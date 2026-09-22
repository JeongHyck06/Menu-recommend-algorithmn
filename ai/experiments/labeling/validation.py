"""모델 응답 파싱과 검증"""

import json
import re
from dataclasses import dataclass, field

from ai.experiments.labeling.schema import (
    ATTRIBUTES,
    MAX_REASON_LENGTH,
    ORIGINAL_TEMPERATURE_COMPATIBLE,
    REQUIRED_RESPONSE_FIELDS,
    UNKNOWN,
    allowed_values,
)

FLAG_TEMPERATURE_CONFLICT = "원본온도충돌"
FLAG_REASON_TRUNCATED = "근거길이초과"
FLAG_EXTRA_ATTRIBUTES = "추가속성무시"

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ResponseValidationError(ValueError):
    """형식 검증 실패, 일시적 API 오류와 구분"""


@dataclass
class ValidatedLabels:
    labels: dict[str, str]
    reason: str
    review_flags: list[str] = field(default_factory=list)


def parse_response_text(text: str) -> dict:
    cleaned = _CODE_FENCE.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"JSON 파싱 실패: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ResponseValidationError("최상위가 객체가 아님")
    return payload


def validate_response(payload: dict, expected_code: str, original_temperature: str | None) -> ValidatedLabels:
    """필수 필드, 식품코드 일치, 허용값 검증 -> 원본 온도 충돌은 검토 플래그"""
    missing = [f for f in REQUIRED_RESPONSE_FIELDS if f not in payload]
    if missing:
        raise ResponseValidationError(f"필수 필드 누락: {missing}")

    if str(payload["식품코드"]) != str(expected_code):
        raise ResponseValidationError(f"식품코드 불일치: {payload['식품코드']} != {expected_code}")

    raw_labels = payload["라벨"]
    if not isinstance(raw_labels, dict):
        raise ResponseValidationError("라벨이 객체가 아님")

    flags: list[str] = []
    labels: dict[str, str] = {}
    for attr in ATTRIBUTES:
        if attr not in raw_labels:
            raise ResponseValidationError(f"속성 누락: {attr}")
        value = raw_labels[attr]
        if value not in allowed_values(attr):
            raise ResponseValidationError(f"허용값 아님: {attr}={value!r}")
        labels[attr] = value
    if set(raw_labels) - set(ATTRIBUTES):
        flags.append(FLAG_EXTRA_ATTRIBUTES)

    reason = payload["근거"]
    if not isinstance(reason, str) or not reason.strip():
        raise ResponseValidationError("근거가 비어 있음")
    reason = reason.strip()
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[:MAX_REASON_LENGTH]
        flags.append(FLAG_REASON_TRUNCATED)

    if _conflicts_with_original(labels["제공온도"], original_temperature):
        flags.append(FLAG_TEMPERATURE_CONFLICT)

    return ValidatedLabels(labels=labels, reason=reason, review_flags=flags)


def _conflicts_with_original(model_value: str, original_temperature: str | None) -> bool:
    if not original_temperature or model_value == UNKNOWN:
        return False
    compatible = ORIGINAL_TEMPERATURE_COMPATIBLE.get(original_temperature)
    if compatible is None:
        return False
    return model_value not in compatible


def check_result_set(result_ids: list[str], expected_ids: list[str]) -> dict[str, list[str]]:
    """결과 목록의 누락과 중복 확인"""
    seen: dict[str, int] = {}
    for rid in result_ids:
        seen[rid] = seen.get(rid, 0) + 1
    return {
        "누락": [uid for uid in expected_ids if uid not in seen],
        "중복": [uid for uid, n in seen.items() if n > 1],
        "대상외": [uid for uid in seen if uid not in set(expected_ids)],
    }
