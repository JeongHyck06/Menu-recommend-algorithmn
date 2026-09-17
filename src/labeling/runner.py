"""라벨링 실행: 캐시 재사용, 재시도, 점진 저장, 비용 기록

API 키는 anthropic SDK가 ANTHROPIC_API_KEY 환경변수에서 읽음, 코드와 로그에 키를 남기지 않음
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.labeling.prompt import LabelingConfig, build_request_params, build_unit_input
from src.labeling.validation import ResponseValidationError, parse_response_text, validate_response

# 공식 가격 확인일 2026-09-17, https://platform.claude.com/docs/en/about-claude/pricing (USD / MTok)
PRICING = {
    "claude-haiku-4-5-20251001": {
        "input_per_mtok": 1.0,
        "output_per_mtok": 5.0,
        "batch_discount": 0.5,
        "checked_on": "2026-09-17",
    },
}

STATUS_SUCCESS = "success"
STATUS_VALIDATION_FAILED = "validation_failed"
STATUS_TRANSIENT_FAILED = "transient_failed"
STATUS_PERMANENT_FAILED = "permanent_failed"

MODE_LIVE = "live"
MODE_MOCK = "mock"

API_KEY_ENV = "ANTHROPIC_API_KEY"


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str, batch: bool = False) -> float:
    price = PRICING[model]
    cost = input_tokens / 1e6 * price["input_per_mtok"] + output_tokens / 1e6 * price["output_per_mtok"]
    if batch:
        cost *= 1 - price["batch_discount"]
    return round(cost, 6)


def make_cache_key(unit_input: dict, model: str, prompt_version: str, schema_version: str) -> str:
    """입력 내용, 모델, 프롬프트·스키마 버전이 모두 같을 때만 같은 키"""
    payload = json.dumps(
        {"input": unit_input, "model": model, "prompt": prompt_version, "schema": schema_version},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResultStore:
    """JSONL 결과 저장소, 한 줄이 한 번의 라벨링 시도 결과"""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def successful_by_key(self, mode: str = MODE_LIVE) -> dict[str, dict]:
        """재사용 가능한 성공 결과, 같은 키가 여러 번 있으면 마지막 것"""
        return {
            r["cache_key"]: r
            for r in self.load()
            if r["status"] == STATUS_SUCCESS and r.get("mode") == mode
        }


def classify_error(exc: Exception) -> str:
    """일시적 오류(재시도)와 영구 오류(중단) 구분"""
    import anthropic

    transient = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    )
    if isinstance(exc, transient):
        return STATUS_TRANSIENT_FAILED
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return STATUS_TRANSIENT_FAILED
    return STATUS_PERMANENT_FAILED


def _response_text(response) -> str:
    return "".join(getattr(block, "text", "") for block in response.content)


def label_unit(
    client,
    unit: pd.Series | dict,
    config: LabelingConfig,
    store: ResultStore,
    mode: str = MODE_LIVE,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """단위 하나 라벨링 -> 저장소 기록, 성공 결과가 있으면 재사용"""
    unit_input = build_unit_input(unit)
    cache_key = make_cache_key(unit_input, config.model, config.prompt_version, config.schema_version)
    cached = store.successful_by_key(mode).get(cache_key)
    if cached is not None:
        return {**cached, "reused": True}

    base = {
        "cache_key": cache_key,
        "라벨링단위ID": dict(unit).get("라벨링단위ID"),
        "식품코드": unit_input["식품코드"],
        "mode": mode,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "schema_version": config.schema_version,
        "input": unit_input,
    }
    params = build_request_params(unit_input, config)
    last_error = None
    last_status = None
    raw_text = None

    for attempt in range(1, config.max_retries + 2):
        try:
            response = client.messages.create(**params)
        except Exception as exc:
            last_status = classify_error(exc)
            last_error = f"{type(exc).__name__}: {exc}"
            if last_status == STATUS_PERMANENT_FAILED:
                break
            sleep(config.retry_wait_seconds * attempt)
            continue

        raw_text = _response_text(response)
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        try:
            validated = validate_response(parse_response_text(raw_text), unit_input["식품코드"], unit_input["온도_원본"])
        except ResponseValidationError as exc:
            last_status = STATUS_VALIDATION_FAILED
            last_error = str(exc)
            store.append({**base, "status": STATUS_VALIDATION_FAILED, "attempt": attempt, "error": last_error,
                          "raw_text": raw_text, "usage": usage,
                          "cost_usd": estimate_cost_usd(usage["input_tokens"], usage["output_tokens"], config.model),
                          "created_at": _now()})
            continue

        record = {
            **base,
            "status": STATUS_SUCCESS,
            "attempt": attempt,
            "labels": validated.labels,
            "reason": validated.reason,
            "review_flags": validated.review_flags,
            "raw_text": raw_text,
            "usage": usage,
            "cost_usd": estimate_cost_usd(usage["input_tokens"], usage["output_tokens"], config.model),
            "created_at": _now(),
        }
        store.append(record)
        return {**record, "reused": False}

    record = {**base, "status": last_status, "attempt": config.max_retries + 1, "error": last_error,
              "raw_text": raw_text, "created_at": _now()}
    store.append(record)
    return {**record, "reused": False}


def run_labeling(
    units: pd.DataFrame,
    config: LabelingConfig,
    store: ResultStore,
    client=None,
    mode: str = MODE_LIVE,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """단위 목록을 순서대로 라벨링, config.enabled가 False면 호출하지 않음"""
    if not config.enabled:
        raise RuntimeError("LabelingConfig.enabled=False 이므로 API를 호출하지 않는다")
    if client is None:
        if not os.environ.get(API_KEY_ENV):
            raise RuntimeError(f"{API_KEY_ENV} 환경변수가 없어 클라이언트를 만들 수 없다")
        import anthropic

        client = anthropic.Anthropic()

    rows = []
    for _, unit in units.iterrows():
        result = label_unit(client, unit, config, store, mode=mode, sleep=sleep)
        usage = result.get("usage") or {}
        rows.append({
            "라벨링단위ID": result.get("라벨링단위ID"),
            "식품코드": result["식품코드"],
            "status": result["status"],
            "reused": result["reused"],
            "attempt": result.get("attempt"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": result.get("cost_usd"),
            "review_flags": ";".join(result.get("review_flags") or []),
            "error": result.get("error"),
        })
    return pd.DataFrame(rows)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
