"""라벨링 실행: 캐시 재사용, 재시도, 점진 저장, 비용 한도와 기록

API 키는 anthropic SDK가 ANTHROPIC_API_KEY 환경변수에서 읽음, 코드와 로그에 키를 남기지 않음
"""

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from experiments.labeling.prompt import LabelingConfig, build_request_params, build_unit_input
from experiments.labeling.validation import ResponseValidationError, parse_response_text, validate_response

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
STATUS_FATAL = "fatal"
STATUS_BUDGET_STOPPED = "budget_stopped"

MODE_LIVE = "live"
MODE_MOCK = "mock"

API_KEY_ENV = "ANTHROPIC_API_KEY"

# 첫 호출 전 다음 호출 비용을 어림할 때 쓰는 입력 토큰 수
DEFAULT_INPUT_TOKEN_GUESS = 1000


class FatalLabelingError(RuntimeError):
    """인증·결제처럼 전체 실행에 영향을 주는 오류, 즉시 중단"""


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


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


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
    """일시적 오류(재시도), 치명적 오류(실행 중단), 영구 오류(해당 단위만 실패) 구분"""
    import anthropic

    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return STATUS_FATAL
    if isinstance(exc, anthropic.BadRequestError) and "credit" in str(exc).lower():
        return STATUS_FATAL
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


def _usage(response) -> dict:
    return {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}


def label_unit(
    client,
    unit: pd.Series | dict,
    config: LabelingConfig,
    store: ResultStore,
    mode: str = MODE_LIVE,
    run_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """단위 하나 라벨링 -> 저장소 기록, 성공 결과가 있으면 재사용

    치명적 오류는 기록 후 FatalLabelingError로 올린다
    """
    unit_input = build_unit_input(unit)
    cache_key = make_cache_key(unit_input, config.model, config.prompt_version, config.schema_version)
    cached = store.successful_by_key(mode).get(cache_key)
    if cached is not None:
        return {**cached, "reused": True}

    base = {
        "cache_key": cache_key,
        "run_id": run_id,
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
            if last_status == STATUS_FATAL:
                store.append({**base, "status": STATUS_FATAL, "attempt": attempt, "error": last_error, "created_at": _now()})
                raise FatalLabelingError(last_error) from exc
            if last_status == STATUS_PERMANENT_FAILED:
                break
            sleep(config.retry_wait_seconds * attempt)
            continue

        raw_text = _response_text(response)
        usage = _usage(response)
        cost = estimate_cost_usd(usage["input_tokens"], usage["output_tokens"], config.model)
        try:
            validated = validate_response(parse_response_text(raw_text), unit_input["식품코드"], unit_input["온도_원본"])
        except ResponseValidationError as exc:
            last_status = STATUS_VALIDATION_FAILED
            last_error = str(exc)
            store.append({**base, "status": STATUS_VALIDATION_FAILED, "attempt": attempt, "error": last_error,
                          "raw_text": raw_text, "usage": usage, "cost_usd": cost, "created_at": _now()})
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
            "cost_usd": cost,
            "created_at": _now(),
        }
        store.append(record)
        return {**record, "reused": False}

    record = {**base, "status": last_status, "attempt": config.max_retries + 1, "error": last_error,
              "raw_text": raw_text, "created_at": _now()}
    store.append(record)
    return {**record, "reused": False}


def make_client():
    """SDK 클라이언트, 내부 재시도를 끄고 자체 재시도만 사용"""
    if not os.environ.get(API_KEY_ENV):
        raise RuntimeError(f"{API_KEY_ENV} 환경변수가 없어 클라이언트를 만들 수 없다")
    import anthropic

    return anthropic.Anthropic(max_retries=0)


def run_labeling(
    units: pd.DataFrame,
    config: LabelingConfig,
    store: ResultStore,
    client=None,
    mode: str = MODE_LIVE,
    run_id: str | None = None,
    budget_usd: float | None = None,
    input_token_guess: int = DEFAULT_INPUT_TOKEN_GUESS,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """단위 목록을 순서대로 라벨링, config.enabled가 False면 호출하지 않음

    budget_usd가 있으면 이번 실행 비용 + 다음 호출 최대 예상 비용이 한도를 넘기 전에 중단
    다음 호출 예상: 지금까지 관측한 최대 입력 토큰(없으면 input_token_guess)과 max_tokens 출력
    """
    if not config.enabled:
        raise RuntimeError("LabelingConfig.enabled=False 이므로 API를 호출하지 않는다")
    if client is None:
        client = make_client()
    run_id = run_id or new_run_id()

    rows = []
    spent = 0.0
    max_input_seen = 0
    for _, unit in units.iterrows():
        if budget_usd is not None:
            next_call = estimate_cost_usd(max(max_input_seen, input_token_guess), config.max_tokens, config.model)
            if spent + next_call > budget_usd:
                rows.append({"라벨링단위ID": unit.get("라벨링단위ID"), "식품코드": unit.get("식품코드"),
                             "status": STATUS_BUDGET_STOPPED, "reused": False, "run_id": run_id,
                             "error": f"한도 {budget_usd} USD, 지출 {spent:.4f} + 다음 호출 최대 {next_call:.4f}"})
                continue

        before = len(store.load())
        result = label_unit(client, unit, config, store, mode=mode, run_id=run_id, sleep=sleep)
        if not result["reused"]:
            for r in store.load()[before:]:
                usage = r.get("usage") or {}
                spent += r.get("cost_usd") or 0.0
                max_input_seen = max(max_input_seen, usage.get("input_tokens") or 0)

        usage = result.get("usage") or {}
        rows.append({
            "라벨링단위ID": result.get("라벨링단위ID"),
            "식품코드": result["식품코드"],
            "status": result["status"],
            "reused": result["reused"],
            "run_id": run_id,
            "attempt": result.get("attempt"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": 0.0 if result["reused"] else result.get("cost_usd"),
            "review_flags": ";".join(result.get("review_flags") or []),
            "error": result.get("error"),
        })
    return pd.DataFrame(rows)


def summarize_run(store: ResultStore, run_id: str, mode: str = MODE_LIVE) -> dict:
    """이번 실행(run_id) 비용과 누적 비용, 재사용 결과는 이번 실행 비용에 포함되지 않음

    검증 실패 후 재시도한 응답도 이번 실행 토큰과 비용에 포함
    """
    records = [r for r in store.load() if r.get("mode") == mode]
    this_run = [r for r in records if r.get("run_id") == run_id]

    def _totals(items):
        return {
            "호출수": sum(1 for r in items if r.get("usage")),
            "입력토큰": sum((r.get("usage") or {}).get("input_tokens", 0) for r in items),
            "출력토큰": sum((r.get("usage") or {}).get("output_tokens", 0) for r in items),
            "비용USD": round(sum(r.get("cost_usd") or 0.0 for r in items), 6),
        }

    statuses = pd.Series([r["status"] for r in this_run]).value_counts().to_dict() if this_run else {}
    return {"run_id": run_id, "이번실행": {**_totals(this_run), "기록상태": statuses}, "누적": _totals(records)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
