"""수동 검토 시트와 품질 지표"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.labeling.runner import MODE_LIVE, STATUS_SUCCESS, STATUS_VALIDATION_FAILED
from experiments.labeling.schema import ATTRIBUTES, SOURCE_MODEL, SOURCE_ORIGINAL, UNKNOWN
from experiments.labeling.validation import FLAG_TEMPERATURE_CONFLICT

REVIEW_NOT_RUN = "미실행"
REVIEW_PENDING = "검토대기"
REVIEW_FAILED = "실패"
REVIEW_APPROVED = "승인"
REVIEW_CORRECTED = "수정"
REVIEW_RECHECK = "재검토"
REVIEWED_STATES = {REVIEW_APPROVED, REVIEW_CORRECTED}

NOT_RUN = "미실행"
NOT_REVIEWED = "미검토"

INPUT_COLUMNS = ["라벨링단위ID", "식품코드", "식품코드목록", "식품명", "메뉴명", "이름접두어", "대표식품명",
                 "식품대분류명", "업체명", "온도", "프랜차이즈여부", "선정사유"]
REVIEW_VALUE_COLUMNS = [f"검토_{a}" for a in ATTRIBUTES]
SHEET_COLUMNS = [*INPUT_COLUMNS, "온도출처", "상태", "model", "run_id", "cache_key", "라벨출처",
                 *[f"모델_{a}" for a in ATTRIBUTES], "근거", "검토플래그",
                 *REVIEW_VALUE_COLUMNS, "검토상태", "검토메모", "오류"]


def latest_records(records: list[dict], mode: str = MODE_LIVE) -> dict[str, dict]:
    """단위별 마지막 기록, 모의 응답(mode != live)은 제외"""
    latest: dict[str, dict] = {}
    for r in records:
        if r.get("mode") != mode:
            continue
        latest[r["라벨링단위ID"]] = r
    return latest


def build_review_sheet(sample: pd.DataFrame, records: list[dict], mode: str = MODE_LIVE) -> pd.DataFrame:
    """샘플 입력 + 모델 라벨 + 검토자 입력 컬럼, 응답이 없으면 검토상태는 미실행"""
    latest = latest_records(records, mode)
    rows = []
    for _, unit in sample.iterrows():
        row = {c: unit.get(c) for c in INPUT_COLUMNS if c in unit}
        row["온도출처"] = SOURCE_ORIGINAL if pd.notna(unit.get("온도")) else None
        record = latest.get(unit["라벨링단위ID"])
        if record is None:
            row["상태"] = REVIEW_NOT_RUN
            row["검토상태"] = REVIEW_NOT_RUN
        elif record["status"] == STATUS_SUCCESS:
            row["상태"] = STATUS_SUCCESS
            row["검토상태"] = REVIEW_PENDING
            for attr in ATTRIBUTES:
                row[f"모델_{attr}"] = record["labels"][attr]
            row["라벨출처"] = SOURCE_MODEL
            row["근거"] = record["reason"]
            row["검토플래그"] = ";".join(record.get("review_flags") or [])
            row["model"] = record["model"]
            row["cache_key"] = record.get("cache_key")
            row["run_id"] = record.get("run_id")
        else:
            row["상태"] = record["status"]
            row["검토상태"] = REVIEW_FAILED
            row["오류"] = record.get("error")
            row["cache_key"] = record.get("cache_key")
            row["run_id"] = record.get("run_id")
        for attr in ATTRIBUTES:
            row.setdefault(f"모델_{attr}", None)
            row[f"검토_{attr}"] = None
        row["검토메모"] = None
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=SHEET_COLUMNS)


def merge_existing_review(sheet: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """기존 검토 파일의 검토값·검토상태·메모를 보존

    cache_key가 같으면 그대로 유지, 다르면(입력·모델·프롬프트·스키마 변경) 값은 남기되 검토상태는 재검토
    """
    if existing is None or existing.empty or "라벨링단위ID" not in existing:
        return sheet
    merged = sheet.copy()
    old = existing.drop_duplicates("라벨링단위ID", keep="last").set_index("라벨링단위ID")
    review_columns = [*REVIEW_VALUE_COLUMNS, "검토메모"]
    for idx, row in merged.iterrows():
        uid = row["라벨링단위ID"]
        if uid not in old.index:
            continue
        prev = old.loc[uid]
        had_review = prev.get("검토상태") in REVIEWED_STATES or any(
            pd.notna(prev.get(c)) for c in review_columns if c in prev
        )
        if not had_review:
            continue
        for c in review_columns:
            if c in prev and pd.notna(prev[c]):
                merged.at[idx, c] = prev[c]
        same_result = (
            row["상태"] == STATUS_SUCCESS
            and pd.notna(prev.get("cache_key"))
            and prev.get("cache_key") == row.get("cache_key")
        )
        if same_result:
            merged.at[idx, "검토상태"] = prev["검토상태"]
        elif row["상태"] == STATUS_SUCCESS:
            merged.at[idx, "검토상태"] = REVIEW_RECHECK
    return merged


def load_review_sheet(path: Path | str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


def save_review_sheet(sheet: pd.DataFrame, path: Path | str) -> Path | None:
    """저장 전 기존 파일을 backups/ 아래에 시각을 붙여 백업, 백업 경로 반환"""
    path = Path(path)
    backup = None
    if path.exists():
        backup_dir = path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"{path.stem}.{stamp}{path.suffix}"
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return backup


def compute_quality_metrics(records: list[dict], review_sheet: pd.DataFrame, mode: str = MODE_LIVE) -> dict:
    """형식 오류율, 미확인 비율, 원본 충돌률, 속성별 수동 검토 일치율

    실제 응답이 없으면 지표를 0이 아니라 미실행으로 표시
    """
    live = [r for r in records if r.get("mode") == mode]
    attempted_units = {r["라벨링단위ID"] for r in live}
    if not attempted_units:
        return {"상태": NOT_RUN, "형식오류율": NOT_RUN, "원본충돌률": NOT_RUN,
                **{f"미확인비율_{a}": NOT_RUN for a in ATTRIBUTES},
                **{f"검토일치율_{a}": NOT_RUN for a in ATTRIBUTES}}

    latest = latest_records(live, mode)
    successes = [r for r in latest.values() if r["status"] == STATUS_SUCCESS]
    validation_failed_units = {r["라벨링단위ID"] for r in live if r["status"] == STATUS_VALIDATION_FAILED}

    metrics = {
        "상태": "실행됨",
        "시도단위수": len(attempted_units),
        "성공단위수": len(successes),
        "형식오류율": round(len(validation_failed_units) / len(attempted_units), 4),
        "원본충돌률": (
            round(sum(FLAG_TEMPERATURE_CONFLICT in (r.get("review_flags") or []) for r in successes) / len(successes), 4)
            if successes else NOT_RUN
        ),
    }
    for attr in ATTRIBUTES:
        metrics[f"미확인비율_{attr}"] = (
            round(sum(r["labels"][attr] == UNKNOWN for r in successes) / len(successes), 4) if successes else NOT_RUN
        )

    reviewed = review_sheet[review_sheet["검토상태"].isin([REVIEW_APPROVED, REVIEW_CORRECTED])]
    for attr in ATTRIBUTES:
        if reviewed.empty:
            metrics[f"검토일치율_{attr}"] = NOT_REVIEWED
            continue
        model_values = reviewed[f"모델_{attr}"]
        reviewer_values = reviewed[f"검토_{attr}"]
        # 승인이고 검토값이 비어 있으면 모델값 인정
        final = reviewer_values.where(reviewer_values.notna(), model_values)
        metrics[f"검토일치율_{attr}"] = round((final == model_values).mean(), 4)
    return metrics
