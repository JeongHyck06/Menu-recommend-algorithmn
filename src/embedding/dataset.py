"""라벨링 단위 기준 음식 정보와 라벨 연결

원본 음식 데이터와 라벨 파일은 읽기만 하고 수정하지 않는다
임베딩 단위는 라벨링단위이며 원본 식품코드목록을 그대로 보존한다
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

UNITS_PATH = DATA_DIR / "processed" / "labeling" / "labeling_units.csv"
LABELS_PATH = DATA_DIR / "processed" / "labeling" / "labels_chat_full_v3.jsonl"
FOODS_PATH = DATA_DIR / "processed" / "food_menu.csv"


@dataclass
class JoinReport:
    """라벨링단위ID 기준 연결 검증 결과"""
    num_units: int = 0
    num_labels: int = 0
    num_foods: int = 0
    num_joined: int = 0
    duplicate_unit_ids: list = field(default_factory=list)
    duplicate_label_ids: list = field(default_factory=list)
    units_without_label: list = field(default_factory=list)
    labels_without_unit: list = field(default_factory=list)
    codelist_mismatch: list = field(default_factory=list)
    codes_missing_in_foods: list = field(default_factory=list)
    codes_not_labeled: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not any([
            self.duplicate_unit_ids, self.duplicate_label_ids,
            self.units_without_label, self.labels_without_unit,
            self.codelist_mismatch, self.codes_missing_in_foods,
        ])

    def summary(self) -> str:
        lines = [
            f"라벨링 단위 {self.num_units}건, 라벨 {self.num_labels}건, 음식 {self.num_foods}행",
            f"연결 성공 {self.num_joined}건",
            f"ID 중복: 단위 {len(self.duplicate_unit_ids)}건, 라벨 {len(self.duplicate_label_ids)}건",
            f"라벨 없는 단위 {len(self.units_without_label)}건, 단위 없는 라벨 {len(self.labels_without_unit)}건",
            f"식품코드목록 불일치 {len(self.codelist_mismatch)}건",
            f"음식 데이터에 없는 식품코드 {len(self.codes_missing_in_foods)}건",
            f"라벨링 대상 외 식품코드 {len(self.codes_not_labeled)}건",
        ]
        return "\n".join(lines)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_codes(codelist):
    return [c.strip() for c in str(codelist or "").split(";") if c.strip()]


def _duplicates(ids):
    seen, dups = set(), []
    for i in ids:
        if i in seen:
            dups.append(i)
        seen.add(i)
    return dups


def load_sources(units_path=UNITS_PATH, labels_path=LABELS_PATH, foods_path=FOODS_PATH):
    """원본 3개 파일 로드"""
    return _read_csv(units_path), _read_jsonl(labels_path), _read_csv(foods_path)


def join_units_labels(units, labels, foods):
    """라벨링단위ID 기준 연결 및 검증

    Returns:
        (joined, report) joined는 라벨링단위ID 정렬 순서의 {unit, label, foods} 목록
    """
    report = JoinReport(num_units=len(units), num_labels=len(labels), num_foods=len(foods))

    unit_ids = [u["라벨링단위ID"] for u in units]
    label_ids = [l["라벨링단위ID"] for l in labels]
    report.duplicate_unit_ids = _duplicates(unit_ids)
    report.duplicate_label_ids = _duplicates(label_ids)

    unit_map = {u["라벨링단위ID"]: u for u in units}
    label_map = {l["라벨링단위ID"]: l for l in labels}
    food_map = {f["식품코드"].strip(): f for f in foods}

    report.units_without_label = sorted(set(unit_map) - set(label_map))
    report.labels_without_unit = sorted(set(label_map) - set(unit_map))

    joined = []
    labeled_codes = set()
    for uid in sorted(set(unit_map) & set(label_map)):
        unit, label = unit_map[uid], label_map[uid]

        unit_codes = _split_codes(unit.get("식품코드목록"))
        label_codes = _split_codes(label.get("식품코드목록"))
        if unit_codes != label_codes:
            report.codelist_mismatch.append(uid)

        labeled_codes.update(unit_codes)
        missing = [c for c in unit_codes if c not in food_map]
        if missing:
            report.codes_missing_in_foods.append((uid, missing))

        joined.append({
            "라벨링단위ID": uid,
            "unit": unit,
            "label": label,
            "식품코드목록": unit.get("식품코드목록", "").strip(),
            "codes": unit_codes,
            "foods": [food_map[c] for c in unit_codes if c in food_map],
        })

    report.num_joined = len(joined)
    report.codes_not_labeled = sorted(set(food_map) - labeled_codes)
    return joined, report
