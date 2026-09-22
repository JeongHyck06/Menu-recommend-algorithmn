"""라벨링 단위 구성과 실험 샘플 추출"""

import hashlib
import math
from collections import Counter

import pandas as pd

from recommender.preprocessing import food_data as fd

# 같은 값을 가진 행끼리만 라벨을 공유한다
UNIT_KEY_COLUMNS = ["메뉴명", "이름접두어", "대표식품명", "식품대분류명", "업체명", "온도"]

NON_FRANCHISE_LABEL = "비프랜차이즈"

REASON_AMBIGUOUS = "경계 메뉴"
REASON_CATEGORY = "카테고리 배분"
REASON_FILL = "잔여 보충"


def _unit_id(values: list[str]) -> str:
    return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:12]


def build_labeling_units(menu: pd.DataFrame) -> pd.DataFrame:
    """식사 후보 행 -> 라벨링 단위

    단위 키가 같은 행(사이즈나 출처만 다른 행)은 하나의 단위로 묶고
    식품코드 목록을 남겨 원본 행으로 되돌아갈 수 있게 함
    """
    df = menu.copy()
    key = df[UNIT_KEY_COLUMNS].astype("object").fillna("")
    key["업체명"] = key["업체명"].replace("", NON_FRANCHISE_LABEL)
    df["라벨링단위ID"] = key.apply(lambda r: _unit_id(r.tolist()), axis=1)

    df = df.sort_values(["라벨링단위ID", "식품코드"], kind="stable")
    grouped = df.groupby("라벨링단위ID", sort=False)
    units = grouped.first()[
        ["식품명", "메뉴명", "이름접두어", "대표식품명", "식품대분류명", "업체명", "온도", "프랜차이즈여부"]
    ]
    units["식품코드"] = grouped["식품코드"].first()
    units["식품코드목록"] = grouped["식품코드"].agg(";".join)
    units["행수"] = grouped.size()
    units["경계메뉴"] = [
        is_ambiguous(rep, cat) for rep, cat in zip(units["대표식품명"], units["식품대분류명"])
    ]
    return units.reset_index()


def is_ambiguous(representative: str, category: str) -> bool:
    """조리법 대분류에서 반찬 키워드를 포함하지만 식사로 남은 메뉴"""
    if category not in fd.COOKING_CATEGORIES:
        return False
    return any(k in str(representative) for k in fd.SIDE_DISH_KEYWORDS)


def _allocate(counts: pd.Series, total: int) -> dict[str, int]:
    """카테고리 단위 수의 제곱근에 비례해 배분, 최대 잔여 방식으로 반올림"""
    weights = counts.map(math.sqrt)
    raw = weights / weights.sum() * total
    alloc = {k: min(int(v), int(counts[k])) for k, v in raw.items()}
    remaining = total - sum(alloc.values())
    order = sorted(raw.index, key=lambda k: (raw[k] - int(raw[k])), reverse=True)
    for k in order:
        if remaining <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            remaining -= 1
    return alloc


def _take(
    pool: pd.DataFrame,
    k: int,
    chosen_ids: set[str],
    rep_counts: Counter,
    cap: int,
) -> list[str]:
    """프랜차이즈와 비프랜차이즈를 번갈아 뽑고 대표식품명 상한 유지"""
    queues = [
        list(pool.loc[~pool["프랜차이즈여부"], "라벨링단위ID"]),
        list(pool.loc[pool["프랜차이즈여부"], "라벨링단위ID"]),
    ]
    rep_of = dict(zip(pool["라벨링단위ID"], pool["대표식품명"]))
    picked: list[str] = []
    turn = 0
    while len(picked) < k and any(queues):
        queue = queues[turn % 2]
        turn += 1
        while queue:
            uid = queue.pop(0)
            rep = rep_of[uid]
            if uid in chosen_ids or rep_counts[rep] >= cap:
                continue
            picked.append(uid)
            chosen_ids.add(uid)
            rep_counts[rep] += 1
            break
    return picked


def sample_experiment_units(
    units: pd.DataFrame,
    n_total: int = 100,
    n_ambiguous: int = 10,
    cap_per_representative: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """실험용 균형 샘플

    1. 경계 메뉴에서 n_ambiguous개
    2. 나머지를 식품대분류별로 배분 (단위 수 제곱근 비례)
    3. 부족하면 남은 풀에서 보충
    모든 단계에서 대표식품명당 cap_per_representative개를 넘지 않음
    """
    shuffled = units.sample(frac=1, random_state=random_state).reset_index(drop=True)
    chosen_ids: set[str] = set()
    rep_counts: Counter = Counter()
    reasons: dict[str, str] = {}

    for uid in _take(shuffled[shuffled["경계메뉴"]], n_ambiguous, chosen_ids, rep_counts, cap_per_representative):
        reasons[uid] = REASON_AMBIGUOUS

    remaining = shuffled[~shuffled["라벨링단위ID"].isin(chosen_ids)]
    quotas = _allocate(remaining["식품대분류명"].value_counts(), n_total - len(chosen_ids))
    for category, quota in quotas.items():
        pool = remaining[remaining["식품대분류명"] == category]
        for uid in _take(pool, quota, chosen_ids, rep_counts, cap_per_representative):
            reasons[uid] = REASON_CATEGORY

    shortfall = n_total - len(chosen_ids)
    if shortfall > 0:
        pool = shuffled[~shuffled["라벨링단위ID"].isin(chosen_ids)]
        for uid in _take(pool, shortfall, chosen_ids, rep_counts, cap_per_representative):
            reasons[uid] = REASON_FILL

    sample = shuffled[shuffled["라벨링단위ID"].isin(chosen_ids)].copy()
    sample["선정사유"] = sample["라벨링단위ID"].map(reasons)
    sample["random_state"] = random_state
    return sample.sort_values(["식품대분류명", "라벨링단위ID"]).reset_index(drop=True)
