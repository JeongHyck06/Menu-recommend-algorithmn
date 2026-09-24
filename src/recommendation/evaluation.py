"""추천 결과 평가: 적합도 판정 세트 관리와 순위 지표

적합도는 2(적합), 1(부분 적합), 0(부적합)이다. 판정은 판정출처(모델추정/사람)와 검토상태를 가지며
승인된 판정만으로도 지표를 계산할 수 있다.

풀링 평가의 한계: 판정 풀에 없는 항목은 미판정이며 지표에서는 0으로 취급하고 미판정 비율을 함께 보고한다.
미판정이 많은 설정의 점수는 낮게 나오므로 미판정 비율 없이 점수만 비교하지 않는다.
"""

import csv
import math
import random
from dataclasses import replace
from pathlib import Path

RELEVANT = 2
UNJUDGED = "미판정"
PENDING = "검토대기"
APPROVED = "승인"

JUDGMENT_COLUMNS = ["질의", "판정기준", "라벨링단위ID", "메뉴명", "업체명", "대표식품명", "식품대분류명",
                    "적합도", "판정출처", "검토상태", "검토메모"]
ITEM_FIELDS = ("라벨링단위ID", "메뉴명", "업체명", "대표식품명", "식품대분류명")


# 판정 풀

def build_pool(recommenders, queries, configs, k=5) -> list:
    """여러 추천기·설정의 상위 k개를 합친 판정 후보 풀, (질의, 라벨링단위ID) 기준 중복 제거"""
    pool, seen = [], set()
    for query in queries:
        for rec in recommenders:
            for config in configs:
                for item in rec.recommend(query, replace(config, top_k=k))["추천"]:
                    key = (query, item["라벨링단위ID"])
                    if key not in seen:
                        seen.add(key)
                        pool.append({"질의": query, **{f: item.get(f) or "" for f in ITEM_FIELDS}})
    return pool


def load_judgments(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def save_judgments(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGMENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({c: row.get(c, "") for c in JUDGMENT_COLUMNS} for row in rows)


def merge_pool(existing, pool, criteria=None) -> list:
    """기존 판정은 그대로 두고 풀에 새로 들어온 항목만 미판정으로 덧붙인다"""
    criteria = criteria or {}
    rows = [dict(r) for r in existing]
    seen = {(r["질의"], r["라벨링단위ID"]) for r in rows}
    for item in pool:
        key = (item["질의"], item["라벨링단위ID"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({**item, "판정기준": criteria.get(item["질의"], ""), "적합도": "",
                     "판정출처": "", "검토상태": UNJUDGED, "검토메모": ""})
    return rows


def judgment_map(rows, approved_only=False) -> dict:
    """{(질의, 라벨링단위ID): 적합도}, 적합도가 비어 있거나 (approved_only면) 승인되지 않은 행은 뺀다"""
    out = {}
    for row in rows:
        grade = str(row.get("적합도", "")).strip()
        if grade == "" or (approved_only and row.get("검토상태") != APPROVED):
            continue
        out[(row["질의"], row["라벨링단위ID"])] = int(float(grade))
    return out


# 순위 지표

def dcg(grades) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades, ideal_grades, k) -> float:
    """ideal_grades는 그 질의에 대해 판정된 모든 적합도, 이상적 순서는 내림차순"""
    ideal = dcg(sorted(ideal_grades, reverse=True)[:k])
    return dcg(list(grades)[:k]) / ideal if ideal > 0 else 0.0


def precision_at_k(grades, k, threshold=RELEVANT) -> float:
    top = list(grades)[:k]
    return sum(1 for g in top if g >= threshold) / k if k else 0.0


def reciprocal_rank(grades, threshold=RELEVANT) -> float:
    for i, g in enumerate(grades):
        if g >= threshold:
            return 1 / (i + 1)
    return 0.0


def evaluate_result(result, jmap, k) -> dict:
    """추천 결과 하나의 지표, 미판정 항목은 0으로 보고 개수를 따로 센다"""
    query = result["질의"]
    items = result["추천"][:k]
    judged = [jmap.get((query, it["라벨링단위ID"])) for it in items]
    grades = [g if g is not None else 0 for g in judged]
    ideal = [g for (q, _), g in jmap.items() if q == query]
    return {
        "질의": query, "반환수": len(items), "미판정수": sum(g is None for g in judged),
        f"P@{k}": precision_at_k(grades, k), f"nDCG@{k}": ndcg_at_k(grades, ideal, k),
        "RR": reciprocal_rank(grades), "적합수": sum(g >= RELEVANT for g in grades),
        "부분적합수": sum(g == 1 for g in grades),
    }


def evaluate_per_query(rec, queries, config, jmap, k=5) -> list:
    """설정 하나에 대한 질의별 지표"""
    return [evaluate_result(rec.recommend(q, replace(config, top_k=k)), jmap, k) for q in queries]


def summarize(per_query, k=5) -> dict:
    n = len(per_query) or 1
    return {
        "질의수": len(per_query),
        f"P@{k}": sum(r[f"P@{k}"] for r in per_query) / n,
        f"nDCG@{k}": sum(r[f"nDCG@{k}"] for r in per_query) / n,
        "MRR": sum(r["RR"] for r in per_query) / n,
        "미판정비율": sum(r["미판정수"] for r in per_query) / max(sum(r["반환수"] for r in per_query), 1),
    }


def evaluate_configs(recommenders, queries, configs, jmap, k=5) -> list:
    """추천기(이름->Recommender) × 설정(이름->PipelineConfig)별 평균 지표"""
    return [
        {"텍스트구성": rec_name, "설정": cfg_name, **summarize(evaluate_per_query(rec, queries, config, jmap, k), k)}
        for rec_name, rec in recommenders.items()
        for cfg_name, config in configs.items()
    ]


def paired_bootstrap(values_a, values_b, iterations=2000, seed=0) -> dict:
    """질의별 지표 두 벌의 평균 차이(b − a)와 95% 부트스트랩 신뢰구간, 질의를 복원 추출한다"""
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("길이가 같은 비어 있지 않은 두 목록이 필요합니다")
    diffs = [b - a for a, b in zip(values_a, values_b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iterations))
    lo, hi = means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]
    return {"평균차이": sum(diffs) / n, "하한95": lo, "상한95": hi, "질의수": n,
            "0포함": lo <= 0 <= hi}
