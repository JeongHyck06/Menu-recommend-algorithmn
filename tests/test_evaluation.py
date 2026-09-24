"""평가 지표와 판정 세트 관리 테스트, 모델·API 호출 없음"""

import math

import pytest

from src.recommendation import PipelineConfig
from src.recommendation.evaluation import (
    APPROVED, PENDING, UNJUDGED, build_pool, dcg, evaluate_configs, evaluate_result, judgment_map,
    load_judgments, merge_pool, ndcg_at_k, precision_at_k, reciprocal_rank, save_judgments,
)
from tests.test_recommendation import _recommender


def test_ranking_metrics_on_toy_grades():
    assert precision_at_k([2, 1, 2, 0, 0], 5) == 0.4
    assert precision_at_k([2, 1, 2, 0, 0], 2) == 0.5
    assert reciprocal_rank([0, 1, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0]) == 0.0
    assert dcg([2, 0]) == pytest.approx(3.0)
    assert ndcg_at_k([2, 2], [2, 2], 2) == pytest.approx(1.0)
    assert ndcg_at_k([0, 2], [2, 0], 2) == pytest.approx(3 / math.log2(3) / 3)
    assert ndcg_at_k([0, 0], [0], 2) == 0.0


def test_judgment_map_filters_empty_and_unapproved():
    rows = [
        {"질의": "q", "라벨링단위ID": "a", "적합도": "2", "검토상태": APPROVED},
        {"질의": "q", "라벨링단위ID": "b", "적합도": "1", "검토상태": PENDING},
        {"질의": "q", "라벨링단위ID": "c", "적합도": "", "검토상태": UNJUDGED},
    ]
    assert judgment_map(rows) == {("q", "a"): 2, ("q", "b"): 1}
    assert judgment_map(rows, approved_only=True) == {("q", "a"): 2}


def test_evaluate_result_counts_unjudged_as_zero_but_reports_them():
    result = {"질의": "q", "추천": [{"라벨링단위ID": u} for u in ("a", "b", "c")]}
    jmap = {("q", "a"): 2, ("q", "c"): 1, ("q", "z"): 2}
    m = evaluate_result(result, jmap, 3)
    assert m["미판정수"] == 1 and m["적합수"] == 1 and m["부분적합수"] == 1
    assert m["P@3"] == pytest.approx(1 / 3) and m["RR"] == 1.0
    assert m["nDCG@3"] == pytest.approx(dcg([2, 0, 1]) / dcg([2, 2, 1]))


def test_pool_merge_preserves_existing_judgments(tmp_path):
    rec, _ = _recommender()
    pool = build_pool([rec], ["피자", "맵지 않고 따뜻한 음식"], [PipelineConfig(), PipelineConfig(candidate_k=8)], k=2)
    keys = {(p["질의"], p["라벨링단위ID"]) for p in pool}
    assert len(keys) == len(pool) and ("피자", "u2") in keys

    existing = [{"질의": "피자", "라벨링단위ID": "u2", "메뉴명": "x", "적합도": "2", "판정출처": "사람",
                 "검토상태": APPROVED, "검토메모": "확인"}]
    merged = merge_pool(existing, pool, {"피자": "피자면 적합"})
    assert merged[0]["적합도"] == "2" and merged[0]["검토상태"] == APPROVED
    new = [r for r in merged if r["라벨링단위ID"] != "u2" or r["질의"] != "피자"]
    assert all(r["검토상태"] == UNJUDGED and r["적합도"] == "" for r in new)
    assert next(r for r in new if r["질의"] == "피자")["판정기준"] == "피자면 적합"

    path = tmp_path / "judgments.csv"
    save_judgments(path, merged)
    reloaded = load_judgments(path)
    assert len(reloaded) == len(merged) and reloaded[0]["검토상태"] == APPROVED
    assert load_judgments(tmp_path / "missing.csv") == []


def test_evaluate_configs_reports_unjudged_ratio():
    rec, _ = _recommender()
    jmap = {("피자", "u2"): 2}
    rows = evaluate_configs({"B": rec}, ["피자"], {"기본": PipelineConfig()}, jmap, k=3)
    assert rows[0]["질의수"] == 1 and rows[0]["P@3"] == pytest.approx(1 / 3)
    assert rows[0]["미판정비율"] == pytest.approx(2 / 3) and rows[0]["MRR"] == 1.0
