"""추천 파이프라인 테스트, 가짜 인덱스와 가짜 질의 인코더 사용 (모델·API 호출 없음)"""

import dataclasses

import numpy as np
import pytest

from src.ranking import RankingConfig
from src.recommendation import (
    EMBEDDING_ONLY, FILTER_ONLY, FULL, STATUS_CONTRADICTION, STATUS_EMPTY_QUERY, STATUS_OK, STATUS_SHORTAGE,
    PipelineConfig, Recommender, result_metrics, result_rows,
)
from src.embedding import DEFAULT_SPEC, TEXT_SPEC_VERSION, EmbeddingStore, build_config, compute_input_hash
from src.retrieval import CandidateIndex, check_compatibility, is_franchise, load_index


def _record(uid, menu, rep, company="", **labels):
    base = {"매운맛": "없음", "국물": "국물없음", "제공온도": "뜨거움", "조리법": "오븐", "기름짐": "높음", "든든함": "미확인"}
    base.update(labels)
    tokens = [f"{k} {v}" for k, v in base.items() if v != "미확인"]
    return {"라벨링단위ID": uid, "메뉴명": menu, "대표식품명": rep, "식품대분류명": "-", "업체명": company,
            "라벨": base, "속성토큰": tokens, "검토상태": "검토대기", "라벨출처": "모델추정"}


RECORDS = [
    _record("u0", "육개장", "육개장", 매운맛="보통", 국물="국물요리", 조리법="끓임", 기름짐="보통"),
    _record("u1", "짬뽕", "짬뽕", 매운맛="강함", 국물="국물요리", 조리법="끓임", 기름짐="보통"),
    _record("u2", "콤비네이션 피자", "피자", "A피자"),
    _record("u3", "콤비네이션피자", "피자", "B피자"),
    _record("u4", "불고기 피자", "피자", "A피자"),
    _record("u5", "고구마 피자", "피자", "C피자", 매운맛="미확인"),
    _record("u6", "김밥", "김밥", 제공온도="상온", 조리법="비조리", 기름짐="낮음", 든든함="보통"),
    _record("u7", "냉면", "냉면", 국물="국물요리", 제공온도="차가움", 조리법="끓임", 기름짐="낮음"),
]

# 문서 i의 벡터는 단위 벡터 e_i, 질의 벡터의 i번째 성분이 곧 유사도가 된다
VECTORS = np.eye(len(RECORDS), dtype=np.float32)


def _query_vector(sims):
    v = np.zeros(len(RECORDS), dtype=np.float32)
    for i, s in enumerate(sims):
        v[i] = s
    return v / np.linalg.norm(v)


# 질의별 유사도 순서를 고정한 가짜 인코더
QUERY_SIMS = {
    "맵지 않고 따뜻한 음식": [0.3, 0.9, 0.85, 0.84, 0.8, 0.83, 0.5, 0.4],
    "국물 없는 매운 음식": [0.9, 0.85, 0.8, 0.79, 0.78, 0.77, 0.5, 0.6],
    "차가운 국물 요리": [0.9, 0.85, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8],
    "맵지 않은 매운 음식": [0.5] * 8,
    "피자": [0.1, 0.1, 0.9, 0.89, 0.88, 0.87, 0.2, 0.2],
    "피자 말고 김밥": [0.1, 0.1, 0.9, 0.89, 0.88, 0.87, 0.6, 0.2],
}


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return _query_vector(QUERY_SIMS.get(text, [0.5] * len(RECORDS)))


def _recommender(config=FULL, candidate_k=4, top_k=3):
    index = CandidateIndex("fake", "B", VECTORS, RECORDS)
    encoder = FakeEncoder()
    cfg = PipelineConfig(top_k=top_k, candidate_k=candidate_k, apply_filters=config.apply_filters, ranking=config.ranking)
    return Recommender(index, encoder, {"name": "fake", "config_hash": "h"}, cfg), encoder


def _ids(result):
    return [it["라벨링단위ID"] for it in result["추천"]]


def test_hard_filter_removes_violations_and_unknown_then_widens_search():
    rec, encoder = _recommender()
    result = rec.recommend("맵지 않고 따뜻한 음식")
    # 상위 4개(짬뽕, 콤비A, 콤비B, 고구마 미확인) 중 통과는 콤비A뿐, 콤비B는 중복 → 8개까지 넓혀 불고기·김밥 확보
    assert result["상태"] == STATUS_OK and result["검색범위"] == [4, 8]
    assert _ids(result) == ["u2", "u4", "u6"]
    assert result["필터제외사유"] == {"매운맛=강함": 1, "매운맛=미확인": 1, "매운맛=보통": 1}
    assert result["임베딩"]["name"] == "fake" and result["설정"]["top_k"] == 3
    assert encoder.calls == ["맵지 않고 따뜻한 음식"]
    assert any("중복: 콤비네이션 피자" == e["제외사유"] for e in result["제외"])
    assert result_metrics(result)["조건위반수"] == 0


def test_embedding_only_keeps_violations_and_metrics_count_them():
    rec, _ = _recommender(EMBEDDING_ONLY)
    result = rec.recommend("맵지 않고 따뜻한 음식")
    assert _ids(result) == ["u1", "u2", "u3"]
    metrics = result_metrics(result)
    assert metrics["조건위반수"] == 1 and metrics["중복메뉴수"] == 1 and metrics["대표식품명반복수"] == 1
    assert metrics["메뉴군반복수"] == 1
    assert result["필터제외"] == 0 and result["검색범위"] == [4]


def test_shortage_reports_actual_count_and_reason_without_relaxing():
    rec, _ = _recommender(top_k=5)
    result = rec.recommend("국물 없는 매운 음식")
    # 국물없음 필수: 피자 4개와 김밥만 통과, 중복·상한(2) 적용 후 3개
    assert result["상태"] == STATUS_SHORTAGE and result["검색범위"][-1] == len(RECORDS)
    assert result["반환수"] == 3 and result["요청수"] == 5
    assert all(it["라벨"]["국물"] == "국물없음" for it in result["추천"])
    assert "3건만 남음" in result["사유"]
    metrics = result_metrics(result)
    assert metrics["반환수"] == 3 and metrics["조건위반수"] == 0


def test_reranking_prefers_matching_preferences_over_similarity():
    rec, _ = _recommender(FILTER_ONLY, candidate_k=8)
    by_similarity = _ids(rec.recommend("차가운 국물 요리"))
    assert by_similarity[0] == "u0"

    rec, _ = _recommender(FULL, candidate_k=8)
    result = rec.recommend("차가운 국물 요리")
    assert _ids(result)[0] == "u7"
    top = result["추천"][0]
    assert top["선호점수"] == 1.0 and abs(top["최종점수"] - (0.7 * top["유사도"] + 0.3)) < 1e-3  # 메뉴 언급 없음
    assert "선호 제공온도=차가움 일치(차가운)" in top["추천근거"] and "유사도" in top["추천근거"]


def test_preference_shortfall_widens_search_up_to_limit():
    rec, _ = _recommender(FULL, candidate_k=2, top_k=2)
    result = rec.recommend("차가운 국물 요리")
    # 상위 2개(육개장, 짬뽕)는 선호 절반만 일치 -> 4, 8로 넓혀 냉면(완전 일치) 확보
    assert result["검색범위"] == [2, 4, 8] and result["확장사유"] == "선호 조건 일치 후보 부족"
    assert _ids(result)[0] == "u7"

    limited = rec.recommend("차가운 국물 요리", PipelineConfig(top_k=2, candidate_k=2, preference_widen_k=2))
    assert limited["검색범위"] == [2] and "u7" not in _ids(limited)

    baseline = rec.recommend("차가운 국물 요리", PipelineConfig(top_k=2, candidate_k=2, apply_filters=False,
                                                          ranking=EMBEDDING_ONLY.ranking))
    assert baseline["검색범위"] == [2]  # 선호 가중치 0이면 넓히지 않는다


def test_zero_candidate_k_terminates_and_widen_limit_is_a_cap():
    rec, _ = _recommender(FULL, candidate_k=0, top_k=2)
    result = rec.recommend("맵지 않고 따뜻한 음식")
    assert result["검색범위"][0] == 1 and result["반환수"] == 2

    rec, _ = _recommender(FULL, candidate_k=3, top_k=2)
    result = rec.recommend("차가운 국물 요리", PipelineConfig(top_k=2, candidate_k=3, preference_widen_k=4))
    assert result["검색범위"] == [3, 4] and result_metrics(result)["확장횟수"] == 1

    with pytest.raises(ValueError, match="비어"):
        Recommender(CandidateIndex("empty", "B", np.zeros((0, 8), dtype=np.float32), []), FakeEncoder())


def test_contradiction_and_empty_query_return_nothing_without_encoding():
    rec, encoder = _recommender()
    result = rec.recommend("맵지 않은 매운 음식")
    assert result["상태"] == STATUS_CONTRADICTION and result["추천"] == [] and "모순" in result["사유"]
    result = rec.recommend("   ")
    assert result["상태"] == STATUS_EMPTY_QUERY and result["추천"] == []
    assert encoder.calls == []
    assert result_metrics(result)["반환수"] == 0


def test_mentioned_menu_is_exempt_from_group_cap():
    rec, _ = _recommender(top_k=3, candidate_k=8)
    capped = rec.recommend("맵지 않고 따뜻한 음식")
    assert sum(it["대표식품명"] == "피자" for it in capped["추천"]) == 2
    pizza = rec.recommend("피자")
    assert [it["대표식품명"] for it in pizza["추천"]] == ["피자", "피자", "피자"]
    assert "u3" not in _ids(pizza)  # 브랜드만 다른 같은 메뉴는 여전히 중복 제거


def test_menu_exclusion_filters_and_counts_violations():
    rec, _ = _recommender(top_k=2, candidate_k=8)
    result = rec.recommend("피자 말고 김밥")
    assert [it["대표식품명"] for it in result["추천"]] == ["김밥", "냉면"]
    assert result["필터제외사유"] == {"메뉴=피자": 4}
    assert result_metrics(result)["조건위반수"] == 0

    baseline = rec.recommend("피자 말고 김밥", dataclasses.replace(EMBEDDING_ONLY, top_k=2))
    assert baseline["추천"][0]["대표식품명"] == "피자"
    assert result_metrics(baseline)["조건위반수"] == 2


def test_results_are_deterministic_and_rows_flatten():
    rec, _ = _recommender()
    first, second = rec.recommend("맵지 않고 따뜻한 음식"), rec.recommend("맵지 않고 따뜻한 음식")
    assert _ids(first) == _ids(second)
    rows = result_rows(first, 모드="full")
    assert rows[0]["모드"] == "full" and rows[0]["순위"] == 1 and "라벨" not in rows[0]
    assert {"메뉴명", "업체명", "주요라벨", "유사도", "선호점수", "최종점수", "추천근거"} <= set(rows[0])


def test_compatibility_check_flags_model_and_source_mismatch():
    good = {"config": {"model_id": "intfloat/multilingual-e5-base",
                       "model_revision": "d128750597153bb5987e10b1c3493a34e5a4502a", "dimension": 768,
                       "normalized": True, "pooling": "mean", "max_seq_length": 512,
                       "query_prefix": "query: ", "document_prefix": "passage: ", "text_spec_version": "v1",
                       "source_files": {"a": "1", "b": "2"}}}
    assert check_compatibility(good, sources={"a": "1", "b": "2"}) == []
    stale = {"config": {**good["config"], "text_spec_version": "v0"}}
    assert any("text_spec_version" in p for p in check_compatibility(stale, sources={"a": "1", "b": "2"}))
    bad = {"config": {**good["config"], "dimension": 384, "source_files": {"a": "1", "b": "x"}}}
    problems = check_compatibility(bad, sources={"a": "1", "b": "2"})
    assert any("dimension" in p for p in problems) and any("b 해시" in p for p in problems)


def test_load_index_excludes_franchise_by_default(tmp_path):
    store = EmbeddingStore(tmp_path)
    records = [{"라벨링단위ID": "u1", "메뉴명": "콤비네이션 피자", "업체명": "A피자"},
               {"라벨링단위ID": "u2", "메뉴명": "육개장", "업체명": ""},
               {"라벨링단위ID": "u3", "메뉴명": "김밥", "업체명": "-"}]
    vectors = np.eye(3, dtype=np.float32)[:, :3].copy()
    vectors = np.pad(vectors, ((0, 0), (0, 765))).astype(np.float32)
    config = build_config(DEFAULT_SPEC.as_dict(), "B", TEXT_SPEC_VERSION,
                          compute_input_hash(["u1", "u2", "u3"], ["a", "b", "c"]), {"f": "h"})
    from src.embedding import result_name
    store.save(result_name(config), vectors, records, config)

    index, ref = load_index("B", store, sources={"f": "h"})
    assert [r["라벨링단위ID"] for r in index.records] == ["u2", "u3"] and index.vectors.shape == (2, 768)
    assert ref["후보범위"] == {"프랜차이즈포함": False, "후보수": 2, "전체수": 3}
    assert not is_franchise(records[2])  # "-"는 빈 업체명

    full, ref = load_index("B", store, sources={"f": "h"}, include_franchise=True)
    assert full.size == 3 and ref["후보범위"]["프랜차이즈포함"] is True


def test_pipeline_config_is_serialisable():
    cfg = PipelineConfig(ranking=RankingConfig(group_cap=3))
    rec, _ = _recommender()
    result = rec.recommend("피자", cfg)
    assert result["설정"]["ranking"]["group_cap"] == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        RankingConfig().group_cap = 1
