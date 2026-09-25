"""필터, 점수, 중복·다양성 제어 테스트, 모델·API 호출 없음"""

import random

import pytest

from src.preprocessing import HARD, SOFT, Condition
from src.ranking import (
    RankingConfig, apply_hard_filters, apply_menu_exclusions, group_of, is_duplicate, mentions, menu_key,
    preference_score, score_candidates, select_top_k,
)


def _cand(uid, menu, rep="피자", company="", sim=0.8, **labels):
    base = {"매운맛": "없음", "국물": "국물없음", "제공온도": "뜨거움", "조리법": "오븐", "기름짐": "높음", "든든함": "미확인"}
    base.update(labels)
    return {"유사도": sim, "record": {"라벨링단위ID": uid, "메뉴명": menu, "대표식품명": rep, "업체명": company, "라벨": base}}


def _cond(attribute, *allowed, strength=HARD, evidence="e"):
    return Condition(attribute, tuple(allowed), strength, evidence, "r")


# 필수 조건 필터

def test_hard_filter_rejects_unknown_and_disallowed_values():
    cands = [_cand("a", "순한피자", 매운맛="없음"), _cand("b", "불피자", 매운맛="강함"), _cand("c", "모름피자", 매운맛="미확인")]
    kept, dropped = apply_hard_filters(cands, [_cond("매운맛", "없음")])
    assert [c["record"]["라벨링단위ID"] for c in kept] == ["a"]
    assert [d["record"]["라벨링단위ID"] for d in dropped] == ["b", "c"]
    assert dropped[0]["제외사유"][0]["value"] == "강함" and dropped[0]["제외사유"][0]["unknown"] is False
    assert dropped[1]["제외사유"][0]["value"] == "미확인" and dropped[1]["제외사유"][0]["unknown"] is True


def test_hard_filter_without_conditions_keeps_everything():
    cands = [_cand("a", "x"), _cand("b", "y", 매운맛="미확인")]
    kept, dropped = apply_hard_filters(cands, [])
    assert len(kept) == 2 and dropped == []


# 선호 점수

def test_preference_score_separates_match_mismatch_unknown():
    labels = {"매운맛": "강함", "국물": "국물요리", "든든함": "미확인"}
    soft = [_cond("매운맛", "보통", "강함", strength=SOFT), _cond("국물", "국물없음", strength=SOFT),
            _cond("든든함", "든든함", strength=SOFT)]
    pref = preference_score(labels, soft)
    assert pref["score"] == 1 / 3
    assert [m["attribute"] for m in pref["matched"]] == ["매운맛"]
    assert [m["attribute"] for m in pref["unmatched"]] == ["국물"]
    assert [m["attribute"] for m in pref["unknown"]] == ["든든함"]
    assert preference_score(labels, [])["score"] == 0.0


def test_score_combines_weights_and_sorts_deterministically():
    soft = [_cond("매운맛", "강함", strength=SOFT)]
    config = RankingConfig(similarity_weight=0.5, preference_weight=0.5)
    cands = [_cand("b", "b", sim=0.80, 매운맛="없음"), _cand("a", "a", sim=0.80, 매운맛="없음"),
             _cand("c", "c", sim=0.70, 매운맛="강함"), _cand("d", "d", sim=0.81, 매운맛="없음")]
    expected = ["c", "d", "a", "b"]  # c: 0.35+0.5, d: 0.405, a/b: 0.40 동점은 ID 순
    for _ in range(5):
        random.shuffle(cands)
        scored = score_candidates(cands, soft, config)
        assert [c["record"]["라벨링단위ID"] for c in scored] == expected
    assert scored[0]["최종점수"] == 0.85 and scored[0]["선호점수"] == 1.0


# 중복

def test_menu_key_and_duplicate_rules():
    assert menu_key("페퍼로니 피자 (L)") == menu_key("페퍼로니피자 씬도우") == "페퍼로니피자"
    assert is_duplicate({"메뉴명": "콤비네이션 피자", "업체명": "A"}, {"메뉴명": "콤비네이션피자", "업체명": "B"})
    assert is_duplicate({"메뉴명": "매운 양념 치킨", "업체명": "A"}, {"메뉴명": "매운양념치킨 반마리", "업체명": "A"})
    assert is_duplicate({"메뉴명": "두부찌개", "업체명": ""}, {"메뉴명": "두부찌개 바지락", "업체명": ""})
    assert not is_duplicate({"메뉴명": "두부찌개", "업체명": "A"}, {"메뉴명": "두부찌개 바지락", "업체명": ""})
    assert not is_duplicate({"메뉴명": "불고기 피자", "업체명": "A"}, {"메뉴명": "고구마 피자", "업체명": "A"})
    assert not is_duplicate({"메뉴명": "", "업체명": "A"}, {"메뉴명": "", "업체명": "A"})
    # 공공 데이터는 어절 단위 접두어만 변형으로 본다
    assert is_duplicate({"메뉴명": "냉국 미역", "업체명": ""}, {"메뉴명": "냉국 미역 오이", "업체명": ""})
    assert not is_duplicate({"메뉴명": "순대", "업체명": ""}, {"메뉴명": "순대볶음 백순대", "업체명": ""})
    assert not is_duplicate({"메뉴명": "두부찌개", "업체명": ""}, {"메뉴명": "굴 두부찌개", "업체명": ""})
    assert not is_duplicate({"메뉴명": "순대", "업체명": "-"}, {"메뉴명": "순대볶음 백순대", "업체명": "-"})


def _scored(*cands):
    return score_candidates(list(cands), [], RankingConfig())


def test_select_collapses_duplicates_and_caps_groups():
    scored = _scored(
        _cand("p1", "콤비네이션 피자", "피자", "A", sim=0.90), _cand("p2", "콤비네이션피자", "피자", "B", sim=0.89),
        _cand("p3", "불고기 피자", "피자", "A", sim=0.88), _cand("p4", "고구마 피자", "피자", "C", sim=0.87),
        _cand("s1", "육개장", "육개장", "", sim=0.86), _cand("s2", "김밥", "김밥", "", sim=0.85),
    )
    selected, skipped = select_top_k(scored, 4, RankingConfig(group_cap=2))
    assert [c["record"]["라벨링단위ID"] for c in selected] == ["p1", "p3", "s1", "s2"]
    assert {s["record"]["라벨링단위ID"]: s["제외사유"] for s in skipped} == {
        "p2": "중복: 콤비네이션 피자", "p4": "대표식품명 상한(2): 피자"}


def test_select_exempts_mentioned_group_and_can_disable_controls():
    scored = _scored(*[_cand(f"p{i}", f"피자{i}", "피자", f"B{i}", sim=0.9 - i / 100) for i in range(5)])
    selected, _ = select_top_k(scored, 5, RankingConfig(group_cap=2), exempt_groups={"피자"})
    assert len(selected) == 5
    selected, _ = select_top_k(scored, 5, RankingConfig(group_cap=2))
    assert len(selected) == 2
    selected, _ = select_top_k(scored, 5, RankingConfig(group_cap=0))
    assert len(selected) == 5


def test_group_of_uses_last_token_for_unbranded_items():
    assert group_of({"대표식품명": "붕어 매운탕", "업체명": ""}) == "매운탕"
    assert group_of({"대표식품명": "붕어 매운탕", "업체명": "-"}) == "매운탕"
    assert group_of({"대표식품명": "두부찌개", "업체명": ""}) == "두부찌개"
    assert group_of({"대표식품명": "콤비네이션 피자", "업체명": "A"}) == "콤비네이션 피자"


def test_select_caps_unbranded_menu_family():
    scored = _scored(_cand("t1", "붕어 매운탕", "붕어 매운탕", "", sim=0.90), _cand("t2", "명태 매운탕", "명태 매운탕", "", sim=0.89),
                     _cand("t3", "꽃게 매운탕", "꽃게 매운탕", "", sim=0.88), _cand("k1", "김밥", "김밥", "", sim=0.87))
    selected, skipped = select_top_k(scored, 3, RankingConfig(group_cap=2))
    assert [c["record"]["라벨링단위ID"] for c in selected] == ["t1", "t2", "k1"]
    assert skipped[0]["제외사유"] == "대표식품명 상한(2): 매운탕"


def test_mentions_uses_whole_tokens_and_category():
    rec = {"대표식품명": "닭튀김", "메뉴명": "매운 양념 치킨", "식품대분류명": "튀김류"}
    assert mentions(rec, "치킨") and not mentions(rec, "치")
    assert mentions({"대표식품명": "피자", "메뉴명": "콤비네이션피자", "식품대분류명": "빵 및 과자류"}, "피자")
    assert mentions({"대표식품명": "비빔밥", "메뉴명": "육회비빔밥", "식품대분류명": "밥류"}, "밥")  # 밥 -> 밥류 별칭
    assert not mentions({"대표식품명": "김치찌개", "메뉴명": "김치찌개", "식품대분류명": "찌개 및 전골류"}, "밥")
    assert mentions({"대표식품명": "국수", "메뉴명": "잔치국수", "식품대분류명": "면 및 만두류"}, "면")
    assert mentions({"대표식품명": "국밥", "메뉴명": "국밥 순대국밥", "식품대분류명": "밥류"}, "밥")
    assert mentions({"대표식품명": "부추전", "메뉴명": "부추전", "식품대분류명": "전·적 및 부침류"}, "전")
    assert mentions({"대표식품명": "부추전", "메뉴명": "부추전", "식품대분류명": "전·적 및 부침류"}, "전·적 및 부침류")


def test_menu_exclusions_drop_matching_candidates_with_reason():
    cands = [_cand("p", "콤비네이션 피자", "피자", "A"), _cand("b", "치즈 버거", "버거", "B")]
    kept, dropped = apply_menu_exclusions(cands, ["피자"])
    assert [c["record"]["라벨링단위ID"] for c in kept] == ["b"]
    assert dropped[0]["제외사유"][0] == {"attribute": "메뉴", "value": "피자", "allowed": [], "evidence": "피자 제외", "unknown": False}
    assert apply_menu_exclusions(cands, [])[1] == []


def test_menu_match_bonus_only_when_weight_set():
    cands = [_cand("p", "콤비네이션 피자", "피자", "A", sim=0.90), _cand("n", "잔치국수", "국수", "", sim=0.85)]
    cands[1]["record"]["식품대분류명"] = "면 및 만두류"
    plain = score_candidates(cands, [], RankingConfig(menu_match_weight=0.0), menu_terms=["면"])
    assert [c["record"]["라벨링단위ID"] for c in plain] == ["p", "n"] and plain[0]["메뉴일치"] is None
    boosted = score_candidates(cands, [], RankingConfig(menu_match_weight=0.3), menu_terms=["면"])
    assert [c["record"]["라벨링단위ID"] for c in boosted] == ["n", "p"] and boosted[0]["메뉴일치"] == "면"
    assert boosted[0]["최종점수"] == pytest.approx(0.7 * 0.85 + 0.3)


def test_group_penalty_promotes_other_groups():
    scored = _scored(_cand("p1", "피자1", "피자", "A", sim=0.90), _cand("p2", "피자2", "피자", "B", sim=0.89),
                     _cand("s1", "육개장", "육개장", "", sim=0.885))
    selected, _ = select_top_k(scored, 2, RankingConfig(group_cap=0, group_penalty=0.01))
    assert [c["record"]["라벨링단위ID"] for c in selected] == ["p1", "s1"]
    selected, _ = select_top_k(scored, 2, RankingConfig(group_cap=0, group_penalty=0.0))
    assert [c["record"]["라벨링단위ID"] for c in selected] == ["p1", "p2"]
