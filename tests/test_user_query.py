"""사용자 조건 추출 규칙 테스트, 모델·API 호출 없음"""

from src.labeling.schema import LABEL_SCHEMA, UNKNOWN
from src.preprocessing import HARD, SOFT, parse_query, support_table
from src.preprocessing.user_query import RULES


def _hard(parsed):
    return {c.attribute: c.allowed for c in parsed.hard}


def _soft(parsed):
    return {c.attribute: c.allowed for c in parsed.soft}


# 긍정·부정

def test_negation_is_hard_and_positive_is_soft():
    parsed = parse_query("맵지 않고 따뜻한 음식")
    assert _hard(parsed) == {"매운맛": ("없음",)}
    assert _soft(parsed) == {"제공온도": ("뜨거움", "따뜻함")}
    assert parsed.hard[0].evidence == "맵지 않고" and parsed.hard[0].strength == HARD
    assert parsed.soft[0].strength == SOFT

    assert _soft(parse_query("매운 음식")) == {"매운맛": ("보통", "강함")}
    assert _hard(parse_query("매운 음식")) == {}


def test_various_negation_forms_do_not_flip_to_positive():
    for text in ("안 매운 음식", "매운 거 싫어", "매운 음식은 빼고", "안 맵게 해줘"):
        parsed = parse_query(text)
        assert _hard(parsed) == {"매운맛": ("없음",)}, text
        assert "매운맛" not in _soft(parsed), text

    assert _hard(parse_query("국물 없는 매운 음식")) == {"국물": ("국물없음",)}
    assert _soft(parse_query("국물 없는 매운 음식")) == {"매운맛": ("보통", "강함")}
    assert _hard(parse_query("느끼하지 않은 담백한 음식")) == {"기름짐": ("낮음", "보통")}
    assert _soft(parse_query("느끼하지 않은 담백한 음식")) == {"기름짐": ("낮음",), "매운맛": ("없음", "약함")}
    assert "튀김" not in _hard(parse_query("튀김 말고 구운 치킨"))["조리법"]


def test_substrings_do_not_trigger_soup_or_spicy():
    assert parse_query("국수 먹고 싶어").soft == []
    assert parse_query("탕수육 먹고 싶어").soft == []
    assert _soft(parse_query("따뜻한 국이나 찌개"))["국물"] == ("국물요리",)
    assert _soft(parse_query("매운탕"))["국물"] == ("국물요리",)


def test_hard_values_never_include_unknown():
    for text in ("맵지 않은", "국물 없는", "느끼하지 않은", "튀김 말고", "너무 맵지 않은", "차갑지 않은", "뜨겁지 않은"):
        for cond in parse_query(text).hard:
            assert UNKNOWN not in cond.allowed, text


# 확정하지 않는 표현

def test_double_negation_and_tolerance_are_unhandled():
    parsed = parse_query("안 매운 건 싫어")
    assert parsed.hard == [] and parsed.soft == []
    assert parsed.unhandled[0]["rule"] == "이중부정"

    parsed = parse_query("매운 것도 괜찮아")
    assert parsed.hard == [] and parsed.soft == []
    assert parsed.unhandled[0]["rule"] == "허용표현"


def test_tolerance_forms_and_lookalikes():
    for text in ("매운 거 상관없어", "매워도 돼요", "국물 없어도 돼"):
        parsed = parse_query(text)
        assert parsed.hard == [] and parsed.soft == [], text
        assert parsed.unhandled[0]["rule"] == "허용표현", text
    # "되게"는 허용 표현이 아니고, "안심"의 "안"은 부정이 아니다
    assert _soft(parse_query("매운 것도 되게 좋아")) == {"매운맛": ("보통", "강함")}
    assert parse_query("매운 것도 되게 좋아").unhandled == []
    assert parse_query("안심스테이크는 싫어").unhandled == []


def test_cool_soup_is_not_cold_temperature():
    parsed = parse_query("시원한 국물이 땡겨")
    assert "제공온도" not in _soft(parsed)
    assert _soft(parsed) == {"국물": ("국물요리",)}
    assert parsed.unhandled[0]["rule"] == "시원한국물"
    assert _soft(parse_query("상큼하고 시원한 음식")) == {"제공온도": ("차가움",)}
    assert _soft(parse_query("시원한 국수")) == {"제공온도": ("차가움",)}
    assert parse_query("시원한 국수").unhandled == []


def test_unknown_taste_words_are_reported_not_guessed():
    parsed = parse_query("단짠단짠한 음식")
    assert parsed.hard == [] and parsed.soft == []
    assert [u["expression"] for u in parsed.unhandled] == ["단짠"]


# 맥락·메뉴 언급

def test_weather_mood_and_time_are_ignored_not_constraints():
    parsed = parse_query("비 오는 날 우울해서 저녁에 얼큰한 국물")
    assert {i["rule"] for i in parsed.ignored} == {"날씨", "기분", "시간"}
    assert parsed.hard == []
    assert _soft(parsed) == {"매운맛": ("보통", "강함"), "제공온도": ("뜨거움", "따뜻함"), "국물": ("국물요리",)}


def test_spicy_hot_word_implies_temperature_but_not_soup():
    assert _soft(parse_query("얼큰한 볶음")) == {"매운맛": ("보통", "강함"), "제공온도": ("뜨거움", "따뜻함"), "조리법": ("볶음",)}
    assert parse_query("국물 없는 얼큰한 음식").contradictions == []


def test_light_maps_only_to_fullness_label():
    parsed = parse_query("차갑고 가볍게 먹을 메뉴")
    assert _soft(parsed) == {"제공온도": ("차가움",), "든든함": ("가벼움",)}
    assert parsed.attributes <= set(LABEL_SCHEMA)


def test_menu_exclusions_are_separated_from_mentions():
    parsed = parse_query("피자 말고 버거")
    assert parsed.menu_terms == ["버거"] and [e["term"] for e in parsed.menu_exclusions] == ["피자"]
    assert parsed.hard == [] and parsed.soft == []
    assert [e["term"] for e in parse_query("치킨은 빼고 가볍게").menu_exclusions] == ["치킨"]
    assert [e["term"] for e in parse_query("라면 아닌 면").menu_exclusions] == ["라면"]
    assert parse_query("라면 아닌 면").menu_terms == ["면"]
    assert parse_query("차가운 면 요리").menu_terms == ["면"]
    assert parse_query("떡볶이 매운 거").menu_exclusions == []


def test_menu_terms_are_recorded_without_conditions():
    parsed = parse_query("피자 먹고 싶은데 느끼하지 않은 걸로")
    assert parsed.menu_terms == ["피자"]
    assert _hard(parsed) == {"기름짐": ("낮음", "보통")}
    assert parse_query("든든한 밥 한 끼").menu_terms == []  # '밥' 한 글자는 메뉴 언급으로 보지 않는다
    assert parse_query("김밥 먹고 싶어").menu_terms == ["김밥"]


def test_rejection_after_preference_inverts_first_condition():
    assert _hard(parse_query("따뜻한 거 싫어")) == {"제공온도": ("상온", "차가움")}
    assert _hard(parse_query("국물 있는 거 싫어")) == {"국물": ("국물약간", "국물없음")}
    assert _hard(parse_query("담백한 건 별로")) == {"기름짐": ("보통", "높음")}
    parsed = parse_query("얼큰한 거 빼고")
    assert _hard(parsed) == {"매운맛": ("없음", "약함")}  # 둘째 조건(제공온도)은 뒤집지 않는다
    for text in ("따뜻한 거 싫어", "국물 있는 거 싫어", "담백한 건 별로", "얼큰한 거 빼고"):
        assert parse_query(text).soft == [], text


def test_more_rejection_forms_and_particles():
    for text in ("매운 걸 빼고", "매운 거 아닌 걸로", "꼭 매운 거 아닌 걸로", "매운 거 안 좋아해", "매운 거 싫고 국물은 괜찮아"):
        assert _hard(parse_query(text)) == {"매운맛": ("없음",)}, text
        assert "매운맛" not in _soft(parse_query(text)), text
    assert _hard(parse_query("너무 매운 거 싫어")) == {"매운맛": ("없음", "약함", "보통")}
    assert _hard(parse_query("국물이 없는 음식")) == {"국물": ("국물없음",)}
    assert _hard(parse_query("국물은 없는 걸로")) == {"국물": ("국물없음",)}
    assert "튀김" not in _hard(parse_query("튀김류는 빼고"))["조리법"]
    assert parse_query("튀김류는 빼고").soft == []
    assert parse_query("단순한 메뉴").soft == []


def test_conflicting_preferences_are_dropped_and_overlapping_ones_intersect():
    parsed = parse_query("시원하고 얼큰한 국물")
    assert _soft(parsed) == {"매운맛": ("보통", "강함"), "국물": ("국물요리",)}
    assert any("상충" in u["reason"] for u in parsed.unhandled)
    assert _soft(parse_query("살짝 매운 매운 음식")) == {"매운맛": ("보통",)}


# 강조, 병합, 모순, 빈 입력

def test_force_words_promote_soft_to_hard():
    assert _hard(parse_query("꼭 매운 걸로")) == {"매운맛": ("보통", "강함")}
    assert _hard(parse_query("무조건 국물 요리")) == {"국물": ("국물요리",)}


def test_same_soft_conditions_are_merged_once():
    parsed = parse_query("따뜻한 국이나 찌개")
    soups = [c for c in parsed.soft if c.attribute == "국물"]
    assert len(soups) == 1 and "찌개" in soups[0].evidence and "국이" in soups[0].evidence


def test_contradictions_are_flagged_and_soft_is_narrowed_by_hard():
    parsed = parse_query("맵지 않은 매운 음식")
    assert parsed.contradictions == [{"attribute": "매운맛", "evidence": "맵지 않은 vs 매운"}]
    assert parsed.soft == []

    parsed = parse_query("맵지 않은데 꼭 매운 걸로")
    assert parsed.contradictions and parsed.hard == []

    assert parse_query("국물 없는 국물 요리").contradictions[0]["attribute"] == "국물"
    assert _soft(parse_query("너무 맵지 않은 매운 음식")) == {"매운맛": ("보통",)}
    assert parse_query("국물 없는 매운 음식").contradictions == []


def test_empty_input():
    for text in ("", "   ", None):
        parsed = parse_query(text)
        assert parsed.is_empty and parsed.hard == [] and parsed.soft == [] and parsed.contradictions == []
        assert parsed.summary() == "조건 없음"


# 문서화

def test_support_table_covers_every_rule_and_schema_values():
    rows = support_table()
    assert [r["규칙"] for r in rows] == [r.name for r in RULES]
    assert all(r["예시"] for r in rows)
    for rule in RULES:
        for attribute, allowed in rule.conditions:
            assert set(allowed) <= set(LABEL_SCHEMA[attribute]["values"])


def test_to_dict_is_json_friendly():
    d = parse_query("맵지 않고 따뜻한 음식").to_dict()
    assert d["hard"][0]["allowed"] == ("없음",) and d["text"] == "맵지 않고 따뜻한 음식"
    assert set(d) == {"text", "hard", "soft", "menu_terms", "menu_exclusions", "unhandled", "ignored", "contradictions"}
