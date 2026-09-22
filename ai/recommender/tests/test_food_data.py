"""음식 데이터 정제 로직 테스트"""

import numpy as np
import pandas as pd
import pytest

from ai.recommender.preprocessing import food_data as fd


@pytest.mark.parametrize(
    "name, is_franchise, expected",
    [
        (
            "커피_카페 라떼 핫(HOT) (L)",
            True,
            {"메뉴명": "카페 라떼", "이름접두어": "커피", "온도": "HOT", "사이즈": "L"},
        ),
        (
            "커피_아메리카노 아이스(ICED)",
            True,
            {"메뉴명": "아메리카노", "이름접두어": "커피", "온도": "ICED", "사이즈": None},
        ),
        (
            "피자_치즈 피자 (L)",
            True,
            {"메뉴명": "치즈 피자", "이름접두어": "피자", "온도": None, "사이즈": "L"},
        ),
        # 사이즈가 아닌 괄호 표기는 이름에 유지
        (
            "케이크_레몬치즈 케이크 (조각)",
            True,
            {"메뉴명": "레몬치즈 케이크 (조각)", "이름접두어": "케이크", "온도": None, "사이즈": None},
        ),
        # 비프랜차이즈 '_'는 재료 변형 구분자 -> 공백
        (
            "된장국_근대",
            False,
            {"메뉴명": "된장국 근대", "이름접두어": None, "온도": None, "사이즈": None},
        ),
        ("흰죽", False, {"메뉴명": "흰죽", "이름접두어": None, "온도": None, "사이즈": None}),
    ],
)
def test_parse_food_name(name, is_franchise, expected):
    assert fd.parse_food_name(name, is_franchise) == expected


def test_parse_weight_splits_value_and_unit():
    series = pd.Series(["291.90ml", "100g", "473ML", np.nan, "1인분"])
    result = fd.parse_weight(series)

    assert result["중량값"].tolist()[:3] == [291.9, 100.0, 473.0]
    assert result["중량단위"].tolist()[:3] == ["ml", "g", "ml"]
    assert result.iloc[3].isna().all()
    assert result.iloc[4].isna().all()


def test_assign_menu_group():
    df = pd.DataFrame(
        {
            "식품대분류명": ["국 및 탕류", "김치류", "빵 및 과자류", "빵 및 과자류", "음료 및 차류", "과일류"],
            "대표식품명": ["된장국", "배추김치", "케이크", "피자", "커피", "사과"],
        }
    )
    assert fd.assign_menu_group(df).tolist() == ["식사", "반찬", "디저트", "식사", "음료", "기타"]


@pytest.mark.parametrize(
    "representative, expected_side",
    [
        ("잔멸치볶음", True),
        ("김치볶음", True),
        ("달걀말이", True),
        ("두부조림", True),
        ("감자튀김", True),
        # 식사 키워드가 있어도 강한 반찬 키워드 우선
        ("소고기 장조림", True),
        ("고구마맛탕", True),
        # 반찬 키워드(감자, 두부, 소시지)가 있어도 식사 키워드로 유지
        ("감자그라탕", False),
        ("마파두부", False),
        ("두부 탕수", False),
        ("제육볶음", False),
        ("고등어구이", False),
        ("닭튀김", False),
        ("돼지갈비찜", False),
    ],
)
def test_is_side_dish(representative, expected_side):
    assert fd.is_side_dish(representative) is expected_side


def test_side_dish_rule_applies_only_to_cooking_categories():
    df = pd.DataFrame(
        {
            "식품대분류명": ["볶음류", "밥류", "국 및 탕류", "면 및 만두류"],
            "대표식품명": ["김치볶음", "김치 볶음밥", "달걀국", "어묵 우동"],
        }
    )
    assert fd.assign_menu_group(df).tolist() == ["반찬", "식사", "식사", "식사"]


def _raw_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "식품명": "흰죽",
        "업체명": "해당없음",
        "식품기원명": "가정식(분석 함량)",
        "영양성분함량기준량": "100g",
        "식품중량": "200g",
        "식품대분류명": "죽 및 스프류",
        "대표식품명": "흰죽",
        "식품중분류명": "해당없음",
        "출처명": "식품의약품안전처",
        "데이터생성방법명": "산출",
        **{c: 1.0 for c in fd.NUTRITION_COLUMNS_ALL},
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_deduplicate_maps_dropped_rows_to_representative():
    df = _raw_frame(
        [
            {"식품코드": "D5", "식품기원명": "초등학교급식"},
            {"식품코드": "D4", "식품기원명": "산업체급식"},
            {"식품코드": "D6", "식품명": "흰죽", "업체명": "A카페"},
            {"식품코드": "D7", "에너지(kcal)": 2.0},
            {"식품코드": "D8", "회분(g)": np.nan},
            {"식품코드": "D9", "회분(g)": np.nan},
        ]
    )
    kept, dedup_map = fd.deduplicate(df)

    assert sorted(kept["식품코드"]) == ["D4", "D6", "D7", "D8"]
    assert dedup_map[["식품코드", "대표식품코드"]].values.tolist() == [["D5", "D4"], ["D9", "D8"]]
    assert dedup_map.loc[0, "식품기원명"] == "초등학교급식"


def test_preprocess_keeps_raw_and_handles_not_applicable():
    raw = _raw_frame(
        [
            {"식품코드": "D1"},
            {"식품코드": "D2", "식품명": "커피_라떼 핫(HOT)", "업체명": "A카페", "식품중분류명": "카페라떼",
             "식품대분류명": "음료 및 차류", "식품중량": "355ml", "지방(g)": np.nan},
        ]
    )
    snapshot = raw.copy()
    clean, dedup_map = fd.preprocess(raw)

    pd.testing.assert_frame_equal(raw, snapshot)
    assert list(clean.columns) == fd.OUTPUT_COLUMNS
    assert dedup_map.empty

    home = clean.loc[clean["식품코드"] == "D1"].iloc[0]
    cafe = clean.loc[clean["식품코드"] == "D2"].iloc[0]
    assert pd.isna(home["업체명"]) and pd.isna(home["식품중분류명"])
    assert not home["프랜차이즈여부"] and cafe["프랜차이즈여부"]
    assert cafe["업체명"] == "A카페" and cafe["식품중분류명"] == "카페라떼"
    assert (cafe["메뉴명"], cafe["온도"], cafe["중량값"], cafe["중량단위"]) == ("라떼", "HOT", 355.0, "ml")
    assert pd.isna(cafe["지방(g)"])
