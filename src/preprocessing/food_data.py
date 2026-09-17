"""공공 음식 영양성분 데이터 정제.

원본 CSV -> 메뉴 그룹 분류 -> 이름 정제 -> 중량 분리 -> 컬럼 선별 -> 중복 통합
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

NOT_APPLICABLE = "해당없음"

# 식품대분류명 기준 메뉴 그룹
MEAL_CATEGORIES = [
    "밥류",
    "면 및 만두류",
    "국 및 탕류",
    "찌개 및 전골류",
    "죽 및 스프류",
    "볶음류",
    "구이류",
    "튀김류",
    "찜류",
    "조림류",
    "전·적 및 부침류",
]
SIDE_DISH_CATEGORIES = [
    "생채·무침류",
    "나물·숙채류",
    "김치류",
    "장아찌·절임류",
    "젓갈류",
    "장류, 양념류",
]
DESSERT_CATEGORIES = ["빵 및 과자류", "유제품류 및 빙과류"]
BEVERAGE_CATEGORIES = ["음료 및 차류"]

# 빵 및 과자류 중 식사로 취급하는 대표식품명
MEAL_BREAD_REPRESENTATIVES = ["피자", "버거", "햄버거", "샌드위치", "핫도그", "토스트"]

MENU_GROUP_MEAL = "식사"
MENU_GROUP_SIDE = "반찬"
MENU_GROUP_DESSERT = "디저트"
MENU_GROUP_BEVERAGE = "음료"
MENU_GROUP_OTHER = "기타"

# 식품명 괄호 안 사이즈 표기
SIZE_TOKENS = {
    "S", "M", "L", "XL", "R", "F", "P", "J", "V", "G", "H", "EX", "ML", "Max",
    "Short", "Tall", "Grande", "Venti", "Mini Venti", "더벤티", "Solo",
    "소", "중", "대", "초대용량",
}

TEMPERATURE_PATTERN = re.compile(r"(?:핫|아이스)?\s*\((HOT|ICED)\)")
PAREN_PATTERN = re.compile(r"\(([^()]*)\)")
WEIGHT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(g|ml)\s*$", re.IGNORECASE)

NUTRITION_COLUMNS_ALL = [
    "에너지(kcal)", "수분(g)", "단백질(g)", "지방(g)", "회분(g)", "탄수화물(g)", "당류(g)",
    "식이섬유(g)", "칼슘(mg)", "철(mg)", "인(mg)", "칼륨(mg)", "나트륨(mg)",
    "비타민 A(μg RAE)", "레티놀(μg)", "베타카로틴(μg)", "티아민(mg)", "리보플라빈(mg)",
    "니아신(mg)", "비타민 C(mg)", "비타민 D(μg)", "콜레스테롤(mg)", "포화지방산(g)",
    "트랜스지방산(g)",
]

# 추천용으로 유지하는 영양성분
NUTRITION_COLUMNS_SELECTED = [
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)",
    "나트륨(mg)", "콜레스테롤(mg)", "포화지방산(g)",
]

OUTPUT_COLUMNS = [
    "식품코드", "식품명", "메뉴명", "이름접두어", "온도", "사이즈",
    "메뉴그룹", "식품대분류명", "대표식품명", "식품중분류명",
    "식품기원명", "업체명", "프랜차이즈여부", "출처명", "데이터생성방법명",
    "영양성분함량기준량", "중량값", "중량단위",
    *NUTRITION_COLUMNS_SELECTED,
]

# 중복 판단 기준: 이름, 업체, 기준량, 중량, 전체 영양성분이 모두 같은 행
DEDUP_KEY_COLUMNS = ["식품명", "업체명", "영양성분함량기준량", "식품중량", *NUTRITION_COLUMNS_ALL]


def load_raw(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def assign_menu_group(df: pd.DataFrame) -> pd.Series:
    category = df["식품대분류명"]
    representative = df["대표식품명"]

    group = pd.Series(MENU_GROUP_OTHER, index=df.index, dtype="object")
    group[category.isin(MEAL_CATEGORIES)] = MENU_GROUP_MEAL
    group[category.isin(SIDE_DISH_CATEGORIES)] = MENU_GROUP_SIDE
    group[category.isin(DESSERT_CATEGORIES)] = MENU_GROUP_DESSERT
    group[category.isin(BEVERAGE_CATEGORIES)] = MENU_GROUP_BEVERAGE

    meal_bread = (category == "빵 및 과자류") & representative.isin(MEAL_BREAD_REPRESENTATIVES)
    group[meal_bread] = MENU_GROUP_MEAL
    return group


def parse_food_name(name: str, is_franchise: bool) -> dict:
    """식품명 -> 메뉴명, 이름접두어, 온도, 사이즈.

    프랜차이즈: 첫 '_' 앞은 메뉴 카테고리 접두어로 분리
    비프랜차이즈: '_'는 재료 변형 구분자이므로 공백으로 치환
    """
    prefix = None
    rest = name
    if is_franchise and "_" in name:
        prefix, rest = name.split("_", 1)
        prefix = prefix.strip()

    temperature = None
    match = TEMPERATURE_PATTERN.search(rest)
    if match:
        temperature = match.group(1)
        rest = TEMPERATURE_PATTERN.sub(" ", rest)

    size = None

    def _pop_size(m: re.Match) -> str:
        nonlocal size
        token = m.group(1).strip()
        if token in SIZE_TOKENS and size is None:
            size = token
            return " "
        return m.group(0)

    rest = PAREN_PATTERN.sub(_pop_size, rest)
    menu_name = re.sub(r"\s+", " ", rest.replace("_", " ")).strip()
    if not menu_name:
        menu_name = name.strip()

    return {"메뉴명": menu_name, "이름접두어": prefix, "온도": temperature, "사이즈": size}


def add_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    is_franchise = df["프랜차이즈여부"]
    parsed = pd.DataFrame(
        [parse_food_name(n, f) for n, f in zip(df["식품명"], is_franchise)],
        index=df.index,
    )
    return pd.concat([df, parsed], axis=1)


def parse_weight(series: pd.Series) -> pd.DataFrame:
    """'291.90ml' -> 중량값 291.9, 중량단위 'ml'. 형식이 다르면 둘 다 NaN."""
    extracted = series.astype("string").str.extract(WEIGHT_PATTERN)
    value = pd.to_numeric(extracted[0], errors="coerce").astype("float64")
    unit = extracted[1].str.lower().astype("object").where(extracted[1].notna(), np.nan)
    return pd.DataFrame({"중량값": value, "중량단위": unit}, index=series.index)


def replace_not_applicable(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].where(out[col] != NOT_APPLICABLE, np.nan)
    return out


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DEDUP_KEY_COLUMNS 기준 중복 행 통합.

    식품코드 오름차순 첫 행 유지 -> 나머지 행은 대표 식품코드에 매핑
    반환: (유지 행, 매핑표[식품코드, 대표식품코드, 식품명, 식품기원명, 업체명])
    """
    sorted_df = df.sort_values("식품코드", kind="stable")
    representative = sorted_df.groupby(DEDUP_KEY_COLUMNS, dropna=False, sort=False)["식품코드"].transform("first")
    keep_mask = sorted_df["식품코드"] == representative

    dropped = sorted_df.loc[~keep_mask, ["식품코드", "식품명", "식품기원명", "업체명"]].copy()
    dropped.insert(1, "대표식품코드", representative[~keep_mask].values)

    kept = sorted_df.loc[keep_mask].sort_index()
    return kept, dropped.reset_index(drop=True)


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 -> (정제 데이터, 중복 매핑표). 원본은 수정하지 않는다."""
    df = raw.copy()
    df["프랜차이즈여부"] = df["업체명"] != NOT_APPLICABLE
    df["메뉴그룹"] = assign_menu_group(df)
    df = add_name_columns(df)
    df = pd.concat([df, parse_weight(df["식품중량"])], axis=1)

    df, dedup_map = deduplicate(df)

    df = replace_not_applicable(df, ["업체명", "식품중분류명"])
    return df[OUTPUT_COLUMNS].reset_index(drop=True), dedup_map


def select_meal_menu(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["메뉴그룹"] == MENU_GROUP_MEAL].reset_index(drop=True)
