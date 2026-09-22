"""음식 임베딩용 텍스트 구성

라벨링 단위 1건을 임베딩 입력 문자열 1개로 변환한다

구성 원칙
- 관리 정보(라벨링단위ID, 처리모델, 검토상태)는 넣지 않는다
- 라벨링 근거 문장은 추정 레시피가 섞여 있어 넣지 않는다
- 원본에 없는 날씨, 기분, 효능 설명을 만들어 넣지 않는다
- 값이 '미확인'인 속성은 텍스트에서 제외한다
- 업체명, 사이즈는 제외한다 (근거: docs/embedding/TEXT_DESIGN.md)
"""

TEXT_SPEC_VERSION = "v1"

UNKNOWN = "미확인"

# 텍스트 B에 포함하는 속성 및 출력 순서
ATTRIBUTE_ORDER = ("매운맛", "국물", "제공온도", "조리법", "기름짐", "든든함")

# 임베딩 텍스트에 절대 넣지 않는 관리 정보 및 자유 서술 필드
EXCLUDED_FIELDS = (
    "라벨링단위ID", "식품코드", "식품코드목록", "행수", "경계메뉴",
    "업체명", "프랜차이즈여부", "사이즈", "근거", "라벨출처",
    "처리방식", "모델", "처리구간", "샘플재사용", "검토상태", "검토메모",
)


def _clean(value):
    """문자열 정리 후 빈 값은 None 반환"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _dedup_keep_order(parts):
    """중복 토큰 제거, 입력 순서 유지"""
    seen = set()
    result = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            result.append(part)
    return result


def build_text_a(unit, label=None):
    """A: 음식명 + 대표식품명 + 분류"""
    parts = [
        _clean(unit.get("메뉴명")),
        _clean(unit.get("대표식품명")),
        _clean(unit.get("식품대분류명")),
    ]
    return " ".join(_dedup_keep_order(parts))


def build_text_b(unit, label=None):
    """B: A + 매운맛, 국물, 제공온도, 조리법, 기름짐, 든든함 (미확인 제외)"""
    parts = [build_text_a(unit)]
    parts.extend(format_attributes(label))
    return " ".join(p for p in parts if p)


def format_attributes(label):
    """라벨 딕셔너리 -> 속성 토큰 목록, 미확인 및 빈 값 제외

    값만으로는 뜻이 통하지 않으므로 속성명을 앞에 붙인다 ('없음' -> '매운맛 없음')
    값에 이미 속성명이 들어 있으면 값만 사용한다 ('국물없음', '든든함')
    """
    attributes = (label or {}).get("라벨") or {}
    tokens = []
    for name in ATTRIBUTE_ORDER:
        value = _clean(attributes.get(name))
        if value is None or value == UNKNOWN:
            continue
        tokens.append(value if name in value else f"{name} {value}")
    return tokens


TEXT_BUILDERS = {"A": build_text_a, "B": build_text_b}


def build_text(variant, unit, label=None):
    """variant('A' 또는 'B')에 맞는 텍스트 생성"""
    if variant not in TEXT_BUILDERS:
        raise ValueError(f"알 수 없는 텍스트 구성: {variant} (사용 가능: {sorted(TEXT_BUILDERS)})")
    return TEXT_BUILDERS[variant](unit, label)
