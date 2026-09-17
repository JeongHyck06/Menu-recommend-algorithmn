"""음식 속성 라벨 스키마

값이 바뀌면 SCHEMA_VERSION을 올림, 캐시 키에 포함되어 이전 결과와 구분됨
"""

SCHEMA_VERSION = "v1"

UNKNOWN = "미확인"

# 속성 -> 허용값 (UNKNOWN은 모든 속성에 추가로 허용)
LABEL_SCHEMA = {
    "매운맛": {
        "values": ["없음", "약함", "보통", "강함"],
        "description": "고추, 고춧가루, 고추장 등에서 오는 매운 정도",
    },
    "국물": {
        "values": ["국물요리", "국물약간", "국물없음"],
        "description": "국물요리는 국·탕·찌개처럼 국물이 중심, 국물약간은 소스나 자작한 국물",
    },
    "제공온도": {
        "values": ["뜨거움", "따뜻함", "상온", "차가움"],
        "description": "일반적으로 제공되는 온도",
    },
    "조리법": {
        "values": ["끓임", "볶음", "구이", "튀김", "찜", "조림", "부침", "오븐", "비조리", "혼합"],
        "description": "대표 조리 방식. 여러 방식이 동등하게 쓰이면 혼합",
    },
    "기름짐": {
        "values": ["낮음", "보통", "높음"],
        "description": "기름을 쓰는 조리법이나 지방이 많은 재료로 느껴지는 기름진 정도",
    },
    "든든함": {
        "values": ["가벼움", "보통", "든든함"],
        "description": "한 끼로서의 포만감",
    },
}

ATTRIBUTES = list(LABEL_SCHEMA)

# 모델 응답 최상위 필수 필드
REQUIRED_RESPONSE_FIELDS = ["식품코드", "라벨", "근거"]

MAX_REASON_LENGTH = 200

# 원본 온도 표기와 양립 가능한 제공온도 값
ORIGINAL_TEMPERATURE_COMPATIBLE = {
    "HOT": {"뜨거움", "따뜻함"},
    "ICED": {"차가움"},
}

# 라벨 출처 구분
SOURCE_ORIGINAL = "원본"
SOURCE_MODEL = "모델추정"


def allowed_values(attribute: str) -> list[str]:
    return [*LABEL_SCHEMA[attribute]["values"], UNKNOWN]


def schema_text() -> str:
    """프롬프트에 넣는 스키마 설명"""
    lines = []
    for attr, spec in LABEL_SCHEMA.items():
        lines.append(f"- {attr}: {' / '.join(allowed_values(attr))}. {spec['description']}")
    return "\n".join(lines)
