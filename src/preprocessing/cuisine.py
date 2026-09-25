"""계열(한식/중식/일식/양식/동남아/분식)과 안주 여부, 맨밥 판정

원본 데이터에는 계열 값이 없어 메뉴명·대표식품명·식품대분류명의 키워드 규칙으로 추정한다.
Claude 채팅으로 라벨링한 파일(CUISINE_LABELS_PATH)이 있으면 그 값을 우선 쓴다.
두 경우 모두 추정값이며 결과의 계열출처에 어느 쪽인지 남긴다.
"""

import json
import re
from pathlib import Path

CUISINES = ("한식", "중식", "일식", "양식", "동남아", "분식")
ANJU_YES, ANJU_NO = "예", "아니오"
CUISINE_LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "labeling" / "cuisine_labels.jsonl"

# 앞 규칙이 우선한다. 어느 것에도 안 걸리면 한식
CUISINE_RULES = (
    ("중식", re.compile(r"짜장|자장|짬뽕|탕수|깐풍|깐쇼|라조|유산슬|팔보채|마파|양장피|울면|기스면|난자완스|멘보샤|잡탕|잡채밥|마라|훠궈|동파육|꿔바로우|유린기|중식")),
    ("일식", re.compile(r"초밥|스시|회덮밥|사시미|우동|소바|메밀|돈까스|돈가스|까스|규동|미소|라멘|데리야끼|가라아게|텐동|카츠|카레|하이라이스|오므라이스|롤$|롤\s")),
    ("양식", re.compile(r"스파게티|파스타|스테이크|함박|햄버거|버거|피자|샌드위치|토스트|오믈렛|스튜|그라탕|리조또|샐러드|핫도그|크로크|수프|스프|폭찹|나폴리탄")),
    ("동남아", re.compile(r"쌀국수|분짜|팟타이|월남쌈|똠얌|나시|커리")),
    ("분식", re.compile(r"떡볶이|라볶이|쫄볶이|김밥|라면|쫄면|순대|어묵|핫바|김말이|컵밥")),
)
ANJU_CATEGORIES = {"전·적 및 부침류", "튀김류", "구이류"}
ANJU_PATTERN = re.compile(
    r"족발|보쌈|골뱅이|닭발|오징어|황태|노가리|먹태|두부김치|계란말이|치킨|닭강정|곱창|막창|대창|삼겹살|육회|나초|소시지|감바스|해물찜|아귀찜|조개|꼬치|안주|부침|전$")
STAPLE_PATTERN = re.compile(
    r"^(?:쌀밥|잡곡밥|현미밥|순 현미밥|보리밥|콩밥|차조밥|밤밥|혼합잡곡밥|기장밥|수수밥|흑미밥|오곡밥|찰밥|팥밥|율무밥|완두콩밥|귀리밥|검정콩밥|강낭콩밥|감자밥|고구마밥|옥수수밥|누룽지)$")


def _text(record) -> str:
    return f"{record.get('메뉴명') or ''} {record.get('대표식품명') or ''}"


def classify_cuisine(record) -> str:
    text = _text(record)
    return next((name for name, pattern in CUISINE_RULES if pattern.search(text)), "한식")


def is_anju(record) -> bool:
    return record.get("식품대분류명") in ANJU_CATEGORIES or bool(ANJU_PATTERN.search(_text(record)))


def is_staple(record) -> bool:
    """쌀밥, 잡곡밥처럼 반찬 없이 단독 메뉴가 되지 않는 맨밥"""
    return bool(STAPLE_PATTERN.match((record.get("대표식품명") or "").strip()))


def load_cuisine_labels(path=CUISINE_LABELS_PATH) -> dict:
    """채팅 라벨링 결과 {라벨링단위ID: {"계열", "안주"}}, 파일이 없으면 빈 dict"""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["라벨링단위ID"]] = row.get("라벨") or {}
    return out


def enrich(record, labels=None) -> dict:
    """record의 라벨과 속성토큰에 계열·안주를 더한 사본. labels(채팅 라벨)가 있으면 우선한다"""
    given = (labels or {}).get(record.get("라벨링단위ID"), {})
    cuisine = given.get("계열") if given.get("계열") in CUISINES else classify_cuisine(record)
    anju = given.get("안주") if given.get("안주") in (ANJU_YES, ANJU_NO) else (ANJU_YES if is_anju(record) else ANJU_NO)
    out = dict(record)
    out["라벨"] = {**(record.get("라벨") or {}), "계열": cuisine, "안주": anju}
    out["속성토큰"] = [*(record.get("속성토큰") or []), cuisine, *(["안주"] if anju == ANJU_YES else [])]
    out["계열출처"] = "채팅 라벨" if given else "키워드 규칙"
    return out
