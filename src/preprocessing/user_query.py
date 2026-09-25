"""사용자 자연어 입력에서 명시적 선호·제외 조건 추출 (규칙 기반)

음식 라벨을 만드는 작업(src/labeling)과 구분한다. 여기서는 사용자 문장만 읽고
저장된 라벨 스키마(src/labeling/schema.py)의 속성·값에 대응하는 조건만 뽑는다.
문장 자체는 바꾸지 않으며, 임베딩에는 원문을 그대로 쓴다.

지원 범위는 RULES 표가 유일한 출처다. support_table()로 표 형태로 확인한다.

조건 강도
- 필수(hard): 부정·제외 표현("맵지 않은", "국물 없는", "느끼하지 않은", "튀김 말고")과
  강조어("꼭", "반드시", "무조건")가 붙은 표현. 허용값 집합으로 표현하며
  '미확인' 라벨은 허용값에 포함되지 않으므로 필수 조건을 충족하지 못한다.
- 선호(soft): 긍정 표현("매운", "따뜻한", "든든한", "가벼운"). 점수에만 반영한다.

확정하지 않는 표현 (unhandled로 기록)
- 이중 부정: "안 매운 건 싫어"
- 허용·관용: "매운 것도 괜찮아", "매워도 돼"
- 온도인지 국물 맛인지 모호한 "시원한 국물"
- 라벨 스키마에 없는 맛: "단짠", "상큼한", "고소한"

무시하는 맥락 (ignored로 기록)
- 날씨, 기분, 시간대는 필수 조건으로 바꾸지 않는다. 임베딩 유사도에만 남는다.

메뉴 언급
- "피자", "치킨", "면"처럼 메뉴 종류를 말하면 menu_terms에 기록한다. 조건이 아니며 랭킹에서 상한 면제·가점에 쓴다.
- "피자 말고", "치킨은 빼고"처럼 메뉴를 제외하면 menu_exclusions에 기록하고 그 메뉴는 필수 제외한다.
- "밥"을 단독으로 말하면 밥류 언급과 한식 선호로 해석한다. 맨밥은 후보에서 빠진다.
- 날씨는 조건이 아니지만 연상 메뉴(비: 전·칼국수·수제비, 추움: 국·찌개, 더움: 냉면·냉국)에 가점을 준다.
- 계열(한식/중식/일식/양식/동남아/분식)과 안주는 원본에 없는 추정값이며 선호 조건으로만 쓴다.

해석하지 않는 것
- "가벼운"은 저장된 든든함 라벨 '가벼움'에만 대응하며 칼로리로 해석하지 않는다.
- 재료 데이터가 없으므로 알레르기·채식 표현은 조건으로 만들지 않는다.
"""

import re
from dataclasses import asdict, dataclass, field

from src.labeling.schema import LABEL_SCHEMA, UNKNOWN
from src.preprocessing.cuisine import ANJU_NO, ANJU_YES, CUISINES

HARD = "hard"
SOFT = "soft"
UNHANDLED = "unhandled"
CONTEXT = "context"

ALL = {attr: tuple(spec["values"]) for attr, spec in LABEL_SCHEMA.items()}
ALL["계열"] = CUISINES          # 키워드 규칙 또는 채팅 라벨로 붙인 추정값 (src/preprocessing/cuisine.py)
ALL["안주"] = (ANJU_YES, ANJU_NO)
_STANDALONE_RICE = r"(?<![가-힣])밥(?=\s|$|[을이도만,.!?]|으로|이나|이랑)"

# 표현 뒤 조사·명사 + 거부어: "매운 거 싫어", "매운 음식은 빼고", "매운 걸 아닌 걸로", "국물 있는 거 싫어"
_REJECT = (r"\s*(?:있는|들어간)?\s*(?:걸|거|것들|것|건|음식|메뉴|맛|류)?\s*[은는이가]?\s*"
           r"(?:싫|빼|말고|못\s*먹|아닌|안\s*좋|별로)")
REJECT_PATTERN = re.compile(_REJECT)


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    pattern: str
    conditions: tuple = ()  # ((속성, 허용값 튜플), ...)
    reason: str = ""
    example: str = ""
    associations: tuple = ()  # 맥락 규칙이 연상하는 메뉴·분류, 선호 가점에만 쓴다

    @property
    def regex(self):
        # 조건 규칙은 근거 표현이 자연스럽도록 뒤따르는 어미까지 포함한다 ("맵지 않" -> "맵지 않고")
        tail = "(?:하고|하게|한|은|는|운|고|게|이|의|을|아|어)?" if self.kind in (HARD, SOFT) else ""
        return re.compile(f"(?:{self.pattern}){tail}")


def _without(attribute, *excluded):
    return tuple(v for v in ALL[attribute] if v not in excluded)


# 우선순위 순서. 앞 규칙이 소비한 구간은 뒤 규칙이 다시 보지 않는다.
RULES = (
    # 확정하지 않는 표현
    Rule("이중부정", UNHANDLED,
         r"(?:안\s*(?:매[운워]|맵|느끼|기름|뜨거|차가|튀긴)[가-힣]*|[가-힣]+지\s*않[은는]|[가-힣]+\s*없는)"
         r"\s*(?:건|것|거|게|음식|메뉴)?[은는]?\s*(?:싫|별로|안\s*좋)",
         reason="이중 부정은 방향을 확정하지 않음", example="안 매운 건 싫어"),
    Rule("허용표현", UNHANDLED,
         r"(?:(?<![가-힣])(?!싫|빼|말고|별로)[가-힣]+\s+)?[가-힣]+\s*(?:거|것|건|음식|메뉴)?(?:도|어도|아도|여도)?\s*"
         r"(?:괜찮|상관\s*없|(?:돼요?|됩니다|되(?:요|죠|지)?)(?![가-힣]))",
         reason="허용·관용 표현은 필수·선호 조건으로 확정하지 않음", example="매운 것도 괜찮아, 매워도 돼"),
    Rule("시원한국물", UNHANDLED, r"시원(?:한|하고|하게)(?=\s*(?:국물|국(?!수)|탕|찌개|해장))",
         reason="'시원한 국물'은 온도가 아닐 수 있어 확정하지 않음", example="시원한 국물"),
    Rule("스키마외맛", UNHANDLED, r"단짠|달콤|달달|새콤|상큼|짭짤|고소|쫄깃|쫀득|부드러운|촉촉",
         reason="라벨 스키마에 없는 맛·식감, 임베딩 유사도에만 맡김", example="단짠단짠한 음식"),

    # 필수 조건 (부정·제외)
    Rule("매운맛_강함제외", HARD, r"너무\s*(?:맵지\s*않|안\s*매[운워]|안\s*맵|매운" + _REJECT + ")",
         (("매운맛", ("없음", "약함", "보통")),), example="너무 맵지 않은, 너무 매운 거 싫어"),
    Rule("매운맛_제외", HARD, r"맵지\s*않|안\s*매[운워]|안\s*맵|매운" + _REJECT,
         (("매운맛", ("없음",)),), example="맵지 않은, 안 매운, 매운 거 싫어"),
    Rule("국물_제외", HARD, r"국물\s*[이은는]?\s*(?:없|안\s*(?:있|들어)|빼|말고|싫)",
         (("국물", ("국물없음",)),), example="국물 없는, 국물이 없는, 국물 빼고"),
    Rule("뜨거움_제외", HARD, r"뜨겁지\s*않|안\s*뜨거|뜨거운" + _REJECT,
         (("제공온도", _without("제공온도", "뜨거움")),), example="뜨겁지 않은"),
    Rule("차가움_제외", HARD, r"차갑지\s*않|안\s*차가|차가운" + _REJECT,
         (("제공온도", _without("제공온도", "차가움")),), example="차갑지 않은"),
    Rule("기름짐_제외", HARD,
         r"느끼하지\s*않|안\s*느끼|느끼한" + _REJECT + r"|기름지지\s*않|안\s*기름|기름기\s*[가은는]?\s*(?:없|적)|기름진" + _REJECT,
         (("기름짐", ("낮음", "보통")),), example="느끼하지 않은, 기름기 적은"),
    Rule("튀김_제외", HARD, r"튀기지\s*않|안\s*튀긴|튀김\s*(?:류)?\s*[은는이]?\s*(?:말고|빼|싫|안\s*먹)",
         (("조리법", _without("조리법", "튀김")),), example="튀김 말고, 튀김류는 빼고"),

    # 선호 조건 (긍정). 앞에 꼭/반드시/무조건이 붙으면 필수로 올리고,
    # 뒤에 거부어("따뜻한 거 싫어")가 붙으면 첫 조건을 뒤집어 필수 제외로 만든다
    Rule("매운맛_강함", SOFT, r"(?:아주|엄청|완전|진짜|매우|겁나|많이)\s*(?:매운|매콤|맵게|얼큰)",
         (("매운맛", ("강함",)),), example="아주 매운"),
    Rule("매운맛_약함", SOFT, r"(?:살짝|약간|조금|덜)\s*(?:매운|매콤|맵게|얼큰)",
         (("매운맛", ("약함", "보통")),), example="살짝 매운"),
    Rule("얼큰", SOFT, r"얼큰|칼칼",
         (("매운맛", ("보통", "강함")), ("제공온도", ("뜨거움", "따뜻함"))), example="얼큰한 (매운 + 뜨거운)"),
    Rule("매운맛", SOFT, r"매운|매콤|맵게|맵고|매워|얼얼|화끈",
         (("매운맛", ("보통", "강함")),), example="매운, 매콤한"),
    Rule("순한", SOFT, r"(?<![가-힣])순(?:한|하게)", (("매운맛", ("없음", "약함")),), example="순한"),
    Rule("국물", SOFT,
         r"국물|국밥|찌개|전골|해장국|국(?=[이에\s,]|$|이나|이랑|을|도)|탕(?=[이에\s,]|$|이나|이랑|을|도)",
         (("국물", ("국물요리",)),), example="국물, 국이나 찌개"),
    Rule("따뜻함", SOFT, r"따뜻|따끈|뜨끈|뜨거|뜨겁|따신",
         (("제공온도", ("뜨거움", "따뜻함")),), example="따뜻한, 뜨끈한"),
    Rule("차가움", SOFT, r"차가|차갑|차게|시원|아이스",
         (("제공온도", ("차가움",)),), example="차가운, 시원한"),
    Rule("기름짐_높음", SOFT, r"기름진|기름지게|기름기\s*많|느끼한|느끼하게|느끼하고",
         (("기름짐", ("높음",)),), example="기름진"),
    Rule("담백", SOFT, r"담백",
         (("기름짐", ("낮음",)), ("매운맛", ("없음", "약함"))), example="담백한 (기름지지 않고 자극 없는)"),
    Rule("든든함", SOFT, r"든든|포만감|배부르|배불리|푸짐|양\s*많",
         (("든든함", ("든든함",)),), example="든든한, 포만감 있는"),
    Rule("가벼움", SOFT, r"가벼운|가볍게|가벼움|가볍고|가벼워|간단|간식",
         (("든든함", ("가벼움",)),), example="가볍게, 간식"),
    Rule("튀김", SOFT, r"튀긴|튀김|바삭", (("조리법", ("튀김",)),), example="바삭한"),
    Rule("구이", SOFT, r"구운|구이|그릴|바베큐|바비큐", (("조리법", ("구이", "오븐")),), example="구운"),
    Rule("볶음", SOFT, r"볶은|볶음", (("조리법", ("볶음",)),), example="볶음"),
    Rule("찜", SOFT, r"찜|찐", (("조리법", ("찜",)),), example="찜"),
    Rule("조림", SOFT, r"조림", (("조리법", ("조림",)),), example="조림"),
    Rule("끓임", SOFT, r"끓인|끓여", (("조리법", ("끓임",)),), example="끓인"),
    Rule("계열_한식", SOFT, r"한식|한국\s*음식|우리\s*음식", (("계열", ("한식",)),), example="한식 위주로"),
    Rule("계열_중식", SOFT, r"중식|중국\s*음식|중국집", (("계열", ("중식",)),), example="중식, 중국집"),
    Rule("계열_일식", SOFT, r"일식|일본\s*음식|일본식", (("계열", ("일식",)),), example="일식"),
    Rule("계열_양식", SOFT, r"양식|서양\s*음식", (("계열", ("양식",)),), example="양식"),
    Rule("계열_동남아", SOFT, r"동남아|베트남|태국", (("계열", ("동남아",)),), example="베트남 음식"),
    Rule("계열_분식", SOFT, r"분식", (("계열", ("분식",)),), example="분식"),
    Rule("안주", SOFT, r"안주|술\s*마시|술이랑|맥주|소주|막걸리", (("안주", (ANJU_YES,)),), example="술안주, 맥주랑"),
    Rule("밥_한식", SOFT, _STANDALONE_RICE, (("계열", ("한식",)),), example="밥 먹고 싶어 (밥은 한식 밥류로 해석)"),

    # 맥락: 기록만 하고 조건으로 쓰지 않는다
    Rule("날씨_비", CONTEXT, r"비\s*(?:오|와|가)|장마|비오는", reason="날씨는 필수 조건이 아님, 연상 메뉴에 가점만",
         example="비 오는 날", associations=("전·적 및 부침류", "칼국수", "수제비")),
    Rule("날씨_추움", CONTEXT, r"눈\s*(?:오|와)|쌀쌀|추운|추워|춥", reason="날씨는 필수 조건이 아님, 연상 메뉴에 가점만",
         example="추운 날", associations=("국 및 탕류", "찌개 및 전골류")),
    Rule("날씨_더움", CONTEXT, r"더운|더워|덥|습한|꿉꿉|폭염", reason="날씨는 필수 조건이 아님, 연상 메뉴에 가점만",
         example="더운 날", associations=("냉면", "냉국", "콩국수")),
    Rule("날씨", CONTEXT, r"날씨", reason="날씨는 필수 조건으로 쓰지 않음, 임베딩 유사도에만 반영", example="날씨 좋은 날"),
    Rule("기분", CONTEXT, r"기분|우울|피곤|힘든|힘들|스트레스|지친",
         reason="기분은 필수 조건으로 쓰지 않음, 임베딩 유사도에만 반영", example="우울할 때"),
    Rule("시간", CONTEXT, r"아침|점심|저녁|야식|주말|혼밥",
         reason="시간대·상황은 필수 조건으로 쓰지 않음, 임베딩 유사도에만 반영", example="저녁밥"),
)

# 조건이 아니라 메뉴 종류 언급. 구간을 소비하지 않고 별도로 기록한다
# 대표식품명·메뉴명 어절·식품대분류명 어절과 그대로 대조하므로 부분 문자열("밥")은 넣지 않는다
MENU_TERMS = (
    r"피자|버거|치킨|샌드위치|토스트|핫도그|파스타|스파게티|떡볶이|김밥|라면|국수|냉면|우동|만두|면"
    r"|볶음밥|비빔밥|덮밥|스테이크|돈까스|돈가스|카레|짜장|짬뽕|찌개|국밥|샐러드|부침개|수제비|칼국수"
    rf"|전(?=\s|$|[을이도만,.!?]|이나|이랑|하고)|(?<![가-힣])죽(?=\s|$|[을이도만,.!?]|이나|이랑)|{_STANDALONE_RICE}"
)
MENU_TERM_PATTERN = re.compile(MENU_TERMS)
# 메뉴 제외: "피자 말고", "치킨은 빼고", "피자 아닌" -> 그 메뉴를 필수 제외한다
MENU_EXCLUDE_PATTERN = re.compile(rf"({MENU_TERMS})\s*(?:류)?\s*[은는이가]?\s*(?:말고|빼고|빼|싫|아닌|제외|안\s*먹)")
FORCE_PATTERN = re.compile(r"(?:꼭|반드시|무조건)\s*$")


@dataclass(frozen=True)
class Condition:
    attribute: str
    allowed: tuple
    strength: str
    evidence: str
    rule: str


@dataclass
class ParsedQuery:
    text: str
    hard: list = field(default_factory=list)
    soft: list = field(default_factory=list)
    menu_terms: list = field(default_factory=list)
    menu_exclusions: list = field(default_factory=list)
    context_terms: list = field(default_factory=list)
    unhandled: list = field(default_factory=list)
    ignored: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def attributes(self) -> set:
        return {c.attribute for c in [*self.hard, *self.soft]}

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [f"필수 {c.attribute}={'/'.join(c.allowed)} ({c.evidence})" for c in self.hard]
        parts += [f"선호 {c.attribute}={'/'.join(c.allowed)} ({c.evidence})" for c in self.soft]
        parts += [f"메뉴 제외 {e['term']}({e['evidence']})" for e in self.menu_exclusions]
        parts += [f"연상 {c['term']}({c['evidence']})" for c in self.context_terms]
        parts += [f"미처리 '{u['expression']}'" for u in self.unhandled]
        parts += [f"모순 {c['attribute']}" for c in self.contradictions]
        return " | ".join(parts) or "조건 없음"


def _overlaps(span, consumed):
    return any(span[0] < e and s < span[1] for s, e in consumed)


def _dedup(items):
    """같은 표현·규칙 반복 제거, 순서 유지"""
    seen, out = set(), []
    for item in items:
        key = (item["expression"], item["rule"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _merge_soft(conditions):
    """같은 속성의 선호는 하나로 합친다

    허용값이 겹치면 교집합("살짝 매운"+"매운" -> 보통), 전혀 겹치지 않으면
    ("시원하고 얼큰한") 어느 쪽도 반영하지 않고 미처리로 기록한다

    Returns:
        (merged, dropped)
    """
    merged, dropped, conflicted = {}, [], set()
    for c in conditions:
        if c.attribute in conflicted:
            continue
        prev = merged.get(c.attribute)
        if prev is None:
            merged[c.attribute] = c
            continue
        common = tuple(v for v in ALL[c.attribute] if v in prev.allowed and v in c.allowed)
        evidence, rule = f"{prev.evidence}, {c.evidence}", f"{prev.rule}, {c.rule}"
        if common:
            merged[c.attribute] = Condition(c.attribute, common, SOFT, evidence, rule)
        else:
            del merged[c.attribute]
            conflicted.add(c.attribute)
            dropped.append({"expression": evidence, "rule": rule,
                            "reason": f"서로 상충하는 선호({c.attribute})는 반영하지 않음"})
    return list(merged.values()), dropped


def _merge_hard(conditions):
    """속성별 허용값 교집합, 비면 모순"""
    merged, contradictions = [], []
    for attribute in ALL:
        group = [c for c in conditions if c.attribute == attribute]
        if not group:
            continue
        allowed = tuple(v for v in ALL[attribute] if all(v in c.allowed for c in group))
        evidence = ", ".join(dict.fromkeys(c.evidence for c in group))
        rule = ", ".join(dict.fromkeys(c.rule for c in group))
        if allowed:
            merged.append(Condition(attribute, allowed, HARD, evidence, rule))
        else:
            contradictions.append({"attribute": attribute, "evidence": evidence})
    return merged, contradictions


def parse_query(text) -> ParsedQuery:
    parsed = ParsedQuery(text=text or "")
    cleaned = parsed.text.strip()
    if not cleaned:
        return parsed

    consumed, hard = [], []
    for rule in RULES:
        for m in rule.regex.finditer(cleaned):
            if _overlaps(m.span(), consumed):
                continue
            consumed.append(m.span())
            evidence = m.group(0)
            if rule.kind == UNHANDLED:
                parsed.unhandled.append({"expression": evidence, "reason": rule.reason, "rule": rule.name})
            elif rule.kind == CONTEXT:
                parsed.ignored.append({"expression": evidence, "reason": rule.reason, "rule": rule.name})
                parsed.context_terms += [{"term": t, "evidence": evidence} for t in rule.associations]
            else:
                conditions, name, end = rule.conditions, rule.name, m.end()
                reject = REJECT_PATTERN.match(cleaned, end) if rule.kind == SOFT else None
                if reject and not _overlaps((end, reject.end()), consumed):
                    attribute, allowed = conditions[0]
                    conditions, name, end = ((attribute, _without(attribute, *allowed)),), f"{rule.name}_거부", reject.end()
                    consumed[-1] = (m.start(), end)
                    strength = HARD
                else:
                    forced = bool(FORCE_PATTERN.search(cleaned[:m.start()]))
                    strength = HARD if rule.kind == HARD or forced else SOFT
                evidence = cleaned[m.start():end]
                for attribute, allowed in conditions:
                    cond = Condition(attribute, tuple(allowed), strength, evidence, name)
                    (hard if strength == HARD else parsed.soft).append(cond)

    exclusions = list(MENU_EXCLUDE_PATTERN.finditer(cleaned))
    parsed.menu_exclusions = [{"term": m.group(1), "evidence": m.group(0)} for m in exclusions]
    excluded_spans = [m.span() for m in exclusions]
    parsed.menu_terms = list(dict.fromkeys(
        m.group(0) for m in MENU_TERM_PATTERN.finditer(cleaned) if not _overlaps(m.span(), excluded_spans)))
    parsed.unhandled = _dedup(parsed.unhandled)
    parsed.ignored = _dedup(parsed.ignored)
    parsed.hard, parsed.contradictions = _merge_hard(hard)
    parsed.soft, conflicting = _merge_soft(parsed.soft)
    parsed.unhandled += conflicting

    # 같은 속성의 선호는 필수 허용값 안으로 좁힌다. 겹치는 값이 없으면 모순이다
    hard_allowed = {c.attribute: c.allowed for c in parsed.hard}
    kept = []
    for cond in parsed.soft:
        if cond.attribute in hard_allowed:
            narrowed = tuple(v for v in hard_allowed[cond.attribute] if v in cond.allowed)
            if not narrowed:
                hard_evidence = next(c.evidence for c in parsed.hard if c.attribute == cond.attribute)
                parsed.contradictions.append({"attribute": cond.attribute,
                                              "evidence": f"{hard_evidence} vs {cond.evidence}"})
                continue
            cond = Condition(cond.attribute, narrowed, SOFT, cond.evidence, cond.rule)
        kept.append(cond)
    parsed.soft = kept
    return parsed


def support_table() -> list:
    """지원 표현 목록, 문서화용"""
    rows = []
    for rule in RULES:
        rows.append({
            "종류": rule.kind, "규칙": rule.name, "예시": rule.example,
            "조건": "; ".join(f"{a}={'/'.join(v)}" for a, v in rule.conditions) or "-",
            "비고": rule.reason or "-",
        })
    return rows


def _check_rules():
    for rule in RULES:
        for attribute, allowed in rule.conditions:
            unknown = set(allowed) - set(ALL[attribute])
            assert not unknown and UNKNOWN not in allowed, f"{rule.name}: 스키마에 없는 값 {unknown}"


_check_rules()
