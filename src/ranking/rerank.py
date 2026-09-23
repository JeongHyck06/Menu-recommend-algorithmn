"""속성 조건 필터, 선호 점수, 재랭킹, 중복·다양성 제어

입력 후보는 src/retrieval의 {"유사도", "record"} 딕셔너리이며 record["라벨"]은
저장된 모델 추정 라벨이다. 라벨이 '미확인'이면 필수 조건을 충족한 것으로 보지 않는다.
"""

import re
from collections import Counter
from dataclasses import dataclass

from src.labeling.schema import UNKNOWN


@dataclass(frozen=True)
class RankingConfig:
    similarity_weight: float = 0.7
    preference_weight: float = 0.3
    group_key: str = "대표식품명"
    group_cap: int = 2            # Top-K 안에서 같은 그룹 최대 수, 0이면 제한 없음
    group_penalty: float = 0.0    # 이미 선택된 같은 그룹 수 × 감점
    collapse_duplicates: bool = True


# 최종 점수 = similarity_weight × 유사도 + preference_weight × 선호점수
# ponytail: 유사도는 원값(0.8~0.9 대역)이라 선호 가중치가 사실상 우선한다, 정규화가 필요하면 후보 내 min-max 추가


def apply_hard_filters(candidates, hard) -> tuple:
    """필수 조건 적용

    Returns:
        (kept, dropped) dropped 항목에는 "제외사유"가 붙는다
    """
    kept, dropped = [], []
    for cand in candidates:
        labels = cand["record"].get("라벨") or {}
        reasons = []
        for cond in hard:
            value = labels.get(cond.attribute) or UNKNOWN
            if value not in cond.allowed:
                reasons.append({"attribute": cond.attribute, "value": value,
                                "allowed": list(cond.allowed), "evidence": cond.evidence,
                                "unknown": value == UNKNOWN})
        if reasons:
            dropped.append({**cand, "제외사유": reasons})
        else:
            kept.append(cand)
    return kept, dropped


def preference_score(labels, soft) -> dict:
    """선호 조건 일치 비율과 근거"""
    matched, unmatched, unknown = [], [], []
    for cond in soft:
        value = (labels or {}).get(cond.attribute) or UNKNOWN
        entry = {"attribute": cond.attribute, "value": value, "evidence": cond.evidence}
        if value == UNKNOWN:
            unknown.append(entry)
        elif value in cond.allowed:
            matched.append(entry)
        else:
            unmatched.append(entry)
    score = len(matched) / len(soft) if soft else 0.0
    return {"score": score, "matched": matched, "unmatched": unmatched, "unknown": unknown}


def score_candidates(candidates, soft, config: RankingConfig) -> list:
    """유사도·선호점수·최종점수 계산 후 결정적 정렬 (최종점수, 유사도 내림차순, ID 오름차순)"""
    scored = []
    for cand in candidates:
        pref = preference_score(cand["record"].get("라벨"), soft)
        final = config.similarity_weight * cand["유사도"] + config.preference_weight * pref["score"]
        scored.append({**cand, "선호점수": pref["score"], "최종점수": final,
                       "선호일치": pref["matched"], "선호불일치": pref["unmatched"], "선호미확인": pref["unknown"]})
    scored.sort(key=lambda c: (-round(c["최종점수"], 9), -round(c["유사도"], 9), c["record"]["라벨링단위ID"]))
    return scored


# 브랜드·사이즈·도우 표기 차이로 갈라진 같은 메뉴를 묶는 정규화
VARIANT_TOKENS = ("리치골드크러스트", "치즈크러스트", "골드크러스트", "크러스트", "씬도우", "밀도우",
                  "오리지널", "라지", "레귤러", "미디움", "반마리", "1인")
_STRIP = re.compile(r"\([^)]*\)|[\s\-_·,./&+]+")


def _company(record) -> str:
    """업체명, 결과 행에서는 빈 업체명을 "-"로 표시하므로 되돌린다"""
    value = record.get("업체명") or ""
    return "" if value == "-" else value


def menu_key(name) -> str:
    key = _STRIP.sub("", str(name or "")).lower()
    for token in VARIANT_TOKENS:
        key = key.replace(token, "")
    return key


def is_duplicate(a, b) -> bool:
    """같은 메뉴의 변형인지 판정

    - 정규화 이름이 같으면 업체가 달라도 같은 메뉴 ("콤비네이션 피자" / "콤비네이션피자")
    - 같은 프랜차이즈 안에서는 띄어쓰기가 불규칙해 정규화 키의 접두어 관계로 본다 ("매운양념치킨" / "매운양념치킨반마리")
    - 공공 데이터(업체명 없음)는 어절 단위 접두어일 때만 변형으로 본다
      ("두부찌개" / "두부찌개 바지락"은 변형, "순대" / "순대볶음 백순대"는 다른 메뉴)
    """
    ka, kb = menu_key(a.get("메뉴명")), menu_key(b.get("메뉴명"))
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    company = _company(a)
    if company != _company(b):
        return False
    if company:
        return ka.startswith(kb) or kb.startswith(ka)
    ta = [menu_key(t) for t in str(a.get("메뉴명")).split()]
    tb = [menu_key(t) for t in str(b.get("메뉴명")).split()]
    n = min(len(ta), len(tb))
    return ta[:n] == tb[:n]


def group_of(record, key="대표식품명") -> str:
    """다양성 상한에 쓰는 그룹 키

    프랜차이즈 항목은 대표식품명(피자, 버거)이 이미 넓은 묶음이지만, 업체명 없는 공공 데이터는
    대표식품명이 곧 메뉴명("붕어 매운탕")이라 마지막 어절("매운탕")로 묶는다
    """
    value = str(record.get(key) or "")
    if _company(record):
        return value
    tokens = value.split()
    return tokens[-1] if tokens else value


def select_top_k(scored, k, config: RankingConfig, exempt_groups=()) -> tuple:
    """정렬된 후보에서 중복·그룹 상한·그룹 감점을 적용해 k개 선택

    exempt_groups에 든 그룹(사용자가 직접 언급한 메뉴)은 상한과 감점을 받지 않는다

    Returns:
        (selected, skipped) skipped 항목에는 "제외사유"가 붙는다
    """
    selected, skipped = [], []
    counts = Counter()
    remaining = list(scored)
    while remaining and len(selected) < k:
        if config.group_penalty:
            best = max(range(len(remaining)), key=lambda i: (
                remaining[i]["최종점수"] - (0 if group_of(remaining[i]["record"], config.group_key) in exempt_groups
                                         else config.group_penalty * counts[group_of(remaining[i]["record"], config.group_key)]),
                -i))
            cand = remaining.pop(best)
        else:
            cand = remaining.pop(0)
        record = cand["record"]
        group = group_of(record, config.group_key)
        dup = next((s for s in selected if is_duplicate(record, s["record"])), None) if config.collapse_duplicates else None
        if dup is not None:
            skipped.append({**cand, "제외사유": f"중복: {dup['record']['메뉴명']}"})
        elif config.group_cap and group not in exempt_groups and counts[group] >= config.group_cap:
            skipped.append({**cand, "제외사유": f"{config.group_key} 상한({config.group_cap}): {group}"})
        else:
            counts[group] += 1
            selected.append(cand)
    return selected, skipped
