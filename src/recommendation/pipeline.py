"""전체 추천 파이프라인

사용자 입력 -> 조건 추출 -> 임베딩 후보 검색 -> 필수 조건 필터 -> 선호 재랭킹
-> 중복·다양성 제어 -> Top-K

후보가 부족하면 검색 범위를 두 배씩 넓히되 필수 조건은 완화하지 않는다.
전체를 훑어도 부족하면 실제 반환 개수와 사유를 결과에 남긴다.
선호 조건을 모두 만족하는 항목이 top_k개 미만이면 preference_widen_k까지만 범위를 넓힌다.
"""

import time
from dataclasses import asdict, dataclass, field

from src.labeling.schema import UNKNOWN
from src.preprocessing import parse_query
from src.ranking import (
    RankingConfig, apply_hard_filters, apply_menu_exclusions, group_of, is_duplicate, mentions, score_candidates,
    select_top_k,
)
from src.retrieval import CandidateIndex, retrieve

STATUS_OK = "ok"
STATUS_EMPTY_QUERY = "empty_query"
STATUS_CONTRADICTION = "contradiction"
STATUS_SHORTAGE = "shortage"


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 5
    candidate_k: int = 100
    preference_widen_k: int = 400   # 선호 일치 항목이 부족할 때 넓히는 상한, candidate_k 이하면 넓히지 않음
    apply_filters: bool = True
    ranking: RankingConfig = field(default_factory=RankingConfig)


# 비교 실험용 사전 설정. preference_weight=0이면 선호 재랭킹과 선호 부족 확장이 모두 꺼진다
EMBEDDING_ONLY = PipelineConfig(apply_filters=False, ranking=RankingConfig(
    similarity_weight=1.0, preference_weight=0.0, menu_match_weight=0.0, group_cap=0, group_penalty=0.0,
    collapse_duplicates=False))
FILTER_ONLY = PipelineConfig(apply_filters=True, ranking=RankingConfig(
    similarity_weight=1.0, preference_weight=0.0, menu_match_weight=0.0, group_cap=0, group_penalty=0.0,
    collapse_duplicates=False))
FULL = PipelineConfig()


class Recommender:
    def __init__(self, index: CandidateIndex, encode_query, embedding_ref=None, config: PipelineConfig = FULL):
        """
        Args:
            index: src.retrieval.load_index 결과
            encode_query: 문장 -> (dim,) 정규화 벡터. E5Embedder.encode_queries를 감싼 함수
            embedding_ref: 결과에 남길 임베딩 출처 (manifest_ref)
        """
        if index.size == 0:
            raise ValueError("비어 있는 임베딩 인덱스입니다")
        self.index = index
        self.encode_query = encode_query
        self.embedding_ref = embedding_ref or {"name": index.name}
        self.config = config
        self._query_cache = {}

    def _vector(self, text):
        if text not in self._query_cache:
            if len(self._query_cache) >= 1000:  # ponytail: 단순 상한, LRU가 필요하면 교체
                self._query_cache.clear()
            self._query_cache[text] = self.encode_query(text)
        return self._query_cache[text]

    def recommend(self, text, config: PipelineConfig = None) -> dict:
        config = config or self.config
        t_start = time.perf_counter()
        parsed = parse_query(text)
        result = {
            "질의": parsed.text, "조건": parsed.to_dict(), "설정": asdict(config), "임베딩": self.embedding_ref,
            "상태": STATUS_OK, "사유": None, "요청수": config.top_k, "반환수": 0,
            "검색범위": [], "확장사유": None, "후보수": 0, "필터통과": 0, "필터제외": 0, "필터제외사유": {},
            "추천": [], "제외": [], "실행시간": {},
        }
        if parsed.is_empty:
            result.update(상태=STATUS_EMPTY_QUERY, 사유="입력이 비어 있음")
            return self._finish(result, t_start)
        if parsed.contradictions:
            names = ", ".join(f"{c['attribute']}({c['evidence']})" for c in parsed.contradictions)
            result.update(상태=STATUS_CONTRADICTION, 사유=f"필수 조건이 서로 모순됨: {names}")
            return self._finish(result, t_start)

        t0 = time.perf_counter()
        query_vector = self._vector(parsed.text)
        result["실행시간"]["질의임베딩"] = time.perf_counter() - t0

        hard = parsed.hard if config.apply_filters else []
        excluded_menus = [e["term"] for e in parsed.menu_exclusions] if config.apply_filters else []
        # 사용자가 언급한 메뉴의 메뉴군은 상한을 면제한다 (대표식품명·메뉴명 어절·대분류 어절 일치, 부분 문자열은 쓰지 않음)
        boost_terms = parsed.menu_terms + [c["term"] for c in parsed.context_terms if c["term"] not in parsed.menu_terms]
        exempt = {group_of(r, config.ranking.group_key) for r in self.index.records
                  if any(mentions(r, term) for term in boost_terms)} if boost_terms else set()
        widen_for_soft = bool(parsed.soft) and config.ranking.preference_weight > 0
        soft_limit = min(config.preference_widen_k, self.index.size)

        k = min(max(config.candidate_k, 1), self.index.size)
        while True:
            t0 = time.perf_counter()
            candidates = retrieve(self.index, query_vector, k)
            result["실행시간"]["검색"] = result["실행시간"].get("검색", 0) + time.perf_counter() - t0
            result["검색범위"].append(k)

            t0 = time.perf_counter()
            kept, dropped = apply_hard_filters(candidates, hard)
            kept, dropped_menu = apply_menu_exclusions(kept, excluded_menus)
            dropped += dropped_menu
            scored = score_candidates(kept, parsed.soft, config.ranking, boost_terms)
            selected, skipped = select_top_k(scored, config.top_k, config.ranking, exempt)
            result["실행시간"]["필터랭킹"] = result["실행시간"].get("필터랭킹", 0) + time.perf_counter() - t0

            if len(selected) < config.top_k and k < self.index.size:
                result["확장사유"], limit = "필수 조건 통과 후보 부족", self.index.size
            elif widen_for_soft and k < soft_limit and sum(c["선호점수"] >= 1.0 for c in selected) < config.top_k:
                result["확장사유"], limit = "선호 조건 일치 후보 부족", soft_limit
            else:
                break
            k = min(k * 2, limit)

        drop_reasons = {}
        for d in dropped:
            for r in d["제외사유"]:
                key = f"{r['attribute']}={r['value']}"
                drop_reasons[key] = drop_reasons.get(key, 0) + 1

        result.update(후보수=len(candidates), 필터통과=len(kept), 필터제외=len(dropped),
                      필터제외사유=dict(sorted(drop_reasons.items(), key=lambda kv: -kv[1])),
                      추천=[self._item(rank, c, hard) for rank, c in enumerate(selected, 1)],
                      제외=[{"메뉴명": s["record"]["메뉴명"], "업체명": s["record"].get("업체명") or "-",
                            "최종점수": round(s["최종점수"], 4), "제외사유": s["제외사유"]} for s in skipped],
                      반환수=len(selected))
        if len(selected) < config.top_k:
            result.update(상태=STATUS_SHORTAGE, 사유=(
                f"전체 {self.index.size}건 중 필수 조건 통과 {len(kept)}건, 중복·상한 제외 후 {len(selected)}건만 남음"))
        return self._finish(result, t_start)

    @staticmethod
    def _finish(result, t_start):
        result["실행시간"]["전체"] = time.perf_counter() - t_start
        return result

    @staticmethod
    def _item(rank, cand, hard):
        record = cand["record"]
        labels = record.get("라벨") or {}
        basis = [f"유사도 {cand['유사도']:.4f}"]
        basis += [f"필수 {c.attribute}={labels.get(c.attribute)} 충족({c.evidence})" for c in hard]
        basis += [f"선호 {m['attribute']}={m['value']} 일치({m['evidence']})" for m in cand["선호일치"]]
        basis += [f"선호 {m['attribute']}={m['value']} 불일치({m['evidence']})" for m in cand["선호불일치"]]
        basis += [f"선호 {m['attribute']} 미확인({m['evidence']})" for m in cand["선호미확인"]]
        if cand.get("메뉴일치"):
            basis.append(f"언급 메뉴 {cand['메뉴일치']} 일치 가점")
        return {
            "순위": rank, "라벨링단위ID": record["라벨링단위ID"], "메뉴명": record["메뉴명"],
            "업체명": record.get("업체명") or "-", "대표식품명": record.get("대표식품명"),
            "식품대분류명": record.get("식품대분류명"), "계열": labels.get("계열"),
            "주요라벨": ", ".join(record.get("속성토큰") or []) or "-", "라벨": labels,
            "유사도": round(cand["유사도"], 4), "선호점수": round(cand["선호점수"], 4),
            "최종점수": round(cand["최종점수"], 4), "추천근거": " / ".join(basis),
            "검토상태": record.get("검토상태"), "라벨출처": record.get("라벨출처"),
        }


def result_metrics(result) -> dict:
    """비교 실험용 관찰 지표, 조건 준수는 저장된 모델 추정 라벨 기준"""
    cond = result["조건"]
    hard, soft = cond["hard"], cond["soft"]
    items = result["추천"]
    value = lambda it, c: it["라벨"].get(c["attribute"]) or UNKNOWN
    violations = sum(1 for it in items for c in hard if value(it, c) not in c["allowed"])
    violations += sum(1 for it in items for e in cond["menu_exclusions"] if mentions(it, e["term"]))
    unknown = sum(1 for it in items if any(value(it, c) == UNKNOWN for c in [*hard, *soft]))
    pref_miss = sum(1 for it in items for c in soft if value(it, c) not in c["allowed"] and value(it, c) != UNKNOWN)
    dup = sum(1 for i, a in enumerate(items) if any(is_duplicate(a, b) for b in items[:i]))
    groups = [it["대표식품명"] for it in items]
    families = [group_of(it) for it in items]
    return {
        "질의": result["질의"], "상태": result["상태"], "요청수": result["요청수"], "반환수": result["반환수"],
        "필수조건수": len(hard), "선호조건수": len(soft), "조건위반수": violations, "미확인포함수": unknown,
        "선호불일치수": pref_miss, "중복메뉴수": dup, "대표식품명반복수": len(groups) - len(set(groups)),
        "메뉴군반복수": len(families) - len(set(families)),
        "확장횟수": max(len(result["검색범위"]) - 1, 0),
        "검색범위": result["검색범위"][-1] if result["검색범위"] else 0, "필터제외": result["필터제외"],
        "실행시간초": round(result["실행시간"].get("전체", 0), 4),
    }


def result_rows(result, **extra) -> list:
    """추천 항목을 표 한 행씩으로 펼친다"""
    return [{**extra, "질의": result["질의"], "상태": result["상태"], **{k: v for k, v in it.items() if k != "라벨"}}
            for it in result["추천"]]
