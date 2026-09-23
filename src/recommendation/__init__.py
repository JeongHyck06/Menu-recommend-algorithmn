"""전체 추천 파이프라인 조합 모듈.

- 조건 추출 → 후보 검색 → 필터 → 재랭킹 → 중복 제어를 연결해 Top-K 메뉴 추천 결과 생성
"""

from .pipeline import (
    EMBEDDING_ONLY, FILTER_ONLY, FULL, STATUS_CONTRADICTION, STATUS_EMPTY_QUERY, STATUS_OK, STATUS_SHORTAGE,
    PipelineConfig, Recommender, result_metrics, result_rows,
)

__all__ = [
    "EMBEDDING_ONLY", "FILTER_ONLY", "FULL", "STATUS_CONTRADICTION", "STATUS_EMPTY_QUERY", "STATUS_OK",
    "STATUS_SHORTAGE", "PipelineConfig", "Recommender", "result_metrics", "result_rows",
]
