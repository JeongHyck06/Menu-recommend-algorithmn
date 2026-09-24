"""음식 속성을 활용한 후보 랭킹 모듈.

- 필수 조건 필터 ('미확인'은 충족으로 보지 않음)
- 선호 조건 일치 점수와 유사도 가중 합산
- 같은 메뉴 변형 중복 제거, 대표식품명 상한·감점 다양성 제어
"""

from .rerank import (
    RankingConfig, apply_hard_filters, apply_menu_exclusions, group_of, is_duplicate, mentions, menu_key,
    preference_score, score_candidates, select_top_k,
)

__all__ = [
    "RankingConfig", "apply_hard_filters", "apply_menu_exclusions", "group_of", "is_duplicate", "mentions",
    "menu_key", "preference_score", "score_candidates", "select_top_k",
]
