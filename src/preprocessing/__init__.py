"""자연어 입력 전처리 및 음식 데이터 정제 모듈.

- 사용자 자연어 입력에서 명시적 선호·제외 조건 추출 (user_query)
- 공공 음식 데이터 정제 및 정규화 (food_data)
"""

from .user_query import HARD, SOFT, Condition, ParsedQuery, parse_query, support_table

__all__ = ["HARD", "SOFT", "Condition", "ParsedQuery", "parse_query", "support_table"]
