"""유사도 기반 후보 검색 모듈.

- 저장된 임베딩 결과의 호환성 확인 및 로드
- 사용자 입력 임베딩과 음식 임베딩 간 유사도 상위 후보 검색
"""

from .candidates import (
    CandidateIndex, check_compatibility, find_result_name, is_franchise, load_index, manifest_ref, retrieve,
)

__all__ = ["CandidateIndex", "check_compatibility", "find_result_name", "is_franchise", "load_index", "manifest_ref",
           "retrieve"]
