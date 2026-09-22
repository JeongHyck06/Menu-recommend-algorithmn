"""문장 임베딩 및 음식 데이터 임베딩 모듈

- 라벨링 단위 기준 음식 정보와 라벨 연결
- 임베딩 텍스트 구성 (A: 기본 정보, B: A + 속성 라벨)
- 로컬 임베딩 모델 실행 및 벡터 저장, 로드
"""

from .dataset import join_units_labels, load_sources
from .embedder import (
    DEFAULT_SPEC, E5Embedder, ModelSpec, choose_batch_size, cosine_top_k, describe_environment, detect_device,
)
from .pipeline import build_records, load_joined, prepare_variant, run_variant, search, source_hashes
from .store import (
    EmbeddingStore, build_config, compute_config_hash, compute_input_hash, result_name, validate_vectors,
)
from .text_builder import TEXT_SPEC_VERSION, build_text, build_text_a, build_text_b, format_attributes

__all__ = [
    "join_units_labels", "load_sources",
    "DEFAULT_SPEC", "E5Embedder", "ModelSpec", "choose_batch_size", "cosine_top_k",
    "describe_environment", "detect_device",
    "build_records", "load_joined", "prepare_variant", "run_variant", "search", "source_hashes",
    "EmbeddingStore", "build_config", "compute_config_hash", "compute_input_hash", "result_name", "validate_vectors",
    "TEXT_SPEC_VERSION", "build_text", "build_text_a", "build_text_b", "format_attributes",
]
