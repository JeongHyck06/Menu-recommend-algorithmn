"""로컬 임베딩 모델 실행

모델 카드가 요구하는 접두어, pooling, 정규화 방식을 그대로 따른다
외부 유료 API를 호출하지 않는다
"""

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """모델 카드에서 확인한 실행 규격"""
    model_id: str
    revision: str
    dimension: int
    query_prefix: str
    document_prefix: str
    pooling: str
    normalize: bool
    max_seq_length: int
    license: str

    def as_dict(self):
        return asdict(self)

    @property
    def slug(self) -> str:
        """저장 경로에 쓰는 짧은 식별자, 'multilingual-e5-base_d1287505'"""
        return f"{self.model_id.split('/')[-1]}_{self.revision[:8]}"


# intfloat/multilingual-e5-base 모델 카드 확인 결과 (2026-09-22)
# - xlm-roberta-base 초기화, 100개 언어, 12 layer, 임베딩 768차원
# - Mr. TyDi ko MRR@10: small 55.4, base 56.6, large 62.5
# - MIT 라이선스, 278M 파라미터 (safetensors 약 1.06GB)
# - 모든 입력에 'query: ' 또는 'passage: ' 접두어 필요, 비영어 텍스트도 동일
# - mean pooling 후 L2 정규화
E5_BASE = ModelSpec(
    model_id="intfloat/multilingual-e5-base",
    revision="d128750597153bb5987e10b1c3493a34e5a4502a",
    dimension=768,
    query_prefix="query: ",
    document_prefix="passage: ",
    pooling="mean",
    normalize=True,
    max_seq_length=512,
    license="mit",
)

DEFAULT_SPEC = E5_BASE


def detect_device() -> str:
    """사용 가능한 가속 장치 확인, cuda -> mps -> cpu 순"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_environment() -> dict:
    """현재 실행 환경의 메모리 및 연산 장치 정보"""
    import os
    import platform

    import torch

    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total_bytes = None

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "total_memory_gb": round(total_bytes / 1024 ** 3, 1) if total_bytes else None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": detect_device(),
    }


def choose_batch_size(device: str, total_memory_gb=None) -> int:
    """장치와 메모리 기준 배치 크기 선택

    입력 텍스트가 80자 이하로 짧아 배치당 메모리는 작지만
    mps는 시스템 메모리를 공유하므로 8GB 이하 기기에서는 32로 제한한다
    """
    if device == "cuda":
        return 128
    if device == "mps":
        return 32 if (total_memory_gb or 0) <= 8 else 64
    return 16


class E5Embedder:
    """multilingual-e5 계열 모델 실행기

    sentence-transformers 설정에 mean pooling과 L2 정규화 모듈이 포함되어 있어
    접두어만 직접 붙이고 나머지는 모델 설정을 따른다
    """

    def __init__(self, spec: ModelSpec = DEFAULT_SPEC, device: str = None,
                 batch_size: int = None, model=None):
        """
        Args:
            model: encode()와 get_sentence_embedding_dimension()을 갖춘 객체,
                   None이면 spec 기준으로 SentenceTransformer를 로드한다
        """
        self.spec = spec
        self.device = device or (None if model is not None else detect_device())
        self.batch_size = batch_size or choose_batch_size(self.device or "cpu")
        self.model = model if model is not None else self._load_model()

        loaded_dim = self.model.get_sentence_embedding_dimension()
        if loaded_dim != spec.dimension:
            raise ValueError(f"모델 차원 불일치: 기대 {spec.dimension}, 실제 {loaded_dim}")

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.spec.model_id,
            revision=self.spec.revision,
            device=self.device,
        )
        model.max_seq_length = self.spec.max_seq_length
        return model

    def _encode(self, texts: Sequence[str], prefix: str, show_progress: bool) -> np.ndarray:
        prefixed = [prefix + text for text in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.spec.normalize,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        """음식 텍스트 임베딩, 'passage: ' 접두어 적용"""
        return self._encode(texts, self.spec.document_prefix, show_progress)

    def encode_queries(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """사용자 문장 임베딩, 'query: ' 접두어 적용"""
        return self._encode(texts, self.spec.query_prefix, show_progress)


def cosine_top_k(query_vectors: np.ndarray, doc_vectors: np.ndarray, k: int = 5):
    """정규화된 벡터 기준 코사인 유사도 Top-K

    Returns:
        (indices, scores) 각각 (n_query, k) 배열
    """
    if query_vectors.ndim == 1:
        query_vectors = query_vectors[None, :]
    scores = query_vectors @ doc_vectors.T
    k = min(k, doc_vectors.shape[0])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)
    return ordered, np.take_along_axis(scores, ordered, axis=1)
