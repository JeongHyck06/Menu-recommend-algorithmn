"""유사도 기반 후보 검색

저장된 임베딩 결과(vectors.npy + manifest.json)를 현재 데이터·모델 규격과 대조해
호환되는 결과만 로드하고, 질의 벡터에 대한 코사인 유사도 상위 후보를 돌려준다.
임베딩을 새로 만들지 않으며 manifest 없는 벡터 파일은 쓰지 않는다.

추천 대상 범위: 기본으로 업체명이 없는 공공 데이터만 후보로 쓴다. 프랜차이즈 메뉴(업체명 있음)는
추천 대상에서 제외하기로 결정했다 (2026-09-25). include_franchise=True로 되돌릴 수 있다.
"""

from dataclasses import dataclass

import numpy as np

from src.embedding import DEFAULT_SPEC, TEXT_SPEC_VERSION, EmbeddingStore, cosine_top_k, source_hashes


@dataclass
class CandidateIndex:
    """호환 확인을 통과한 임베딩 한 벌"""
    name: str
    text_variant: str
    vectors: np.ndarray
    records: list

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def subset(self, keep) -> "CandidateIndex":
        """keep(record) -> bool 인 항목만 남긴 새 인덱스, 벡터 행과 records 순서를 함께 유지한다"""
        rows = [i for i, r in enumerate(self.records) if keep(r)]
        return CandidateIndex(self.name, self.text_variant, self.vectors[rows], [self.records[i] for i in rows])


def is_franchise(record) -> bool:
    """업체명이 있으면 프랜차이즈, 결과 행의 "-" 표시는 빈 업체명으로 본다"""
    return (record.get("업체명") or "").strip() not in ("", "-")


def check_compatibility(manifest, spec=DEFAULT_SPEC, sources=None) -> list:
    """manifest와 현재 모델 규격·원본 데이터 해시의 불일치 목록, 비어 있으면 호환"""
    cfg = manifest.get("config") or {}
    expected = {
        "model_id": spec.model_id, "model_revision": spec.revision, "dimension": spec.dimension,
        "normalized": spec.normalize, "pooling": spec.pooling, "max_seq_length": spec.max_seq_length,
        "query_prefix": spec.query_prefix, "document_prefix": spec.document_prefix,
        "text_spec_version": TEXT_SPEC_VERSION,
    }
    problems = [f"{k}: 저장 {cfg.get(k)!r} != 기대 {v!r}" for k, v in expected.items() if cfg.get(k) != v]
    current = sources if sources is not None else source_hashes()
    stored = cfg.get("source_files") or {}
    problems += [f"{name} 해시 불일치" for name, h in current.items() if stored.get(name) != h]
    return problems


def manifest_ref(manifest) -> dict:
    """추천 결과에 남기는 임베딩 출처 정보"""
    cfg = manifest.get("config") or {}
    return {
        "name": manifest.get("name"), "config_hash": manifest.get("config_hash"),
        "model_id": cfg.get("model_id"), "model_revision": cfg.get("model_revision"),
        "text_variant": cfg.get("text_variant"), "text_spec_version": cfg.get("text_spec_version"),
        "dimension": manifest.get("dimension"), "num_vectors": manifest.get("num_vectors"),
        "source_files": cfg.get("source_files"), "created_at": manifest.get("created_at"),
    }


def find_result_name(store, text_variant, spec=DEFAULT_SPEC) -> str:
    """모델·리비전·텍스트 구성이 맞는 저장 결과 이름, 없거나 여러 개면 오류"""
    names = [
        r["name"] for r in store.list_results()
        if r["model_id"] == spec.model_id and r["model_revision"] == spec.revision[:8]
        and r["text_variant"] == text_variant
    ]
    if len(names) != 1:
        raise FileNotFoundError(
            f"텍스트 {text_variant} 결과가 {len(names)}개입니다 ({names}), 정확히 하나여야 합니다"
        )
    return names[0]


def load_index(text_variant, store=None, spec=DEFAULT_SPEC, sources=None, include_franchise=False) -> tuple:
    """호환 검사를 통과한 벡터·records 로드

    Args:
        include_franchise: False면 업체명이 있는 프랜차이즈 항목을 후보에서 뺀다

    Returns:
        (CandidateIndex, manifest_ref) manifest_ref에 후보 범위를 함께 남긴다
    """
    store = store or EmbeddingStore()
    name = find_result_name(store, text_variant, spec)
    vectors, manifest = store.load(name)
    problems = check_compatibility(manifest, spec, sources)
    if problems:
        raise ValueError(f"{name}은 현재 데이터·규격과 호환되지 않습니다: " + "; ".join(problems))
    index = CandidateIndex(name, text_variant, vectors, manifest["records"])
    if not include_franchise:
        index = index.subset(lambda r: not is_franchise(r))
    ref = manifest_ref(manifest)
    ref["후보범위"] = {"프랜차이즈포함": include_franchise, "후보수": index.size, "전체수": len(manifest["records"])}
    return index, ref


def retrieve(index: CandidateIndex, query_vector: np.ndarray, k: int) -> list:
    """코사인 유사도 상위 k개 후보, 유사도 내림차순"""
    indices, scores = cosine_top_k(query_vector, index.vectors, k=k)
    return [
        {"index": int(i), "유사도": float(s), "record": index.records[int(i)]}
        for i, s in zip(indices[0], scores[0])
    ]
