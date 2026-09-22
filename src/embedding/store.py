"""임베딩 벡터 및 메타데이터 저장, 로드, 재사용 판정

결과 하나는 디렉터리 하나에 저장한다
  data/embeddings/<모델_리비전_텍스트구성_설정해시>/vectors.npy
  data/embeddings/<모델_리비전_텍스트구성_설정해시>/manifest.json
벡터 행 순서와 manifest의 records 순서는 항상 같다
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings"

# 메타데이터 구조가 바뀌면 올린다
MANIFEST_VERSION = 3

ID_FIELD = "라벨링단위ID"

VECTOR_FILE = "vectors.npy"
MANIFEST_FILE = "manifest.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_input_hash(uids, texts) -> str:
    """임베딩 입력 전체(ID + 텍스트, 순서 포함)의 해시"""
    if len(uids) != len(texts):
        raise ValueError(f"uids({len(uids)})와 texts({len(texts)}) 길이가 다릅니다")
    payload = "\n".join(f"{uid}\t{text}" for uid, text in zip(uids, texts))
    return sha256_text(payload)


def compute_config_hash(config: dict) -> str:
    """모델, 텍스트 구성, 입력 데이터가 모두 일치하는지 판정할 설정 해시"""
    return sha256_text(json.dumps(config, sort_keys=True, ensure_ascii=False))


def build_config(model_spec: dict, text_variant: str, text_spec_version: str,
                 input_hash: str, source_hashes: dict) -> dict:
    """재사용 판정에 쓰는 설정 딕셔너리, 여기 들어간 값이 바뀌면 재계산한다"""
    return {
        "manifest_version": MANIFEST_VERSION,
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "dimension": model_spec["dimension"],
        "normalized": model_spec["normalize"],
        "pooling": model_spec["pooling"],
        "document_prefix": model_spec["document_prefix"],
        "query_prefix": model_spec["query_prefix"],
        "max_seq_length": model_spec["max_seq_length"],
        "text_variant": text_variant,
        "text_spec_version": text_spec_version,
        "input_data_hash": input_hash,
        "source_files": source_hashes,
    }


def result_name(config: dict) -> str:
    """설정에서 결과 디렉터리 이름 생성

    예: multilingual-e5-base_d1287505_textA_v1_3fa2c91d
    설정 해시에 입력 해시가 포함되므로 데이터나 설정이 바뀌면 이름도 바뀐다
    """
    model_slug = f"{config['model_id'].split('/')[-1]}_{config['model_revision'][:8]}"
    return (f"{model_slug}_text{config['text_variant']}_{config['text_spec_version']}"
            f"_{compute_config_hash(config)[:8]}")


def validate_vectors(vectors: np.ndarray, dimension: int = None, normalized: bool = None,
                     norm_tolerance: float = 1e-3) -> dict:
    """벡터 배열 검증, 실패하면 ValueError

    Returns:
        형태, dtype, 노름 범위 등 검증 통계
    """
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("(n, dim) 2차원 배열이 필요합니다")
    if vectors.dtype != np.float32:
        raise ValueError(f"float32 배열이 필요합니다: {vectors.dtype}")
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError(f"벡터 차원({vectors.shape[1]})이 설정({dimension})과 다릅니다")
    if not np.isfinite(vectors).all():
        raise ValueError("NaN 또는 무한대 값이 있습니다")

    norms = np.linalg.norm(vectors, axis=1)
    min_norm, max_norm = float(norms.min()), float(norms.max())
    if normalized and (abs(min_norm - 1.0) > norm_tolerance or abs(max_norm - 1.0) > norm_tolerance):
        raise ValueError(f"L2 정규화되지 않았습니다: 노름 범위 {min_norm:.6f}~{max_norm:.6f}")

    return {
        "num_vectors": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "finite": True,
        "min_norm": min_norm,
        "max_norm": max_norm,
    }


def _check_ids(records):
    ids = [r.get(ID_FIELD) for r in records]
    if any(i in (None, "") for i in ids):
        raise ValueError(f"records에 {ID_FIELD}가 비어 있는 항목이 있습니다")
    if len(set(ids)) != len(ids):
        raise ValueError(f"records의 {ID_FIELD}가 중복됩니다")


class EmbeddingStore:
    """결과를 이름별 디렉터리에 저장하고 설정이 같을 때만 재사용한다"""

    def __init__(self, output_dir=DEFAULT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def result_dir(self, name: str) -> Path:
        return self.output_dir / name

    def vector_path(self, name: str) -> Path:
        return self.result_dir(name) / VECTOR_FILE

    def manifest_path(self, name: str) -> Path:
        return self.result_dir(name) / MANIFEST_FILE

    def exists(self, name: str) -> bool:
        return self.vector_path(name).exists() and self.manifest_path(name).exists()

    def read_manifest(self, name: str) -> dict:
        with open(self.manifest_path(name), encoding="utf-8") as f:
            return json.load(f)

    def save(self, name: str, vectors: np.ndarray, records: list, config: dict,
             overwrite: bool = False) -> tuple:
        """벡터와 메타데이터 저장

        같은 이름에 다른 설정의 결과가 있으면 항상 거부한다
        같은 설정의 결과가 있으면 overwrite=True일 때만 덮어쓴다

        Args:
            name: 결과 디렉터리 이름, 보통 result_name(config)
            vectors: (n, dim) float32 배열
            records: 행 순서와 1:1로 대응하는 항목 메타데이터
            config: build_config 결과
        """
        stats = validate_vectors(vectors, config["dimension"], config.get("normalized"))
        if vectors.shape[0] != len(records):
            raise ValueError(f"벡터 행수({vectors.shape[0]})와 records({len(records)})가 다릅니다")
        _check_ids(records)

        config_hash = compute_config_hash(config)
        target = self.result_dir(name)
        if self.manifest_path(name).exists():
            stored_hash = self.read_manifest(name).get("config_hash")
            if stored_hash != config_hash:
                raise FileExistsError(
                    f"다른 설정의 결과가 이미 있습니다: {target} (저장된 해시 {str(stored_hash)[:8]}, "
                    f"요청 해시 {config_hash[:8]}), 다른 이름을 쓰거나 직접 정리하세요"
                )
            if not overwrite:
                raise FileExistsError(f"같은 설정의 결과가 이미 있습니다: {target}, overwrite=True로 다시 저장하세요")
        elif target.exists() and any(target.iterdir()):
            raise FileExistsError(f"manifest 없는 파일이 있는 경로입니다: {target}, 덮어쓰지 않습니다")

        target.mkdir(parents=True, exist_ok=True)
        np.save(self.vector_path(name), vectors)

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config,
            "config_hash": config_hash,
            "num_vectors": stats["num_vectors"],
            "dimension": stats["dimension"],
            "dtype": stats["dtype"],
            "vector_stats": stats,
            "vector_file": VECTOR_FILE,
            "records": records,
        }
        with open(self.manifest_path(name), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return self.vector_path(name), self.manifest_path(name)

    def load(self, name: str, config: dict = None) -> tuple:
        """저장된 벡터와 메타데이터 로드

        벡터 형태, dtype, 유한성, 정규화, records 길이, ID 유일성을 검증하고
        config를 주면 설정 해시 일치까지 확인한다
        """
        if not self.exists(name):
            raise FileNotFoundError(f"저장된 임베딩이 없습니다: {self.result_dir(name)}")

        vectors = np.load(self.vector_path(name))
        manifest = self.read_manifest(name)
        stored_config = manifest.get("config") or {}

        validate_vectors(vectors, stored_config.get("dimension"), stored_config.get("normalized"))
        if vectors.shape[0] != manifest.get("num_vectors") or vectors.shape[1] != manifest.get("dimension"):
            raise ValueError(f"벡터 형태 {vectors.shape}가 manifest({manifest.get('num_vectors')}, "
                             f"{manifest.get('dimension')})와 다릅니다")
        if str(vectors.dtype) != manifest.get("dtype"):
            raise ValueError(f"벡터 dtype({vectors.dtype})이 manifest({manifest.get('dtype')})와 다릅니다")
        if vectors.shape[0] != len(manifest["records"]):
            raise ValueError(f"벡터 행수({vectors.shape[0]})와 records({len(manifest['records'])})가 어긋났습니다")
        _check_ids(manifest["records"])
        if config is not None and manifest.get("config_hash") != compute_config_hash(config):
            raise ValueError("요청한 설정과 저장된 설정이 다릅니다: " + self._diff(stored_config, config))
        return vectors, manifest

    def reuse_check(self, name: str, config: dict) -> tuple:
        """재사용 가능 여부 판정

        Returns:
            (재사용가능 여부, 사유 문자열)
        """
        if not self.exists(name):
            return False, "저장된 결과 없음"

        manifest = self.read_manifest(name)
        if manifest.get("config_hash") != compute_config_hash(config):
            return False, "설정 불일치: " + self._diff(manifest.get("config") or {}, config)
        return True, "설정 일치"

    @staticmethod
    def _diff(stored: dict, config: dict) -> str:
        changed = [k for k in set(stored) | set(config) if stored.get(k) != config.get(k)]
        return ", ".join(sorted(changed))

    def list_results(self) -> list:
        """저장된 결과 목록, manifest 없는 벡터 파일은 출처 미확인으로 표시"""
        rows = []
        for path in sorted(self.output_dir.iterdir()):
            if path.is_dir() and (path / MANIFEST_FILE).exists():
                m = self.read_manifest(path.name)
                cfg = m.get("config") or {}
                rows.append({
                    "name": path.name, "model_id": cfg.get("model_id"),
                    "model_revision": (cfg.get("model_revision") or "")[:8] or None,
                    "text_variant": cfg.get("text_variant"), "num_vectors": m.get("num_vectors"),
                    "dimension": m.get("dimension"), "created_at": m.get("created_at"),
                })
            elif path.suffix == ".npy":
                arr = np.load(path, mmap_mode="r")
                rows.append({
                    "name": path.name, "model_id": None, "model_revision": None,
                    "text_variant": None, "num_vectors": int(arr.shape[0]),
                    "dimension": int(arr.shape[1]) if arr.ndim == 2 else None, "created_at": None,
                })
        return rows
