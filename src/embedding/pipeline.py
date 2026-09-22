"""음식 임베딩 파이프라인

데이터 연결 -> 텍스트 구성 -> 임베딩 -> 저장, 재사용 판정까지 담당한다
"""

from pathlib import Path

from .dataset import FOODS_PATH, LABELS_PATH, UNITS_PATH, join_units_labels, load_sources
from .embedder import DEFAULT_SPEC, E5Embedder, cosine_top_k
from .store import EmbeddingStore, build_config, compute_input_hash, result_name, sha256_file
from .text_builder import TEXT_SPEC_VERSION, build_text, format_attributes


def build_records(joined, text_variant):
    """임베딩 입력 텍스트와 추적용 메타데이터 생성

    라벨 값은 전량 모델 추정이며 검토 상태를 함께 기록한다

    Returns:
        (texts, records) 두 리스트의 순서는 임베딩 행 순서와 같다
    """
    texts, records = [], []
    for item in joined:
        unit, label = item["unit"], item["label"]
        text = build_text(text_variant, unit, label)
        texts.append(text)
        records.append({
            "라벨링단위ID": item["라벨링단위ID"],
            "메뉴명": unit.get("메뉴명", ""),
            "대표식품명": unit.get("대표식품명", ""),
            "식품대분류명": unit.get("식품대분류명", ""),
            "업체명": unit.get("업체명", ""),
            "식품코드목록": item["식품코드목록"],
            "식품코드수": len(item["codes"]),
            "embedding_text": text,
            "라벨": label.get("라벨", {}),
            "속성토큰": format_attributes(label),
            "라벨출처": label.get("라벨출처", ""),
            "검토상태": label.get("검토상태", ""),
            "라벨모델": label.get("모델", ""),
        })
    return texts, records


def source_hashes(units_path=UNITS_PATH, labels_path=LABELS_PATH, foods_path=FOODS_PATH):
    """입력 원본 파일의 내용 해시"""
    return {
        Path(p).name: sha256_file(p)
        for p in (units_path, labels_path, foods_path)
    }


def prepare_variant(text_variant, joined, spec=DEFAULT_SPEC, sources=None):
    """텍스트, records, 재사용 판정용 config 생성"""
    texts, records = build_records(joined, text_variant)
    input_hash = compute_input_hash([r["라벨링단위ID"] for r in records], texts)
    config = build_config(
        spec.as_dict(), text_variant, TEXT_SPEC_VERSION, input_hash,
        sources if sources is not None else source_hashes(),
    )
    return texts, records, config


def run_variant(text_variant, joined, embedder=None, spec=DEFAULT_SPEC, store=None,
                sources=None, name=None, save=True, force=False, batch_size=None):
    """텍스트 구성 하나에 대해 임베딩 생성 또는 재사용

    Args:
        embedder: None이면 재사용 판정 후 필요할 때만 모델을 로드한다
        name: 결과 이름, None이면 모델·리비전·텍스트 구성·설정 해시로 만든다
        force: True면 설정이 같아도 다시 계산해 덮어쓴다

    Returns:
        (vectors, manifest, info)
    """
    store = store or EmbeddingStore()
    texts, records, config = prepare_variant(text_variant, joined, spec, sources)
    name = name or result_name(config)

    reusable, reason = store.reuse_check(name, config)
    if reusable and not force:
        vectors, manifest = store.load(name, config)
        return vectors, manifest, {"name": name, "reused": True, "reason": reason}

    embedder = embedder or E5Embedder(spec=spec, batch_size=batch_size)
    vectors = embedder.encode_documents(texts)
    if save:
        store.save(name, vectors, records, config, overwrite=force)
        vectors, manifest = store.load(name, config)
    else:
        manifest = {"name": name, "config": config, "records": records}
    return vectors, manifest, {"name": name, "reused": False, "reason": reason}


def load_joined(units_path=UNITS_PATH, labels_path=LABELS_PATH, foods_path=FOODS_PATH):
    """원본 로드 후 연결까지 한 번에 수행"""
    units, labels, foods = load_sources(units_path, labels_path, foods_path)
    return join_units_labels(units, labels, foods)


def search(query_vectors, doc_vectors, records, queries, k=5):
    """코사인 유사도 Top-K 검색 결과를 표로 쓰기 좋은 행 목록으로 반환"""
    indices, scores = cosine_top_k(query_vectors, doc_vectors, k=k)
    rows = []
    for qi, query in enumerate(queries):
        for rank, (idx, score) in enumerate(zip(indices[qi], scores[qi]), start=1):
            record = records[int(idx)]
            rows.append({
                "질의": query,
                "순위": rank,
                "메뉴명": record["메뉴명"],
                "업체명": record["업체명"] or "-",
                "유사도": round(float(score), 4),
                "주요라벨": ", ".join(record["속성토큰"]) or "-",
                "라벨링단위ID": record["라벨링단위ID"],
                "대표식품명": record["대표식품명"],
                "식품대분류명": record["식품대분류명"],
            })
    return rows
