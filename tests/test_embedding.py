"""음식 임베딩 핵심 로직 테스트, 실제 모델 다운로드나 API 호출 없음"""

import numpy as np
import pytest

from src.embedding import (
    DEFAULT_SPEC, E5Embedder, EmbeddingStore, ModelSpec, TEXT_SPEC_VERSION,
    build_config, build_records, build_text_a, build_text_b, choose_batch_size, compute_input_hash,
    cosine_top_k, format_attributes, join_units_labels, result_name, run_variant, search, validate_vectors,
)


def _unit(uid, menu, rep, category, codes, company=""):
    return {
        "라벨링단위ID": uid, "메뉴명": menu, "대표식품명": rep, "식품대분류명": category,
        "업체명": company, "식품코드목록": ";".join(codes), "행수": str(len(codes)), "경계메뉴": "False",
    }


def _label(uid, codes, **attrs):
    labels = {"매운맛": "없음", "국물": "국물없음", "제공온도": "뜨거움", "조리법": "오븐", "기름짐": "높음", "든든함": "미확인"}
    labels.update(attrs)
    return {
        "라벨링단위ID": uid, "식품코드목록": ";".join(codes), "라벨": labels,
        "근거": "치즈 도우라 기름짐 높음", "라벨출처": "모델추정", "모델": "테스트모델", "검토상태": "검토대기", "검토메모": "",
    }


def _food(code):
    return {"식품코드": code, "식품명": code}


def _fixture():
    units = [
        _unit("u1", "치즈 피자", "피자", "빵 및 과자류", ["c1", "c2"], company="A피자"),
        _unit("u2", "육개장", "육개장", "국 및 탕류", ["c3"]),
    ]
    labels = [
        _label("u1", ["c1", "c2"]),
        _label("u2", ["c3"], 매운맛="보통", 국물="국물요리", 조리법="끓임", 기름짐="보통", 든든함="든든함"),
    ]
    foods = [_food("c1"), _food("c2"), _food("c3")]
    return units, labels, foods


class FakeModel:
    """encode 입력을 기록하는 가짜 모델, 결정적 벡터 반환"""

    def __init__(self, dimension=768):
        self.dimension = dimension
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
        self.calls.append(list(texts))
        rng = np.random.default_rng([abs(hash(t)) % (2 ** 32) for t in texts] or [0])
        vectors = rng.standard_normal((len(texts), self.dimension)).astype(np.float32)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def _embedder(dimension=768):
    return E5Embedder(spec=DEFAULT_SPEC, device="cpu", model=FakeModel(dimension))


# 데이터 연결

def test_join_units_labels_is_clean_and_preserves_codes():
    joined, report = join_units_labels(*_fixture())
    assert report.is_clean
    assert [j["라벨링단위ID"] for j in joined] == ["u1", "u2"]
    assert joined[0]["codes"] == ["c1", "c2"]
    assert [f["식품코드"] for f in joined[0]["foods"]] == ["c1", "c2"]


def test_join_units_labels_reports_mismatches():
    units, labels, foods = _fixture()
    labels[0]["식품코드목록"] = "c1"
    labels.append(_label("u9", ["c9"]))
    units.append(_unit("u3", "김밥", "김밥", "밥류", ["c4"]))
    units.append(_unit("u3", "김밥", "김밥", "밥류", ["c4"]))
    foods.pop()

    _, report = join_units_labels(units, labels, foods)
    assert not report.is_clean
    assert report.codelist_mismatch == ["u1"]
    assert report.labels_without_unit == ["u9"]
    assert report.units_without_label == ["u3"]
    assert report.duplicate_unit_ids == ["u3"]
    assert report.codes_missing_in_foods == [("u2", ["c3"])]


# 텍스트 구성

def test_text_a_and_b_composition():
    units, labels, _ = _fixture()
    assert build_text_a(units[0]) == "치즈 피자 피자 빵 및 과자류"
    assert build_text_b(units[1], labels[1]) == (
        "육개장 국 및 탕류 매운맛 보통 국물요리 제공온도 뜨거움 조리법 끓임 기름짐 보통 든든함"
    )


def test_text_excludes_unknown_and_admin_fields():
    units, labels, _ = _fixture()
    text = build_text_b(units[0], labels[0])
    assert "미확인" not in text and "든든함" not in text
    for banned in ("A피자", "u1", "c1", "치즈 도우", "모델추정", "검토대기", "테스트모델"):
        assert banned not in text


def test_format_attributes_prefixes_attribute_name_only_when_needed():
    tokens = format_attributes(_label("u", ["c"], 매운맛="없음", 국물="국물약간", 제공온도="미확인", 든든함="가벼움"))
    assert tokens == ["매운맛 없음", "국물약간", "조리법 오븐", "기름짐 높음", "든든함 가벼움"]
    assert format_attributes({}) == []


def test_build_records_keeps_tracking_fields_and_label_provenance():
    joined, _ = join_units_labels(*_fixture())
    texts, records = build_records(joined, "B")
    assert len(texts) == len(records) == 2
    assert records[0]["식품코드목록"] == "c1;c2" and records[0]["식품코드수"] == 2
    assert records[0]["embedding_text"] == texts[0]
    assert records[0]["라벨출처"] == "모델추정" and records[0]["검토상태"] == "검토대기"
    assert "확정속성" not in records[0]
    assert records[1]["속성토큰"][0] == "매운맛 보통"


# 접두어 구분

def test_embedder_applies_query_and_passage_prefix():
    embedder = _embedder()
    docs = embedder.encode_documents(["육개장 국 및 탕류"])
    queries = embedder.encode_queries(["얼큰한 국물"])
    assert embedder.model.calls == [["passage: 육개장 국 및 탕류"], ["query: 얼큰한 국물"]]
    assert docs.shape == (1, 768) and docs.dtype == np.float32
    assert np.isclose(np.linalg.norm(queries[0]), 1.0)


def test_embedder_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="차원 불일치"):
        _embedder(dimension=384)


def test_choose_batch_size_by_device():
    assert choose_batch_size("cuda") == 128
    assert choose_batch_size("mps", 8.0) == 32
    assert choose_batch_size("mps", 16.0) == 64
    assert choose_batch_size("cpu") == 16


# 검증

def test_validate_vectors_detects_nan_and_unnormalized():
    ok = np.eye(3, dtype=np.float32)
    assert validate_vectors(ok, 3, True)["max_norm"] == 1.0
    with pytest.raises(ValueError, match="NaN"):
        validate_vectors(np.array([[np.nan, 0.0]], dtype=np.float32), 2, True)
    with pytest.raises(ValueError, match="정규화"):
        validate_vectors(ok * 2, 3, True)
    with pytest.raises(ValueError, match="float32"):
        validate_vectors(ok.astype(np.float64), 3, True)


def test_cosine_top_k_orders_by_similarity():
    docs = np.eye(4, dtype=np.float32)
    query = np.array([[0.1, 0.9, 0.0, 0.3]], dtype=np.float32)
    idx, scores = cosine_top_k(query, docs, k=2)
    assert idx.tolist() == [[1, 3]]
    assert np.allclose(scores, [[0.9, 0.3]])


# 저장, 로드, 재사용

def _config(text_variant="A", **overrides):
    spec = DEFAULT_SPEC.as_dict()
    spec.update(overrides)
    return build_config(spec, text_variant, TEXT_SPEC_VERSION, compute_input_hash(["u1", "u2"], ["a", "b"]), {"f": "h"})


def _records():
    return [{"라벨링단위ID": "u1", "메뉴명": "a"}, {"라벨링단위ID": "u2", "메뉴명": "b"}]


def _vectors(n=2, dim=768):
    v = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_store_save_load_roundtrip(tmp_path):
    store = EmbeddingStore(tmp_path)
    config = _config()
    name = result_name(config)
    store.save(name, _vectors(), _records(), config)

    vectors, manifest = store.load(name, config)
    assert vectors.shape == (2, 768) and vectors.dtype == np.float32
    assert manifest["config"]["model_revision"] == DEFAULT_SPEC.revision
    assert [r["라벨링단위ID"] for r in manifest["records"]] == ["u1", "u2"]
    assert store.reuse_check(name, config) == (True, "설정 일치")
    assert name.startswith("multilingual-e5-base_d1287505_textA_v1_")


def test_result_name_changes_with_model_or_text_variant():
    base = result_name(_config())
    assert result_name(_config("B")) != base
    assert result_name(_config(revision="0" * 40)) != base
    assert "textB" in result_name(_config("B"))


def test_store_refuses_overwrite_with_different_config(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.save("shared", _vectors(), _records(), _config())
    with pytest.raises(FileExistsError, match="다른 설정"):
        store.save("shared", _vectors(), _records(), _config("B"))
    with pytest.raises(FileExistsError, match="같은 설정"):
        store.save("shared", _vectors(), _records(), _config())
    store.save("shared", _vectors(), _records(), _config(), overwrite=True)
    assert store.reuse_check("shared", _config("B"))[0] is False
    assert "text_variant" in store.reuse_check("shared", _config("B"))[1]


def test_store_refuses_legacy_files_without_manifest(tmp_path):
    store = EmbeddingStore(tmp_path)
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "vectors.npy").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="manifest 없는"):
        store.save("legacy", _vectors(), _records(), _config())


def test_store_save_and_load_validate_records_and_vectors(tmp_path):
    store = EmbeddingStore(tmp_path)
    config = _config()
    with pytest.raises(ValueError, match="중복"):
        store.save("dup", _vectors(), [{"라벨링단위ID": "u1"}, {"라벨링단위ID": "u1"}], config)
    with pytest.raises(ValueError, match="행수"):
        store.save("len", _vectors(3), _records(), config)
    with pytest.raises(ValueError, match="차원"):
        store.save("dim", _vectors(2, 384), _records(), config)

    store.save("ok", _vectors(), _records(), config)
    np.save(store.vector_path("ok"), _vectors(3))
    with pytest.raises(ValueError, match="형태"):
        store.load("ok")
    np.save(store.vector_path("ok"), _vectors())
    with pytest.raises(ValueError, match="저장된 설정이 다릅니다"):
        store.load("ok", _config("B"))


def test_store_lists_results_and_legacy_vectors(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.save("new", _vectors(), _records(), _config())
    np.save(tmp_path / "old.npy", _vectors(2, 384))
    rows = {r["name"]: r for r in store.list_results()}
    assert rows["new"]["model_id"] == DEFAULT_SPEC.model_id and rows["new"]["dimension"] == 768
    assert rows["old.npy"]["model_id"] is None and rows["old.npy"]["dimension"] == 384


def test_run_variant_reuses_only_when_config_matches(tmp_path):
    joined, _ = join_units_labels(*_fixture())
    store = EmbeddingStore(tmp_path)
    sources = {"units": "h1", "labels": "h2", "foods": "h3"}
    embedder = _embedder()

    vectors, manifest, info = run_variant("A", joined, embedder=embedder, store=store, sources=sources)
    assert info["reused"] is False and vectors.shape == (2, 768)
    assert len(embedder.model.calls) == 1
    assert manifest["config"]["text_variant"] == "A"

    again, _, info = run_variant("A", joined, embedder=embedder, store=store, sources=sources)
    assert info["reused"] is True and len(embedder.model.calls) == 1
    assert np.array_equal(again, vectors)

    _, _, info = run_variant("A", joined, embedder=embedder, store=store, sources={"units": "changed"})
    assert info["reused"] is False and len(embedder.model.calls) == 2

    _, _, info = run_variant("A", joined, embedder=embedder, store=store, sources=sources, force=True)
    assert info["reused"] is False and len(embedder.model.calls) == 3


def test_search_rows_include_labels_and_company():
    joined, _ = join_units_labels(*_fixture())
    _, records = build_records(joined, "B")
    docs = np.eye(2, dtype=np.float32)
    rows = search(np.array([[0.0, 1.0]], dtype=np.float32), docs, records, ["얼큰한 국물"], k=2)
    assert rows[0]["메뉴명"] == "육개장" and rows[0]["순위"] == 1 and rows[0]["업체명"] == "-"
    assert rows[0]["주요라벨"].startswith("매운맛 보통")
    assert rows[1]["업체명"] == "A피자"
