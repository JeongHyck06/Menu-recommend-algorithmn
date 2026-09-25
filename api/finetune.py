"""좋아요 피드백으로 e5 파인튜닝

좋아요 받은 (질의, 메뉴) 쌍이 지난 시도보다 MIN_NEW개 이상 늘면 기본 모델에서 다시 학습하고,
승인 판정 세트 nDCG@5가 현재 모델보다 떨어지지 않을 때만 state/current.json을 새 모델로 바꾼다

python -m api.finetune
"""

import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

from api.feedback import STATE_DIR, FeedbackStore

CURRENT = STATE_DIR / "current.json"
EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "evaluation"
MIN_NEW = int(os.environ.get("FINETUNE_MIN_NEW", "200"))


def read_current() -> dict:
    try:
        return json.loads(CURRENT.read_text())
    except (OSError, ValueError):
        return {}


def write_current(data: dict):
    tmp = CURRENT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(CURRENT)


def training_examples(pairs, records) -> list:
    """(질의, 메뉴 임베딩 원문), 검색어와 대표식품명이 같은 첫 메뉴를 정답으로 쓴다"""
    texts = {}
    for r in records:
        texts.setdefault(r["대표식품명"] or r["메뉴명"], r["embedding_text"])
    return [(q, texts[k]) for q, k in pairs if k in texts]


def _recommender(index, embedder, ref):
    from src.recommendation import Recommender
    return Recommender(index, lambda t: embedder.encode_queries([t])[0], ref)


def load_active():
    """(추천기, 모델 이름), 교체된 파인튜닝 모델이 현재 후보와 맞으면 그것을, 아니면 기본 모델"""
    from src.embedding import E5Embedder
    from src.retrieval import CandidateIndex, load_index

    index, ref = load_index("B")
    name = read_current().get("dir")
    path = STATE_DIR / name if name else None
    if path and (path / "ids.json").exists():
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if json.loads((path / "ids.json").read_text()) == [r["라벨링단위ID"] for r in index.records]:
            embedder = E5Embedder(model=SentenceTransformer(str(path), device="cpu"))
            ft = CandidateIndex(f"{index.name}+{name}", index.text_variant, np.load(path / "vectors.npy"),
                                index.records)
            return _recommender(ft, embedder, ref), name
    return _recommender(index, E5Embedder(), ref), "base"


def ndcg(rec) -> float:
    from src.recommendation import FULL
    from src.recommendation.evaluation import evaluate_configs, judgment_map, load_judgments

    jmap = judgment_map(load_judgments(EVAL_DIR / "judgments.csv"), approved_only=True)
    with open(EVAL_DIR / "queries.csv", newline="", encoding="utf-8-sig") as f:
        queries = [row["질의"] for row in csv.DictReader(f)]
    return evaluate_configs({"x": rec}, queries, {"full": FULL}, jmap)[0]["nDCG@5"]


def train(examples, epochs=1, batch_size=16, lr=2e-5):
    """MultipleNegativesRankingLoss, 같은 배치의 다른 메뉴를 오답으로 쓴다
    ponytail: 같은 메뉴가 한 배치에 두 번 들면 서로 오답이 되는 잡음, 데이터가 많아지면 배치 구성 개선
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    from src.embedding import DEFAULT_SPEC as spec

    model = SentenceTransformer(spec.model_id, revision=spec.revision, device="cpu")
    model.max_seq_length = spec.max_seq_length
    loss_fn = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            if len(batch) < 2:
                continue
            features = [model.preprocess([spec.query_prefix + q for q, _ in batch]),
                        model.preprocess([spec.document_prefix + d for _, d in batch])]
            loss = loss_fn(features, None)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    return model


def run(min_new=MIN_NEW) -> dict:
    from src.embedding import E5Embedder
    from src.retrieval import CandidateIndex, load_index

    pairs = FeedbackStore().liked_pairs()
    current = read_current()
    if len(pairs) - current.get("tried_pairs", 0) < min_new:
        return {"status": "skip", "pairs": len(pairs), "need": current.get("tried_pairs", 0) + min_new}

    index, ref = load_index("B")
    examples = training_examples(pairs, index.records)
    model = train(examples)
    embedder = E5Embedder(model=model)
    vectors = embedder.encode_documents([r["embedding_text"] for r in index.records], show_progress=False)
    ft_rec = _recommender(CandidateIndex(index.name + "+new", index.text_variant, vectors, index.records),
                          embedder, ref)
    active_rec, active = load_active()
    new_score, old_score = ndcg(ft_rec), ndcg(active_rec)
    result = {"pairs": len(pairs), "examples": len(examples), "ndcg": new_score, "active_ndcg": old_score}

    current["tried_pairs"] = len(pairs)
    current["last_attempt"] = {**result, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if new_score >= old_score:
        import numpy as np

        name = time.strftime("ft-%Y%m%d%H%M%S")
        path = STATE_DIR / name
        model.save(str(path))
        np.save(path / "vectors.npy", vectors)
        (path / "ids.json").write_text(json.dumps([r["라벨링단위ID"] for r in index.records]))
        current["dir"] = name
        for old in STATE_DIR.glob("ft-*"):
            if old.name != name:
                shutil.rmtree(old, ignore_errors=True)
    write_current(current)
    return {"status": "swapped" if new_score >= old_score else "kept", **result}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
