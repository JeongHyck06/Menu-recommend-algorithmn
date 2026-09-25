"""승인 판정과 좋아요 피드백으로 e5 파인튜닝

학습 데이터는 학습용 질의의 승인 판정 적합도 2 쌍과 좋아요 받은 (질의, 메뉴) 쌍이다
평가용 질의(해시로 약 1/3)는 학습에서 빼고 평가에만 쓴다
좋아요 쌍이 지난 시도보다 MIN_NEW개 이상 늘면 기본 모델에서 다시 학습하고,
평가용 질의에서 nDCG@5가 0.01 이상 오르고 정답 회수율이 떨어지지 않을 때만 state/current.json을 새 모델로 바꾼다

python -m api.finetune          조건을 채웠을 때만 학습
python -m api.finetune --force  바로 학습
"""

import csv
import hashlib
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


def is_test_query(query: str) -> bool:
    return int(hashlib.md5(query.encode()).hexdigest(), 16) % 3 == 0


def judgment_examples(records) -> list:
    """학습용 질의의 승인 판정 적합도 2 쌍, (질의, 메뉴 임베딩 원문)"""
    from src.recommendation.evaluation import APPROVED, load_judgments

    texts = {r["라벨링단위ID"]: r["embedding_text"] for r in records}
    return [(row["질의"], texts[row["라벨링단위ID"]]) for row in load_judgments(EVAL_DIR / "judgments.csv")
            if row["검토상태"] == APPROVED and row["적합도"] == "2" and not is_test_query(row["질의"])
            and row["라벨링단위ID"] in texts]


def training_examples(pairs, records) -> list:
    """(질의, 메뉴 임베딩 원문), 검색어와 대표식품명이 같은 첫 메뉴를 정답으로 쓴다"""
    texts = {}
    for r in records:
        texts.setdefault(r["대표식품명"] or r["메뉴명"], r["embedding_text"])
    return [(q, texts[k]) for q, k in pairs if k in texts and not is_test_query(q)]


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


def evaluate(rec) -> dict:
    """평가용 질의 기준 지표
    판정 풀이 기본 모델 결과로 만들어져 미판정을 0점으로 보면 새 모델이 불리하므로
    nDCG@5는 판정된 항목만 남긴 순위(condensed)로, 정답 회수율은 임베딩 상위 10개 중 적합도 2 비율로 잰다
    """
    from dataclasses import replace

    import numpy as np

    from src.embedding import cosine_top_k
    from src.recommendation import FULL
    from src.recommendation.evaluation import judgment_map, load_judgments, ndcg_at_k

    jmap = judgment_map(load_judgments(EVAL_DIR / "judgments.csv"), approved_only=True)
    with open(EVAL_DIR / "queries.csv", newline="", encoding="utf-8-sig") as f:
        queries = [row["질의"] for row in csv.DictReader(f) if is_test_query(row["질의"])]
    ids = [r["라벨링단위ID"] for r in rec.index.records]
    ndcgs, recalls = [], []
    for q in queries:
        items = rec.recommend(q, replace(FULL, top_k=20))["추천"]
        grades = [jmap[(q, it["라벨링단위ID"])] for it in items if (q, it["라벨링단위ID"]) in jmap][:5]
        ndcgs.append(ndcg_at_k(grades, [g for (qq, _), g in jmap.items() if qq == q], 5))
        positives = {i for (qq, i), g in jmap.items() if qq == q and g == 2}
        if positives:
            top, _ = cosine_top_k(rec.encode_query(q), rec.index.vectors, k=10)
            recalls.append(len({ids[i] for i in top[0]} & positives) / min(10, len(positives)))
    return {"ndcg": round(float(np.mean(ndcgs)), 4), "recall": round(float(np.mean(recalls)), 4)}


def better(new: dict, old: dict) -> bool:
    return new["ndcg"] >= old["ndcg"] + 0.01 and new["recall"] >= old["recall"]


def unique_batches(examples, batch_size) -> list:
    """한 배치에 같은 질의나 같은 메뉴가 두 번 들면 서로의 정답을 오답으로 배우므로 겹치지 않게 나눈다"""
    batches = []
    for ex in examples:
        for b in batches:
            if len(b) < batch_size and all(ex[0] != q and ex[1] != d for q, d in b):
                b.append(ex)
                break
        else:
            batches.append([ex])
    return batches


def train(examples, epochs=1, batch_size=16, lr=2e-5):
    """MultipleNegativesRankingLoss, 같은 배치의 다른 메뉴를 오답으로 쓴다"""
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
        for batch in unique_batches(examples, batch_size):
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


def run(min_new=MIN_NEW, force=False) -> dict:
    from src.embedding import E5Embedder
    from src.retrieval import CandidateIndex, load_index

    pairs = FeedbackStore().liked_pairs()
    current = read_current()
    if not force and len(pairs) - current.get("tried_pairs", 0) < min_new:
        return {"status": "skip", "pairs": len(pairs), "need": current.get("tried_pairs", 0) + min_new}

    index, ref = load_index("B")
    examples = judgment_examples(index.records) + training_examples(pairs, index.records)
    model = train(examples)
    embedder = E5Embedder(model=model)
    vectors = embedder.encode_documents([r["embedding_text"] for r in index.records], show_progress=False)
    ft_rec = _recommender(CandidateIndex(index.name + "+new", index.text_variant, vectors, index.records),
                          embedder, ref)
    active_rec, active = load_active()
    new_score, old_score = evaluate(ft_rec), evaluate(active_rec)
    swap = better(new_score, old_score)
    result = {"pairs": len(pairs), "examples": len(examples), "new": new_score, "active": old_score}

    current["tried_pairs"] = len(pairs)
    current["last_attempt"] = {**result, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if swap:
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
    return {"status": "swapped" if swap else "kept", **result}


if __name__ == "__main__":
    import sys

    print(json.dumps(run(force="--force" in sys.argv), ensure_ascii=False))
