"""docs/REPORT.md 8장 파인튜닝 실험의 수치와 그림 10을 다시 만든다

실행: .venv/bin/python docs/finetune_experiment.py              학습부터 다시 (약 5분)
      .venv/bin/python docs/finetune_experiment.py --plot-only  저장된 수치로 그림만
결과: docs/figures/finetune.json, docs/figures/fig10_finetune.png
학습은 api.finetune과 같은 코드(승인 판정 적합도 2 쌍, 평가용 질의 제외)를 시드별로 돌린다.
"""

import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"
SEEDS = [f"파인튜닝 시드{s}" for s in (0, 1, 2)]
METRICS = ("nDCG", "미판정", "condensed", "recall")
GRAY, BLUE, ORANGE, INK2, GRID = "#a3a29b", "#2a78d6", "#eb6834", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def run() -> dict:
    import torch

    from api import finetune as ft
    from src.embedding import E5Embedder, cosine_top_k
    from src.recommendation import FULL
    from src.recommendation.evaluation import (evaluate_per_query, judgment_map, load_judgments, ndcg_at_k,
                                               paired_bootstrap)
    from src.retrieval import CandidateIndex, load_index

    jmap = judgment_map(load_judgments(ft.EVAL_DIR / "judgments.csv"), approved_only=True)
    queries = sorted({q for q, _ in jmap})
    split = {"학습": [q for q in queries if not ft.is_test_query(q)], "평가": [q for q in queries if ft.is_test_query(q)]}
    index, ref = load_index("B")
    ids = [r["라벨링단위ID"] for r in index.records]
    examples = ft.judgment_examples(index.records)

    def per_query(rec, qs):
        rows = []
        for q, s in zip(qs, evaluate_per_query(rec, qs, FULL, jmap, 5)):
            items = rec.recommend(q, replace(FULL, top_k=20))["추천"]
            grades = [jmap[(q, it["라벨링단위ID"])] for it in items if (q, it["라벨링단위ID"]) in jmap][:5]
            pos = {i for (qq, i), g in jmap.items() if qq == q and g == 2}
            top, _ = cosine_top_k(rec.encode_query(q), rec.index.vectors, k=10)
            rows.append({
                "nDCG": s["nDCG@5"], "미판정": s["미판정수"] / max(s["반환수"], 1),
                "condensed": ndcg_at_k(grades, [g for (qq, _), g in jmap.items() if qq == q], 5),
                "recall": len({ids[i] for i in top[0]} & pos) / min(10, len(pos)) if pos else None,
            })
        return rows

    def fine_tuned(seed, batches=None):
        random.seed(seed)
        torch.manual_seed(seed)
        original = ft.unique_batches
        if batches:
            ft.unique_batches = batches
        try:
            model = ft.train(list(examples))
        finally:
            ft.unique_batches = original
        emb = E5Embedder(model=model)
        vec = emb.encode_documents([r["embedding_text"] for r in index.records], show_progress=False)
        return ft._recommender(CandidateIndex("ft", "B", vec, index.records), emb, ref)

    naive = lambda ex, size: [ex[i:i + size] for i in range(0, len(ex), size)]
    models = {"기본": ft._recommender(index, E5Embedder(), ref)}
    models.update({name: fine_tuned(s) for s, name in enumerate(SEEDS)})
    models["중복 배치 시드0"] = fine_tuned(0, naive)

    mean = lambda rows, k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 4)
    raw = {name: {part: per_query(rec, qs) for part, qs in split.items()} for name, rec in models.items()}
    result = {
        "학습쌍": len(examples), "질의수": {k: len(v) for k, v in split.items()},
        "모델": {name: {part: {m: mean(rows, m) for m in METRICS} for part, rows in r.items()} for name, r in raw.items()},
    }
    result["시드평균"] = {part: {m: round(float(np.mean([result["모델"][s][part][m] for s in SEEDS])), 4)
                              for m in METRICS} for part in split}
    for m in ("condensed", "recall"):
        pairs = [(a[m], b[m]) for a, b in zip(raw["기본"]["평가"], raw[SEEDS[0]]["평가"]) if a[m] is not None]
        result[f"부트스트랩_평가_{m}_시드0"] = paired_bootstrap([a for a, _ in pairs], [b for _, b in pairs])
    (OUT / "finetune.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def plot(result: dict):
    metrics = [("nDCG", "nDCG@5\n(미판정 0점)"), ("condensed", "condensed\nnDCG@5"), ("recall", "정답 회수율@10")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(metrics))
    for ax, part in zip(axes, ("학습", "평가")):
        base = [result["모델"]["기본"][part][m] for m, _ in metrics]
        vals = np.array([[result["모델"][s][part][m] for m, _ in metrics] for s in SEEDS])
        naive = [result["모델"]["중복 배치 시드0"][part][m] for m, _ in metrics]
        ax.bar(x - 0.27, base, 0.26, color=GRAY, label="기본 모델")
        ax.bar(x, vals.mean(0), 0.26, color=BLUE, label="미세조정 (시드 3개 평균, 막대는 범위)",
               yerr=[vals.mean(0) - vals.min(0), vals.max(0) - vals.mean(0)], capsize=3, ecolor=INK2)
        ax.bar(x + 0.27, naive, 0.26, color=ORANGE, label="중복 허용 배치 (시드 0)")
        ax.set_xticks(x, [label for _, label in metrics])
        ax.set_title(f"{part}용 질의 {result['질의수'][part]}개", color=INK2, fontsize=10)
        ax.set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.1))
    fig.savefig(OUT / "fig10_finetune.png")
    plt.close(fig)


if __name__ == "__main__":
    data = json.loads((OUT / "finetune.json").read_text()) if "--plot-only" in sys.argv else run()
    plot(data)
    print(json.dumps({k: data[k] for k in ("학습쌍", "질의수", "시드평균")}, ensure_ascii=False))
