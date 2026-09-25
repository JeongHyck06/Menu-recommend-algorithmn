"""docs/REPORT.md의 그림 1~9와 수치를 다시 만든다

실행: .venv/bin/python docs/make_figures.py
결과: docs/figures/*.png, docs/figures/stats.json
승인된 판정(검토상태=승인)만 쓴다. 재정렬 실험(그림 8)은 코드를 되돌렸으므로 당시 기록값을 그대로 그린다.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embedding import E5Embedder
from src.recommendation import Recommender
from src.recommendation.evaluation import (evaluate_per_query, judgment_map, load_judgments, paired_bootstrap,
                                           summarize)
from src.recommendation.pipeline import EMBEDDING_ONLY, FILTER_ONLY, FULL
from src.retrieval import load_index

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EVAL = ROOT / "data" / "processed" / "evaluation"

# 기본 팔레트 (dataviz reference palette, light)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


rows = load_judgments(EVAL / "judgments.csv")
jmap = judgment_map(rows, approved_only=True)
queries = sorted({q for q, _ in jmap})
all_queries = json.load(open(EVAL / "eval_run_config.json", encoding="utf-8"))["질의"]
query_type = {r["질의"]: r["유형"] for r in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig"))}

index_b, ref_b = load_index("B")
index_a, ref_a = load_index("A")
embedder = E5Embedder()
encode = lambda t: embedder.encode_queries([t])[0]
rec_b, rec_a = Recommender(index_b, encode, ref_b), Recommender(index_a, encode, ref_a)

R = lambda **kw: replace(FULL.ranking, **kw)
CONFIGS = {
    "임베딩만": EMBEDDING_ONLY,
    "조건 필터": FILTER_ONLY,
    "필터+선호 재랭킹\n+중복 제어 (이전 가점)": replace(FULL, ranking=R(menu_match_weight=0.15, context_match_weight=0.15)),
    "최종 (메뉴 0.3, 연상 0.05)": FULL,
}
per = {name: evaluate_per_query(rec_b, queries, cfg, jmap) for name, cfg in CONFIGS.items()}
summary = {name: summarize(pq) for name, pq in per.items()}
final = "최종 (메뉴 0.3, 연상 0.05)"
final_ndcg = [r["nDCG@5"] for r in per[final]]
boot = {name: paired_bootstrap([r["nDCG@5"] for r in pq], final_ndcg) for name, pq in per.items() if name != final}
text_a = summarize(evaluate_per_query(rec_a, queries, FULL, jmap))

# 그림 1: 단계별 성능
fig, ax = plt.subplots(figsize=(8, 3.8))
names = list(CONFIGS)
metrics = [("P@5", BLUE), ("nDCG@5", ORANGE), ("MRR", AQUA)]
y = np.arange(len(names))
h = 0.26
for i, (m, color) in enumerate(metrics):
    vals = [summary[n][m] for n in names]
    ax.barh(y + (i - 1) * h, vals, height=h - 0.03, color=color, label=m)
    for yy, v in zip(y + (i - 1) * h, vals):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3)
ax.set_title(f"그림 4. 파이프라인 단계별 성능 (승인 판정, 질의 {len(queries)}개, 텍스트 B)", loc="left", fontsize=11, color=INK, pad=24)
save(fig, "fig4_ablation.png")

# 그림 2: 질의별 nDCG@5, 임베딩만 vs 최종
emb = {r["질의"]: r["nDCG@5"] for r in per["임베딩만"]}
fin = {r["질의"]: r["nDCG@5"] for r in per[final]}
order = sorted(queries, key=lambda q: (fin[q], emb[q]))
fig, ax = plt.subplots(figsize=(8, 9))
yy = np.arange(len(order))
ax.hlines(yy, [emb[q] for q in order], [fin[q] for q in order], color=GRID, linewidth=2)
ax.scatter([emb[q] for q in order], yy, s=36, color=GRAY, label="임베딩만", zorder=3, edgecolors="white", linewidths=1.5)
ax.scatter([fin[q] for q in order], yy, s=36, color=BLUE, label="최종", zorder=3, edgecolors="white", linewidths=1.5)
ax.set_yticks(yy, order, fontsize=8.5)
ax.set_xlim(-0.03, 1.03)
ax.set_xlabel("nDCG@5")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", frameon=False)
ax.set_title("그림 6. 질의별 nDCG@5 (최종 점수 오름차순)", loc="left", fontsize=11, color=INK)
save(fig, "fig6_per_query.png")

# 그림 3: 후보 1,088건의 속성별 미확인 비율
ATTRS = ["매운맛", "국물", "제공온도", "조리법", "기름짐", "든든함"]
label_dist = {a: Counter((r.get("라벨") or {}).get(a) or "미확인" for r in index_b.records) for a in ATTRS}
unknown = [label_dist[a]["미확인"] / index_b.size for a in ATTRS]
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh(ATTRS, unknown, color=BLUE, height=0.55)
for a, v in zip(ATTRS, unknown):
    ax.text(v + 0.004, a, f"{v:.1%}", va="center", fontsize=9, color=INK2)
ax.invert_yaxis()
ax.set_xlim(0, max(unknown) * 1.2)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", visible=False)
ax.set_title(f"그림 1. 추천 후보 {index_b.size:,}건의 속성별 '미확인' 라벨 비율", loc="left", fontsize=11, color=INK)
save(fig, "fig1_unknown_labels.png")

# 그림 4: 언급 메뉴 가점 x 연상 가점
menu_ws, ctx_ws = [0.0, 0.15, 0.3, 0.5], [0.0, 0.05, 0.15]
grid = np.zeros((len(menu_ws), len(ctx_ws)))
for i, m in enumerate(menu_ws):
    for j, c in enumerate(ctx_ws):
        cfg = replace(FULL, ranking=R(menu_match_weight=m, context_match_weight=c))
        grid[i, j] = summarize(evaluate_per_query(rec_b, queries, cfg, jmap))["nDCG@5"]
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.imshow(grid, cmap="Blues", vmin=grid.min() - 0.03, vmax=grid.max(), aspect="auto")
for i in range(len(menu_ws)):
    for j in range(len(ctx_ws)):
        dark = grid[i, j] > (grid.min() + grid.max()) / 2
        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9,
                color="white" if dark else INK, fontweight="bold" if (menu_ws[i], ctx_ws[j]) == (0.3, 0.05) else None)
ax.set_xticks(range(len(ctx_ws)), [str(c) for c in ctx_ws])
ax.set_yticks(range(len(menu_ws)), [str(m) for m in menu_ws])
ax.set_xlabel("연상 가점 (context_match_weight)")
ax.set_ylabel("언급 메뉴 가점 (menu_match_weight)")
ax.grid(False)
ax.set_title("그림 7. 가점 조합별 nDCG@5", loc="left", fontsize=11, color=INK)
save(fig, "fig7_weight_grid.png")

# 그림 5: cross-encoder 재정렬 실험 (2026-09-25 기록값, 승인 전 모델 추정 판정 기준, 미판정 0%)
ce = [("임베딩만 (e5)", 0.5018, GRAY), ("CE 단독", 0.5818, ORANGE), ("e5 + 조건 필터", 0.6140, GRAY),
      ("CE + 조건 필터", 0.6180, ORANGE), ("CE + 필터 + 선호 (라벨 텍스트)", 0.7647, ORANGE),
      ("CE + 필터 + 선호 (이름 텍스트)", 0.7723, ORANGE), ("e5 + 필터 + 선호 (당시 기준)", 0.8125, BLUE)]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh([c[0] for c in ce], [c[1] for c in ce], color=[c[2] for c in ce], height=0.6)
for name, v, _ in ce:
    ax.text(v + 0.01, name, f"{v:.3f}", va="center", fontsize=9, color=INK2)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("nDCG@5")
ax.grid(axis="y", visible=False)
ax.set_title("그림 8. cross-encoder(bge-reranker-v2-m3) 재정렬 비교", loc="left", fontsize=11, color=INK)
save(fig, "fig8_cross_encoder.png")

# 그림 6: 질의별 적합 메뉴가 걸친 식품대분류 수
cat_of = {r["라벨링단위ID"]: r["식품대분류명"] for r in index_b.records}
spread = defaultdict(set)
for (q, uid), g in jmap.items():
    if g == 2 and uid in cat_of:
        spread[q].add(cat_of[uid])
spread_counts = Counter(len(spread[q]) for q in queries if spread[q])
fig, ax = plt.subplots(figsize=(6, 2.8))
ks = sorted(spread_counts)
ax.bar([str(k) for k in ks], [spread_counts[k] for k in ks], color=BLUE, width=0.6)
for k in ks:
    ax.text(str(k), spread_counts[k] + 0.2, str(spread_counts[k]), ha="center", fontsize=9, color=INK2)
ax.set_xlabel("적합(2점) 메뉴가 속한 식품대분류 수")
ax.set_ylabel("질의 수")
ax.grid(axis="x", visible=False)
ax.set_title("그림 9. 질의별 정답의 대분류 분산", loc="left", fontsize=11, color=INK)
save(fig, "fig9_category_spread.png")

# 그림 7: 교차 검증 혼동 행렬
cross = list(csv.DictReader(open(EVAL / "judgments_crosscheck.csv", encoding="utf-8-sig")))
cm = np.zeros((3, 3), dtype=int)
for r in cross:
    cm[int(r["기존적합도"]), int(r["교차적합도"])] += 1
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cm, cmap="Blues", aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_xticks(range(3), ["0", "1", "2"])
ax.set_yticks(range(3), ["0", "1", "2"])
ax.set_xlabel("교차 판정 (Sonnet)")
ax.set_ylabel("기존 판정")
ax.grid(False)
ax.set_title("그림 3. 판정 교차 검증 (공공 데이터 971행)", loc="left", fontsize=11, color=INK)
save(fig, "fig3_crosscheck.png")

# 수치 기록
fresh = Recommender(index_b, encode, ref_b)  # 질의 임베딩 캐시 없이 잰다
times = [fresh.recommend(q)["실행시간"]["전체"] for q in queries]
stats = {
    "질의수": len(queries), "제외질의": [q for q in all_queries if q not in queries],
    "질의유형": dict(Counter(query_type.get(q, "?") for q in queries)),
    "판정": {"전체": len(rows), "승인": sum(r["검토상태"] == "승인" for r in rows), "승인판정수": len(jmap),
             "적합도분포": dict(Counter(jmap.values()))},
    "요약": {n.replace("\n", " "): {k: round(v, 4) for k, v in s.items()} for n, s in summary.items()},
    "텍스트A_최종": {k: round(v, 4) for k, v in text_a.items()},
    "부트스트랩_최종대비": {n.replace("\n", " "): {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
                    for n, b in boot.items()},
    "질의별": {q: {"임베딩만": round(emb[q], 4), "최종": round(fin[q], 4)} for q in queries},
    "라벨분포": {a: dict(label_dist[a]) for a in ATTRS}, "후보수": index_b.size,
    "가점격자": {"menu": menu_ws, "context": ctx_ws, "nDCG@5": grid.round(4).tolist()},
    "대분류분산": dict(spread_counts),
    "교차검증": {"행": len(cross), "일치": int(np.trace(cm)), "행렬": cm.tolist()},
    "지연초": {"평균": round(float(np.mean(times)), 4), "최대": round(float(np.max(times)), 4)},
}
json.dump(stats, open(OUT / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: stats[k] for k in ("요약", "텍스트A_최종", "부트스트랩_최종대비", "판정", "지연초", "대분류분산", "교차검증")},
                 ensure_ascii=False, indent=1))

# 보강 수치: 판정 일치도, 유형별 성능, 후보 내 유사도 분포, 파서 커버리지
from src.preprocessing import parse_query
from src.retrieval import retrieve


def kappa(m, weights=None):
    m = np.asarray(m, float)
    n = m.sum()
    k = len(m)
    w = np.ones((k, k)) - np.eye(k) if weights is None else np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float)
    expected = np.outer(m.sum(1), m.sum(0)) / n
    return 1 - (w * m).sum() / (w * expected).sum()


TYPE = {"5단계 기존": "기본", "5단계 복합": "복합", "추가 복합": "복합", "5단계 부정": "부정", "추가 부정": "부정",
        "추가 맥락": "맥락", "추가 메뉴언급": "메뉴 언급", "5단계 미확정": "미확정"}
by_type = defaultdict(lambda: defaultdict(list))
for name in ("임베딩만", final):
    for r in per[name]:
        by_type[TYPE.get(query_type.get(r["질의"]), "?")][name].append(r["nDCG@5"])
spreads, top1 = [], []
for q in queries:
    sims = [c["유사도"] for c in retrieve(index_b, encode(q), 100)]
    spreads.append(sims[0] - sims[-1])
    top1.append(sims[0])
parsed = [parse_query(q) for q in queries]
extra = {
    "카파": {"단순": round(kappa(cm), 4), "이차가중": round(kappa(cm, "quadratic"), 4)},
    "유형별_nDCG": {t: {n: round(float(np.mean(v)), 4) for n, v in d.items()} | {"질의수": len(d[final])}
                  for t, d in by_type.items()},
    "후보100_유사도": {"1위평균": round(float(np.mean(top1)), 4), "1위와100위차_평균": round(float(np.mean(spreads)), 4),
                  "1위와100위차_최대": round(float(np.max(spreads)), 4)},
    "파서": {"조건1개이상": sum(bool(p.hard or p.soft) for p in parsed), "필수포함": sum(bool(p.hard) for p in parsed),
           "메뉴언급": sum(bool(p.menu_terms) for p in parsed), "맥락": sum(bool(p.context_terms) for p in parsed),
           "미처리포함": sum(bool(p.unhandled) for p in parsed), "조건없음": [p.text for p in parsed if not (p.hard or p.soft)]},
}
stats.update(extra)

# 그림 2: 후보 100개 안의 유사도 폭과 선호 한 단계의 크기
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.hist(np.array(spreads) * 0.7, bins=np.arange(0, 0.335, 0.005), color=BLUE, edgecolor="white", linewidth=1)
top = ax.get_ylim()[1]
for n_soft, style, y in ((1, "-", 0.92), (2, "--", 0.78), (3, ":", 0.92)):
    ax.axvline(0.3 / n_soft, color=INK2, linestyle=style, linewidth=1.2)
    ax.text(0.3 / n_soft + 0.003, top * y, f"선호 1단계\n(조건 {n_soft}개)", fontsize=8, color=INK2, va="top")
ax.axvline(0.05, color=ORANGE, linewidth=1.5)
ax.text(0.053, top * 0.55, "연상 가점\n0.05", fontsize=8, color=INK2, va="top")
ax.set_xlim(0, 0.34)
ax.set_xlabel("0.7 x (1위 유사도 - 100위 유사도)")
ax.set_ylabel("질의 수")
ax.grid(axis="x", visible=False)
ax.set_title("그림 2. 유사도 항의 점수 폭과 선호·가점 항의 크기 비교", loc="left", fontsize=11, color=INK)
save(fig, "fig2_similarity_spread.png")

# 그림 5: 질의 유형별 nDCG@5
types = [t for t in ("부정", "기본", "맥락", "복합", "메뉴 언급", "미확정") if t in by_type]
fig, ax = plt.subplots(figsize=(7, 3.2))
yy = np.arange(len(types))
for off, name, color, label in ((-0.2, "임베딩만", GRAY, "임베딩만"), (0.2, final, BLUE, "최종")):
    vals = [np.mean(by_type[t][name]) for t in types]
    ax.barh(yy + off, vals, height=0.37, color=color, label=label)
    for y0, v in zip(yy + off, vals):
        ax.text(v + 0.01, y0, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(yy, [f"{t} ({len(by_type[t][final])})" for t in types])
ax.invert_yaxis()
ax.set_xlim(0, 1.1)
ax.set_xlabel("nDCG@5")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
ax.set_title("그림 5. 질의 유형별 nDCG@5 (괄호는 질의 수)", loc="left", fontsize=11, color=INK, pad=24)
save(fig, "fig5_by_type.png")
json.dump(stats, open(OUT / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(extra, ensure_ascii=False, indent=1))
