"""메뉴 추천 API

POST /parse      문장 -> 그룹별 태그 (맛, 메뉴, 상황)
POST /recommend  문장 -> Top-K 메뉴
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.preprocessing import parse_query
from src.recommendation import PipelineConfig

# 맥락 규칙 근거는 "비 오"처럼 잘려 있어 다시 파싱되는 자연스러운 표현으로 바꾼다
CONTEXT_LABELS = {"날씨_비": "비 오는 날", "날씨_추움": "추운 날", "날씨_더움": "더운 날"}

state = {}


def extract_tags(text: str) -> list:
    parsed = parse_query(text)
    tags = [{"label": c.evidence, "group": "맛", "avoid": c.strength == "hard"} for c in [*parsed.hard, *parsed.soft]]
    tags += [{"label": u["expression"], "group": "맛", "avoid": False} for u in parsed.unhandled]
    tags += [{"label": e["evidence"], "group": "메뉴", "avoid": True} for e in parsed.menu_exclusions]
    tags += [{"label": t, "group": "메뉴", "avoid": False} for t in parsed.menu_terms]
    tags += [{"label": CONTEXT_LABELS.get(i["rule"], i["expression"]), "group": "상황", "avoid": False}
             for i in parsed.ignored]
    seen, unique = set(), []
    for tag in tags:
        if tag["label"] not in seen:
            seen.add(tag["label"])
            unique.append(tag)
    return unique


def pick_menus(items: list, limit: int) -> list:
    """대표식품명이 같은 메뉴는 같은 검색어가 되므로 첫 항목만 남긴다"""
    menus, seen = [], set()
    for it in items:
        keyword = it["대표식품명"] or it["메뉴명"]
        if keyword in seen:
            continue
        seen.add(keyword)
        menus.append({"keyword": keyword, "name": it["메뉴명"], "category": it["식품대분류명"],
                      "cuisine": it["계열"], "score": it["최종점수"], "labels": it["주요라벨"]})
        if len(menus) == limit:
            break
    return menus


@asynccontextmanager
async def lifespan(_app):
    from src.embedding import E5Embedder
    from src.recommendation import Recommender
    from src.retrieval import load_index

    index, ref = load_index("B")
    embedder = E5Embedder()
    state["rec"] = Recommender(index, lambda t: embedder.encode_queries([t])[0], ref)
    yield


app = FastAPI(lifespan=lifespan)


class TextIn(BaseModel):
    text: str = Field(max_length=200)
    limit: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health():
    return {"ok": "rec" in state}


@app.post("/parse")
def parse(body: TextIn):
    return {"tags": extract_tags(body.text)}


@app.post("/recommend")
def recommend(body: TextIn):
    result = state["rec"].recommend(body.text, PipelineConfig(top_k=body.limit * 3))
    return {"status": result["상태"], "reason": result["사유"], "menus": pick_menus(result["추천"], body.limit)}
