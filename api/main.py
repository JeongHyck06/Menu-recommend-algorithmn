"""메뉴 추천 API

POST /parse      문장 -> 그룹별 태그 (맛, 메뉴, 상황)
POST /recommend  문장 -> Top-K 메뉴, 피드백 가점 반영
POST /feedback   추천 결과 좋아요·별로예요 저장

하루에 한 번 별도 프로세스로 api.finetune을 돌리고 모델이 바뀌면 다시 불러온다
"""

import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.feedback import FeedbackStore
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


def pick_menus(items: list, limit: int, boost=lambda keyword: 0.0) -> list:
    """피드백 가점을 더해 다시 정렬하고, 대표식품명이 같은 메뉴는 같은 검색어가 되므로 첫 항목만 남긴다"""
    keyed = [(it["대표식품명"] or it["메뉴명"], it) for it in items]
    scored = sorted(((it["최종점수"] + boost(k), k, it) for k, it in keyed), key=lambda x: -x[0])
    menus, seen = [], set()
    for score, keyword, it in scored:
        if keyword in seen:
            continue
        seen.add(keyword)
        menus.append({"keyword": keyword, "name": it["메뉴명"], "category": it["식품대분류명"],
                      "cuisine": it["계열"], "score": round(score, 4), "labels": it["주요라벨"]})
        if len(menus) == limit:
            break
    return menus


def retrain_loop(interval: int):
    from api.finetune import load_active, read_current

    while True:
        time.sleep(interval)
        run = subprocess.run([sys.executable, "-m", "api.finetune"], capture_output=True, text=True)
        print("finetune", run.returncode, run.stdout[-500:], run.stderr[-500:], flush=True)
        if read_current().get("dir", "base") != state["model"]:
            state["rec"], state["model"] = load_active()


@asynccontextmanager
async def lifespan(_app):
    from api.finetune import load_active

    state["feedback"] = FeedbackStore()
    state["rec"], state["model"] = load_active()
    interval = int(os.environ.get("FINETUNE_INTERVAL", "0"))
    if interval:
        threading.Thread(target=retrain_loop, args=(interval,), daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


class TextIn(BaseModel):
    text: str = Field(max_length=200)
    limit: int = Field(default=5, ge=1, le=10)


class FeedbackIn(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    keyword: str = Field(min_length=1, max_length=50)
    menu: str = Field(default="", max_length=100)
    place_id: str = Field(default="", max_length=30)
    liked: bool


@app.get("/health")
def health():
    return {"ok": "rec" in state, "model": state.get("model"), "feedback": state["feedback"].size()}


@app.post("/feedback")
def feedback(body: FeedbackIn):
    state["feedback"].add(body.query.strip(), body.keyword, body.menu, body.place_id, body.liked)
    return {"ok": True}


@app.post("/parse")
def parse(body: TextIn):
    return {"tags": extract_tags(body.text)}


@app.post("/recommend")
def recommend(body: TextIn):
    result = state["rec"].recommend(body.text, PipelineConfig(top_k=body.limit * 3))
    boost = lambda keyword: state["feedback"].boost(body.text, keyword)
    return {"status": result["상태"], "reason": result["사유"], "menus": pick_menus(result["추천"], body.limit, boost)}
