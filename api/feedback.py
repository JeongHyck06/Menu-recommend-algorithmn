"""사용자 피드백 저장과 재랭킹 가점

질의를 공백으로 나눈 단어와 메뉴 검색어 조합별로 좋아요·별로예요 수를 세고
(좋아요 - 별로예요) / (전체 + 사전값)을 단어 평균해 가점으로 쓴다
"""

import os
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

STATE_DIR = Path(os.environ.get("STATE_DIR", Path(__file__).resolve().parents[1] / "state"))


def tokens(query: str) -> set:
    return set(query.split())


class FeedbackStore:
    def __init__(self, path=None):
        path = Path(path or STATE_DIR / "feedback.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, created_at REAL, query TEXT, keyword TEXT,"
            " menu TEXT, place_id TEXT, liked INTEGER)")
        self.lock = threading.Lock()
        self.counts = defaultdict(lambda: [0, 0])
        for query, keyword, liked in self.conn.execute("SELECT query, keyword, liked FROM feedback"):
            self._count(query, keyword, liked)

    def _count(self, query, keyword, liked):
        for t in tokens(query):
            self.counts[(t, keyword)][0 if liked else 1] += 1

    def add(self, query, keyword, menu, place_id, liked: bool):
        with self.lock:
            self.conn.execute("INSERT INTO feedback (created_at, query, keyword, menu, place_id, liked)"
                              " VALUES (?, ?, ?, ?, ?, ?)", (time.time(), query, keyword, menu, place_id, int(liked)))
            self.conn.commit()
            self._count(query, keyword, liked)

    def boost(self, query, keyword, weight=0.15, prior=2.0) -> float:
        """-weight ~ +weight, 피드백이 없으면 0"""
        ts = tokens(query)
        if not ts:
            return 0.0
        total = 0.0
        for t in ts:
            likes, dislikes = self.counts.get((t, keyword), (0, 0))
            total += (likes - dislikes) / (likes + dislikes + prior)
        return weight * total / len(ts)

    def liked_pairs(self) -> list:
        """파인튜닝용 (질의, 메뉴 검색어), 좋아요가 별로예요보다 많은 쌍만"""
        return self.conn.execute("SELECT query, keyword FROM feedback GROUP BY query, keyword"
                                 " HAVING SUM(liked) * 2 > COUNT(*)").fetchall()

    def size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
