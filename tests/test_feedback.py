from api.feedback import FeedbackStore
from api.finetune import better, is_test_query, training_examples, unique_batches
from api.main import pick_menus


def test_boost_follows_likes_and_dislikes(tmp_path):
    store = FeedbackStore(tmp_path / "f.db")
    assert store.boost("라면", "라면") == 0.0
    store.add("얼큰한 라면", "라면", "라면", "1", True)
    store.add("얼큰한 라면", "라면", "라면", "2", True)
    store.add("얼큰한 라면", "우동", "우동", "3", False)
    assert store.boost("라면", "라면") > 0
    assert store.boost("얼큰한", "우동") < 0
    assert store.boost("국밥", "라면") == 0.0
    assert abs(store.boost("라면", "라면")) <= 0.15


def test_counts_survive_reopen_and_liked_pairs(tmp_path):
    store = FeedbackStore(tmp_path / "f.db")
    store.add("라면", "라면", "라면", "1", True)
    store.add("라면", "우동", "우동", "1", True)
    store.add("라면", "우동", "우동", "2", False)
    reopened = FeedbackStore(tmp_path / "f.db")
    assert reopened.boost("라면", "라면") == store.boost("라면", "라면")
    assert reopened.liked_pairs() == [("라면", "라면")]
    assert reopened.size() == 3


def test_pick_menus_reorders_by_feedback():
    item = lambda food, score: {"메뉴명": food, "대표식품명": food, "식품대분류명": "면류", "계열": "분식",
                                "최종점수": score, "주요라벨": "-"}
    items = [item("우동", 0.90), item("라면", 0.85)]
    boost = lambda k: 0.1 if k == "라면" else -0.1
    assert [m["keyword"] for m in pick_menus(items, 2, boost)] == ["라면", "우동"]
    assert [m["keyword"] for m in pick_menus(items, 2)] == ["우동", "라면"]


def test_training_examples_maps_keyword_and_skips_test_queries():
    records = [{"대표식품명": "라면", "메뉴명": "라면 라면만", "embedding_text": "라면 국물요리"},
               {"대표식품명": "라면", "메뉴명": "라면 해물", "embedding_text": "해물 라면"}]
    train_q = next(q for q in ("얼큰한 라면", "라면 먹고 싶어", "매운 라면", "라면 한 그릇") if not is_test_query(q))
    test_q = next(q for q in ("얼큰한 라면", "라면 먹고 싶어", "매운 라면", "라면 한 그릇") if is_test_query(q))
    pairs = [(train_q, "라면"), (test_q, "라면"), ("피자", "피자")]
    assert training_examples(pairs, records) == [(train_q, "라면 국물요리")]


def test_chosen_counts_double_and_migrates_old_table(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE feedback (id INTEGER PRIMARY KEY, created_at REAL, query TEXT, keyword TEXT,"
                " menu TEXT, place_id TEXT, liked INTEGER)")
    old.execute("INSERT INTO feedback (created_at, query, keyword, menu, place_id, liked) VALUES (0, '라면', '우동', '', '', 1)")
    old.commit()
    store = FeedbackStore(path)
    store.add("라면", "라면", "라면", "1", False, chosen=True)
    assert store.counts[("라면", "라면")] == [2, 0]
    assert store.counts[("라면", "우동")] == [1, 0]
    assert store.boost("라면", "라면") > store.boost("라면", "우동")


def test_swap_needs_clear_gain_without_recall_drop():
    old = {"ndcg": 0.776, "recall": 0.439}
    assert not better({"ndcg": 0.779, "recall": 0.408}, old)
    assert not better({"ndcg": 0.780, "recall": 0.45}, old)
    assert better({"ndcg": 0.79, "recall": 0.439}, old)


def test_unique_batches_never_repeat_query_or_menu():
    ex = [("a", "1"), ("a", "2"), ("b", "1"), ("b", "3"), ("c", "4")]
    batches = unique_batches(ex, 16)
    assert sorted(x for b in batches for x in b) == sorted(ex)
    assert all(len({q for q, _ in b}) == len(b) == len({d for _, d in b}) for b in batches)
