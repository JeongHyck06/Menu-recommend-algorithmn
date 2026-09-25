from api.feedback import FeedbackStore
from api.finetune import training_examples
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


def test_training_examples_maps_keyword_to_menu_text():
    records = [{"대표식품명": "라면", "메뉴명": "라면 라면만", "embedding_text": "라면 국물요리"},
               {"대표식품명": "라면", "메뉴명": "라면 해물", "embedding_text": "해물 라면"}]
    assert training_examples([("얼큰한 라면", "라면"), ("피자", "피자")], records) == [("얼큰한 라면", "라면 국물요리")]
