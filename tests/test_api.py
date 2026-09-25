from api.main import extract_tags, pick_menus


def test_extract_tags_groups():
    tags = extract_tags("비 오는 날 얼큰한 국물, 피자 말고 혼밥")
    by_label = {t["label"]: t for t in tags}
    assert by_label["얼큰한"]["group"] == "맛"
    assert by_label["국물"]["group"] == "맛"
    assert by_label["피자 말고"] == {"label": "피자 말고", "group": "메뉴", "avoid": True}
    assert by_label["비 오는 날"]["group"] == "상황"
    assert by_label["혼밥"]["group"] == "상황"
    assert len(tags) == len(by_label)


def test_extract_tags_empty():
    assert extract_tags("") == []


def test_pick_menus_dedups_keyword():
    item = lambda name, food, score: {"메뉴명": name, "대표식품명": food, "식품대분류명": "밥류", "계열": "한식",
                                      "최종점수": score, "주요라벨": "-"}
    menus = pick_menus([item("국밥 순대국밥", "국밥", 1.2), item("국밥 굴", "국밥", 1.1), item("파전", "파전", 0.9)], 5)
    assert [m["keyword"] for m in menus] == ["국밥", "파전"]
