"""계열·안주·맨밥 규칙 테스트"""

from src.preprocessing.cuisine import classify_cuisine, enrich, is_anju, is_staple


def _r(menu, rep=None, cat=""):
    return {"라벨링단위ID": "x", "메뉴명": menu, "대표식품명": rep or menu, "식품대분류명": cat, "라벨": {}, "속성토큰": []}


def test_cuisine_rules_and_default():
    assert classify_cuisine(_r("잡탕밥")) == "중식"
    assert classify_cuisine(_r("짬뽕라면")) == "중식"
    assert classify_cuisine(_r("연어롤")) == "일식"
    assert classify_cuisine(_r("소고기 덮밥")) == "한식"
    assert classify_cuisine(_r("미트 스파게티")) == "양식"
    assert classify_cuisine(_r("분짜")) == "동남아"
    assert classify_cuisine(_r("김치 볶음밥")) == "한식"
    assert classify_cuisine(_r("떡볶이 고추장 어묵")) == "분식"


def test_anju_and_staple():
    assert is_anju(_r("부추전", cat="전·적 및 부침류")) and is_anju(_r("족발", cat="찜류"))
    assert not is_anju(_r("무 된장국", cat="국 및 탕류"))
    assert is_staple(_r("쌀밥")) and is_staple(_r("혼합잡곡밥")) and not is_staple(_r("곤드레밥"))
    assert not is_staple(_r("국밥"))


def test_enrich_prefers_chat_labels_and_adds_tokens():
    rec = enrich(_r("탕수육", cat="튀김류"))
    assert rec["라벨"]["계열"] == "중식" and rec["라벨"]["안주"] == "예" and rec["속성토큰"] == ["중식", "안주"]
    rec = enrich(_r("탕수육", cat="튀김류"), {"x": {"계열": "한식", "안주": "아니오"}})
    assert rec["라벨"]["계열"] == "한식" and rec["속성토큰"] == ["한식"] and rec["계열출처"] == "채팅 라벨"
