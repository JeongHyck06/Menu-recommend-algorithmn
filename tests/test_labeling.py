"""LLM 라벨링 핵심 로직 테스트, 실제 API는 호출하지 않음"""

import json
from types import SimpleNamespace

import anthropic
import httpx
import numpy as np
import pandas as pd
import pytest

from src.labeling import review, runner, sampling, schema, validation
from src.labeling.prompt import LabelingConfig, build_batch_requests, build_request_params, build_unit_input


def _menu_frame() -> pd.DataFrame:
    rows = []
    code = 0

    def add(rep, category, menu_name, company=None, size=None):
        nonlocal code
        code += 1
        rows.append({
            "식품코드": f"D{code:04d}", "식품명": f"피자_{menu_name} ({size})" if company else rep,
            "메뉴명": menu_name, "이름접두어": "피자" if company else None,
            "대표식품명": rep, "식품대분류명": category, "업체명": company, "온도": None,
            "사이즈": size, "프랜차이즈여부": company is not None,
        })

    for category, reps in {
        "국 및 탕류": ["된장국", "미역국", "육개장"],
        "밥류": ["비빔밥", "김밥"],
        "구이류": ["감자그라탕", "고등어구이"],
    }.items():
        for rep in reps:
            add(rep, category, rep)
    # 프랜차이즈 피자: 업체 2곳 x 메뉴 3개 x 사이즈 2개
    for company in ["A피자", "B피자"]:
        for menu_idx in range(3):
            for size in ["L", "M"]:
                add("피자", "빵 및 과자류", f"치즈 피자 {menu_idx}", company, size)
    return pd.DataFrame(rows)


def test_build_labeling_units_groups_only_identical_context():
    menu = _menu_frame()
    units = sampling.build_labeling_units(menu)

    assert units["행수"].sum() == len(menu)
    assert units["라벨링단위ID"].is_unique
    # 사이즈만 다른 프랜차이즈 행은 한 단위, 비프랜차이즈 단일 행은 각자 단위
    pizza = units[units["대표식품명"] == "피자"]
    assert (pizza["행수"] == 2).all()
    assert set(pizza["식품코드목록"].iloc[0].split(";")) <= set(menu["식품코드"])
    assert units.loc[units["대표식품명"] == "감자그라탕", "경계메뉴"].all()
    assert not units.loc[units["대표식품명"] == "고등어구이", "경계메뉴"].any()


def test_sample_experiment_units_is_balanced_and_deterministic():
    units = sampling.build_labeling_units(_menu_frame())
    sample = sampling.sample_experiment_units(units, n_total=8, n_ambiguous=1, cap_per_representative=2, random_state=1)
    again = sampling.sample_experiment_units(units, n_total=8, n_ambiguous=1, cap_per_representative=2, random_state=1)

    assert len(sample) == 8
    assert sample["라벨링단위ID"].tolist() == again["라벨링단위ID"].tolist()
    assert sample["대표식품명"].value_counts().max() <= 2
    assert (sample["선정사유"] == sampling.REASON_AMBIGUOUS).sum() == 1
    assert sample["프랜차이즈여부"].nunique() == 2
    assert sample["식품대분류명"].nunique() >= 3


def _unit_input(code="D0001", temperature=None) -> dict:
    return build_unit_input({
        "식품코드": code, "식품명": "육개장", "메뉴명": "육개장", "이름접두어": np.nan,
        "대표식품명": "육개장", "식품대분류명": "국 및 탕류", "업체명": np.nan, "온도": temperature,
    })


def _payload(code="D0001", **overrides) -> dict:
    labels = {"매운맛": "보통", "국물": "국물요리", "제공온도": "뜨거움", "조리법": "끓임", "기름짐": "보통", "든든함": "든든함"}
    labels.update(overrides)
    return {"식품코드": code, "라벨": labels, "근거": "고춧가루 육수의 국 요리"}


def test_validate_response_accepts_valid_payload_and_code_fence():
    text = "```json\n" + json.dumps(_payload(), ensure_ascii=False) + "\n```"
    result = validation.validate_response(validation.parse_response_text(text), "D0001", None)
    assert result.labels["국물"] == "국물요리"
    assert result.review_flags == []


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"식품코드": "D0001", "라벨": {}}, "필수 필드 누락"),
        (_payload(code="D9999"), "식품코드 불일치"),
        ({**_payload(), "라벨": {k: v for k, v in _payload()["라벨"].items() if k != "국물"}}, "속성 누락"),
        (_payload(매운맛="아주매움"), "허용값 아님"),
        ({**_payload(), "근거": ""}, "근거가 비어"),
    ],
)
def test_validate_response_rejects_invalid(payload, message):
    with pytest.raises(validation.ResponseValidationError, match=message):
        validation.validate_response(payload, "D0001", None)


def test_parse_response_text_rejects_non_json():
    with pytest.raises(validation.ResponseValidationError, match="JSON 파싱 실패"):
        validation.parse_response_text("육개장은 맵다")


def test_original_temperature_conflict_is_flagged_not_overwritten():
    result = validation.validate_response(_payload(제공온도="차가움"), "D0001", "HOT")
    assert result.labels["제공온도"] == "차가움"
    assert validation.FLAG_TEMPERATURE_CONFLICT in result.review_flags

    ok = validation.validate_response(_payload(제공온도="미확인"), "D0001", "HOT")
    assert ok.review_flags == []


def test_check_result_set_reports_missing_and_duplicates():
    report = validation.check_result_set(["a", "a", "c"], ["a", "b"])
    assert report == {"누락": ["b"], "중복": ["a"], "대상외": ["c"]}


def test_cache_key_changes_with_input_model_or_versions():
    base = runner.make_cache_key(_unit_input(), "m1", "p1", "s1")
    assert base == runner.make_cache_key(_unit_input(), "m1", "p1", "s1")
    assert base != runner.make_cache_key(_unit_input(code="D0002"), "m1", "p1", "s1")
    assert base != runner.make_cache_key(_unit_input(), "m2", "p1", "s1")
    assert base != runner.make_cache_key(_unit_input(), "m1", "p2", "s1")
    assert base != runner.make_cache_key(_unit_input(), "m1", "p1", "s2")


def test_estimate_cost_uses_pricing_and_batch_discount():
    assert runner.estimate_cost_usd(1_000_000, 0, runner.PRICING and "claude-haiku-4-5-20251001") == 1.0
    assert runner.estimate_cost_usd(0, 1_000_000, "claude-haiku-4-5-20251001") == 5.0
    assert runner.estimate_cost_usd(1_000_000, 1_000_000, "claude-haiku-4-5-20251001", batch=True) == 3.0


class FakeClient:
    """messages.create 응답을 순서대로 돌려주는 테스트용 클라이언트"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            content=[SimpleNamespace(text=outcome)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


def _unit_row(code="D0001"):
    return pd.Series({
        "라벨링단위ID": f"u-{code}", "식품코드": code, "식품명": "육개장", "메뉴명": "육개장", "이름접두어": np.nan,
        "대표식품명": "육개장", "식품대분류명": "국 및 탕류", "업체명": np.nan, "온도": np.nan, "프랜차이즈여부": False,
    })


def _api_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    return cls("err", response=response, body=None)


def test_run_labeling_refuses_when_disabled(tmp_path):
    with pytest.raises(RuntimeError, match="enabled=False"):
        runner.run_labeling(pd.DataFrame([_unit_row()]), LabelingConfig(), runner.ResultStore(tmp_path / "r.jsonl"))


def test_label_unit_saves_success_and_reuses_cache(tmp_path):
    store = runner.ResultStore(tmp_path / "r.jsonl")
    config = LabelingConfig(enabled=True)
    client = FakeClient([json.dumps(_payload(), ensure_ascii=False)])

    first = runner.label_unit(client, _unit_row(), config, store, mode=runner.MODE_MOCK, sleep=lambda s: None)
    second = runner.label_unit(client, _unit_row(), config, store, mode=runner.MODE_MOCK, sleep=lambda s: None)

    assert first["status"] == runner.STATUS_SUCCESS and not first["reused"]
    assert first["cost_usd"] == runner.estimate_cost_usd(100, 50, config.model)
    assert second["reused"] and client.calls == 1
    assert len(store.load()) == 1

    # 프롬프트 버전이 바뀌면 캐시를 쓰지 않는다
    client.outcomes.append(json.dumps(_payload(), ensure_ascii=False))
    third = runner.label_unit(client, _unit_row(), LabelingConfig(enabled=True, prompt_version="v2"), store,
                              mode=runner.MODE_MOCK, sleep=lambda s: None)
    assert not third["reused"] and client.calls == 2


def test_label_unit_retries_transient_then_succeeds(tmp_path):
    store = runner.ResultStore(tmp_path / "r.jsonl")
    client = FakeClient([_api_error(anthropic.InternalServerError, 500), json.dumps(_payload(), ensure_ascii=False)])
    waits = []

    result = runner.label_unit(client, _unit_row(), LabelingConfig(enabled=True, max_retries=2), store,
                               mode=runner.MODE_MOCK, sleep=waits.append)

    assert result["status"] == runner.STATUS_SUCCESS and result["attempt"] == 2
    assert waits == [2.0]


def test_label_unit_records_validation_failure_separately(tmp_path):
    store = runner.ResultStore(tmp_path / "r.jsonl")
    client = FakeClient(["not json", json.dumps(_payload(code="D9999"), ensure_ascii=False)])

    result = runner.label_unit(client, _unit_row(), LabelingConfig(enabled=True, max_retries=1), store,
                               mode=runner.MODE_MOCK, sleep=lambda s: None)

    assert result["status"] == runner.STATUS_VALIDATION_FAILED
    statuses = [r["status"] for r in store.load()]
    assert statuses == [runner.STATUS_VALIDATION_FAILED] * 3
    assert "식품코드 불일치" in result["error"]
    assert store.successful_by_key(runner.MODE_MOCK) == {}


def test_label_unit_stops_on_permanent_error(tmp_path):
    store = runner.ResultStore(tmp_path / "r.jsonl")
    client = FakeClient([_api_error(anthropic.AuthenticationError, 401)])

    result = runner.label_unit(client, _unit_row(), LabelingConfig(enabled=True, max_retries=3), store,
                               mode=runner.MODE_MOCK, sleep=lambda s: None)

    assert result["status"] == runner.STATUS_PERMANENT_FAILED and client.calls == 1


def test_run_labeling_resumes_and_summarizes(tmp_path):
    store = runner.ResultStore(tmp_path / "r.jsonl")
    units = pd.DataFrame([_unit_row("D0001"), _unit_row("D0002")])
    config = LabelingConfig(enabled=True)

    client = FakeClient([json.dumps(_payload("D0001"), ensure_ascii=False), _api_error(anthropic.RateLimitError, 429),
                         _api_error(anthropic.RateLimitError, 429), _api_error(anthropic.RateLimitError, 429)])
    summary = runner.run_labeling(units, config, store, client=client, mode=runner.MODE_MOCK, sleep=lambda s: None)
    assert summary["status"].tolist() == [runner.STATUS_SUCCESS, runner.STATUS_TRANSIENT_FAILED]

    # 재개: 성공한 단위는 재사용, 실패한 단위만 다시 호출
    client = FakeClient([json.dumps(_payload("D0002"), ensure_ascii=False)])
    resumed = runner.run_labeling(units, config, store, client=client, mode=runner.MODE_MOCK, sleep=lambda s: None)
    assert resumed["reused"].tolist() == [True, False]
    assert resumed["status"].tolist() == [runner.STATUS_SUCCESS] * 2
    assert client.calls == 1


def test_batch_requests_share_params_with_direct_call():
    units = pd.DataFrame([_unit_row("D0001")])
    config = LabelingConfig()
    batch = build_batch_requests(units, config)
    direct = build_request_params(build_unit_input(units.iloc[0]), config)

    assert batch[0]["custom_id"] == "u-D0001"
    assert batch[0]["params"] == direct
    assert "thinking" not in direct


def _sample_frame():
    row = _unit_row("D0001").to_dict()
    row.update({"식품코드목록": "D0001", "선정사유": "카테고리 배분"})
    return pd.DataFrame([row])


def test_review_sheet_and_metrics_show_not_run_without_live_results():
    sheet = review.build_review_sheet(_sample_frame(), [])
    assert sheet.loc[0, "검토상태"] == review.REVIEW_NOT_RUN
    assert pd.isna(sheet.loc[0, "모델_매운맛"])

    metrics = review.compute_quality_metrics([], sheet)
    assert metrics["상태"] == review.NOT_RUN and metrics["형식오류율"] == review.NOT_RUN


def test_mock_results_do_not_enter_live_sheet_or_metrics(tmp_path):
    store = runner.ResultStore(tmp_path / "r.jsonl")
    client = FakeClient([json.dumps(_payload(), ensure_ascii=False)])
    runner.label_unit(client, _unit_row(), LabelingConfig(enabled=True), store, mode=runner.MODE_MOCK, sleep=lambda s: None)

    records = store.load()
    sheet = review.build_review_sheet(_sample_frame(), records)
    assert sheet.loc[0, "검토상태"] == review.REVIEW_NOT_RUN
    assert review.compute_quality_metrics(records, sheet)["상태"] == review.NOT_RUN

    mock_sheet = review.build_review_sheet(_sample_frame(), records, mode=runner.MODE_MOCK)
    assert mock_sheet.loc[0, "검토상태"] == review.REVIEW_PENDING
    assert mock_sheet.loc[0, "모델_국물"] == "국물요리"


def test_quality_metrics_with_review_values():
    records = [
        {"mode": "live", "라벨링단위ID": "u-D0001", "status": "validation_failed"},
        {"mode": "live", "라벨링단위ID": "u-D0001", "status": "success", "model": "m",
         "labels": {**_payload()["라벨"], "기름짐": schema.UNKNOWN}, "reason": "r", "review_flags": [validation.FLAG_TEMPERATURE_CONFLICT]},
    ]
    sheet = review.build_review_sheet(_sample_frame(), records)
    sheet.loc[0, "검토상태"] = review.REVIEW_CORRECTED
    sheet.loc[0, "검토_매운맛"] = "강함"

    metrics = review.compute_quality_metrics(records, sheet)
    assert metrics["형식오류율"] == 1.0
    assert metrics["원본충돌률"] == 1.0
    assert metrics["미확인비율_기름짐"] == 1.0 and metrics["미확인비율_국물"] == 0.0
    assert metrics["검토일치율_매운맛"] == 0.0 and metrics["검토일치율_국물"] == 1.0
