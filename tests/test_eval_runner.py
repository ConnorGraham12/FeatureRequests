"""The eval runner's plumbing, with a fake API client. No paid calls."""

import json
from argparse import Namespace

import pytest

from app.ai.llm import LLMError, LLMResult
from app.config import Settings
from app.pipeline.schemas import CategoryAssignment, DuplicateJudgment, SummaryUpdate
from evals.orbit import flows, pricing, run_eval
from evals.orbit.metrics import metric_ids_for_flow
from tests.fakes import FakeEmbedder


def canned_answer(output_type):
    if output_type is DuplicateJudgment:
        return DuplicateJudgment(
            duplicate_of_feature_id=None, confidence=0.9, reasoning=".", rejected_candidates=[]
        )
    if output_type is CategoryAssignment:
        return CategoryAssignment(
            action="assign", category_id=1, new_category_name=None, new_category_description=None,
            feature_summary="x", confidence=0.9, reasoning=".",
        )
    if output_type is SummaryUpdate:
        return SummaryUpdate(
            summary="Users want a dark theme. It should follow the OS.", changed=True, reasoning="."
        )
    if output_type is flows.SummaryJudgment:
        passed = flows.SummaryCheck(passed=True, reasoning="ok")
        return flows.SummaryJudgment(
            keeps_existing=passed, adds_new_info=passed, no_fabrication=passed, flag_consistent=passed
        )
    raise AssertionError(f"no canned answer for {output_type}")


class FakeClaudeClient:
    """Stands in for ClaudeClient. Can pretend to be a different model, or fail every call."""

    def __init__(self, model_name: str, served_model_name: str | None = None, always_fail: bool = False):
        self.served_model_name = served_model_name or model_name
        self.always_fail = always_fail

    def request_structured_output(self, *, system_prompt, user_message, output_type):
        if self.always_fail:
            raise LLMError("BadRequestError: nope") from ValueError("not retryable")
        return LLMResult(
            output=canned_answer(output_type),
            model=self.served_model_name,
            latency_ms=1,
            input_tokens=1000,
            output_tokens=100,
            request_id="r",
            fallback_used=False,
            raw_response="{}",
        )


@pytest.fixture
def eval_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "EVAL_OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(flows, "get_shared_embedder", lambda model_name: FakeEmbedder())
    settings = Settings(_env_file=None, llm_model="claude-haiku-5-5")
    args = Namespace(variant="baseline", reps=2, cases=None, concurrency=4, timeout_s=30, verbose=False)
    return tmp_path, settings, args


def use_fake_app_client(monkeypatch, **fake_options):
    """Replace the real API client. fake_options apply to the model under test, not the judge."""

    def create_fake_client(settings, timeout_seconds):
        if settings.llm_model == flows.JUDGE_MODEL:
            return FakeClaudeClient(settings.llm_model)
        return FakeClaudeClient(settings.llm_model, **fake_options)

    monkeypatch.setattr(run_eval, "create_claude_client_for_eval", create_fake_client)


def read_output_rows(tmp_path, flow_name, file_name="results.jsonl"):
    path = tmp_path / "out" / flow_name / "baseline" / file_name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("flow_class", [flows.DuplicatesFlow, flows.CategoriesFlow, flows.SummariesFlow])
def test_every_row_is_complete_and_rerunning_adds_nothing(eval_setup, monkeypatch, flow_class):
    tmp_path, settings, args = eval_setup
    use_fake_app_client(monkeypatch)
    flow = flow_class(settings, tmp_path)

    run_eval.run_flow(flow, args, settings)
    rows_after_first_run = read_output_rows(tmp_path, flow.name)
    run_eval.run_flow(flow, args, settings)  # resume: everything is already done

    assert len(rows_after_first_run) == len(flow.cases) * 2
    assert len(read_output_rows(tmp_path, flow.name)) == len(rows_after_first_run)

    expected_metric_ids = set(metric_ids_for_flow(flow.name))
    for row in rows_after_first_run:
        assert set(row["grade"]) == expected_metric_ids
        assert row["status"] == "ok"
        assert row["model"] == "claude-haiku-5-5"
        trace_path = tmp_path / "out" / flow.name / "baseline" / "traces" / f"{row['prompt_id']}_rep{row['rep']}.json"
        assert trace_path.exists()
        model_was_called = flow.name != "duplicates" or row["meta"]["llm_called"]
        if model_was_called:
            assert row["usage"]["input_tokens"] > 0
            assert row["cost_usd"] > 0

    if flow.name == "summaries":
        for row in rows_after_first_run:
            assert row["judge_model"] == flows.JUDGE_MODEL
            assert row["judge_usage"]["input_tokens"] > 0


def test_api_errors_go_to_the_errors_file_not_the_results(eval_setup, monkeypatch):
    tmp_path, settings, args = eval_setup
    use_fake_app_client(monkeypatch, always_fail=True)

    run_eval.run_flow(flows.CategoriesFlow(settings, tmp_path), args, settings)

    assert read_output_rows(tmp_path, "categories") == []
    errors = read_output_rows(tmp_path, "categories", "errors.jsonl")
    assert errors
    for error in errors:
        assert error["failure_class"] == "serving_error"


def test_a_different_served_model_is_an_error(eval_setup, monkeypatch):
    tmp_path, settings, args = eval_setup
    use_fake_app_client(monkeypatch, served_model_name="claude-opus-5-5")
    args.cases = "c01"

    run_eval.run_flow(flows.CategoriesFlow(settings, tmp_path), args, settings)

    assert read_output_rows(tmp_path, "categories") == []
    failure_classes = {error["failure_class"] for error in read_output_rows(tmp_path, "categories", "errors.jsonl")}
    assert failure_classes == {"model_mismatch"}


def test_a_crash_in_one_case_is_recorded_and_the_run_continues(eval_setup, monkeypatch):
    tmp_path, settings, args = eval_setup
    use_fake_app_client(monkeypatch)
    flow = flows.CategoriesFlow(settings, tmp_path)
    real_run_case = flow.run_case

    def crash_on_c01(item, model, judge_model):
        if item.id == "c01":
            raise RuntimeError("boom")
        return real_run_case(item, model, judge_model)

    monkeypatch.setattr(flow, "run_case", crash_on_c01)
    args.cases = "c01,c02"
    args.reps = 1
    run_eval.run_flow(flow, args, settings)

    assert [row["prompt_id"] for row in read_output_rows(tmp_path, "categories")] == ["c02"]
    errors = read_output_rows(tmp_path, "categories", "errors.jsonl")
    assert errors[0]["failure_class"] == "harness_error"
    assert "boom" in errors[0]["message"]


def test_an_unpriced_model_is_recorded_as_an_error_not_a_crash(eval_setup, monkeypatch):
    tmp_path, settings, args = eval_setup
    use_fake_app_client(monkeypatch, served_model_name="claude-haiku-5-5-preview")
    monkeypatch.setattr(pricing, "PRICES_PER_MILLION_TOKENS", {"claude-sonnet-5-5": {"in": 2.0, "out": 10.0}})
    args.cases = "c01"
    args.reps = 1

    run_eval.run_flow(flows.CategoriesFlow(settings, tmp_path), args, settings)

    assert read_output_rows(tmp_path, "categories") == []
    assert "no price" in read_output_rows(tmp_path, "categories", "errors.jsonl")[0]["message"]
