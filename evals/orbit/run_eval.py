"""Eval runner for the Orbit triage pipeline.

    python -m evals.orbit.run_eval --flow duplicates [--reps 3] [--cases d01,h01]
    python -m evals.orbit.run_eval --flow categories
    python -m evals.orbit.run_eval --flow summaries
    python -m evals.orbit.run_eval --self-test        free: known answers through the graders
    python -m evals.orbit.run_eval --judge-check      paid: 3 known-bad summaries through the judge
    python -m evals.orbit.run_eval --approve-harness  the user approves the current eval files

The pieces:
    flows.py             the three evals and their graders
    recording_model.py   retries, deadlines and the served-model check around the app's client
    results.py           writing rows and transcripts, and the printed summary
    harness_approval.py  the approval gate
    pricing.py           what each row cost
"""

import argparse
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from app.ai.llm import LLMOutputError, LLMRefusalError
from app.config import Settings, get_settings
from evals.orbit.dataset import Item, load_corpus, load_probes
from evals.orbit.flows import (
    FLOW_CLASSES,
    JUDGE_MODEL,
    CaseOutcome,
    SummariesFlow,
    check_summary_format,
    describe_item_for_report,
    grade_category_choice,
    grade_duplicate_verdict,
    grade_summary,
)
from evals.orbit.harness_approval import (
    EVAL_OUTPUT_ROOT,
    load_or_create_flow_state,
    record_harness_approval,
    stop_unless_harness_is_approved,
)
from evals.orbit.metrics import METRICS_BY_FLOW, all_zero_grade
from evals.orbit.pricing import calculate_cost_usd, find_price
from evals.orbit.recording_model import (
    HarnessError,
    RecordedCall,
    RecordingLanguageModel,
    create_claude_client_for_eval,
)
from evals.orbit.results import (
    ResultsWriter,
    add_up_token_usage,
    build_trace,
    summarize_results,
)


def judge_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"llm_model": JUDGE_MODEL, "llm_effort": "medium"})


# --- Running one (case, rep) -----------------------------------------------------


class CaseRunner:
    """Runs a single (case, rep) of a flow and writes its result or error."""

    def __init__(self, flow, settings: Settings, writer: ResultsWriter, timeout_seconds: float):
        self.flow = flow
        self.settings = settings
        self.writer = writer
        self.timeout_seconds = timeout_seconds
        self.app_client = create_claude_client_for_eval(settings, timeout_seconds)
        self.judge_client = create_claude_client_for_eval(judge_settings(settings), timeout_seconds)

    def run(self, item: Item, rep: int) -> str:
        """Run and record one (case, rep). Never raises: every failure is recorded.

        Returns a one-line description for the progress output.
        """
        deadline = time.monotonic() + self.timeout_seconds
        model = RecordingLanguageModel(self.app_client, self.settings.llm_model, deadline)
        judge_model = RecordingLanguageModel(self.judge_client, JUDGE_MODEL, deadline)
        started_at = time.perf_counter()

        try:
            status, outcome, note = self.run_flow_case(item, model, judge_model)
            latency_seconds = time.perf_counter() - started_at
            row = self.build_result_row(item, rep, status, outcome, model, judge_model, latency_seconds)
        except HarnessError as error:
            self.write_error_row(item, rep, error.failure_class, str(error), error.attempts, model, judge_model, error.result)
            return f"{item.id}#{rep} ERROR {error.failure_class}"
        except Exception as error:  # a bug in the eval itself; record it with the billed usage
            message = f"{type(error).__name__}: {error}"
            self.write_error_row(item, rep, "harness_error", message, None, model, judge_model, None)
            return f"{item.id}#{rep} ERROR harness_error: {error}"

        if model.calls:
            trace_note = note
        else:
            trace_note = note or "No LLM call: no candidates above the similarity threshold."
        trace = build_trace(model.calls, outcome.extra_trace_turns, trace_note)
        self.writer.write_result(row, trace)
        return f"{item.id}#{rep} {outcome.grade}"

    def run_flow_case(self, item: Item, model, judge_model) -> tuple[str, CaseOutcome, str | None]:
        """Return (status, outcome, note). Refusals and truncation are graded, not errors."""
        try:
            outcome = self.flow.run_case(item, model, judge_model)
            return "ok", outcome, None
        except LLMRefusalError as refusal:
            grade = all_zero_grade(self.flow.name)
            grade["refused"] = 1.0
            outcome = CaseOutcome(grade=grade, prompt_text=describe_item_for_report(item))
            return "ok", outcome, f"Refused: {refusal}"
        except LLMOutputError as error:
            outcome = CaseOutcome(grade=all_zero_grade(self.flow.name), prompt_text=describe_item_for_report(item))
            return "truncated", outcome, str(error)

    def build_result_row(
        self,
        item: Item,
        rep: int,
        status: str,
        outcome: CaseOutcome,
        model: RecordingLanguageModel,
        judge_model: RecordingLanguageModel,
        latency_seconds: float,
    ) -> dict:
        app_calls = model.calls
        judge_calls = outcome.judge_calls or judge_model.calls

        app_usage = add_up_token_usage(app_calls)
        if app_calls:
            served_model = app_calls[0].result.model
            app_cost = calculate_cost_usd(served_model, app_usage)
        else:
            # The duplicates flow skips the model when the search finds no candidates.
            served_model = self.settings.llm_model
            app_cost = 0.0

        judge_usage = None
        served_judge_model = None
        judge_cost = 0.0
        if judge_calls:
            judge_usage = add_up_token_usage(judge_calls)
            served_judge_model = judge_calls[0].result.model
            judge_cost = calculate_cost_usd(served_judge_model, judge_usage)

        if status == "truncated":
            stop_reason = "max_tokens"
        elif outcome.grade.get("refused"):
            stop_reason = "refusal"
        else:
            stop_reason = "end_turn"

        attempts = 0
        for call in app_calls + judge_calls:
            attempts += call.attempts

        return {
            "prompt_id": item.id,
            "rep": rep,
            "prompt": outcome.prompt_text,
            "tags": self.flow.tags_for(item),
            "status": status,
            "stop_reason": stop_reason,
            "grade": outcome.grade,
            "explanation": outcome.explanation or None,
            "model": served_model,
            "usage": app_usage,
            "judge_model": served_judge_model,
            "judge_usage": judge_usage,
            "cost_usd": round(app_cost + judge_cost, 6),
            "latency_s": round(latency_seconds, 3),
            "attempts": attempts,
            "meta": outcome.meta,
        }

    def write_error_row(
        self,
        item: Item,
        rep: int,
        failure_class: str,
        message: str,
        attempts: int | None,
        model: RecordingLanguageModel,
        judge_model: RecordingLanguageModel,
        unrecorded_result,
    ) -> None:
        """Errors never go into results.jsonl, so a re-run tries the case again."""
        billed_calls = list(model.calls) + list(judge_model.calls)
        if unrecorded_result is not None:
            billed_calls.append(RecordedCall("", "", unrecorded_result, attempts or 1))
        if attempts is None:
            attempts = sum(call.attempts for call in billed_calls)

        served_model = None
        if billed_calls:
            served_model = billed_calls[0].result.model

        self.writer.write_error(
            {
                "prompt_id": item.id,
                "rep": rep,
                "failure_class": failure_class,
                "message": message,
                "attempts": attempts,
                "model": served_model,
                "usage": add_up_token_usage(billed_calls),
            }
        )


# --- Running a whole flow ---------------------------------------------------------


def select_cases(all_cases: list[Item], case_ids_argument: str | None) -> list[Item]:
    if not case_ids_argument:
        return all_cases
    wanted_ids = set(case_ids_argument.split(","))
    return [case for case in all_cases if case.id in wanted_ids]


def run_flow(flow, args, settings: Settings) -> None:
    writer = ResultsWriter(EVAL_OUTPUT_ROOT / flow.name / args.variant)
    finished = writer.find_finished_case_reps()
    cases = select_cases(flow.cases, args.cases)

    work_to_do = []
    for case in cases:
        for rep in range(args.reps):
            if (case.id, rep) not in finished:
                work_to_do.append((case, rep))
    print(
        f"[{flow.name}] {len(work_to_do)} to run ({len(finished)} already done), "
        f"model {settings.llm_model}"
    )

    case_runner = CaseRunner(flow, settings, writer, args.timeout_s)
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for case, rep in work_to_do:
            futures.append(pool.submit(case_runner.run, case, rep))
        for number_done, future in enumerate(as_completed(futures), start=1):
            progress_line = future.result()
            if args.verbose or "ERROR" in progress_line:
                print(f"  {number_done}/{len(work_to_do)} {progress_line}")
    print(f"[{flow.name}] finished in {time.perf_counter() - started_at:.1f}s")

    case_ids = {case.id for case in cases}
    summarize_results(flow.name, writer.output_dir, case_ids)


# --- Free self-test and paid judge check ----------------------------------------


class SelfTestReport:
    def __init__(self):
        self.failures = 0

    def check(self, description: str, passed: bool, detail: str) -> None:
        if not passed:
            self.failures += 1
        print(f"  {'PASS' if passed else 'FAIL'} {description}: {detail}")


def share(values: list[float]) -> float:
    return sum(values) / len(values)


def run_self_test() -> int:
    """Known answers through the programmatic graders. No API calls."""
    probes = load_probes()
    all_items = load_corpus() + probes
    non_duplicate_share = sum(1 for probe in probes if probe.duplicate_of is None) / len(probes)
    report = SelfTestReport()

    print("duplicates grader:")
    oracle = share([grade_duplicate_verdict(p, p.duplicate_of)["correct"] for p in probes])
    report.check("oracle", oracle == 1.0, f"{oracle:.1%}")
    never_merge = share([grade_duplicate_verdict(p, None)["correct"] for p in probes])
    report.check("never merge = share of non-duplicates", abs(never_merge - non_duplicate_share) < 1e-9, f"{never_merge:.1%}")
    always_c27 = share([grade_duplicate_verdict(p, "c27")["correct"] for p in probes])
    report.check("always merge into c27", always_c27 < 0.1, f"{always_c27:.1%}")

    print("categories grader:")
    oracle = share([grade_category_choice(i, "assign", i.category)["correct"] for i in all_items])
    report.check("oracle", oracle == 1.0, f"{oracle:.1%}")
    constant = share([grade_category_choice(i, "assign", "Tasks & Projects")["correct"] for i in all_items])
    report.check("constant 'Tasks & Projects'", constant < 0.35, f"{constant:.1%}")
    always_new = share([grade_category_choice(i, "propose_new", None)["correct"] for i in all_items])
    report.check("always propose new", always_new == 0.0, f"{always_new:.1%}")

    print("summary format check:")
    existing = "A dark color theme for the web app."
    format_cases = [
        ("Users want a dark theme. It should follow the OS setting.", True),
        ("Dark mode.", False),
        ("Several users ask for a dark theme. It reduces eye strain.", False),
        ("Users want a dark theme. 12 customers requested it.", False),
        (existing, True),  # an unchanged one-sentence summary is allowed
    ]
    for text, should_pass in format_cases:
        passed, reason = check_summary_format(text, existing)
        report.check(repr(text), passed == should_pass, reason)

    if report.failures:
        return 1
    return 0


def run_judge_check(settings: Settings, timeout_seconds: float) -> int:
    """Known-bad summaries must fail the judge. Makes 3 paid judge calls."""
    flow = SummariesFlow(settings, Path(tempfile.mkdtemp()))
    judge_client = create_claude_client_for_eval(judge_settings(settings), timeout_seconds)
    item = next(case for case in flow.cases if case.id == "d02")
    existing_summary = flow.existing_feature_by_probe_id[item.id].description

    bad_summaries = [
        ("empty", ""),
        ("I don't know", "I don't know."),
        ("confident, wrong feature",
         "Users want to export their tasks to a CSV file. This lets finance work with the data in spreadsheets."),
    ]
    failures = 0
    for label, bad_summary in bad_summaries:
        judge_model = RecordingLanguageModel(judge_client, JUDGE_MODEL, time.monotonic() + timeout_seconds)
        result = grade_summary(flow.judge_prompt, judge_model, existing_summary, item, bad_summary, True)
        failed_as_expected = result.grade["all_pass"] == 0.0 and result.grade["keeps_existing"] == 0.0
        if not failed_as_expected:
            failures += 1
        print(f"  {'PASS' if failed_as_expected else 'FAIL'} {label}: {result.grade}")
        if not failed_as_expected:
            print(f"    {result.explanation}")
    if failures:
        return 1
    return 0


# --- Command line -----------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.orbit.run_eval")
    parser.add_argument("--flow", choices=list(METRICS_BY_FLOW) + ["all"], default="all")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--model", help="model under test (default: the LLM_MODEL setting)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cases", help="comma-separated case ids, for a pilot")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--approve-harness", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--judge-check", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="only recompute summary.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"baseline|v\d+", args.variant):
        parser.error("--variant must be 'baseline' or v<N>")
    return args


def main() -> int:
    args = parse_arguments()
    load_dotenv()  # the same credential source as the app (app/main.py)

    if args.self_test:
        return run_self_test()

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    for model_name in (settings.llm_model, JUDGE_MODEL):
        if find_price(model_name) is None:
            print(f"No price for {model_name}; add it to evals/orbit/pricing.py first.", file=sys.stderr)
            return 2
    model_source = "--model" if args.model else "the LLM_MODEL setting"
    print(f"Model under test: {settings.llm_model} (from {model_source}); judge: {JUDGE_MODEL}")

    if args.judge_check:
        return run_judge_check(settings, args.timeout_s)

    if args.flow == "all":
        flow_names = list(METRICS_BY_FLOW)
    else:
        flow_names = [args.flow]

    if args.summarize:
        for flow_name in flow_names:
            summarize_results(flow_name, EVAL_OUTPUT_ROOT / flow_name / args.variant)
        return 0

    if args.approve_harness:
        # Approving never makes paid calls in the same command.
        for flow_name in flow_names:
            state = load_or_create_flow_state(flow_name, args.reps)
            record_harness_approval(flow_name, state)
        return 0

    for flow_name in flow_names:
        state = load_or_create_flow_state(flow_name, args.reps)
        stop_unless_harness_is_approved(flow_name, state)

    # SQLite files can stay locked on Windows until the process exits.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        for flow_name in flow_names:
            flow = FLOW_CLASSES[flow_name](settings, Path(temp_dir))
            run_flow(flow, args, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
