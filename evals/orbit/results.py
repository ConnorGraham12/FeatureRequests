"""Writing eval results to disk, and summarizing them.

Per flow and variant, under .claude/hillclimb/<flow>/<variant>/:
    results.jsonl             one row per (case, rep), appended as each finishes
    traces/<id>_rep<k>.json   the full exchange with the model for that row
    errors.jsonl              attempts that produced nothing to score
    summary.json              the aggregate numbers printed at the end
"""

import json
import math
import threading
from pathlib import Path
from typing import Any

from evals.orbit.metrics import METRICS_BY_FLOW
from evals.orbit.recording_model import RecordedCall


class ResultsWriter:
    """Thread-safe writer for one flow's results directory."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results_path = output_dir / "results.jsonl"
        self.errors_path = output_dir / "errors.jsonl"
        self.traces_dir = output_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def find_finished_case_reps(self) -> set[tuple[str, int]]:
        """(case id, rep) pairs that already have a result, so a re-run skips them."""
        finished = set()
        if not self.results_path.exists():
            return finished
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            finished.add((row["prompt_id"], row["rep"]))
        return finished

    def write_result(self, row: dict, trace: list[dict]) -> None:
        trace_path = self.traces_dir / f"{row['prompt_id']}_rep{row['rep']}.json"
        with self._lock:
            trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
            with self.results_path.open("a", encoding="utf-8") as results_file:
                results_file.write(json.dumps(row) + "\n")

    def write_error(self, row: dict) -> None:
        with self._lock:
            with self.errors_path.open("a", encoding="utf-8") as errors_file:
                errors_file.write(json.dumps(row) + "\n")


def add_up_token_usage(calls: list[RecordedCall]) -> dict[str, int]:
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    for call in calls:
        usage["input_tokens"] += call.result.input_tokens
        usage["output_tokens"] += call.result.output_tokens
        usage["cache_read_input_tokens"] += call.result.cache_read_input_tokens
        usage["cache_creation_input_tokens"] += call.result.cache_creation_input_tokens
    return usage


def build_trace(app_calls: list[RecordedCall], extra_turns: list[dict], note: str | None) -> list[dict]:
    """The conversation as the report expects it: system, user and assistant turns."""
    trace = []
    for call in app_calls:
        trace.append({"role": "system", "content": call.system_prompt})
        trace.append({"role": "user", "content": call.user_message})
        trace.append({"role": "assistant", "content": call.result.output.model_dump_json(indent=2)})
    if note:
        trace.append({"role": "assistant", "content": note})
    trace.extend(extra_turns)
    return trace


# --- Summary --------------------------------------------------------------------


def mean_with_95_percent_margin(values: list[float]) -> tuple[float, float]:
    """Return (mean, half-width of the 95% confidence interval)."""
    count = len(values)
    if count == 0:
        return math.nan, math.nan
    mean = sum(values) / count
    if count == 1:
        return mean, math.nan
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    standard_error = math.sqrt(variance) / math.sqrt(count)
    return mean, 1.96 * standard_error


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def group_rows_by_case(rows: list[dict]) -> dict[str, list[dict]]:
    rows_by_case: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_case.setdefault(row["prompt_id"], []).append(row)
    return rows_by_case


def share_of_rows(rows: list[dict], metric_id: str) -> float:
    if not rows:
        return math.nan
    return sum(row["grade"][metric_id] for row in rows) / len(rows)


def summarize_duplicate_detection(ok_rows: list[dict]) -> dict[str, float]:
    """Recall, specificity and precision, plus how often the true feature was even a candidate."""
    duplicate_rows = [row for row in ok_rows if row["meta"]["gold"]]
    non_duplicate_rows = [row for row in ok_rows if not row["meta"]["gold"]]
    rows_that_merged = [row for row in ok_rows if row["meta"]["predicted"]]

    correct_merges = sum(row["grade"]["correct"] for row in duplicate_rows)
    if rows_that_merged:
        precision = correct_merges / len(rows_that_merged)
    else:
        precision = math.nan

    found_in_candidates = [row["meta"]["gold_in_candidates"] for row in duplicate_rows]
    if found_in_candidates:
        candidate_recall = sum(found_in_candidates) / len(found_in_candidates)
    else:
        candidate_recall = math.nan

    stats = {
        "recall": share_of_rows(duplicate_rows, "correct"),
        "specificity": share_of_rows(non_duplicate_rows, "correct"),
        "precision": precision,
        "candidate_recall": candidate_recall,
    }
    for kind in ("paraphrase", "extension", "hard_negative", "easy_negative"):
        rows_of_kind = [row for row in ok_rows if row["tags"][0] == kind]
        if rows_of_kind:
            stats[f"correct_{kind}"] = share_of_rows(rows_of_kind, "correct")
    return stats


def middle_value(sorted_values: list[float]) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[len(sorted_values) // 2]


def summarize_results(flow_name: str, output_dir: Path, only_case_ids: set[str] | None = None) -> dict[str, Any]:
    """Print the headline numbers for a flow and save them to summary.json."""
    rows = read_rows(output_dir / "results.jsonl")
    if only_case_ids:
        rows = [row for row in rows if row["prompt_id"] in only_case_ids]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    error_rows = read_rows(output_dir / "errors.jsonl")
    ok_rows_by_case = group_rows_by_case(ok_rows)

    summary: dict[str, Any] = {
        "cases": len(ok_rows_by_case),
        "rows": len(rows),
        "truncated": len(rows) - len(ok_rows),
        "errors": len(error_rows),
    }
    print(
        f"[{flow_name}] {summary['cases']} cases, {len(ok_rows)} ok rows, "
        f"{summary['truncated']} truncated, {summary['errors']} errors"
    )

    for metric in METRICS_BY_FLOW[flow_name]:
        # Average each case over its reps first, so every case counts equally.
        per_case_means = []
        for case_rows in ok_rows_by_case.values():
            per_case_means.append(share_of_rows(case_rows, metric["id"]))
        mean, margin = mean_with_95_percent_margin(per_case_means)
        summary[metric["id"]] = mean
        print(f"  {metric['label']:<14} {mean:6.1%}  +/- {margin:.1%}")

    if flow_name == "duplicates":
        detection = summarize_duplicate_detection(ok_rows)
        summary.update(detection)
        print(
            f"  recall {detection['recall']:.1%} | specificity {detection['specificity']:.1%} | "
            f"precision {detection['precision']:.1%} | candidate recall {detection['candidate_recall']:.1%}"
        )

    costs = sorted(row["cost_usd"] for row in rows)
    latencies = sorted(row["latency_s"] for row in ok_rows)
    summary["cost_total"] = sum(costs)
    summary["cost_per_row_median"] = middle_value(costs)
    summary["latency_median_s"] = middle_value(latencies)
    print(
        f"  cost ${summary['cost_total']:.4f} total (median ${summary['cost_per_row_median']:.5f}/row)"
        f" | median latency {summary['latency_median_s']:.1f}s"
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
