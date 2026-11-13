"""The harness-approval gate.

The runner refuses to run until the user has approved the current version of
the eval files (runner, grader, dataset, judge prompt). If any of them change,
the runner stops again until the user re-approves. This stops anyone, including
an AI assistant, from quietly changing the grader or the answer key between runs.

Approval is given by the user running:
    python -m evals.orbit.run_eval --approve-harness
"""

import hashlib
import json
import sys
from pathlib import Path

from evals.orbit.metrics import METRICS_BY_FLOW
from evals.orbit.pricing import PRICES_PER_MILLION_TOKENS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_OUTPUT_ROOT = PROJECT_ROOT / ".claude" / "hillclimb"

HARNESS_FILES = [
    "evals/orbit/run_eval.py",
    "evals/orbit/flows.py",
    "evals/orbit/metrics.py",
    "evals/orbit/results.py",
    "evals/orbit/pricing.py",
    "evals/orbit/recording_model.py",
    "evals/orbit/harness_approval.py",
    "evals/orbit/dataset.py",
    "evals/orbit/taxonomy.json",
    "evals/orbit/corpus.jsonl",
    "evals/orbit/probes.jsonl",
    "evals/orbit/judge_summary_v1.txt",
    "requirements.txt",
]


def calculate_harness_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in HARNESS_FILES:
        digest.update(relative_path.encode())
        digest.update((PROJECT_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def state_file_path(flow_name: str) -> Path:
    return EVAL_OUTPUT_ROOT / flow_name / "_state.json"


def load_or_create_flow_state(flow_name: str, reps: int) -> dict:
    """Read the flow's _state.json, creating it with the report settings if missing."""
    path = state_file_path(flow_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    state = {
        "flow": flow_name,
        "reps": reps,
        "metrics": METRICS_BY_FLOW[flow_name],
        "perf_fields": [
            {"id": "cost_usd", "label": "Cost", "unit": "$"},
            {"id": "latency_s", "label": "Latency", "unit": "s"},
        ],
        "prices": PRICES_PER_MILLION_TOKENS,
        "harness_paths": HARNESS_FILES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def record_harness_approval(flow_name: str, state: dict) -> None:
    fingerprint = calculate_harness_fingerprint()
    state["harness_sha"] = fingerprint
    state["harness_paths"] = HARNESS_FILES
    state_file_path(flow_name).write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[{flow_name}] harness approved ({fingerprint[:12]})")


def stop_unless_harness_is_approved(flow_name: str, state: dict) -> None:
    if state.get("harness_sha") == calculate_harness_fingerprint():
        return
    print(
        f"[{flow_name}] Eval files changed since they were last approved (or were never approved):\n  "
        + "\n  ".join(HARNESS_FILES)
        + "\nReview them, then re-run with --approve-harness.",
        file=sys.stderr,
    )
    sys.exit(2)
