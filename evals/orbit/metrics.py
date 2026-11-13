"""The metrics each eval flow records. The first "binary" metric is the report's headline."""

METRICS_BY_FLOW = {
    "duplicates": [
        {"id": "correct", "kind": "binary", "label": "Correct"},
        {"id": "false_merge", "kind": "binary", "label": "Wrong merge", "better": "lower"},
        {"id": "missed_dup", "kind": "binary", "label": "Missed dup", "better": "lower"},
        {"id": "refused", "kind": "binary", "label": "Refused", "better": "lower"},
    ],
    "categories": [
        {"id": "correct", "kind": "binary", "label": "Correct"},
        {"id": "exact", "kind": "binary", "label": "Exact gold"},
        {"id": "invented", "kind": "binary", "label": "New category", "better": "lower"},
        {"id": "refused", "kind": "binary", "label": "Refused", "better": "lower"},
    ],
    "summaries": [
        {"id": "all_pass", "kind": "binary", "label": "All checks"},
        {"id": "keeps_existing", "kind": "judge", "label": "Keeps points"},
        {"id": "adds_new_info", "kind": "judge", "label": "Adds new info"},
        {"id": "no_fabrication", "kind": "judge", "label": "No invention"},
        {"id": "flag_consistent", "kind": "judge", "label": "Flag right"},
        {"id": "format_ok", "kind": "binary", "label": "Format"},
        {"id": "refused", "kind": "binary", "label": "Refused", "better": "lower"},
    ],
}


def metric_ids_for_flow(flow_name: str) -> list[str]:
    return [metric["id"] for metric in METRICS_BY_FLOW[flow_name]]


def all_zero_grade(flow_name: str) -> dict[str, float]:
    """A grade with every metric at 0, used for refused or truncated answers."""
    grade = {}
    for metric_id in metric_ids_for_flow(flow_name):
        grade[metric_id] = 0.0
    return grade
