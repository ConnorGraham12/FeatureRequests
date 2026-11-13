import json

import pytest

from app.observability.correlation import correlation_scope, new_correlation_id, run_scope
from app.observability.decision_log import (
    Actor,
    DecisionLogger,
    EventType,
    MissingCorrelationIdError,
    TokenUsage,
)


def test_events_share_correlation_id_and_chain_by_parent(tmp_path):
    log = DecisionLogger(tmp_path / "decisions.jsonl")
    cid = new_correlation_id()

    with correlation_scope(cid):
        received = log.record(EventType.SUBMISSION_RECEIVED, Actor.system(), submission_id=1)
        judged = log.record(
            EventType.DUPLICATE_JUDGMENT,
            Actor.ai("claude-opus-5-5"),
            parent_event_id=received.event_id,
            submission_id=1,
            confidence=0.81,
            tokens=TokenUsage(input=612, output=88),
        )

    events = log.read_correlation(cid)
    assert [e.event_type for e in events] == [
        EventType.SUBMISSION_RECEIVED,
        EventType.DUPLICATE_JUDGMENT,
    ]
    assert events[1].parent_event_id == received.event_id
    assert events[1].event_id == judged.event_id


def test_one_json_line_per_event_with_compact_fields(tmp_path):
    path = tmp_path / "decisions.jsonl"
    log = DecisionLogger(path)
    with correlation_scope(new_correlation_id()):
        log.record(EventType.SUBMISSION_RECEIVED, Actor.system())
        log.record(
            EventType.CATEGORY_PROPOSED,
            Actor.ai("claude-opus-5-5"),
            tokens=TokenUsage(input=10, output=2),
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert "confidence" not in first  # None fields are omitted
    assert second["tokens"] == {"in": 10, "out": 2}


def test_human_decision_days_later_rejoins_correlation(tmp_path):
    log = DecisionLogger(tmp_path / "decisions.jsonl")
    cid = new_correlation_id()
    with correlation_scope(cid):
        log.record(EventType.SUBMISSION_RECEIVED, Actor.system())

    # Later request, different context: pass the stored ID explicitly.
    log.record(EventType.MERGE_DECISION, Actor.human("pm"), correlation_id=cid)

    assert len(log.read_correlation(cid)) == 2


def test_recording_without_correlation_id_raises(tmp_path):
    log = DecisionLogger(tmp_path / "decisions.jsonl")
    with pytest.raises(MissingCorrelationIdError):
        log.record(EventType.SUBMISSION_RECEIVED, Actor.system())


def test_debug_only_written_when_verbose(tmp_path):
    quiet = DecisionLogger(tmp_path / "quiet.jsonl")
    loud = DecisionLogger(tmp_path / "loud.jsonl", verbose=True)
    with correlation_scope(new_correlation_id()):
        assert quiet.record(EventType.PIPELINE_ERROR, Actor.system(), debug={"x": 1}).debug is None
        assert loud.record(EventType.PIPELINE_ERROR, Actor.system(), debug={"x": 1}).debug == {"x": 1}


def test_run_id_recorded_inside_eval_run(tmp_path):
    log = DecisionLogger(tmp_path / "decisions.jsonl")
    with run_scope("eval-001"), correlation_scope(new_correlation_id()):
        event = log.record(EventType.SUBMISSION_RECEIVED, Actor.system())
    assert event.run_id == "eval-001"
