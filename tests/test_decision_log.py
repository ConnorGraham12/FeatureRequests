import json

import pytest

from app.observability.correlation import (
    bind_correlation_id,
    bind_eval_run_id,
    generate_correlation_id,
)
from app.observability.decision_log import (
    Actor,
    DecisionLogger,
    EventType,
    MissingCorrelationIdError,
    TokenUsage,
)


def test_events_share_correlation_id_and_link_to_their_parent(tmp_path):
    decision_log = DecisionLogger(tmp_path / "decisions.jsonl")
    correlation_id = generate_correlation_id()

    with bind_correlation_id(correlation_id):
        received = decision_log.write_event(
            EventType.SUBMISSION_RECEIVED, Actor.for_system(), submission_id=1
        )
        judged = decision_log.write_event(
            EventType.DUPLICATE_JUDGMENT,
            Actor.for_ai_model("claude-haiku-5-5"),
            parent_event_id=received.event_id,
            submission_id=1,
            confidence=0.81,
            tokens=TokenUsage(input=612, output=88),
        )

    events = decision_log.read_events_for_correlation_id(correlation_id)
    assert [event.event_type for event in events] == [
        EventType.SUBMISSION_RECEIVED,
        EventType.DUPLICATE_JUDGMENT,
    ]
    assert events[1].parent_event_id == received.event_id
    assert events[1].event_id == judged.event_id


def test_each_event_is_one_json_line_without_empty_fields(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    decision_log = DecisionLogger(log_path)
    with bind_correlation_id(generate_correlation_id()):
        decision_log.write_event(EventType.SUBMISSION_RECEIVED, Actor.for_system())
        decision_log.write_event(
            EventType.CATEGORY_PROPOSED,
            Actor.for_ai_model("claude-haiku-5-5"),
            tokens=TokenUsage(input=10, output=2),
        )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first_event = json.loads(lines[0])
    second_event = json.loads(lines[1])
    assert "confidence" not in first_event
    assert second_event["tokens"] == {"in": 10, "out": 2}


def test_a_later_human_decision_joins_the_same_trace(tmp_path):
    decision_log = DecisionLogger(tmp_path / "decisions.jsonl")
    correlation_id = generate_correlation_id()
    with bind_correlation_id(correlation_id):
        decision_log.write_event(EventType.SUBMISSION_RECEIVED, Actor.for_system())

    # A later request in a different context passes the stored ID explicitly.
    decision_log.write_event(
        EventType.MERGE_DECISION, Actor.for_human_reviewer("pm"), correlation_id=correlation_id
    )

    assert len(decision_log.read_events_for_correlation_id(correlation_id)) == 2


def test_writing_without_a_correlation_id_raises(tmp_path):
    decision_log = DecisionLogger(tmp_path / "decisions.jsonl")
    with pytest.raises(MissingCorrelationIdError):
        decision_log.write_event(EventType.SUBMISSION_RECEIVED, Actor.for_system())


def test_debug_details_are_only_written_when_enabled(tmp_path):
    without_debug = DecisionLogger(tmp_path / "plain.jsonl")
    with_debug = DecisionLogger(tmp_path / "debug.jsonl", include_debug_details=True)
    with bind_correlation_id(generate_correlation_id()):
        plain_event = without_debug.write_event(
            EventType.PIPELINE_ERROR, Actor.for_system(), debug={"x": 1}
        )
        debug_event = with_debug.write_event(
            EventType.PIPELINE_ERROR, Actor.for_system(), debug={"x": 1}
        )
    assert plain_event.debug is None
    assert debug_event.debug == {"x": 1}


def test_eval_run_id_is_recorded_during_an_eval_run(tmp_path):
    decision_log = DecisionLogger(tmp_path / "decisions.jsonl")
    with bind_eval_run_id("eval-001"):
        with bind_correlation_id(generate_correlation_id()):
            event = decision_log.write_event(EventType.SUBMISSION_RECEIVED, Actor.for_system())
    assert event.run_id == "eval-001"
