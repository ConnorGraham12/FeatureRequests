"""Append-only JSONL log of every runtime decision, made by the AI or a human.

Each decision is one line in the file. Every line carries a correlation ID, so
the full history of a submission can be found with:

    grep <correlation_id> logs/decisions.jsonl

The database holds the current state; this log records how it was reached.
"""

import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.observability.correlation import get_current_correlation_id, get_current_eval_run_id


class EventType(StrEnum):
    SUBMISSION_RECEIVED = "submission.received"
    DUPLICATE_CANDIDATES = "duplicate.candidates"
    DUPLICATE_JUDGMENT = "duplicate.judgment"
    SUMMARY_PROPOSED = "summary.proposed"
    CATEGORY_PROPOSED = "category.proposed"
    MERGE_DECISION = "merge.decision"
    CATEGORY_DECISION = "category.decision"
    CATEGORY_CREATED = "category.created"
    CATEGORY_REASSIGNED = "category.reassigned"
    FEATURE_EDITED = "feature.edited"
    FEATURE_ARCHIVED = "feature.archived"
    FEATURE_RESTORED = "feature.restored"
    SUBMISSION_UNMERGED = "submission.unmerged"
    SUBMISSION_MOVED = "submission.moved"
    SUBMISSION_ARCHIVED = "submission.archived"
    SUBMISSION_RESTORED = "submission.restored"
    TAXONOMY_PROPOSED = "taxonomy.proposed"
    TAXONOMY_DECISION = "taxonomy.decision"
    PIPELINE_ERROR = "pipeline.error"


class Actor(BaseModel):
    """Who made a decision: the system, an AI model, or a human reviewer."""

    type: Literal["system", "ai", "human"]
    model: str | None = None
    reviewer: str | None = None

    @classmethod
    def for_system(cls) -> "Actor":
        return cls(type="system")

    @classmethod
    def for_ai_model(cls, model_name: str) -> "Actor":
        return cls(type="ai", model=model_name)

    @classmethod
    def for_human_reviewer(cls, reviewer_name: str) -> "Actor":
        return cls(type="human", reviewer=reviewer_name)


class TokenUsage(BaseModel):
    """Token counts. Written to the log as {"in": ..., "out": ...}."""

    model_config = ConfigDict(populate_by_name=True)

    input: int = Field(alias="in")
    output: int = Field(alias="out")


class DecisionEvent(BaseModel):
    timestamp: datetime
    correlation_id: str
    event_id: str
    parent_event_id: str | None = None
    run_id: str | None = None
    event_type: EventType
    actor: Actor
    submission_id: int | None = None
    feature_id: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    reasoning: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    latency_ms: int | None = None
    tokens: TokenUsage | None = None
    debug: dict[str, Any] | None = None


class MissingCorrelationIdError(RuntimeError):
    """Raised when a decision is written while no correlation ID is bound."""


class DecisionLogger:
    def __init__(self, log_file_path: Path, include_debug_details: bool = False) -> None:
        self.log_file_path = Path(log_file_path)
        self.include_debug_details = include_debug_details
        self._write_lock = threading.Lock()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def write_event(
        self,
        event_type: EventType,
        actor: Actor,
        *,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        submission_id: int | None = None,
        feature_id: int | None = None,
        payload: dict[str, Any] | None = None,
        confidence: float | None = None,
        reasoning: str | None = None,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
        latency_ms: int | None = None,
        tokens: TokenUsage | None = None,
        debug: dict[str, Any] | None = None,
    ) -> DecisionEvent:
        """Append one decision to the log file and return it.

        If no correlation ID is passed, the currently bound one is used. A
        decision with no correlation ID at all is a bug, so this raises rather
        than writing a line that can't be traced. `debug` (full prompts and raw
        responses) is only written when debug details are enabled.
        """
        if correlation_id is None:
            correlation_id = get_current_correlation_id()
        if correlation_id is None:
            raise MissingCorrelationIdError(f"No correlation ID is bound for {event_type}")

        if self.include_debug_details:
            debug_to_write = debug
        else:
            debug_to_write = None

        event = DecisionEvent(
            timestamp=datetime.now(UTC),
            correlation_id=correlation_id,
            event_id=str(uuid.uuid4()),
            parent_event_id=parent_event_id,
            run_id=get_current_eval_run_id(),
            event_type=event_type,
            actor=actor,
            submission_id=submission_id,
            feature_id=feature_id,
            payload=payload or {},
            confidence=confidence,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            latency_ms=latency_ms,
            tokens=tokens,
            debug=debug_to_write,
        )

        # Empty fields are left out to keep each line short and readable.
        line = event.model_dump_json(exclude_none=True, by_alias=True)
        with self._write_lock:
            with self.log_file_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        return event

    def read_all_events(self) -> list[DecisionEvent]:
        """Read every event in the log, in the order they were written.

        This scans the whole file, which is fine at prototype log sizes.
        """
        if not self.log_file_path.exists():
            return []

        events = []
        with self.log_file_path.open(encoding="utf-8") as log_file:
            for line in log_file:
                if line.strip():
                    events.append(DecisionEvent.model_validate_json(line))
        return events

    def read_events_for_correlation_id(self, correlation_id: str) -> list[DecisionEvent]:
        matching_events = []
        for event in self.read_all_events():
            if event.correlation_id == correlation_id:
                matching_events.append(event)
        return matching_events

    def read_events_for_feature(self, feature_id: int, event_type: EventType) -> list[DecisionEvent]:
        matching_events = []
        for event in self.read_all_events():
            if event.feature_id == feature_id and event.event_type is event_type:
                matching_events.append(event)
        return matching_events
