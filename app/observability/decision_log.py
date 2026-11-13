"""Append-only JSONL log of every runtime decision, AI or human.

One line per decision block. Every line carries a correlation ID, so the full
history of a submission can be recovered with:

    grep <correlation_id> logs/decisions.jsonl

The database holds current state; this log records how that state was reached.
"""

import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.observability.correlation import get_correlation_id, get_run_id


class EventType(StrEnum):
    SUBMISSION_RECEIVED = "submission.received"
    DUPLICATE_CANDIDATES = "duplicate.candidates"
    DUPLICATE_JUDGMENT = "duplicate.judgment"
    SUMMARY_PROPOSED = "summary.proposed"
    CATEGORY_PROPOSED = "category.proposed"
    MERGE_DECISION = "merge.decision"
    CATEGORY_DECISION = "category.decision"
    CATEGORY_CREATED = "category.created"
    TAXONOMY_PROPOSED = "taxonomy.proposed"
    TAXONOMY_DECISION = "taxonomy.decision"
    PIPELINE_ERROR = "pipeline.error"


class Actor(BaseModel):
    type: Literal["system", "ai", "human"]
    model: str | None = None
    reviewer: str | None = None

    @classmethod
    def system(cls) -> "Actor":
        return cls(type="system")

    @classmethod
    def ai(cls, model: str) -> "Actor":
        return cls(type="ai", model=model)

    @classmethod
    def human(cls, reviewer: str) -> "Actor":
        return cls(type="human", reviewer=reviewer)


class TokenUsage(BaseModel):
    # Written as {"in": .., "out": ..}; accepts either name when constructed.
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
    """Raised when a decision is recorded outside any correlation scope."""


class DecisionLogger:
    def __init__(self, path: Path, verbose: bool = False) -> None:
        self.path = Path(path)
        self.verbose = verbose
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
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
        """Append one decision block and return it.

        `correlation_id` defaults to the one bound in the current context. A
        decision with no correlation ID is a bug, so it raises instead of
        writing an orphaned line. `debug` (full prompts, raw responses) is only
        written when the logger is verbose.
        """
        correlation_id = correlation_id or get_correlation_id()
        if correlation_id is None:
            raise MissingCorrelationIdError(f"No correlation ID bound for {event_type}")

        event = DecisionEvent(
            timestamp=datetime.now(UTC),
            correlation_id=correlation_id,
            event_id=str(uuid.uuid4()),
            parent_event_id=parent_event_id,
            run_id=get_run_id(),
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
            debug=debug if self.verbose else None,
        )
        line = event.model_dump_json(exclude_none=True, by_alias=True)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return event

    def read_correlation(self, correlation_id: str) -> list[DecisionEvent]:
        """Return all events for one correlation ID, in write order."""
        if not self.path.exists():
            return []
        events = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                event = DecisionEvent.model_validate_json(line)
                if event.correlation_id == correlation_id:
                    events.append(event)
        return events
