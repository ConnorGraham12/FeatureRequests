from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.ai.embeddings import Embedder
from app.ai.llm import LLM, LLMError, LLMResult
from app.ai.prompts import Prompt
from app.config import Settings
from app.db.models import Proposal
from app.domain.enums import ProposalStatus
from app.observability.decision_log import (
    Actor,
    DecisionEvent,
    DecisionLogger,
    EventType,
    TokenUsage,
)


@dataclass
class TriageContext:
    """Everything a service needs besides the DB session."""

    settings: Settings
    log: DecisionLogger
    embedder: Embedder
    llm: LLM

    @property
    def reviewer(self) -> str:
        return self.settings.reviewer_name

    def log_ai(
        self,
        event_type: EventType,
        result: LLMResult,
        prompt: Prompt,
        user_message: str,
        *,
        parent_event_id: str | None,
        payload: dict[str, Any],
        confidence: float | None = None,
        reasoning: str | None = None,
        submission_id: int | None = None,
        feature_id: int | None = None,
    ) -> DecisionEvent:
        if result.fallback_used:
            payload = {**payload, "fallback_used": True}
        return self.log.record(
            event_type,
            Actor.ai(result.model),
            parent_event_id=parent_event_id,
            submission_id=submission_id,
            feature_id=feature_id,
            payload=payload,
            confidence=confidence,
            reasoning=reasoning,
            prompt_version=prompt.id,
            prompt_hash=prompt.hash,
            latency_ms=result.latency_ms,
            tokens=TokenUsage(input=result.input_tokens, output=result.output_tokens),
            debug={
                "system": prompt.text,
                "user": user_message,
                "request_id": result.request_id,
                "raw_response": result.raw_response,
            },
        )

    def log_error(
        self,
        stage: str,
        error: LLMError,
        *,
        parent_event_id: str | None,
        submission_id: int | None = None,
    ) -> DecisionEvent:
        return self.log.record(
            EventType.PIPELINE_ERROR,
            Actor.system(),
            parent_event_id=parent_event_id,
            submission_id=submission_id,
            payload={"stage": stage, "error": type(error).__name__, "message": str(error)},
        )


def mark_decided(
    proposal: Proposal,
    status: ProposalStatus,
    reviewer: str,
    decision_payload: dict[str, Any] | None,
) -> None:
    proposal.status = status
    proposal.decided_by = reviewer
    proposal.decided_at = datetime.now(UTC)
    proposal.decision_payload = decision_payload or None
