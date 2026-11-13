from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.ai.embeddings import Embedder
from app.ai.llm import LanguageModel, LLMError, LLMResult
from app.ai.prompts import PromptTemplate
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
    """Everything the services need apart from a database session."""

    settings: Settings
    decision_log: DecisionLogger
    embedder: Embedder
    language_model: LanguageModel

    @property
    def reviewer_name(self) -> str:
        return self.settings.reviewer_name

    def write_ai_decision_event(
        self,
        event_type: EventType,
        result: LLMResult,
        prompt: PromptTemplate,
        user_message: str,
        *,
        parent_event_id: str | None,
        payload: dict[str, Any],
        confidence: float | None = None,
        reasoning: str | None = None,
        submission_id: int | None = None,
        feature_id: int | None = None,
    ) -> DecisionEvent:
        """Log a decision the model made, with its prompt version, tokens and timing."""
        payload_to_log = dict(payload)
        if result.fallback_used:
            payload_to_log["fallback_used"] = True

        debug_details = {
            "system": prompt.text,
            "user": user_message,
            "request_id": result.request_id,
            "raw_response": result.raw_response,
        }
        return self.decision_log.write_event(
            event_type,
            Actor.for_ai_model(result.model),
            parent_event_id=parent_event_id,
            submission_id=submission_id,
            feature_id=feature_id,
            payload=payload_to_log,
            confidence=confidence,
            reasoning=reasoning,
            prompt_version=prompt.versioned_name,
            prompt_hash=prompt.content_hash,
            latency_ms=result.latency_ms,
            tokens=TokenUsage(input=result.input_tokens, output=result.output_tokens),
            debug=debug_details,
        )

    def write_pipeline_error_event(
        self,
        failed_stage: str,
        error: LLMError,
        *,
        parent_event_id: str | None,
        submission_id: int | None = None,
    ) -> DecisionEvent:
        """Log that an AI step failed, so the failure appears in the submission's trace."""
        return self.decision_log.write_event(
            EventType.PIPELINE_ERROR,
            Actor.for_system(),
            parent_event_id=parent_event_id,
            submission_id=submission_id,
            payload={
                "stage": failed_stage,
                "error": type(error).__name__,
                "message": str(error),
            },
        )


def record_decision_on_proposal(
    proposal: Proposal,
    new_status: ProposalStatus,
    reviewer_name: str,
    decision_details: dict[str, Any] | None,
) -> None:
    """Mark a proposal as decided: who decided, when, the outcome and any edits."""
    proposal.status = new_status
    proposal.decided_by = reviewer_name
    proposal.decided_at = datetime.now(UTC)
    if decision_details:
        proposal.decision_payload = decision_details
    else:
        proposal.decision_payload = None
