"""Runs the AI pipeline on a new submission and turns its output into proposals.

Nothing here changes a submission's feature or category. That only happens in
app/services/review.py, after a human decision.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.embeddings import build_embedding_text
from app.ai.llm import LLMError
from app.ai.vectors import to_blob
from app.db.models import Feature, Proposal, Submission
from app.domain.enums import ProposalType
from app.observability.correlation import correlation_scope, get_correlation_id, new_correlation_id
from app.observability.decision_log import Actor, EventType
from app.pipeline import stages
from app.services.context import TriageContext


def submit(session: Session, ctx: TriageContext, title: str, description: str) -> Submission:
    """Store a submission, then propose a merge or a category for review."""
    correlation_id = get_correlation_id() or new_correlation_id()
    with correlation_scope(correlation_id):
        vector = ctx.embedder.embed([build_embedding_text(title, description)])[0]
        submission = Submission(
            correlation_id=correlation_id,
            title=title,
            description=description,
            embedding=to_blob(vector),
            embedding_model=ctx.embedder.model_name,
        )
        session.add(submission)
        session.flush()

        received = ctx.log.record(
            EventType.SUBMISSION_RECEIVED,
            Actor.system(),
            submission_id=submission.id,
            payload={"title": title, "embedding_model": ctx.embedder.model_name},
        )

        settings = ctx.settings
        candidates = stages.find_candidates(
            session,
            vector,
            k=settings.duplicate_top_k,
            threshold=settings.duplicate_similarity_threshold,
            exclude_submission_id=submission.id,
        )
        candidates_event = ctx.log.record(
            EventType.DUPLICATE_CANDIDATES,
            Actor.system(),
            parent_event_id=received.event_id,
            submission_id=submission.id,
            payload={
                "threshold": settings.duplicate_similarity_threshold,
                "k": settings.duplicate_top_k,
                "candidates": [
                    {
                        "feature_id": c.feature_id,
                        "similarity": round(c.similarity, 4),
                        "matched_submission_id": c.submission_id,
                    }
                    for c in candidates
                ],
            },
        )

        parent_event_id = candidates_event.event_id
        duplicate_check = None
        if candidates:
            merged, parent_event_id, duplicate_check = _check_duplicates(
                session, ctx, submission, candidates, candidates_event.event_id
            )
            if merged:
                session.commit()
                return submission
            # Not a duplicate, or the check failed (logged as pipeline.error):
            # either way it is categorized as a new feature.

        run_categorization(session, ctx, submission, parent_event_id, duplicate_check)
        session.commit()
        return submission


def _check_duplicates(
    session: Session,
    ctx: TriageContext,
    submission: Submission,
    candidates: list[stages.Candidate],
    parent_event_id: str,
) -> tuple[bool, str, dict | None]:
    """Return (merge proposed?, last event id, summary of a negative check)."""
    features = {
        f.id: f
        for f in session.scalars(
            select(Feature).where(Feature.id.in_([c.feature_id for c in candidates]))
        )
    }
    pairs = [(c, features[c.feature_id]) for c in candidates]
    candidate_payload = [
        {"feature_id": c.feature_id, "title": f.title, "similarity": round(c.similarity, 4)}
        for c, f in pairs
    ]
    prompt = stages.prompt_for("duplicate_judge")

    try:
        result, user = stages.judge_duplicate(ctx.llm, submission.title, submission.description, pairs)
    except LLMError as error:
        error_event = ctx.log_error(
            "duplicate.judgment", error, parent_event_id=parent_event_id, submission_id=submission.id
        )
        return False, error_event.event_id, {
            "error": "The duplicate check failed, so possible duplicates were not ruled out.",
            "candidates": candidate_payload,
        }

    judgment = result.output
    rejected = [r.model_dump() for r in judgment.rejected_candidates]
    judged = ctx.log_ai(
        EventType.DUPLICATE_JUDGMENT,
        result,
        prompt,
        user,
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        payload={"duplicate_of": judgment.duplicate_of_feature_id, "rejected": rejected},
        confidence=judgment.confidence,
        reasoning=judgment.reasoning,
    )

    if judgment.duplicate_of_feature_id is None:
        summary = {
            "candidates": candidate_payload,
            "rejected": rejected,
            "reasoning": judgment.reasoning,
        }
        return False, judged.event_id, summary

    target = features[judgment.duplicate_of_feature_id]
    session.add(
        Proposal(
            correlation_id=submission.correlation_id,
            type=ProposalType.DUPLICATE,
            submission_id=submission.id,
            feature_id=target.id,
            payload={
                "duplicate_of": target.id,
                "candidates": candidate_payload,
                "rejected": rejected,
            },
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            model=result.model,
            prompt_version=prompt.id,
            ai_event_id=judged.event_id,
        )
    )
    _propose_summary(session, ctx, submission, target, judged.event_id)
    return True, judged.event_id, None


def _propose_summary(
    session: Session,
    ctx: TriageContext,
    submission: Submission,
    feature: Feature,
    parent_event_id: str,
) -> None:
    prompt = stages.prompt_for("summary")
    try:
        result, user = stages.propose_summary(ctx.llm, feature, submission.title, submission.description)
    except LLMError as error:
        # The merge can still be reviewed; the summary just stays as it is.
        ctx.log_error("summary.proposed", error, parent_event_id=parent_event_id, submission_id=submission.id)
        return

    update = result.output
    payload = {"old": feature.summary, "new": update.summary, "changed": update.changed}
    event = ctx.log_ai(
        EventType.SUMMARY_PROPOSED,
        result,
        prompt,
        user,
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        feature_id=feature.id,
        payload=payload,
        reasoning=update.reasoning,
    )
    session.add(
        Proposal(
            correlation_id=submission.correlation_id,
            type=ProposalType.SUMMARY,
            submission_id=submission.id,
            feature_id=feature.id,
            payload=payload,
            reasoning=update.reasoning,
            model=result.model,
            prompt_version=prompt.id,
            ai_event_id=event.event_id,
        )
    )


def run_categorization(
    session: Session,
    ctx: TriageContext,
    submission: Submission,
    parent_event_id: str | None,
    duplicate_check: dict | None = None,
) -> Proposal | None:
    """Propose a category for a submission that will become a new feature.

    Returns None if the LLM call failed; the reviewer then picks by hand.
    """
    prompt = stages.prompt_for("categorize")
    try:
        result, user = stages.categorize(ctx.llm, session, submission.title, submission.description)
    except LLMError as error:
        ctx.log_error("category.proposed", error, parent_event_id=parent_event_id, submission_id=submission.id)
        return None

    assignment = result.output
    payload = {
        "action": assignment.action,
        "category_id": assignment.category_id,
        "new_category_name": assignment.new_category_name,
        "new_category_description": assignment.new_category_description,
        "feature_summary": assignment.feature_summary,
    }
    event = ctx.log_ai(
        EventType.CATEGORY_PROPOSED,
        result,
        prompt,
        user,
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        payload=payload,
        confidence=assignment.confidence,
        reasoning=assignment.reasoning,
    )
    if duplicate_check:
        payload["duplicate_check"] = duplicate_check

    proposal = Proposal(
        correlation_id=submission.correlation_id,
        type=ProposalType.CATEGORY if assignment.action == "assign" else ProposalType.NEW_CATEGORY,
        submission_id=submission.id,
        payload=payload,
        confidence=assignment.confidence,
        reasoning=assignment.reasoning,
        model=result.model,
        prompt_version=prompt.id,
        ai_event_id=event.event_id,
    )
    session.add(proposal)
    return proposal
