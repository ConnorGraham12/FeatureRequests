"""Runs the AI pipeline on a new submission and stores what it suggests as proposals.

Nothing here attaches a submission to a feature or sets a category. That only
happens in app/services/review.py, after a human decision.

Flow for a new submission:
    1. Store it and its embedding.
    2. Find similar existing features (embedding search).
    3. If there are any, ask the model whether it's a duplicate of one of them.
       If yes: propose the merge and an updated feature summary, then stop.
    4. Otherwise ask the model for a category, and propose that.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.embeddings import build_text_to_embed
from app.ai.llm import LLMError
from app.ai.vectors import vector_to_bytes
from app.db.models import Feature, Proposal, Submission
from app.domain.enums import ProposalType
from app.observability.correlation import (
    bind_correlation_id,
    generate_correlation_id,
    get_current_correlation_id,
)
from app.observability.decision_log import Actor, EventType
from app.pipeline import stages
from app.services.context import TriageContext


@dataclass
class DuplicateCheckOutcome:
    merge_was_proposed: bool
    last_event_id: str
    # When no merge was proposed: what was checked, shown on the review page.
    details_for_reviewer: dict | None = None


def process_new_submission(
    session: Session, context: TriageContext, title: str, description: str
) -> Submission:
    """Store a new submission, then propose either a merge or a category for review."""
    correlation_id = get_current_correlation_id()
    if correlation_id is None:
        correlation_id = generate_correlation_id()

    with bind_correlation_id(correlation_id):
        stored = store_submission_with_embedding(
            session, context, title, description, correlation_id
        )
        submission = stored.submission

        candidates = find_and_log_duplicate_candidates(
            session, context, submission, stored.embedding, parent_event_id=stored.received_event_id
        )
        last_event_id = candidates.logged_event_id

        details_for_reviewer = None
        if candidates.found:
            outcome = check_for_duplicate(
                session, context, submission, candidates.found, parent_event_id=last_event_id
            )
            if outcome.merge_was_proposed:
                session.commit()
                return submission
            # Not a duplicate, or the check failed: either way it becomes a new feature.
            last_event_id = outcome.last_event_id
            details_for_reviewer = outcome.details_for_reviewer

        propose_category_for_submission(
            session, context, submission, last_event_id, details_for_reviewer
        )
        session.commit()
        return submission


@dataclass
class StoredSubmission:
    submission: Submission
    embedding: np.ndarray
    received_event_id: str


def store_submission_with_embedding(
    session: Session, context: TriageContext, title: str, description: str, correlation_id: str
) -> StoredSubmission:
    """Save the submission with its embedding and log that it arrived."""
    text_to_embed = build_text_to_embed(title, description)
    embedding = context.embedder.embed_texts([text_to_embed])[0]

    submission = Submission(
        correlation_id=correlation_id,
        title=title,
        description=description,
        embedding=vector_to_bytes(embedding),
        embedding_model=context.embedder.model_name,
    )
    session.add(submission)
    session.flush()  # assigns submission.id

    received_event = context.decision_log.write_event(
        EventType.SUBMISSION_RECEIVED,
        Actor.for_system(),
        submission_id=submission.id,
        payload={"title": title, "embedding_model": context.embedder.model_name},
    )
    return StoredSubmission(
        submission=submission, embedding=embedding, received_event_id=received_event.event_id
    )


@dataclass
class LoggedCandidates:
    found: list[stages.DuplicateCandidate]
    logged_event_id: str


def find_and_log_duplicate_candidates(
    session: Session,
    context: TriageContext,
    submission: Submission,
    embedding: np.ndarray,
    *,
    parent_event_id: str,
) -> LoggedCandidates:
    settings = context.settings
    candidates = stages.find_duplicate_candidates(
        session,
        embedding,
        max_candidates=settings.duplicate_top_k,
        min_similarity=settings.duplicate_similarity_threshold,
        submission_id_to_skip=submission.id,
    )

    candidates_for_log = []
    for candidate in candidates:
        candidates_for_log.append(
            {
                "feature_id": candidate.feature_id,
                "similarity": round(candidate.similarity, 4),
                "matched_submission_id": candidate.most_similar_submission_id,
            }
        )
    event = context.decision_log.write_event(
        EventType.DUPLICATE_CANDIDATES,
        Actor.for_system(),
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        payload={
            "threshold": settings.duplicate_similarity_threshold,
            "k": settings.duplicate_top_k,
            "candidates": candidates_for_log,
        },
    )
    return LoggedCandidates(found=candidates, logged_event_id=event.event_id)


def describe_candidates_for_reviewer(
    candidates: list[stages.DuplicateCandidate], features_by_id: dict[int, Feature]
) -> list[dict]:
    descriptions = []
    for candidate in candidates:
        descriptions.append(
            {
                "feature_id": candidate.feature_id,
                "title": features_by_id[candidate.feature_id].title,
                "similarity": round(candidate.similarity, 4),
            }
        )
    return descriptions


def check_for_duplicate(
    session: Session,
    context: TriageContext,
    submission: Submission,
    candidates: list[stages.DuplicateCandidate],
    *,
    parent_event_id: str,
) -> DuplicateCheckOutcome:
    """Ask the model whether the submission duplicates a candidate, and log its answer.

    If it does, a merge proposal and a summary-update proposal are created.
    """
    candidate_feature_ids = [candidate.feature_id for candidate in candidates]
    features_by_id = {}
    for feature in session.scalars(select(Feature).where(Feature.id.in_(candidate_feature_ids))):
        features_by_id[feature.id] = feature

    candidates_with_features = []
    for candidate in candidates:
        candidates_with_features.append((candidate, features_by_id[candidate.feature_id]))
    candidates_for_reviewer = describe_candidates_for_reviewer(candidates, features_by_id)

    prompt = stages.load_current_prompt("duplicate_judge")
    try:
        result, user_message = stages.ask_model_if_duplicate(
            context.language_model, submission.title, submission.description, candidates_with_features
        )
    except LLMError as error:
        error_event = context.write_pipeline_error_event(
            "duplicate.judgment", error, parent_event_id=parent_event_id, submission_id=submission.id
        )
        return DuplicateCheckOutcome(
            merge_was_proposed=False,
            last_event_id=error_event.event_id,
            details_for_reviewer={
                "error": "The duplicate check failed, so possible duplicates were not ruled out.",
                "candidates": candidates_for_reviewer,
            },
        )

    judgment = result.output
    rejected_candidates = []
    for rejected in judgment.rejected_candidates:
        rejected_candidates.append(rejected.model_dump())

    judgment_event = context.write_ai_decision_event(
        EventType.DUPLICATE_JUDGMENT,
        result,
        prompt,
        user_message,
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        payload={"duplicate_of": judgment.duplicate_of_feature_id, "rejected": rejected_candidates},
        confidence=judgment.confidence,
        reasoning=judgment.reasoning,
    )

    if judgment.duplicate_of_feature_id is None:
        return DuplicateCheckOutcome(
            merge_was_proposed=False,
            last_event_id=judgment_event.event_id,
            details_for_reviewer={
                "candidates": candidates_for_reviewer,
                "rejected": rejected_candidates,
                "reasoning": judgment.reasoning,
            },
        )

    target_feature = features_by_id[judgment.duplicate_of_feature_id]
    merge_proposal = Proposal(
        correlation_id=submission.correlation_id,
        type=ProposalType.DUPLICATE,
        submission_id=submission.id,
        feature_id=target_feature.id,
        payload={
            "duplicate_of": target_feature.id,
            "candidates": candidates_for_reviewer,
            "rejected": rejected_candidates,
        },
        confidence=judgment.confidence,
        reasoning=judgment.reasoning,
        model=result.model,
        prompt_version=prompt.versioned_name,
        ai_event_id=judgment_event.event_id,
    )
    session.add(merge_proposal)

    propose_summary_update(
        session, context, submission, target_feature, parent_event_id=judgment_event.event_id
    )
    return DuplicateCheckOutcome(merge_was_proposed=True, last_event_id=judgment_event.event_id)


def propose_summary_update(
    session: Session,
    context: TriageContext,
    submission: Submission,
    feature: Feature,
    *,
    parent_event_id: str,
) -> None:
    """Ask the model how the feature summary should change if this merge is approved."""
    prompt = stages.load_current_prompt("summary")
    try:
        result, user_message = stages.ask_model_for_merged_summary(
            context.language_model, feature, submission.title, submission.description
        )
    except LLMError as error:
        # The merge can still be reviewed; the summary just stays as it is.
        context.write_pipeline_error_event(
            "summary.proposed", error, parent_event_id=parent_event_id, submission_id=submission.id
        )
        return

    update = result.output
    payload = {"old": feature.summary, "new": update.summary, "changed": update.changed}
    summary_event = context.write_ai_decision_event(
        EventType.SUMMARY_PROPOSED,
        result,
        prompt,
        user_message,
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        feature_id=feature.id,
        payload=payload,
        reasoning=update.reasoning,
    )
    summary_proposal = Proposal(
        correlation_id=submission.correlation_id,
        type=ProposalType.SUMMARY,
        submission_id=submission.id,
        feature_id=feature.id,
        payload=payload,
        reasoning=update.reasoning,
        model=result.model,
        prompt_version=prompt.versioned_name,
        ai_event_id=summary_event.event_id,
    )
    session.add(summary_proposal)


def propose_category_for_submission(
    session: Session,
    context: TriageContext,
    submission: Submission,
    parent_event_id: str | None,
    duplicate_check_details: dict | None = None,
) -> Proposal | None:
    """Ask the model for a category for a submission that will become a new feature.

    Returns None if the model call failed; the reviewer then picks a category by hand.
    """
    prompt = stages.load_current_prompt("categorize")
    try:
        result, user_message = stages.ask_model_for_category(
            context.language_model, session, submission.title, submission.description
        )
    except LLMError as error:
        context.write_pipeline_error_event(
            "category.proposed", error, parent_event_id=parent_event_id, submission_id=submission.id
        )
        return None

    assignment = result.output
    logged_payload = {
        "action": assignment.action,
        "category_id": assignment.category_id,
        "new_category_name": assignment.new_category_name,
        "new_category_description": assignment.new_category_description,
        "feature_summary": assignment.feature_summary,
    }
    category_event = context.write_ai_decision_event(
        EventType.CATEGORY_PROPOSED,
        result,
        prompt,
        user_message,
        parent_event_id=parent_event_id,
        submission_id=submission.id,
        payload=logged_payload,
        confidence=assignment.confidence,
        reasoning=assignment.reasoning,
    )

    # The review page also shows what the duplicate check found; the log
    # already has that in its own event, so it is only added here.
    proposal_payload = dict(logged_payload)
    if duplicate_check_details:
        proposal_payload["duplicate_check"] = duplicate_check_details

    if assignment.action == "assign":
        proposal_type = ProposalType.CATEGORY
    else:
        proposal_type = ProposalType.NEW_CATEGORY

    proposal = Proposal(
        correlation_id=submission.correlation_id,
        type=proposal_type,
        submission_id=submission.id,
        payload=proposal_payload,
        confidence=assignment.confidence,
        reasoning=assignment.reasoning,
        model=result.model,
        prompt_version=prompt.versioned_name,
        ai_event_id=category_event.event_id,
    )
    session.add(proposal)
    return proposal
