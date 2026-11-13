"""Human decisions on proposals. The only place submissions join features.

Each decision runs in the submission's own correlation scope, so it lands in
the same trace as the AI events it decides on, however much later it happens.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Category, Feature, Proposal, Submission
from app.domain.enums import (
    CategoryStatus,
    Decision,
    ProposalStatus,
    ProposalType,
    SubmissionStatus,
)
from app.domain.proposals import InvalidDecisionError, validate_decision
from app.observability.correlation import correlation_scope
from app.observability.decision_log import Actor, EventType
from app.services.context import TriageContext, mark_decided
from app.services.triage import run_categorization


class NotFoundError(LookupError):
    pass


def _get_proposal(session: Session, proposal_id: int, types: set[ProposalType]) -> Proposal:
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.type not in types:
        raise NotFoundError(f"No {'/'.join(types)} proposal with id {proposal_id}")
    return proposal


def _ensure_pending_submission(submission: Submission | None) -> Submission:
    if submission is None:
        raise NotFoundError("Proposal has no submission")
    if submission.status is not SubmissionStatus.PENDING:
        raise InvalidDecisionError(f"Submission is already {submission.status}")
    return submission


def _active_category(session: Session, category_id: int | None) -> Category:
    category = session.get(Category, category_id) if category_id is not None else None
    if category is None or category.status is not CategoryStatus.ACTIVE:
        raise InvalidDecisionError("Choose an active category")
    return category


def _create_category(session: Session, ctx: TriageContext, name: str | None, description: str | None, parent_event_id: str | None, source_proposal_id: int) -> Category:
    name = (name or "").strip()
    description = (description or "").strip()
    if not name or not description:
        raise InvalidDecisionError("A new category needs a name and a description")
    if session.scalar(select(Category).where(Category.name == name)) is not None:
        raise InvalidDecisionError(f'A category named "{name}" already exists')

    category = Category(name=name, description=description, status=CategoryStatus.ACTIVE)
    session.add(category)
    session.flush()
    ctx.log.record(
        EventType.CATEGORY_CREATED,
        Actor.human(ctx.reviewer),
        parent_event_id=parent_event_id,
        payload={
            "category_id": category.id,
            "name": name,
            "description": description,
            "source_proposal_id": source_proposal_id,
        },
    )
    return category


# --- Merges ------------------------------------------------------------------


def decide_merge(
    session: Session,
    ctx: TriageContext,
    proposal_id: int,
    approve: bool,
    edited_summary: str | None = None,
) -> Submission:
    """Approve (optionally editing the summary) or reject a duplicate proposal.

    Rejecting makes the submission a new feature, so categorization runs next.
    """
    proposal = _get_proposal(session, proposal_id, {ProposalType.DUPLICATE})
    submission = _ensure_pending_submission(proposal.submission)
    summary_proposal = session.scalar(
        select(Proposal).where(
            Proposal.submission_id == submission.id,
            Proposal.type == ProposalType.SUMMARY,
            Proposal.status == ProposalStatus.PENDING,
        )
    )
    edited_summary = (edited_summary or "").strip() or None

    with correlation_scope(proposal.correlation_id):
        if not approve:
            status = validate_decision(proposal.type, proposal.status, Decision.REJECT)
            mark_decided(proposal, status, ctx.reviewer, None)
            if summary_proposal is not None:
                mark_decided(summary_proposal, ProposalStatus.REJECTED, ctx.reviewer, None)
            event = ctx.log.record(
                EventType.MERGE_DECISION,
                Actor.human(ctx.reviewer),
                parent_event_id=proposal.ai_event_id,
                submission_id=submission.id,
                feature_id=proposal.feature_id,
                payload={"decision": Decision.REJECT},
            )
            proposal.decision_event_id = event.event_id
            run_categorization(session, ctx, submission, event.event_id)
            session.commit()
            return submission

        status = validate_decision(proposal.type, proposal.status, Decision.APPROVE)
        feature = session.get(Feature, proposal.feature_id)
        if feature is None:
            raise NotFoundError(f"Feature {proposal.feature_id} no longer exists")

        # The review form pre-fills the AI's summary; unchanged text is not an edit.
        baseline = summary_proposal.payload["new"] if summary_proposal else feature.summary
        if edited_summary == baseline.strip():
            edited_summary = None

        payload: dict = {"decision": Decision.APPROVE}
        if summary_proposal is not None:
            summary_decision = Decision.EDIT if edited_summary else Decision.APPROVE
            summary_payload = {"summary": edited_summary} if edited_summary else None
            summary_status = validate_decision(
                summary_proposal.type, summary_proposal.status, summary_decision, summary_payload
            )
            mark_decided(summary_proposal, summary_status, ctx.reviewer, summary_payload)
            feature.summary = edited_summary or summary_proposal.payload["new"]
            payload |= {
                "summary_decision": summary_decision,
                "summary_event_id": summary_proposal.ai_event_id,
            }
        elif edited_summary:
            feature.summary = edited_summary
            payload["summary_decision"] = "manual"
        if edited_summary:
            payload["summary"] = edited_summary

        mark_decided(proposal, status, ctx.reviewer, None)
        submission.feature_id = feature.id
        submission.status = SubmissionStatus.MERGED

        event = ctx.log.record(
            EventType.MERGE_DECISION,
            Actor.human(ctx.reviewer),
            parent_event_id=proposal.ai_event_id,
            submission_id=submission.id,
            feature_id=feature.id,
            payload=payload,
        )
        proposal.decision_event_id = event.event_id
        if summary_proposal is not None:
            summary_proposal.decision_event_id = event.event_id
        session.commit()
        return submission


# --- Categories --------------------------------------------------------------


def decide_category(
    session: Session,
    ctx: TriageContext,
    proposal_id: int,
    decision: Decision,
    *,
    category_id: int | None = None,
    summary: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Feature:
    """Decide a category or new-category proposal, creating the new feature.

    Rejecting never discards the submission: the reviewer's category is used.
    """
    proposal = _get_proposal(session, proposal_id, {ProposalType.CATEGORY, ProposalType.NEW_CATEGORY})
    submission = _ensure_pending_submission(proposal.submission)

    # The form pre-fills the AI's summary; unchanged text is not an edit.
    if (summary or "").strip() == (proposal.payload.get("feature_summary") or "").strip():
        summary = None
    decision_payload = {
        key: value
        for key, value in {
            "category_id": category_id,
            "summary": (summary or "").strip() or None,
            "name": (name or "").strip() or None,
            "description": (description or "").strip() or None,
        }.items()
        if value is not None
    }
    status = validate_decision(proposal.type, proposal.status, decision, decision_payload)

    with correlation_scope(proposal.correlation_id):
        created = None
        if decision is Decision.REJECT:
            category = _active_category(session, category_id)
        elif proposal.type is ProposalType.CATEGORY:
            chosen = category_id if decision is Decision.EDIT else proposal.payload.get("category_id")
            if chosen is None:
                raise InvalidDecisionError("The AI suggested no valid category; reject and choose one")
            category = _active_category(session, chosen)
        else:
            use_edits = decision is Decision.EDIT
            category = created = _create_category(
                session,
                ctx,
                name if use_edits else proposal.payload.get("new_category_name"),
                description if use_edits else proposal.payload.get("new_category_description"),
                proposal.ai_event_id,
                proposal.id,
            )

        feature = Feature(
            title=submission.title,
            summary=decision_payload.get("summary")
            or proposal.payload.get("feature_summary")
            or submission.description,
            category=category,
        )
        session.add(feature)
        session.flush()
        submission.feature_id = feature.id
        submission.status = SubmissionStatus.NEW_FEATURE
        mark_decided(proposal, status, ctx.reviewer, decision_payload)

        ai_choice = (
            {"category_id": proposal.payload.get("category_id")}
            if proposal.type is ProposalType.CATEGORY
            else {"new_category_name": proposal.payload.get("new_category_name")}
        )
        event = ctx.log.record(
            EventType.CATEGORY_DECISION,
            Actor.human(ctx.reviewer),
            parent_event_id=proposal.ai_event_id,
            submission_id=submission.id,
            feature_id=feature.id,
            payload={
                "decision": decision,
                "ai_choice": ai_choice,
                "final_category_id": category.id,
                "created_category": created is not None,
                "summary_edited": "summary" in decision_payload,
            },
        )
        proposal.decision_event_id = event.event_id
        session.commit()
        return feature


def assign_manually(
    session: Session,
    ctx: TriageContext,
    submission_id: int,
    category_id: int,
    summary: str | None = None,
) -> Feature:
    """Categorize a submission with no pending AI proposal (e.g. the AI call failed)."""
    submission = _ensure_pending_submission(session.get(Submission, submission_id))
    has_pending = session.scalar(
        select(Proposal.id).where(
            Proposal.submission_id == submission.id, Proposal.status == ProposalStatus.PENDING
        )
    )
    if has_pending:
        raise InvalidDecisionError("This submission has a pending AI proposal; decide that instead")

    with correlation_scope(submission.correlation_id):
        category = _active_category(session, category_id)
        feature = Feature(
            title=submission.title,
            summary=(summary or "").strip() or submission.description,
            category=category,
        )
        session.add(feature)
        session.flush()
        submission.feature_id = feature.id
        submission.status = SubmissionStatus.NEW_FEATURE
        ctx.log.record(
            EventType.CATEGORY_DECISION,
            Actor.human(ctx.reviewer),
            submission_id=submission.id,
            feature_id=feature.id,
            payload={"decision": "manual", "final_category_id": category.id},
        )
        session.commit()
        return feature
