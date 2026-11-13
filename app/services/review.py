"""Human decisions on AI proposals. This is the only place submissions join features.

There is one function per action a reviewer can take:

    Duplicate proposals:
        approve_merge                    merge the submission into the suggested feature
        reject_merge                     not a duplicate; ask the AI for a category instead

    Category proposals (the submission becomes a new feature):
        approve_suggested_category       use the existing category the AI suggested
        create_suggested_new_category    create the new category the AI suggested
        create_edited_new_category       create a new category with the reviewer's name/description
        reject_category_suggestion       use a category the reviewer chooses instead

    No AI proposal (the AI call failed):
        assign_category_manually

Changes to features after triage (edit, archive, change category, unmerge...)
are in app/services/feature_management.py.

Each decision is logged under the submission's own correlation ID, so it lands
in the same trace as the AI events it decides on, however much later it happens.
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
from app.domain.proposals import (
    InvalidDecisionError,
    ensure_category_is_chosen,
    ensure_proposal_is_pending,
    ensure_text_is_present,
)
from app.observability.correlation import bind_correlation_id
from app.observability.decision_log import Actor, EventType
from app.services.context import TriageContext, record_decision_on_proposal
from app.services.triage import propose_category_for_submission


class NotFoundError(LookupError):
    pass


# --- Loading and checking ------------------------------------------------------


def load_proposal(session: Session, proposal_id: int, allowed_types: set[ProposalType]) -> Proposal:
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.type not in allowed_types:
        allowed = " or ".join(sorted(allowed_types))
        raise NotFoundError(f"No {allowed} proposal with id {proposal_id}")
    return proposal


def load_pending_submission(proposal: Proposal) -> Submission:
    """Return the proposal's submission, checking it is still awaiting review."""
    submission = proposal.submission
    if submission is None:
        raise NotFoundError("This proposal has no submission")
    if submission.status is not SubmissionStatus.PENDING:
        raise InvalidDecisionError(f"This submission is already {submission.status}")
    return submission


def load_active_category(session: Session, category_id: int) -> Category:
    category = session.get(Category, category_id)
    if category is None or category.status is not CategoryStatus.ACTIVE:
        raise InvalidDecisionError("Choose an active category")
    return category


def find_pending_summary_proposal(session: Session, submission_id: int) -> Proposal | None:
    query = select(Proposal).where(
        Proposal.submission_id == submission_id,
        Proposal.type == ProposalType.SUMMARY,
        Proposal.status == ProposalStatus.PENDING,
    )
    return session.scalar(query)


def get_edited_text(submitted_text: str | None, original_text: str | None) -> str | None:
    """Return the reviewer's text only if they actually changed it.

    Review forms are pre-filled with the AI's text, so submitting it unchanged
    is not an edit.
    """
    submitted = (submitted_text or "").strip()
    original = (original_text or "").strip()
    if not submitted or submitted == original:
        return None
    return submitted


def create_active_category(
    session: Session,
    context: TriageContext,
    name: str,
    description: str,
    *,
    parent_event_id: str | None,
    source_proposal_id: int | None,
) -> Category:
    """Create a new active category and log it. The name must not already exist."""
    name = ensure_text_is_present(name, "A category name")
    description = ensure_text_is_present(description, "A category description")

    existing = session.scalar(select(Category).where(Category.name == name))
    if existing is not None:
        raise InvalidDecisionError(f'A category named "{name}" already exists')

    category = Category(name=name, description=description, status=CategoryStatus.ACTIVE)
    session.add(category)
    session.flush()  # assigns category.id

    context.decision_log.write_event(
        EventType.CATEGORY_CREATED,
        Actor.for_human_reviewer(context.reviewer_name),
        parent_event_id=parent_event_id,
        payload={
            "category_id": category.id,
            "name": name,
            "description": description,
            "source_proposal_id": source_proposal_id,
        },
    )
    return category


# --- Merges --------------------------------------------------------------------


def approve_merge(
    session: Session,
    context: TriageContext,
    proposal_id: int,
    edited_summary: str | None = None,
) -> Submission:
    """Merge the submission into the feature the AI matched it to.

    The feature summary becomes the AI's proposed update, or the reviewer's
    edited version if they changed it.
    """
    merge_proposal = load_proposal(session, proposal_id, {ProposalType.DUPLICATE})
    ensure_proposal_is_pending(merge_proposal.status)
    submission = load_pending_submission(merge_proposal)

    feature = session.get(Feature, merge_proposal.feature_id)
    if feature is None:
        raise NotFoundError(f"Feature {merge_proposal.feature_id} no longer exists")
    if feature.is_archived:
        raise InvalidDecisionError(
            "That feature has been deleted. Restore it first, or reject the merge."
        )
    summary_proposal = find_pending_summary_proposal(session, submission.id)

    with bind_correlation_id(merge_proposal.correlation_id):
        decision_details: dict = {"decision": Decision.APPROVE}

        if summary_proposal is not None:
            ai_summary = summary_proposal.payload["new"]
            reviewer_summary = get_edited_text(edited_summary, ai_summary)
            if reviewer_summary is not None:
                feature.summary = reviewer_summary
                record_decision_on_proposal(
                    summary_proposal, ProposalStatus.EDITED, context.reviewer_name,
                    {"summary": reviewer_summary},
                )
                decision_details["summary_decision"] = Decision.EDIT
                decision_details["summary"] = reviewer_summary
            else:
                feature.summary = ai_summary
                record_decision_on_proposal(
                    summary_proposal, ProposalStatus.APPROVED, context.reviewer_name, None
                )
                decision_details["summary_decision"] = Decision.APPROVE
            decision_details["summary_event_id"] = summary_proposal.ai_event_id
        else:
            # No AI summary was proposed (that call failed); the reviewer may still edit it.
            reviewer_summary = get_edited_text(edited_summary, feature.summary)
            if reviewer_summary is not None:
                feature.summary = reviewer_summary
                decision_details["summary_decision"] = "manual"
                decision_details["summary"] = reviewer_summary

        record_decision_on_proposal(merge_proposal, ProposalStatus.APPROVED, context.reviewer_name, None)
        submission.feature_id = feature.id
        submission.status = SubmissionStatus.MERGED

        merge_event = context.decision_log.write_event(
            EventType.MERGE_DECISION,
            Actor.for_human_reviewer(context.reviewer_name),
            parent_event_id=merge_proposal.ai_event_id,
            submission_id=submission.id,
            feature_id=feature.id,
            payload=decision_details,
        )
        merge_proposal.decision_event_id = merge_event.event_id
        if summary_proposal is not None:
            summary_proposal.decision_event_id = merge_event.event_id
        session.commit()
        return submission


def reject_merge(session: Session, context: TriageContext, proposal_id: int) -> Submission:
    """The submission is not a duplicate. It will become a new feature, so the
    AI is asked for a category next."""
    merge_proposal = load_proposal(session, proposal_id, {ProposalType.DUPLICATE})
    ensure_proposal_is_pending(merge_proposal.status)
    submission = load_pending_submission(merge_proposal)
    summary_proposal = find_pending_summary_proposal(session, submission.id)

    with bind_correlation_id(merge_proposal.correlation_id):
        record_decision_on_proposal(merge_proposal, ProposalStatus.REJECTED, context.reviewer_name, None)
        if summary_proposal is not None:
            record_decision_on_proposal(
                summary_proposal, ProposalStatus.REJECTED, context.reviewer_name, None
            )

        merge_event = context.decision_log.write_event(
            EventType.MERGE_DECISION,
            Actor.for_human_reviewer(context.reviewer_name),
            parent_event_id=merge_proposal.ai_event_id,
            submission_id=submission.id,
            feature_id=merge_proposal.feature_id,
            payload={"decision": Decision.REJECT},
        )
        merge_proposal.decision_event_id = merge_event.event_id

        propose_category_for_submission(session, context, submission, merge_event.event_id)
        session.commit()
        return submission


# --- Categories for a new feature -------------------------------------------------


def create_feature_for_submission(
    session: Session, submission: Submission, category: Category, summary: str
) -> Feature:
    """Create the feature this submission starts, and attach the submission to it."""
    feature = Feature(title=submission.title, summary=summary, category=category)
    session.add(feature)
    session.flush()  # assigns feature.id
    submission.feature_id = feature.id
    submission.status = SubmissionStatus.NEW_FEATURE
    return feature


def choose_feature_summary(proposal: Proposal, submission: Submission, edited_summary: str | None) -> tuple[str, bool]:
    """Return (summary to use, whether the reviewer edited it).

    Uses the reviewer's edit if there is one, otherwise the AI's summary, and
    falls back to the submission's own description.
    """
    ai_summary = proposal.payload.get("feature_summary")
    reviewer_summary = get_edited_text(edited_summary, ai_summary)
    if reviewer_summary is not None:
        return reviewer_summary, True
    if ai_summary:
        return ai_summary, False
    return submission.description, False


def describe_ai_category_choice(proposal: Proposal) -> dict:
    if proposal.type is ProposalType.CATEGORY:
        return {"category_id": proposal.payload.get("category_id")}
    return {"new_category_name": proposal.payload.get("new_category_name")}


def finish_category_decision(
    session: Session,
    context: TriageContext,
    proposal: Proposal,
    submission: Submission,
    *,
    decision: Decision,
    new_status: ProposalStatus,
    category: Category,
    edited_summary: str | None,
    created_category: bool,
    decision_details: dict | None,
) -> Feature:
    """Shared final steps for every category decision: create the feature,
    mark the proposal decided, log the decision and commit."""
    summary, summary_was_edited = choose_feature_summary(proposal, submission, edited_summary)
    if summary_was_edited:
        if decision_details is None:
            decision_details = {}
        decision_details["summary"] = summary

    feature = create_feature_for_submission(session, submission, category, summary)
    record_decision_on_proposal(proposal, new_status, context.reviewer_name, decision_details)

    decision_event = context.decision_log.write_event(
        EventType.CATEGORY_DECISION,
        Actor.for_human_reviewer(context.reviewer_name),
        parent_event_id=proposal.ai_event_id,
        submission_id=submission.id,
        feature_id=feature.id,
        payload={
            "decision": decision,
            "ai_choice": describe_ai_category_choice(proposal),
            "final_category_id": category.id,
            "created_category": created_category,
            "summary_edited": summary_was_edited,
        },
    )
    proposal.decision_event_id = decision_event.event_id
    session.commit()
    return feature


def approve_suggested_category(
    session: Session, context: TriageContext, proposal_id: int, edited_summary: str | None = None
) -> Feature:
    """Accept the existing category the AI suggested."""
    proposal = load_proposal(session, proposal_id, {ProposalType.CATEGORY})
    ensure_proposal_is_pending(proposal.status)
    submission = load_pending_submission(proposal)

    suggested_category_id = proposal.payload.get("category_id")
    if suggested_category_id is None:
        raise InvalidDecisionError("The AI suggested no valid category; reject it and choose one")

    with bind_correlation_id(proposal.correlation_id):
        category = load_active_category(session, suggested_category_id)
        return finish_category_decision(
            session, context, proposal, submission,
            decision=Decision.APPROVE,
            new_status=ProposalStatus.APPROVED,
            category=category,
            edited_summary=edited_summary,
            created_category=False,
            decision_details=None,
        )


def create_suggested_new_category(
    session: Session, context: TriageContext, proposal_id: int, edited_summary: str | None = None
) -> Feature:
    """Create the new category the AI proposed, exactly as it described it."""
    proposal = load_proposal(session, proposal_id, {ProposalType.NEW_CATEGORY})
    ensure_proposal_is_pending(proposal.status)
    submission = load_pending_submission(proposal)

    with bind_correlation_id(proposal.correlation_id):
        category = create_active_category(
            session, context,
            proposal.payload.get("new_category_name"),
            proposal.payload.get("new_category_description"),
            parent_event_id=proposal.ai_event_id,
            source_proposal_id=proposal.id,
        )
        return finish_category_decision(
            session, context, proposal, submission,
            decision=Decision.APPROVE,
            new_status=ProposalStatus.APPROVED,
            category=category,
            edited_summary=edited_summary,
            created_category=True,
            decision_details=None,
        )


def create_edited_new_category(
    session: Session,
    context: TriageContext,
    proposal_id: int,
    name: str,
    description: str,
    edited_summary: str | None = None,
) -> Feature:
    """Create a new category, using the reviewer's name and description
    instead of the AI's."""
    proposal = load_proposal(session, proposal_id, {ProposalType.NEW_CATEGORY})
    ensure_proposal_is_pending(proposal.status)
    submission = load_pending_submission(proposal)
    name = ensure_text_is_present(name, "A category name")
    description = ensure_text_is_present(description, "A category description")

    with bind_correlation_id(proposal.correlation_id):
        category = create_active_category(
            session, context, name, description,
            parent_event_id=proposal.ai_event_id,
            source_proposal_id=proposal.id,
        )
        return finish_category_decision(
            session, context, proposal, submission,
            decision=Decision.EDIT,
            new_status=ProposalStatus.EDITED,
            category=category,
            edited_summary=edited_summary,
            created_category=True,
            decision_details={"name": name, "description": description},
        )


def reject_category_suggestion(
    session: Session,
    context: TriageContext,
    proposal_id: int,
    chosen_category_id: int | None,
    edited_summary: str | None = None,
) -> Feature:
    """Reject the AI's category (existing or new) and use the reviewer's choice.

    The submission is never discarded: it is saved with the chosen category.
    """
    proposal = load_proposal(session, proposal_id, {ProposalType.CATEGORY, ProposalType.NEW_CATEGORY})
    ensure_proposal_is_pending(proposal.status)
    submission = load_pending_submission(proposal)
    chosen_category_id = ensure_category_is_chosen(chosen_category_id)

    with bind_correlation_id(proposal.correlation_id):
        category = load_active_category(session, chosen_category_id)
        return finish_category_decision(
            session, context, proposal, submission,
            decision=Decision.REJECT,
            new_status=ProposalStatus.REJECTED,
            category=category,
            edited_summary=edited_summary,
            created_category=False,
            decision_details={"category_id": category.id},
        )


def assign_category_manually(
    session: Session,
    context: TriageContext,
    submission_id: int,
    category_id: int,
    summary: str | None = None,
) -> Feature:
    """Categorize a submission that has no AI proposal, e.g. because the AI call failed."""
    submission = session.get(Submission, submission_id)
    if submission is None:
        raise NotFoundError(f"No submission with id {submission_id}")
    if submission.status is not SubmissionStatus.PENDING:
        raise InvalidDecisionError(f"This submission is already {submission.status}")

    pending_proposal_query = select(Proposal.id).where(
        Proposal.submission_id == submission.id,
        Proposal.status == ProposalStatus.PENDING,
    )
    if session.scalar(pending_proposal_query) is not None:
        raise InvalidDecisionError("This submission has a pending AI proposal; decide that instead")

    with bind_correlation_id(submission.correlation_id):
        category = load_active_category(session, category_id)
        feature_summary = (summary or "").strip() or submission.description
        feature = create_feature_for_submission(session, submission, category, feature_summary)

        context.decision_log.write_event(
            EventType.CATEGORY_DECISION,
            Actor.for_human_reviewer(context.reviewer_name),
            submission_id=submission.id,
            feature_id=feature.id,
            payload={"decision": "manual", "final_category_id": category.id},
        )
        session.commit()
        return feature
