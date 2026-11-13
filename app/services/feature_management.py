"""Changes a reviewer makes to features after triage.

    Feature toolbar:
        edit_feature_details                change the title and summary
        archive_feature / restore_feature   "Delete" hides the feature; it can be restored
        move_feature_to_existing_category   fix a wrong category
        move_feature_to_new_category        ...or create the right one

    Each merged submission inside a feature:
        unmerge_submission                  split it out into its own new feature
        move_submission_to_feature          attach it to a different feature
        archive_submission / restore_submission

Every change is logged. Feature changes use the correlation ID of the
submission that created the feature, so they appear in that feature's original
trace. Submission changes use that submission's own correlation ID.

A feature always keeps at least one active submission: the last one can't be
unmerged, moved or deleted. Delete the whole feature instead.
"""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.models import Category, Feature, Submission
from app.domain.enums import SubmissionStatus
from app.domain.proposals import InvalidDecisionError, ensure_text_is_present
from app.observability.correlation import bind_correlation_id
from app.observability.decision_log import Actor, EventType
from app.services.context import TriageContext
from app.services.review import NotFoundError, create_active_category, load_active_category


# --- Loading and checking ---------------------------------------------------------


def load_feature(session: Session, feature_id: int) -> Feature:
    feature = session.get(Feature, feature_id)
    if feature is None:
        raise NotFoundError(f"No feature with id {feature_id}")
    if not feature.submissions:
        raise InvalidDecisionError("This feature has no submissions")
    return feature


def load_active_feature(session: Session, feature_id: int) -> Feature:
    """Load a feature that hasn't been deleted (archived)."""
    feature = load_feature(session, feature_id)
    if feature.is_archived:
        raise InvalidDecisionError("This feature has been deleted. Restore it first.")
    return feature


def load_triaged_submission(session: Session, submission_id: int) -> Submission:
    """Load a submission that belongs to a feature that hasn't been deleted."""
    submission = session.get(Submission, submission_id)
    if submission is None:
        raise NotFoundError(f"No submission with id {submission_id}")
    if submission.feature is None:
        raise InvalidDecisionError("This submission hasn't been triaged yet")
    if submission.feature.is_archived:
        raise InvalidDecisionError("This submission's feature has been deleted. Restore it first.")
    return submission


def ensure_not_the_last_active_submission(submission: Submission, action: str) -> None:
    active_submissions = submission.feature.active_submissions
    if len(active_submissions) == 1 and active_submissions[0].id == submission.id:
        raise InvalidDecisionError(
            f"Can't {action} the only submission in this feature. Delete the feature instead."
        )


def get_original_correlation_id(feature: Feature) -> str:
    """The correlation ID of the submission that created the feature."""
    return feature.submissions[0].correlation_id


def write_human_event(context: TriageContext, event_type: EventType, **event_fields) -> None:
    context.decision_log.write_event(
        event_type, Actor.for_human_reviewer(context.reviewer_name), **event_fields
    )


# --- Feature toolbar: edit ------------------------------------------------------------


def edit_feature_details(
    session: Session, context: TriageContext, feature_id: int, title: str, summary: str
) -> Feature:
    """Change a feature's title and summary. The original submissions are not changed."""
    feature = load_active_feature(session, feature_id)
    new_title = ensure_text_is_present(title, "A title")
    new_summary = ensure_text_is_present(summary, "A summary")

    changes = {}
    if new_title != feature.title:
        changes["title"] = {"old": feature.title, "new": new_title}
    if new_summary != feature.summary:
        changes["summary"] = {"old": feature.summary, "new": new_summary}
    if not changes:
        raise InvalidDecisionError("Nothing was changed")

    with bind_correlation_id(get_original_correlation_id(feature)):
        feature.title = new_title
        feature.summary = new_summary
        write_human_event(context, EventType.FEATURE_EDITED, feature_id=feature.id, payload=changes)
        session.commit()
        return feature


# --- Feature toolbar: delete (archive) and restore -------------------------------------


def archive_feature(session: Session, context: TriageContext, feature_id: int) -> Feature:
    """Hide the feature from lists and duplicate search. Nothing is removed."""
    feature = load_active_feature(session, feature_id)
    with bind_correlation_id(get_original_correlation_id(feature)):
        feature.archived_at = datetime.now(UTC)
        write_human_event(
            context,
            EventType.FEATURE_ARCHIVED,
            feature_id=feature.id,
            payload={"title": feature.title, "submission_ids": [s.id for s in feature.submissions]},
        )
        session.commit()
        return feature


def restore_feature(session: Session, context: TriageContext, feature_id: int) -> Feature:
    feature = load_feature(session, feature_id)
    if not feature.is_archived:
        raise InvalidDecisionError("This feature isn't deleted")
    with bind_correlation_id(get_original_correlation_id(feature)):
        feature.archived_at = None
        write_human_event(context, EventType.FEATURE_RESTORED, feature_id=feature.id)
        session.commit()
        return feature


# --- Feature toolbar: change category ------------------------------------------------


def log_feature_category_change(
    context: TriageContext,
    feature: Feature,
    previous_category: Category | None,
    new_category: Category,
    *,
    created_category: bool,
    reason: str | None,
) -> None:
    payload = {
        "from_category_id": previous_category.id if previous_category else None,
        "from_category_name": previous_category.name if previous_category else None,
        "to_category_id": new_category.id,
        "to_category_name": new_category.name,
        "created_category": created_category,
        "submission_ids": [submission.id for submission in feature.submissions],
    }
    reason = (reason or "").strip()
    if reason:
        payload["reason"] = reason
    write_human_event(context, EventType.CATEGORY_REASSIGNED, feature_id=feature.id, payload=payload)


def move_feature_to_existing_category(
    session: Session,
    context: TriageContext,
    feature_id: int,
    category_id: int,
    reason: str | None = None,
) -> Feature:
    """Move a feature to another existing category, e.g. after a mis-click."""
    feature = load_active_feature(session, feature_id)
    previous_category = feature.category

    with bind_correlation_id(get_original_correlation_id(feature)):
        new_category = load_active_category(session, category_id)
        if previous_category is not None and new_category.id == previous_category.id:
            raise InvalidDecisionError(f'This feature is already in "{new_category.name}"')

        feature.category = new_category
        log_feature_category_change(
            context, feature, previous_category, new_category, created_category=False, reason=reason
        )
        session.commit()
        return feature


def move_feature_to_new_category(
    session: Session,
    context: TriageContext,
    feature_id: int,
    name: str,
    description: str,
    reason: str | None = None,
) -> Feature:
    """Create a new category and move the feature into it."""
    feature = load_active_feature(session, feature_id)
    previous_category = feature.category

    with bind_correlation_id(get_original_correlation_id(feature)):
        new_category = create_active_category(
            session, context, name, description, parent_event_id=None, source_proposal_id=None
        )
        feature.category = new_category
        log_feature_category_change(
            context, feature, previous_category, new_category, created_category=True, reason=reason
        )
        session.commit()
        return feature


# --- Merged submissions: unmerge and move -------------------------------------------


def unmerge_submission(session: Session, context: TriageContext, submission_id: int) -> Feature:
    """Split a submission out of its feature into a new feature of its own.

    The new feature takes the submission's title and description and the old
    feature's category. Returns the new feature. The old feature's summary is
    left alone, so the reviewer may want to edit it if it mentioned this request.
    """
    submission = load_triaged_submission(session, submission_id)
    if submission.is_archived:
        raise InvalidDecisionError("Restore this submission before unmerging it")
    ensure_not_the_last_active_submission(submission, "unmerge")
    old_feature = submission.feature

    with bind_correlation_id(submission.correlation_id):
        new_feature = Feature(
            title=submission.title,
            summary=submission.description,
            category=old_feature.category,
        )
        session.add(new_feature)
        session.flush()  # assigns new_feature.id

        submission.feature_id = new_feature.id
        submission.status = SubmissionStatus.NEW_FEATURE
        write_human_event(
            context,
            EventType.SUBMISSION_UNMERGED,
            submission_id=submission.id,
            feature_id=new_feature.id,
            payload={"from_feature_id": old_feature.id, "to_feature_id": new_feature.id},
        )
        session.commit()
        return new_feature


def move_submission_to_feature(
    session: Session, context: TriageContext, submission_id: int, target_feature_id: int
) -> Feature:
    """Attach a submission to a different existing feature. Returns that feature."""
    submission = load_triaged_submission(session, submission_id)
    if submission.is_archived:
        raise InvalidDecisionError("Restore this submission before moving it")
    old_feature = submission.feature
    if target_feature_id == old_feature.id:
        raise InvalidDecisionError("The submission is already in that feature")
    ensure_not_the_last_active_submission(submission, "move")
    target_feature = load_active_feature(session, target_feature_id)

    with bind_correlation_id(submission.correlation_id):
        submission.feature_id = target_feature.id
        submission.status = SubmissionStatus.MERGED
        write_human_event(
            context,
            EventType.SUBMISSION_MOVED,
            submission_id=submission.id,
            feature_id=target_feature.id,
            payload={"from_feature_id": old_feature.id, "to_feature_id": target_feature.id},
        )
        session.commit()
        return target_feature


# --- Merged submissions: delete (archive) and restore ---------------------------------


def archive_submission(session: Session, context: TriageContext, submission_id: int) -> Submission:
    """Hide one submission: it no longer counts towards its feature or matches in
    duplicate search. Nothing is removed."""
    submission = load_triaged_submission(session, submission_id)
    if submission.is_archived:
        raise InvalidDecisionError("This submission is already deleted")
    ensure_not_the_last_active_submission(submission, "delete")

    with bind_correlation_id(submission.correlation_id):
        submission.archived_at = datetime.now(UTC)
        write_human_event(
            context,
            EventType.SUBMISSION_ARCHIVED,
            submission_id=submission.id,
            feature_id=submission.feature_id,
        )
        session.commit()
        return submission


def restore_submission(session: Session, context: TriageContext, submission_id: int) -> Submission:
    submission = load_triaged_submission(session, submission_id)
    if not submission.is_archived:
        raise InvalidDecisionError("This submission isn't deleted")

    with bind_correlation_id(submission.correlation_id):
        submission.archived_at = None
        write_human_event(
            context,
            EventType.SUBMISSION_RESTORED,
            submission_id=submission.id,
            feature_id=submission.feature_id,
        )
        session.commit()
        return submission
