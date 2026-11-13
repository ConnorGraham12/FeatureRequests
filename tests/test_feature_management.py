"""Feature toolbar actions (edit, delete, restore) and merged-submission actions
(unmerge, move, delete, restore)."""

import pytest

from app.db.models import Proposal
from app.domain.enums import ProposalStatus, ProposalType, SubmissionStatus
from app.domain.proposals import InvalidDecisionError
from app.observability.correlation import bind_correlation_id, generate_correlation_id
from app.observability.decision_log import EventType
from app.pipeline.schemas import CategoryAssignment, DuplicateJudgment, SummaryUpdate
from app.services import feature_management, review, triage
from tests.fakes import FakeEmbedder

DARK_MODE = ("Dark mode please", "The white background is harsh at night, can we get a dark theme")
NIGHT_THEME = ("Night theme option", "Can we get a dark theme, the white background is harsh at night")
SLACK = ("Create tasks from Slack", "Turn a Slack message into a task with one click")


def category_answer(category_id):
    return CategoryAssignment(
        action="assign", category_id=category_id, new_category_name=None,
        new_category_description=None, feature_summary="Users want this.",
        confidence=0.9, reasoning=".",
    )


def submit(session, context, request):
    title, description = request
    with bind_correlation_id(generate_correlation_id()):
        return triage.process_new_submission(session, context, title, description)


def pending_proposal(session, submission, proposal_type):
    return (
        session.query(Proposal)
        .filter_by(submission_id=submission.id, type=proposal_type, status=ProposalStatus.PENDING)
        .one()
    )


def create_feature(session, context, fake_model, request, category):
    fake_model.queue_answers(category_answer(category.id))
    submission = submit(session, context, request)
    proposal = pending_proposal(session, submission, ProposalType.CATEGORY)
    return review.approve_suggested_category(session, context, proposal.id), submission


def merge_into(session, context, fake_model, feature, request):
    fake_model.queue_answers(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.9, reasoning=".", rejected_candidates=[]),
        SummaryUpdate(summary=feature.summary, changed=False, reasoning="."),
    )
    submission = submit(session, context, request)
    proposal = pending_proposal(session, submission, ProposalType.DUPLICATE)
    review.approve_merge(session, context, proposal.id)
    return submission


@pytest.fixture
def ui_category(categories):
    return categories["UI/UX & Accessibility"]


@pytest.fixture
def merged_feature(session, context, fake_model, ui_category):
    """A dark-mode feature with two submissions: the original and a merged duplicate."""
    feature, original = create_feature(session, context, fake_model, DARK_MODE, ui_category)
    duplicate = merge_into(session, context, fake_model, feature, NIGHT_THEME)
    return feature, original, duplicate


def last_event(context, correlation_id):
    return context.decision_log.read_events_for_correlation_id(correlation_id)[-1]


# --- Edit ---------------------------------------------------------------------------


def test_edit_changes_title_and_summary_and_logs_old_and_new(session, context, fake_model, ui_category):
    feature, original = create_feature(session, context, fake_model, DARK_MODE, ui_category)

    feature_management.edit_feature_details(
        session, context, feature.id, "Dark mode", "A dark theme for the web app."
    )

    session.refresh(feature)
    assert feature.title == "Dark mode"
    assert feature.summary == "A dark theme for the web app."
    assert original.title == "Dark mode please"  # the submission is never changed
    event = last_event(context, original.correlation_id)
    assert event.event_type is EventType.FEATURE_EDITED
    assert event.payload["title"] == {"old": "Dark mode please", "new": "Dark mode"}


def test_edit_with_no_changes_or_blank_title_is_refused(session, context, fake_model, ui_category):
    feature, _ = create_feature(session, context, fake_model, DARK_MODE, ui_category)

    with pytest.raises(InvalidDecisionError, match="Nothing was changed"):
        feature_management.edit_feature_details(session, context, feature.id, feature.title, feature.summary)
    with pytest.raises(InvalidDecisionError, match="A title is required"):
        feature_management.edit_feature_details(session, context, feature.id, "  ", "x")


# --- Delete (archive) and restore ------------------------------------------------------


def test_deleted_feature_is_hidden_from_duplicate_search_and_can_be_restored(
    session, context, fake_model, ui_category
):
    feature, original = create_feature(session, context, fake_model, DARK_MODE, ui_category)

    feature_management.archive_feature(session, context, feature.id)

    # A near-identical request now finds no candidates, so it goes straight to categorization.
    fake_model.queue_answers(category_answer(ui_category.id))
    later = submit(session, context, NIGHT_THEME)
    assert pending_proposal(session, later, ProposalType.CATEGORY)
    candidates_event = context.decision_log.read_events_for_correlation_id(later.correlation_id)[1]
    assert candidates_event.payload["candidates"] == []

    feature_management.restore_feature(session, context, feature.id)
    session.refresh(feature)
    assert not feature.is_archived
    types = [e.event_type for e in context.decision_log.read_events_for_correlation_id(original.correlation_id)]
    assert types[-2:] == [EventType.FEATURE_ARCHIVED, EventType.FEATURE_RESTORED]


def test_deleted_feature_cannot_be_edited_or_merged_into(session, context, fake_model, ui_category):
    feature, _ = create_feature(session, context, fake_model, DARK_MODE, ui_category)
    fake_model.queue_answers(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.9, reasoning=".", rejected_candidates=[]),
        SummaryUpdate(summary="x", changed=True, reasoning="."),
    )
    waiting = submit(session, context, NIGHT_THEME)  # merge proposal waiting for review
    merge_proposal = pending_proposal(session, waiting, ProposalType.DUPLICATE)

    feature_management.archive_feature(session, context, feature.id)

    with pytest.raises(InvalidDecisionError, match="deleted"):
        feature_management.edit_feature_details(session, context, feature.id, "New", "New")
    with pytest.raises(InvalidDecisionError, match="deleted"):
        review.approve_merge(session, context, merge_proposal.id)


# --- Merged submissions --------------------------------------------------------------


def test_unmerge_splits_a_submission_into_its_own_feature(session, context, merged_feature, ui_category):
    feature, original, duplicate = merged_feature

    new_feature = feature_management.unmerge_submission(session, context, duplicate.id)

    session.refresh(feature)
    assert [s.id for s in feature.active_submissions] == [original.id]
    assert new_feature.title == "Night theme option"
    assert new_feature.category_id == ui_category.id
    assert duplicate.status is SubmissionStatus.NEW_FEATURE
    event = last_event(context, duplicate.correlation_id)
    assert event.event_type is EventType.SUBMISSION_UNMERGED
    assert event.payload == {"from_feature_id": feature.id, "to_feature_id": new_feature.id}


def test_move_attaches_a_submission_to_another_feature(
    session, context, fake_model, merged_feature, categories
):
    feature, _, duplicate = merged_feature
    other_feature, _ = create_feature(session, context, fake_model, SLACK, categories["Integrations & API"])

    feature_management.move_submission_to_feature(session, context, duplicate.id, other_feature.id)

    session.refresh(other_feature)
    assert duplicate.id in [s.id for s in other_feature.active_submissions]
    assert duplicate.status is SubmissionStatus.MERGED
    assert last_event(context, duplicate.correlation_id).event_type is EventType.SUBMISSION_MOVED


def test_deleted_submission_stops_counting_and_can_be_restored(session, context, merged_feature):
    feature, original, duplicate = merged_feature

    feature_management.archive_submission(session, context, duplicate.id)
    session.refresh(feature)
    assert [s.id for s in feature.active_submissions] == [original.id]
    assert len(feature.submissions) == 2  # kept, just hidden

    feature_management.restore_submission(session, context, duplicate.id)
    session.refresh(feature)
    assert len(feature.active_submissions) == 2


def test_deleted_submission_is_left_out_of_duplicate_search(session, context, fake_model, merged_feature):
    from app.pipeline import stages

    feature, original, duplicate = merged_feature
    feature_management.archive_submission(session, context, duplicate.id)

    embedding = FakeEmbedder().embed_texts(["Night theme option\nCan we get a dark theme"])[0]
    candidates = stages.find_duplicate_candidates(session, embedding, max_candidates=5, min_similarity=0.0)

    matched_submission_ids = {c.most_similar_submission_id for c in candidates}
    assert duplicate.id not in matched_submission_ids
    assert original.id in matched_submission_ids


@pytest.mark.parametrize("action", ["unmerge", "move", "archive"])
def test_the_last_submission_in_a_feature_cannot_be_removed(
    session, context, fake_model, ui_category, categories, action
):
    feature, only_submission = create_feature(session, context, fake_model, DARK_MODE, ui_category)
    other_feature, _ = create_feature(session, context, fake_model, SLACK, categories["Integrations & API"])

    with pytest.raises(InvalidDecisionError, match="only submission"):
        if action == "unmerge":
            feature_management.unmerge_submission(session, context, only_submission.id)
        elif action == "move":
            feature_management.move_submission_to_feature(session, context, only_submission.id, other_feature.id)
        else:
            feature_management.archive_submission(session, context, only_submission.id)
