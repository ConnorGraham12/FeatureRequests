"""End-to-end triage flows, using a fake language model and embedder."""

import pytest

from app.db.models import Category, Proposal
from app.domain.enums import ProposalStatus, ProposalType, SubmissionStatus
from app.domain.proposals import InvalidDecisionError, ProposalAlreadyDecidedError
from app.observability.correlation import bind_correlation_id, generate_correlation_id
from app.observability.decision_log import EventType
from app.pipeline.schemas import (
    CategoryAssignment,
    DuplicateJudgment,
    ProposedCategory,
    RejectedCandidate,
    SummaryUpdate,
    TaxonomyProposal,
)
from app.services import feature_management, review, taxonomy, triage

DARK_MODE = ("Dark mode please", "The white background is harsh at night, can we get a dark theme")
NIGHT_THEME = ("Night theme option", "Can we get a dark theme, the white background is harsh at night")
SLACK = ("Create tasks from Slack", "Turn a Slack message into a task with one click")


# --- Helpers ------------------------------------------------------------------


def category_answer(category_id, summary="Users want this.", confidence=0.9):
    return CategoryAssignment(
        action="assign",
        category_id=category_id,
        new_category_name=None,
        new_category_description=None,
        feature_summary=summary,
        confidence=confidence,
        reasoning="Fits the category.",
    )


def new_category_answer(name, description):
    return CategoryAssignment(
        action="propose_new",
        category_id=None,
        new_category_name=name,
        new_category_description=description,
        feature_summary="Users want this.",
        confidence=0.5,
        reasoning="Nothing fits.",
    )


def duplicate_answer(feature_id, confidence=0.9):
    return DuplicateJudgment(
        duplicate_of_feature_id=feature_id,
        confidence=confidence,
        reasoning="Same ask.",
        rejected_candidates=[],
    )


def summary_answer(text):
    return SummaryUpdate(summary=text, changed=True, reasoning="Adds detail.")


def submit(session, context, request):
    title, description = request
    with bind_correlation_id(generate_correlation_id()):
        return triage.process_new_submission(session, context, title, description)


def find_pending_proposal(session, submission, proposal_type):
    for proposal in session.query(Proposal).filter_by(submission_id=submission.id, type=proposal_type):
        if proposal.status is ProposalStatus.PENDING:
            return proposal
    raise AssertionError(f"No pending {proposal_type} proposal")


def create_approved_feature(session, context, fake_model, request, category):
    fake_model.queue_answers(category_answer(category.id, summary="Users want a dark theme."))
    submission = submit(session, context, request)
    proposal = find_pending_proposal(session, submission, ProposalType.CATEGORY)
    feature = review.approve_suggested_category(session, context, proposal.id)
    return feature, submission


def event_types_for(context, correlation_id):
    events = context.decision_log.read_events_for_correlation_id(correlation_id)
    return [event.event_type for event in events]


# --- New submissions and categories --------------------------------------------


def test_new_submission_is_not_saved_to_a_feature_until_approved(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    fake_model.queue_answers(category_answer(ui_category.id))

    submission = submit(session, context, DARK_MODE)

    assert submission.status is SubmissionStatus.PENDING
    assert submission.feature_id is None
    proposal = find_pending_proposal(session, submission, ProposalType.CATEGORY)
    assert proposal.payload["category_id"] == ui_category.id

    feature = review.approve_suggested_category(session, context, proposal.id)
    assert feature.category_id == ui_category.id
    assert feature.summary == "Users want this."
    assert submission.status is SubmissionStatus.NEW_FEATURE

    events = context.decision_log.read_events_for_correlation_id(submission.correlation_id)
    assert [event.event_type for event in events] == [
        EventType.SUBMISSION_RECEIVED,
        EventType.DUPLICATE_CANDIDATES,
        EventType.CATEGORY_PROPOSED,
        EventType.CATEGORY_DECISION,
    ]
    assert events[-1].parent_event_id == events[-2].event_id
    assert events[-1].actor.reviewer == "pm"


def test_rejected_category_is_saved_with_the_reviewers_choice(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    integrations_category = categories["Integrations & API"]
    fake_model.queue_answers(category_answer(ui_category.id))
    submission = submit(session, context, SLACK)
    proposal = find_pending_proposal(session, submission, ProposalType.CATEGORY)

    with pytest.raises(InvalidDecisionError, match="Choose a category"):
        review.reject_category_suggestion(session, context, proposal.id, chosen_category_id=None)

    feature = review.reject_category_suggestion(
        session, context, proposal.id, chosen_category_id=integrations_category.id
    )
    assert feature.category_id == integrations_category.id
    assert session.get(Proposal, proposal.id).status is ProposalStatus.REJECTED
    assert session.get(Proposal, proposal.id).decision_payload == {"category_id": integrations_category.id}


def test_new_category_suggestion_creates_the_category(session, context, fake_model):
    fake_model.queue_answers(new_category_answer("Automation", "Rules and recurring work."))
    submission = submit(session, context, ("Recurring tasks", "Repeat a task every week"))
    proposal = find_pending_proposal(session, submission, ProposalType.NEW_CATEGORY)

    feature = review.create_suggested_new_category(session, context, proposal.id)

    assert feature.category.name == "Automation"
    assert EventType.CATEGORY_CREATED in event_types_for(context, submission.correlation_id)


def test_new_category_can_be_created_with_reviewer_edits(session, context, fake_model):
    fake_model.queue_answers(new_category_answer("Automation", "Rules."))
    submission = submit(session, context, ("Recurring tasks", "Repeat a task every week"))
    proposal = find_pending_proposal(session, submission, ProposalType.NEW_CATEGORY)

    feature = review.create_edited_new_category(
        session, context, proposal.id, "Automation & Workflows", "Rules, triggers and recurring tasks."
    )

    assert feature.category.name == "Automation & Workflows"
    assert session.get(Proposal, proposal.id).status is ProposalStatus.EDITED


def test_rejecting_a_new_category_records_only_the_chosen_category(session, context, fake_model, categories):
    integrations_category = categories["Integrations & API"]
    fake_model.queue_answers(new_category_answer("Unclear", "Vague requests."))
    submission = submit(session, context, ("A", "B"))
    proposal = find_pending_proposal(session, submission, ProposalType.NEW_CATEGORY)

    review.reject_category_suggestion(
        session, context, proposal.id, chosen_category_id=integrations_category.id
    )

    assert session.get(Proposal, proposal.id).decision_payload == {"category_id": integrations_category.id}


def test_unchanged_prefilled_summary_is_not_an_edit(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    fake_model.queue_answers(category_answer(ui_category.id, summary="Users want a dark theme."))
    submission = submit(session, context, DARK_MODE)
    proposal = find_pending_proposal(session, submission, ProposalType.CATEGORY)

    review.approve_suggested_category(
        session, context, proposal.id, edited_summary="Users want a dark theme."
    )

    decision_event = context.decision_log.read_events_for_correlation_id(submission.correlation_id)[-1]
    assert decision_event.payload["summary_edited"] is False


def test_model_failure_is_logged_and_allows_manual_category(session, context, fake_model, categories):
    fake_model.queue_failure(CategoryAssignment, "API down")
    submission = submit(session, context, SLACK)

    assert not submission.proposals
    events = context.decision_log.read_events_for_correlation_id(submission.correlation_id)
    assert events[-1].event_type is EventType.PIPELINE_ERROR
    assert events[-1].payload["stage"] == "category.proposed"

    feature = review.assign_category_manually(
        session, context, submission.id, categories["Integrations & API"].id
    )
    assert feature.category.name == "Integrations & API"


def test_decided_submission_cannot_be_decided_again(session, context, fake_model, categories):
    _, submission = create_approved_feature(
        session, context, fake_model, DARK_MODE, categories["UI/UX & Accessibility"]
    )
    proposal = submission.proposals[0]
    with pytest.raises(ProposalAlreadyDecidedError):
        review.approve_suggested_category(session, context, proposal.id)


# --- Duplicates and merges ---------------------------------------------------


def test_duplicate_is_merged_with_the_approved_summary(session, context, fake_model, categories):
    feature, original = create_approved_feature(
        session, context, fake_model, DARK_MODE, categories["UI/UX & Accessibility"]
    )
    fake_model.queue_answers(
        duplicate_answer(feature.id, confidence=0.94),
        summary_answer("Users want a dark theme for night use."),
    )

    second = submit(session, context, NIGHT_THEME)

    merge_proposal = find_pending_proposal(session, second, ProposalType.DUPLICATE)
    assert merge_proposal.feature_id == feature.id
    assert merge_proposal.payload["candidates"][0]["feature_id"] == feature.id
    assert second.feature_id is None  # not merged until a human approves

    review.approve_merge(session, context, merge_proposal.id)

    session.refresh(feature)
    assert [submission.id for submission in feature.submissions] == [original.id, second.id]
    assert second.status is SubmissionStatus.MERGED
    assert feature.summary == "Users want a dark theme for night use."

    events = context.decision_log.read_events_for_correlation_id(second.correlation_id)
    assert [event.event_type for event in events] == [
        EventType.SUBMISSION_RECEIVED,
        EventType.DUPLICATE_CANDIDATES,
        EventType.DUPLICATE_JUDGMENT,
        EventType.SUMMARY_PROPOSED,
        EventType.MERGE_DECISION,
    ]
    merge_event = events[-1]
    assert merge_event.parent_event_id == events[2].event_id
    assert merge_event.payload["summary_decision"] == "approve"


def test_reviewer_edited_summary_is_used_and_logged(session, context, fake_model, categories):
    feature, _ = create_approved_feature(
        session, context, fake_model, DARK_MODE, categories["UI/UX & Accessibility"]
    )
    fake_model.queue_answers(duplicate_answer(feature.id), summary_answer("AI summary."))
    second = submit(session, context, NIGHT_THEME)
    merge_proposal = find_pending_proposal(session, second, ProposalType.DUPLICATE)

    review.approve_merge(session, context, merge_proposal.id, edited_summary="My summary.")

    session.refresh(feature)
    assert feature.summary == "My summary."
    merge_event = context.decision_log.read_events_for_correlation_id(second.correlation_id)[-1]
    assert merge_event.payload["summary_decision"] == "edit"
    assert merge_event.payload["summary"] == "My summary."


def test_rejected_merge_asks_for_a_category_next(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    feature, _ = create_approved_feature(session, context, fake_model, DARK_MODE, ui_category)
    fake_model.queue_answers(duplicate_answer(feature.id, confidence=0.6), summary_answer("x"))
    second = submit(session, context, NIGHT_THEME)
    merge_proposal = find_pending_proposal(session, second, ProposalType.DUPLICATE)

    fake_model.queue_answers(category_answer(ui_category.id, summary="Users want a night theme."))
    review.reject_merge(session, context, merge_proposal.id)

    assert session.get(Proposal, merge_proposal.id).status is ProposalStatus.REJECTED
    category_proposal = find_pending_proposal(session, second, ProposalType.CATEGORY)
    new_feature = review.approve_suggested_category(session, context, category_proposal.id)
    assert new_feature.id != feature.id

    assert event_types_for(context, second.correlation_id)[-3:] == [
        EventType.MERGE_DECISION,
        EventType.CATEGORY_PROPOSED,
        EventType.CATEGORY_DECISION,
    ]


def test_not_a_duplicate_goes_straight_to_a_category(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    feature, _ = create_approved_feature(session, context, fake_model, DARK_MODE, ui_category)
    fake_model.queue_answers(
        DuplicateJudgment(
            duplicate_of_feature_id=None,
            confidence=0.8,
            reasoning="Different ask.",
            rejected_candidates=[RejectedCandidate(feature_id=feature.id, reason="Different.")],
        ),
        category_answer(ui_category.id),
    )
    second = submit(session, context, NIGHT_THEME)

    category_proposal = find_pending_proposal(session, second, ProposalType.CATEGORY)
    checked_candidates = category_proposal.payload["duplicate_check"]["candidates"]
    assert checked_candidates[0]["feature_id"] == feature.id
    for proposal in second.proposals:
        assert proposal.type is not ProposalType.DUPLICATE


def test_duplicate_id_the_model_was_not_shown_is_ignored(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    create_approved_feature(session, context, fake_model, DARK_MODE, ui_category)
    fake_model.queue_answers(duplicate_answer(999, confidence=0.99), category_answer(ui_category.id))

    second = submit(session, context, NIGHT_THEME)

    assert find_pending_proposal(session, second, ProposalType.CATEGORY)


# --- Fixing a feature's category ----------------------------------------------


def test_moving_a_feature_is_logged_in_its_original_trace(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    integrations_category = categories["Integrations & API"]
    feature, original = create_approved_feature(
        session, context, fake_model, DARK_MODE, integrations_category  # the mistake
    )

    feature_management.move_feature_to_existing_category(
        session, context, feature.id, ui_category.id, reason="Picked by mistake"
    )

    session.refresh(feature)
    assert feature.category_id == ui_category.id
    change_event = context.decision_log.read_events_for_correlation_id(original.correlation_id)[-1]
    assert change_event.event_type is EventType.CATEGORY_REASSIGNED
    assert change_event.payload["from_category_name"] == "Integrations & API"
    assert change_event.payload["to_category_name"] == "UI/UX & Accessibility"
    assert change_event.payload["reason"] == "Picked by mistake"
    feature_events = context.decision_log.read_events_for_feature(
        feature.id, EventType.CATEGORY_REASSIGNED
    )
    assert feature_events == [change_event]


def test_moving_a_feature_into_a_new_category(session, context, fake_model, categories):
    feature, _ = create_approved_feature(
        session, context, fake_model, ("A", "B"), categories["Integrations & API"]
    )

    feature_management.move_feature_to_new_category(
        session, context, feature.id, "Needs Clarification", "Requests too vague to act on yet."
    )

    session.refresh(feature)
    assert feature.category.name == "Needs Clarification"


def test_moving_a_feature_to_its_current_category_is_refused(session, context, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    feature, _ = create_approved_feature(session, context, fake_model, DARK_MODE, ui_category)

    with pytest.raises(InvalidDecisionError, match="already in"):
        feature_management.move_feature_to_existing_category(session, context, feature.id, ui_category.id)


# --- Taxonomy -------------------------------------------------------------------


def test_taxonomy_creates_only_the_categories_the_reviewer_kept(session, context, fake_model):
    fake_model.queue_answers(
        TaxonomyProposal(
            categories=[
                ProposedCategory(name="Notifications", description="Alerts and digests"),
                ProposedCategory(name="Reporting", description="Dashboards and exports"),
            ],
            reasoning="Two clear areas.",
        )
    )
    proposal = taxonomy.request_taxonomy_proposal(session, context, [DARK_MODE, SLACK])

    created = taxonomy.approve_taxonomy(
        session,
        context,
        proposal.id,
        [{"name": "Notifications & Alerts", "description": "Email, push and in-app alerts"}],
    )

    assert [category.name for category in created] == ["Notifications & Alerts"]
    assert session.get(Proposal, proposal.id).status is ProposalStatus.EDITED
    assert event_types_for(context, proposal.correlation_id) == [
        EventType.TAXONOMY_PROPOSED,
        EventType.TAXONOMY_DECISION,
        EventType.CATEGORY_CREATED,
    ]


def test_keeping_the_taxonomy_unchanged_is_a_plain_approval(session, context, fake_model):
    proposed = [ProposedCategory(name="Reporting", description="Dashboards and exports")]
    fake_model.queue_answers(TaxonomyProposal(categories=proposed, reasoning="."))
    proposal = taxonomy.request_taxonomy_proposal(session, context, [DARK_MODE])

    taxonomy.approve_taxonomy(
        session, context, proposal.id, [{"name": "Reporting", "description": "Dashboards and exports"}]
    )

    assert session.get(Proposal, proposal.id).status is ProposalStatus.APPROVED
    assert session.query(Category).filter_by(name="Reporting").count() == 1


def test_rejecting_the_taxonomy_creates_nothing(session, context, fake_model):
    proposed = [ProposedCategory(name="Reporting", description="Dashboards and exports")]
    fake_model.queue_answers(TaxonomyProposal(categories=proposed, reasoning="."))
    proposal = taxonomy.request_taxonomy_proposal(session, context, [DARK_MODE])

    taxonomy.reject_taxonomy(session, context, proposal.id)

    assert session.get(Proposal, proposal.id).status is ProposalStatus.REJECTED
    assert session.query(Category).count() == 0
