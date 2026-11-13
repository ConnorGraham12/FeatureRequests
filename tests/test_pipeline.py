"""End-to-end triage flows against a fake LLM and embedder."""

import pytest

from app.db.models import Category, Feature, Proposal, Submission
from app.domain.enums import Decision, ProposalStatus, ProposalType, SubmissionStatus
from app.domain.proposals import InvalidDecisionError
from app.observability.correlation import correlation_scope, new_correlation_id
from app.observability.decision_log import EventType
from app.pipeline.schemas import (
    CategoryAssignment,
    DuplicateJudgment,
    ProposedCategory,
    RejectedCandidate,
    SummaryUpdate,
    TaxonomyProposal,
)
from app.services import review, taxonomy, triage

DARK_MODE = ("Dark mode please", "The white background is harsh at night, can we get a dark theme")
NIGHT_THEME = ("Night theme option", "Can we get a dark theme, the white background is harsh at night")
SLACK = ("Create tasks from Slack", "Turn a Slack message into a task with one click")


def assign(category_id, summary="Users want this.", confidence=0.9):
    return CategoryAssignment(
        action="assign", category_id=category_id, new_category_name=None,
        new_category_description=None, feature_summary=summary,
        confidence=confidence, reasoning="Fits the category.",
    )


def submit(session, ctx, request):
    with correlation_scope(new_correlation_id()):
        return triage.submit(session, ctx, *request)


def pending(session, submission, type_):
    return next(
        p for p in session.query(Proposal).filter_by(submission_id=submission.id, type=type_)
        if p.status is ProposalStatus.PENDING
    )


def make_feature(session, ctx, llm, request, category):
    llm.queue(assign(category.id, summary="Users want a dark theme."))
    submission = submit(session, ctx, request)
    proposal = pending(session, submission, ProposalType.CATEGORY)
    return review.decide_category(session, ctx, proposal.id, Decision.APPROVE), submission


def test_new_submission_gets_category_proposal_and_is_not_saved_until_approved(session, ctx, llm, categories):
    ui = categories["UI/UX & Accessibility"]
    llm.queue(assign(ui.id))

    submission = submit(session, ctx, DARK_MODE)

    assert submission.status is SubmissionStatus.PENDING
    assert submission.feature_id is None
    proposal = pending(session, submission, ProposalType.CATEGORY)
    assert proposal.payload["category_id"] == ui.id

    feature = review.decide_category(session, ctx, proposal.id, Decision.APPROVE)
    assert feature.category_id == ui.id
    assert feature.summary == "Users want this."
    assert submission.status is SubmissionStatus.NEW_FEATURE

    events = ctx.log.read_correlation(submission.correlation_id)
    assert [e.event_type for e in events] == [
        EventType.SUBMISSION_RECEIVED,
        EventType.DUPLICATE_CANDIDATES,
        EventType.CATEGORY_PROPOSED,
        EventType.CATEGORY_DECISION,
    ]
    assert events[-1].parent_event_id == events[-2].event_id
    assert events[-1].actor.reviewer == "pm"


def test_rejected_category_is_saved_with_reviewers_choice(session, ctx, llm, categories):
    ui, integrations = categories["UI/UX & Accessibility"], categories["Integrations & API"]
    llm.queue(assign(ui.id))
    submission = submit(session, ctx, SLACK)
    proposal = pending(session, submission, ProposalType.CATEGORY)

    with pytest.raises(InvalidDecisionError):
        review.decide_category(session, ctx, proposal.id, Decision.REJECT)
    session.rollback()

    feature = review.decide_category(
        session, ctx, proposal.id, Decision.REJECT, category_id=integrations.id
    )
    assert feature.category_id == integrations.id
    assert session.get(Proposal, proposal.id).status is ProposalStatus.REJECTED


def test_duplicate_is_merged_with_approved_summary(session, ctx, llm, categories):
    feature, original = make_feature(session, ctx, llm, DARK_MODE, categories["UI/UX & Accessibility"])
    llm.queue(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.94,
                          reasoning="Same ask.", rejected_candidates=[]),
        SummaryUpdate(summary="Users want a dark theme for night use.", changed=True, reasoning="Adds night use."),
    )

    second = submit(session, ctx, NIGHT_THEME)

    duplicate = pending(session, second, ProposalType.DUPLICATE)
    assert duplicate.feature_id == feature.id
    assert duplicate.payload["candidates"][0]["feature_id"] == feature.id
    assert second.feature_id is None  # not merged until a human approves

    review.decide_merge(session, ctx, duplicate.id, approve=True)

    session.refresh(feature)
    assert [s.id for s in feature.submissions] == [original.id, second.id]
    assert second.status is SubmissionStatus.MERGED
    assert feature.summary == "Users want a dark theme for night use."

    events = ctx.log.read_correlation(second.correlation_id)
    assert [e.event_type for e in events] == [
        EventType.SUBMISSION_RECEIVED,
        EventType.DUPLICATE_CANDIDATES,
        EventType.DUPLICATE_JUDGMENT,
        EventType.SUMMARY_PROPOSED,
        EventType.MERGE_DECISION,
    ]
    merge = events[-1]
    assert merge.parent_event_id == events[2].event_id
    assert merge.payload["summary_decision"] == "approve"


def test_edited_summary_is_used_and_logged(session, ctx, llm, categories):
    feature, _ = make_feature(session, ctx, llm, DARK_MODE, categories["UI/UX & Accessibility"])
    llm.queue(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.9, reasoning="Same.", rejected_candidates=[]),
        SummaryUpdate(summary="AI summary.", changed=True, reasoning="."),
    )
    second = submit(session, ctx, NIGHT_THEME)
    duplicate = pending(session, second, ProposalType.DUPLICATE)

    review.decide_merge(session, ctx, duplicate.id, approve=True, edited_summary="My summary.")

    session.refresh(feature)
    assert feature.summary == "My summary."
    merge = ctx.log.read_correlation(second.correlation_id)[-1]
    assert merge.payload["summary_decision"] == "edit"
    assert merge.payload["summary"] == "My summary."


def test_rejected_merge_runs_categorization(session, ctx, llm, categories):
    ui = categories["UI/UX & Accessibility"]
    feature, _ = make_feature(session, ctx, llm, DARK_MODE, ui)
    llm.queue(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.6, reasoning="Maybe.", rejected_candidates=[]),
        SummaryUpdate(summary="x", changed=True, reasoning="."),
    )
    second = submit(session, ctx, NIGHT_THEME)
    duplicate = pending(session, second, ProposalType.DUPLICATE)

    llm.queue(assign(ui.id, summary="Users want a night theme."))
    review.decide_merge(session, ctx, duplicate.id, approve=False)

    assert session.get(Proposal, duplicate.id).status is ProposalStatus.REJECTED
    category = pending(session, second, ProposalType.CATEGORY)
    new_feature = review.decide_category(session, ctx, category.id, Decision.APPROVE)
    assert new_feature.id != feature.id

    types = [e.event_type for e in ctx.log.read_correlation(second.correlation_id)]
    assert types[-3:] == [EventType.MERGE_DECISION, EventType.CATEGORY_PROPOSED, EventType.CATEGORY_DECISION]


def test_not_duplicate_verdict_goes_straight_to_categorization(session, ctx, llm, categories):
    ui = categories["UI/UX & Accessibility"]
    feature, _ = make_feature(session, ctx, llm, DARK_MODE, ui)
    llm.queue(
        DuplicateJudgment(
            duplicate_of_feature_id=None, confidence=0.8, reasoning="Different ask.",
            rejected_candidates=[RejectedCandidate(feature_id=feature.id, reason="Different.")],
        ),
        assign(ui.id),
    )
    second = submit(session, ctx, NIGHT_THEME)

    proposal = pending(session, second, ProposalType.CATEGORY)
    assert proposal.payload["duplicate_check"]["candidates"][0]["feature_id"] == feature.id
    assert not [p for p in second.proposals if p.type is ProposalType.DUPLICATE]


def test_hallucinated_duplicate_id_is_ignored(session, ctx, llm, categories):
    ui = categories["UI/UX & Accessibility"]
    make_feature(session, ctx, llm, DARK_MODE, ui)
    llm.queue(
        DuplicateJudgment(duplicate_of_feature_id=999, confidence=0.99, reasoning="?", rejected_candidates=[]),
        assign(ui.id),
    )
    second = submit(session, ctx, NIGHT_THEME)
    assert pending(session, second, ProposalType.CATEGORY)


def test_new_category_proposal_creates_category_on_approval(session, ctx, llm):
    llm.queue(CategoryAssignment(
        action="propose_new", category_id=None, new_category_name="Automation",
        new_category_description="Rules and recurring work.", feature_summary="Users want recurring tasks.",
        confidence=0.7, reasoning="No categories exist.",
    ))
    submission = submit(session, ctx, ("Recurring tasks", "Repeat a task every week"))
    proposal = pending(session, submission, ProposalType.NEW_CATEGORY)

    feature = review.decide_category(session, ctx, proposal.id, Decision.APPROVE)

    assert feature.category.name == "Automation"
    types = [e.event_type for e in ctx.log.read_correlation(submission.correlation_id)]
    assert EventType.CATEGORY_CREATED in types


def test_llm_failure_is_logged_and_allows_manual_assignment(session, ctx, llm, categories):
    llm.fail_next(CategoryAssignment, "API down")
    submission = submit(session, ctx, SLACK)

    assert not submission.proposals
    events = ctx.log.read_correlation(submission.correlation_id)
    assert events[-1].event_type is EventType.PIPELINE_ERROR
    assert events[-1].payload["stage"] == "category.proposed"

    feature = review.assign_manually(session, ctx, submission.id, categories["Integrations & API"].id)
    assert feature.category.name == "Integrations & API"


def test_decided_submission_cannot_be_decided_again(session, ctx, llm, categories):
    feature, submission = make_feature(session, ctx, llm, DARK_MODE, categories["UI/UX & Accessibility"])
    proposal = next(p for p in submission.proposals if p.type is ProposalType.CATEGORY)
    with pytest.raises(InvalidDecisionError):
        review.decide_category(session, ctx, proposal.id, Decision.APPROVE)


def test_taxonomy_edits_create_only_kept_categories(session, ctx, llm):
    llm.queue(TaxonomyProposal(
        categories=[
            ProposedCategory(name="Notifications", description="Alerts and digests"),
            ProposedCategory(name="Reporting", description="Dashboards and exports"),
        ],
        reasoning="Two clear areas.",
    ))
    proposal = taxonomy.propose(session, ctx, [DARK_MODE, SLACK])

    created = taxonomy.decide(
        session, ctx, proposal.id, Decision.EDIT,
        [{"name": "Notifications & Alerts", "description": "Email, push and in-app alerts"}],
    )

    assert [c.name for c in created] == ["Notifications & Alerts"]
    assert session.get(Proposal, proposal.id).status is ProposalStatus.EDITED
    types = [e.event_type for e in ctx.log.read_correlation(proposal.correlation_id)]
    assert types == [EventType.TAXONOMY_PROPOSED, EventType.TAXONOMY_DECISION, EventType.CATEGORY_CREATED]


def test_unchanged_taxonomy_counts_as_approval(session, ctx, llm):
    categories = [ProposedCategory(name="Reporting", description="Dashboards and exports")]
    llm.queue(TaxonomyProposal(categories=categories, reasoning="."))
    proposal = taxonomy.propose(session, ctx, [DARK_MODE])

    taxonomy.decide(session, ctx, proposal.id, Decision.EDIT, [c.model_dump() for c in categories])

    assert session.get(Proposal, proposal.id).status is ProposalStatus.APPROVED
    assert session.query(Category).filter_by(name="Reporting").count() == 1
