"""The pages render and the forms drive the same human-gated flow."""

from app.db.models import Feature, Proposal, Submission
from app.domain.enums import ProposalType
from app.pipeline.schemas import CategoryAssignment, DuplicateJudgment, SummaryUpdate


def assign(category_id):
    return CategoryAssignment(
        action="assign", category_id=category_id, new_category_name=None, new_category_description=None,
        feature_summary="Users want a dark theme.", confidence=0.9, reasoning="Theme request.",
    )


def test_pages_render(client):
    for path in ["/", "/review", "/features", "/categories"]:
        assert client.get(path).status_code == 200, path


def test_empty_submission_is_rejected(client):
    response = client.post("/submissions", data={"title": " ", "description": ""}, follow_redirects=False)
    assert response.status_code == 303
    assert "error=" in response.headers["location"]


def test_submit_review_and_merge_through_forms(client, session, llm, categories):
    ui = categories["UI/UX & Accessibility"]

    llm.queue(assign(ui.id))
    response = client.post("/submissions", data={"title": "Dark mode please",
                           "description": "The white background is harsh at night"})
    assert response.status_code == 200
    assert "Category suggestion" in response.text
    assert "UI/UX &amp; Accessibility" in response.text

    first = session.query(Submission).one()
    proposal = session.query(Proposal).filter_by(submission_id=first.id).one()
    response = client.post(f"/proposals/{proposal.id}/category",
                           data={"decision": "approve", "summary": "Users want a dark theme."})
    assert response.status_code == 200
    assert "Original (created this feature)" in response.text

    feature = session.query(Feature).one()
    llm.queue(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.94, reasoning="Same ask.",
                          rejected_candidates=[]),
        SummaryUpdate(summary="Users want a dark theme at night.", changed=True, reasoning="Adds night."),
    )
    response = client.post("/submissions", data={"title": "Night theme option",
                           "description": "The white background is harsh at night, dark theme please"})
    assert "Duplicate detected" in response.text
    assert "Users want a dark theme at night." in response.text

    session.expire_all()
    duplicate = session.query(Proposal).filter_by(type=ProposalType.DUPLICATE).one()
    response = client.post(f"/proposals/{duplicate.id}/merge",
                           data={"action": "approve", "summary": "Users want a dark theme at night."})
    assert response.status_code == 200
    assert "2 submissions" in response.text
    assert "LLM confidence 0.94" in response.text


def test_reject_without_category_shows_error(client, session, llm, categories):
    llm.queue(assign(categories["UI/UX & Accessibility"].id))
    client.post("/submissions", data={"title": "Dark mode", "description": "Please add it"})
    proposal = session.query(Proposal).one()

    response = client.post(f"/proposals/{proposal.id}/category", data={"decision": "reject"})

    assert "requires: category_id" in response.text
    session.expire_all()
    assert session.query(Submission).one().feature_id is None
