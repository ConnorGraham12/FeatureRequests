"""The pages render, and the forms drive the same human-gated flow."""

from app.db.models import Feature, Proposal, Submission
from app.domain.enums import ProposalType
from app.pipeline.schemas import CategoryAssignment, DuplicateJudgment, SummaryUpdate


def category_answer(category_id):
    return CategoryAssignment(
        action="assign",
        category_id=category_id,
        new_category_name=None,
        new_category_description=None,
        feature_summary="Users want a dark theme.",
        confidence=0.9,
        reasoning="Theme request.",
    )


def test_every_page_renders(client):
    for path in ["/", "/review", "/features", "/categories"]:
        assert client.get(path).status_code == 200, path


def test_empty_submission_is_refused_with_a_message(client):
    response = client.post(
        "/submissions", data={"title": " ", "description": ""}, follow_redirects=False
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]


def test_submit_review_and_merge_through_the_forms(client, session, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]

    fake_model.queue_answers(category_answer(ui_category.id))
    response = client.post(
        "/submissions",
        data={"title": "Dark mode please", "description": "The white background is harsh at night"},
    )
    assert response.status_code == 200
    assert "Category suggestion" in response.text
    assert "UI/UX &amp; Accessibility" in response.text

    first_submission = session.query(Submission).one()
    category_proposal = session.query(Proposal).filter_by(submission_id=first_submission.id).one()
    response = client.post(
        f"/proposals/{category_proposal.id}/category",
        data={"decision": "approve", "summary": "Users want a dark theme."},
    )
    assert response.status_code == 200
    assert "Original (created this feature)" in response.text

    feature = session.query(Feature).one()
    fake_model.queue_answers(
        DuplicateJudgment(
            duplicate_of_feature_id=feature.id,
            confidence=0.94,
            reasoning="Same ask.",
            rejected_candidates=[],
        ),
        SummaryUpdate(summary="Users want a dark theme at night.", changed=True, reasoning="Adds night."),
    )
    response = client.post(
        "/submissions",
        data={
            "title": "Night theme option",
            "description": "The white background is harsh at night, dark theme please",
        },
    )
    assert "Duplicate detected" in response.text
    assert "Users want a dark theme at night." in response.text

    session.expire_all()
    merge_proposal = session.query(Proposal).filter_by(type=ProposalType.DUPLICATE).one()
    response = client.post(
        f"/proposals/{merge_proposal.id}/merge",
        data={"action": "approve", "summary": "Users want a dark theme at night."},
    )
    assert response.status_code == 200
    assert "2 submissions" in response.text
    assert "LLM confidence 0.94" in response.text


def test_rejecting_without_a_category_shows_an_error(client, session, fake_model, categories):
    fake_model.queue_answers(category_answer(categories["UI/UX & Accessibility"].id))
    client.post("/submissions", data={"title": "Dark mode", "description": "Please add it"})
    proposal = session.query(Proposal).one()

    response = client.post(f"/proposals/{proposal.id}/category", data={"decision": "reject"})

    assert "Choose a category" in response.text
    session.expire_all()
    assert session.query(Submission).one().feature_id is None


def test_change_category_through_the_feature_page(client, session, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    integrations_category = categories["Integrations & API"]
    fake_model.queue_answers(category_answer(ui_category.id))
    client.post("/submissions", data={"title": "Dark mode", "description": "Please add it"})
    proposal = session.query(Proposal).one()
    client.post(
        f"/proposals/{proposal.id}/category",
        data={"decision": "reject", "category_id": str(integrations_category.id)},
    )
    feature = session.query(Feature).one()

    page = client.get(f"/features/{feature.id}")
    assert "Change category" in page.text

    response = client.post(
        f"/features/{feature.id}/category",
        data={"category_id": str(ui_category.id), "reason": "Mis-click"},
    )
    assert response.status_code == 200
    assert "Integrations &amp; API → UI/UX &amp; Accessibility" in response.text
    assert '"Mis-click"' in response.text


def test_change_category_needs_exactly_one_target(client, session, fake_model, categories):
    ui_category = categories["UI/UX & Accessibility"]
    fake_model.queue_answers(category_answer(ui_category.id))
    client.post("/submissions", data={"title": "Dark mode", "description": "Please add it"})
    proposal = session.query(Proposal).one()
    client.post(f"/proposals/{proposal.id}/category", data={"decision": "approve"})
    feature = session.query(Feature).one()

    neither = client.post(f"/features/{feature.id}/category", data={})
    both = client.post(
        f"/features/{feature.id}/category",
        data={"category_id": str(ui_category.id), "new_name": "X", "new_description": "Y"},
    )

    assert "Choose an existing category or name a new one" in neither.text
    assert "not both" in both.text


def create_feature_through_forms(client, session, fake_model, category, title="Dark mode"):
    fake_model.queue_answers(category_answer(category.id))
    client.post("/submissions", data={"title": title, "description": "Please add it"})
    session.expire_all()
    proposal = session.query(Proposal).order_by(Proposal.id.desc()).first()
    client.post(f"/proposals/{proposal.id}/category", data={"decision": "approve"})
    session.expire_all()
    return session.query(Feature).order_by(Feature.id.desc()).first()


def test_feature_toolbar_edit_delete_and_restore(client, session, fake_model, categories):
    feature = create_feature_through_forms(client, session, fake_model, categories["UI/UX & Accessibility"])

    page = client.get(f"/features/{feature.id}")
    assert 'aria-label="Feature actions"' in page.text
    for label in [">Edit<", ">Change category<", ">Delete<"]:
        assert label in page.text

    edited = client.post(f"/features/{feature.id}/edit", data={"title": "Dark theme", "summary": "A dark theme."})
    assert "Feature updated." in edited.text
    assert "<h1>Dark theme</h1>" in edited.text

    deleted = client.post(f"/features/{feature.id}/archive")
    assert "Restore feature" in deleted.text
    assert "Dark theme" not in client.get("/features").text
    assert "Dark theme" in client.get("/features?archived=1").text

    restored = client.post(f"/features/{feature.id}/restore")
    assert "Feature restored." in restored.text
    assert "Dark theme" in client.get("/features").text


def test_merged_submission_toolbar_unmerge(client, session, fake_model, categories):
    feature = create_feature_through_forms(client, session, fake_model, categories["UI/UX & Accessibility"])
    fake_model.queue_answers(
        DuplicateJudgment(duplicate_of_feature_id=feature.id, confidence=0.9, reasoning=".", rejected_candidates=[]),
        SummaryUpdate(summary="Users want a dark theme.", changed=False, reasoning="."),
    )
    client.post("/submissions", data={"title": "Dark mode again", "description": "Please add it"})
    session.expire_all()
    merge = session.query(Proposal).filter_by(type=ProposalType.DUPLICATE).one()
    client.post(f"/proposals/{merge.id}/merge", data={"action": "approve"})

    page = client.get(f"/features/{feature.id}")
    assert page.text.count(">Unmerge<") == 2
    assert page.text.count(">Move to…<") == 2

    duplicate_id = merge.submission_id
    response = client.post(f"/submissions/{duplicate_id}/unmerge")
    assert "is now its own feature" in response.text
    assert ">Unmerge<" not in response.text  # one submission left, so no submission actions
    assert "The only submission in this feature" in response.text
