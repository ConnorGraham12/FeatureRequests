from sqlalchemy import select

from app.db.models import Category, Feature, Proposal, Submission
from app.domain.enums import CategoryStatus, ProposalStatus, ProposalType, SubmissionStatus
from app.observability.correlation import generate_correlation_id


def test_health_check_returns_a_new_correlation_id(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert len(response.headers["X-Correlation-ID"]) == 36


def test_valid_incoming_correlation_id_is_kept(client):
    correlation_id = generate_correlation_id()
    response = client.get("/health", headers={"X-Correlation-ID": correlation_id})
    assert response.headers["X-Correlation-ID"] == correlation_id


def test_invalid_incoming_correlation_id_is_replaced(client):
    response = client.get("/health", headers={"X-Correlation-ID": "not-a-uuid"})
    assert response.headers["X-Correlation-ID"] != "not-a-uuid"


def test_merged_feature_keeps_every_submission(session):
    category = Category(
        name="UI/UX & Accessibility", description="Themes, layout", status=CategoryStatus.ACTIVE
    )
    feature = Feature(
        title="Dark mode for the web app", summary="Users want a dark theme.", category=category
    )
    original = Submission(
        correlation_id=generate_correlation_id(),
        title="Dark mode please",
        description="The white background is harsh at night.",
        status=SubmissionStatus.NEW_FEATURE,
        feature=feature,
    )
    duplicate = Submission(
        correlation_id=generate_correlation_id(),
        title="Night theme option",
        description="Would love a dark theme for the dashboard.",
        status=SubmissionStatus.MERGED,
        feature=feature,
    )
    session.add_all([original, duplicate])
    session.flush()
    session.add(
        Proposal(
            correlation_id=duplicate.correlation_id,
            type=ProposalType.DUPLICATE,
            submission_id=duplicate.id,
            feature_id=feature.id,
            payload={"duplicate_of": feature.id},
            confidence=0.94,
        )
    )
    session.commit()
    session.expire_all()

    stored_feature = session.scalars(select(Feature)).one()
    titles = [submission.title for submission in stored_feature.submissions]
    assert titles == ["Dark mode please", "Night theme option"]
    assert stored_feature.category.name == "UI/UX & Accessibility"

    stored_proposal = session.scalars(select(Proposal)).one()
    assert stored_proposal.status is ProposalStatus.PENDING
    assert stored_proposal.payload == {"duplicate_of": stored_feature.id}
