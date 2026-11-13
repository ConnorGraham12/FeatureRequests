import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.models import Category
from app.domain.enums import CategoryStatus
from app.main import create_app
from tests.fakes import FakeEmbedder, FakeLanguageModel


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        decision_log_path=tmp_path / "decisions.jsonl",
        duplicate_similarity_threshold=0.5,
    )


@pytest.fixture
def fake_model() -> FakeLanguageModel:
    return FakeLanguageModel()


@pytest.fixture
def client(settings, fake_model):
    app = create_app(settings, embedder=FakeEmbedder(), language_model=fake_model)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session(client):
    with client.app.state.session_factory() as db_session:
        yield db_session


@pytest.fixture
def context(client):
    return client.app.state.triage


@pytest.fixture
def categories(session):
    """Two active categories, keyed by name."""
    ui_category = Category(
        name="UI/UX & Accessibility",
        description="Themes, layout, accessibility",
        status=CategoryStatus.ACTIVE,
    )
    integrations_category = Category(
        name="Integrations & API",
        description="Slack, GitHub, webhooks, API",
        status=CategoryStatus.ACTIVE,
    )
    session.add_all([ui_category, integrations_category])
    session.commit()
    return {
        ui_category.name: ui_category,
        integrations_category.name: integrations_category,
    }
