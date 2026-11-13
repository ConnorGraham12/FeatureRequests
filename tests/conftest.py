import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.models import Category
from app.domain.enums import CategoryStatus
from app.main import build_context, create_app
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        decision_log_path=tmp_path / "decisions.jsonl",
        duplicate_similarity_threshold=0.5,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings, llm):
    with TestClient(create_app(settings, embedder=FakeEmbedder(), llm=llm)) as c:
        yield c


@pytest.fixture
def session(client):
    with client.app.state.session_factory() as s:
        yield s


@pytest.fixture
def ctx(client):
    return client.app.state.triage


@pytest.fixture
def categories(session):
    """Two active categories, returned by name."""
    items = [
        Category(name="UI/UX & Accessibility", description="Themes, layout, accessibility", status=CategoryStatus.ACTIVE),
        Category(name="Integrations & API", description="Slack, GitHub, webhooks, API", status=CategoryStatus.ACTIVE),
    ]
    session.add_all(items)
    session.commit()
    return {c.name: c for c in items}


@pytest.fixture
def standalone_ctx(settings):
    """A context for tests that don't need the web app."""
    return build_context(settings, FakeEmbedder(), FakeLLM())
