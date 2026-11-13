from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request

from app.ai.embeddings import Embedder, get_shared_embedder
from app.ai.llm import ClaudeClient, LanguageModel
from app.config import Settings, get_settings
from app.db.session import create_database_engine, create_session_factory, create_tables
from app.observability.correlation import (
    bind_correlation_id,
    generate_correlation_id,
    is_valid_correlation_id,
)
from app.observability.decision_log import DecisionLogger
from app.services.context import TriageContext
from app.web.routes import router

CORRELATION_HEADER = "X-Correlation-ID"

load_dotenv()


def build_triage_context(
    settings: Settings,
    embedder: Embedder | None = None,
    language_model: LanguageModel | None = None,
) -> TriageContext:
    """Build the services' shared context. Tests pass a fake embedder and model."""
    if embedder is None:
        embedder = get_shared_embedder(settings.embedding_model)
    if language_model is None:
        language_model = ClaudeClient(settings)

    decision_log = DecisionLogger(
        settings.decision_log_path, include_debug_details=settings.decision_log_verbose
    )
    return TriageContext(
        settings=settings,
        decision_log=decision_log,
        embedder=embedder,
        language_model=language_model,
    )


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    language_model: LanguageModel | None = None,
) -> FastAPI:
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def set_up_and_tear_down(app: FastAPI) -> AsyncIterator[None]:
        engine = create_database_engine(settings.database_url)
        create_tables(engine)
        app.state.settings = settings
        app.state.session_factory = create_session_factory(engine)
        app.state.triage = build_triage_context(settings, embedder, language_model)
        yield
        engine.dispose()

    app = FastAPI(title="Feature Request Triage", lifespan=set_up_and_tear_down)
    app.include_router(router)

    @app.middleware("http")
    async def bind_correlation_id_for_request(request: Request, call_next):
        """Use the caller's correlation ID if it sent a valid one, otherwise start one.

        A new submission adopts this ID. Decisions on an existing submission
        bind that submission's stored ID instead, inside the service layer.
        """
        incoming_id = request.headers.get(CORRELATION_HEADER, "")
        if is_valid_correlation_id(incoming_id):
            correlation_id = incoming_id
        else:
            correlation_id = generate_correlation_id()

        with bind_correlation_id(correlation_id):
            response = await call_next(request)
        response.headers[CORRELATION_HEADER] = correlation_id
        return response

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
