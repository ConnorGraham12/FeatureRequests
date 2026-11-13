from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.ai.embeddings import Embedder, get_embedder
from app.ai.llm import LLM, ClaudeClient
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.observability.correlation import (
    correlation_scope,
    is_valid_correlation_id,
    new_correlation_id,
)
from app.observability.decision_log import DecisionLogger
from app.services.context import TriageContext
from app.web.routes import router

CORRELATION_HEADER = "X-Correlation-ID"


def build_context(
    settings: Settings, embedder: Embedder | None = None, llm: LLM | None = None
) -> TriageContext:
    return TriageContext(
        settings=settings,
        log=DecisionLogger(settings.decision_log_path, verbose=settings.decision_log_verbose),
        embedder=embedder or get_embedder(settings.embedding_model),
        llm=llm or ClaudeClient(settings),
    )


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake `embedder` and `llm`."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        init_db(engine)
        app.state.settings = settings
        app.state.session_factory = make_session_factory(engine)
        app.state.triage = build_context(settings, embedder, llm)
        yield
        engine.dispose()

    app = FastAPI(title="Feature Request Triage", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        # Accept a caller-supplied ID or start one. A new submission adopts
        # this ID; decisions on an existing submission re-bind its stored ID.
        incoming = request.headers.get(CORRELATION_HEADER, "")
        correlation_id = incoming if is_valid_correlation_id(incoming) else new_correlation_id()
        with correlation_scope(correlation_id):
            response = await call_next(request)
        response.headers[CORRELATION_HEADER] = correlation_id
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
