"""Thin wrapper over the Claude API for structured (Pydantic) outputs.

Every pipeline stage calls `structured()` and gets back the parsed output plus
the metadata the decision log needs: model, latency, tokens, request ID.
"""

import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from app.config import Settings

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Any failure to get a usable answer from the model."""


class LLMRefusalError(LLMError):
    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"Model refused (category={category}): {explanation}")
        self.category = category
        self.explanation = explanation


class LLMOutputError(LLMError):
    """The response had no usable structured output (e.g. hit max_tokens)."""


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    output: T
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    request_id: str | None
    fallback_used: bool
    raw_response: str


class LLM(Protocol):
    def structured(self, *, system: str, user: str, output_type: type[T]) -> LLMResult[T]: ...


class ClaudeClient:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        # Created on first use, so the app starts without credentials. They
        # resolve from the environment (ANTHROPIC_API_KEY or an `ant auth
        # login` profile). Retries on 429/5xx are handled by the SDK.
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()
            except anthropic.AnthropicError as error:
                raise LLMError(f"Claude API client unavailable: {error}") from error
        return self._client

    def structured(self, *, system: str, user: str, output_type: type[T]) -> LLMResult[T]:
        kwargs: dict = {
            "model": self.settings.llm_model,
            "max_tokens": self.settings.llm_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": output_type,
            "output_config": {"effort": self.settings.llm_effort},
        }
        if self.settings.llm_fallbacks_enabled:
            # On a safety-classifier refusal, the API re-runs the request on a
            # fallback model chosen by refusal category.
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        started = time.perf_counter()
        try:
            response = self.client.beta.messages.parse(**kwargs)
        except anthropic.APIError as error:
            # Already retried by the SDK where retryable; the caller degrades
            # to manual review instead of failing the request.
            raise LLMError(f"{type(error).__name__}: {error}") from error
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.stop_reason == "refusal":
            details = response.stop_details
            raise LLMRefusalError(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )
        if response.stop_reason == "max_tokens":
            raise LLMOutputError("Response truncated at max_tokens")
        if response.parsed_output is None:
            raise LLMOutputError("Response contained no structured output")

        iterations = getattr(response.usage, "iterations", None) or []
        return LLMResult(
            output=response.parsed_output,
            model=response.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=response._request_id,
            fallback_used=any(it.type == "fallback_message" for it in iterations),
            raw_response=response.to_json(),
        )
