"""A small wrapper over the Claude API that returns structured (Pydantic) output.

Every pipeline stage calls `request_structured_output()` and gets back the
parsed answer plus what the decision log needs: model, latency, tokens and the
API request ID.
"""

import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from app.config import Settings

OutputModel = TypeVar("OutputModel", bound=BaseModel)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Any failure to get a usable answer from the model."""


class LLMRefusalError(LLMError):
    """The model declined to answer."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"Model refused (category={category}): {explanation}")
        self.category = category
        self.explanation = explanation


class LLMOutputError(LLMError):
    """The response had no usable structured output, e.g. it hit max_tokens."""


@dataclass(frozen=True)
class LLMResult(Generic[OutputModel]):
    output: OutputModel
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    request_id: str | None
    fallback_used: bool
    raw_response: str
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LanguageModel(Protocol):
    """Anything that can answer a prompt with structured output. Tests use a fake."""

    def request_structured_output(
        self, *, system_prompt: str, user_message: str, output_type: type[OutputModel]
    ) -> LLMResult[OutputModel]: ...


class ClaudeClient:
    def __init__(self, settings: Settings, api_client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings
        self._api_client = api_client

    def _get_or_create_api_client(self) -> anthropic.Anthropic:
        """Create the API client on first use, so the app can start without credentials.

        Credentials come from the environment: ANTHROPIC_API_KEY, or an
        `ant auth login` profile. The SDK retries 429 and 5xx errors itself.
        """
        if self._api_client is None:
            try:
                self._api_client = anthropic.Anthropic()
            except anthropic.AnthropicError as error:
                raise LLMError(f"Claude API client unavailable: {error}") from error
        return self._api_client

    def request_structured_output(
        self, *, system_prompt: str, user_message: str, output_type: type[OutputModel]
    ) -> LLMResult[OutputModel]:
        request_arguments: dict = {
            "model": self.settings.llm_model,
            "max_tokens": self.settings.llm_max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "output_format": output_type,
            "output_config": {"effort": self.settings.llm_effort},
        }
        if self.settings.llm_fallbacks_enabled:
            # If a safety classifier refuses, the API re-runs the request on a
            # fallback model chosen by the refusal category.
            request_arguments["betas"] = [REFUSAL_FALLBACK_BETA]
            request_arguments["fallbacks"] = "default"

        api_client = self._get_or_create_api_client()
        started_at = time.perf_counter()
        try:
            response = api_client.beta.messages.parse(**request_arguments)
        except anthropic.APIError as error:
            # The SDK has already retried where that makes sense. The caller
            # falls back to manual review instead of failing the request.
            raise LLMError(f"{type(error).__name__}: {error}") from error
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None)
            raise LLMRefusalError(category, explanation)
        if response.stop_reason == "max_tokens":
            raise LLMOutputError("Response was cut off at max_tokens")
        if response.parsed_output is None:
            raise LLMOutputError("Response contained no structured output")

        return LLMResult(
            output=response.parsed_output,
            model=response.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=response._request_id,
            fallback_used=_response_used_a_fallback_model(response),
            raw_response=response.to_json(),
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None) or 0,
            cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", None) or 0,
        )


def _response_used_a_fallback_model(response) -> bool:
    iterations = getattr(response.usage, "iterations", None) or []
    for iteration in iterations:
        if iteration.type == "fallback_message":
            return True
    return False
