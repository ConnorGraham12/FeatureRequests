"""Wraps the app's ClaudeClient for eval runs.

Adds what the eval needs on top of the app's client:
  - retries with jittered backoff on rate limits and server errors, counted
  - a wall-clock deadline per case
  - a check that the model that answered is the model that was asked for
  - a record of every call, so the runner can save transcripts and token usage
"""

import random
import time
from dataclasses import dataclass, field

import anthropic

from app.ai.llm import ClaudeClient, LLMError, LLMOutputError, LLMRefusalError, LLMResult
from app.config import Settings

MAX_ATTEMPTS_PER_CALL = 5
MAX_BACKOFF_SECONDS = 30.0

RETRYABLE_API_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)


class HarnessError(Exception):
    """A failure that produced nothing to score. Recorded in errors.jsonl, never as a result.

    failure_class is one of: "timeout", "serving_error", "model_mismatch", "harness_error".
    """

    def __init__(self, failure_class: str, message: str, attempts: int, result: LLMResult | None = None):
        super().__init__(message)
        self.failure_class = failure_class
        self.attempts = attempts
        self.result = result


@dataclass
class RecordedCall:
    system_prompt: str
    user_message: str
    result: LLMResult
    attempts: int


def create_claude_client_for_eval(settings: Settings, timeout_seconds: float) -> ClaudeClient:
    """The app's ClaudeClient with the SDK's own retries turned off,
    so RecordingLanguageModel can count every attempt."""
    api_client = anthropic.Anthropic(max_retries=0, timeout=timeout_seconds)
    return ClaudeClient(settings, api_client=api_client)


def wait_before_retry(attempt_number: int) -> None:
    backoff_seconds = min(MAX_BACKOFF_SECONDS, 2**attempt_number)
    jitter = random.uniform(0.5, 1.0)
    time.sleep(backoff_seconds * jitter)


@dataclass
class RecordingLanguageModel:
    client: ClaudeClient
    expected_model: str
    deadline: float  # a time.monotonic() value
    calls: list[RecordedCall] = field(default_factory=list)

    def request_structured_output(self, *, system_prompt: str, user_message: str, output_type: type):
        for attempt_number in range(1, MAX_ATTEMPTS_PER_CALL + 1):
            if time.monotonic() > self.deadline:
                raise HarnessError("timeout", "case wall-clock ceiling reached", attempt_number - 1)

            try:
                result = self.client.request_structured_output(
                    system_prompt=system_prompt, user_message=user_message, output_type=output_type
                )
            except (LLMRefusalError, LLMOutputError):
                # Refusals and truncation are graded outcomes; the runner handles them.
                raise
            except LLMError as error:
                is_retryable = isinstance(error.__cause__, RETRYABLE_API_ERRORS)
                is_last_attempt = attempt_number == MAX_ATTEMPTS_PER_CALL
                if not is_retryable or is_last_attempt:
                    raise HarnessError("serving_error", str(error), attempt_number) from error
                wait_before_retry(attempt_number)
                continue

            if not result.model.startswith(self.expected_model):
                raise HarnessError(
                    "model_mismatch",
                    f"asked for {self.expected_model}, served by {result.model}",
                    attempt_number,
                    result,
                )

            self.calls.append(RecordedCall(system_prompt, user_message, result, attempt_number))
            return result

        raise AssertionError("unreachable: the loop always returns or raises")
