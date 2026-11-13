"""Test doubles: deterministic embeddings and a scripted language model."""

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np

from app.ai.llm import LLMError, LLMResult

EMBEDDING_SIZE = 256


class FakeEmbedder:
    """A hashed bag of words: texts that share words get a high cosine similarity."""

    model_name = "fake-bag-of-words"

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_SIZE), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vectors[row, word_hash % EMBEDDING_SIZE] += 1.0
            length = np.linalg.norm(vectors[row])
            if length:
                vectors[row] /= length
        return vectors


class FakeLanguageModel:
    """Returns queued answers, one queue per output type.

    Queue an LLMError (or use queue_failure) to make the next call for that
    output type fail.
    """

    def __init__(self) -> None:
        self.queued_answers: dict[type, deque] = defaultdict(deque)
        self.calls: list[tuple[type, str]] = []

    def queue_answers(self, *answers) -> "FakeLanguageModel":
        for answer in answers:
            self.queued_answers[type(answer)].append(answer)
        return self

    def queue_failure(self, output_type: type, message: str = "boom") -> "FakeLanguageModel":
        self.queued_answers[output_type].append(LLMError(message))
        return self

    def request_structured_output(self, *, system_prompt: str, user_message: str, output_type: type):
        self.calls.append((output_type, user_message))
        if not self.queued_answers[output_type]:
            raise AssertionError(f"No queued answer for {output_type.__name__}")

        answer = self.queued_answers[output_type].popleft()
        if isinstance(answer, LLMError):
            raise answer
        return LLMResult(
            output=answer,
            model="fake-model",
            latency_ms=5,
            input_tokens=100,
            output_tokens=20,
            request_id="req_test",
            fallback_used=False,
            raw_response="{}",
        )
