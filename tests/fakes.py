"""Test doubles: deterministic embeddings and a scripted LLM."""

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np

from app.ai.llm import LLMError, LLMResult

DIM = 256


class FakeEmbedder:
    """Hashed bag of words: texts sharing words get high cosine similarity."""

    model_name = "fake-bow"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class FakeLLM:
    """Returns queued outputs per output type; an LLMError in the queue is raised."""

    def __init__(self) -> None:
        self.queues: dict[type, deque] = defaultdict(deque)
        self.calls: list[tuple[type, str]] = []

    def queue(self, *outputs) -> "FakeLLM":
        for output in outputs:
            key = LLMError if isinstance(output, LLMError) else type(output)
            self.queues[key].append(output)
        return self

    def fail_next(self, output_type: type, message: str = "boom") -> "FakeLLM":
        self.queues[output_type].append(LLMError(message))
        return self

    def structured(self, *, system: str, user: str, output_type: type):
        self.calls.append((output_type, user))
        if not self.queues[output_type]:
            raise AssertionError(f"No scripted output for {output_type.__name__}")
        output = self.queues[output_type].popleft()
        if isinstance(output, LLMError):
            raise output
        return LLMResult(
            output=output,
            model="fake-model",
            latency_ms=5,
            input_tokens=100,
            output_tokens=20,
            request_id="req_test",
            fallback_used=False,
            raw_response="{}",
        )
