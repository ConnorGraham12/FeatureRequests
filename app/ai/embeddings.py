"""Local text embeddings behind a small interface, so the model can be swapped."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.ai.vectors import DTYPE


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalized embeddings."""
        ...


def build_embedding_text(title: str, description: str) -> str:
    return f"{title.strip()}\n{description.strip()}"


class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model locally on CPU.

    The model is loaded on first use, since loading takes a few seconds and
    downloads weights the first time.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.astype(DTYPE)


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    return SentenceTransformerEmbedder(model_name)
