"""Local text embeddings, behind a small interface so the model can be swapped."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.ai.vectors import VECTOR_DTYPE


class Embedder(Protocol):
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized float32 vector per text, as an (n, d) matrix."""
        ...


def build_text_to_embed(title: str, description: str) -> str:
    """The text a submission is embedded as: its title, then its description."""
    return f"{title.strip()}\n{description.strip()}"


class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model locally on the CPU.

    The model is loaded the first time it's needed, because loading takes a few
    seconds and downloads the weights on first use.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._loaded_model = None

    def _get_loaded_model(self):
        if self._loaded_model is None:
            from sentence_transformers import SentenceTransformer

            self._loaded_model = SentenceTransformer(self.model_name, device="cpu")
        return self._loaded_model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        model = self._get_loaded_model()
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(VECTOR_DTYPE)


@lru_cache
def get_shared_embedder(model_name: str) -> Embedder:
    """Return one embedder per model name, so the model is only loaded once."""
    return SentenceTransformerEmbedder(model_name)
