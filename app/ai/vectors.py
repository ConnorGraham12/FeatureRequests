"""Embedding storage and brute-force cosine search.

Fine for thousands of submissions. Swap for sqlite-vec or pgvector when it isn't.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DTYPE = np.float32


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


@dataclass(frozen=True)
class Match:
    id: int
    similarity: float


def top_k(
    query: np.ndarray,
    candidates: np.ndarray,
    ids: Sequence[int],
    k: int,
    threshold: float = -1.0,
) -> list[Match]:
    """Return up to `k` candidates with cosine similarity >= `threshold`, best first.

    `candidates` is an (n, d) matrix whose rows line up with `ids`.
    """
    if len(ids) == 0:
        return []
    if candidates.shape[0] != len(ids):
        raise ValueError("candidates and ids must have the same length")

    q = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(candidates, axis=1)
    norms[norms == 0] = 1.0
    scores = (candidates / norms[:, None]) @ q

    order = np.argsort(-scores)[:k]
    return [
        Match(id=int(ids[i]), similarity=float(scores[i]))
        for i in order
        if scores[i] >= threshold
    ]
