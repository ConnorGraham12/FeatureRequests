"""Converting embeddings to and from bytes for storage, and comparing them."""

import numpy as np

VECTOR_DTYPE = np.float32


def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=VECTOR_DTYPE).tobytes()


def bytes_to_vector(stored_bytes: bytes) -> np.ndarray:
    return np.frombuffer(stored_bytes, dtype=VECTOR_DTYPE)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Similarity between two vectors, from -1 (opposite) to 1 (identical direction)."""
    first_length = np.linalg.norm(first)
    second_length = np.linalg.norm(second)
    if first_length == 0 or second_length == 0:
        return 0.0
    return float(np.dot(first, second) / (first_length * second_length))
