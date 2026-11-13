import numpy as np
import pytest

from app.ai.vectors import bytes_to_vector, cosine_similarity, vector_to_bytes


def test_vector_survives_round_trip_through_bytes():
    vector = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.array_equal(bytes_to_vector(vector_to_bytes(vector)), vector)


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_similarity_of_a_zero_vector_is_zero():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
