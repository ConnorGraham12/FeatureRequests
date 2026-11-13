import numpy as np

from app.ai.vectors import from_blob, to_blob, top_k


def test_blob_round_trip():
    v = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.array_equal(from_blob(to_blob(v)), v)


def test_top_k_orders_by_similarity_and_applies_threshold():
    query = np.array([1.0, 0.0])
    candidates = np.array([[0.0, 1.0], [1.0, 0.1], [0.7, 0.7]])
    matches = top_k(query, candidates, ids=[10, 20, 30], k=5, threshold=0.6)

    assert [m.id for m in matches] == [20, 30]
    assert matches[0].similarity > matches[1].similarity


def test_top_k_limits_results_and_handles_empty():
    candidates = np.eye(3)
    assert len(top_k(np.array([1.0, 1.0, 1.0]), candidates, ids=[1, 2, 3], k=2)) == 2
    assert top_k(np.array([1.0]), np.empty((0, 1)), ids=[], k=5) == []
