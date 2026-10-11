"""Tests for the exact kNN reference implementation."""

import numpy as np
import pytest

from ann import knn_bruteforce, l2_distance


def test_l2_distance_on_a_known_pair():
    # 3-4-5 triangle, so the answer is checkable by hand.
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert l2_distance(a, b) == pytest.approx(5.0)


def test_knn_returns_neighbours_in_ascending_distance_order():
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ]
    )
    query = np.array([0.1, 0.0])

    indices, distances = knn_bruteforce(vectors, query, k=2)

    assert list(indices) == [0, 1]
    assert distances[0] <= distances[1]
