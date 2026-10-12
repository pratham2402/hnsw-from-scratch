"""Exact nearest-neighbour search.

This module is the reference implementation for the project: approximate
indexes are evaluated against what these functions return, so they need to be
correct rather than fast.
"""

from __future__ import annotations

import numpy as np


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1-D vectors.

    Args:
        a: 1-D array of shape (d,).
        b: 1-D array of shape (d,).

    Returns:
        The Euclidean distance as a float.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def knn_bruteforce(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours of `query` among `vectors`.

    Args:
        vectors: 2-D array of shape (n, d) — the corpus.
        query: 1-D array of shape (d,) — the query point.
        k: How many neighbours to return.

    Returns:
        ``(indices, distances)``, each an array of length ``min(k, n)`` sorted
        by distance ascending, so ``indices[0]`` is the nearest neighbour.

    Raises:
        ValueError: if `k` is not positive.
    """
    raise NotImplementedError
