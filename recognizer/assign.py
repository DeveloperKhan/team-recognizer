"""Decode a whole team at once: six tiles, six different species.

Tournament rules allow a species only once per team, so when two tiles' best guesses collide the
assignment that maximizes the team's total score wins (Hungarian algorithm over each tile's top
candidates). The solver is implemented here rather than taken from SciPy, which would be most of
the deployment's size for this one call.
"""

from __future__ import annotations

import numpy as np

TOP_K = 10


def min_cost_assignment(cost: np.ndarray) -> list[int]:
    """For an (n, m) cost matrix with n <= m, the column assigned to each row, all distinct,
    minimizing the total cost (Hungarian algorithm with potentials, O(n^2 m))."""
    n, m = cost.shape
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner = np.zeros(m + 1, int)  # owner[j] = row (1-based) assigned to column j; 0 = free
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    result = [0] * n
    for j in range(1, m + 1):
        if owner[j]:
            result[owner[j] - 1] = j - 1
    return result


def assign_team(scores: np.ndarray, top_k: int = TOP_K) -> list[int]:
    """Class index per tile, all distinct, maximizing the summed score.

    `scores` is (n_tiles, n_classes).
    """
    candidates = sorted({int(c) for row in scores for c in np.argsort(-row)[:top_k]})
    cols = min_cost_assignment(-scores[:, candidates])
    return [candidates[c] for c in cols]
