"""Decode a whole team at once: six tiles, six different species.

Tournament rules allow a species only once per team, so when two tiles' best guesses collide the
assignment that maximizes the team's total score wins (Hungarian algorithm over each tile's top
candidates).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

TOP_K = 10


def assign_team(scores: np.ndarray, top_k: int = TOP_K) -> list[int]:
    """Class index per tile, all distinct, maximizing the summed score.

    `scores` is (n_tiles, n_classes).
    """
    n_tiles = scores.shape[0]
    candidates = sorted({int(c) for row in scores for c in np.argsort(-row)[:top_k]})
    cost = -scores[:, candidates]
    rows, cols = linear_sum_assignment(cost)
    out = [0] * n_tiles
    for r, c in zip(rows, cols):
        out[r] = candidates[c]
    return out
