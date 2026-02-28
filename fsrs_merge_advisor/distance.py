from __future__ import annotations

import math
from typing import Sequence

from .reference_covariance import FSRS6_RECENCY_DIM, FSRS6_RECENCY_INV_COVARIANCE_21

def get_validated_fsrs6_inverse_covariance(
    rows: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not rows:
        raise ValueError("Cannot compute inverse covariance of empty data")

    dim = len(rows[0])
    if any(len(row) != dim for row in rows):
        raise ValueError("All vectors must have the same length")

    if dim != FSRS6_RECENCY_DIM:
        raise ValueError("Not FSRS6 valid params")

    return [list(row) for row in FSRS6_RECENCY_INV_COVARIANCE_21]


def mahalanobis_distance(
    left: Sequence[float],
    right: Sequence[float],
    inv_covariance: Sequence[Sequence[float]],
) -> float:
    if len(left) != len(right):
        raise ValueError("Input vectors must have the same length")

    dim = len(left)
    if any(len(row) != dim for row in inv_covariance):
        raise ValueError("Inverse covariance matrix shape does not match vector size")

    delta = [left[i] - right[i] for i in range(dim)]
    transformed = [sum(inv_covariance[i][j] * delta[j] for j in range(dim)) for i in range(dim)]
    squared = sum(delta[i] * transformed[i] for i in range(dim))
    return math.sqrt(max(squared, 0.0))
