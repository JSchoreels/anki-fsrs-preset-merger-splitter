from __future__ import annotations

import math
from typing import Sequence

from .reference_covariance import FSRS6_RECENCY_DIM, FSRS6_RECENCY_INV_COVARIANCE_21


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(matrix)
    if n == 0:
        raise ValueError("Cannot invert an empty matrix")

    augmented = [list(row) + ident for row, ident in zip(matrix, _identity(n))]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        pivot = augmented[pivot_row][col]
        if abs(pivot) < 1e-15:
            raise ValueError("Matrix is singular")

        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        scale = 1.0 / pivot
        for j in range(2 * n):
            augmented[col][j] *= scale

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(2 * n):
                augmented[row][j] -= factor * augmented[col][j]

    return [row[n:] for row in augmented]


def covariance_matrix(rows: Sequence[Sequence[float]], regularization: float = 1e-6) -> list[list[float]]:
    if not rows:
        raise ValueError("Cannot compute covariance of empty data")

    dim = len(rows[0])
    if dim == 0:
        raise ValueError("Cannot compute covariance of zero-length vectors")

    if any(len(row) != dim for row in rows):
        raise ValueError("All vectors must have the same length")

    if len(rows) < 2:
        cov = _identity(dim)
    else:
        means = [sum(row[i] for row in rows) / len(rows) for i in range(dim)]
        cov = [[0.0 for _ in range(dim)] for _ in range(dim)]
        for row in rows:
            centered = [row[i] - means[i] for i in range(dim)]
            for i in range(dim):
                for j in range(dim):
                    cov[i][j] += centered[i] * centered[j]

        denom = float(len(rows) - 1)
        for i in range(dim):
            for j in range(dim):
                cov[i][j] /= denom

    for i in range(dim):
        cov[i][i] += regularization

    return cov


def inverse_covariance(rows: Sequence[Sequence[float]], regularization: float = 1e-6) -> list[list[float]]:
    cov = covariance_matrix(rows, regularization=regularization)
    return _invert_matrix(cov)


def inverse_covariance_for_vectors(
    rows: Sequence[Sequence[float]],
    regularization: float = 1e-6,
) -> list[list[float]]:
    if not rows:
        raise ValueError("Cannot compute inverse covariance of empty data")

    dim = len(rows[0])
    if any(len(row) != dim for row in rows):
        raise ValueError("All vectors must have the same length")

    if dim == FSRS6_RECENCY_DIM:
        return [list(row) for row in FSRS6_RECENCY_INV_COVARIANCE_21]

    return inverse_covariance(rows, regularization=regularization)


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
