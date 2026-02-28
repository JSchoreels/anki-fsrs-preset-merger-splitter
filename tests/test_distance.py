from fsrs_merge_advisor.distance import (
    covariance_matrix,
    inverse_covariance,
    inverse_covariance_for_vectors,
    mahalanobis_distance,
)
from fsrs_merge_advisor.reference_covariance import (
    FSRS6_RECENCY_DIM,
    FSRS6_RECENCY_INV_COVARIANCE_21,
)


def test_covariance_matrix_regularizes_diagonal():
    rows = [[1.0, 2.0], [3.0, 4.0]]
    cov = covariance_matrix(rows, regularization=0.1)

    assert cov[0][0] > 0.1
    assert cov[1][1] > 0.1


def test_inverse_covariance_and_mahalanobis_positive():
    rows = [[1.0, 2.0], [2.0, 2.5], [3.0, 4.0], [4.0, 5.0]]
    inv_cov = inverse_covariance(rows)

    dist = mahalanobis_distance([1.0, 2.0], [4.0, 5.0], inv_cov)

    assert dist > 0


def test_inverse_covariance_for_vectors_uses_reference_for_21_params():
    rows = [[float(i)] * FSRS6_RECENCY_DIM for i in range(1, 4)]

    inv_cov = inverse_covariance_for_vectors(rows)

    assert inv_cov == FSRS6_RECENCY_INV_COVARIANCE_21


def test_inverse_covariance_for_vectors_falls_back_for_other_dimensions():
    rows = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]

    inv_cov = inverse_covariance_for_vectors(rows)

    assert len(inv_cov) == 2
    assert len(inv_cov[0]) == 2
