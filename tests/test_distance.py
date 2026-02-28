import pytest

from fsrs_merge_advisor.distance import (
    get_validated_fsrs6_inverse_covariance,
    mahalanobis_distance,
)
from fsrs_merge_advisor.reference_covariance import (
    FSRS6_RECENCY_DIM,
    FSRS6_RECENCY_INV_COVARIANCE_21,
)


def test_mahalanobis_positive_with_reference_inverse_covariance():
    dist = mahalanobis_distance(
        [1.0] * FSRS6_RECENCY_DIM,
        [2.0] * FSRS6_RECENCY_DIM,
        FSRS6_RECENCY_INV_COVARIANCE_21,
    )

    assert dist > 0


def test_get_validated_fsrs6_inverse_covariance_uses_reference_for_21_params():
    rows = [[float(i)] * FSRS6_RECENCY_DIM for i in range(1, 4)]

    inv_cov = get_validated_fsrs6_inverse_covariance(rows)

    assert inv_cov == FSRS6_RECENCY_INV_COVARIANCE_21


def test_get_validated_fsrs6_inverse_covariance_rejects_non_fsrs6_dimensions():
    rows = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]

    with pytest.raises(ValueError, match="Not FSRS6 valid params"):
        get_validated_fsrs6_inverse_covariance(rows)
