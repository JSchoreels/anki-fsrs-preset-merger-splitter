"""FSRS deck merge advisor addon package."""

from .analyzer import (
    DistanceResult,
    FSRSProfile,
    analyze_profiles,
    extract_fsrs_weights,
    pairwise_distance_matrix,
    recommend_shared_preset,
)

__all__ = [
    "DistanceResult",
    "FSRSProfile",
    "analyze_profiles",
    "extract_fsrs_weights",
    "pairwise_distance_matrix",
    "recommend_shared_preset",
]
