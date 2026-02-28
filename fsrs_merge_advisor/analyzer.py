from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .distance import inverse_covariance_for_vectors, mahalanobis_distance
from .reference_covariance import (
    FSRS6_RECENCY_DIM,
    FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD,
)


@dataclass(frozen=True)
class FSRSProfile:
    profile_id: int
    profile_name: str
    weights: tuple[float, ...]

    @property
    def deck_id(self) -> int:
        return self.profile_id

    @property
    def deck_name(self) -> str:
        return self.profile_name


@dataclass(frozen=True)
class DistanceResult:
    profile: FSRSProfile
    nearest_profile_name: str | None
    nearest_distance: float | None
    should_share_preset: bool | None

    @property
    def nearest_deck_name(self) -> str | None:
        return self.nearest_profile_name


def _to_float_tuple(values: Sequence[Any]) -> tuple[float, ...] | None:
    try:
        converted = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    return converted if converted else None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _field_any(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = _field(obj, name)
        if value is not None:
            return value
    return None


def _iter_candidate_sequences(config: Any) -> list[Sequence[Any]]:
    candidates: list[Sequence[Any]] = []

    for name in (
        "fsrsParams6",
        "fsrs_params6",
        "fsrsParams5",
        "fsrs_params5",
        "fsrsParams",
        "fsrs_params",
        "fsrsWeights",
        "fsrs_weights",
    ):
        value = _field(config, name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            candidates.append(value)

    fsrs_obj = _field(config, "fsrs")
    if fsrs_obj is not None:
        for name in ("weights", "params", "parameters"):
            nested = _field(fsrs_obj, name)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                candidates.append(nested)

    return candidates


def recommend_shared_preset(
    parameter_count: int,
    nearest_distance: float | None,
) -> bool | None:
    if nearest_distance is None:
        return None
    if parameter_count != FSRS6_RECENCY_DIM:
        return None
    return nearest_distance < FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD


def extract_fsrs_weights(config: Any) -> tuple[float, ...] | None:
    """Try FSRS-specific config locations across Anki versions."""
    if not config:
        return None

    for candidate in _iter_candidate_sequences(config):
        converted = _to_float_tuple(candidate)
        if converted is not None:
            return converted

    return None


def analyze_profiles(profiles: Sequence[FSRSProfile]) -> list[DistanceResult]:
    if not profiles:
        return []

    grouped: dict[int, list[FSRSProfile]] = {}
    for profile in profiles:
        grouped.setdefault(len(profile.weights), []).append(profile)

    results: list[DistanceResult] = []
    for same_length_profiles in grouped.values():
        if len(same_length_profiles) == 1:
            only = same_length_profiles[0]
            results.append(
                DistanceResult(
                    profile=only,
                    nearest_profile_name=None,
                    nearest_distance=None,
                    should_share_preset=None,
                )
            )
            continue

        vectors = [list(p.weights) for p in same_length_profiles]
        inv_cov = inverse_covariance_for_vectors(vectors)

        for idx, profile in enumerate(same_length_profiles):
            nearest_name: str | None = None
            nearest_distance: float | None = None

            for other_idx, other in enumerate(same_length_profiles):
                if idx == other_idx:
                    continue

                dist = mahalanobis_distance(vectors[idx], vectors[other_idx], inv_cov)
                if nearest_distance is None or dist < nearest_distance:
                    nearest_distance = dist
                    nearest_name = other.profile_name

            results.append(
                DistanceResult(
                    profile=profile,
                    nearest_profile_name=nearest_name,
                    nearest_distance=nearest_distance,
                    should_share_preset=recommend_shared_preset(
                        parameter_count=len(profile.weights),
                        nearest_distance=nearest_distance,
                    ),
                )
            )

    return sorted(results, key=lambda result: result.profile.profile_name.lower())


def pairwise_distance_matrix(
    profiles: Sequence[FSRSProfile],
) -> tuple[list[FSRSProfile], list[list[float | None]]]:
    sorted_profiles = sorted(profiles, key=lambda profile: profile.profile_name.lower())
    if not sorted_profiles:
        return [], []

    by_length: dict[int, list[int]] = {}
    for idx, profile in enumerate(sorted_profiles):
        by_length.setdefault(len(profile.weights), []).append(idx)

    matrix: list[list[float | None]] = [
        [None for _ in range(len(sorted_profiles))] for _ in range(len(sorted_profiles))
    ]
    for idx in range(len(sorted_profiles)):
        matrix[idx][idx] = 0.0

    for group_indexes in by_length.values():
        if len(group_indexes) < 2:
            continue

        vectors = [list(sorted_profiles[i].weights) for i in group_indexes]
        inv_cov = inverse_covariance_for_vectors(vectors)

        for left_offset, left_idx in enumerate(group_indexes):
            for right_offset, right_idx in enumerate(group_indexes):
                if right_offset <= left_offset:
                    continue

                dist = mahalanobis_distance(vectors[left_offset], vectors[right_offset], inv_cov)
                matrix[left_idx][right_idx] = dist
                matrix[right_idx][left_idx] = dist

    return sorted_profiles, matrix
