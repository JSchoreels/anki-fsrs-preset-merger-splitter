from fsrs_merge_advisor.analyzer import (
    FSRSProfile,
    analyze_profiles,
    extract_fsrs_weights,
    pairwise_distance_matrix,
    recommend_shared_preset,
)


class _ConfigObject:
    def __init__(self, fsrs_weights):
        self.fsrs_weights = fsrs_weights


def test_extract_fsrs_weights_from_flat_key():
    config = {"fsrsWeights": [1, 2, 3]}
    assert extract_fsrs_weights(config) == (1.0, 2.0, 3.0)


def test_extract_fsrs_weights_prefers_fsrs_params6_over_legacy_weights():
    config = {
        "fsrsWeights": [1] * 17,
        "fsrsParams5": [2] * 19,
        "fsrsParams6": [3] * 21,
    }
    assert extract_fsrs_weights(config) == tuple([3.0] * 21)


def test_extract_fsrs_weights_falls_back_when_fsrs_params6_empty():
    config = {
        "fsrsParams6": [],
        "fsrsParams5": [2] * 19,
    }
    assert extract_fsrs_weights(config) == tuple([2.0] * 19)


def test_extract_fsrs_weights_from_object_attribute():
    config = _ConfigObject([0.11, 0.22, 0.33])
    assert extract_fsrs_weights(config) == (0.11, 0.22, 0.33)


def test_extract_fsrs_weights_from_nested_key():
    config = {"fsrs": {"weights": [0.1, 0.2, 0.3]}}
    assert extract_fsrs_weights(config) == (0.1, 0.2, 0.3)


def test_extract_fsrs_weights_rejects_invalid_sequence():
    config = {"fsrsWeights": [1, "x", 3]}
    assert extract_fsrs_weights(config) is None


def test_extract_fsrs_weights_does_not_use_generic_weights_field():
    config = {"weights": [9, 9, 9], "fsrsWeights": [1, 2, 3]}
    assert extract_fsrs_weights(config) == (1.0, 2.0, 3.0)


def test_extract_fsrs_weights_ignores_non_fsrs_weights_when_no_fsrs_key():
    config = {"weights": [9, 9, 9]}
    assert extract_fsrs_weights(config) is None


def test_analyze_profiles_returns_nearest_deck_name():
    profiles = [
        FSRSProfile(profile_id=1, profile_name="A", weights=(1.0, 2.0, 3.0)),
        FSRSProfile(profile_id=2, profile_name="B", weights=(1.1, 2.1, 3.1)),
        FSRSProfile(profile_id=3, profile_name="C", weights=(9.0, 9.0, 9.0)),
    ]

    results = analyze_profiles(profiles)
    by_name = {res.profile.profile_name: res for res in results}

    assert by_name["A"].nearest_profile_name == "B"
    assert by_name["B"].nearest_profile_name == "A"
    assert by_name["C"].nearest_profile_name in {"A", "B"}


def test_analyze_profiles_handles_single_profile_group():
    profiles = [
        FSRSProfile(profile_id=1, profile_name="A", weights=(1.0, 2.0)),
        FSRSProfile(profile_id=2, profile_name="B", weights=(1.0, 2.0, 3.0)),
    ]

    results = analyze_profiles(profiles)

    assert len(results) == 2
    assert all(res.nearest_profile_name is None for res in results)
    assert all(res.nearest_distance is None for res in results)
    assert all(res.should_share_preset is None for res in results)


def test_recommend_shared_preset_threshold_logic():
    assert recommend_shared_preset(parameter_count=21, nearest_distance=3.79) is True
    assert recommend_shared_preset(parameter_count=21, nearest_distance=3.8) is False
    assert recommend_shared_preset(parameter_count=20, nearest_distance=2.0) is None
    assert recommend_shared_preset(parameter_count=21, nearest_distance=None) is None


def test_pairwise_distance_matrix_same_length_profiles():
    profiles = [
        FSRSProfile(profile_id=1, profile_name="A", weights=(1.0, 2.0, 3.0)),
        FSRSProfile(profile_id=2, profile_name="B", weights=(1.1, 2.1, 3.1)),
        FSRSProfile(profile_id=3, profile_name="C", weights=(9.0, 9.0, 9.0)),
    ]

    ordered, matrix = pairwise_distance_matrix(profiles)

    assert [p.profile_name for p in ordered] == ["A", "B", "C"]
    assert len(matrix) == 3
    assert matrix[0][0] == 0.0
    assert matrix[1][1] == 0.0
    assert matrix[2][2] == 0.0
    assert matrix[0][1] is not None
    assert matrix[0][1] == matrix[1][0]
    assert matrix[0][2] == matrix[2][0]
    assert matrix[1][2] == matrix[2][1]


def test_pairwise_distance_matrix_cross_dimension_is_none():
    profiles = [
        FSRSProfile(profile_id=1, profile_name="A", weights=(1.0, 2.0)),
        FSRSProfile(profile_id=2, profile_name="B", weights=(1.1, 2.1)),
        FSRSProfile(profile_id=3, profile_name="C", weights=(1.0, 2.0, 3.0)),
    ]

    ordered, matrix = pairwise_distance_matrix(profiles)

    assert [p.profile_name for p in ordered] == ["A", "B", "C"]
    assert matrix[0][1] is not None
    assert matrix[0][2] is None
    assert matrix[1][2] is None
