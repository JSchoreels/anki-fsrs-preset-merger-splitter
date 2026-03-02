# Architecture

## Runtime flow (deck-computed mode)

1. Menu action `FSRS Deck Proximity (Computed)` calls `_show_deck_computed_results()`.
2. User chooses scope (leaf only vs include middle/root).
3. `_load_computed_deck_profiles()` loops selected decks:
   - review-count-based cache check,
   - compute params via backend `compute_fsrs_params` when cache miss,
   - keep only FSRS6-valid profiles.
4. `pairwise_distance_matrix()` computes distances.
5. Similarity groups are built with `similarity_groups_from_matrix(..., min_group_size=1)`.
6. Group editor dialog `_show_similarity_groups()` opens as primary view.

## Runtime flow (preset mode)

1. Menu action `FSRS Preset Proximity` calls `_show_preset_results()`.
2. `_load_profiles()` extracts FSRS params from existing deck presets.
3. `_show_results_for_profiles()` renders table + matrix UI.

## Distance stack

- `extract_fsrs_weights()` -> `transform_params_for_distance()` ->
  `get_validated_fsrs6_inverse_covariance()` -> `mahalanobis_distance()`.
- `get_validated_fsrs6_inverse_covariance()` validates vector shape and returns the shipped matrix.

## Grouping model

- Grouping uses complete-link style merge:
  - clusters merge only if *all cross-pairs* are below threshold.
- Therefore, each final group should satisfy:
  - every pairwise distance inside group `< threshold`.

## Persistence model

- Deck param cache and preset backup each use:
  - preferred profile-folder path,
  - fallback legacy addon-folder path.
- On successful read from legacy path, data is re-saved to preferred path.

## Main code anchors

- UI/actions/orchestration: `fsrs_merge_advisor/addon.py`
- Distance analysis: `fsrs_merge_advisor/analyzer.py`
- Math core: `fsrs_merge_advisor/distance.py`
- Constants/matrix: `fsrs_merge_advisor/reference_covariance.py`
- Utility helpers: `fsrs_merge_advisor/deck_tools.py`

