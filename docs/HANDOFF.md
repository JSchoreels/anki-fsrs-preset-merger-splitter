# Handoff Notes

## 1) Project identity and paths

- Current repository path:
  - `/Users/jschoreels/workspace/anki-fsrs-preset-advisor`
- Addon package name:
  - `anki-fsrs-preset-advisor`
- Anki addon sync target:
  - `/Users/jschoreels/Library/Application Support/Anki2/addons21/anki-fsrs-preset-advisor`

## 2) Core behavior currently implemented

### Distance model

- Mahalanobis distance is implemented internally (no SciPy dependency).
- FSRS6-only comparisons:
  - 21 parameters required.
  - non-21 params are marked as `Not FSRS6 valid params`.
- Distance transform:
  - natural log (`ln`) applied to first 4 FSRS params before distance.
- Covariance strategy:
  - uses fixed provided inverse covariance matrix from `reference_covariance.py`.
  - dynamic covariance computation was removed.
- Threshold default:
  - `3.8` (`FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD`).

### Menu actions in Anki Tools

- `FSRS Preset Proximity`
  - uses existing preset configs and their FSRS params.
- `FSRS Deck Proximity (Computed)`
  - computes params per deck scope first, then shows proximity/grouping UI.

### Deck computation scope

- Default mode: leaf decks only.
- Optional mode: include middle/root decks too (user prompt).
- Search construction:
  - leaf: `did:<deck_id> -is:suspended`
  - include middle/root: `deck:"<deck_name>" -is:suspended`

### Progress and cancellation

- Deck optimization shows progress dialog with cancel.
- If canceled:
  - warns if nothing computed,
  - otherwise shows partial results.

## 3) Current UI model

Primary screen for deck-computed flow is similarity groups editor (`_show_similarity_groups`).

### Left panel

- Explicitly labeled as presets editor.
- Decks grouped by their *existing preset*.
- Drag/drop allowed only within left pane lists.
- Save button applies left-pane assignment edits to actual deck preset assignment.

### Right panel

- Explicitly labeled as similarity groups.
- Recommends groups from pairwise distances.
- Includes singleton groups.
- Group header shows compact metric:
  - `Group X (Max: <value>)`.
- Deck row metric uses max distance to others in group (not avg).
- Value coloring:
  - only item-level max value is colorized (`green` if `< 3.8`, `red` if `>= 3.8`).
- Drag/drop allowed only within right pane lists.

### Cross-pane interaction rules

- Cross-panel drag/drop is blocked.
- Clicking one item deselects others and selects corresponding item in opposite panel.

### Controls

- Threshold slider at top of group editor:
  - recomputes recommended right-side groups dynamically.
- `See Proximity Matrix` button:
  - opens proximity table + pairwise matrix combined view.
- `Reset Groups`:
  - resets both panels to initial state at dialog open.
- `Store Preset Backup` / `Revert Preset Backup`.
- `Use Recommended Group`:
  - creates/reuses advisor presets and assigns decks.
- `Save Preset Changes`:
  - persists manual left-panel edits.

## 4) Preset creation/apply behavior

When `Use Recommended Group` is clicked:

- Advisor preset naming:
  - `FSRS Preset Advisor : Group X`
- Reuse behavior:
  - if same advisor preset name already exists, it is reused (no new `(2)` names by default).
- Singleton groups:
  - also become advisor presets and receive assignments.
- Optional defaults source:
  - `Preset Default Values` dropdown can force copying settings from selected preset.
  - otherwise auto mode clones from a source preset found in the group.
- Assignment application:
  - done via deck config assignment APIs, then `mw.reset()`.

## 5) Persistence files

### Deck params cache

- File name: `deck_params_cache.json`
- Preferred location: active Anki profile folder.
- Legacy fallback: addon folder.
- Reuse condition: same review count for scope + cached params available.

### Preset backup

- File name: `deck_preset_backup.json`
- Preferred location: active Anki profile folder.
- Legacy fallback: addon folder.
- Used for manual backup/revert and auto backup before save/recommended apply operations.

## 6) Important modules

- `fsrs_merge_advisor/addon.py`
  - UI, menu wiring, cache/backup I/O, deck compute orchestration, preset reassignment.
- `fsrs_merge_advisor/analyzer.py`
  - profile extraction, log transform, pairwise distance matrix orchestration.
- `fsrs_merge_advisor/distance.py`
  - pure Mahalanobis distance + validation/getter for fixed inverse covariance.
- `fsrs_merge_advisor/reference_covariance.py`
  - shipped matrix and constants.
- `fsrs_merge_advisor/deck_tools.py`
  - pure helpers (grouping, thresholds, labels, search query construction, etc.).

## 7) Tests status

- Test suite:
  - `pytest -q`
- Last known status before this docs update:
  - `38 passed`.
- Most pure logic is covered in:
  - `tests/test_deck_tools.py`
  - `tests/test_analyzer.py`
  - `tests/test_distance.py`

## 8) Packaging / sync scripts

- Local sync:
  - `copy_to_anki.sh`
- Package build:
  - `package_ankiaddon.sh`
- Existing dist file may still include old basename:
  - `dist/anki-fsrs-preset-merger-splitter.ankiaddon`
  - this is an artifact naming leftover; script output now defaults to new name.

## 9) Known pending items for next thread

- New user request not yet implemented:
  - after `Use Recommended Group` creates presets, prompt user to optimize those newly created presets.
- Suggestion for implementation of pending item:
  - capture newly created preset IDs during `_use_recommended_groups()`.
  - after assignment success, show confirmation dialog.
  - if accepted, run FSRS optimization per created preset and update preset params.
  - show progress + failures summary.

## 10) Practical code anchors

- Similarity groups editor:
  - `fsrs_merge_advisor/addon.py` function `_show_similarity_groups()`.
- Recommended group apply handler:
  - nested `_use_recommended_groups()` inside `_show_similarity_groups()`.
- Deck-computed main flow:
  - `_show_deck_computed_results()`.
- Proximity matrix/table dialog:
  - `_show_results_for_profiles()` and `_show_pairwise_distances()`.

