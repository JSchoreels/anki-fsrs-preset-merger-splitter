# anki-fsrs-preset-advisor docs

## Purpose

`anki-fsrs-preset-advisor` is an Anki add-on to:
- compute FSRS params per deck scope,
- measure preset/deck proximity with Mahalanobis distance,
- recommend and edit grouping of decks into presets.

## Quickstart

- Run tests:
  - `pytest -q`
- Sync addon to local Anki addons folder:
  - `./copy_to_anki.sh`
- Build `.ankiaddon` package:
  - `./package_ankiaddon.sh`

## Main entrypoints

- Addon bootstrap:
  - `__init__.py`
- Menu actions and UI flow:
  - `fsrs_merge_advisor/addon.py`
- Distance/profile logic:
  - `fsrs_merge_advisor/analyzer.py`
  - `fsrs_merge_advisor/distance.py`
  - `fsrs_merge_advisor/reference_covariance.py`
- Pure UI/group helper functions:
  - `fsrs_merge_advisor/deck_tools.py`

## Handoff

See `docs/HANDOFF.md` for feature-level status, design decisions, persistence files, and pending work.
