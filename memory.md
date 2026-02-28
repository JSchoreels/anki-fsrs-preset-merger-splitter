# Project Memory

## Anki Addon Vision

- Build an Anki addon to check whether decks should be merged based on the distance between their FSRS parameters.
- Compute FSRS parameters for each deck separately.
- Compare resulting per-deck parameters to determine whether decks are close enough.
- Preferred distance metric: Mahalanobis distance (`scipy.spatial.distance.mahalanobis`).
- Open technical uncertainty: importing SciPy inside Anki might be difficult, so a reimplementation may be needed.
- Optimization should ideally be temporary for computation only.
- Likely need to split existing decks into single preset during computation, unless `fsrs-rs` allows per-deck optimization without modifying presets.
- Need a small UI listing:
  - each deck,
  - its ideal/optimized parameters,
  - which deck(s) it is closest to.

## Notes from user

- Keep feature scope strict to requested logic.
- If uncertain about a feature, ask for clarification.
- Cover implemented behavior with tests.
