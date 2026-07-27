# Changelog

All notable changes to Keystress-AI. Format follows [Keep a Changelog](https://keepachangelog.com/);
versioning is not yet semantic — the project is a pre-release research prototype.

Entries reference the feature IDs in [`docs/FEATURES.md`](docs/FEATURES.md).

---

## [Unreleased] — Phase 0: honesty and buildability

### Added
- `docs/AUDIT.md` — Phase 0 baseline audit of the inherited codebase: layout mismatches, the
  inline-HTML boundary, fragile patterns, the full metric-claim sweep, and verification of the
  privacy boundary.
- `docs/DECISIONS.md` — running log of non-obvious decisions (D-001 … ).
- `CHANGELOG.md` — this file.

### Changed
- Documentation relocated from `files/` to `docs/`, matching every internal cross-reference and the
  target layout in `ARCHITECTURE.md` §3 (D-002).
- Root `README.md` replaced with the honest version: no unqualified accuracy figures, explicit
  synthetic-data status, no "detects burnout" capability claim (D-002). The claims it replaced are
  catalogued in `docs/AUDIT.md` §5.
- `docs/PROJECT_REPORT.md` — three factual errors corrected inline and marked, and the ~90% metrics
  table now carries its data source on every row (D-003).

### Fixed
- Corrected the record that the repository had no git history. It had 7 commits and a live remote;
  history was preserved rather than re-initialised (D-001).

---

### F1 — Honest metrics and data-source labelling everywhere

#### Added
- `src/disclosure.py` — single source of truth for the disclosure contract: the disclaimer, the
  data-source vocabulary, the versioned feature set, and `format_metric`/`format_percentage`
  helpers that make emitting an unqualified number require deliberately bypassing them.
- **Required disclosure fields on every prediction response**: `data_source`, `model_version`,
  `feature_set`, `disclaimer`, and `insufficient_data`, per `docs/CLAUDE.md` §5.
- Model metadata sidecar `models/model_metadata.json`, written at training time, giving
  `model_version` a real provenance rather than a hardcoded string. Version identifiers are
  deterministic (`rf-v1-synthetic-s42-n1500`) so a clean checkout reproduces them (supports F13).
- `tools/check_metric_qualifiers.py` — build-time check that fails when a metric appears without a
  data-source qualifier nearby, with a `metrics-ok: <reason>` escape hatch that is inert unless a
  reason is written (D-007).
- `tests/test_metric_qualifiers.py` — 26 tests, including proof that the checker rejects a
  known-bad fixture, resists CSS and line-number false positives, and that the live repository
  passes its own rule.
- `/api/health` now reports `model_version`, `data_source`, and `feature_set`, so an operator can
  never be unsure whether an instance is serving synthetic-trained predictions.

#### Changed
- **Every metric the system emits now states its data source.** Training output, CLI prediction
  output, the API response, and the UI all carry the qualifier.
- Indicator labels reworded from asserted states ("Low Burnout") to indicators
  ("Low (indicator)"), and descriptions rewritten to describe the *signal* rather than the person.
- UI: replaced the "Detect academic burnout" tagline, added a prominent synthetic-data research
  banner, removed the hardcoded "Confidence: 85%" placeholder, and attached the model version and
  source qualifier to every displayed percentage. Confidence is labelled uncalibrated (F7 pending).
- `probabilities` is now an ordered array with a parallel `labels` array, matching
  `ARCHITECTURE.md` §4.3 (D-004). **Breaking change** to `/api/predict`.
- Training reports and feature-importance output are labelled synthetic-derived.

#### Fixed
- **Degenerate sessions no longer receive invented predictions.** An all-zero feature vector now
  returns `insufficient_data: true` with a null prediction instead of a confident-looking result
  derived from nothing — HARD RULE 6 (D-005).
- A non-result no longer falls back to the reassuring "low" styling; it renders as an explicit
  unknown.
- **Default bind changed from `0.0.0.0` to `127.0.0.1`** (HARD RULE 5), overridable via
  `KEYSTRESS_HOST` with a warning on non-loopback addresses. Pulled forward from F3 because the
  code contradicted its own startup message (D-006).
