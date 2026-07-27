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

---

### F11 — Kill fragile patterns: imports, globals, packaging

#### Added
- **Real installable package.** `pyproject.toml` with `pip install -e .`, console scripts
  `keystress` and `keystress-train`, and ruff/pytest configuration. Layout follows
  `ARCHITECTURE.md` §3: `keystress/{api,core,ml,web}` (D-008).
- `keystress/core/model.py` — immutable `ModelBundle` (estimator + scaler + metadata) and a
  `ModelRegistry` that swaps bundles atomically. Injectable through `create_app()`, which is what
  makes the API testable without a trained artifact on disk (D-009).
- `keystress/config.py` — environment-driven settings with loopback-by-default binding and paths
  resolved relative to the project root rather than the working directory.
- `keystress/app.py` — `create_app()` application factory; blueprints in `keystress/api/`.
- `/readyz` readiness endpoint, distinct from liveness: it reports whether a prediction can actually
  be served.
- Server-side input validation on `/api/predict`: payload shape, event-count bounds, and numeric
  timestamps, with clear 400s instead of a 500.

#### Changed
- **Removed the `sys.path` manipulation** (`app.py:18`) — imports now resolve through the package.
- **Removed module-level mutable model state.** The `model`/`scaler` globals and `load_models()`
  are gone (D-009).
- **`print` replaced with `logging`** throughout; all emoji removed from output. Emoji in log lines
  raise `UnicodeEncodeError` on legacy Windows console codepages, turning a cosmetic banner into a
  startup crash.
- **Removed the unused `LogisticRegression` import** together with `train_logistic_regression()`,
  the dead function that was its only referent.
- `src/` replaced by `keystress/` using `git mv`, preserving per-file history. No compatibility
  shim (D-008).
- Minimum Python raised to 3.9; 3.8 is end-of-life (D-011).
- README rewritten with the real layout, working commands, and the configuration table.

#### Fixed
- **The synthetic generator no longer creates artificial point masses.** `max(floor, x)` clamping
  collapsed ~2.3% of one class onto exactly `0.01` and ~3% of another onto exactly `0.5` — a
  generator artifact a tree model can split on to score well for the wrong reason. Replaced with
  rejection sampling; verified 1500/1500 distinct values with none at either floor. The
  data-generating process is now versioned (`g2`) and appears in every model identity (D-010).
- Model artifacts that are missing or corrupt now produce a clear 503 and a degraded `/readyz`
  rather than a stack trace or a startup crash (HARD RULE 6).
- `TypingSession` switched from wall-clock `time.time()` to `time.monotonic()`, and accepts an
  injected timestamp so tests need not sleep.

---

### F10 — Extract the frontend from the Python string

#### Added
- `keystress/web/` — the frontend as real files: `index.html` (145 lines),
  `static/styles.css` (428 lines), `static/app.js` (194 lines). Build-free: Flask serves the
  directory, so editing the frontend needs no toolchain (D-013).
- `tests/test_characterization.py` — 67 tests pinning the rendered page and the `/api/predict`
  round-trip, **written and run green before the extraction, then run green after it with no
  assertion changed**. That is the evidence the extraction preserved behaviour.
- `tests/conftest.py` — shared fixtures, including an in-memory `ModelBundle` so API tests never
  depend on a trained artifact on disk, and `page_bundle()`, which fetches `/` plus every linked
  same-origin asset (D-012).

#### Changed
- **No HTML, CSS, or JavaScript remains in any Python file.** `keystress/app.py` drops from 817 to
  180 lines; the 677-line `HTML_TEMPLATE` string literal is gone.
- `/` now renders `web/index.html`; assets are served from `/static/`.
- The frontend privacy contract is documented at the top of `app.js`, where the keydown handler
  lives, rather than being implicit.

#### Notes
- The two CDN `<link>` tags (Google Fonts, Font Awesome) were carried across unchanged. Removing
  them is F16; doing it here would have made this change more than behaviour-preserving.

---

### F12 — Test suite including a dedicated privacy test

#### Added
- **`tests/test_privacy.py` (26 tests)** asserting the project's central guarantee across three
  independent layers: response and session-metadata key allowlists; equivalence tests proving that
  sessions differing only in typed content produce byte-identical output; and source-level scans
  ensuring no core module reads a content-bearing field and the frontend stores a boolean rather
  than a key. Hostile payloads carrying `key`, `char`, `code`, `keyCode`, `text`, `clipboard`,
  `window`, and `url` are pushed through the full HTTP round-trip and asserted to vanish.
  **Verified by injecting a real leak and confirming three tests fail** (D-014).
- `tests/test_core.py` (66 tests) — collection, features, model loading, inference, disclosure
  helpers, and CLI formatting.
- `tests/test_ml.py` (41 tests) — synthetic generation, training, versioning, metadata. Includes
  explicit regression tests for the clamp point-mass defect, so it becomes a build failure rather
  than something noticed years later.
- `tests/test_api.py` (36 tests) — payload validation, error paths, the app factory, configuration,
  and static asset serving.
- `pytest` `privacy` marker so the privacy suite can be run and enforced independently in CI.

#### Fixed
- **Corrupt model artifacts no longer escape as a stack trace.** A corrupt pickle raises `IndexError`
  from inside `pickle.pop_mark`, which the original narrow exception tuple did not catch — found by
  the new test, not by inspection. Now handled at the deserialisation boundary and reported as a
  clean 503 (HARD RULE 6, D-015).

#### Notes
- 265 tests; 92% statement coverage overall, 93-100% across every `core/` and `api/` module.
- Tests never read the real `models/` directory. `conftest.py` builds an in-memory `ModelBundle`, so
  results do not depend on whether a developer has run training — a hidden disk dependency that one
  test did have until it was caught and fixed.

---

### F13 — CI, dependency pinning, and reproducible builds

#### Added
- `.github/workflows/ci.yml` with six jobs, each mapping to a Phase 0 exit criterion:
  `lint`, `test` (Python 3.9-3.12 matrix), `privacy`, `honesty`, `reproducible`, and `install`
  (Ubuntu + Windows, proving `pip install -e .` works and the package imports without the old
  `sys.path` hack).
- **CI verifies its own guards can fail** (D-018): the honesty job runs the metric checker against a
  known-bad fixture and fails if it passes; the privacy job injects a real content leak, requires the
  suite to catch it, restores the file, and confirms the restoration with `git diff --exit-code`.
- `tools/check_reproducible_build.py` — builds twice from scratch in separate directories and
  compares the dataset digest, predictions, full class probabilities, metrics, and model version
  (D-017).
- `requirements-lock.txt` — fully pinned dependency set. Pinning scikit-learn matters specifically
  here: estimator internals can change between minor versions, so an unpinned upgrade could silently
  change what a "reproducible" build produces.
- `TestSuiteIntegrity` in the privacy suite, failing if the number of privacy tests drops — a suite
  that collects zero tests otherwise passes.
- Reproducibility tests running a smaller build comparison locally, including one asserting that
  *different* seeds produce *different* builds, so the check cannot trivially pass.

#### Fixed
- **Model builds were not actually reproducible.** `RandomForestClassifier(n_jobs=-1)` made fitting
  non-deterministic at float precision even with `random_state` fixed: six refits produced six
  different probability arrays. Found by the new reproducibility check, not by inspection — the
  project would have shipped a false reproducibility claim. Training is now single-threaded, at a
  measured cost of 0.371s for the full 1500-sample fit (D-016). This also stops the confidence figure
  shown to a user from varying between otherwise identical runs.

#### Fixed (follow-up)
- **The metric-qualifier checker had a regex backtracking flaw.** Given `0.371s` the engine matched
  `0.371`, saw the forbidden unit suffix, then gave back a digit to match `0.37` — whose next
  character is `1`, not a unit — and reported a metric. Any decimal followed by a unit could be
  partially matched this way. Found when the checker flagged a timing measurement in this very
  changelog. The fractional part is now anchored so it cannot be shortened, with regression tests
  covering both the false positive and the real metrics that must still be caught.
