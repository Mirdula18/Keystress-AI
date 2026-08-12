# Changelog

All notable changes to Keystress-AI. Format follows [Keep a Changelog](https://keepachangelog.com/);
versioning is not yet semantic — the project is a pre-release research prototype.

Entries reference the feature IDs in [`docs/FEATURES.md`](docs/FEATURES.md).

---

## [Unreleased] — Phase 1: ethical core + validity spine

### F2 — Consent, disclaimer & deletion flow

The first thing the project stores about a person, and the gate that must exist before it does.
Nothing is analysed without a recorded consent, nothing is stored without a second, separate
opt-in, and everything stored can be viewed and permanently deleted by the person it belongs to
(HARD RULE 4).

**Added**
- `keystress/core/consent.py` — the consent wording as versioned content (`CONSENT_VERSION`,
  `CONSENT_SUMMARY`). Every consent record stores the version in force when it was given, so a
  later policy change cannot silently reinterpret what someone agreed to.
- `keystress/core/storage.py` — SQLite store (stdlib, no new dependency) for consent records and
  opt-in donations. A donation is filtered to exactly the five `FEATURES_V1` values and coerced to
  `float` at the boundary, so no content-bearing field can reach disk even if a caller passes one.
  Deletion cascades: rows are removed, not flagged.
- `keystress/api/consent.py` — `GET /api/consent/policy`, `POST /api/consent`,
  `PATCH /api/consent/<id>` (change or withdraw), `POST /api/donate`, `GET /api/data/<id>`,
  `DELETE /api/data/<id>`. The anonymous `participant_id` is the only credential.
- **Consent gate on `/api/predict`**: without a valid token the endpoint returns `403` and no
  analysis runs. Config-gated by `KEYSTRESS_REQUIRE_CONSENT` (on by default).
- **UI gate** (`web/index.html`, `web/static/app.js`): the typing card ships hidden and is revealed
  only after consent is given against the policy text fetched from the API. Neither checkbox is
  pre-ticked. A "Your data" panel shows the stored record verbatim as JSON, toggles the donation
  opt-in, and deletes everything. The results card states plainly whether the session was stored.
- Config surface: `KEYSTRESS_STORE_PATH`, `KEYSTRESS_REQUIRE_CONSENT`.
- `tests/test_storage.py`, `tests/test_consent_api.py` — the store, the endpoints, the gate, the
  UI markup, and withdrawal. `tests/test_privacy.py` gains a section asserting that a *stored*
  donation is content-free, including a scan of the raw database bytes.
- Test-isolation guard: an autouse fixture fails any test that would open the real consent
  database, so consent rows can never be written into the working tree by a test run.

**Decisions.** D-022 (F2 design: token model, two-part consent, withdrawal vs deletion).

### F16 — Self-hosted assets, offline capability & a strict CSP

The page now loads nothing from anywhere but this origin, and the Content-Security-Policy says so
instead of merely claiming to.

**Changed**
- **Strict Content-Security-Policy.** Every directive is `'self'` or `'none'` — no CDN hosts, no
  `'unsafe-inline'` in `script-src` or `style-src`. The only non-`'self'` source left is `data:` in
  `img-src`, for the inline SVG favicon the page carries rather than fetches. This closes the
  compromise D-021 recorded when it shipped the loose policy.
- **No inline behaviour.** The nine `onclick`/`onchange` attributes are replaced by a
  `CONTROL_BINDINGS` table in `app.js`; a missing element id is logged rather than thrown, so one
  typo cannot unbind every other control.
- **No inline styling.** `style="display: none"` on the gated cards becomes an `.is-hidden` class
  toggled through `showCard`/`hideCard`; the probability bars take their zero width from the
  stylesheet. Scripted width changes go through the CSSOM, which CSP does not restrict.
- `keystress/security.py` exposes `CSP_DIRECTIVES` and `STRICT_DIRECTIVES`, so the policy is
  asserted directive by directive rather than string-matched.

**Added**
- `tests/test_offline_assets.py` — no subresource is fetched from another origin (the footer's
  GitHub link is navigation, not a fetch, and stays); no inline handler, `<script>` block, or
  `style` attribute may reappear; every linked local asset is actually served; every id in the
  binding table exists in the markup.
- `tests/test_security.py` gains `TestStrictContentSecurityPolicy`: the policy forbids inline and
  remote sources, **and** the served page complies with it. A strict header over a page that needs
  inline script would pass an audit while silently losing its buttons.

**Decisions.** D-023 (strict CSP, and the permanent frontend constraint it imposes).

### Accessibility — the page keeps the promises its markup makes

**Fixed**
- The live region (`#announcer`) and the advertised `Ctrl`+`Enter` shortcut existed in markup only.
  Nothing wrote to the region, so a screen-reader user got silence at every state change — consent
  accepted, analysis started, result ready, data deleted are each conveyed only by a card appearing
  — and the shortcut the page told users about did nothing at all.
- The loader card carried a permanent `aria-hidden="true"`, which silenced the `role="status"` line
  inside it: the one element whose entire job is to be announced never could be. It is hidden by
  `display: none` when inactive, which already removes it from the accessibility tree.
- The results card had no accessible name, so it did not appear in a screen reader's region list.

**Added**
- `announce()` plus a call at every otherwise-silent transition. The result announcement repeats the
  same uncalibrated/synthetic qualifier the page displays — HARD RULE 3 governs the spoken channel
  too — and a test asserts no announcement can read the typing box.
- `handleShortcut()`: Ctrl+Enter, and Cmd+Enter for a Mac, gated by the same minimum-session-length
  check as the button so it cannot submit what the button refuses. `event.key` is compared and
  discarded, exactly as the Backspace check is.
- `tests/test_accessibility.py` — pairs each affordance in the markup with the code obliged to
  honour it, so none can quietly go back to being decorative.

### F3 — Privacy hardening & local-first defaults

Defence-in-depth for the serving path, ahead of the project moving from a localhost tool toward a
consent-gated, publicly hostable data-donation site (see D-019).

**Added**
- Request-body cap (`MAX_CONTENT_LENGTH`, default 1 MiB): an oversized payload is rejected with
  `413` before it is parsed, so it cannot exhaust memory. Complements the semantic event-count cap
  (D-021).
- Per-client rate limiting on `/api/predict` via Flask-Limiter (default `60/minute`), returning
  `429` with a `Retry-After` header. Only the model endpoint is throttled; health and static assets
  are not.
- Security headers on every response — `Content-Security-Policy`, `X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`, `Permissions-Policy`; HSTS is
  added only on a secure origin (`keystress/security.py`).
- Config surface: `KEYSTRESS_MAX_CONTENT_LENGTH`, `KEYSTRESS_RATE_LIMIT`,
  `KEYSTRESS_RATE_LIMIT_ENABLED`.
- Runtime dependency `flask-limiter` (pinned with its transitive set in `requirements-lock.txt`).
- `tests/test_security.py` — asserts the headers, the 413 cap, and the 429 limit; the shared test
  fixtures disable rate limiting so they stay hermetic (D-021).

**Decisions.** D-019 (F4 delivered as a crowdsourced, consent-gated donation site with honest
framing), D-020 (Copenhagen Burnout Inventory as the labelling instrument), D-021 (F3 technical
choices).

---

## Phase 0 — honesty and buildability

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

---

### Gap-filling hardening (Phase 0 follow-up)

#### Fixed
- **`NaN`/`Inf` timestamps no longer poison the privacy boundary.** `record_keypress` and
  `process_keystroke_data` reject non-finite timestamps with a `ValueError`; previously they passed
  the numeric check and would have turned every downstream aggregate (durations, delays, features,
  and ultimately model input) into `NaN`/`Inf` — silently breaking the keep-typed-content-private
  and metrics-tell-the-truth guarantees (HARD RULES 1 and 3).
- **`evaluate_model` no longer crashes on degenerate predictions.** `classification_report` is
  now called with explicit `labels=[0, 1, 2]`, so a model that predicts fewer classes than the
  three-class taxonomy still produces a report (with `zero_division=0` rows) instead of raising
  `ValueError` from scikit-learn (HARD RULE 6).
- **Frontend gate now matches the server minimum.** The page allowed sessions the server rejects:
  the analyze button enabled at 20 keystrokes while the API refuses anything under
  `MIN_KEYSTROKE_EVENTS` (5). Both sites now use 5, with a comment documenting the contract.

#### Changed
- `tests/test_core.py`, `tests/test_api.py`, `tests/test_ml.py`, and `tests/test_characterization.py`
  — 33 new tests covering the previously untested branches above: non-finite timestamp rejection,
  degenerate-prediction evaluation, maximum-size payload acceptance, corrupt-artifact auto-training
  and its swallowed-failure path, config env overrides, feature-vector shape, out-of-range class
  fallback, uneven synthetic splits, and more. Test count: 307.
- `tests/test_metric_qualifiers.py` — 5 new tests for file scanning and integer-score detection.
