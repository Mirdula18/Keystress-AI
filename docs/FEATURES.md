# FEATURES.md — Keystress-AI Upgrade Specifications

20 upgrades that turn Keystress-AI from a synthetic-data prototype into an honest, ethical,
scientifically defensible research instrument for the typing-dynamics→wellbeing hypothesis.

**Format:** Goal · Why it matters · Approach · Touches · Libraries · Acceptance · Constraints.
**Effort:** 🟢 low · 🟡 medium · 🔴 high **Priority:** ⭐ = Phase-1 core.

> Guiding principle: features that increase *honesty and validity* rank above features that add
> surface capability. A smaller validated claim beats a bigger unvalidated one.

---

## Theme A — Integrity & Ethics (the foundation — nothing ships without these)

### F1 · Honest metrics & "synthetic" labeling everywhere 🟢 ⭐
**Goal.** Every accuracy/confidence number in UI, API, README, and logs states its data source.
**Why it matters.** The headline ~90% currently reads as real-world performance; it measures the
synthetic generator. This is the project's biggest credibility risk.
**Approach.** Add required `data_source` + `disclaimer` + `model_version` to the prediction
contract. Rewrite README/UI so no metric appears without "on synthetic data." Add a build-time
check that fails if a metric string lacks a source qualifier.
**Touches.** `predict`, response schema, README, UI results card.
**Acceptance.** No unqualified metric exists anywhere; API response always carries `data_source`;
CI check enforces it.

### F2 · Consent, disclaimer & data-deletion flow 🟡 ⭐
**Goal.** Explicit informed consent before any analysis; visible "not a medical assessment"
disclaimer; user can view and delete anything stored about them.
**Why it matters.** This is health-adjacent inference about a person. Consent and transparency are
ethical and, in many jurisdictions, legal requirements.
**Approach.** Consent gate before the typing test; plain-language explanation of what's collected
(metadata only) and why; opt-in for any data donation (F3); delete endpoint.
**Touches.** new `api/consent.py`, UI gate, storage layer.
**Acceptance.** No prediction runs without consent; disclaimer always visible on results; a user
can delete their data and it is actually gone; nothing is stored without opt-in.

### F3 · Privacy hardening & local-first defaults 🟢 ⭐
**Goal.** Localhost binding by default; treat raw timing data as sensitive; validate/limit input.
**Why it matters.** `0.0.0.0` + no auth + timing data is an unnecessary exposure.
**Approach.** Default host `127.0.0.1` (network bind is explicit opt-in). Input validation and rate
limiting on `/api/predict`. No raw timing data leaves the machine unless the user donates it (F3
consent). Security headers; no CDN dependency for core function (self-host fonts/icons — see F16).
**Touches.** `config.py`, `api/predict.py`, app factory.
**Libraries.** Flask-Limiter.
**Acceptance.** Default run is localhost-only; malformed/oversized payloads rejected cleanly; no
third-party network call required to produce a prediction.

---

## Theme B — Scientific Validity (the whole point)

### F4 · Real-data collection harness & study protocol 🔴 ⭐
**Goal.** Tooling to collect *real*, consented, labeled typing sessions paired with a validated
burnout/wellbeing instrument.
**Why it matters.** This is the only thing that can break the synthetic circularity. Everything
else is supporting infrastructure for this.
**Approach.** In `research/`: a session collector that pairs anonymized typing metadata with a
self-report label from a recognized short instrument (e.g. an academic-burnout or wellbeing scale —
document which, with citation and licensing). Anonymized participant IDs; consent-gated; export to a
labeled dataset schema. Include a written study protocol and data-handling policy.
**Touches.** new `research/` package, dataset schema, protocol doc.
**Acceptance.** Can collect and store consented, labeled real sessions with no content capture;
dataset schema documented; protocol covers consent, anonymization, retention, withdrawal.
**Constraints.** Offline/research path only — never on the live serving path. Cite the chosen
instrument; respect its license.

### F5 · Validation harness & honest evaluation 🟡 ⭐
**Goal.** Measure real performance on held-out *real* data: accuracy, per-class precision/recall,
calibration, confusion, and baseline comparison (vs. majority-class and vs. random).
**Why it matters.** Turns "~90% on synthetic" into a defensible (likely humbler) real number, or
honestly reports that the signal is weak.
**Approach.** `ml/evaluate.py`: stratified/grouped splits (never leak a participant across
train/test), calibration curves, comparison to trivial baselines, subgroup breakdowns. Persist an
eval report per model version.
**Touches.** `ml/evaluate.py`, model registry, eval report page.
**Libraries.** scikit-learn (metrics, calibration), matplotlib for report plots.
**Acceptance.** Reports real metrics with proper participant-grouped splits; always shows trivial
baselines alongside; a model with no real data is clearly marked "synthetic — not validated."
**Depends on.** F4.

### F6 · Per-user baseline & personalization 🟡
**Goal.** Score burnout as deviation from a user's *own* typing norm, not from population classes.
**Why it matters.** Between-person variation swamps the burnout signal in a population model;
within-person change is far more plausible and more private.
**Approach.** Maintain a local rolling baseline per user (stored locally / opt-in); features become
relative (z-score vs personal baseline). "Insufficient baseline" state until enough sessions exist.
**Touches.** `core/baseline.py`, features, response (`insufficient_data`).
**Acceptance.** With a baseline, output reflects change-from-personal-norm; cold-start users get an
honest "not enough data yet" instead of a confident guess; baseline stays local by default.

### F7 · Confidence calibration & abstention 🟡
**Goal.** Report calibrated confidence and abstain ("insufficient/uncertain") when appropriate.
**Why it matters.** Raw `predict_proba` max is not a real probability; overconfident wellbeing
claims are harmful.
**Approach.** Calibrate (Platt/isotonic) on real validation data; define an abstention threshold and
minimum-input-quality gate; surface `insufficient_data` in the contract.
**Touches.** `predict`, `ml/evaluate.py`, response schema, UI.
**Acceptance.** Confidence is calibrated (reliability curve shown in eval report); low-quality or
too-short sessions abstain instead of guessing.
**Depends on.** F5.

### F8 · Feature-set expansion & versioning 🟡
**Goal.** Add richer, still-private timing features (e.g. burst/rhythm patterns, pause-length
distribution shape, correction-timing) with a versioned feature set.
**Why it matters.** Five aggregate features are thin; better *private* features may carry more
signal — but only real validation (F5) proves which help.
**Approach.** Add candidate features behind `FEATURE_SET_VERSION`; select via validation, not
intuition; keep every feature reconstructing-content-proof.
**Touches.** `core/features.py`, training, registry.
**Acceptance.** New feature set is versioned; each added feature justified by measured lift on real
data; privacy test proves no feature leaks content.
**Depends on.** F5.

### F9 · Drift & fairness monitoring 🟡
**Goal.** Detect input drift over time and check performance across subgroups (device, language,
typing proficiency) for large disparities.
**Why it matters.** A wellbeing model that works only for fast English typists on laptops would be
quietly discriminatory.
**Approach.** Track input distribution drift; report subgroup metrics in the eval report; flag
disparities. Requires subgroup metadata collected *with consent* in F4.
**Touches.** `ml/evaluate.py`, monitoring.
**Acceptance.** Eval report includes subgroup breakdowns and a drift indicator; large disparities
are surfaced, not hidden.
**Depends on.** F4, F5.

---

## Theme C — Structure & Quality (Claude owns the whole codebase)

### F10 · Extract the frontend from the Python string 🟡 ⭐
**Goal.** Move ~680 lines of embedded HTML/CSS/JS into real `web/` static files + templates.
**Why it matters.** An inline string literal is unmaintainable and untestable; this unblocks all UI
work (F2, F16, F17).
**Approach.** Extract to `web/` with a build-free static setup; keep behavior identical;
characterization test on the rendered page + `/api/predict` round-trip first.
**Touches.** `app.py`, new `web/`.
**Acceptance.** No HTML/JS in Python; page and prediction flow behave identically; frontend files
are independently editable.

### F11 · Kill fragile patterns: imports, globals, packaging 🟢 ⭐
**Goal.** Remove `sys.path` hacks and module-global mutable model state; make it a proper package.
**Why it matters.** These are latent bugs and the reason the app is hard to test.
**Approach.** Proper package + `pyproject.toml`; model loader/singleton replaces mutable globals;
remove unused `LogisticRegression`; fix `typing_consistency` clamp; remove emoji from logs (use
`logging`, not `print`).
**Touches.** package layout, `core/model.py`, training, logging.
**Acceptance.** No `sys.path` manipulation; models loaded via a documented loader; `pip install -e .`
works; logging replaces prints.

### F12 · Test suite incl. a privacy test 🟡 ⭐
**Goal.** Real pytest coverage of collection, features, prediction, API — plus an explicit privacy
test that asserts no character/content data can be stored or returned.
**Why it matters.** `pytest` is a dependency but no tests exist; the privacy guarantee is currently
unverified.
**Approach.** Unit tests for each module; API tests; a dedicated `test_privacy.py` that feeds events
and asserts only metadata survives anywhere in the pipeline and response.
**Touches.** `tests/`.
**Acceptance.** Meaningful coverage on core paths; privacy test fails loudly if any content-bearing
field appears; runs in CI (F13).

### F13 · CI/CD, pinning & reproducibility 🟢 ⭐
**Goal.** GitHub Actions (lint, test, privacy test, metric-qualifier check); pinned dependencies;
reproducible model builds.
**Why it matters.** No CI, minimum-only version pins, no reproducibility.
**Approach.** Pin versions (lockfile); CI matrix; seed and record randomness in synthetic gen and
training so model builds are reproducible; artifact + eval report on release.
**Touches.** `.github/workflows/`, `requirements`/lock, training seeds.
**Acceptance.** CI blocks merge on failing tests/lint/privacy/metric checks; a clean checkout
reproduces the same synthetic model; deps pinned.

### F14 · Production serving & config 🟢
**Goal.** Real WSGI server, env-driven config, `.env.example`, health/readiness endpoints.
**Why it matters.** Flask dev server in production is unsafe; config is undocumented.
**Approach.** gunicorn/waitress; `config.py` + `.env.example`; `/api/health` + `/readyz`.
**Touches.** entrypoint, config, docs.
**Acceptance.** Runs under a production WSGI server; config documented; health/readiness reflect
model state.

### F15 · Docker & one-command dev 🟢
**Goal.** Containerized app + reproducible dev environment.
**Why it matters.** No containerization today; onboarding is manual.
**Approach.** Dockerfile + compose; `make dev` / `docker compose up` runs the app and tests.
**Acceptance.** `docker compose up` serves the app; image builds in CI; no secrets baked in.
**Depends on.** F13, F14.

---

## Theme D — Product & Reach (only meaningful once the core is honest)

### F16 · Self-hosted assets & offline capability 🟢
**Goal.** Remove hard CDN dependencies (Google Fonts, Font Awesome) so the tool works offline and
leaks nothing to third parties.
**Why it matters.** CDN calls contradict the privacy-first positioning and break offline use.
**Approach.** Vendor fonts/icons locally; no external requests on the core path.
**Touches.** `web/` assets.
**Acceptance.** App fully functional with no network; no third-party requests from the page.
**Depends on.** F10.

### F17 · Personal insights & trends dashboard (local) 🟡
**Goal.** Let a *consenting* user see their own typing-metric trends over time, framed as
self-reflection — never as diagnosis.
**Why it matters.** Within-person trends are the scientifically honest, genuinely useful framing,
and they keep data local to the user.
**Approach.** Local, opt-in history; trend charts of personal metrics and deviation-from-baseline;
prominent "reflection, not diagnosis" framing; easy delete.
**Touches.** `web/`, local storage, `core/baseline.py`.
**Acceptance.** Trends shown only for opted-in users; framed as reflection with disclaimer; data is
local and deletable.
**Depends on.** F2, F6.

### F18 · Wellbeing resources & safe messaging 🟢
**Goal.** When indicators are elevated, surface supportive, non-alarming resources and clear "this
is not a diagnosis; here's where to get real help" messaging.
**Why it matters.** Duty of care: a wellbeing tool must handle a "high" result responsibly.
**Approach.** Curated, region-aware support resources; calm, non-deterministic language; never
imply certainty; encourage talking to a real person/professional.
**Touches.** UI results path, content.
**Acceptance.** Elevated results show supportive resources and a clear non-diagnostic message; tone
reviewed for harm; no alarming or deterministic phrasing.
**Constraints.** Follows the wellbeing/safe-messaging rules in `CLAUDE.md §2`.

### F19 · Explainability of a prediction 🟡
**Goal.** Show which features drove a given result (e.g. "longer pauses than your baseline"),
honestly and understandably.
**Why it matters.** Users deserve to understand a wellbeing indicator; opacity erodes trust.
**Approach.** Per-prediction feature contributions (relative to baseline where available); plain
language; always paired with uncertainty.
**Touches.** `predict`, UI.
**Acceptance.** Each result explains its main contributing features in lay terms with uncertainty;
explanations never overstate certainty.
**Depends on.** F6.

### F20 · Model card & research writeup 🟢
**Goal.** A model card documenting data, intended use, limitations, metrics-by-source, and ethical
considerations; plus an honest writeup of what the synthetic vs real results show.
**Why it matters.** This is what makes the project stand out academically — rigor and honesty,
not a fake accuracy number.
**Approach.** Standard model-card template; intended-use and out-of-scope sections; metrics labeled
by data source; known limitations and fairness notes.
**Touches.** `docs/MODEL_CARD.md`, README.
**Acceptance.** Model card complete and honest; clearly states synthetic-vs-real status; out-of-scope
uses (diagnosis, high-stakes decisions about individuals) explicitly listed.
**Depends on.** F5.

---

## Dependency summary

```
F1, F2, F3 ─▶ everything (integrity gate)
F10 ─▶ F16, F17    F11 ─▶ testable everything    F12, F13 ─▶ release quality
F4 ─▶ F5 ─▶ F7, F8, F9, F20
F6 ─▶ F17, F19
F13, F14 ─▶ F15
```

**Phase-1 core (⭐):** F1, F2, F3, F4, F5, F10, F11, F12, F13 — see `ROADMAP.md`.
The honest-core (F1–F3) plus the validity spine (F4–F5) plus the quality base (F10–F13) come first;
capability features follow only once the project tells the truth about itself.
