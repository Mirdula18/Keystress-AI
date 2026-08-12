# ROADMAP.md — Keystress-AI Implementation Plan

Phased plan for full Claude ownership of the 20 upgrades in `FEATURES.md`. Each phase leaves the
project working and *more honest* than before.

**How to use.** You (Claude Code) own the whole build. Do Phase 0 → 4 in order; within a phase,
respect the dependency summary in `FEATURES.md`. Keep a running `DECISIONS.md`. Because there is no
second agent, you are also the reviewer — hold your own PRs to the Definition of Done.

---

## ⚠️ Phase 0 — Tell the truth + make it buildable *(before any capability work)*

The project currently overstates itself and can't be tested. Fix both.

- [x] **Repo + path audit.** Map real layout vs `CLAUDE.md §3`. Note the inline-HTML boundary,
      `sys.path` use, and global model state. Initialize git properly (history is empty today).
- [x] **F1 — Honest metrics.** Add `data_source`/`disclaimer`/`model_version` to the response;
      strip every unqualified accuracy claim from README/UI; add the metric-qualifier check.
- [x] **F11 — Kill fragile patterns.** Package layout + `pyproject.toml`; model loader replaces
      globals; remove `sys.path` hack, unused `LogisticRegression`, emoji prints; fix the
      `typing_consistency` clamp; switch to `logging`.
- [x] **F10 — Extract the frontend** from the Python string into `web/` (behavior-preserving;
      characterization test first).
- [x] **F12 — Test suite + privacy test.** Cover core modules and add `test_privacy.py` asserting
      no content data survives anywhere.
- [x] **F13 — CI + pinning + reproducible builds.** Actions run lint/tests/privacy/metric checks;
      pin deps; seed synthetic gen + training.

**Exit criteria:** no unqualified metric anywhere; app is a proper installable package with no
inline HTML or global model state; privacy test green in CI; synthetic model builds reproducibly.

> **Phase 0 complete.** All five exit criteria verified — see `CHANGELOG.md` and `docs/DECISIONS.md`
> (D-001 … D-018). 274 tests, 92% statement coverage. Three real defects were found by the new
> checks rather than by inspection: a non-reproducible model build (`n_jobs=-1`), a corrupt-artifact
> crash path, and a backtracking flaw in the metric checker itself.

---

## Phase 1 — Ethical core + validity spine ⭐

Make it safe to put in front of a person, and start breaking the synthetic circularity.

- [x] **F2 — Consent, disclaimer & deletion flow.** *Consent gate on `/api/predict` (403 without
      it), versioned policy text, separate donate opt-in, view/withdraw/delete endpoints, and a UI
      gate in front of the typing test. See `CHANGELOG.md` F2 and D-022.*
- [x] **F3 — Privacy hardening & local-first defaults** (localhost bind, validation, rate limit,
      no CDN requirement on the core path). *Body cap → 413, per-client rate limit → 429, security
      headers, prediction path CDN-free. See `CHANGELOG.md` F3 and D-021.*
- [ ] **F4 — Real-data collection harness & study protocol** (consented, labeled, content-free;
      offline research path; cite the chosen wellbeing instrument).
- [ ] **F5 — Validation harness** (participant-grouped splits, trivial baselines, calibration,
      eval report per model version).

**Exit criteria:** no analysis without consent; disclaimer always shown; a real (even if small)
labeled dataset can be collected and evaluated with honest, participant-grouped metrics reported
alongside trivial baselines. *This is the version that can be presented with integrity.*

---

## Phase 2 — Make the science better (not just bigger)

- [ ] **F6 — Per-user baseline & personalization** (deviation-from-own-norm; cold-start abstention).
- [ ] **F7 — Confidence calibration & abstention.**
- [ ] **F8 — Feature-set expansion & versioning** (selected by measured lift on real data).
- [ ] **F9 — Drift & fairness monitoring** (subgroup metrics; disparity flags).

**Exit criteria:** individual scoring reflects personal change with calibrated confidence and honest
abstention; feature additions justified by real-data lift; fairness/drift reported, not hidden.

---

## Phase 3 — Product & duty of care

- [ ] **F14 — Production serving & config** (WSGI, `.env.example`, health/readiness).
- [ ] **F15 — Docker & one-command dev.**
- [x] **F16 — Self-hosted assets & offline capability.** *No third-party request from the
      page; fonts and icons are local (inline SVG), and the CSP is strict — every directive
      `'self'` or `'none'`, closing the D-021 compromise. See `CHANGELOG.md` F16 and D-023.*
- [ ] **F17 — Personal insights & trends dashboard** (opt-in, local, reflection-framed).
- [ ] **F18 — Wellbeing resources & safe messaging** for elevated results.
- [ ] **F19 — Explainability of a prediction.**

**Exit criteria:** deployable and offline-capable; consenting users see local, reflection-framed
trends and explanations; elevated results handled with care and real-help messaging.

---

## Phase 4 — Rigor as the headline

- [ ] **F20 — Model card & honest research writeup** (metrics-by-source, intended use, out-of-scope,
      limitations, fairness).

**Exit criteria:** a model card and writeup that make the project stand out through honesty and
rigor — clearly separating synthetic from real results and naming out-of-scope uses.

---

## Suggested branch names

```
chore/phase0-audit
feat/F1-honest-metrics
feat/F11-package-and-loader
feat/F10-extract-frontend
feat/F12-tests-and-privacy
feat/F13-ci-and-pinning
feat/F2-consent-flow
feat/F4-real-data-harness
feat/F5-validation-harness
...
```

## Per-feature Definition of Done (from `CLAUDE.md §6`)

Acceptance criteria met · HARD RULES respected (privacy / no clinical claims / honest metrics /
consent / local-first) · tests incl. the privacy test · type hints · no global mutable model state ·
every surfaced metric carries `data_source` · CHANGELOG + DECISIONS updated · no secrets; user data
opt-in and deletable.

---

## Milestone framing (for a report, demo, or résumé)

- **M0 (Phase 0):** "Honest and buildable" — no overstated accuracy, real package, tested, CI.
- **M1 (Phase 1):** "Ethical and testable science" — consent, privacy, real data collection, honest
  validation. *This is the credible turning point.*
- **M2 (Phase 2):** "Personal and calibrated" — within-person modeling, calibration, fairness.
- **M3 (Phase 3):** "A careful product" — deployable, offline, reflective insights, duty of care.
- **M4 (Phase 4):** "Rigor as the differentiator" — a model card and writeup that stand on honesty.

---

## A note on scope and honesty

If real-data collection (F4) shows the typing→burnout signal is weak, **that is a valid and valuable
result** — report it. The project's worth is the rigorous, privacy-preserving methodology and the
honest evaluation of a plausible hypothesis, not a specific accuracy number. Do not tune toward an
impressive figure at the expense of truth.
