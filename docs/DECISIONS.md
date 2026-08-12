# DECISIONS.md — Non-obvious choices and their reasoning

A running log of decisions whose rationale is not evident from the code: model changes,
feature-set versions, validation methodology, and departures from the received brief.

Newest entries at the bottom. Each entry: context, decision, rationale, consequences.

---

## D-001 — Preserve git history rather than re-initialising

**Phase 0 · `chore/phase0-audit`**

**Context.** The takeover brief and the inherited `PROJECT_REPORT.md` both stated the repository had
no commits and instructed a clean `git init`. The audit found 7 commits on `main` tracking a live
GitHub remote, plus a second remote branch.

**Decision.** Do not re-initialise. Preserve all history and build Phase 0 as feature branches off
`main`.

**Rationale.** Re-initialising would erase the original author's (Mirdula R.) authorship record and
desynchronise from a live remote — an irreversible loss for no benefit. The stated *intent* of the
roadmap item was sound git hygiene, which branches deliver without destruction. Where a brief's
premise is factually wrong, the underlying goal wins over the literal instruction.

**Consequences.** All Phase 0 work appears as merge commits on top of existing history. Anyone
expecting a single squashed root commit will not find one.

---

## D-002 — Documentation relocated from `files/` to `docs/`

**Phase 0 · `chore/phase0-audit`**

**Context.** The doc set arrived in an untracked `files/` directory, while every cross-reference
inside those documents (`docs/CLAUDE.md §1`, `docs/ROADMAP.md`, the README doc table) pointed at
`docs/`.

**Decision.** Move the five docs to `docs/`, promote `files/README.md` to the repository root
(replacing the overstated original), and move `PROJECT_REPORT.md` into `docs/`.

**Rationale.** `files/` was a misnamed drop, not an intentional layout — the documents' own
self-references establish `docs/` as the intended location, and `ARCHITECTURE.md` §3 lists `docs/`
in the target module layout. `CLAUDE.md` §3 requires reporting moves rather than making them
silently; the mapping is recorded in `AUDIT.md` §1.2 and was reported before execution.

**Consequences.** The inherited root `README.md` (with its unqualified accuracy claims) is replaced
wholesale rather than edited. Its claims are preserved in `AUDIT.md` §5 as the F1 sweep list, so the
record of what was corrected survives the replacement.

---

## D-003 — `PROJECT_REPORT.md` corrected in place rather than deleted

**Phase 0 · `chore/phase0-audit`**

**Context.** The inherited project report contained three factual errors: the empty-git claim, a
"detects burnout" capability claim, and a bare list of ~90% metrics presented without a data source.

**Decision.** Retain the document as a marked historical baseline snapshot, correcting the three
errors inline with explicit **[corrected]** markers rather than silently rewriting or deleting it.

**Rationale.** Deleting it would destroy the record of what the project previously claimed about
itself, which is precisely the thing Phase 0 exists to correct. Silent rewriting would be
indistinguishable from the overstatement it replaces. Visible corrections leave an auditable trail:
a reader can see both the original claim and why it was wrong.

**Consequences.** The document reads as a hybrid of snapshot and correction. This is intentional and
its header says so.

---

## D-004 — `probabilities` changed from a labelled dict to an ordered array

**Phase 0 · F1 · `feat/F1-honest-metrics`**

**Context.** The inherited response returned `probabilities` as a dict keyed by label string
(`{"Low Burnout": 0.9, ...}`), while `ARCHITECTURE.md` §4.3 specifies an ordered array
(`[0.20, 0.62, 0.18]`). F1 also reworded every label, which would have silently broken any consumer
keying on the old strings.

**Decision.** Emit `probabilities` as an array ordered by class index, plus a parallel `labels` array
giving the position meanings.

**Rationale.** Following the documented contract beats preserving an undocumented one. Keying
probabilities by display text couples data to presentation: the label rewording this feature required
would have broken the frontend invisibly, returning `undefined` percentages rather than an error.
With an array plus `labels`, the frontend renders whatever the server says the classes are and never
hard-codes them — which matters again at F8 when the feature set and classes may change.

**Consequences.** A breaking change to `/api/predict` for any external consumer. Acceptable
pre-release, and the frontend was updated in the same commit.

---

## D-005 — Degenerate sessions abstain instead of being scored

**Phase 0 · F1 · `feat/F1-honest-metrics`**

**Context.** A session with no measurable duration yields an all-zero feature vector
(`feature_engineering.py:41-48`). The inherited code passed that vector to the model, which returned
a confident-looking classification derived from no information at all.

**Decision.** Add `has_sufficient_signal()`. When every feature is zero, return
`insufficient_data: true` with `prediction`, `confidence`, and `probabilities` all `null`, and an
explanation instead of a score.

**Rationale.** HARD RULE 6 forbids a silent fake result, and scoring an all-zero vector is exactly
that. The gate is deliberately minimal — all-zero only — because a real input-quality threshold
requires calibration data that does not exist yet (F7). Guessing a threshold now would substitute
one arbitrary number for another.

**Consequences.** `/api/predict` can return a `null` prediction, so every consumer must handle the
`insufficient_data` branch. The frontend styles it as an explicit non-result rather than defaulting
to the reassuring "low" styling, which the inherited `level_classes.get(..., 'low')` fallback would
otherwise have done — a non-result must never read as good news.

---

## D-006 — Localhost binding pulled forward from F3 into Phase 0

**Phase 0 · F1 · `feat/F1-honest-metrics`**

**Context.** `app.py` bound to `0.0.0.0` while its own startup banner advertised `127.0.0.1`. Host
binding formally belongs to F3 in Phase 1.

**Decision.** Fix it now: default `127.0.0.1`, overridable through `KEYSTRESS_HOST`, with a printed
warning when the override is not a loopback address.

**Rationale.** This was a live violation of HARD RULE 5, not a missing feature, and the code
contradicted its own user-facing message. Leaving a known hard-rule breach in place for a whole phase
to respect a roadmap boundary would be process over substance. The full config layer still lands in
F3/F14.

**Consequences.** Anyone relying on network access from another machine must now set
`KEYSTRESS_HOST` explicitly. That is the intended direction of the change.

---

## D-007 — The metric-qualifier check is deliberately narrow

**Phase 0 · F1 · `feat/F1-honest-metrics`**

**Context.** F1 requires a build-time check that fails when a metric string lacks a source
qualifier. The obvious implementation — flag every number near a metric-ish word — produced
false positives on CSS values (`font-size: 0.9rem`), documentation line-number tables
(`| app.py | 715 | ... confidence ... |`), and this project's own feature identifiers, where a bare
`F1` means `FEATURES.md` item 1 rather than F1-score.

**Decision.** Narrow the matcher: percentages and 0-1 decimals always count; bare integers count only
within 12 characters of the metric word and only in the 0-100 range; unit-suffixed numbers never
count; `f1` matches only in its scored forms. Provide a `metrics-ok: <reason>` escape hatch that is
inert without a written reason.

**Rationale.** A checker that cries wolf gets disabled, and a disabled checker protects nothing. The
guard is only worth having if its output is trustworthy enough that a failure is always acted on.
Requiring a reason on every exception means the escape hatch leaves an audit trail instead of a
silent bypass.

**Consequences.** The check will miss exotic phrasings such as "accuracy was ninety percent".
Accepted: `tests/test_metric_qualifiers.py` asserts against known-bad fixtures so the guard is proven
able to fail, and the residual gap is narrower than the false-positive noise the broad version
created. Documents that legitimately quote the removed claims (`docs/AUDIT.md` §5) carry explicit
exception markers.

---

## D-008 — `src/` replaced outright by the `keystress/` package, no compatibility shim

**Phase 0 · F11 · `feat/F11-package-and-loader`**

**Context.** F11 requires a real installable package. The inherited `src/` was not a package in any
meaningful sense — it was importable only because `app.py` inserted the repository root into
`sys.path` at line 18.

**Decision.** Move every module into `keystress/` with `git mv` (preserving file history), delete
`src/`, and ship no deprecation shim.

**Rationale.** A shim exists to protect external importers, and there are none: the project is
pre-release, nothing depends on `src.*`, and the only in-repo consumer was `app.py` itself. Adding a
shim would mean writing dead code on day one and then needing a second commit to remove it. `git mv`
keeps `git log --follow` working, so the history argument for a gentle transition does not apply.

**Consequences.** Any external snippet doing `from src.predict import ...` breaks immediately rather
than deprecating. Given zero known consumers, an immediate clear break beats a silent shim.

---

## D-009 — Model state moved into an injectable registry, not a module singleton

**Phase 0 · F11 · `feat/F11-package-and-loader`**

**Context.** The inherited app kept `model` and `scaler` as module-level globals, mutated by
`load_models()` and read by two request handlers. This made the app untestable without touching disk,
allowed a half-loaded state (model assigned, scaler not), and hid model identity from the response.

**Decision.** Introduce an immutable `ModelBundle` (estimator + scaler + metadata, frozen) held by a
`ModelRegistry` that swaps bundles atomically under a lock. The registry is attached to
`app.extensions` and can be injected into `create_app()`.

**Rationale.** Binding the estimator, scaler, and metadata into one frozen object makes the
partially-loaded state unrepresentable, and means a prediction can never be served without knowing
its data source — the exact failure F1 exists to prevent. Injection through the app factory is what
makes the API testable with a fixture model instead of a trained artifact, which the whole F12 test
suite depends on.

**Consequences.** Model access is now `current_app.extensions["keystress_registry"]` rather than a
bare import. Slightly more ceremony at the call site, in exchange for testability and an
unrepresentable broken state.

---

## D-010 — The synthetic generator now resamples instead of clamping (g1 → g2)

**Phase 0 · F11 · `feat/F11-package-and-loader`**

**Context.** The inherited generator bounded out-of-range draws with `max(floor, x)`. That does not
truncate a distribution — it piles every rejected draw onto one exact value. Measured on the
inherited generator: roughly 2.3% of the low-burnout class sat on exactly `0.01` for
`typing_consistency`, and about 3% of the high-burnout class on exactly `0.5` for `avg_typing_speed`.

**Decision.** Replace clamping with rejection sampling (`_truncated_normal`), scale the Beta-drawn
`backspace_ratio` rather than clipping it, and tag the data-generating process
`SYNTHETIC_GENERATOR_VERSION = "g2"`, recorded in every model version string
(`rf-v1-synthetic-g2-s42-n1500`).

**Rationale.** A point mass is a generator artifact a tree model can split on directly, letting it
score well by detecting the author's clamp rather than any pattern. In a project whose central
problem is already that synthetic scores overstate real capability, leaving an artifact that
*further* inflates them is the worst available option. Verified after the change: 1500/1500 distinct
values, none at either floor.

**Consequences.** The synthetic distribution changed, so models built before and after are not
comparable and prior artifacts must be regenerated. The generator version in the model identity makes
that visible rather than silent. Switching to `np.random.default_rng` also changes the random stream,
so seed 42 no longer reproduces the old dataset — intended, and the reason the version tag exists.

---

## D-011 — Minimum Python raised from 3.8 to 3.9

**Phase 0 · F11 · `feat/F11-package-and-loader`**

**Context.** `docs/CLAUDE.md` §4 states "Python 3.8+ (target 3.11)". Python 3.8 reached end of life in
October 2024 and receives no security fixes.

**Decision.** Set `requires-python = ">=3.9"` in `pyproject.toml`.

**Rationale.** Shipping a stated floor on an unsupported interpreter is a security position the
project cannot defend, particularly one handling data it describes as sensitive. 3.9 is the lowest
still-supported version. This is a deliberate, minimal deviation from the brief rather than an
oversight.

**Consequences.** Divergence from `CLAUDE.md` §4, recorded here rather than by silently editing the
brief. Should be reconciled when that document is next revised.

---

## D-012 — Characterization tests assert on the page *bundle*, not the `/` response body

**Phase 0 · F10 · `feat/F10-extract-frontend`**

**Context.** F10 required behaviour-preserving extraction with a characterization test written first.
The obvious form — assert that `GET /` contains `.result-icon` and `function displayResults` — passes
before extraction and fails after it, purely because the bytes moved into linked files. Such a test
does not protect the refactor; it forbids it.

**Decision.** Add a `page_bundle()` helper that fetches `/` plus every same-origin stylesheet and
script it links, and assert CSS/JS content against that. Assertions about *markup* (element IDs, the
research banner, document structure) still read the `/` body directly, since that content must remain
in the page itself.

**Rationale.** The bundle is what a browser ends up with, which is the actual observable behaviour.
The helper is indifferent to storage location but not to content: deleting a CSS rule or a JS function
still fails, and a broken asset link fails loudly because the helper asserts each linked asset returns
200. The distinction is deliberate — location-independent for assets, location-sensitive for markup
that must not be deferred behind a second request.

**Consequences.** The suite was written, run green pre-extraction, then run green post-extraction
**without modification to any assertion**. That is the evidence the extraction preserved behaviour.
The helper also gives F16 (self-hosted assets) a natural place to assert that no third-party origins
remain.

---

## D-013 — `index.html` is a Jinja template solely for `url_for`

**Phase 0 · F10 · `feat/F10-extract-frontend`**

**Context.** The extracted page needs to reference `styles.css` and `app.js`. Hardcoding
`/static/styles.css` works but silently breaks if the app is ever mounted under a path prefix.

**Decision.** Serve `web/index.html` through `render_template`, using
`url_for('static', filename=...)` for the two asset URLs and nothing else. `template_folder` points at
`web/`, `static_folder` at `web/static/`.

**Rationale.** Two `url_for` calls buy correct URLs under any mount point at effectively zero cost,
and the setup stays build-free as F10 requires — editing the frontend needs no toolchain, and the
file remains valid, readable HTML. Introducing server-side rendering of *content* would recreate the
Python/markup coupling this feature exists to remove, so the template contains no logic, no loops, and
no interpolated content.

**Consequences.** `web/index.html` is not directly openable in a browser as a file, since the two
`url_for` expressions need rendering. Acceptable: the page cannot function without the API anyway.

---

## D-014 — The privacy test is verified by injecting a leak, not by passing

**Phase 0 · F12 · `feat/F12-tests-and-privacy`**

**Context.** `tests/test_privacy.py` asserts the project's central guarantee. A privacy test that
passes tells you nothing on its own — a test asserting `True is True` also passes.

**Decision.** Validate the suite by mutation: temporarily add a content leak
(`"keys_typed": [e.get("key") for e in keystroke_events]` to the session record), confirm the suite
fails, then revert. Recorded here so the verification is repeatable rather than a one-off.

**Result.** Three independent tests caught it — the response-shape allowlist, the
identical-timing/different-content equivalence test, and the source-level scan. Independent detection
matters: a single guard can be defeated by a change that routes around it, three cannot easily be.

**Rationale.** The guarantee is the product. Evidence that the guard fires beats evidence that it is
silent. The three-layer structure is deliberate: shape assertions catch new keys, equivalence tests
catch content influencing output through any path, and source scanning catches the code being written
before it has an effect.

**Consequences.** Re-run the mutation whenever the privacy test changes shape. If a future change
makes only one layer fire, the other two have been weakened and should be investigated.

---

## D-015 — Broad `except Exception` at the deserialisation boundary

**Phase 0 · F12 · `feat/F12-tests-and-privacy`**

**Context.** `load_bundle` originally caught
`(OSError, ValueError, EOFError, AttributeError, ImportError)`. A test feeding it a corrupt artifact
showed a real `IndexError` escaping from `pickle.pop_mark` as a raw traceback — a HARD RULE 6
violation that only appeared because the test existed.

**Decision.** Catch `Exception` at that one call site, log the concrete type, and re-raise as
`ModelUnavailableError` with the original chained.

**Rationale.** `docs/CLAUDE.md` §6 forbids a *bare* `except`; this is not one, and it is confined to a
single deserialisation call. Enumerating every way a corrupt pickle can fail is a losing game — the
narrow tuple was already wrong, and the next unpickling bug would be a new escape. The correct rule at
this boundary is that no artifact problem may reach a user as a stack trace. Nothing is hidden: the
exception type is logged and the original is chained.

**Consequences.** A genuine bug inside a well-formed artifact's deserialisation is reported as
"model unavailable" rather than crashing. Acceptable — from the user's position the model is in fact
unavailable, and the operator gets the real exception in the log.

---

## D-016 — Training is single-threaded, because `n_jobs=-1` is not reproducible

**Phase 0 · F13 · `feat/F13-ci-and-pinning`**

**Context.** The inherited trainer used `RandomForestClassifier(..., n_jobs=-1)`. The new
reproducibility check failed: two builds with identical seeds produced models whose `predict_proba`
differed on one probe row.

**Investigation.** Parallel fitting varies floating-point accumulation order, so `random_state` alone
does not pin the result. Measured directly — six refits with `n_jobs=-1` produced six different
probability arrays; with `n_jobs=1`, all six were byte-identical. The divergence is in *fitting*, not
prediction: repeated `predict_proba` calls on one fitted model always agreed.

**Decision.** Set `n_jobs=1` for training.

**Rationale.** Two things were at stake, and neither is worth 0.37 seconds:

1. F13 requires that a clean checkout reproduce the same synthetic model. With `n_jobs=-1` that
   claim was simply false — and would have been shipped as true had the check not existed.
2. The confidence figure shown to a user would vary between otherwise identical runs. Tiny, but this
   project presents uncertainty as a feature; that number should not wobble for reasons unrelated to
   the input.

Measured cost of the full 1500-sample fit at `n_jobs=1`: 0.371s.

**Consequences.** Training does not use multiple cores. Irrelevant at this scale. If the dataset ever
grows by orders of magnitude, the trade-off should be re-examined and recorded here — not silently
reversed for speed.

---

## D-017 — Reproducibility is defined behaviourally, not as byte-identical artifacts

**Phase 0 · F13 · `feat/F13-ci-and-pinning`**

**Context.** The obvious implementation of "reproducible model builds" is to hash the `.pkl` files and
compare. Joblib artifacts can differ between runs for reasons unrelated to the model — memory layout,
serialisation ordering, library build details.

**Decision.** `tools/check_reproducible_build.py` builds twice in separate temporary directories and
compares: the SHA-256 of the generated dataset (byte-exact), predictions and full class probabilities
on a fixed seven-point probe set spanning the feature space (exact float equality), the reported
metrics, and the model version string.

**Rationale.** Byte comparison of pickles fails both ways: false alarms from irrelevant differences,
and — more dangerous — a green result that says nothing about whether the model behaves the same.
Behavioural equivalence is both stricter where it matters and stable where it does not. The dataset
*is* compared byte-exactly, because that is where a seeding regression appears first and there is no
excuse for it to vary.

**Consequences.** Two models that behave identically but serialise differently pass, which is the
intended semantics. A test asserting that *different* seeds produce *different* builds guards against
the check trivially passing.

---

## D-018 — CI verifies that its own guards can fail

**Phase 0 · F13 · `feat/F13-ci-and-pinning`**

**Context.** The CI pipeline enforces two guarantees that are otherwise invisible: no content capture,
and no unqualified metrics. Both are enforced by code that could itself be broken, disabled, or
emptied — and in every one of those cases CI would stay green.

**Decision.** Each guard job proves the guard is live:

- **honesty**: runs the metric checker against a known-bad fixture and fails if it passes.
- **privacy**: injects a real content leak into the privacy boundary, requires the suite to fail,
  restores the file, and then runs `git diff --exit-code` to confirm the tree was restored.
- **privacy** additionally relies on `TestSuiteIntegrity` inside the suite, which fails if the number
  of privacy tests drops — because a suite collecting zero tests passes.

**Rationale.** A green check that cannot go red is worse than no check: it manufactures confidence.
For a project whose two headline claims are "we never capture content" and "every metric states its
source", the guards deserve the same scepticism as the code.

**Consequences.** The privacy job mutates a tracked file during the run. It is confined to a
throwaway CI checkout, restored immediately, and the restoration is verified — but anyone reading the
workflow should understand it is deliberate. The suite-size floor must be raised as tests are added.

---

## D-019 — F4 becomes a crowdsourced, consent-gated donation site, not a lab study

**Phase 1 · `feat/F3-privacy-hardening`**

**Context.** `FEATURES.md` F4 describes a real-data collection harness and imagines a formal study
protocol. The project owner's intent is broader and simpler: an open-source website anyone can open,
do the typing exercise, and contribute to research on typing-dynamics vs. burnout.

**Decision.** Deliver F4 as a **public, consent-gated data-donation site**. A visitor consents, does
the typing test (metadata only), **and** completes a validated burnout questionnaire (D-020); the
paired, anonymised record is the labelled real dataset that finally breaks the synthetic
circularity. On the result page the visitor sees **their questionnaire score plus their own typing
metrics** — never a typing-model "burnout verdict", because the model is still synthetic-trained and
unvalidated (HARD RULE 2/3, `CLAUDE.md` §1). The typing indicator earns a place on the page only
after F5 validation on real data. The harness is built self-hostable first; public hosting follows
once consent, deletion, and data-handling are solid.

**Rationale.** Crowdsourcing is a legitimate, faster path to a labelled dataset than a formal study,
and the questionnaire gives each visitor genuine value without the project having to overstate the
typing model. Showing an unvalidated typing verdict would violate the honesty rules the whole
rebuild exists to enforce.

**Consequences.** The project shifts from strictly local-first (HARD RULE 5) toward hosted central
collection: a database of people's data, real deletion/withdrawal, and privacy/legal obligations
(consent, retention, likely research-ethics review). F3 (this branch) hardens the serving path in
preparation. F2 gains a real storage and deletion layer; F5 is the gate before any typing indicator
is shown.

---

## D-020 — Copenhagen Burnout Inventory as the labelling instrument

**Phase 1 · `feat/F3-privacy-hardening`**

**Context.** F4/F5 need a recognised burnout instrument to supply the ground-truth label. For a
public, open-source site, licensing is the deciding constraint: the instrument must be freely usable
and adaptable by anyone who forks the project.

**Decision.** Use the **Copenhagen Burnout Inventory (CBI)** (Kristensen et al., 2005) — a
public-domain, free-to-use, validated instrument (19 items; Personal / Work-related / Client-related
subscales; frequency responses scored 0–100 and averaged per subscale). For an academic-burnout
context the site will use the **Personal Burnout** subscale plus a **studies-adapted Work-related**
subscale and drop Client-related; the exact wording and scoring will be documented in the F4
protocol with citation.

**Rationale.** The obvious student instrument, the Maslach Burnout Inventory – Student Survey, is
**licensed and paid per administration** — incompatible with an open site anyone can run. CBI is
public domain, cross-culturally validated, and adaptable, so it fits both the science and the
licence model. Verified against primary/reference sources rather than memory.

**Consequences.** The dataset schema (F4) carries per-subscale CBI scores as the label; the study
protocol must cite Kristensen et al. and record the exact adaptation used, since a modified
instrument is no longer the validated original and must be described honestly.

---

## D-021 — F3 hardening: two payload guards, an in-memory limiter, and a deliberately loose CSP

**Phase 1 · `feat/F3-privacy-hardening`**

**Context.** F3 adds input limits, rate limiting, and security headers to a serving path that will
soon face the public internet rather than only localhost.

**Decision.** Three choices worth recording:

- **Two payload limits, not one.** `MAX_CONTENT_LENGTH` (default 1 MiB) is the outer memory guard —
  it rejects an oversized body with `413` *before* it is parsed. `MAX_KEYSTROKE_EVENTS` stays as the
  *semantic* guard for a payload that is small in bytes but absurd in event count. They protect
  different failure modes, so both remain.
- **In-memory rate-limit storage.** Correct for the single-process dev server shipped today; a
  multi-process production deployment (F14) will point the limiter at a shared backend so the limit
  holds across workers. The shared in-process counter is why the test fixtures disable rate limiting
  by default — otherwise independent tests would couple through it.
- **A CSP that still allows `'unsafe-inline'` and two CDNs.** The current page carries inline
  `onclick` handlers and `style="width:…"` attributes and pulls fonts/icons from a CDN, so a strict
  `'self'` policy would break it. The policy tightens to `'self'` with no CDN and no `'unsafe-inline'`
  in F16, which vendors the assets and moves the handlers into `app.js`. The prediction path itself
  already makes no third-party call, satisfying the F3 acceptance criterion.

**Rationale.** Ship real hardening now without breaking a working page, and be explicit about the
one directive that is weaker than it should be so it is closed deliberately in F16 rather than
forgotten.

**Consequences.** F16 must tighten `keystress/security.py` (drop the CDN and `'unsafe-inline'`
entries) once assets are local. F14 must set a shared `RATELIMIT_STORAGE_URI`.

---

## D-022 — F2 consent: an opaque token, two independent agreements, and withdrawal separate from erasure

**Phase 1 · `feat/F2-consent-flow`**

**Context.** F2 introduces the first persistence of anything about a person, and the gate that must
precede any analysis at all. Several designs would satisfy the acceptance criteria on paper; these
are the ones chosen and why.

**Decision.** Four choices worth recording:

- **An anonymous UUID as the only credential, held in `localStorage`.** No accounts, no email, no
  password. The token carries no personal content, so losing it leaks nothing — it merely makes the
  stored rows unreachable, which is a tolerable failure mode for data the participant can delete
  anyway. Accounts would mean collecting identifying information in order to protect
  non-identifying information, which is backwards for a project whose product is privacy.

- **Consent is two independent agreements, not one.** `analysis` permits processing this session;
  `donate` permits storing its features. Neither implies the other, and only `analysis` gates
  `/api/predict`. Bundling them would make "I want to try the tool" indistinguishable from "I want
  to contribute to your dataset", which is precisely the conflation informed consent exists to
  prevent. The default for both is unticked, and a test asserts neither ships pre-checked.

- **The policy wording is versioned content, and every record stores the version in force.** A
  later edit to the wording cannot retroactively reinterpret what someone agreed to; the record
  says which text it was. Bumping `CONSENT_VERSION` on a meaning-changing edit is a documented
  obligation in `keystress/core/consent.py`.

- **Withdrawal (`PATCH /api/consent/<id>`) does not delete; deletion (`DELETE /api/data/<id>`)
  does.** Turning off donation stops new storage and leaves existing rows alone. Silently
  destroying data as a side effect of a settings change would be its own unpleasant surprise, and
  a participant who wants erasure has an explicit, clearly labelled control for it. `PATCH`
  requires both fields to be stated, so "omitted" can never be ambiguous between *unchanged* and
  *withdrawn*.

**Rationale.** The acceptance criteria ("no prediction without consent … nothing stored without
opt-in … deletion is real") are the floor, not the design. Each choice above resolves a case where
a compliant-but-thoughtless implementation would still treat the participant badly.

**Consequences.** The whitelist in `Store.save_donation` is now a load-bearing privacy boundary and
must be updated deliberately when F8 versions the feature set — adding a feature there is adding a
field to what is persisted about people. F4 builds its labelled dataset on this store and will need
the CBI scores (D-020) joined to `participant_id`. F17's trends dashboard depends on donated
history existing, so it must degrade honestly for the analysis-only participant who has none.
The consent gate is enforced server-side and defaults on; `KEYSTRESS_REQUIRE_CONSENT=false` exists
for tests and must never be set in a deployment.

---

## D-023 — The CSP goes strict, and the frontend accepts the constraint that comes with it

**Phase 3 · `feat/F16-offline-and-strict-csp`**

**Context.** D-021 shipped a Content-Security-Policy that allowed `'unsafe-inline'` for both script
and style and named two CDN hosts, and said plainly that F16 would close it. Meanwhile the web
branch removed the Google Fonts and Font Awesome dependencies, so by the time the branches merged
the policy permitted three things the page no longer used and one — inline script — that it still
did. A policy that over-permits is not neutral: `script-src 'unsafe-inline'` re-authorises exactly
the injected inline script the header exists to block, so the header was mostly decoration.

**Decision.** Remove the four causes and make every directive `'self'` or `'none'`:

- **Inline handlers become a binding table in `app.js`.** Nine `onclick`/`onchange` attributes are
  gone; `CONTROL_BINDINGS` maps element id → event → function, so adding a control is adding a row.
  A missing element is logged, not thrown, so one bad id cannot unbind the other eight — and
  `tests/test_offline_assets.py` catches the bad id instead.
- **Inline styles become classes.** `style="display: none"` on the gated cards becomes `.is-hidden`,
  toggled through `showCard`/`hideCard`; `style="width: 0%"` on the probability bars moves into the
  `.fill` rule. The widths written during a result are CSSOM assignments, which CSP does not
  restrict, so the animation is unaffected.
- **`img-src` keeps `data:`.** The favicon is an inline SVG the page carries rather than fetches.
  This is the one non-`'self'` source left, and it enables no execution.
- **`CSP_DIRECTIVES` becomes public and `STRICT_DIRECTIVES` names the two that must never be
  weakened.** The policy is asserted directive by directive rather than substring-matched.

**Rationale.** The alternative — keeping `'unsafe-inline'` because the page happens to need it — is
how a security header becomes a checkbox that passes an audit and stops nothing. The work to remove
it is small and bounded, and it was already owed: D-021 recorded the debt explicitly.

**Consequences.** A real, permanent constraint on the frontend: no inline handler, no inline
`<style>`, no `style="…"` attribute may be added to `web/index.html` again. Anything that must run
belongs in `app.js`; anything that must look different belongs in a class. The failure mode is
silent — a blocked handler is a button that does nothing, with no visible error — so it is enforced
by test rather than by review. F14 must keep the policy intact when it introduces a production
server, and any future embedded chart or third-party widget is now a decision to re-open this one,
not a detail.
