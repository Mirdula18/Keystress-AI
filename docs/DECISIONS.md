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
