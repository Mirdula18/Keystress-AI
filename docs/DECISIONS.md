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
