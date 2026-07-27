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
