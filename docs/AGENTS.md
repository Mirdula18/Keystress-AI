# CLAUDE.md — Project Ownership Brief

> This project is a **complete takeover by Claude Code**. You own the codebase end to end:
> architecture, implementation, tests, docs, and release quality. There is no separate base-code
> agent — you write it all.
>
> Read this file, `ARCHITECTURE.md`, `FEATURES.md`, and `ROADMAP.md` before touching code.
> (`AGENTS.md` is a symlink/copy of this file for non-Claude tooling.)

---

## 1. What this project is — and what it is NOT

**Keystress-AI** infers early academic burnout risk from **typing dynamics only** — timing,
pauses, correction frequency — never the characters typed. Original work by Mirdula R.

**Read this carefully, it governs every decision below:**

The current model is trained on **synthetic data whose class labels were hand-authored by the data
generator.** `generate_synthetic_data.py` defines three burnout classes as chosen statistical
distributions; `train_model.py` then trains a classifier to recover those distributions. The
README's ~90% accuracy therefore measures *how separable the author made the synthetic classes* —
**not** any ability to detect real burnout. This is circular and must not be presented as
real-world performance.

So:
- This is a **methodology prototype and research instrument**, not a validated burnout detector.
- **You may not claim, imply, or let the UI imply clinical or diagnostic validity.** No "detects
  burnout," no accuracy number without the words "on synthetic data" attached, until real
  validation exists (see `ROADMAP.md`).
- The central goal of the roadmap is to make the project **honestly useful**: a strong, ethical,
  well-validated pipeline for *researching* the typing-dynamics→wellbeing hypothesis — with the
  scientific caveats built in, not bolted on.

If a task would require presenting the synthetic-trained model as a real detector, **stop and flag
it.**

---

## 2. Non-negotiable constraints (HARD RULES)

1. **Privacy is the product. Never capture content.** Only keystroke *metadata* (timestamps,
   backspace/correction flags, derived timing features) may be recorded — never characters, key
   codes that reveal characters, clipboard, or window/app context. Any feature that could
   reconstruct typed text is forbidden. Raw timing data is sensitive; treat it as such.

2. **No clinical or diagnostic claims.** This tool does not diagnose. Outputs are risk *indicators*
   for reflection/research, always shown with uncertainty and a "not a medical assessment"
   disclaimer. Never recommend it be used to evaluate, rank, or act on a specific person without
   their informed consent.

3. **Honest metrics only.** Every reported number states its data source. Synthetic-data results
   are always labeled as such. Real-world claims require the validation harness (F-series) and a
   real dataset. Never publish an unqualified accuracy figure.

4. **Informed consent & transparency by design.** Anyone whose typing is analyzed must know it is
   happening, what is collected, and be able to opt out and delete their data. No silent/background
   keystroke capture. No network exfiltration of raw timing data without explicit consent.

5. **Local-first, minimal exposure.** Default to localhost binding, not `0.0.0.0`. No telemetry.
   Any persistence of user data is opt-in and deletable.

6. **Graceful degradation.** Missing model, missing data, or a failed prediction must produce a
   clear message, never a crash or a silent fake result.

---

## 3. Current repository (audit before trusting this)

```
Keystress-AI/
├── app.py                      # 817 lines: Flask app + ~680 lines of HTML/CSS/JS inlined as a string
├── requirements.txt            # minimum-version pins only
├── README.md  LICENSE  .gitignore
└── src/
    ├── collect_typing_data.py  # TypingSession, process_keystroke_data()  — privacy boundary lives here
    ├── feature_engineering.py  # 5 features from session metadata
    ├── generate_synthetic_data.py  # ⚠ hand-authored class distributions (the circularity source)
    ├── train_model.py          # RandomForest (+ unused LogisticRegression); StandardScaler; joblib
    └── predict.py              # inference + formatting
```

Known defects to fix early (see `ROADMAP.md` Phase 0):
- `app.py:18` `sys.path` manipulation — fragile imports.
- Module-level mutable globals (`model`, `scaler`) mutated in `load_models()`.
- ~680 lines of HTML/CSS/JS embedded as a Python string literal.
- Unused `LogisticRegression` import; emoji in prints; `typing_consistency` clamp can misbehave.
- No tests despite `pytest` in requirements. Empty `.git/` — no history yet.

> Map to the real repo first. If paths differ, report the mapping; don't silently move files.

---

## 4. Tech stack

**Backend:** Python 3.8+ (target 3.11), Flask, scikit-learn, pandas, numpy, joblib.
**Frontend:** vanilla JS + HTML/CSS (to be extracted from the Python string into real files).
**Testing:** pytest. **New deps** must be justified in the PR and permissively licensed.

Keep the stack small. This project's virtue is that it's lean and self-contained — don't turn it
into a framework zoo. Add a real WSGI server (gunicorn/waitress) for anything past dev.

---

## 5. Data contracts (extend additively)

- **Keystroke event (client→API):** `{ t: float_ms, backspace: bool }` per event. Never add a
  field that carries character identity.
- **Session metadata:** `total_keys, backspace_count, duration, inter_key_delays[]`.
- **5 features:** `avg_typing_speed, avg_inter_key_delay, max_pause_duration, backspace_ratio,
  typing_consistency`. If you add features (F-series), version the feature set and keep the model
  aware of which version it was trained on.
- **Prediction response:** `{ prediction:int, label, confidence, probabilities[], description,
  data_source:"synthetic"|"real", model_version, disclaimer }` — `data_source` and `disclaimer`
  are **required** new fields (Hard Rule 3).

---

## 6. Definition of Done (every task)

- [ ] Meets the feature's acceptance criteria in `FEATURES.md`.
- [ ] Respects all HARD RULES (privacy, no clinical claims, honest metrics, consent, local-first).
- [ ] Tests added, including a privacy test asserting no character data is ever stored/returned.
- [ ] Type hints; no bare `except`; no module-global mutable state for models (use a loader/singleton).
- [ ] Any metric surfaced carries its `data_source`. No unqualified accuracy anywhere.
- [ ] Docstrings + `CHANGELOG.md` entry referencing the feature ID.
- [ ] No secrets committed; user data (if any) is opt-in, local, and deletable.

---

## 7. Working style

- Small, reviewable PRs; one feature per branch (`feat/F3-real-data-harness`).
- Because you own the whole project, keep a running `DECISIONS.md` of non-obvious choices
  (model changes, feature-set versions, validation methodology) so the reasoning is auditable.
- Prefer honesty over impressiveness: a smaller, well-validated claim beats a big unvalidated one.
- When the science is uncertain (it often is here), say so in the UI and docs rather than papering
  over it.

---

## 8. What NOT to do

- Do not capture or enable reconstruction of typed content — ever.
- Do not present synthetic-data performance as real-world performance.
- Do not add diagnostic/clinical language or deploy against people without consent.
- Do not bind to `0.0.0.0` by default, add telemetry, or exfiltrate raw timing data.
- Do not keep training on hand-labeled synthetic data as the *only* data source past Phase 1 —
  the roadmap's point is to move beyond it.
