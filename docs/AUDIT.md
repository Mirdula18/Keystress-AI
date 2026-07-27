# AUDIT.md — Phase 0 Baseline Audit

Audit of the inherited codebase, performed before any Phase 0 code changes. Records the starting
state so every subsequent change is traceable to a named defect.

**Audited commit:** `128cd6a` ("added requirements.txt") on `main`.

---

## 1. Two corrections to the received brief

### 1.1 Git history was not empty

Both the takeover brief and `PROJECT_REPORT.md` stated that `.git/` existed but held no commits, and
called for a clean `git init`. This was false. The repository at audit time held **7 commits** and a
live remote:

```
origin  https://github.com/Mirdula18/Keystress-AI.git
128cd6a added requirements.txt
bdaa3c4 Merge pull request #1 from Mirdula18/copilot/create-data-collection-module
11f00f1 Fix Flask debug mode security issue
f4945d9 Address code review feedback: use joblib, fix security issues
422ede7 Add complete AI-based burnout detection system with Flask UI
87a4ac6 Initial plan
4cbf67b Initial commit
```

A re-initialization would have destroyed the original author's authorship history and desynchronised
from a live remote carrying an additional branch (`copilot/create-data-collection-module`).
**Decision: history preserved, no re-init.** See `DECISIONS.md` D-001.

### 1.2 Documentation lived in `files/`, not `docs/`

The doc set shipped in an untracked `files/` directory while every internal cross-reference pointed
at `docs/`. Relocated as part of this branch:

| From | To |
|---|---|
| `files/{CLAUDE,ARCHITECTURE,FEATURES,ROADMAP,AGENTS}.md` | `docs/` |
| `files/README.md` | `README.md` (replacing the overstated root README) |
| `PROJECT_REPORT.md` | `docs/PROJECT_REPORT.md` (factual errors corrected inline) |

---

## 2. Real layout vs `CLAUDE.md` §3

| §3 claims | Reality at audit |
|---|---|
| `app.py` 817 lines, ~680 inline HTML | Confirmed — 817 lines; inline string is 677 lines |
| `requirements.txt` minimum-version pins | Confirmed — 6 deps, all `>=`, no lockfile |
| `src/` with 5 modules | Confirmed, plus an undocumented `src/__init__.py` |
| `data/`, `models/` | **Neither exists.** Both are gitignored (`.gitignore:210-211`) and created at first run, so a clean checkout always retrains — a reproducibility gap addressed by F13 |
| "Empty `.git/` — no history" | False, see §1.1 |
| Docs under `docs/` | False, see §1.2 |

## 3. Inline-HTML boundary (F10 extraction seam)

| Item | Location |
|---|---|
| String literal `HTML_TEMPLATE = '''` … `'''` | `app.py:61-739` |
| HTML content | `app.py:62-738` (677 lines) |
| ├─ `<style>` | `app.py:70-463` (394 lines CSS) |
| ├─ `<body>` markup | `app.py:465-586` (122 lines) |
| └─ `<script>` | `app.py:588-736` (149 lines JS) |

**Complete reference list:** one call site — `index()` at `app.py:742-745`, which returns the raw
string rather than using `render_template`. No other route, helper, or test referenced it, making
this a clean single-seam extraction.

Dead imports in the same module: `render_template` (`app.py:20`), `numpy as np` (`app.py:21`),
`predict_burnout` and `BURNOUT_LABELS` (`app.py:25`).

Two CDN `<link>` tags at `app.py:68-69` (Google Fonts, cdnjs Font Awesome) issue third-party
requests on page load, contradicting the local-first rule. Tracked as F16; deliberately left intact
during F10, which is behaviour-preserving.

## 4. Fragile patterns (F11 targets)

| Pattern | Location at audit |
|---|---|
| `sys.path` manipulation | `app.py:15`, `app.py:18` |
| Module-level mutable globals | `app.py:31-32` (`model`, `scaler`) |
| …mutated by | `load_models()` `app.py:35-57` — `global` at :37, assigned :40, :55-56 |
| …read by | `api_predict()` :758 (declares `global` but never assigns — redundant), used :779; `health_check()` :797 |
| Unused `LogisticRegression` | `src/train_model.py:19` — referenced only by `train_logistic_regression()` (`:109-129`), which nothing calls |
| Emoji in prints | `app.py:41,43,57,803,812-815`; `src/predict.py:170-185` |
| `print` instead of `logging` | all five `src/` modules and `app.py`; exceptions printed at `app.py:788` |
| `typing_consistency` clamp | Generator-side `max(0.01, ...)` at `generate_synthetic_data.py:59,77,95` — artificial point mass, see `PROJECT_REPORT.md` §7 item 10. Extractor applies no clamp |

Additional defects found during audit, not in the received list:

- **`host='0.0.0.0'`** (`app.py:817`) violates HARD RULE 5 while the startup banner (`app.py:813`)
  advertises `http://127.0.0.1:5000` — code and message disagree. Pulled forward from F3 into
  Phase 0 as a one-line hard-rule fix.
- **`typing_consistency` is misnamed.** It is a standard deviation, so a higher value means *less*
  consistency; README and UI read it in the opposite direction. Feature-semantics issue for F8.
- `app.secret_key` (`app.py:28`) regenerates per restart and is currently unused; becomes relevant
  at F2.
- Zero-duration sessions yield all-zero features (`feature_engineering.py:41-48`) which are then
  scored as a confident prediction rather than abstaining — a HARD RULE 6 gap. Full abstention logic
  is F7; Phase 0 surfaces an honest `insufficient_data` field.

## 5. Metric-claim sweep (F1 target list)

| File | Line | Claim |
|---|---|---|
| README.md | 10 | "**detects** early academic burnout" |
| README.md | 35 | "Provides real-time burnout risk assessment" |
| README.md | 43 | "Random Forest classifier with **85%+ accuracy**" |
| README.md | 75 | "**~90% accuracy**, balanced precision/recall" |
| README.md | 196-199 | Performance table: accuracy/precision/recall/F1 all ~90%, unqualified |
| README.md | 202-206 | Feature-importance ranking presented as established fact |
| app.py | 469 | UI tagline "**Detect** academic burnout through typing patterns" |
| app.py | 533 | Hardcoded placeholder "Confidence: 85%" in results markup |
| app.py | 715 | Renders confidence with no data-source qualifier |
| app.py | 541 | "Risk Assessment Breakdown" |
| app.py | 803 | Startup banner "Academic Burnout **Detection**" |
| src/predict.py | 22-26 | Labels are bare "Low/Medium/High Burnout" — asserts a state |
| src/predict.py | 29-36 | Descriptions assert the user's condition |
| src/predict.py | 170-177 | Formatted CLI output: unqualified risk level and confidence |
| src/train_model.py | 250-253 | Prints accuracy/precision/recall/F1 with no source qualifier |
| PROJECT_REPORT.md | 225-228 | ~90% × 4, unqualified |

The prediction response (`src/predict.py:142-153`) carried **none** of `data_source`,
`model_version`, or `disclaimer`, all three required by `CLAUDE.md` §5.

## 6. Privacy boundary — verified content-free

`process_keystroke_data()` (`src/collect_typing_data.py:145-184`) reads exactly two keys:
`event['timestamp']` (:168) and `event.get('is_backspace', False)` (:169). Nothing else is read,
stored, or returned.

```
JS keydown (app.py:601-616)
  event.key === 'Backspace'  ->  boolean only; key identity discarded at the source
  push {timestamp: performance.now()/1000, is_backspace: bool}
    -> POST /api/predict (app.py:666-680)
      -> process_keystroke_data (app.py:772-773)
        -> {total_keys, backspace_count, duration, inter_key_delays[], start_time, end_time}
          -> extract_typing_features (app.py:776) -> 5 floats
            -> get_prediction_details (app.py:779)
              -> response {prediction, label, description, confidence, probabilities, features}
```

No character, key code, textarea value, clipboard content, or window context enters the pipeline at
any point. `performance.now()` is relative to page load, so timestamps carry no wall-clock
fingerprint. The `features` field returned to the client is five aggregate floats.

Three findings for F12/F3:

1. **The guarantee is enforced client-side only.** The server ignores unknown fields rather than
   rejecting them, so a modified client could POST `{timestamp, is_backspace, char: "p"}` and the
   request would be accepted. Nothing persists it, so there is no leak today, but the guarantee
   belongs in a server-side allowlist schema rather than in client convention. The privacy test
   covers this case explicitly.
2. `total_keys` is a length signal. It is inherent to the method and not reconstructive, but §4.1
   forbids length deltas that reveal content, so it is documented as an accepted bounded disclosure.
3. `TypingSession.record_keypress` (`src/collect_typing_data.py:40`) uses wall-clock `time.time()`.
   It is reachable only from the unused local demo path, never from a web request.

## 7. Git and tooling state

Branch `main`, working tree clean apart from the untracked `files/` and `PROJECT_REPORT.md`. No
tags, no CI workflows, no `tests/` directory despite `pytest>=7.0.0` being a declared dependency.
Python 3.11.9 with all six dependencies importable — the application builds and runs as inherited.
