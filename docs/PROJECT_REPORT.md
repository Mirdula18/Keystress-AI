# Keystress-AI — Detailed Project Report

> **Historical baseline snapshot.** This describes the codebase *as inherited*, before the Phase 0
> rebuild. It is retained for traceability. For the current state see [AUDIT.md](./AUDIT.md),
> [ARCHITECTURE.md](./ARCHITECTURE.md), and the root `CHANGELOG.md`.
>
> Three factual errors in the original snapshot are corrected inline below and marked **[corrected]**.

---

## 1. Overview

**Keystress-AI** is a privacy-preserving machine learning research prototype that *investigates
whether* typing behaviour patterns (speed, pauses, correction frequency) relate to academic burnout
— without ever capturing typed content. It is not a validated detector. **[corrected]** — the
original text read "detects early academic burnout", a capability claim unsupported by any
real-world evidence (see §8).

- **Author**: Mirdula R
- **License**: MIT (2025)
- **Language**: Python 3.8+
- **Framework**: Flask + scikit-learn
- **Status**: **[corrected]** — the original snapshot stated "No git commits yet (`.git/` exists but
  is empty)". This is false. At the time of the Phase 0 audit the repository had **7 commits** on
  `main`, tracking `https://github.com/Mirdula18/Keystress-AI.git`, with an additional remote branch
  `copilot/create-data-collection-module`. History was preserved; no re-initialization was performed.

---

## 2. Project Structure

```
Keystress-AI/
├── app.py                              # Flask web app (817 lines, self-contained)
├── requirements.txt                    # 6 dependencies
├── README.md                           # Comprehensive docs (249 lines)
├── LICENSE                             # MIT
├── .gitignore                          # Standard Python gitignore + project-specific
└── src/
    ├── __init__.py                     # Package marker
    ├── collect_typing_data.py          # Keystroke metadata capture (208 lines)
    ├── feature_engineering.py          # Raw data → 5 ML features (179 lines)
    ├── generate_synthetic_data.py      # Synthetic training data gen (130 lines)
    ├── train_model.py                  # Model training & evaluation (284 lines)
    └── predict.py                      # Inference & prediction output (209 lines)
```

**Generated at runtime** (gitignored):

- `data/synthetic_typing_data.csv` — Training dataset (1500 samples)
- `models/burnout_model.pkl` — Trained Random Forest
- `models/scaler.pkl` — StandardScaler

---

## 3. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  BROWSER (JavaScript)                                   │
│  - Records keystroke timestamps + is_backspace boolean  │
│  - Never captures actual typed characters               │
│  - POST /api/predict  {keystroke_events: [...]}        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  FLASK API  (app.py)                                    │
│  /api/predict  →  /api/health                           │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  collect_typing_data.process_keystroke_data()           │
│  → Extracts: total_keys, backspace_count, duration,    │
│    inter_key_delays                                     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  feature_engineering.extract_typing_features()          │
│  → Computes 5 features from session metadata            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  predict.get_prediction_details()                       │
│  → Scaler.transform → RF.predict + predict_proba        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  JSON Response: {prediction, label, confidence,         │
│                  probabilities, description}            │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Module Details

### 4.1 `collect_typing_data.py` — Data Collection Layer

| Component | Purpose |
|---|---|
| `TypingSession` (dataclass) | Stores timestamps list + backspace flags; computes inter-key delays |
| `TypingDataCollector` | Wrapper class for session management |
| `process_keystroke_data()` | Processes raw JSON from web frontend into session metadata dict |

**Key privacy guarantee**: Only records `timestamp` (float) and `is_backspace` (bool) per keystroke. No character data is ever stored.

### 4.2 `feature_engineering.py` — Feature Extraction

Converts raw session metadata into 5 ML features:

| Feature | Formula | Burnout Signal |
|---|---|---|
| `avg_typing_speed` | `total_keys / duration` | Lower → higher risk |
| `avg_inter_key_delay` | `mean(inter_key_delays)` | Higher → higher risk |
| `max_pause_duration` | `max(inter_key_delays)` | Higher → higher risk |
| `backspace_ratio` | `backspace_count / total_keys` | Higher → higher risk |
| `typing_consistency` | `std(inter_key_delays)` | Higher → higher risk |

Also includes: `batch_extract_features()`, `normalize_features()` (min-max), `get_feature_summary()`.

### 4.3 `generate_synthetic_data.py` — Training Data Generation

Generates 1,500 synthetic samples (500/class) using statistically modeled distributions:

| Parameter | Low Burnout (0) | Medium (1) | High (2) |
|---|---|---|---|
| `avg_typing_speed` | N(5.0, 0.8) keys/s | N(3.5, 1.0) | N(2.0, 0.8) |
| `avg_inter_key_delay` | N(0.2, 0.05) sec | N(0.35, 0.1) | N(0.6, 0.15) |
| `max_pause_duration` | Exp(1.0) + 0.5 | Exp(2.0) + 1.0 | Exp(3.5) + 2.0 |
| `backspace_ratio` | Beta(2, 20) ~9% | Beta(3, 15) ~17% | Beta(4, 10) ~29% |
| `typing_consistency` | N(0.05, 0.02) | N(0.12, 0.04) | N(0.25, 0.08) |

Values are clamped to physically realistic ranges (e.g., `backspace_ratio` in [0, 0.5]).

### 4.4 `train_model.py` — Model Training

- **Primary model**: `RandomForestClassifier` (100 trees, max_depth=10, min_samples_split=5)
- **Alternative**: `LogisticRegression` (multinomial, LBFGS solver) — defined but unused
- **Preprocessing**: `StandardScaler` on all 5 features
- **Split**: 80/20 train/test, stratified
- **Evaluation metrics**: accuracy, precision, recall, F1 (all weighted), confusion matrix, classification report
- **Persistence**: `joblib.dump()` to `models/burnout_model.pkl` + `models/scaler.pkl`

### 4.5 `predict.py` — Inference

- `predict_burnout()` — Returns `(level_int, label, description, confidence)`
- `get_prediction_details()` — Returns full dict with probabilities per class, features used, and description
- `format_prediction_output()` — CLI-friendly formatted string with ASCII bar chart
- Burnout labels: `{0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}`

### 4.6 `app.py` — Web Application (817 lines)

The Flask app is **self-contained** — the entire HTML/CSS/JS frontend is embedded as an inline string (`HTML_TEMPLATE`), no separate template files or static assets.

**Routes**:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the full SPA |
| `/api/predict` | POST | Accepts keystroke JSON, returns prediction |
| `/api/health` | GET | Health check (`{status, model_loaded}`) |

**Frontend features**:

- Real-time keystroke collection via `keydown` + `input` events
- Live stats display (keys pressed, corrections, duration, speed)
- Minimum 20 keystrokes required before analysis is enabled
- Loading spinner during prediction
- Results card with color-coded risk level, confidence %, probability bars
- Responsive design (CSS Grid + mobile breakpoint at 600px)
- External dependencies: Google Fonts (Inter), Font Awesome 6.4.0 (via CDN)

**Startup behavior**: If `models/burnout_model.pkl` is missing, the app auto-generates synthetic data and trains the model before starting.

**Security**: Debug mode defaults to `false`; controlled via `FLASK_DEBUG` env var. Host is `0.0.0.0:5000`.

---

## 5. Dependencies

```
numpy>=1.21.0          # Numerical computation
pandas>=1.3.0          # Data manipulation
scikit-learn>=1.0.1    # ML models + preprocessing
joblib>=1.2.0          # Model serialization
flask>=2.3.2           # Web framework
pytest>=7.0.0          # Testing (optional, no tests exist)
```

---

## 6. Privacy & Security Analysis

| Aspect | Status |
|---|---|
| Content capture | None — only timestamps + backspace flag |
| Data storage | All in-memory; files are model artifacts only |
| HTTPS | Not enforced (runs on HTTP) |
| CORS | Not configured |
| Rate limiting | None |
| Input validation | Minimal (checks for empty data and < 5 keystrokes) |
| Secret key | `os.urandom(24)` fallback — acceptable for dev |
| Host binding | `0.0.0.0` — exposes to network, not just localhost |

---

## 7. Identified Issues & Gaps

### Functional

1. **No tests** — `pytest` is in requirements but no test files exist
2. **No CI/CD** — No GitHub Actions workflow
3. **No production WSGI server** — Uses Flask dev server in production (`app.run()`)
4. **CORS not configured** — API cannot be consumed from different origins
5. **No input sanitization** on keystroke events beyond length check

### Code Quality

6. **`sys.path` manipulation** in `app.py:18` — fragile import approach
7. **Global mutable state** — `model` and `scaler` are module-level globals mutated in `load_models()`
8. **Emoji in print statements** (`app.py:41,57`) — may fail on some terminals
9. **Unused imports** — `LogisticRegression` is imported in `train_model.py` but never used in the main pipeline
10. **`typing_consistency` clamp creates an artificial point mass** **[corrected]** — the original
    wording ("can go negative") understated this. The clamp lives in the *generator*, not the
    feature extractor: `max(0.01, np.random.normal(0.05, 0.02))` collapses roughly 2.3% of the
    low-burnout class onto the single value `0.01`, giving the Random Forest a discrete
    class-0 fingerprint that no real typing session would ever produce. The same defect applies to
    `max(0.5, np.random.normal(2.0, 0.8))` for the high-burnout class (~3% clipped). Meanwhile
    `feature_engineering.extract_typing_features()` applies **no clamp at all**, so inference-time
    values can land in ranges the training distribution structurally excludes.

### Deployment

11. **No Dockerfile** or containerization
12. **No `.env.example`** for environment variable documentation
13. **No version pinning** — only minimum versions specified
14. **HTML embedded in Python** — ~680 lines of HTML/CSS/JS as a string literal in `app.py` (lines 61-739)

---

## 8. Model Performance — synthetic data only, not a real-world result **[corrected]**

The original snapshot reproduced the README's figures as a bare list, which read as system
performance. They are not. Every number below is measured **on synthetic data whose class labels
were hand-authored by `generate_synthetic_data.py`** — the classifier is scored on its ability to
recover distributions the generator itself defined. The figures therefore quantify *how separable
the author made three synthetic classes*, and carry **no** information about detecting real burnout.

| Metric | Score | Data source |
|---|---|---|
| Accuracy | ~90% | synthetic (hand-authored labels) |
| Precision (weighted) | ~90% | synthetic (hand-authored labels) |
| Recall (weighted) | ~90% | synthetic (hand-authored labels) |
| F1-Score (weighted) | ~90% | synthetic (hand-authored labels) |

Feature importance ranking — also **synthetic-derived**, reflecting the generator's chosen
distributions rather than any measured property of human typing:

1. `avg_typing_speed`
2. `typing_consistency`
3. `avg_inter_key_delay`
4. `max_pause_duration`
5. `backspace_ratio`

No real-world performance figure exists for this system, and none may be quoted until the F4
real-data harness and F5 validation harness produce one.

---

## 9. Summary Statistics

| Metric | Value |
|---|---|
| Total Python files | 6 |
| Total lines of Python | ~1,836 |
| Total lines of JS/HTML/CSS | ~680 (embedded in app.py) |
| ML features | 5 |
| Training samples | 1,500 (synthetic) |
| Model type | Random Forest (100 trees) |
| API endpoints | 3 (`/`, `/api/predict`, `/api/health`) |
| External CDN deps | 2 (Google Fonts, Font Awesome) |
