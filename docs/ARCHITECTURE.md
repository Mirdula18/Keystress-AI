# ARCHITECTURE.md — Keystress-AI System Design

Current architecture, target architecture after the upgrades, and the contracts that hold it
together. Read with `CLAUDE.md` and `FEATURES.md`.

---

## 1. Current architecture (baseline)

```
Browser (JS embedded in app.py)
  keydown/input → {t, backspace}[]  ── POST /api/predict ──▶ Flask (app.py)
                                                               │
                            collect_typing_data.process_keystroke_data()
                                                               │ session metadata
                            feature_engineering.extract_typing_features()  → 5 features
                                                               │
                            predict.get_prediction_details()  → scaler.transform → RF.predict_proba
                                                               │
                         JSON {prediction, label, confidence, probabilities, description}

Startup: if models/burnout_model.pkl missing → generate_synthetic_data → train_model → serve
```

### The core scientific problem (see CLAUDE.md §1)

```
generate_synthetic_data.py          train_model.py                README
  defines class distributions  ───▶   learns to separate    ───▶   "~90% accuracy"
  (low/med/high, by hand)             those same distributions       (measures the generator,
                                                                      not burnout)
```
The loop is closed on synthetic assumptions. **Breaking this loop with real data is the point of
the roadmap** — treat everything else as supporting that goal.

### Baseline weaknesses

| Area | Problem |
|------|---------|
| Validity | model recovers hand-authored labels; no ground truth; no real data |
| Metrics | accuracy reported without "synthetic" qualifier |
| Ethics | no consent flow, no disclaimer, health-adjacent inference |
| Privacy | `0.0.0.0` binding, no auth, raw timing data not treated as sensitive |
| Structure | 680 lines of HTML/JS in a Python string; `sys.path` hacks; global model state |
| Ops | no tests, no CI, no WSGI server, no containerization, no version pinning |
| Modeling | single feature set; no per-user baseline; population model applied to individuals |

---

## 2. Target architecture (after upgrades)

✦ = new or reworked.

```
        Consent gate ✦
             │
 Browser (real static files ✦) ── {t, backspace}[] ──▶ Flask API
   consent banner, disclaimer,                              │
   opt-in data donation ✦                    input validation ✦
                                                            │
                                    collect → features (versioned ✦)
                                                            │
                              ┌─────────────────────────────┴───────────────┐
                              │ Inference                                    │
                              │  population model  +  per-user baseline ✦    │
                              │  calibrated confidence ✦  +  data_source tag │
                              │  "insufficient data" path ✦                  │
                              └─────────────────────────────┬───────────────┘
                                                            │
                          response {..., data_source, model_version, disclaimer}

  Research / validation track ✦ (offline, separate from serving path)
    real-data harness ✦ → labeled study data → retrain → validation metrics ✦
    synthetic generator (kept for tests/augmentation, clearly labeled) ✦
    model registry + eval report ✦   drift & fairness checks ✦
```

---

## 3. Target module layout

```
keystress/
├── app.py                 # thin Flask entrypoint / app factory
├── config.py              # env-driven settings; localhost default; feature flags
├── api/
│   ├── predict.py         # /api/predict, validation, response assembly
│   ├── health.py          # /api/health, /readyz
│   └── consent.py         # ✦ consent + data-donation + delete endpoints
├── core/
│   ├── collect.py         # process_keystroke_data (privacy boundary — heavily tested)
│   ├── features.py        # feature extraction; FEATURE_SET_VERSION
│   ├── baseline.py        # ✦ per-user rolling baseline / personalization
│   └── model.py           # loader/singleton (replaces global mutable state)
├── ml/
│   ├── synthetic.py       # generator — kept, clearly labeled, for tests/augmentation
│   ├── train.py           # training; model registry write
│   ├── evaluate.py        # ✦ validation harness, calibration, fairness
│   └── registry.py        # ✦ versioned model artifacts + metadata
├── research/              # ✦ real-data study tooling (offline; not on serving path)
├── web/                   # ✦ extracted static frontend (html/css/js)
├── tests/                 # unit + privacy + api tests
└── docs/  DECISIONS.md
```

---

## 4. Key contracts

### 4.1 Keystroke event (privacy-bounded)

```jsonc
{ "t": 1234.5, "backspace": false }   // milliseconds since session start, correction flag
```
**Forbidden fields (never add):** key, char, code, which, target value, text length deltas that
reveal content, clipboard, focused element identity.

### 4.2 Feature set (versioned)

```python
FEATURE_SET_VERSION = "v1"
FEATURES_V1 = ["avg_typing_speed", "avg_inter_key_delay",
               "max_pause_duration", "backspace_ratio", "typing_consistency"]
```
Any change bumps the version; models record the version they were trained on and refuse mismatched
inputs.

### 4.3 Prediction response (honest-by-construction)

```jsonc
{
  "prediction": 1,
  "label": "Medium (indicator)",
  "confidence": 0.62,               // calibrated (F-series), not raw max-proba. metrics-ok: schema example value, not a measured result
  "probabilities": [0.20, 0.62, 0.18],
  "description": "...",
  "data_source": "synthetic",       // REQUIRED: "synthetic" until real validation exists
  "model_version": "rf-v1-synthetic-2025xx",
  "disclaimer": "Research indicator only — not a medical or diagnostic assessment.",
  "insufficient_data": false        // true when input is too short/noisy to score
}
```

### 4.4 Model registry entry

```jsonc
{ "model_version": "...", "trained_on": "synthetic|study-name",
  "feature_set": "v1", "metrics": {...}, "data_source": "synthetic|real",
  "created_at": "...", "notes": "..." }
```

---

## 5. Personalization note (why it matters here)

Typing speed varies enormously *between* healthy people, so a population model comparing an
individual to global class distributions is weak by design. The target adds a **per-user baseline**
(F-series): burnout signal is a *deviation from that person's own norm*, which is both more
plausible scientifically and more privacy-respecting (the comparison stays local to the user).
Population classification remains available but is explicitly the weaker path.

---

## 6. Serving vs research separation

The live serving path must stay simple, fast, and privacy-bounded. All real-data collection,
labeling, retraining, and validation live in `research/` and run offline. The serving model is only
ever updated by promoting a registry entry — never trained inline on user data during a request.
