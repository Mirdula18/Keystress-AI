# Keystress-AI

**A privacy-preserving research instrument for studying whether typing dynamics relate to academic
wellbeing** — analyzing typing *rhythm* (speed, pauses, corrections) while never capturing the
characters typed. Original work by Mirdula R.

- **License:** MIT · **Stack:** Python · Flask · scikit-learn

> ⚠️ **Not a diagnostic tool.** Keystress-AI produces *research indicators*, not medical or
> clinical assessments. The current model is trained on **synthetic data**, so reported accuracy
> reflects the synthetic generator — **not** real-world burnout detection. Do not use it to
> evaluate or make decisions about any person. See `docs/CLAUDE.md §1`.

> **Status:** methodology prototype under a full rebuild toward an honest, ethical, validated
> research instrument. See the docs below.

---

## 📚 Documentation (read in this order)

| File | Purpose |
|------|---------|
| **[docs/CLAUDE.md](./docs/CLAUDE.md)** | Ownership brief: what this is/ISN'T, hard rules, Definition of Done. (`AGENTS.md` mirrors it.) |
| **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** | Current + target design, the circularity problem, module layout, data contracts. |
| **[docs/FEATURES.md](./docs/FEATURES.md)** | The 20 upgrades, each with acceptance criteria. |
| **[docs/ROADMAP.md](./docs/ROADMAP.md)** | Phased plan, starting with honesty and buildability. |
| [docs/AUDIT.md](./docs/AUDIT.md) | Phase 0 baseline audit: what was inherited and what was wrong with it. |
| [docs/DECISIONS.md](./docs/DECISIONS.md) | Running log of non-obvious decisions and their reasoning. |
| [CHANGELOG.md](./CHANGELOG.md) | Change history by feature ID. |

---

## 🧠 How it works

```
browser records {timestamp, backspace?} per keystroke   (never the characters)
   → 5 timing features (speed, inter-key delay, max pause, backspace ratio, consistency)
   → model → risk indicator + calibrated confidence + data-source label + disclaimer
```

The privacy boundary lives in data collection: only timing metadata and a correction flag ever
leave the keyboard. No character, key code, or content is recorded.

---

## 🚦 Non-negotiables (see `docs/CLAUDE.md §2`)

- **Privacy is the product** — only keystroke *metadata*; never content, ever.
- **No clinical or diagnostic claims** — risk indicators for reflection/research, with uncertainty.
- **Honest metrics only** — every number states its data source; synthetic results are labeled.
- **Informed consent & transparency** — no analysis without consent; users can delete their data.
- **Local-first** — localhost by default, no telemetry, no third-party calls on the core path.

---

## 🔬 The honest status of the model

The shipped model is trained on synthetic data whose burnout classes were **defined by hand** in the
generator. Its ~90% "accuracy" measures how separable those authored classes are — it is **not**
evidence the system detects real burnout. Establishing real performance requires consented, labeled,
real-world data (`FEATURES.md` F4) and honest, participant-grouped validation (F5). Until then, all
results are marked `data_source: synthetic`.

If validation shows the signal is weak, that is a legitimate and useful finding. The value here is
rigorous, privacy-preserving methodology — not a headline number.

---

## 🏁 Quick start

```bash
pip install -e .              # installs the `keystress` package
python -m keystress           # serves on http://127.0.0.1:5000 (loopback by default)
```

On first run, if no model exists, one is trained from synthetic data automatically. To do
it explicitly:

```bash
python -m keystress.ml.synthetic    # generate the synthetic dataset
python -m keystress.ml.train        # train and print a source-labelled evaluation report
```

Development:

```bash
pip install -e ".[dev]"
pytest                                        # tests, including the privacy test
ruff check keystress tools tests              # lint
python tools/check_metric_qualifiers.py       # no unqualified metrics anywhere
```

Configuration (all optional, safe defaults):

| Variable | Default | Purpose |
|---|---|---|
| `KEYSTRESS_HOST` | `127.0.0.1` | Bind address. Non-loopback values log a warning — raw keystroke timing is sensitive. |
| `KEYSTRESS_PORT` | `5000` | Bind port. |
| `KEYSTRESS_DEBUG` | `false` | Flask debug mode. |
| `KEYSTRESS_LOG_LEVEL` | `INFO` | Application log level. |
| `KEYSTRESS_AUTO_TRAIN` | `true` | Train from synthetic data at startup when no model is found. |
| `KEYSTRESS_MAX_CONTENT_LENGTH` | `1048576` | Max request body in bytes; larger payloads get `413` before parsing (F3). |
| `KEYSTRESS_RATE_LIMIT` | `60/minute` | Per-client limit on `/api/predict`; over-limit gets `429` (F3). |
| `KEYSTRESS_RATE_LIMIT_ENABLED` | `true` | Master switch for rate limiting (F3). |

Docker (after F15): `docker compose up`.

---

## 📁 Layout

```
keystress/
├── app.py          # Flask application factory
├── config.py       # env-driven settings; loopback default
├── api/            # HTTP layer: predict, health/readyz
├── core/           # domain: collect (privacy boundary), features, model loader, inference
├── ml/             # offline: synthetic generation, training  (never on the serving path)
└── web/            # extracted frontend (F10)
tools/              # repository checks (metric qualifiers)
tests/              # unit, API, and privacy tests
```

---

## 🤖 Build ownership

This project is a **complete takeover by Claude Code**: architecture, implementation, tests, and
docs. `docs/CLAUDE.md` is authoritative. Start with **Phase 0** of `docs/ROADMAP.md` — honesty
(remove overstated metrics) and buildability (package, extract frontend, tests, CI) — before any
capability features.

---

## 📄 License

MIT. Any wellbeing/burnout instrument used for labeling (F4) retains its own license and citation.
