# STUDY_PROTOCOL.md — collecting the real dataset (F4)

This is the protocol for the only data collection Keystress-AI does: pairing a consented,
content-free typing session with a self-reported burnout score. It covers what is
collected, from whom, under what consent, how long it is kept, and how someone withdraws.

It is written to be read by three people: a participant who wants to know what happens to
their data, a researcher deciding whether the resulting dataset is worth anything, and
whoever runs a copy of this site next.

> **Status: no data has been collected yet.** This protocol describes the harness that
> exists in the code, not a study that has run. Every claim below is about what the
> software does, and can be checked against `keystress/research/` and `tests/`.

---

## 1. Why collect anything at all

The shipped model is trained on synthetic data whose burnout classes were written by hand
in `keystress/ml/synthetic.py`. A classifier trained to recover authored classes tells you
how separable the author made them — nothing about people (`docs/CLAUDE.md` §1). Every
other feature in this project is infrastructure around that hole.

The only thing that fills it is real sessions with real labels. This protocol is how they
are obtained, and D-019 fixes the shape: **a public, consent-gated donation site**, not a
lab study. Anyone can open it, do the typing exercise, answer the questionnaire, and see
their own score. If they opt in, the pair becomes a row in the research dataset.

**What a participant gets:** their own questionnaire score, banded and explained. **What
they do not get:** a typing-model verdict. The model is synthetic-trained and unvalidated;
showing its guess beside a real self-report would invite the reader to conclude the two
measure the same thing. That comparison is F5's job, after validation, and only if it
survives it.

---

## 2. What is collected

### 2.1 Typing side — timing only

Per keystroke, the browser records two things: a timestamp and whether the key was
Backspace. Key identity is discarded in the browser, before anything is sent
(`web/static/app.js`, and `keystress/core/collect.py` re-enforces it server-side because a
modified client can send anything).

Those events are reduced to five aggregate features before storage:

| Feature | Meaning |
| --- | --- |
| `avg_typing_speed` | keys per second over the session |
| `avg_inter_key_delay` | mean gap between consecutive keys |
| `max_pause_duration` | longest gap |
| `backspace_ratio` | corrections ÷ total keys |
| `typing_consistency` | standard deviation of inter-key delays (higher = *less* consistent) |

**Raw events are never stored.** The donation endpoint extracts features and persists only
those. There is no table, column, or log line anywhere that holds a keystroke.

**One accepted disclosure:** `avg_typing_speed` and the session duration together imply
roughly how many keys were pressed. That is inherent to timing analysis, cannot
reconstruct content, and is recorded in `docs/AUDIT.md` §6 rather than left implicit.

### 2.2 Label side — the questionnaire

A studies-adapted short form of the **Copenhagen Burnout Inventory** (Kristensen et al.,
2005). Thirteen items: six personal-burnout, seven studies-related. Each is answered on a
five-point scale anchored at 0, 25, 50, 75, 100; one item ("enough energy for family and
friends") is reverse-scored. A subscale score is the mean of its items; the overall score
is the mean across all thirteen.

**Instrument choice (D-020).** The CBI is free to use and adapt with citation, which is
what makes an open, self-hostable site possible. The obvious student alternative — the
Maslach Burnout Inventory Student Survey — is licensed per administration, so a fork of
this repository could not legally run it.

**Adaptation, stated plainly.** Two changes: the client-related subscale is dropped (a
student has no clients), and the work-related items are re-worded to ask about studies.
**The psychometric evidence for the CBI was established for the CBI, not for this
shortened re-wording.** No claim is made that this version is validated. Any publication
using this dataset must describe the instrument as adapted and say how.

**Before you field this with participants**, verify the item wording against Kristensen et
al. (2005) directly. The wording in `keystress/research/instrument.py` was checked against
public reference descriptions of the instrument, and published versions differ over which
option set the last three studies-related items take. Both sets map to identical anchors,
so subscale means are unaffected — but "close enough" is not a standard a study should
adopt. If you change anything, bump `INSTRUMENT_VERSION`.

### 2.3 What is never collected

No name, email, account, age, gender, institution, IP-derived location, or device
fingerprint. No free-text field exists anywhere in the flow — deliberately, because a
comments box is the one control that could carry content and is where content would
eventually arrive. No third-party request is made by the page at all (F16), so no analytics
or CDN provider learns that a visit happened.

The absence of demographics has a real cost: **subgroup fairness analysis (F9) is
impossible on this dataset.** That is a deliberate trade — collecting sensitive attributes
from anonymous volunteers to check for bias creates the very risk it investigates — and it
must be stated as a limitation rather than quietly ignored.

---

## 3. Consent

Consent is two independent agreements (D-022), neither pre-ticked:

1. **Analysis** — required. Permits processing this session's timing. Without it
   `/api/predict` and `/api/questionnaire` refuse outright (403), enforced server-side.
2. **Donation** — optional. Permits *storing* the session's features and questionnaire
   answers for research. Without it, everything is scored in memory and discarded.

The policy wording is versioned (`CONSENT_VERSION`) and every record stores the version in
force when it was given, so a later edit cannot retroactively reinterpret what someone
agreed to.

**Identity.** A participant is one opaque UUID minted at consent, held in the browser's
`localStorage`. There is no account and no way to link a participant to a person. Losing
the token does not expose anything — it makes the rows unreachable, which is a tolerable
failure mode for data the participant could delete anyway.

**Minors and vulnerable participants.** This protocol assumes adult volunteers who can
consent for themselves. Deploying it to a population that cannot — school pupils, or any
group where a teacher or employer could apply pressure to participate — requires
institutional ethics review first, and the "anyone can open it" framing stops being
adequate.

---

## 4. Retention, withdrawal, and deletion

| Action | Effect |
| --- | --- |
| Withdraw donation (`PATCH /api/consent/<id>`) | Stops new storage. Existing rows are untouched. |
| Withdraw analysis | The token stops working; nothing further can be analysed or answered. |
| Delete (`DELETE /api/data/<id>`) | Consent record, every donation, and every questionnaire response are removed. Rows are deleted, not flagged. |
| View (`GET /api/data/<id>`) | Returns everything held about that participant, verbatim. |

Withdrawal is deliberately *not* deletion. Silently destroying data as a side effect of a
settings change would be its own unpleasant surprise; someone who wants erasure has an
explicit, clearly labelled control for it.

**Retention.** Data is kept until the participant deletes it or the operator retires the
deployment. There is no automatic expiry, because a research dataset that silently
evaporates is not reproducible — but note the consequence honestly: *indefinite retention
of donated data is only acceptable while deletion stays genuinely available.* An operator
who takes the site down owes participants either a deletion window announced in advance or
destruction of the database.

**Exports are generated, not accumulated.** `keystress-export` reads the live store each
time, so a deleted participant disappears from every future export. Any export file
already written is a copy outside the deletion path — treat exported files as
distributions, keep track of them, and delete them when you delete the store. This is the
weakest link in the deletion promise and it is a handling obligation, not a software one.

---

## 5. The dataset

`keystress-export --out data/labelled_sessions.csv` produces `labelled-v1`:

```
participant_id, donation_id, response_id,
session_created_at, response_created_at, feature_set,
avg_typing_speed, avg_inter_key_delay, max_pause_duration,
backspace_ratio, typing_consistency,
instrument_version, personal_score, studies_score, overall_score, label
```

- `participant_id` is the **grouping key**. One person may contribute many sessions, so
  any evaluation must split by participant, never by row (F5). Dropping this column makes
  the dataset unusable for honest evaluation, so it is never dropped when sharing.
- `overall_score` (0–100) is the label to prefer. `label` (0/1/2) is the banded version —
  below 50, 50–74, 75+ — provided because the current model is a three-class classifier.
  The bands are a reporting convention from the CBI literature, **not diagnostic
  thresholds**.
- The synthetic model's own prediction is **not** exported, though the store keeps it for
  auditing. A file holding both the real label and the current model's guess is one
  careless join away from training on its own output.

Every export prints its own warnings — too few sessions, too few participants, a class
that never appears, one participant dominating — under the heading **"THIS DATASET IS NOT
YET EVIDENCE"**.

---

## 6. Threats to validity, stated up front

A dataset collected this way has real limitations, and a paper that hides them is worse
than no paper:

- **Self-selection.** People who click a burnout site are not a random sample of students,
  and are plausibly more burned out than average.
- **One session, one score.** The CBI asks about recent weeks; a typing session is two
  minutes. Any correlation is between a momentary measurement and a retrospective one.
- **Uncontrolled conditions.** Keyboard, device, posture, time of day, and whether someone
  was interrupted are all unknown and all affect typing timing — plausibly more than
  burnout does.
- **A fixed prompt.** Everyone types the same sentence, which controls content but means
  the task is copying, not composing. Whether copy-typing rhythm carries the same signal as
  natural writing is untested.
- **No demographics** (§2.3), so no subgroup analysis is possible.
- **An adapted instrument** (§2.2), so the CBI's published properties do not transfer
  automatically.

**If the data shows the typing→burnout signal is weak, that is the result.** It is a
useful one, it gets reported, and it is not a reason to keep tuning until a number looks
better (`ROADMAP.md`, "A note on scope and honesty").

---

## 7. Running a collection yourself

1. Read this file and `docs/CLAUDE.md` §2 in full.
2. Verify the instrument wording against the primary source (§2.2).
3. Decide whether your context needs ethics review (§3) — if participants are recruited
   through an institution, assume yes.
4. Deploy with `KEYSTRESS_REQUIRE_CONSENT=true` (the default; it exists as a switch for
   tests and must never be false in a deployment).
5. Keep the store somewhere backed up but access-controlled: it holds no identities, but
   it does hold people's self-reported wellbeing.
6. Export with `keystress-export`, read the warnings it prints, and evaluate with F5's
   participant-grouped harness — never with a plain random split.

---

## References

Kristensen, T. S., Borritz, M., Villadsen, E., & Christensen, K. B. (2005). The Copenhagen
Burnout Inventory: A new tool for the assessment of burnout. *Work & Stress*, 19(3),
192–207. https://doi.org/10.1080/02678370500297720

Related decisions: **D-019** (crowdsourced donation site rather than a lab study),
**D-020** (instrument choice), **D-022** (consent design), **D-024** (F4 harness design).
