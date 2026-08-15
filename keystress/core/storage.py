"""
Consent and donation storage (F2).

A small SQLite store backing three ethical guarantees:

- **Consent is recorded.** Every participant is an anonymous UUID with a consent record
  stating what they agreed to and under which policy version (:mod:`keystress.core.consent`).
- **Nothing is stored without opt-in.** A prediction stores nothing. Only an explicit
  donation writes a row, and only when the participant has ``donate`` consent.
- **Deletion is real.** Deleting a participant removes their consent record *and* every
  donation; the rows are gone, not flagged.

Privacy boundary (HARD RULE 1)
------------------------------
A donation persists **only** the five derived timing features in
:data:`keystress.core.disclosure.FEATURES_V1`. :meth:`Store.save_donation` filters to
exactly that whitelist and coerces each to ``float`` before writing, so no keystroke
event, character, or unknown content-bearing field can ever reach disk — even if a caller
passes one by mistake. The privacy test asserts this.

SQLite is stdlib, so this adds no dependency (CLAUDE.md §4). A fresh connection is opened
per operation, which is the simplest thing that is safe under the threaded dev server; the
volume this handles makes connection reuse a non-issue.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .consent import CONSENT_VERSION
from .disclosure import FEATURE_SET_VERSION, FEATURES_V1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    participant_id   TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    consent_version  TEXT NOT NULL,
    consent_analysis INTEGER NOT NULL,
    consent_donate   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS donations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id TEXT NOT NULL REFERENCES participants(participant_id) ON DELETE CASCADE,
    created_at     TEXT NOT NULL,
    feature_set    TEXT NOT NULL,
    features_json  TEXT NOT NULL,
    data_source    TEXT,
    prediction     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_donations_participant ON donations(participant_id);

-- Questionnaire responses (F4). This is the *label* side of the labelled dataset: a
-- donation supplies the typing features, a response supplies the self-reported burnout
-- score, and `donation_id` pairs them.
--
-- `items_json` holds integer scale values keyed by item id and nothing else. There is no
-- free-text column here by design: a comments box is the one field that could carry
-- content, and it would be the obvious place for it to arrive.
CREATE TABLE IF NOT EXISTS responses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id     TEXT NOT NULL REFERENCES participants(participant_id) ON DELETE CASCADE,
    donation_id        INTEGER REFERENCES donations(id) ON DELETE CASCADE,
    created_at         TEXT NOT NULL,
    instrument_version TEXT NOT NULL,
    items_json         TEXT NOT NULL,
    personal_score     REAL NOT NULL,
    studies_score      REAL NOT NULL,
    overall_score      REAL NOT NULL,
    label              INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_responses_participant ON responses(participant_id);
CREATE INDEX IF NOT EXISTS idx_responses_donation ON responses(donation_id);
"""


def _now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _response_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a `responses` row to the shape the API and export share."""
    return {
        "id": row["id"],
        "donation_id": row["donation_id"],
        "created_at": row["created_at"],
        "instrument_version": row["instrument_version"],
        "item_scores": json.loads(row["items_json"]),
        "personal_score": row["personal_score"],
        "studies_score": row["studies_score"],
        "overall_score": row["overall_score"],
        "label": row["label"],
    }


class Store:
    """
    SQLite-backed store for consent records and opt-in donations.

    Parameters:
        path: Database file path. Its parent directory is created on demand. Use
            ``":memory:"`` only with a single connection — this class opens one per call,
            so an in-memory database would not persist between them; tests use a temp file.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with foreign keys on and dict-like rows."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_schema(self) -> None:
        """Create tables if they do not yet exist. Idempotent."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- consent ---------------------------------------------------------------------

    def create_participant(self, *, analysis: bool, donate: bool) -> dict[str, Any]:
        """
        Record a new consent and return the participant.

        Parameters:
            analysis: Consent to have this session's timing analysed. Required upstream.
            donate: Opt-in to store this session's timing features for research.

        Returns:
            dict: The participant record, including its generated ``participant_id``.
        """
        participant_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO participants (participant_id, created_at, updated_at, "
                "consent_version, consent_analysis, consent_donate) VALUES (?, ?, ?, ?, ?, ?)",
                (participant_id, now, now, CONSENT_VERSION, int(analysis), int(donate)),
            )
        return {
            "participant_id": participant_id,
            "created_at": now,
            "updated_at": now,
            "consent_version": CONSENT_VERSION,
            "analysis": analysis,
            "donate": donate,
        }

    def get_participant(self, participant_id: str) -> dict[str, Any] | None:
        """Return a participant record, or ``None`` if unknown."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM participants WHERE participant_id = ?", (participant_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "participant_id": row["participant_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "consent_version": row["consent_version"],
            "analysis": bool(row["consent_analysis"]),
            "donate": bool(row["consent_donate"]),
        }

    def update_consent(
        self, participant_id: str, *, analysis: bool, donate: bool
    ) -> dict[str, Any] | None:
        """
        Change what a participant consents to (including withdrawing).

        Returns:
            dict | None: The updated record, or ``None`` if the participant is unknown.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE participants SET consent_analysis = ?, consent_donate = ?, "
                "updated_at = ? WHERE participant_id = ?",
                (int(analysis), int(donate), _now(), participant_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_participant(participant_id)

    def has_analysis_consent(self, participant_id: str | None) -> bool:
        """Report whether a participant exists and consents to analysis."""
        if not participant_id:
            return False
        record = self.get_participant(participant_id)
        return bool(record and record["analysis"])

    def has_donate_consent(self, participant_id: str | None) -> bool:
        """Report whether a participant exists and consents to donation."""
        if not participant_id:
            return False
        record = self.get_participant(participant_id)
        return bool(record and record["donate"])

    # -- donations -------------------------------------------------------------------

    def save_donation(
        self,
        participant_id: str,
        features: dict[str, Any],
        *,
        data_source: str | None = None,
        prediction: int | None = None,
    ) -> int:
        """
        Persist an opt-in donation of timing features.

        Only the five :data:`FEATURES_V1` values are written, each coerced to ``float``;
        any other key in ``features`` is dropped at the boundary so content can never be
        stored (HARD RULE 1). Missing features default to ``0.0``.

        Parameters:
            participant_id: The donating participant. Must have donate consent.
            features: Extracted timing features. Filtered to the whitelist.
            data_source: The model's data source, when a prediction accompanied the donation.
            prediction: The model's class index, when one accompanied the donation.

        Returns:
            int: The new donation row id.

        Raises:
            PermissionError: If the participant lacks donate consent. The caller should
                have checked, but the store enforces it as the last line of defence.
        """
        if not self.has_donate_consent(participant_id):
            raise PermissionError("Participant has not consented to donation.")

        safe_features = {name: float(features.get(name, 0.0)) for name in FEATURES_V1}
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO donations (participant_id, created_at, feature_set, "
                "features_json, data_source, prediction) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    participant_id,
                    _now(),
                    FEATURE_SET_VERSION,
                    json.dumps(safe_features),
                    data_source,
                    prediction,
                ),
            )
            return int(cursor.lastrowid)

    def list_donations(self, participant_id: str) -> list[dict[str, Any]]:
        """Return every donation for a participant, newest first, features decoded."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM donations WHERE participant_id = ? ORDER BY id DESC",
                (participant_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "feature_set": row["feature_set"],
                "features": json.loads(row["features_json"]),
                "data_source": row["data_source"],
                "prediction": row["prediction"],
            }
            for row in rows
        ]

    def feature_history(
        self,
        participant_id: str,
        *,
        limit: int = 30,
    ) -> list[dict[str, float]]:
        """
        Return a participant's own past feature rows, newest first (F6).

        This is the input to :func:`keystress.core.baseline.build_baseline`. It reads only
        the requesting participant's rows: a personal baseline is built from one person's
        history and is meaningless to anyone else, which is a large part of why comparing
        someone against themselves is more private than comparing them against a
        population.

        Parameters:
            participant_id: Whose history to read.
            limit: Maximum rows, newest first. A baseline tracks a person as they change
                rather than averaging their whole past, so old rows are simply not read.

        Returns:
            list[dict]: Feature dictionaries, newest first. Empty for an unknown
            participant or one who has donated nothing - which is the cold-start state,
            not an error.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT features_json FROM donations WHERE participant_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (participant_id, int(limit)),
            ).fetchall()
        return [json.loads(row["features_json"]) for row in rows]

    # -- questionnaire responses (F4) ------------------------------------------------

    def save_response(
        self,
        participant_id: str,
        result: Any,
        *,
        donation_id: int | None = None,
    ) -> int:
        """
        Persist a scored questionnaire response.

        Like :meth:`save_donation`, this is gated on donate consent and filters what it
        writes. Only integer scale values keyed by a *known* item id are stored; anything
        else in ``result.item_scores`` is dropped at the boundary. A questionnaire is a
        richer object than a feature vector, so the filter matters more here, not less.

        Parameters:
            participant_id: The responding participant. Must have donate consent.
            result: A :class:`keystress.research.scoring.ScoreResult`.
            donation_id: The typing donation this response labels, when there is one.
                A response with no donation is still worth keeping - it is a valid
                questionnaire, just not part of a labelled pair - so this is nullable
                rather than required.

        Returns:
            int: The new response row id.

        Raises:
            PermissionError: If the participant lacks donate consent.
            ValueError: If ``donation_id`` belongs to a different participant. Pairing
                one person's typing with another's questionnaire would produce a
                confidently mislabelled training row, which is worse than no row.
        """
        from ..research.instrument import REQUIRED_ITEM_IDS
        from ..research.scoring import SUBSCALES

        if not self.has_donate_consent(participant_id):
            raise PermissionError("Participant has not consented to donation.")

        if donation_id is not None and not self._owns_donation(participant_id, donation_id):
            raise ValueError(
                f"Donation {donation_id} does not belong to participant {participant_id}"
            )

        safe_items = {
            item_id: int(value)
            for item_id, value in result.item_scores.items()
            if item_id in REQUIRED_ITEM_IDS
        }
        scores = result.subscale_scores

        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO responses (participant_id, donation_id, created_at, "
                "instrument_version, items_json, personal_score, studies_score, "
                "overall_score, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    participant_id,
                    donation_id,
                    _now(),
                    result.instrument_version,
                    json.dumps(safe_items),
                    float(scores[SUBSCALES[0]]),
                    float(scores[SUBSCALES[1]]),
                    float(result.overall_score),
                    int(result.label),
                ),
            )
            return int(cursor.lastrowid)

    def _owns_donation(self, participant_id: str, donation_id: int) -> bool:
        """Report whether a donation row belongs to this participant."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM donations WHERE id = ? AND participant_id = ?",
                (donation_id, participant_id),
            ).fetchone()
        return row is not None

    def list_responses(self, participant_id: str) -> list[dict[str, Any]]:
        """Return every questionnaire response for a participant, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM responses WHERE participant_id = ? ORDER BY id DESC",
                (participant_id,),
            ).fetchall()
        return [_response_row_to_dict(row) for row in rows]

    def labelled_records(self) -> list[dict[str, Any]]:
        """
        Return every donation paired with the questionnaire response that labels it.

        This is the labelled real dataset (F4) in row form:
        :mod:`keystress.research.dataset` turns it into a file, and F5 evaluates against
        it. Only paired rows are returned - a typing donation with no questionnaire has
        no label, and a questionnaire with no donation has no features.

        The join is on ``donation_id`` rather than on "same participant, nearby time",
        because a participant may contribute many sessions and guessing which
        questionnaire belongs to which session would silently mislabel rows.

        Returns:
            list[dict]: One record per labelled pair, ordered oldest first so an export
            is stable across runs. ``participant_id`` is included as the grouping key F5
            needs to keep one person out of both sides of a split.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT d.participant_id AS participant_id, d.id AS donation_id, "
                "       d.created_at AS session_created_at, d.feature_set AS feature_set, "
                "       d.features_json AS features_json, d.data_source AS data_source, "
                "       d.prediction AS prediction, "
                "       r.id AS response_id, r.created_at AS response_created_at, "
                "       r.instrument_version AS instrument_version, "
                "       r.personal_score AS personal_score, r.studies_score AS studies_score, "
                "       r.overall_score AS overall_score, r.label AS label "
                "FROM donations d "
                "JOIN responses r ON r.donation_id = d.id "
                "ORDER BY d.id ASC"
            ).fetchall()

        return [
            {
                "participant_id": row["participant_id"],
                "donation_id": row["donation_id"],
                "response_id": row["response_id"],
                "session_created_at": row["session_created_at"],
                "response_created_at": row["response_created_at"],
                "feature_set": row["feature_set"],
                "features": json.loads(row["features_json"]),
                "instrument_version": row["instrument_version"],
                "personal_score": row["personal_score"],
                "studies_score": row["studies_score"],
                "overall_score": row["overall_score"],
                "label": row["label"],
                "model_data_source": row["data_source"],
                "model_prediction": row["prediction"],
            }
            for row in rows
        ]

    # -- transparency & deletion -----------------------------------------------------

    def participant_summary(self, participant_id: str) -> dict[str, Any] | None:
        """
        Return everything stored about a participant, for the "view my data" endpoint.

        "Everything" is meant literally: every table that can hold a row about this
        person appears here. When F4 added questionnaire responses, leaving them out
        would have turned an honest transparency endpoint into a partial one — the worst
        kind, because it looks complete.

        Returns:
            dict | None: Consent record, donations, and questionnaire responses, or
            ``None`` if the participant is unknown.
        """
        record = self.get_participant(participant_id)
        if record is None:
            return None
        return {
            **record,
            "donations": self.list_donations(participant_id),
            "responses": self.list_responses(participant_id),
        }

    def delete_participant(self, participant_id: str) -> bool:
        """
        Permanently delete a participant's consent record and all their donations.

        Returns:
            bool: ``True`` if a participant was deleted, ``False`` if none existed.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM participants WHERE participant_id = ?", (participant_id,)
            )
            # ON DELETE CASCADE removes donations; PRAGMA foreign_keys is set per connection.
            return cursor.rowcount > 0
