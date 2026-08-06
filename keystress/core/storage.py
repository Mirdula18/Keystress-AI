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
"""


def _now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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

    # -- transparency & deletion -----------------------------------------------------

    def participant_summary(self, participant_id: str) -> dict[str, Any] | None:
        """
        Return everything stored about a participant, for the "view my data" endpoint.

        Returns:
            dict | None: Consent record plus all donations, or ``None`` if unknown.
        """
        record = self.get_participant(participant_id)
        if record is None:
            return None
        return {**record, "donations": self.list_donations(participant_id)}

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
