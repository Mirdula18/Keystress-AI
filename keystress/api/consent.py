"""
Consent, transparency, and data-deletion endpoints (F2).

This blueprint owns the ethical surface that :mod:`keystress.core.storage` backs:

- ``GET  /api/consent/policy`` - the plain-language policy and its version.
- ``POST /api/consent``        - record consent, returns an anonymous ``participant_id``.
- ``POST /api/donate``         - opt-in donation of a session's timing features.
- ``GET  /api/data/<id>``      - show everything stored about a participant.
- ``DELETE /api/data/<id>``    - permanently delete it.

The ``participant_id`` is the only credential. It is an opaque UUID with no personal
content; the client holds it (e.g. in ``localStorage``) and presents it to run a
prediction, donate, view, or delete. Losing it simply means the stored data becomes
unreachable and can be left to deletion — there is nothing sensitive to leak.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from ..core.baseline import BASELINE_WINDOW, build_baseline
from ..core.collect import process_keystroke_data
from ..core.consent import CONSENT_SUMMARY, CONSENT_VERSION
from ..core.disclosure import DISCLAIMER
from ..core.features import extract_typing_features
from ..core.inference import get_prediction_details
from ..core.model import ModelUnavailableError
from ..extensions import limiter, predict_rate_limit
from .predict import validate_payload

logger = logging.getLogger(__name__)

bp = Blueprint("consent", __name__)


def _store():
    """Return the app's consent/donation store."""
    return current_app.extensions["keystress_store"]


def _error(message: str, status: int) -> tuple[Any, int]:
    """Build a JSON error response matching the shape used across the API."""
    return jsonify({"error": message}), status


def _consent_id_from_request() -> str | None:
    """
    Extract the participant token from a request.

    Preferred in the ``X-Consent-Id`` header; also accepted as a ``consent_id`` field in
    the JSON body so a simple ``fetch`` need not set custom headers.
    """
    header = request.headers.get("X-Consent-Id")
    if header:
        return header.strip()
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        value = payload.get("consent_id")
        if isinstance(value, str):
            return value.strip()
    return None


@bp.route("/api/consent/policy")
def consent_policy() -> tuple[Any, int]:
    """Return the current consent policy so the UI can display it before asking."""
    return jsonify({
        "consent_version": CONSENT_VERSION,
        "summary": CONSENT_SUMMARY,
        "disclaimer": DISCLAIMER,
    }), 200


@bp.route("/api/consent", methods=["POST"])
@limiter.limit(predict_rate_limit)
def record_consent() -> tuple[Any, int]:
    """
    Record a participant's consent.

    Body: ``{"analysis": true, "donate": false}``. ``analysis`` must be true — it is the
    consent to process the session at all, without which nothing may run (HARD RULE 4).
    ``donate`` is the separate opt-in to store timing features for research.

    Returns:
        tuple: ``(json, 201)`` with the new ``participant_id`` on success.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)

    analysis = bool(payload.get("analysis", False))
    donate = bool(payload.get("donate", False))

    if not analysis:
        return _error(
            "Analysis consent is required. Without it, nothing can be analysed.", 400
        )

    record = _store().create_participant(analysis=analysis, donate=donate)
    logger.info("Recorded consent %s (donate=%s)", record["participant_id"], donate)
    return jsonify({
        "participant_id": record["participant_id"],
        "consent_version": record["consent_version"],
        "analysis": record["analysis"],
        "donate": record["donate"],
    }), 201


@bp.route("/api/consent/<participant_id>", methods=["PATCH"])
def change_consent(participant_id: str) -> tuple[Any, int]:
    """
    Change what a participant consents to, including withdrawing it.

    HARD RULE 4 requires that a participant can *opt out*, not only delete. Withdrawing
    ``donate`` stops further storage while leaving the tool usable; withdrawing
    ``analysis`` makes ``/api/predict`` refuse the token — a full opt-out that stops
    short of erasure. Erasure is the separate, explicit ``DELETE /api/data/<id>``:
    withdrawing consent deliberately does not delete what was already donated, because
    silently destroying data on a settings change would be its own surprise.

    Body: ``{"analysis": bool, "donate": bool}``. Both are required. A partial update
    would leave "omitted" ambiguous between *unchanged* and *withdrawn*, and consent is
    the last place to tolerate that ambiguity.

    Returns:
        tuple: ``(json, 200)`` with the updated record, or 404 if unknown.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)
    if "analysis" not in payload or "donate" not in payload:
        return _error("Both 'analysis' and 'donate' must be stated explicitly.", 400)

    record = _store().update_consent(
        participant_id,
        analysis=bool(payload["analysis"]),
        donate=bool(payload["donate"]),
    )
    if record is None:
        return _error("No data found for this identifier.", 404)

    logger.info(
        "Updated consent %s (analysis=%s, donate=%s)",
        participant_id, record["analysis"], record["donate"],
    )
    return jsonify({
        "participant_id": record["participant_id"],
        "consent_version": record["consent_version"],
        "analysis": record["analysis"],
        "donate": record["donate"],
    }), 200


@bp.route("/api/donate", methods=["POST"])
@limiter.limit(predict_rate_limit)
def donate() -> tuple[Any, int]:
    """
    Store a session's timing features for research — only with donate consent.

    Accepts the same ``keystroke_events`` as ``/api/predict`` plus a participant token.
    The privacy boundary runs server-side: events become aggregate features and only those
    features are stored (:meth:`keystress.core.storage.Store.save_donation`). Raw events
    are never persisted.

    Returns:
        tuple: ``(json, 201)`` with the ``donation_id`` on success.
    """
    consent_id = _consent_id_from_request()
    store = _store()
    if not store.has_donate_consent(consent_id):
        return _error("Donation requires an explicit opt-in consent.", 403)

    payload = request.get_json(silent=True)
    events, message = validate_payload(payload)
    if events is None:
        return _error(message, 400)

    try:
        session_data = process_keystroke_data(events)
    except ValueError as exc:
        logger.info("Rejected malformed donation payload: %s", exc)
        return _error(f"Malformed keystroke data: {exc}", 400)

    features = extract_typing_features(session_data)

    # A prediction may accompany the donation, but it is optional: the research value is
    # the timing features paired with a real label (F4), not the synthetic model's guess.
    data_source: str | None = None
    prediction: int | None = None
    registry = current_app.extensions["keystress_registry"]
    try:
        bundle = registry.get()
        result = get_prediction_details(features, bundle)
        data_source = result.get("data_source")
        prediction = result.get("prediction")
    except (ModelUnavailableError, ValueError, AttributeError):
        logger.info("Donation stored without a prediction; model unavailable or mismatched.")

    donation_id = store.save_donation(
        consent_id, features, data_source=data_source, prediction=prediction
    )
    return jsonify({"donation_id": donation_id, "stored_features": list(features)}), 201


@bp.route("/api/data/<participant_id>")
def view_data(participant_id: str) -> tuple[Any, int]:
    """
    Return everything stored about a participant (transparency).

    Since F6 this includes the *derived* personal baseline as well as the stored rows.
    A baseline is not a row in a table — it is computed on demand from the donations
    already shown above it — but it is still a description of the participant that the
    system holds and acts on, and "we only show you what is literally stored" is the kind
    of narrow reading that makes a transparency endpoint less honest than it looks. The
    numbers are shown, not just their existence, so someone can check what "your usual
    pace" has been taken to mean.
    """
    store = _store()
    summary = store.participant_summary(participant_id)
    if summary is None:
        return _error("No data found for this identifier.", 404)

    baseline = build_baseline(store.feature_history(participant_id, limit=BASELINE_WINDOW))
    summary["personal_baseline"] = {
        **baseline.as_payload(),
        "centre": baseline.centre,
        "spread": baseline.spread,
        "note": (
            "Derived from the sessions above, not stored separately. Deleting your data "
            "removes it because it removes what it is computed from."
        ),
    }
    return jsonify(summary), 200


@bp.route("/api/data/<participant_id>", methods=["DELETE"])
def delete_data(participant_id: str) -> tuple[Any, int]:
    """Permanently delete a participant's consent record and every donation."""
    deleted = _store().delete_participant(participant_id)
    if not deleted:
        return _error("No data found for this identifier.", 404)
    logger.info("Deleted all data for participant %s", participant_id)
    return jsonify({"deleted": True}), 200
