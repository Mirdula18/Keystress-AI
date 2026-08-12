"""
Research endpoints (F4): the questionnaire that supplies the real label.

- ``GET  /api/instrument``     - the questionnaire, its scales, and its provenance.
- ``POST /api/questionnaire``  - submit answers; returns the participant's own scores.

Why this is a separate blueprint from ``consent``
-------------------------------------------------
Consent is infrastructure every path depends on. This is the research path — the one
place the project asks a person for something *about themselves* rather than about their
typing, and the only path that produces labelled data. Keeping it separate makes the
boundary visible: if this blueprint were not registered, the site would still work as a
typing demo and would collect no self-report at all.

What the participant gets back (D-019)
--------------------------------------
Their **own questionnaire score**, banded, with the caveat attached — and nothing derived
from the typing model. The model is synthetic-trained and unvalidated, so presenting its
guess next to a real self-report score would invite exactly the inference this project
refuses to support: that the typing model measured the same thing. The typing indicator
earns a place beside this score only after F5 validates it on real data.

Storage is gated twice over: an answer is only stored if the participant holds a donate
opt-in, and :meth:`keystress.core.storage.Store.save_response` refuses regardless if they
do not. Someone who wants to see their own score without contributing it simply gets the
score — that is a legitimate way to use the site, and it costs the research nothing to
allow it.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from ..extensions import limiter, predict_rate_limit
from ..research.instrument import as_payload as instrument_payload
from ..research.scoring import score_responses

logger = logging.getLogger(__name__)

bp = Blueprint("research", __name__)


def _store():
    """Return the app's consent/donation store."""
    return current_app.extensions["keystress_store"]


def _error(message: str, status: int) -> tuple[Any, int]:
    """Build a JSON error response matching the shape used across the API."""
    return jsonify({"error": message}), status


def _consent_id_from_request() -> str | None:
    """
    Extract the participant token, header first then body.

    Deliberately identical to the consent blueprint's helper rather than imported from
    it: these are two independent surfaces, and the day one grows a different token rule
    it should not silently change the other.
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


@bp.route("/api/instrument")
def instrument() -> tuple[Any, int]:
    """
    Return the questionnaire and everything needed to present it honestly.

    Unauthenticated on purpose: the wording of a questionnaire someone is about to be
    asked to complete is not a secret, and being able to read it before consenting is
    part of informed consent, not a leak.

    Returns:
        tuple: ``(json, 200)`` with items, scales, citation, adaptation note, disclaimer.
    """
    return jsonify(instrument_payload()), 200


@bp.route("/api/questionnaire", methods=["POST"])
@limiter.limit(predict_rate_limit)
def submit_questionnaire() -> tuple[Any, int]:
    """
    Score a completed questionnaire, and store it when the participant has opted in.

    Body: ``{"responses": {"p1": 75, ...}, "donation_id": 12}``. ``donation_id`` is
    optional and names the typing session these answers label.

    Requires analysis consent — the same gate ``/api/predict`` uses. A questionnaire is
    processing someone's data about themselves, so it may not run for a token the
    participant has withdrawn.

    Returns:
        tuple: ``(json, 200)`` with the participant's subscale scores, band, and caveat.
        ``stored`` says plainly whether the answers were kept, so donation is never
        silent in either direction.
    """
    consent_id = _consent_id_from_request()
    store = _store()

    if current_app.config.get("KEYSTRESS_REQUIRE_CONSENT", True) and (
        not store.has_analysis_consent(consent_id)
    ):
        return _error("This questionnaire requires your recorded consent.", 403)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)

    responses = payload.get("responses")
    if not isinstance(responses, dict):
        return _error("Field 'responses' must be an object of item id to value.", 400)

    try:
        result = score_responses(responses)
    except ValueError as exc:
        # Incomplete, unknown item, or an off-scale value. All are the client's problem
        # and all would corrupt the dataset if accepted.
        logger.info("Rejected questionnaire submission: %s", exc)
        return _error(str(exc), 400)

    donation_id = payload.get("donation_id")
    if donation_id is not None and not isinstance(donation_id, int):
        return _error("Field 'donation_id' must be an integer when present.", 400)

    stored = False
    if store.has_donate_consent(consent_id):
        try:
            store.save_response(consent_id, result, donation_id=donation_id)
            stored = True
        except ValueError as exc:
            # A donation id belonging to someone else. Refuse rather than store the
            # response unpaired: a client sending the wrong id is confused about which
            # session it is labelling, and guessing on its behalf risks a wrong label.
            logger.warning("Refused to pair questionnaire with donation: %s", exc)
            return _error("That typing session does not belong to this participant.", 400)

    response = result.as_payload()
    response["stored"] = stored
    response["storage_note"] = (
        "Your answers were stored for research. You can view or delete them at any time."
        if stored else
        "Your answers were not stored. They were scored and discarded."
    )
    return jsonify(response), 200
