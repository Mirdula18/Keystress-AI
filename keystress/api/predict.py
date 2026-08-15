"""
Prediction API endpoint.

Owns request validation, the call into the inference layer, and response assembly. The
model is reached through the app's :class:`~keystress.core.model.ModelRegistry` rather
than a module global, so handlers hold no mutable state of their own.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from ..core.baseline import BASELINE_WINDOW, build_baseline, personal_summary
from ..core.collect import process_keystroke_data
from ..core.features import extract_typing_features
from ..core.inference import get_prediction_details
from ..core.model import ModelUnavailableError
from ..extensions import limiter, predict_rate_limit

logger = logging.getLogger(__name__)

bp = Blueprint("predict", __name__)

#: Minimum events required to attempt a prediction. Below this there is not enough timing
#: signal for the aggregate features to mean anything.
MIN_KEYSTROKE_EVENTS = 5

#: Upper bound on accepted events, a semantic guard against a nonsensical session. The
#: outer memory guard is ``Settings.max_content_length``, which rejects an oversized body
#: with 413 before it is ever parsed (F3); this cap catches a payload that is small in
#: bytes but absurd in event count.
MAX_KEYSTROKE_EVENTS = 100_000

#: CSS class per indicator level. A missing prediction maps to "unknown", never to "low" —
#: a non-result must not be styled as reassuring news.
LEVEL_CLASSES = {0: "low", 1: "medium", 2: "high"}


def _error(message: str, status: int) -> tuple[Any, int]:
    """
    Build a JSON error response.

    Parameters:
        message: User-facing message. Never contains internal detail.
        status: HTTP status code.

    Returns:
        tuple: ``(response, status)``.
    """
    return jsonify({"error": message}), status


def validate_payload(payload: Any) -> tuple[Any, str]:
    """
    Validate an incoming prediction request.

    Parameters:
        payload: Parsed JSON body.

    Returns:
        tuple: ``(events, "")`` when valid, otherwise ``(None, message)``.
    """
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    events = payload.get("keystroke_events")
    if events is None:
        return None, "No keystroke data provided."

    if not isinstance(events, list):
        return None, "'keystroke_events' must be a list."

    if len(events) < MIN_KEYSTROKE_EVENTS:
        return None, (
            f"Insufficient keystroke data: at least {MIN_KEYSTROKE_EVENTS} events are "
            "needed to measure timing."
        )

    if len(events) > MAX_KEYSTROKE_EVENTS:
        return None, f"Too many keystroke events (limit {MAX_KEYSTROKE_EVENTS})."

    return events, ""


@bp.route("/api/predict", methods=["POST"])
@limiter.limit(predict_rate_limit)
def api_predict() -> tuple[Any, int]:
    """
    Score a typing session and return a burnout risk indicator.

    Accepts ``{"keystroke_events": [{"timestamp": float, "is_backspace": bool}, ...]}``.
    Any additional field in an event is ignored by the privacy boundary in
    :func:`keystress.core.collect.process_keystroke_data` and never appears in the
    response.

    Returns:
        tuple: ``(json_response, status_code)``. The success body always carries
        ``data_source``, ``model_version``, ``disclaimer``, and ``insufficient_data``.
    """
    payload = request.get_json(silent=True)

    # Consent gate (F2, HARD RULE 4): no analysis without a recorded consent. The token is
    # taken from the X-Consent-Id header, falling back to a body field so a bare fetch
    # works. Enforcement is config-gated so tests unrelated to consent can opt out.
    if current_app.config.get("KEYSTRESS_REQUIRE_CONSENT", True):
        consent_id = request.headers.get("X-Consent-Id")
        if not consent_id and isinstance(payload, dict):
            consent_id = payload.get("consent_id")
        store = current_app.extensions.get("keystress_store")
        if store is None or not store.has_analysis_consent(consent_id):
            return _error(
                "Analysis consent is required. Record consent at POST /api/consent first.",
                403,
            )

    events, message = validate_payload(payload)
    if events is None:
        return _error(message, 400)

    try:
        session_data = process_keystroke_data(events)
    except ValueError as exc:
        # Malformed timing data: the client's problem, and safe to describe.
        logger.info("Rejected malformed keystroke payload: %s", exc)
        return _error(f"Malformed keystroke data: {exc}", 400)

    features = extract_typing_features(session_data)

    try:
        bundle = current_app.extensions["keystress_registry"].get()
    except ModelUnavailableError as exc:
        # Graceful degradation (HARD RULE 6): a clear message, not a crash or a fake result.
        logger.error("Prediction unavailable: %s", exc)
        return _error(str(exc), 503)

    try:
        result: dict[str, Any] = get_prediction_details(features, bundle)
    except (ValueError, AttributeError):
        # A model that cannot score the current feature set - e.g. a feature-set mismatch.
        # Logged with the traceback for the operator; the client gets a plain message.
        logger.exception("Prediction failed for a well-formed request")
        return _error(
            "The model could not score this session. It may have been trained on a "
            "different feature set; retrain with `keystress-train`.", 500
        )

    result["level_class"] = LEVEL_CLASSES.get(result["prediction"], "unknown")

    # F6: how this session compares with the participant's *own* previous sessions. Added
    # additively (CLAUDE.md §5) and only when there is history to compare against, which
    # requires the donate opt-in — a baseline needs stored sessions, and nothing is stored
    # without one.
    personal = _personal_block(_consent_id(payload), features)
    if personal is not None:
        result["personal"] = personal

    return jsonify(result), 200


def _consent_id(payload: Any) -> str | None:
    """Extract the participant token from the header, falling back to the body."""
    consent_id = request.headers.get("X-Consent-Id")
    if not consent_id and isinstance(payload, dict):
        consent_id = payload.get("consent_id")
    return consent_id.strip() if isinstance(consent_id, str) and consent_id.strip() else None


def _personal_block(consent_id: str | None, features: dict[str, float]) -> dict[str, Any] | None:
    """
    Build the personal-baseline block for a session, or ``None`` when there can be none.

    ``None`` rather than an empty block for the two cases where a baseline is not merely
    unavailable but *inapplicable*: no participant token, or a participant who has not
    opted into storage and therefore has no history by their own choice. Sending a
    cold-start block to someone who has deliberately stored nothing would read as a nudge
    to opt in, on a page whose whole design is that opting in is genuinely optional.

    Parameters:
        consent_id: The participant token, if the request carried one.
        features: This session's features.

    Returns:
        dict | None: The block from :func:`keystress.core.baseline.personal_summary`.
    """
    if not consent_id:
        return None

    store = current_app.extensions.get("keystress_store")
    if store is None or not store.has_donate_consent(consent_id):
        return None

    # The history read here is the participant's previous sessions only: this one is not
    # donated until the client calls /api/donate, so a session is never compared with a
    # baseline it is itself part of.
    history = store.feature_history(consent_id, limit=BASELINE_WINDOW)
    baseline = build_baseline(history)
    return personal_summary(baseline, features)
