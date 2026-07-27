"""
Health and readiness endpoints.

``/api/health`` reports liveness plus the identity of the loaded model. Reporting the
model version and data source here means an operator can never be unsure whether a
running instance is serving synthetic-trained predictions (F1).

``/readyz`` reports whether the service can actually serve a prediction, which is a
different question from whether the process is alive.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, jsonify

from ..core.disclosure import DISCLAIMER

bp = Blueprint("health", __name__)


def _registry():
    """Return the app's model registry."""
    return current_app.extensions["keystress_registry"]


@bp.route("/api/health")
def health_check() -> tuple[Any, int]:
    """
    Report service liveness and the identity of the loaded model.

    Returns:
        tuple: ``(json_response, 200)``. Always 200 while the process is alive; use
        ``/readyz`` to ask whether it can serve.
    """
    bundle = _registry().get_or_none()

    body: dict[str, Any] = {
        "status": "healthy",
        "model_loaded": bundle is not None,
        "disclaimer": DISCLAIMER,
    }

    if bundle is None:
        body.update({
            "model_version": None,
            "data_source": None,
            "feature_set": None,
        })
    else:
        body.update({
            "model_version": bundle.model_version,
            "data_source": bundle.data_source,
            "feature_set": bundle.feature_set,
        })

    return jsonify(body), 200


@bp.route("/readyz")
def readiness_check() -> tuple[Any, int]:
    """
    Report whether the service can serve a prediction.

    Returns:
        tuple: ``(json_response, 200)`` when a model is loaded, otherwise
        ``(json_response, 503)``.
    """
    bundle = _registry().get_or_none()

    if bundle is None:
        return jsonify({
            "ready": False,
            "reason": "No trained model is loaded; predictions are unavailable.",
        }), 503

    return jsonify({
        "ready": True,
        "model_version": bundle.model_version,
        "data_source": bundle.data_source,
    }), 200
