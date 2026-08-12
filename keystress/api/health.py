"""
Health and readiness endpoints.

``/api/health`` reports liveness plus the identity of the loaded model. Reporting the
model version and data source here means an operator can never be unsure whether a
running instance is serving synthetic-trained predictions (F1).

``/readyz`` reports whether the service can actually serve a prediction, which is a
different question from whether the process is alive.

Both now also report the model's **validation status** (F5). "A model is loaded" and "that
model has ever been tested against real people" are different facts, and an operator
reading a healthy-looking health check should not have to infer the second from the first.
The status is derived from the evaluation reports on disk, never set by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify

from ..core.disclosure import DISCLAIMER
from ..ml.validation import disclosure_for, find_report

bp = Blueprint("health", __name__)


def _registry():
    """Return the app's model registry."""
    return current_app.extensions["keystress_registry"]


def _report_dir() -> Path:
    """
    Return the directory holding evaluation reports.

    Resolved from the model path rather than the working directory, so a service started
    from anywhere still finds the reports that sit beside its artifacts.
    """
    configured = current_app.config.get("KEYSTRESS_EVAL_REPORT_DIR")
    if configured:
        return Path(configured)
    return Path(current_app.config["KEYSTRESS_MODEL_PATH"]).parent / "eval"


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
        report = find_report(bundle.model_version, _report_dir())
        body.update({
            "model_version": bundle.model_version,
            "data_source": bundle.data_source,
            "feature_set": bundle.feature_set,
            # Deriving this on each request costs one small file read and means the
            # answer cannot go stale after an evaluation runs.
            "validation": disclosure_for(bundle.metadata, report),
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
        # Ready to serve is not the same as fit to be believed, and the two are reported
        # separately rather than collapsed into one green tick.
        "validation_status": disclosure_for(
            bundle.metadata, find_report(bundle.model_version, _report_dir())
        )["status"],
    }), 200
