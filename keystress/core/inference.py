"""
Inference for Keystress-AI.

Turns typing features into a burnout *risk indicator* — never a diagnosis, never an
unqualified number.

Every response carries the disclosure fields required by ``docs/CLAUDE.md`` §5:

``data_source``
    Where the model's training data came from. ``"synthetic"`` for the shipped model,
    meaning its scores describe hand-authored generator classes rather than real burnout.
``model_version``
    Which model produced the result, taken from the loaded bundle's metadata.
``disclaimer``
    Plain-language statement that this is not a medical assessment.
``insufficient_data``
    ``True`` when the session carries too little signal to score, in which case no
    prediction is invented (HARD RULE 6 — never a silent fake result).

Indicator levels:
    0 / "Low (indicator)"    - timing sits where the model associates lower indicators
    1 / "Medium (indicator)" - mixed signals
    2 / "High (indicator)"   - timing sits where the model associates higher indicators
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .disclosure import (
    DISCLAIMER,
    FEATURE_SET_VERSION,
    FEATURES_V1,
    SHIPPED_DATA_SOURCE,
    SYNTHETIC_MODEL_NOTICE,
    format_percentage,
)
from .model import ModelBundle

logger = logging.getLogger(__name__)


# Indicator labels. Deliberately phrased as indicators rather than states: the model
# observes typing rhythm, it does not observe a person's condition.
BURNOUT_LABELS: dict[int, str] = {
    0: "Low (indicator)",
    1: "Medium (indicator)",
    2: "High (indicator)",
}

#: Ordered label list matching class index order.
ORDERED_LABELS: list[str] = [BURNOUT_LABELS[0], BURNOUT_LABELS[1], BURNOUT_LABELS[2]]

# Descriptions describe the *signal*, not the user. No second-person diagnosis, no
# instruction to act, no implied certainty.
BURNOUT_DESCRIPTIONS: dict[int, str] = {
    0: "This session's typing rhythm falls where the model associates lower burnout "
       "indicators. That is a statement about timing patterns in one session, not about "
       "how you are doing.",
    1: "This session's typing rhythm falls where the model associates mixed burnout "
       "indicators. Typing varies for many ordinary reasons - tiredness, an unfamiliar "
       "keyboard, distraction - so this on its own means very little.",
    2: "This session's typing rhythm falls where the model associates higher burnout "
       "indicators. This is one noisy signal from one session and cannot tell you whether "
       "you are burned out. If something feels off, talking to someone you trust will tell "
       "you far more than this will.",
}

#: Returned when a session carries too little signal to score.
INSUFFICIENT_DATA_DESCRIPTION = (
    "There is not enough timing signal in this session to produce an indicator. This "
    "usually means too few keystrokes, or keystrokes with no measurable time between "
    "them. Nothing was scored - no result is better than a made-up one."
)

INSUFFICIENT_DATA_LABEL = "Insufficient data"


def feature_vector(typing_features: dict[str, float]) -> np.ndarray:
    """
    Build the model input vector in the order the feature set defines.

    Parameters:
        typing_features: Feature dictionary.

    Returns:
        np.ndarray: Shape ``(1, 5)`` feature matrix.
    """
    return np.array([[float(typing_features.get(name, 0.0)) for name in FEATURES_V1]])


def has_sufficient_signal(typing_features: dict[str, float]) -> bool:
    """
    Decide whether a session carries enough signal to score at all.

    A session with no measurable duration produces an all-zero feature vector. Scoring
    that yields a confident-looking result derived from nothing, which HARD RULE 6
    forbids. This is the minimal Phase 0 gate; calibrated abstention with a real
    input-quality threshold arrives with F7.

    Parameters:
        typing_features: Feature dictionary.

    Returns:
        bool: ``False`` when every feature is zero, otherwise ``True``.
    """
    return any(float(typing_features.get(name, 0.0)) != 0.0 for name in FEATURES_V1)


def disclosure_fields(bundle: ModelBundle | None) -> dict[str, Any]:
    """
    Build the disclosure block attached to every response.

    Parameters:
        bundle: The loaded model bundle, or ``None`` when unavailable.

    Returns:
        dict: ``data_source``, ``model_version``, ``feature_set``, ``disclaimer``.
    """
    if bundle is None:
        return {
            "data_source": SHIPPED_DATA_SOURCE,
            "model_version": "unknown",
            "feature_set": FEATURE_SET_VERSION,
            "disclaimer": DISCLAIMER,
        }
    return {
        "data_source": bundle.data_source,
        "model_version": bundle.model_version,
        "feature_set": bundle.feature_set,
        "disclaimer": DISCLAIMER,
    }


def insufficient_data_response(typing_features: dict[str, float],
                               bundle: ModelBundle | None = None) -> dict[str, Any]:
    """
    Build the response for a session that cannot be scored.

    Parameters:
        typing_features: Feature dictionary.
        bundle: The loaded model bundle, if any.

    Returns:
        dict: A response with a null prediction and an explanation.
    """
    return {
        "prediction": None,
        "label": INSUFFICIENT_DATA_LABEL,
        "description": INSUFFICIENT_DATA_DESCRIPTION,
        "confidence": None,
        "probabilities": None,
        "labels": list(ORDERED_LABELS),
        "insufficient_data": True,
        "features": dict(typing_features),
        **disclosure_fields(bundle),
    }


def predict_burnout(typing_features: dict[str, float],
                    bundle: ModelBundle) -> tuple[int | None, str, str, float | None]:
    """
    Predict a burnout indicator level from typing features.

    Parameters:
        typing_features: The five ``FEATURES_V1`` values.
        bundle: A loaded model bundle.

    Returns:
        tuple: ``(level, label, description, confidence)``. When the session carries
        insufficient signal, level and confidence are ``None`` and the label reports that
        rather than guessing.
    """
    if not has_sufficient_signal(typing_features):
        return None, INSUFFICIENT_DATA_LABEL, INSUFFICIENT_DATA_DESCRIPTION, None

    scaled = bundle.scaler.transform(feature_vector(typing_features))
    prediction = int(bundle.estimator.predict(scaled)[0])
    probabilities = bundle.estimator.predict_proba(scaled)[0]

    return (
        prediction,
        BURNOUT_LABELS.get(prediction, "Unknown"),
        BURNOUT_DESCRIPTIONS.get(prediction, ""),
        float(np.max(probabilities)),
    )


def get_prediction_details(typing_features: dict[str, float],
                           bundle: ModelBundle) -> dict[str, Any]:
    """
    Produce the full prediction response, including required disclosure fields.

    The response always contains ``data_source``, ``model_version``, ``disclaimer``, and
    ``insufficient_data``. It never contains anything derived from typed content — only
    the five aggregate timing features.

    Parameters:
        typing_features: Dictionary of typing features.
        bundle: A loaded model bundle.

    Returns:
        dict: Response matching the ``ARCHITECTURE.md`` §4.3 contract.
    """
    if not has_sufficient_signal(typing_features):
        logger.info("Session had insufficient signal to score; abstaining")
        return insufficient_data_response(typing_features, bundle)

    scaled = bundle.scaler.transform(feature_vector(typing_features))
    prediction = int(bundle.estimator.predict(scaled)[0])
    probabilities = bundle.estimator.predict_proba(scaled)[0]

    return {
        "prediction": prediction,
        "label": BURNOUT_LABELS.get(prediction, "Unknown"),
        "description": BURNOUT_DESCRIPTIONS.get(prediction, ""),
        # Raw max-probability, NOT a calibrated probability. F7 replaces this.
        "confidence": float(np.max(probabilities)),
        # Ordered by class index, per ARCHITECTURE.md §4.3. `labels` gives the position
        # meanings so consumers never hard-code label strings.
        "probabilities": [float(p) for p in probabilities],
        "labels": list(ORDERED_LABELS),
        "insufficient_data": False,
        "features": dict(typing_features),
        **disclosure_fields(bundle),
    }


def format_prediction_output(prediction_result: dict[str, Any]) -> str:
    """
    Format prediction results for display.

    Parameters:
        prediction_result: Dictionary from :func:`get_prediction_details`.

    Returns:
        str: Formatted output in which no metric appears without its data source.
    """
    data_source = prediction_result.get("data_source", SHIPPED_DATA_SOURCE)
    model_version = prediction_result.get("model_version", "unknown")

    output = [
        "=" * 72,
        "BURNOUT RISK INDICATOR (research output - not a diagnosis)",
        "=" * 72,
    ]

    if prediction_result.get("insufficient_data"):
        output.extend([
            f"\nResult: {prediction_result['label']}",
            f"\n{prediction_result['description']}",
            f"\nModel: {model_version} (trained on {data_source} data)",
            f"\n{prediction_result.get('disclaimer', DISCLAIMER)}",
        ])
        return "\n".join(output)

    output.extend([
        f"\nIndicator level: {prediction_result['label']}",
        format_percentage("Model confidence", prediction_result["confidence"], data_source),
        f"\n{prediction_result['description']}",
        f"\nProbability breakdown ({data_source}-trained model):",
    ])

    for level, prob in zip(prediction_result["labels"], prediction_result["probabilities"]):
        bar = "#" * int(prob * 20)
        output.append(f"  {level}: {bar} {prob:.1%} ({data_source}-trained)")

    features = prediction_result["features"]
    output.extend([
        "\nSession timing features:",
        f"  - Typing speed: {features.get('avg_typing_speed', 0):.2f} keys/sec",
        f"  - Avg key delay: {features.get('avg_inter_key_delay', 0):.3f} sec",
        f"  - Max pause: {features.get('max_pause_duration', 0):.2f} sec",
        f"  - Correction rate: {features.get('backspace_ratio', 0):.1%} of keystrokes",
        f"  - Delay variability: {features.get('typing_consistency', 0):.3f} sec "
        f"(std dev - higher means less consistent)",
        f"\nModel: {model_version} (trained on {data_source} data, feature set "
        f"{prediction_result.get('feature_set', FEATURE_SET_VERSION)})",
    ])

    if data_source == "synthetic":
        output.append(f"\n{SYNTHETIC_MODEL_NOTICE}")
    output.append(f"\n{prediction_result.get('disclaimer', DISCLAIMER)}")

    return "\n".join(output)
