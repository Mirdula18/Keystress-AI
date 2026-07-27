"""
Prediction Module for Keystress-AI

Turns typing features into a burnout *risk indicator* — never a diagnosis, never an
unqualified number.

Every response this module produces carries four disclosure fields required by
``docs/CLAUDE.md`` §5:

``data_source``
    Where the model's training data came from. ``"synthetic"`` for the shipped model,
    meaning its scores describe hand-authored generator classes rather than real burnout.
``model_version``
    Which model produced the result, read from the metadata sidecar written at training.
``disclaimer``
    Plain-language statement that this is not a medical assessment.
``insufficient_data``
    ``True`` when the session carries too little signal to score, in which case no
    prediction is invented (``docs/CLAUDE.md`` §2 rule 6 — never a silent fake result).

Indicator levels:
    0 / "Low (indicator)"    - timing sits where the model associates lower indicators
    1 / "Medium (indicator)" - mixed signals
    2 / "High (indicator)"   - timing sits where the model associates higher indicators
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .disclosure import (
    DISCLAIMER,
    FEATURE_SET_VERSION,
    FEATURES_V1,
    SHIPPED_DATA_SOURCE,
    SYNTHETIC_MODEL_NOTICE,
    format_percentage,
)

DEFAULT_MODEL_PATH = 'models/burnout_model.pkl'
DEFAULT_SCALER_PATH = 'models/scaler.pkl'
DEFAULT_METADATA_PATH = 'models/model_metadata.json'


# Indicator labels. Deliberately phrased as indicators rather than states: the model
# observes typing rhythm, it does not observe a person's condition.
BURNOUT_LABELS: Dict[int, str] = {
    0: "Low (indicator)",
    1: "Medium (indicator)",
    2: "High (indicator)",
}

# Descriptions describe the *signal*, not the user. No second-person diagnosis, no
# instruction to act, no implied certainty.
BURNOUT_DESCRIPTIONS: Dict[int, str] = {
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


def load_model_metadata(metadata_path: str = DEFAULT_METADATA_PATH) -> Dict[str, Any]:
    """
    Load the disclosure metadata written alongside a trained model.

    A model without metadata is reported honestly as unknown rather than being given a
    flattering default: an unlabelled model is exactly the situation the disclosure
    contract exists to expose.

    Parameters:
        metadata_path: Path to the metadata sidecar.

    Returns:
        dict: Metadata with at least ``model_version`` and ``data_source`` populated.
    """
    if not os.path.exists(metadata_path):
        return {
            'model_version': 'unknown',
            'data_source': SHIPPED_DATA_SOURCE,
            'feature_set': FEATURE_SET_VERSION,
            'metrics_caveat': SYNTHETIC_MODEL_NOTICE,
        }

    try:
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {
            'model_version': 'unknown',
            'data_source': SHIPPED_DATA_SOURCE,
            'feature_set': FEATURE_SET_VERSION,
            'metrics_caveat': SYNTHETIC_MODEL_NOTICE,
        }

    metadata.setdefault('model_version', 'unknown')
    metadata.setdefault('data_source', SHIPPED_DATA_SOURCE)
    metadata.setdefault('feature_set', FEATURE_SET_VERSION)
    return metadata


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH,
                       scaler_path: str = DEFAULT_SCALER_PATH):
    """
    Load the trained model and scaler.

    Parameters:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler

    Returns:
        tuple: (model, scaler)

    Raises:
        FileNotFoundError: If either artifact is missing.
    """
    import joblib

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Trained model not found. Please run train_model.py first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def _feature_vector(typing_features: Dict[str, float]) -> np.ndarray:
    """
    Build the model input vector in the order the feature set defines.

    Parameters:
        typing_features: Feature dictionary.

    Returns:
        np.ndarray: Shape ``(1, 5)`` feature matrix.
    """
    return np.array([[float(typing_features.get(name, 0.0)) for name in FEATURES_V1]])


def has_sufficient_signal(typing_features: Dict[str, float]) -> bool:
    """
    Decide whether a session carries enough signal to score at all.

    A session with no measurable duration produces an all-zero feature vector. Scoring
    that vector yields a confident-looking result derived from nothing, which
    ``docs/CLAUDE.md`` §2 rule 6 forbids. This is the minimal Phase 0 gate; calibrated
    abstention with a real quality threshold arrives with F7.

    Parameters:
        typing_features: Feature dictionary.

    Returns:
        bool: ``False`` when every feature is zero, otherwise ``True``.
    """
    return any(float(typing_features.get(name, 0.0)) != 0.0 for name in FEATURES_V1)


def _disclosure_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the disclosure block attached to every response.

    Parameters:
        metadata: Model metadata from :func:`load_model_metadata`.

    Returns:
        dict: ``data_source``, ``model_version``, ``feature_set``, ``disclaimer``.
    """
    return {
        'data_source': metadata.get('data_source', SHIPPED_DATA_SOURCE),
        'model_version': metadata.get('model_version', 'unknown'),
        'feature_set': metadata.get('feature_set', FEATURE_SET_VERSION),
        'disclaimer': DISCLAIMER,
    }


def predict_burnout(typing_features: Dict[str, float],
                    model=None, scaler=None) -> Tuple[Optional[int], str, str, Optional[float]]:
    """
    Predict a burnout indicator level from typing features.

    Parameters:
        typing_features: Dictionary containing the five ``FEATURES_V1`` values.
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)

    Returns:
        tuple: ``(level_number, level_label, description, confidence)``. When the session
        carries insufficient signal, level and confidence are ``None`` and the label
        reports that rather than guessing.
    """
    if not has_sufficient_signal(typing_features):
        return None, "Insufficient data", INSUFFICIENT_DATA_DESCRIPTION, None

    if model is None or scaler is None:
        model, scaler = load_trained_model()

    features_scaled = scaler.transform(_feature_vector(typing_features))

    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))

    label = BURNOUT_LABELS.get(prediction, "Unknown")
    description = BURNOUT_DESCRIPTIONS.get(prediction, "")

    return prediction, label, description, confidence


def get_prediction_details(typing_features: Dict[str, float],
                           model=None, scaler=None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Produce the full prediction response, including required disclosure fields.

    The returned dictionary always contains ``data_source``, ``model_version``,
    ``disclaimer``, and ``insufficient_data``. It never contains anything derived from
    typed content — only the five aggregate timing features.

    Parameters:
        typing_features: Dictionary of typing features
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)
        metadata: Pre-loaded model metadata (optional)

    Returns:
        dict: Prediction response matching the ``ARCHITECTURE.md`` §4.3 contract.
    """
    if metadata is None:
        metadata = load_model_metadata()

    disclosure = _disclosure_fields(metadata)

    if not has_sufficient_signal(typing_features):
        return {
            'prediction': None,
            'label': "Insufficient data",
            'description': INSUFFICIENT_DATA_DESCRIPTION,
            'confidence': None,
            'probabilities': None,
            'labels': [BURNOUT_LABELS[0], BURNOUT_LABELS[1], BURNOUT_LABELS[2]],
            'insufficient_data': True,
            'features': dict(typing_features),
            **disclosure,
        }

    if model is None or scaler is None:
        model, scaler = load_trained_model()

    features_scaled = scaler.transform(_feature_vector(typing_features))

    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]

    return {
        'prediction': prediction,
        'label': BURNOUT_LABELS.get(prediction, "Unknown"),
        'description': BURNOUT_DESCRIPTIONS.get(prediction, ""),
        # Raw max-probability, NOT a calibrated probability. F7 replaces this.
        'confidence': float(np.max(probabilities)),
        # Ordered by class index, per the ARCHITECTURE.md §4.3 contract. `labels` gives
        # the position meanings so consumers never hard-code label strings.
        'probabilities': [float(p) for p in probabilities],
        'labels': [BURNOUT_LABELS[0], BURNOUT_LABELS[1], BURNOUT_LABELS[2]],
        'insufficient_data': False,
        'features': dict(typing_features),
        **disclosure,
    }


def format_prediction_output(prediction_result: Dict[str, Any]) -> str:
    """
    Format prediction results for display.

    Parameters:
        prediction_result: Dictionary from :func:`get_prediction_details`

    Returns:
        str: Formatted output string in which no metric appears without its data source.
    """
    data_source = prediction_result.get('data_source', SHIPPED_DATA_SOURCE)

    output = []
    output.append("=" * 72)
    output.append("BURNOUT RISK INDICATOR (research output - not a diagnosis)")
    output.append("=" * 72)

    if prediction_result.get('insufficient_data'):
        output.append(f"\nResult: {prediction_result['label']}")
        output.append(f"\n{prediction_result['description']}")
        output.append(f"\nModel: {prediction_result.get('model_version', 'unknown')} "
                      f"(trained {data_source})")
        output.append(f"\n{prediction_result.get('disclaimer', DISCLAIMER)}")
        return "\n".join(output)

    output.append(f"\nIndicator level: {prediction_result['label']}")
    output.append(format_percentage("Model confidence",
                                    prediction_result['confidence'], data_source))
    output.append(f"\n{prediction_result['description']}")

    output.append(f"\nProbability breakdown ({data_source}-trained model):")
    for level, prob in zip(prediction_result['labels'], prediction_result['probabilities']):
        bar = "#" * int(prob * 20)
        output.append(f"  {level}: {bar} {prob:.1%} ({data_source}-trained)")

    output.append("\nSession timing features:")
    features = prediction_result['features']
    output.append(f"  - Typing speed: {features.get('avg_typing_speed', 0):.2f} keys/sec")
    output.append(f"  - Avg key delay: {features.get('avg_inter_key_delay', 0):.3f} sec")
    output.append(f"  - Max pause: {features.get('max_pause_duration', 0):.2f} sec")
    output.append(f"  - Correction rate: {features.get('backspace_ratio', 0):.1%} of keystrokes")
    output.append(f"  - Delay variability: {features.get('typing_consistency', 0):.3f} sec "
                  f"(std dev - higher means less consistent)")

    output.append(f"\nModel: {prediction_result.get('model_version', 'unknown')} "
                  f"(trained on {data_source} data, feature set "
                  f"{prediction_result.get('feature_set', FEATURE_SET_VERSION)})")
    if data_source == 'synthetic':
        output.append(f"\n{SYNTHETIC_MODEL_NOTICE}")
    output.append(f"\n{prediction_result.get('disclaimer', DISCLAIMER)}")

    return "\n".join(output)


if __name__ == "__main__":
    print("Prediction Module Demo")
    print("=" * 72)

    sample_features = {
        'avg_typing_speed': 3.5,
        'avg_inter_key_delay': 0.35,
        'max_pause_duration': 2.5,
        'backspace_ratio': 0.15,
        'typing_consistency': 0.12,
    }

    try:
        result = get_prediction_details(sample_features)
        print(format_prediction_output(result))
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Please train the model first by running: python -m src.train_model")
