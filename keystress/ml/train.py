"""
Model Training for Keystress-AI.

Trains a Random Forest to classify burnout indicator levels from typing features, and
writes a metadata sidecar recording what data it was trained on.

**What the metrics from this module mean.** With the default dataset the labels were
authored by ``keystress.ml.synthetic``, so every score here measures how separable those
hand-built classes are. That is a property of the generator, not of human typing. Every
metric this module prints or stores carries its data source (F1).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..core.disclosure import (
    FEATURE_SET_VERSION,
    FEATURES_V1,
    SHIPPED_DATA_SOURCE,
    SYNTHETIC_MODEL_NOTICE,
    format_metric,
)
from .synthetic import SYNTHETIC_GENERATOR_VERSION

logger = logging.getLogger(__name__)

#: Feature columns, sourced from the versioned feature set so training, inference, and
#: metadata cannot drift apart silently.
FEATURE_COLUMNS: list[str] = list(FEATURES_V1)

DEFAULT_DATA_PATH = Path("data") / "synthetic_typing_data.csv"
DEFAULT_MODEL_PATH = Path("models") / "burnout_model.pkl"
DEFAULT_SCALER_PATH = Path("models") / "scaler.pkl"
DEFAULT_METADATA_PATH = Path("models") / "model_metadata.json"

#: Display names for the three indicator classes in evaluation reports.
CLASS_DISPLAY_NAMES = ["Low", "Medium", "High"]


def load_training_data(filepath: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the training dataset.

    Parameters:
        filepath: Path to the CSV dataset.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset is missing.
        ValueError: If required columns are absent, which would otherwise surface as an
            obscure KeyError deep inside the split.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Generate it with `python -m keystress.ml.synthetic`."
        )

    df = pd.read_csv(path)

    required = set(FEATURE_COLUMNS) | {"burnout_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Training data at {path} is missing required columns: {sorted(missing)}. "
            f"Expected feature set {FEATURE_SET_VERSION}."
        )

    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2,
                 random_state: int = 42) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, StandardScaler]:
    """
    Split and scale the dataset for training.

    Parameters:
        df: Full dataset.
        test_size: Proportion held out for testing.
        random_state: Seed for the split.

    Returns:
        tuple: ``(X_train, X_test, y_train, y_test, scaler)``.

    Note:
        The scaler is fitted on the full dataset before splitting, which leaks test-set
        distribution information into the transform. With synthetic data and a
        distribution-free tree model the practical effect is negligible, but it is a real
        methodological flaw and is fixed properly in F5, where participant-grouped splits
        replace this entirely. Recorded rather than silently carried.
    """
    X = df[FEATURE_COLUMNS].values
    y = df["burnout_level"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


#: Fit single-threaded. See :func:`train_random_forest` for why this is not a performance
#: oversight.
TRAINING_N_JOBS = 1


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 100,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters:
        X_train: Scaled training features.
        y_train: Training labels.
        n_estimators: Number of trees.
        random_state: Seed, fixed so builds are reproducible (F13).

    Returns:
        RandomForestClassifier: The fitted model.

    Note:
        ``n_jobs`` is deliberately 1, not -1. The inherited code used ``n_jobs=-1``, which
        makes fitting **non-deterministic at float precision even with `random_state`
        fixed**: parallel tree construction varies the floating-point accumulation order,
        so two runs with identical seeds produce models whose ``predict_proba`` differs in
        the last bits. Measured directly — six refits with ``n_jobs=-1`` produced six
        different probability arrays; with ``n_jobs=1`` all six were byte-identical.

        Two reasons this matters more than the lost parallelism:

        1. F13 requires that a clean checkout reproduce the same model. With ``n_jobs=-1``
           that claim is false, and this was found by the reproducibility check failing.
        2. The confidence figure shown to a user would vary between otherwise identical
           runs. Tiny, but this project reports uncertainty as a feature; that number
           should not wobble for reasons unrelated to the input.

        The cost is negligible at this scale (1500 samples, 100 shallow trees — well under
        a second). Revisit only if the dataset grows by orders of magnitude, and then
        record the trade-off rather than silently trading determinism for speed.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=TRAINING_N_JOBS,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """
    Evaluate a trained model.

    The returned metrics are meaningless without knowing their data source; callers must
    pair them with one. :func:`build_model_metadata` does this by construction.

    Parameters:
        model: Trained classifier.
        X_test: Scaled test features.
        y_test: True labels.

    Returns:
        dict: Accuracy, weighted precision/recall/F1, confusion matrix, and report.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_DISPLAY_NAMES, zero_division=0
        ),
    }


def get_feature_importance(model: Any,
                           feature_names: list[str] | None = None) -> dict[str, float]:
    """
    Extract feature importances from a fitted tree model.

    Parameters:
        model: Trained model.
        feature_names: Feature names; defaults to :data:`FEATURE_COLUMNS`.

    Returns:
        dict: Feature name to importance, empty if the model exposes none.
    """
    names = feature_names if feature_names is not None else FEATURE_COLUMNS

    if not hasattr(model, "feature_importances_"):
        return {}
    return dict(zip(names, model.feature_importances_.tolist()))


def build_model_version(data_source: str, n_samples: int, random_state: int) -> str:
    """
    Build a deterministic model version identifier.

    Derived only from inputs that determine the model, so an identical run produces an
    identical string. Deliberately contains no timestamp — a clean checkout must reproduce
    the same version (F13). The generator version is included so a change to the
    data-generating process is visible in the model identity.

    Parameters:
        data_source: ``"synthetic"`` or ``"real"``.
        n_samples: Number of training samples.
        random_state: Seed used throughout.

    Returns:
        str: e.g. ``"rf-v1-synthetic-g2-s42-n1500"``.
    """
    parts = ["rf", FEATURE_SET_VERSION, data_source]
    if data_source == "synthetic":
        parts.append(SYNTHETIC_GENERATOR_VERSION)
    parts.extend([f"s{random_state}", f"n{n_samples}"])
    return "-".join(parts)


def build_model_metadata(metrics: dict[str, Any], n_samples: int, random_state: int,
                         data_source: str = SHIPPED_DATA_SOURCE) -> dict[str, Any]:
    """
    Describe a trained model and what its metrics actually mean.

    Parameters:
        metrics: Metrics from :func:`evaluate_model`.
        n_samples: Number of samples in the training dataset.
        random_state: Seed used throughout the pipeline.
        data_source: Source of the training data.

    Returns:
        dict: Registry-shaped metadata (``ARCHITECTURE.md`` §4.4).
    """
    metadata: dict[str, Any] = {
        "model_version": build_model_version(data_source, n_samples, random_state),
        "model_type": "RandomForestClassifier",
        "trained_on": data_source,
        "data_source": data_source,
        "feature_set": FEATURE_SET_VERSION,
        "features": list(FEATURE_COLUMNS),
        "random_seed": random_state,
        "n_samples": n_samples,
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision_weighted": float(metrics["precision"]),
            "recall_weighted": float(metrics["recall"]),
            "f1_weighted": float(metrics["f1_score"]),
        },
        # The metrics above are meaningless without this field. Never drop it.
        "metrics_data_source": data_source,
        "metrics_caveat": SYNTHETIC_MODEL_NOTICE if data_source == "synthetic" else "",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if data_source == "synthetic":
        metadata["generator_version"] = SYNTHETIC_GENERATOR_VERSION
    return metadata


def save_model(model: Any, scaler: StandardScaler,
               model_path: Path = DEFAULT_MODEL_PATH,
               scaler_path: Path = DEFAULT_SCALER_PATH,
               metadata: dict[str, Any] | None = None,
               metadata_path: Path = DEFAULT_METADATA_PATH) -> None:
    """
    Save the model, scaler, and disclosure metadata.

    Parameters:
        model: Trained model.
        scaler: Fitted scaler.
        model_path: Destination for the model.
        scaler_path: Destination for the scaler.
        metadata: Metadata from :func:`build_model_metadata`.
        metadata_path: Destination for the metadata sidecar.
    """
    import joblib

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved model to %s and scaler to %s", model_path, scaler_path)

    if metadata is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Saved model metadata to %s", metadata_path)


def train_and_evaluate(data_path: Path = DEFAULT_DATA_PATH,
                       save_models: bool = True,
                       random_state: int = 42,
                       data_source: str = SHIPPED_DATA_SOURCE,
                       model_path: Path = DEFAULT_MODEL_PATH,
                       scaler_path: Path = DEFAULT_SCALER_PATH,
                       metadata_path: Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """
    Run the full training pipeline: load, split, fit, evaluate, save.

    Parameters:
        data_path: Path to the training dataset.
        save_models: Whether to persist artifacts.
        random_state: Seed applied to splitting and fitting.
        data_source: Source of the training data, recorded in metadata.
        model_path: Destination for the model.
        scaler_path: Destination for the scaler.
        metadata_path: Destination for the metadata sidecar.

    Returns:
        dict: ``model``, ``scaler``, ``metrics``, ``feature_importance``, ``metadata``.
    """
    df = load_training_data(data_path)
    n_samples = len(df)
    logger.info("Loaded %d samples from %s (data source: %s)", n_samples, data_path, data_source)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df, random_state=random_state)
    logger.info("Split into %d training and %d test samples", len(X_train), len(X_test))

    model = train_random_forest(X_train, y_train, random_state=random_state)
    metrics = evaluate_model(model, X_test, y_test)
    importance = get_feature_importance(model)
    metadata = build_model_metadata(metrics, n_samples, random_state, data_source)

    logger.info(
        "Trained %s: accuracy %.4f measured on %s data",
        metadata["model_version"], metrics["accuracy"], data_source,
    )

    if save_models:
        save_model(model, scaler, model_path, scaler_path, metadata, metadata_path)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": importance,
        "metadata": metadata,
        "data_source": data_source,
    }


def format_evaluation_report(metrics: dict[str, Any], importance: dict[str, float],
                             metadata: dict[str, Any]) -> str:
    """
    Render a human-readable evaluation report.

    No number in the output appears without its data source.

    Parameters:
        metrics: Metrics from :func:`evaluate_model`.
        importance: Feature importances.
        metadata: Model metadata.

    Returns:
        str: The formatted report.
    """
    data_source = metadata.get("data_source", SHIPPED_DATA_SOURCE)

    lines = [
        "=" * 72,
        f"MODEL EVALUATION - measured on {data_source.upper()} DATA",
        "=" * 72,
    ]
    if data_source == "synthetic":
        lines.extend([SYNTHETIC_MODEL_NOTICE, "-" * 72])

    lines.extend([
        format_metric("Accuracy ", metrics["accuracy"], data_source),
        format_metric("Precision", metrics["precision"], data_source),
        format_metric("Recall   ", metrics["recall"], data_source),
        format_metric("F1-Score ", metrics["f1_score"], data_source),
        "",
        f"Confusion matrix ({data_source} data):",
        "            Predicted",
        "            Low  Med  High",
    ])

    cm = metrics["confusion_matrix"]
    for name, row in zip(CLASS_DISPLAY_NAMES, cm):
        lines.append(f"Actual {name:<5} {row[0]:4d} {row[1]:4d} {row[2]:5d}")

    lines.extend([
        "",
        f"Classification report ({data_source} data):",
        metrics["classification_report"],
        f"Feature importance (derived from {data_source} data):",
    ])
    for feature, score in sorted(importance.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  {feature}: {score:.4f} ({data_source}-derived)")

    lines.extend([
        "",
        f"Model version: {metadata['model_version']}",
        f"Feature set:   {metadata['feature_set']}",
        f"Data source:   {data_source}",
    ])
    return "\n".join(lines)


def main() -> int:
    """
    CLI entrypoint: generate data if needed, train, and report.

    Returns:
        int: Process exit code, 1 if training could not proceed.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not DEFAULT_DATA_PATH.exists():
        from .synthetic import generate_synthetic_typing_data, save_synthetic_data

        logger.info("No dataset found; generating the default synthetic dataset")
        save_synthetic_data(generate_synthetic_typing_data(n_samples=1500))

    try:
        results = train_and_evaluate()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        return 1

    print(format_evaluation_report(
        results["metrics"], results["feature_importance"], results["metadata"]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
