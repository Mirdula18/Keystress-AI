"""
Evaluation metrics (F5): per-class performance and calibration.

Two families live here, and the second is the one this project actually needs most.

**Classification metrics** say how often the model is right. They are computed per class,
not only in aggregate, because a weighted average over an imbalanced dataset is dominated
by the majority class — a model that never once predicts "high" can still post a
respectable weighted score while being useless for the only class anyone cares about.
Macro averages are reported alongside for exactly that reason: they weight the rare class
equally, and the gap between the two numbers is itself informative.

**Calibration** asks a different question: when the model says 0.8, is it right 80% of the
time? The prediction endpoint shows a confidence figure to a person looking at a wellbeing
indicator, so the honesty of that number is not an academic concern. ``predict_proba`` from
a random forest is a vote share, not a probability, and is typically overconfident.
Until this is measured on real data the UI calls it "uncalibrated"; these functions are how
that word eventually gets replaced with evidence — or kept.

Everything here is plain NumPy over scikit-learn's outputs. No plotting library is pulled
in: the reliability data is returned as numbers, and the report prints them as a table
(D-025). A text table is greppable, diffable, and readable over SSH, which a PNG is not.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

#: Default number of equal-width bins for a reliability table.
DEFAULT_CALIBRATION_BINS: Final[int] = 10


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: np.ndarray | list[int],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute overall and per-class classification metrics.

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Every class the task defines, including any absent from ``y_true`` —
            a class that never appears must show as zero, not vanish from the table.
        class_names: Display names, defaulting to ``class <i>``.

    Returns:
        dict: ``accuracy``, macro and weighted precision/recall/F1, a ``per_class`` block,
        the confusion matrix, and ``support`` per class. Every value is a plain float or
        list, so the whole thing is JSON-serialisable for the report.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = [int(label) for label in classes]

    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty test set")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred differ in length: {y_true.shape} vs {y_pred.shape}"
        )

    names = class_names or [f"class {label}" for label in labels]

    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = [int((y_true == label).sum()) for label in labels]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": support[index],
                # How often the model chose this class at all. A zero here with non-zero
                # support is the specific failure a weighted average hides: the class
                # exists in the data and the model never predicts it.
                "predicted_count": int((y_pred == labels[index]).sum()),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": names,
        "classes": labels,
        "n_samples": int(y_true.size),
    }


def never_predicted_classes(metrics: dict[str, Any]) -> list[str]:
    """
    Return classes that occur in the data but are never predicted.

    Parameters:
        metrics: A :func:`classification_metrics` result.

    Returns:
        list[str]: Display names of classes with support but no predictions.
    """
    return [
        name for name, stats in metrics["per_class"].items()
        if stats["support"] > 0 and stats["predicted_count"] == 0
    ]


def reliability_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> list[dict[str, Any]]:
    """
    Bin predictions by confidence and report accuracy within each bin.

    This is a reliability curve as numbers. A well-calibrated model has ``accuracy`` close
    to ``mean_confidence`` in every populated bin; a systematically higher confidence than
    accuracy is overconfidence, which is the usual finding for tree ensembles and the one
    that matters when a confidence figure is shown to a person.

    Parameters:
        y_true: True labels.
        probabilities: Predicted probabilities, shape ``(n_samples, n_classes)``.
        n_bins: Number of equal-width confidence bins between 0 and 1.

    Returns:
        list[dict]: One entry per bin with its range, count, mean confidence, and
        accuracy. Empty bins are included with ``count`` 0 so the table has a fixed shape
        across models and reports stay comparable.

    Raises:
        ValueError: If shapes disagree or ``n_bins`` is below 1.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2-D array of shape (n_samples, n_classes)")
    if probabilities.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"probabilities has {probabilities.shape[0]} rows for {y_true.shape[0]} labels"
        )

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict[str, Any]] = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        # Upper edge inclusive on the last bin only, so a confidence of exactly 1.0 lands
        # somewhere rather than being dropped.
        in_bin = (
            (confidence >= low) & (confidence <= high) if index == n_bins - 1
            else (confidence >= low) & (confidence < high)
        )
        count = int(in_bin.sum())
        table.append({
            "bin_lower": float(low),
            "bin_upper": float(high),
            "count": count,
            "mean_confidence": float(confidence[in_bin].mean()) if count else None,
            "accuracy": float(correct[in_bin].mean()) if count else None,
        })
    return table


def expected_calibration_error(table: list[dict[str, Any]]) -> float:
    """
    Compute the expected calibration error from a reliability table.

    ECE is the sample-weighted mean gap between confidence and accuracy across bins: 0 is
    perfect, and larger is worse. It is a summary and hides direction — over- and
    under-confidence look identical — so the report prints the table too.

    Parameters:
        table: A :func:`reliability_table` result.

    Returns:
        float: The error, or ``0.0`` for an empty table.
    """
    total = sum(row["count"] for row in table)
    if total == 0:
        return 0.0
    return float(sum(
        row["count"] / total * abs(row["mean_confidence"] - row["accuracy"])
        for row in table if row["count"]
    ))


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """
    Multi-class Brier score: mean squared error against the one-hot truth.

    Unlike accuracy it rewards being uncertain when uncertain, which is the behaviour this
    project wants from a wellbeing indicator. Lower is better; 0 is perfect.

    Parameters:
        y_true: True labels.
        probabilities: Predicted probabilities, shape ``(n_samples, n_classes)``.

    Returns:
        float: The score.

    Raises:
        ValueError: If shapes disagree or the input is empty.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.size == 0:
        raise ValueError("Cannot compute a Brier score on an empty test set")
    if probabilities.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"probabilities has {probabilities.shape[0]} rows for {y_true.shape[0]} labels"
        )

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(y_true.size), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """
    Compute the full calibration picture: table, ECE, Brier, and a verdict.

    Parameters:
        y_true: True labels.
        probabilities: Predicted probabilities.
        n_bins: Bins for the reliability table.

    Returns:
        dict: ``reliability_table``, ``expected_calibration_error``, ``brier_score``,
        ``mean_confidence``, ``accuracy``, and ``verdict`` — a plain-language reading such
        as "overconfident", so the report does not require the reader to interpret ECE.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    table = reliability_table(y_true, probabilities, n_bins=n_bins)
    confidence = probabilities.max(axis=1)
    accuracy = float((probabilities.argmax(axis=1) == y_true).mean())
    mean_confidence = float(confidence.mean())
    gap = mean_confidence - accuracy

    # A tenth of the scale is a deliberately loose threshold: below it, a small dataset
    # cannot distinguish miscalibration from noise, and calling that "overconfident"
    # would be its own overclaim.
    if abs(gap) < 0.1:
        verdict = "roughly calibrated on this data"
    elif gap > 0:
        verdict = "overconfident - it claims more certainty than it earns on this data"
    else:
        verdict = "underconfident - it is right more often than it claims on this data"

    return {
        "reliability_table": table,
        "expected_calibration_error": expected_calibration_error(table),
        "brier_score": brier_score(y_true, probabilities),
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
        "confidence_minus_accuracy": gap,
        "verdict": verdict,
        "n_bins": n_bins,
    }
