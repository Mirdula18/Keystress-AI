"""
Trivial baselines (F5).

A model's score means nothing on its own. "Sixty-two percent" sounds like a finding until
you notice that always guessing the most common class scores sixty percent, at which point
the model has demonstrated two points of skill and a great deal of machinery.

This is not a hypothetical risk for this project. Burnout classes are unlikely to be
balanced in a volunteer sample, so the majority-class baseline may be high — and it is
exactly the number an enthusiastic reading would forget to mention. So the evaluation
harness computes these unconditionally and the report prints them **beside** every model
metric, never in an appendix.

Three baselines, because they fail differently:

``majority``
    Always predicts the most frequent training class. The hardest to beat when classes
    are imbalanced, and the one that most often embarrasses a real model.
``stratified``
    Samples predictions from the training class distribution. Beats nothing, but shows
    what "getting the distribution right and the individuals wrong" scores.
``uniform``
    Samples uniformly across classes. The floor: what pure chance looks like when the
    classes are ignored entirely.

All three are deterministic given a seed, so a report can be regenerated exactly.
"""

from __future__ import annotations

from collections import Counter
from typing import Final

import numpy as np

#: Baseline names, in the order they should be reported.
BASELINE_NAMES: Final[tuple[str, ...]] = ("majority", "stratified", "uniform")

#: What each baseline does, carried into the report so a reader need not look it up.
BASELINE_DESCRIPTIONS: Final[dict[str, str]] = {
    "majority": "always predicts the most common class in the training data",
    "stratified": "predicts at random, matching the training class distribution",
    "uniform": "predicts uniformly at random across the classes",
}


def majority_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """
    Predict the most frequent training class for everything.

    Parameters:
        y_train: Training labels.
        n_test: Number of predictions to produce.

    Returns:
        np.ndarray: ``n_test`` copies of the majority class.

    Raises:
        ValueError: If ``y_train`` is empty; there is no majority of nothing.
    """
    if len(y_train) == 0:
        raise ValueError("Cannot compute a majority baseline from an empty training set")

    # Counter.most_common breaks ties by insertion order, which depends on row order.
    # Sorting the tied classes keeps the baseline reproducible across shuffles.
    counts = Counter(int(label) for label in y_train)
    top = max(counts.values())
    majority = min(label for label, count in counts.items() if count == top)
    return np.full(n_test, majority, dtype=int)


def stratified_baseline(y_train: np.ndarray, n_test: int, seed: int = 42) -> np.ndarray:
    """
    Predict at random, following the training class distribution.

    Parameters:
        y_train: Training labels.
        n_test: Number of predictions to produce.
        seed: Seed, so the report is reproducible.

    Returns:
        np.ndarray: Sampled predictions.

    Raises:
        ValueError: If ``y_train`` is empty.
    """
    if len(y_train) == 0:
        raise ValueError("Cannot compute a stratified baseline from an empty training set")

    labels, counts = np.unique(np.asarray(y_train, dtype=int), return_counts=True)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(labels, size=n_test, p=probabilities).astype(int)


def uniform_baseline(classes: np.ndarray, n_test: int, seed: int = 42) -> np.ndarray:
    """
    Predict uniformly at random across the classes.

    Parameters:
        classes: The classes to choose between.
        n_test: Number of predictions to produce.
        seed: Seed, so the report is reproducible.

    Returns:
        np.ndarray: Sampled predictions.

    Raises:
        ValueError: If ``classes`` is empty.
    """
    classes = np.asarray(classes, dtype=int)
    if classes.size == 0:
        raise ValueError("Cannot compute a uniform baseline with no classes")

    rng = np.random.default_rng(seed)
    return rng.choice(classes, size=n_test).astype(int)


def baseline_predictions(
    y_train: np.ndarray,
    n_test: int,
    *,
    classes: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Produce every baseline's predictions for a test set.

    Parameters:
        y_train: Training labels.
        n_test: Size of the test set.
        classes: Classes the task uses. Defaults to those present in ``y_train``; pass
            explicitly when a class is absent from training but defined by the task,
            otherwise the uniform baseline would quietly never predict it.
        seed: Seed for the two random baselines.

    Returns:
        dict[str, np.ndarray]: Predictions keyed by baseline name.
    """
    known_classes = (
        np.unique(np.asarray(y_train, dtype=int)) if classes is None
        else np.asarray(classes, dtype=int)
    )
    return {
        "majority": majority_baseline(y_train, n_test),
        "stratified": stratified_baseline(y_train, n_test, seed=seed),
        "uniform": uniform_baseline(known_classes, n_test, seed=seed),
    }


def beats_every_baseline(model_score: float, baseline_scores: dict[str, float]) -> bool:
    """
    Report whether a model score exceeds every baseline score.

    Deliberately strict — equal is not better — and deliberately blunt: this returns the
    one fact a reader wants and the report prints it in words, so "the model did not beat
    always guessing the most common class" cannot be lost between two rows of a table.

    Parameters:
        model_score: The model's score on some metric.
        baseline_scores: The same metric for each baseline.

    Returns:
        bool: ``True`` only if the model strictly exceeds all of them. With no baselines
        to compare against, ``False`` — an unbeaten claim needs something to have beaten.
    """
    if not baseline_scores:
        return False
    return all(model_score > score for score in baseline_scores.values())
