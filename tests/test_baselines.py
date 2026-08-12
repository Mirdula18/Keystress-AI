"""
Trivial baseline tests (F5).

These are small functions, and the tests are correspondingly small — except for the
reproducibility and tie-breaking ones, which cover the two ways a baseline can quietly
stop being a fair comparison: changing between runs, or depending on row order.
"""

from __future__ import annotations

import numpy as np
import pytest

from keystress.ml.baselines import (
    BASELINE_DESCRIPTIONS,
    BASELINE_NAMES,
    baseline_predictions,
    beats_every_baseline,
    majority_baseline,
    stratified_baseline,
    uniform_baseline,
)


class TestMajority:
    def test_predicts_the_most_common_class(self) -> None:
        y_train = np.array([0, 0, 0, 1, 2])
        assert set(majority_baseline(y_train, 4)) == {0}

    def test_produces_one_prediction_per_test_row(self) -> None:
        assert len(majority_baseline(np.array([1, 1, 2]), 7)) == 7

    def test_ties_break_deterministically_regardless_of_order(self) -> None:
        """
        `Counter.most_common` breaks ties by insertion order, so a shuffled training set
        would silently change the baseline — and with it the bar the model is measured
        against.
        """
        first = majority_baseline(np.array([2, 2, 1, 1]), 3)
        second = majority_baseline(np.array([1, 1, 2, 2]), 3)
        assert list(first) == list(second)

    def test_an_empty_training_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            majority_baseline(np.array([]), 3)


class TestStratified:
    def test_only_predicts_known_classes(self) -> None:
        predictions = stratified_baseline(np.array([0, 0, 1, 1, 2]), 100)
        assert set(predictions) <= {0, 1, 2}

    def test_roughly_follows_the_training_distribution(self) -> None:
        y_train = np.array([0] * 90 + [1] * 10)
        predictions = stratified_baseline(y_train, 2000, seed=1)
        share_of_zero = float((predictions == 0).mean())
        assert 0.85 < share_of_zero < 0.95

    def test_is_reproducible_for_a_seed(self) -> None:
        y_train = np.array([0, 1, 1, 2])
        assert list(stratified_baseline(y_train, 20, seed=5)) == list(
            stratified_baseline(y_train, 20, seed=5)
        )

    def test_an_empty_training_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            stratified_baseline(np.array([]), 3)


class TestUniform:
    def test_covers_the_classes_roughly_evenly(self) -> None:
        predictions = uniform_baseline(np.array([0, 1, 2]), 3000, seed=2)
        shares = [float((predictions == label).mean()) for label in (0, 1, 2)]
        assert all(0.28 < share < 0.39 for share in shares)

    def test_is_reproducible_for_a_seed(self) -> None:
        assert list(uniform_baseline(np.array([0, 1, 2]), 15, seed=3)) == list(
            uniform_baseline(np.array([0, 1, 2]), 15, seed=3)
        )

    def test_no_classes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no classes"):
            uniform_baseline(np.array([]), 3)


class TestBaselinePredictions:
    def test_returns_every_named_baseline(self) -> None:
        result = baseline_predictions(np.array([0, 1, 1, 2]), 10)
        assert set(result) == set(BASELINE_NAMES)

    def test_each_baseline_predicts_for_every_test_row(self) -> None:
        for predictions in baseline_predictions(np.array([0, 1]), 8).values():
            assert len(predictions) == 8

    def test_classes_can_be_declared_beyond_those_seen_in_training(self) -> None:
        """
        A class the task defines but training never saw must still be reachable by the
        uniform baseline, or the "floor" is computed over the wrong number of options.
        """
        predictions = baseline_predictions(
            np.array([0, 0, 1]), 400, classes=np.array([0, 1, 2]), seed=4
        )
        assert 2 in set(predictions["uniform"])

    def test_every_baseline_is_described_for_the_report(self) -> None:
        # A reader should not have to look up what "stratified" means to read the table.
        assert set(BASELINE_DESCRIPTIONS) == set(BASELINE_NAMES)


class TestComparison:
    def test_true_only_when_strictly_better_than_all(self) -> None:
        assert beats_every_baseline(0.7, {"majority": 0.6, "uniform": 0.33})
        assert not beats_every_baseline(0.6, {"majority": 0.6})
        assert not beats_every_baseline(0.5, {"majority": 0.6, "uniform": 0.33})

    def test_no_baselines_means_no_claim(self) -> None:
        # "Unbeaten" with nothing to beat is not a result.
        assert not beats_every_baseline(0.99, {})
