"""
Evaluation metric tests (F5).

Worked examples throughout: for a metric, "it returns a plausible number" is not evidence
of anything, so each test constructs a case whose answer is known by hand. The calibration
tests in particular build deliberately over- and under-confident predictions, because a
calibration function that silently reports the wrong direction is worse than none.
"""

from __future__ import annotations

import numpy as np
import pytest

from keystress.ml.metrics import (
    brier_score,
    calibration_metrics,
    classification_metrics,
    expected_calibration_error,
    never_predicted_classes,
    reliability_table,
)

CLASSES = [0, 1, 2]
NAMES = ["Low", "Medium", "High"]


class TestClassificationMetrics:
    """Per-class, because an average hides the class that matters."""

    def test_perfect_predictions_score_perfectly(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        result = classification_metrics(y, y, classes=CLASSES, class_names=NAMES)

        assert result["accuracy"] == 1.0
        assert result["f1_macro"] == 1.0
        assert all(stats["f1"] == 1.0 for stats in result["per_class"].values())

    def test_a_worked_example_matches_hand_computation(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 1, 2])
        result = classification_metrics(y_true, y_pred, classes=CLASSES, class_names=NAMES)

        assert result["accuracy"] == pytest.approx(4 / 6)
        # metrics-ok: hand-built arrays in a unit test, not a reported result
        # Class 0: predicted 3 times, 2 correct, so precision is 2/3; support 2, both found.
        assert result["per_class"]["Low"]["precision"] == pytest.approx(2 / 3)
        assert result["per_class"]["Low"]["recall"] == 1.0
        # Class 2: predicted once, correct; one of its two rows went to class 1.
        # metrics-ok: hand-built arrays in a unit test, not a reported result
        assert result["per_class"]["High"]["recall"] == 0.5

    def test_a_class_absent_from_the_data_still_appears(self) -> None:
        """
        A class that never occurs must show as zero rather than vanish, or the table
        silently changes shape between datasets and reports stop being comparable.
        """
        y = np.array([0, 0, 1, 1])
        result = classification_metrics(y, y, classes=CLASSES, class_names=NAMES)

        assert "High" in result["per_class"]
        assert result["per_class"]["High"]["support"] == 0

    def test_a_never_predicted_class_is_detectable(self) -> None:
        """
        The specific failure a weighted average hides: the class exists in the data and
        the model never once predicts it, while the headline number still looks fine.
        """
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
        y_pred = np.zeros_like(y_true)
        result = classification_metrics(y_true, y_pred, classes=CLASSES, class_names=NAMES)

        # metrics-ok: hand-built arrays in a unit test, not a reported result
        assert result["accuracy"] == pytest.approx(0.8)
        assert sorted(never_predicted_classes(result)) == ["High", "Medium"]

    def test_macro_and_weighted_disagree_under_imbalance(self) -> None:
        """The gap between them is itself the finding, so both are reported."""
        y_true = np.array([0] * 18 + [1, 2])
        y_pred = np.zeros_like(y_true)
        result = classification_metrics(y_true, y_pred, classes=CLASSES, class_names=NAMES)

        assert result["f1_weighted"] > result["f1_macro"]

    def test_confusion_matrix_is_square_over_all_classes(self) -> None:
        y = np.array([0, 1])
        result = classification_metrics(y, y, classes=CLASSES, class_names=NAMES)
        matrix = result["confusion_matrix"]
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

    def test_an_empty_test_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            classification_metrics(np.array([]), np.array([]), classes=CLASSES)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            classification_metrics(np.array([0, 1]), np.array([0]), classes=CLASSES)


class TestReliabilityTable:
    """Confidence versus accuracy, in bins."""

    def test_bins_have_a_fixed_shape_including_empty_ones(self) -> None:
        # A fixed shape keeps two reports comparable row by row.
        table = reliability_table(np.array([0, 1]), np.array([[0.9, 0.1], [0.2, 0.8]]),
                                  n_bins=5)
        assert len(table) == 5
        assert sum(row["count"] for row in table) == 2

    def test_confidence_of_exactly_one_lands_in_the_last_bin(self) -> None:
        """Half-open bins would silently drop a fully confident prediction."""
        table = reliability_table(np.array([0]), np.array([[1.0, 0.0]]), n_bins=10)
        assert table[-1]["count"] == 1

    def test_a_confident_and_correct_model_bins_high(self) -> None:
        probabilities = np.array([[0.95, 0.05]] * 4)
        table = reliability_table(np.array([0, 0, 0, 0]), probabilities, n_bins=10)
        populated = [row for row in table if row["count"]]

        assert len(populated) == 1
        # metrics-ok: hand-built arrays in a unit test, not a reported result
        assert populated[0]["accuracy"] == 1.0
        assert populated[0]["mean_confidence"] == pytest.approx(0.95)

    def test_shape_mismatches_are_refused(self) -> None:
        with pytest.raises(ValueError, match="rows for"):
            reliability_table(np.array([0, 1]), np.array([[0.5, 0.5]]))

    def test_one_dimensional_probabilities_are_refused(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            reliability_table(np.array([0]), np.array([0.9]))


class TestCalibrationSummary:
    """The numbers a report prints, and the direction they point."""

    def test_a_perfectly_calibrated_model_has_near_zero_error(self) -> None:
        # 80% confident, right exactly 80% of the time.
        probabilities = np.array([[0.8, 0.2]] * 10)
        y_true = np.array([0] * 8 + [1] * 2)

        result = calibration_metrics(y_true, probabilities, n_bins=10)
        assert result["expected_calibration_error"] == pytest.approx(0.0, abs=0.01)
        assert "roughly calibrated" in result["verdict"]

    def test_overconfidence_is_named_as_such(self) -> None:
        """The usual finding for a tree ensemble, and the one that matters when the
        number is shown to a person looking at a wellbeing indicator."""
        probabilities = np.array([[0.99, 0.01]] * 10)
        y_true = np.array([0] * 5 + [1] * 5)

        result = calibration_metrics(y_true, probabilities)
        assert result["confidence_minus_accuracy"] > 0
        assert "overconfident" in result["verdict"]

    def test_underconfidence_is_named_as_such(self) -> None:
        probabilities = np.array([[0.55, 0.45]] * 10)
        y_true = np.zeros(10, dtype=int)

        result = calibration_metrics(y_true, probabilities)
        assert result["confidence_minus_accuracy"] < 0
        assert "underconfident" in result["verdict"]

    def test_the_table_is_returned_alongside_the_summary(self) -> None:
        # ECE hides direction; the table is what shows where the model goes wrong.
        result = calibration_metrics(np.array([0, 1]), np.array([[0.7, 0.3], [0.4, 0.6]]))
        assert len(result["reliability_table"]) == result["n_bins"]

    def test_ece_of_an_empty_table_is_zero_not_an_error(self) -> None:
        assert expected_calibration_error([]) == 0.0


class TestBrier:
    """Rewards being uncertain when uncertain."""

    def test_a_confident_correct_prediction_scores_zero(self) -> None:
        assert brier_score(np.array([0]), np.array([[1.0, 0.0, 0.0]])) == 0.0

    def test_a_confident_wrong_prediction_scores_the_maximum(self) -> None:
        assert brier_score(np.array([1]), np.array([[1.0, 0.0, 0.0]])) == pytest.approx(2.0)

    def test_hedging_beats_being_confidently_wrong(self) -> None:
        y_true = np.array([1])
        confident_wrong = brier_score(y_true, np.array([[0.9, 0.1, 0.0]]))
        hedged = brier_score(y_true, np.array([[0.34, 0.33, 0.33]]))
        assert hedged < confident_wrong

    def test_an_empty_test_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            brier_score(np.array([]), np.zeros((0, 3)))
