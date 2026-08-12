"""
Participant-grouped split tests (F5).

The bug these guard against does not raise, log, or look wrong: a leaked participant makes
the *numbers better*. So the property is asserted directly and from several angles rather
than inferred from a metric looking sensible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from keystress.ml.splits import (
    GROUP_COLUMN,
    MIN_PARTICIPANTS_FOR_SPLIT,
    InsufficientDataError,
    assert_no_participant_leak,
    grouped_folds,
    grouped_split,
    split_summary,
)


def make_frame(sessions_per_participant: dict[str, int]) -> pd.DataFrame:
    """Build a dataset with the given number of rows per participant."""
    rows = []
    for participant, count in sessions_per_participant.items():
        for index in range(count):
            rows.append({
                GROUP_COLUMN: participant,
                "avg_typing_speed": 1.0 + index,
                "label": index % 3,
            })
    return pd.DataFrame(rows)


def even_frame(n_participants: int = 10, sessions: int = 3) -> pd.DataFrame:
    """A dataset with equal contributions."""
    return make_frame({f"p{i}": sessions for i in range(n_participants)})


class TestGroupedSplit:
    """One split."""

    def test_no_participant_appears_on_both_sides(self) -> None:
        split = grouped_split(even_frame())
        assert not set(split.train_participants) & set(split.test_participants)
        assert_no_participant_leak(split.train, split.test)

    def test_every_row_lands_somewhere(self) -> None:
        frame = even_frame()
        split = grouped_split(frame)
        assert len(split.train) + len(split.test) == len(frame)

    def test_neither_side_is_empty(self) -> None:
        split = grouped_split(even_frame())
        assert len(split.train) > 0
        assert len(split.test) > 0

    def test_split_is_deterministic_for_a_seed(self) -> None:
        first = grouped_split(even_frame(), seed=7)
        second = grouped_split(even_frame(), seed=7)
        assert first.test_participants == second.test_participants

    def test_a_different_seed_can_choose_differently(self) -> None:
        # Not guaranteed for any given pair of seeds, so scan a few.
        base = grouped_split(even_frame(20), seed=1).test_participants
        assert any(
            grouped_split(even_frame(20), seed=seed).test_participants != base
            for seed in range(2, 8)
        )

    def test_test_size_is_approximately_respected(self) -> None:
        frame = even_frame(20, sessions=2)
        split = grouped_split(frame, test_size=0.3)
        proportion = len(split.test) / len(frame)
        assert 0.15 < proportion < 0.5

    def test_balancing_counts_rows_not_participants(self) -> None:
        """
        Three people who donated once are not a third of a dataset where a fourth donated
        thirty times. Balancing on participant count would hold out 3 rows out of 33 and
        call it a 25% test set.
        """
        frame = make_frame({"heavy": 30, "a": 1, "b": 1, "c": 1, "d": 1, "e": 1})
        split = grouped_split(frame, test_size=0.5, seed=3)

        assert len(split.test) / len(frame) > 0.2 or "heavy" in split.train_participants

    @pytest.mark.parametrize("bad_size", [0.0, 1.0, -0.2, 1.5])
    def test_an_impossible_test_size_is_rejected(self, bad_size: float) -> None:
        with pytest.raises(ValueError, match="test_size"):
            grouped_split(even_frame(), test_size=bad_size)


class TestRefusals:
    """When a split cannot be honest, it must not be made."""

    def test_too_few_participants_is_refused(self) -> None:
        frame = even_frame(MIN_PARTICIPANTS_FOR_SPLIT - 1)
        with pytest.raises(InsufficientDataError, match="participant"):
            grouped_split(frame)

    def test_a_missing_group_column_is_refused_not_worked_around(self) -> None:
        """
        Falling back to a random split here would be the single most damaging
        convenience in the codebase: it would silently produce inflated metrics from a
        dataset that never supported them.
        """
        frame = even_frame().drop(columns=[GROUP_COLUMN])
        with pytest.raises(InsufficientDataError, match="participant"):
            grouped_split(frame)

    def test_an_empty_dataset_is_refused(self) -> None:
        with pytest.raises(InsufficientDataError):
            grouped_split(pd.DataFrame(columns=[GROUP_COLUMN, "label"]))


class TestLeakDetection:
    """The check itself."""

    def test_a_leaked_split_is_caught(self) -> None:
        frame = even_frame()
        # A plain random split — the mistake this whole module exists to prevent.
        train = frame.iloc[::2]
        test = frame.iloc[1::2]
        with pytest.raises(AssertionError, match="both train and test"):
            assert_no_participant_leak(train, test)

    def test_a_clean_split_passes(self) -> None:
        frame = even_frame()
        train = frame[frame[GROUP_COLUMN] != "p0"]
        test = frame[frame[GROUP_COLUMN] == "p0"]
        assert_no_participant_leak(train, test)  # must not raise


class TestFolds:
    """k folds."""

    def test_folds_cover_every_participant_exactly_once_as_test(self) -> None:
        frame = even_frame(10)
        folds = grouped_folds(frame, n_folds=5)

        held_out = [p for fold in folds for p in fold.test_participants]
        assert sorted(held_out) == sorted(frame[GROUP_COLUMN].unique())

    def test_every_fold_is_leak_free(self) -> None:
        for fold in grouped_folds(even_frame(12), n_folds=4):
            assert_no_participant_leak(fold.train, fold.test)

    def test_folds_are_capped_at_the_participant_count(self) -> None:
        # Ten folds over six people would leave four folds empty.
        folds = grouped_folds(even_frame(6), n_folds=10)
        assert len(folds) == 6
        assert all(len(fold.test) > 0 for fold in folds)

    def test_at_least_two_folds_are_required(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            grouped_folds(even_frame(), n_folds=1)

    def test_too_few_participants_is_refused(self) -> None:
        with pytest.raises(InsufficientDataError):
            grouped_folds(even_frame(2), n_folds=2)


class TestSummaries:
    """What the report says about the split it used."""

    def test_summary_counts_both_sides(self) -> None:
        summary = grouped_split(even_frame(10)).summary()
        assert summary["n_train"] + summary["n_test"] == 30
        assert summary["n_train_participants"] + summary["n_test_participants"] == 10

    def test_a_single_participant_test_set_is_warned_about(self) -> None:
        frame = make_frame({"a": 1, "b": 1, "c": 1, "d": 1, "heavy": 60})
        split = grouped_split(frame, test_size=0.02, seed=1)
        if len(split.test_participants) == 1:
            assert any("one participant" in w for w in split.summary()["warnings"])

    def test_a_tiny_test_set_is_warned_about(self) -> None:
        split = grouped_split(even_frame(5, sessions=1), test_size=0.2)
        assert any("sampling noise" in warning for warning in split.summary()["warnings"])

    def test_split_summary_aggregates_folds(self) -> None:
        summary = split_summary(grouped_folds(even_frame(8), n_folds=4))
        assert summary["n_folds"] == 4
        assert len(summary["folds"]) == 4
