"""
Participant-grouped splits (F5).

The single most effective way to overstate a model like this one is a random split. One
person contributing several sessions has a typing rhythm that is recognisable across all
of them; put some of their sessions in train and the rest in test and the model can score
well by identifying *the person*, not their burnout. The resulting number is real,
reproducible, and meaningless.

So every split here is by **participant**, never by row: everything one person contributed
lands wholly on one side. That is the property this module exists to guarantee, and
:func:`assert_no_participant_leak` states it as a check the caller can run against any
split, including one produced elsewhere.

The cost is honest and worth naming: grouped splits are noisier. With few participants the
test set may be dominated by two or three people, and the metric will swing depending on
who they are. That is not a flaw in the method — it is the actual uncertainty, which a
random split hides rather than removes. :func:`split_summary` reports how few participants
a split rests on so the reader can weigh it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Column identifying who contributed a row. Exported by
#: :mod:`keystress.research.dataset` and required by every function here.
GROUP_COLUMN = "participant_id"

#: A split needs at least this many participants to mean anything. Two participants means
#: a "held-out set" is one person, and the metric describes that person.
MIN_PARTICIPANTS_FOR_SPLIT = 4


class InsufficientDataError(ValueError):
    """
    Raised when a dataset cannot support an honest split.

    A distinct exception because the correct response is different from other errors:
    collect more data. Silently returning a degenerate split — an empty test set, or one
    holding a single participant — would produce a number that looks like a result.
    """


@dataclass(frozen=True)
class Split:
    """
    A train/test partition and the facts needed to judge it.

    Attributes:
        train: Training rows.
        test: Held-out rows.
        train_participants: Participant ids in the training half.
        test_participants: Participant ids in the held-out half.
        seed: Seed used to assign participants.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    train_participants: tuple[str, ...]
    test_participants: tuple[str, ...]
    seed: int

    def summary(self) -> dict[str, Any]:
        """
        Describe the split, including why it might be too small to trust.

        Returns:
            dict: Row and participant counts per side, plus ``warnings``.
        """
        warnings: list[str] = []
        if len(self.test_participants) < 2:
            warnings.append(
                "The held-out set contains one participant, so any metric computed on it "
                "describes that individual rather than the population."
            )
        if len(self.test) < 10:
            warnings.append(
                f"Only {len(self.test)} held-out session(s); metrics from a set this "
                "small are dominated by sampling noise."
            )
        return {
            "n_train": len(self.train),
            "n_test": len(self.test),
            "n_train_participants": len(self.train_participants),
            "n_test_participants": len(self.test_participants),
            "seed": self.seed,
            "warnings": warnings,
        }


def assert_no_participant_leak(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Verify that no participant appears on both sides of a split.

    Cheap enough to run on every split, including ones this module did not create. The
    failure it catches is invisible in the metrics — a leaked split produces *better*
    numbers, not worse ones, which is exactly why it needs asserting rather than
    reviewing.

    Parameters:
        train: Training rows.
        test: Held-out rows.

    Raises:
        AssertionError: If any participant id occurs in both frames.
    """
    shared = set(train[GROUP_COLUMN]) & set(test[GROUP_COLUMN])
    assert not shared, (
        f"{len(shared)} participant(s) appear in both train and test: "
        f"{sorted(shared)[:3]}. Metrics from this split would be inflated by the model "
        "recognising the person rather than the signal."
    )


def grouped_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.3,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
) -> Split:
    """
    Split a dataset by participant.

    Participants are shuffled with a fixed seed and assigned whole to one side, taking
    participants into the test set until it reaches ``test_size`` of the rows. Balancing
    on rows rather than on participant count matters when contributions are uneven: three
    people who each donated once are not a third of a dataset where a fourth donated
    thirty times.

    Parameters:
        frame: Labelled dataset, carrying ``group_column``.
        test_size: Target proportion of *rows* held out, 0-1 exclusive.
        seed: Seed for the participant shuffle.
        group_column: Column identifying the participant.

    Returns:
        Split: The partition, with both participant lists recorded.

    Raises:
        InsufficientDataError: If the frame is empty, lacks the group column, or holds
            fewer than :data:`MIN_PARTICIPANTS_FOR_SPLIT` participants.
        ValueError: If ``test_size`` is not strictly between 0 and 1.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    if group_column not in frame.columns:
        raise InsufficientDataError(
            f"Dataset has no {group_column!r} column, so it cannot be split by "
            "participant. A random split would inflate every metric; refusing rather "
            "than falling back to one."
        )

    participants = list(dict.fromkeys(frame[group_column].tolist()))
    if len(participants) < MIN_PARTICIPANTS_FOR_SPLIT:
        raise InsufficientDataError(
            f"Only {len(participants)} participant(s); at least "
            f"{MIN_PARTICIPANTS_FOR_SPLIT} are needed for a grouped split to describe "
            "anything beyond those individuals."
        )

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(participants))

    counts = frame[group_column].value_counts().to_dict()
    target_rows = len(frame) * test_size

    test_participants: list[str] = []
    held = 0
    # Leave at least one participant on each side: a target that would consume everyone
    # is a smaller failure than an empty train set discovered later.
    for participant in order[:-1]:
        if held >= target_rows:
            break
        test_participants.append(participant)
        held += counts[participant]

    if not test_participants:
        test_participants = [order[0]]

    test_ids = set(test_participants)
    test = frame[frame[group_column].isin(test_ids)].copy()
    train = frame[~frame[group_column].isin(test_ids)].copy()

    assert_no_participant_leak(train, test)

    logger.info(
        "Grouped split: %d train rows (%d participants) / %d test rows (%d participants)",
        len(train), len(train[group_column].unique()),
        len(test), len(test_ids),
    )

    return Split(
        train=train,
        test=test,
        train_participants=tuple(dict.fromkeys(train[group_column].tolist())),
        test_participants=tuple(test_participants),
        seed=seed,
    )


def grouped_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
) -> list[Split]:
    """
    Build k participant-grouped folds.

    Preferred over a single split whenever there are enough participants: with a small
    dataset one split's metric depends heavily on who happened to land in the test set,
    and k folds at least make that variability visible instead of hiding it behind one
    number.

    Parameters:
        frame: Labelled dataset.
        n_folds: Number of folds. Reduced to the participant count if there are fewer
            participants than folds, since an empty fold helps nobody.
        seed: Seed for the participant shuffle.
        group_column: Column identifying the participant.

    Returns:
        list[Split]: One split per fold, each holding out a distinct participant group.

    Raises:
        InsufficientDataError: If there are fewer than
            :data:`MIN_PARTICIPANTS_FOR_SPLIT` participants.
        ValueError: If ``n_folds`` is less than 2.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    if group_column not in frame.columns:
        raise InsufficientDataError(f"Dataset has no {group_column!r} column")

    participants = list(dict.fromkeys(frame[group_column].tolist()))
    if len(participants) < MIN_PARTICIPANTS_FOR_SPLIT:
        raise InsufficientDataError(
            f"Only {len(participants)} participant(s); at least "
            f"{MIN_PARTICIPANTS_FOR_SPLIT} are needed for grouped folds."
        )

    effective_folds = min(n_folds, len(participants))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(participants))
    buckets: list[list[str]] = [list(bucket) for bucket in np.array_split(order, effective_folds)]

    splits: list[Split] = []
    for bucket in buckets:
        test_ids = set(bucket)
        test = frame[frame[group_column].isin(test_ids)].copy()
        train = frame[~frame[group_column].isin(test_ids)].copy()
        assert_no_participant_leak(train, test)
        splits.append(Split(
            train=train,
            test=test,
            train_participants=tuple(dict.fromkeys(train[group_column].tolist())),
            test_participants=tuple(bucket),
            seed=seed,
        ))
    return splits


def split_summary(splits: Sequence[Split]) -> dict[str, Any]:
    """
    Summarise one or more splits.

    Parameters:
        splits: Splits to describe.

    Returns:
        dict: Fold count, per-fold summaries, and the union of their warnings.
    """
    summaries = [split.summary() for split in splits]
    warnings: list[str] = []
    for index, summary in enumerate(summaries):
        warnings.extend(f"fold {index}: {warning}" for warning in summary["warnings"])
    return {
        "n_folds": len(summaries),
        "folds": summaries,
        "warnings": warnings,
    }
