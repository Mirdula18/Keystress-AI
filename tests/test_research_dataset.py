"""
Labelled dataset export tests (F4).

The export is the artifact everything else in the research path exists to produce, so the
tests cover both what it contains and what it must refuse to contain — in particular the
model's own prediction, whose presence beside the label would be one careless join away
from training on the current model's output.
"""

from __future__ import annotations

import csv
import json

import pytest

from keystress.core.disclosure import FEATURES_V1
from keystress.core.storage import Store
from keystress.research import instrument as inst
from keystress.research.dataset import (
    DATASET_COLUMNS,
    DATASET_VERSION,
    build_rows,
    describe,
    export,
    format_summary,
    main,
    write_csv,
    write_jsonl,
)
from keystress.research.scoring import score_responses


def _answers(value: int) -> dict[str, int]:
    return {item.id: value for item in inst.ITEMS}


def add_labelled_session(store: Store, participant: str, *, answer: int = 50,
                         speed: float = 3.0) -> int:
    """Donate a typing session and label it with a questionnaire."""
    features = dict.fromkeys(FEATURES_V1, 1.0)
    features["avg_typing_speed"] = speed
    donation_id = store.save_donation(features=features, participant_id=participant,
                                      data_source="synthetic", prediction=1)
    store.save_response(participant, score_responses(_answers(answer)),
                        donation_id=donation_id)
    return donation_id


@pytest.fixture
def populated_store(store: Store) -> Store:
    """Two participants, three labelled sessions between them."""
    first = store.create_participant(analysis=True, donate=True)["participant_id"]
    second = store.create_participant(analysis=True, donate=True)["participant_id"]
    add_labelled_session(store, first, answer=0, speed=2.0)
    add_labelled_session(store, first, answer=50, speed=4.0)
    add_labelled_session(store, second, answer=100, speed=6.0)
    return store


class TestRows:
    """What a row contains."""

    def test_one_row_per_labelled_pair(self, populated_store: Store) -> None:
        assert len(build_rows(populated_store)) == 3

    def test_rows_carry_every_declared_column(self, populated_store: Store) -> None:
        for row in build_rows(populated_store):
            assert set(row) == set(DATASET_COLUMNS)

    def test_features_are_floats(self, populated_store: Store) -> None:
        row = build_rows(populated_store)[0]
        for name in FEATURES_V1:
            assert isinstance(row[name], float)

    def test_the_grouping_key_is_present(self, populated_store: Store) -> None:
        # Without it, F5 cannot keep one person out of both sides of a split.
        assert all(row["participant_id"] for row in build_rows(populated_store))

    def test_the_model_prediction_is_not_exported(self, populated_store: Store) -> None:
        """
        The store keeps the synthetic model's guess for auditing; the dataset must not.

        A file holding both the real label and the current model's prediction is one
        careless join away from training on its own output — manufacturing exactly the
        circularity this dataset exists to escape.
        """
        for column in ("prediction", "model_prediction", "data_source", "model_data_source"):
            assert column not in DATASET_COLUMNS
        for row in build_rows(populated_store):
            assert "prediction" not in row

    def test_rows_are_oldest_first(self, populated_store: Store) -> None:
        speeds = [row["avg_typing_speed"] for row in build_rows(populated_store)]
        assert speeds == [2.0, 4.0, 6.0]

    def test_an_empty_store_exports_no_rows(self, store: Store) -> None:
        assert build_rows(store) == []


class TestDescribe:
    """The summary that travels with an export."""

    def test_counts_records_and_participants(self, populated_store: Store) -> None:
        summary = describe(build_rows(populated_store))
        assert summary["n_records"] == 3
        assert summary["n_participants"] == 2
        assert summary["dataset_version"] == DATASET_VERSION

    def test_label_counts_are_named_not_numbered(self, populated_store: Store) -> None:
        summary = describe(build_rows(populated_store))
        assert set(summary["label_counts"]) <= {
            "below the burnout range", "moderate", "high",
        }

    def test_a_small_dataset_is_warned_about(self, populated_store: Store) -> None:
        """
        Three sessions from two people is not evidence, and the export must say so at the
        moment it is produced rather than in a footnote nobody reads later.
        """
        warnings = describe(build_rows(populated_store))["warnings"]
        assert any("too few" in warning for warning in warnings)
        assert any("participant" in warning for warning in warnings)

    def test_a_missing_class_is_warned_about(self, store: Store) -> None:
        participant = store.create_participant(analysis=True, donate=True)["participant_id"]
        add_labelled_session(store, participant, answer=0)

        warnings = describe(build_rows(store))["warnings"]
        assert any("cannot be evaluated on a class that never appears" in w for w in warnings)

    def test_one_dominant_participant_is_warned_about(self, store: Store) -> None:
        heavy = store.create_participant(analysis=True, donate=True)["participant_id"]
        light = store.create_participant(analysis=True, donate=True)["participant_id"]
        for _ in range(5):
            add_labelled_session(store, heavy)
        add_labelled_session(store, light)

        warnings = describe(build_rows(store))["warnings"]
        assert any("more than half the rows" in warning for warning in warnings)

    def test_an_empty_dataset_describes_itself_without_crashing(self, store: Store) -> None:
        summary = describe(build_rows(store))
        assert summary["n_records"] == 0
        assert summary["sessions_per_participant"] == {"min": 0, "max": 0}
        assert summary["warnings"]


class TestWriting:
    """The files themselves."""

    def test_csv_has_a_header_and_one_row_each(self, populated_store: Store, tmp_path) -> None:
        path = write_csv(build_rows(populated_store), tmp_path / "out" / "data.csv")

        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 3
        assert list(rows[0]) == list(DATASET_COLUMNS)

    def test_jsonl_keeps_column_order(self, populated_store: Store, tmp_path) -> None:
        path = write_jsonl(build_rows(populated_store), tmp_path / "data.jsonl")
        lines = path.read_text(encoding="utf-8").strip().splitlines()

        assert len(lines) == 3
        assert list(json.loads(lines[0])) == list(DATASET_COLUMNS)

    def test_export_returns_the_summary_with_the_path(
        self, populated_store: Store, tmp_path
    ) -> None:
        summary = export(populated_store, tmp_path / "data.csv")
        assert summary["n_records"] == 3
        assert summary["path"].endswith("data.csv")

    def test_unknown_format_is_rejected(self, populated_store: Store, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            export(populated_store, tmp_path / "data.parquet", fmt="parquet")

    def test_export_columns_are_exactly_the_declared_schema(
        self, populated_store: Store, tmp_path
    ) -> None:
        """
        HARD RULE 1 at the last boundary: the file that leaves the machine.

        An allowlist, not a blocklist of suspicious words — the feature name
        `avg_inter_key_delay` legitimately contains "key", and a blocklist that flagged
        it would be turned off within a week. Every column in the file must be one this
        module declares, so a new content-bearing column cannot appear without appearing
        in `DATASET_COLUMNS` first, where it is reviewable.
        """
        path = tmp_path / "data.csv"
        export(populated_store, path)

        header = path.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(DATASET_COLUMNS)


class TestSummaryRendering:
    """What a human sees."""

    def test_warnings_are_printed_prominently(self, populated_store: Store) -> None:
        report = format_summary(describe(build_rows(populated_store)))
        assert "THIS DATASET IS NOT YET EVIDENCE" in report

    def test_an_empty_dataset_renders(self, store: Store) -> None:
        report = format_summary(describe(build_rows(store)))
        assert "Records:            0" in report
        assert "(none)" in report


class TestCli:
    """The `keystress-export` entrypoint."""

    def test_exports_from_a_given_store(self, populated_store: Store, tmp_path, capsys) -> None:
        out = tmp_path / "export.csv"
        exit_code = main(["--store", str(populated_store.path), "--out", str(out)])

        assert exit_code == 0
        assert out.exists()
        assert "LABELLED DATASET" in capsys.readouterr().out

    def test_jsonl_format_is_selectable(self, populated_store: Store, tmp_path) -> None:
        out = tmp_path / "export.jsonl"
        assert main(["--store", str(populated_store.path), "--out", str(out),
                     "--format", "jsonl"]) == 0
        assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 3

    def test_a_missing_store_fails_loudly(self, tmp_path) -> None:
        # Exporting nothing because the path was wrong must not look like exporting
        # nothing because nobody has contributed yet.
        assert main(["--store", str(tmp_path / "absent.db"), "--out", str(tmp_path / "x.csv")]) == 1
