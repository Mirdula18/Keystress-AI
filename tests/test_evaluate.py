"""
Validation harness tests (F5).

Two kinds of claim are checked here. The mechanical ones — a report has these fields, the
file is written where it says — are cheap. The ones that matter are about what the harness
*refuses* to do: call a synthetic run validation, produce a report without its caveats, or
split a dataset in a way that would inflate the result.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from keystress.core.disclosure import FEATURES_V1
from keystress.core.model import ModelBundle
from keystress.ml.evaluate import (
    HEADLINE_METRIC,
    evaluate,
    format_report,
    load_labelled_dataset,
    save_report,
)
from keystress.ml.splits import InsufficientDataError
from keystress.ml.validation import (
    EVALUATED_NO_SKILL,
    NOT_VALIDATED,
    VALIDATED_ON_REAL,
)


def labelled_frame(n_participants: int = 12, sessions: int = 4, seed: int = 5) -> pd.DataFrame:
    """
    A labelled dataset with a genuine (if noisy) relationship between speed and label.

    Built so a real model can plausibly beat the baselines — a dataset with no signal at
    all would make every "did it beat the baseline" test trivially negative and hide a
    harness that always says no.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(n_participants):
        label = participant % 3
        for _ in range(sessions):
            row = {
                "participant_id": f"p{participant}",
                "label": label,
                "avg_typing_speed": 5.0 - label * 1.5 + rng.normal(0, 0.2),
                "avg_inter_key_delay": 0.2 + label * 0.1 + rng.normal(0, 0.02),
                "max_pause_duration": 1.0 + label * 0.5 + rng.normal(0, 0.1),
                "backspace_ratio": 0.05 + label * 0.05 + rng.normal(0, 0.01),
                "typing_consistency": 0.1 + label * 0.05 + rng.normal(0, 0.01),
            }
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_bundle() -> ModelBundle:
    """A model trained on the same kind of data the tests evaluate against."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    frame = labelled_frame(seed=99)
    X = frame[list(FEATURES_V1)].to_numpy()
    y = frame["label"].to_numpy()

    scaler = StandardScaler().fit(X)
    estimator = RandomForestClassifier(n_estimators=20, random_state=1, n_jobs=1)
    estimator.fit(scaler.transform(X), y)

    return ModelBundle(
        estimator=estimator,
        scaler=scaler,
        metadata={
            "model_version": "rf-v1-synthetic-test",
            "data_source": "synthetic",
            "feature_set": "v1",
        },
    )


class TestLoading:
    def test_loads_an_exported_dataset(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        labelled_frame().to_csv(path, index=False)
        assert len(load_labelled_dataset(path)) == 48

    def test_a_missing_file_is_reported_clearly(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="keystress-export"):
            load_labelled_dataset(tmp_path / "absent.csv")

    def test_a_dataset_without_the_grouping_column_is_refused(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        labelled_frame().drop(columns=["participant_id"]).to_csv(path, index=False)
        with pytest.raises(InsufficientDataError, match="participant_id"):
            load_labelled_dataset(path)

    def test_an_empty_dataset_is_refused(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        labelled_frame().iloc[:0].to_csv(path, index=False)
        with pytest.raises(InsufficientDataError, match="no rows"):
            load_labelled_dataset(path)


class TestReportContents:
    def test_report_names_the_model_and_the_data_source(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        assert report["model_version"] == "rf-v1-synthetic-test"
        assert report["dataset"]["data_source"] == "real"
        assert report["metrics_data_source"] == "real"

    def test_baselines_are_always_present(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        assert set(report["baselines"]) == {"majority", "stratified", "uniform"}
        assert set(report["baseline_headline_scores"]) == {"majority", "stratified", "uniform"}

    def test_the_headline_metric_is_named_once(self, fitted_bundle) -> None:
        # So the comparison, the report, and the validation verdict cannot diverge.
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        assert report["headline_metric"] == HEADLINE_METRIC
        assert report["headline_score"] == report["model"][HEADLINE_METRIC]

    def test_calibration_travels_with_the_report(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        assert "expected_calibration_error" in report["calibration"]
        assert report["calibration"]["verdict"]

    def test_the_split_was_participant_grouped(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        assert report["split_strategy"] == "participant-grouped"

    def test_an_unknown_data_source_is_refused(self, fitted_bundle) -> None:
        # A report that cannot say what it measured is worse than no report.
        with pytest.raises(ValueError, match="Unknown data_source"):
            evaluate(fitted_bundle, labelled_frame(), data_source="probably-real")

    def test_evaluation_is_reproducible(self, fitted_bundle) -> None:
        frame = labelled_frame()
        first = evaluate(fitted_bundle, frame, data_source="real", seed=11)
        second = evaluate(fitted_bundle, frame, data_source="real", seed=11)
        assert first["headline_score"] == second["headline_score"]
        assert first["baseline_headline_scores"] == second["baseline_headline_scores"]


class TestWarningsTravelWithTheReport:
    """A report read six months later must carry its own caveats."""

    def test_a_synthetic_run_says_it_cannot_validate(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="synthetic")
        assert any("cannot validate" in warning for warning in report["warnings"])

    def test_a_thin_participant_pool_is_warned_about(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(n_participants=6), data_source="real")
        assert any("participant" in warning for warning in report["warnings"])

    def test_failing_to_beat_a_baseline_is_stated_in_words(self, fitted_bundle) -> None:
        """
        The number alone can be read charitably; the sentence cannot. A model that loses
        to "always guess the most common class" must say so in the report.
        """
        noise = labelled_frame()
        rng = np.random.default_rng(3)
        noise["label"] = rng.integers(0, 3, size=len(noise))

        report = evaluate(fitted_bundle, noise, data_source="real")
        if not report["beats_all_baselines"]:
            assert any("did not beat" in warning for warning in report["warnings"])
            assert any("no useful skill" in warning for warning in report["warnings"])

    def test_warnings_are_persisted_not_only_printed(self, fitted_bundle, tmp_path) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="synthetic")
        path = save_report(report, tmp_path)
        assert json.loads(path.read_text(encoding="utf-8"))["warnings"]


class TestValidationStatus:
    """The distinction the whole project turns on."""

    def test_a_synthetic_run_never_confers_validation(self, fitted_bundle) -> None:
        """However good the numbers are. This is the failure CLAUDE.md §1 is about."""
        report = evaluate(fitted_bundle, labelled_frame(), data_source="synthetic")
        assert report["validation_status"] == NOT_VALIDATED

    def test_a_real_run_that_beats_the_baselines_validates(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        expected = VALIDATED_ON_REAL if report["beats_all_baselines"] else EVALUATED_NO_SKILL
        assert report["validation_status"] == expected

    def test_a_real_run_with_no_skill_is_its_own_state(self, fitted_bundle) -> None:
        """
        "Evaluated and found wanting" is a real result, distinct from "never tested". It
        gets its own status rather than being flattened into not-validated.
        """
        noise = labelled_frame()
        rng = np.random.default_rng(7)
        noise["label"] = rng.integers(0, 3, size=len(noise))

        report = evaluate(fitted_bundle, noise, data_source="real")
        if not report["beats_all_baselines"]:
            assert report["validation_status"] == EVALUATED_NO_SKILL
            assert "did NOT beat" in report["validation_explanation"]

    def test_the_explanation_is_carried_alongside_the_status(self, fitted_bundle) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="synthetic")
        assert "synthetic" in report["validation_explanation"].lower()


class TestPersistence:
    def test_report_is_written_per_model_version_and_dataset(
        self, fitted_bundle, tmp_path
    ) -> None:
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        path = save_report(report, tmp_path)

        assert path.name.startswith("rf-v1-synthetic-test__real__")
        assert json.loads(path.read_text(encoding="utf-8"))["model_version"] == (
            "rf-v1-synthetic-test"
        )

    def test_report_is_json_serialisable_in_full(self, fitted_bundle) -> None:
        # numpy types leak into JSON silently until something tries to write them.
        report = evaluate(fitted_bundle, labelled_frame(), data_source="real")
        json.dumps(report)


class TestFormatting:
    """What a person sees in the terminal."""

    def test_the_data_source_is_in_the_headline(self, fitted_bundle) -> None:
        text = format_report(evaluate(fitted_bundle, labelled_frame(), data_source="synthetic"))
        assert "measured on SYNTHETIC DATA" in text

    def test_baselines_are_printed_beside_the_model(self, fitted_bundle) -> None:
        text = format_report(evaluate(fitted_bundle, labelled_frame(), data_source="real"))
        assert "MODEL vs TRIVIAL BASELINES" in text
        for name in ("majority", "stratified", "uniform"):
            assert name in text

    def test_the_baseline_verdict_is_a_sentence(self, fitted_bundle) -> None:
        text = format_report(evaluate(fitted_bundle, labelled_frame(), data_source="real"))
        assert "beat every trivial baseline" in text or "did NOT beat" in text

    def test_warnings_are_printed_under_a_heading_that_cannot_be_missed(
        self, fitted_bundle
    ) -> None:
        text = format_report(evaluate(fitted_bundle, labelled_frame(), data_source="synthetic"))
        assert "READ THIS BEFORE QUOTING ANY NUMBER ABOVE" in text

    def test_per_class_rows_are_printed(self, fitted_bundle) -> None:
        text = format_report(evaluate(fitted_bundle, labelled_frame(), data_source="real"))
        for name in ("Low", "Medium", "High"):
            assert name in text


class TestCli:
    """The `keystress-evaluate` entrypoint."""

    def _install_model(self, bundle: ModelBundle, tmp_path, monkeypatch) -> None:
        """Write the bundle to disk and point the settings at it."""
        import joblib

        model_path = tmp_path / "model.pkl"
        scaler_path = tmp_path / "scaler.pkl"
        metadata_path = tmp_path / "metadata.json"

        joblib.dump(bundle.estimator, model_path)
        joblib.dump(bundle.scaler, scaler_path)
        metadata_path.write_text(json.dumps(bundle.metadata), encoding="utf-8")

        monkeypatch.setenv("KEYSTRESS_MODEL_PATH", str(model_path))
        monkeypatch.setenv("KEYSTRESS_SCALER_PATH", str(scaler_path))
        monkeypatch.setenv("KEYSTRESS_METADATA_PATH", str(metadata_path))

    def test_data_source_must_be_stated(self, tmp_path) -> None:
        """
        Inferring it from the filename would let a mislabelled file silently promote a
        model to "validated" — the one mistake this harness exists to prevent.
        """
        from keystress.ml.evaluate import main

        with pytest.raises(SystemExit):
            main(["--dataset", str(tmp_path / "x.csv")])

    def test_no_model_on_disk_exits_one_with_advice(self, tmp_path, monkeypatch, caplog) -> None:
        from keystress.ml.evaluate import main

        monkeypatch.setenv("KEYSTRESS_MODEL_PATH", str(tmp_path / "absent.pkl"))
        monkeypatch.setenv("KEYSTRESS_SCALER_PATH", str(tmp_path / "absent-scaler.pkl"))

        assert main(["--dataset", str(tmp_path / "d.csv"), "--data-source", "real"]) == 1
        assert "keystress-train" in caplog.text

    def test_a_missing_dataset_exits_one(self, fitted_bundle, tmp_path, monkeypatch) -> None:
        from keystress.ml.evaluate import main

        self._install_model(fitted_bundle, tmp_path, monkeypatch)
        assert main([
            "--dataset", str(tmp_path / "absent.csv"), "--data-source", "real",
        ]) == 1

    def test_a_full_run_writes_a_report(self, fitted_bundle, tmp_path, monkeypatch, capsys) -> None:
        from keystress.ml.evaluate import main

        self._install_model(fitted_bundle, tmp_path, monkeypatch)
        dataset = tmp_path / "labelled.csv"
        labelled_frame().to_csv(dataset, index=False)
        report_dir = tmp_path / "eval"

        exit_code = main([
            "--dataset", str(dataset), "--data-source", "real",
            "--report-dir", str(report_dir),
        ])

        assert exit_code == 0
        assert "MODEL EVALUATION" in capsys.readouterr().out
        assert list(report_dir.glob("*.json")), "no report was persisted"

    def test_a_model_with_no_skill_still_exits_zero(
        self, fitted_bundle, tmp_path, monkeypatch
    ) -> None:
        """
        A negative result is a successful evaluation. Exiting non-zero would make an
        honest finding look like a broken run, and would eventually get "fixed".
        """
        from keystress.ml.evaluate import main

        self._install_model(fitted_bundle, tmp_path, monkeypatch)
        noise = labelled_frame()
        rng = np.random.default_rng(13)
        noise["label"] = rng.integers(0, 3, size=len(noise))
        dataset = tmp_path / "noise.csv"
        noise.to_csv(dataset, index=False)

        assert main([
            "--dataset", str(dataset), "--data-source", "real",
            "--report-dir", str(tmp_path / "eval"),
        ]) == 0

    def test_no_save_skips_persistence(self, fitted_bundle, tmp_path, monkeypatch) -> None:
        from keystress.ml.evaluate import main

        self._install_model(fitted_bundle, tmp_path, monkeypatch)
        dataset = tmp_path / "labelled.csv"
        labelled_frame().to_csv(dataset, index=False)
        report_dir = tmp_path / "eval"

        assert main([
            "--dataset", str(dataset), "--data-source", "real",
            "--report-dir", str(report_dir), "--no-save",
        ]) == 0
        assert not report_dir.exists()
