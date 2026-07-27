"""
Tests for the offline ML pipeline: synthetic generation and training (F12, F13).

The generator tests are the most important here. The inherited generator created an
artificial point mass by clamping, which a tree model could split on to score well by
detecting the author's clamp rather than any pattern. These tests make that class of
defect a build failure rather than something noticed years later.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from keystress.core.disclosure import FEATURES_V1
from keystress.ml.synthetic import (
    CLASS_PARAMETERS,
    MAX_BACKSPACE_RATIO,
    SYNTHETIC_GENERATOR_VERSION,
    _truncated_normal,
    generate_synthetic_typing_data,
    save_synthetic_data,
)
from keystress.ml.train import (
    FEATURE_COLUMNS,
    build_model_metadata,
    build_model_version,
    evaluate_model,
    format_evaluation_report,
    get_feature_importance,
    load_training_data,
    prepare_data,
    save_model,
    train_and_evaluate,
    train_random_forest,
)

# --------------------------------------------------------------------------------------
# Synthetic generation
# --------------------------------------------------------------------------------------


class TestSyntheticGeneration:
    """Shape, determinism, and the absence of the inherited clamp artifact."""

    def test_produces_requested_sample_count(self) -> None:
        assert len(generate_synthetic_typing_data(n_samples=300, random_state=7)) == 300

    def test_has_the_v1_columns_plus_label(self) -> None:
        df = generate_synthetic_typing_data(n_samples=90, random_state=7)
        assert set(df.columns) == set(FEATURES_V1) | {"burnout_level"}

    def test_classes_are_balanced(self) -> None:
        counts = generate_synthetic_typing_data(
            n_samples=300, random_state=7
        )["burnout_level"].value_counts()
        assert set(counts.index) == {0, 1, 2}
        assert counts.tolist() == [100, 100, 100]

    def test_is_deterministic_for_a_seed(self) -> None:
        """F13 depends on this: a clean checkout must reproduce the dataset."""
        first = generate_synthetic_typing_data(n_samples=150, random_state=42)
        second = generate_synthetic_typing_data(n_samples=150, random_state=42)
        assert first.equals(second)

    def test_different_seeds_differ(self) -> None:
        first = generate_synthetic_typing_data(n_samples=150, random_state=1)
        second = generate_synthetic_typing_data(n_samples=150, random_state=2)
        assert not first.equals(second)


class TestNoPointMass:
    """
    Regression tests for the clamp defect (D-010).

    `max(floor, x)` does not truncate a distribution — it piles every rejected draw onto
    one exact value, giving a tree model a discrete artifact to split on. These tests fail
    if clamping is ever reintroduced.
    """

    @pytest.fixture(scope="class")
    def dataset(self):
        return generate_synthetic_typing_data(n_samples=3000, random_state=99)

    @pytest.mark.parametrize("column,floor", [
        ("typing_consistency", 0.01),
        ("avg_typing_speed", 0.5),
        ("avg_inter_key_delay", 0.05),
    ])
    def test_no_value_sits_exactly_on_a_floor(self, dataset, column: str, floor: float) -> None:
        at_floor = int((dataset[column] == floor).sum())
        assert at_floor == 0, (
            f"{at_floor} rows sit exactly on the {column} floor {floor}: "
            "clamping has been reintroduced, creating a generator artifact"
        )

    @pytest.mark.parametrize("column", list(FEATURES_V1))
    def test_values_are_effectively_continuous(self, dataset, column: str) -> None:
        """No value should repeat: a duplicate is the signature of a clamp."""
        most_common = dataset[column].value_counts().iloc[0]
        assert most_common == 1, (
            f"{column} has a value repeated {most_common} times; "
            "continuous draws should never collide"
        )

    def test_all_values_stay_above_their_floors(self, dataset) -> None:
        for level, params in CLASS_PARAMETERS.items():
            subset = dataset[dataset["burnout_level"] == level]
            for column, (_, _, floor) in params.items():
                assert (subset[column] > floor).all(), f"class {level} {column} below floor"

    def test_backspace_ratio_stays_in_range(self, dataset) -> None:
        assert dataset["backspace_ratio"].between(0.0, MAX_BACKSPACE_RATIO).all()

    def test_backspace_ratio_upper_tail_is_not_flattened(self, dataset) -> None:
        """Scaling a Beta draw preserves its shape; clipping would stack values on 0.5."""
        assert int((dataset["backspace_ratio"] == MAX_BACKSPACE_RATIO).sum()) == 0

    def test_pauses_are_positive(self, dataset) -> None:
        assert (dataset["max_pause_duration"] > 0).all()


class TestTruncatedNormal:
    """The replacement sampler."""

    def test_all_draws_exceed_the_floor(self) -> None:
        rng = np.random.default_rng(3)
        samples = _truncated_normal(rng, mean=0.05, std=0.02, floor=0.01, size=20000)
        assert (samples > 0.01).all()

    def test_no_point_mass_at_the_floor(self) -> None:
        rng = np.random.default_rng(3)
        samples = _truncated_normal(rng, mean=0.05, std=0.02, floor=0.01, size=20000)
        assert int((samples == 0.01).sum()) == 0

    def test_mean_is_close_to_requested(self) -> None:
        """Truncation shifts the mean slightly upward; it must not shift it wildly."""
        rng = np.random.default_rng(3)
        samples = _truncated_normal(rng, mean=5.0, std=0.8, floor=0.5, size=20000)
        assert samples.mean() == pytest.approx(5.0, abs=0.05)

    def test_impossible_bound_raises_instead_of_looping_forever(self) -> None:
        rng = np.random.default_rng(3)
        with pytest.raises(RuntimeError, match="converge"):
            _truncated_normal(rng, mean=0.0, std=0.001, floor=100.0, size=10)


class TestSaveSyntheticData:
    """Persistence."""

    def test_writes_a_readable_csv(self, tmp_path) -> None:
        df = generate_synthetic_typing_data(n_samples=60, random_state=5)
        path = save_synthetic_data(df, tmp_path / "nested" / "out.csv")

        assert path.exists()
        assert len(load_training_data(path)) == 60


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


class TestTrainingPipeline:
    """Training, evaluation, and the metadata that gives metrics meaning."""

    @pytest.fixture(scope="class")
    def dataset_path(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("data") / "synthetic.csv"
        return save_synthetic_data(
            generate_synthetic_typing_data(n_samples=600, random_state=11), path
        )

    def test_load_rejects_a_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_data(tmp_path / "absent.csv")

    def test_load_rejects_a_dataset_missing_columns(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_training_data(path)

    def test_prepare_splits_and_scales(self, dataset_path) -> None:
        df = load_training_data(dataset_path)
        X_train, X_test, y_train, y_test, scaler = prepare_data(df, random_state=11)

        assert len(X_train) + len(X_test) == len(df)
        assert X_train.shape[1] == len(FEATURE_COLUMNS)
        assert scaler.mean_ is not None

    def test_split_is_stratified(self, dataset_path) -> None:
        df = load_training_data(dataset_path)
        _, _, y_train, y_test, _ = prepare_data(df, random_state=11)
        assert set(np.unique(y_train)) == set(np.unique(y_test)) == {0, 1, 2}

    def test_training_produces_a_usable_model(self, dataset_path) -> None:
        df = load_training_data(dataset_path)
        X_train, X_test, y_train, y_test, _ = prepare_data(df, random_state=11)
        model = train_random_forest(X_train, y_train, random_state=11)

        metrics = evaluate_model(model, X_test, y_test)
        for key in ("accuracy", "precision", "recall", "f1_score",
                    "confusion_matrix", "classification_report"):
            assert key in metrics

    def test_feature_importance_covers_every_feature(self, dataset_path) -> None:
        df = load_training_data(dataset_path)
        X_train, _, y_train, _, _ = prepare_data(df, random_state=11)
        importance = get_feature_importance(train_random_forest(X_train, y_train, random_state=11))
        assert set(importance) == set(FEATURE_COLUMNS)

    def test_feature_importance_empty_for_a_model_without_it(self) -> None:
        assert get_feature_importance(object()) == {}


class TestModelVersioning:
    """Version identity, which F13's reproducibility check relies on."""

    def test_version_is_deterministic(self) -> None:
        assert (build_model_version("synthetic", 1500, 42)
                == build_model_version("synthetic", 1500, 42))

    def test_version_contains_no_timestamp(self) -> None:
        """A timestamp would make a clean checkout produce a different version."""
        version = build_model_version("synthetic", 1500, 42)
        assert not any(part.startswith("20") and part[:4].isdigit()
                       for part in version.split("-"))

    def test_version_records_the_generator(self) -> None:
        """A change to the data-generating process must be visible in model identity."""
        assert SYNTHETIC_GENERATOR_VERSION in build_model_version("synthetic", 1500, 42)

    def test_version_reflects_its_inputs(self) -> None:
        versions = {
            build_model_version("synthetic", 1500, 42),
            build_model_version("synthetic", 1500, 7),
            build_model_version("synthetic", 900, 42),
            build_model_version("real", 1500, 42),
        }
        assert len(versions) == 4, "different inputs must yield different versions"


class TestModelMetadata:
    """Metadata is what makes a stored metric interpretable (F1)."""

    @pytest.fixture
    def metrics(self):
        return {
            # metrics-ok: fixture placeholders exercising the metadata shape, not results
            "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9,
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "classification_report": "report",
        }

    def test_metadata_records_the_data_source(self, metrics) -> None:
        metadata = build_model_metadata(metrics, 1500, 42, "synthetic")
        assert metadata["data_source"] == "synthetic"
        assert metadata["metrics_data_source"] == "synthetic"

    def test_synthetic_metadata_carries_the_caveat(self, metrics) -> None:
        metadata = build_model_metadata(metrics, 1500, 42, "synthetic")
        assert "synthetic" in metadata["metrics_caveat"].lower()
        assert metadata["metrics_caveat"], "a synthetic model must carry its caveat"

    def test_metadata_records_feature_set_and_seed(self, metrics) -> None:
        metadata = build_model_metadata(metrics, 1500, 42, "synthetic")
        assert metadata["feature_set"] == "v1"
        assert metadata["random_seed"] == 42
        assert metadata["features"] == list(FEATURE_COLUMNS)

    def test_metadata_is_json_serialisable(self, metrics) -> None:
        json.dumps(build_model_metadata(metrics, 1500, 42, "synthetic"))

    def test_report_qualifies_every_metric(self, metrics) -> None:
        metadata = build_model_metadata(metrics, 1500, 42, "synthetic")
        report = format_evaluation_report(metrics, {"avg_typing_speed": 0.5}, metadata)

        assert report.count("on synthetic data") >= 4, "each headline metric needs a qualifier"
        assert "synthetic-derived" in report, "feature importance must be labelled too"


class TestEndToEndTraining:
    """The full pipeline writing real artifacts."""

    def test_train_and_evaluate_writes_all_three_artifacts(self, tmp_path) -> None:
        data_path = save_synthetic_data(
            generate_synthetic_typing_data(n_samples=300, random_state=13),
            tmp_path / "data.csv",
        )
        model_path = tmp_path / "m" / "model.pkl"
        scaler_path = tmp_path / "m" / "scaler.pkl"
        metadata_path = tmp_path / "m" / "meta.json"

        results = train_and_evaluate(
            data_path=data_path, model_path=model_path,
            scaler_path=scaler_path, metadata_path=metadata_path,
            random_state=13,
        )

        assert model_path.exists() and scaler_path.exists() and metadata_path.exists()
        assert results["metadata"]["data_source"] == "synthetic"

        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert stored["model_version"] == results["metadata"]["model_version"]

    def test_trained_model_loads_through_the_registry(self, tmp_path) -> None:
        """Training and loading must agree on the artifact format."""
        from keystress.core.model import ModelRegistry

        data_path = save_synthetic_data(
            generate_synthetic_typing_data(n_samples=300, random_state=13),
            tmp_path / "data.csv",
        )
        paths = (tmp_path / "model.pkl", tmp_path / "scaler.pkl", tmp_path / "meta.json")
        train_and_evaluate(
            data_path=data_path, model_path=paths[0],
            scaler_path=paths[1], metadata_path=paths[2], random_state=13,
        )

        bundle = ModelRegistry().load(*paths)
        assert bundle.data_source == "synthetic"
        assert bundle.feature_set == "v1"

    def test_save_model_without_metadata_writes_no_sidecar(self, tmp_path) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        X = np.random.default_rng(0).normal(size=(30, 5))
        y = np.array([0, 1, 2] * 10)
        scaler = StandardScaler().fit(X)
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

        metadata_path = tmp_path / "meta.json"
        save_model(model, scaler, tmp_path / "m.pkl", tmp_path / "s.pkl",
                   metadata=None, metadata_path=metadata_path)

        assert not metadata_path.exists()


class TestReproducibility:
    """
    Reproducible builds (F13).

    The CI job runs the full check; these run a smaller version so a seeding regression
    fails locally before it reaches a pull request.
    """

    def test_two_builds_agree(self, tmp_path) -> None:
        from tools.check_reproducible_build import build_once, compare

        first = build_once(tmp_path / "a", 42, 300)
        second = build_once(tmp_path / "b", 42, 300)

        failures = compare(first, second)
        assert not failures, "builds diverged:\n" + "\n".join(failures)

    def test_different_seeds_produce_different_builds(self, tmp_path) -> None:
        """Confirms the check compares something real rather than trivially passing."""
        from tools.check_reproducible_build import build_once, compare

        first = build_once(tmp_path / "a", 42, 300)
        second = build_once(tmp_path / "b", 7, 300)

        assert compare(first, second), "different seeds should not produce identical builds"

    def test_probe_predictions_are_stable_across_reloads(self, tmp_path) -> None:
        """A saved and reloaded model must behave identically to the in-memory one."""
        import numpy as np

        from keystress.core.model import load_bundle
        from tools.check_reproducible_build import PROBE_FEATURES

        data_path = save_synthetic_data(
            generate_synthetic_typing_data(n_samples=300, random_state=21),
            tmp_path / "data.csv",
        )
        paths = (tmp_path / "m.pkl", tmp_path / "s.pkl", tmp_path / "meta.json")
        results = train_and_evaluate(
            data_path=data_path, model_path=paths[0],
            scaler_path=paths[1], metadata_path=paths[2], random_state=21,
        )

        probe = np.array(PROBE_FEATURES)
        in_memory = results["model"].predict(results["scaler"].transform(probe))

        bundle = load_bundle(*paths)
        reloaded = bundle.estimator.predict(bundle.scaler.transform(probe))

        assert in_memory.tolist() == reloaded.tolist()
