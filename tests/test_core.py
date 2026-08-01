"""
Unit tests for the core domain: collection, features, model loading, inference (F12).

These cover the paths the API tests exercise only indirectly, and the edge cases that
produced silent wrong answers in the inherited code — zero-duration sessions, single-event
sessions, missing artifacts, and corrupt metadata.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from keystress.core.collect import (
    TypingDataCollector,
    TypingSession,
    empty_session_metadata,
    process_keystroke_data,
)
from keystress.core.disclosure import (
    DATA_SOURCE_QUALIFIERS,
    DISCLAIMER,
    FEATURE_SET_VERSION,
    FEATURES_V1,
    SHIPPED_DATA_SOURCE,
    format_metric,
    format_percentage,
    qualifier_for,
)
from keystress.core.features import (
    batch_extract_features,
    extract_typing_features,
    features_to_dataframe,
    get_feature_summary,
    normalize_features,
)
from keystress.core.inference import (
    BURNOUT_LABELS,
    disclosure_fields,
    feature_vector,
    get_prediction_details,
    has_sufficient_signal,
    predict_burnout,
)
from keystress.core.model import (
    ModelBundle,
    ModelRegistry,
    ModelUnavailableError,
    default_metadata,
    load_bundle,
    read_metadata,
)
from tests.conftest import make_events

# --------------------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------------------


class TestProcessKeystrokeData:
    """The privacy-boundary function's functional behaviour."""

    def test_empty_input_returns_zeroed_metadata(self) -> None:
        assert process_keystroke_data([]) == empty_session_metadata()

    def test_counts_and_duration(self) -> None:
        session = process_keystroke_data(make_events(count=10, interval=0.5))
        assert session["total_keys"] == 10
        assert session["duration"] == pytest.approx(4.5)
        assert len(session["inter_key_delays"]) == 9

    def test_backspaces_counted(self) -> None:
        session = process_keystroke_data(make_events(count=12, backspace_every=3))
        assert session["backspace_count"] == 4

    def test_single_event_has_zero_duration(self) -> None:
        session = process_keystroke_data([{"timestamp": 5.0}])
        assert session["total_keys"] == 1
        assert session["duration"] == 0.0
        assert session["inter_key_delays"] == []

    def test_is_backspace_defaults_to_false(self) -> None:
        session = process_keystroke_data([{"timestamp": 0.0}, {"timestamp": 1.0}])
        assert session["backspace_count"] == 0

    def test_inter_key_delays_are_differences(self) -> None:
        events = [{"timestamp": t} for t in (0.0, 0.5, 1.5, 4.0)]
        assert process_keystroke_data(events)["inter_key_delays"] == pytest.approx(
            [0.5, 1.0, 2.5]
        )

    @pytest.mark.parametrize("bad_event", [
        {"is_backspace": False},
        {"timestamp": "not-a-number"},
        {"timestamp": None},
        {"timestamp": True},
    ])
    def test_malformed_events_raise_value_error(self, bad_event) -> None:
        """Malformed input fails loudly rather than being coerced into a plausible number."""
        with pytest.raises(ValueError):
            process_keystroke_data([bad_event, {"timestamp": 1.0}])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_timestamps_raise_value_error(self, value: float) -> None:
        """
        NaN/Inf pass the numeric isinstance check but must still be rejected.

        A non-finite timestamp would turn every aggregate (duration, delays, features)
        into NaN/Inf that silently reaches a model - the exact class of fake result
        HARD RULE 6 forbids.
        """
        with pytest.raises(ValueError):
            process_keystroke_data([{"timestamp": value}, {"timestamp": 1.0}])

    def test_non_mapping_event_raises(self) -> None:
        with pytest.raises(ValueError):
            process_keystroke_data(["not-a-dict"])

    def test_out_of_order_timestamps_produce_negative_delays(self) -> None:
        """
        Documents current behaviour: ordering is not enforced.

        Real clients send monotonic `performance.now()` values. Rejecting or sorting
        out-of-order input is an input-quality decision that belongs with F7's quality
        gate, not a silent fix here.
        """
        session = process_keystroke_data([{"timestamp": t} for t in (0.0, 2.0, 1.0)])
        assert min(session["inter_key_delays"]) < 0


class TestTypingSession:
    """The local-capture dataclass."""

    def test_records_and_computes(self) -> None:
        session = TypingSession()
        for i in range(5):
            session.record_keypress(is_backspace=(i == 2), timestamp=float(i))

        assert session.get_total_keys() == 5
        assert session.get_backspace_count() == 1
        assert session.get_duration() == pytest.approx(4.0)
        assert session.get_inter_key_delays() == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_empty_session_duration_is_zero(self) -> None:
        assert TypingSession().get_duration() == 0.0

    def test_reset_clears_everything(self) -> None:
        session = TypingSession()
        session.record_keypress(timestamp=1.0)
        session.reset()
        assert session.get_total_keys() == 0
        assert session.start_time is None

    def test_collector_round_trip(self) -> None:
        collector = TypingDataCollector()
        collector.start_session()
        for i in range(4):
            collector.session.record_keypress(timestamp=float(i))

        data = collector.get_session_data()
        assert data["total_keys"] == 4
        assert set(data) == {
            "total_keys", "backspace_count", "duration",
            "inter_key_delays", "start_time", "end_time",
        }

    def test_collector_end_session_stamps_the_end_time(self) -> None:
        collector = TypingDataCollector()
        collector.start_session()
        collector.record_key(is_backspace=False)
        session = collector.end_session()

        assert session.end_time is not None
        assert session.end_time >= session.start_time

    def test_record_keypress_rejects_a_non_finite_timestamp(self) -> None:
        session = TypingSession()
        with pytest.raises(ValueError):
            session.record_keypress(timestamp=float("nan"))
        with pytest.raises(ValueError):
            session.record_keypress(timestamp=float("inf"))
        assert session.get_total_keys() == 0, "a rejected keypress must not be recorded"


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------


class TestExtractTypingFeatures:
    """Feature extraction, including the edge cases that used to produce fake results."""

    def test_returns_the_v1_feature_set(self) -> None:
        session = process_keystroke_data(make_events())
        assert set(extract_typing_features(session)) == set(FEATURES_V1)

    def test_speed_is_keys_per_second(self) -> None:
        session = process_keystroke_data(make_events(count=11, interval=1.0))
        # 11 keys spanning 10 seconds.
        assert extract_typing_features(session)["avg_typing_speed"] == pytest.approx(1.1)

    def test_backspace_ratio(self) -> None:
        session = process_keystroke_data(make_events(count=10, backspace_every=2))
        assert extract_typing_features(session)["backspace_ratio"] == pytest.approx(0.5)

    def test_max_pause_is_the_largest_gap(self) -> None:
        events = [{"timestamp": t} for t in (0.0, 0.1, 0.2, 3.7, 3.8)]
        features = extract_typing_features(process_keystroke_data(events))
        assert features["max_pause_duration"] == pytest.approx(3.5)

    def test_uniform_typing_has_zero_variability(self) -> None:
        session = process_keystroke_data(make_events(count=20, interval=0.25))
        assert extract_typing_features(session)["typing_consistency"] == pytest.approx(0.0)

    def test_zero_duration_returns_all_zeros(self) -> None:
        session = process_keystroke_data([{"timestamp": 1.0}] * 5)
        assert extract_typing_features(session) == dict.fromkeys(FEATURES_V1, 0.0)

    def test_empty_session_returns_all_zeros(self) -> None:
        assert extract_typing_features(empty_session_metadata()) == dict.fromkeys(
            FEATURES_V1, 0.0
        )

    def test_missing_keys_do_not_raise(self) -> None:
        assert extract_typing_features({}) == dict.fromkeys(FEATURES_V1, 0.0)

    def test_keys_without_delays_still_yield_speed(self) -> None:
        """
        A session with keys and a duration but no inter-key delay list.

        Only reachable through a hand-crafted record, but it must not crash and must not
        fabricate delay-derived values.
        """
        features = extract_typing_features({
            "total_keys": 3, "backspace_count": 0, "duration": 1.0,
            "inter_key_delays": [],
        })
        assert features["avg_typing_speed"] == pytest.approx(3.0)
        assert features["avg_inter_key_delay"] == 0.0
        assert features["max_pause_duration"] == 0.0
        assert features["typing_consistency"] == 0.0


class TestFeatureHelpers:
    """Supporting transforms."""

    def test_features_to_dataframe_column_order(self) -> None:
        df = features_to_dataframe(dict.fromkeys(FEATURES_V1, 1.0))
        assert list(df.columns) == list(FEATURES_V1)
        assert len(df) == 1

    def test_batch_extract(self) -> None:
        sessions = [process_keystroke_data(make_events(count=n)) for n in (10, 20, 30)]
        assert len(batch_extract_features(sessions)) == 3

    def test_batch_extract_empty(self) -> None:
        assert batch_extract_features([]).empty

    def test_normalize_scales_to_unit_range(self) -> None:
        df = features_to_dataframe(dict.fromkeys(FEATURES_V1, 1.0))
        df.loc[1] = dict.fromkeys(FEATURES_V1, 3.0)
        normalized = normalize_features(df)
        assert normalized["avg_typing_speed"].tolist() == [0.0, 1.0]

    def test_normalize_constant_column_is_zero_not_nan(self) -> None:
        df = features_to_dataframe(dict.fromkeys(FEATURES_V1, 2.0))
        assert normalize_features(df)["avg_typing_speed"].tolist() == [0.0]

    def test_normalize_with_explicit_columns_only_scales_those(self) -> None:
        df = features_to_dataframe(dict.fromkeys(FEATURES_V1, 1.0))
        df.loc[1] = dict.fromkeys(FEATURES_V1, 1.0)
        df["extra_column"] = [10.0, 20.0]

        normalized = normalize_features(df, feature_columns=["avg_typing_speed"])
        assert normalized["avg_typing_speed"].tolist() == [0.0, 0.0]
        assert normalized["extra_column"].tolist() == [10.0, 20.0], (
            "columns not listed must be left untouched"
        )

    def test_feature_summary(self) -> None:
        sessions = [process_keystroke_data(make_events(count=n)) for n in (10, 20, 30)]
        summary = get_feature_summary(batch_extract_features(sessions))
        assert set(summary) == set(FEATURES_V1)
        assert set(summary["avg_typing_speed"]) == {"mean", "std", "min", "max"}

    def test_feature_summary_skips_missing_columns(self) -> None:
        """A dataframe without every v1 feature still summarises the ones it has."""
        df = features_to_dataframe(dict.fromkeys(FEATURES_V1, 1.0)).drop(
            columns=["backspace_ratio"]
        )
        summary = get_feature_summary(df)
        assert "backspace_ratio" not in summary
        assert set(summary) == set(FEATURES_V1) - {"backspace_ratio"}


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------


class TestModelBundle:
    """The immutable bundle."""

    def test_exposes_metadata(self, model_bundle: ModelBundle) -> None:
        assert model_bundle.model_version == "rf-v1-synthetic-test"
        assert model_bundle.data_source == "synthetic"
        assert model_bundle.feature_set == "v1"

    def test_describe_names_the_data_source(self, model_bundle: ModelBundle) -> None:
        assert "synthetic" in model_bundle.describe()

    def test_is_immutable(self, model_bundle: ModelBundle) -> None:
        """Frozen so a half-swapped bundle is unrepresentable."""
        with pytest.raises(FrozenInstanceError):
            model_bundle.estimator = None  # type: ignore[misc]

    def test_missing_metadata_reports_unknown_not_a_flattering_default(self) -> None:
        bundle = ModelBundle(estimator=None, scaler=None, metadata={})
        assert bundle.model_version == "unknown"

    def test_missing_source_and_feature_set_get_the_shipped_defaults(self) -> None:
        """Provenance and feature-set defaults must still say something honest."""
        bundle = ModelBundle(estimator=None, scaler=None, metadata={"model_version": "x"})
        assert bundle.data_source == SHIPPED_DATA_SOURCE
        assert bundle.feature_set == FEATURE_SET_VERSION


class TestModelRegistry:
    """Lifecycle of the loader that replaced the globals."""

    def test_starts_empty(self) -> None:
        registry = ModelRegistry()
        assert not registry.is_loaded
        assert registry.get_or_none() is None

    def test_get_raises_when_empty(self) -> None:
        with pytest.raises(ModelUnavailableError):
            ModelRegistry().get()

    def test_set_then_get(self, model_bundle: ModelBundle) -> None:
        registry = ModelRegistry()
        registry.set(model_bundle)
        assert registry.is_loaded
        assert registry.get() is model_bundle

    def test_set_replaces_atomically(self, model_bundle: ModelBundle) -> None:
        registry = ModelRegistry()
        registry.set(model_bundle)
        replacement = ModelBundle(
            estimator=model_bundle.estimator,
            scaler=model_bundle.scaler,
            metadata={"model_version": "second", "data_source": "synthetic"},
        )
        registry.set(replacement)
        assert registry.get().model_version == "second"

    def test_clear(self, model_bundle: ModelBundle) -> None:
        registry = ModelRegistry()
        registry.set(model_bundle)
        registry.clear()
        assert not registry.is_loaded

    def test_load_missing_artifacts_raises_actionable_error(self, tmp_path) -> None:
        with pytest.raises(ModelUnavailableError) as exc:
            ModelRegistry().load(
                tmp_path / "nope.pkl", tmp_path / "nope2.pkl", tmp_path / "meta.json"
            )
        assert "keystress-train" in str(exc.value), "error should say how to fix it"

    def test_failed_load_leaves_previous_model_in_place(self, model_bundle, tmp_path) -> None:
        """A bad reload must not tear down a working model."""
        registry = ModelRegistry()
        registry.set(model_bundle)

        with pytest.raises(ModelUnavailableError):
            registry.load(tmp_path / "a.pkl", tmp_path / "b.pkl", tmp_path / "c.json")

        assert registry.get() is model_bundle

    def test_corrupt_artifact_raises_clean_error(self, tmp_path) -> None:
        model = tmp_path / "model.pkl"
        scaler = tmp_path / "scaler.pkl"
        model.write_bytes(b"this is not a pickle")
        scaler.write_bytes(b"neither is this")

        with pytest.raises(ModelUnavailableError):
            load_bundle(model, scaler, tmp_path / "meta.json")


class TestReadMetadata:
    """Metadata reading degrades honestly."""

    def test_missing_file_returns_unknown(self, tmp_path) -> None:
        assert read_metadata(tmp_path / "absent.json")["model_version"] == "unknown"

    def test_corrupt_json_returns_unknown(self, tmp_path) -> None:
        path = tmp_path / "meta.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert read_metadata(path)["model_version"] == "unknown"

    def test_non_object_json_returns_unknown(self, tmp_path) -> None:
        path = tmp_path / "meta.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_metadata(path)["model_version"] == "unknown"

    def test_unreadable_metadata_path_returns_unknown(self, tmp_path) -> None:
        """An OSError while reading (here: a directory where a file was expected)."""
        path = tmp_path / "meta.json"
        path.mkdir()
        assert read_metadata(path)["model_version"] == "unknown"

    def test_valid_metadata_is_read(self, tmp_path) -> None:
        path = tmp_path / "meta.json"
        path.write_text(json.dumps({
            "model_version": "rf-v1-synthetic-g2-s42-n1500",
            "data_source": "synthetic",
            "feature_set": "v1",
        }), encoding="utf-8")
        assert read_metadata(path)["model_version"] == "rf-v1-synthetic-g2-s42-n1500"

    def test_partial_metadata_gets_safe_defaults(self, tmp_path) -> None:
        path = tmp_path / "meta.json"
        path.write_text(json.dumps({"model_version": "x"}), encoding="utf-8")
        metadata = read_metadata(path)
        assert metadata["data_source"] == "synthetic"
        assert metadata["feature_set"] == "v1"

    def test_default_metadata_declares_a_source(self) -> None:
        assert default_metadata()["data_source"] in DATA_SOURCE_QUALIFIERS


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


class TestInference:
    """Scoring and abstention."""

    def test_sufficient_signal_detection(self) -> None:
        assert not has_sufficient_signal(dict.fromkeys(FEATURES_V1, 0.0))
        assert has_sufficient_signal({**dict.fromkeys(FEATURES_V1, 0.0),
                                      "avg_typing_speed": 3.0})

    def test_predict_returns_label_and_confidence(self, model_bundle) -> None:
        features = extract_typing_features(process_keystroke_data(make_events()))
        level, label, description, confidence = predict_burnout(features, model_bundle)

        assert level in (0, 1, 2)
        assert label == BURNOUT_LABELS[level]
        assert description
        assert 0.0 <= confidence <= 1.0  # metrics-ok: bounds check, not a published metric

    def test_predict_abstains_on_zero_features(self, model_bundle) -> None:
        level, label, _, confidence = predict_burnout(
            dict.fromkeys(FEATURES_V1, 0.0), model_bundle
        )
        assert level is None
        assert confidence is None
        assert label == "Insufficient data"

    def test_details_carry_all_disclosure_fields(self, model_bundle) -> None:
        features = extract_typing_features(process_keystroke_data(make_events()))
        result = get_prediction_details(features, model_bundle)
        for key in ("data_source", "model_version", "feature_set", "disclaimer",
                    "insufficient_data"):
            assert key in result

    def test_abstention_response_still_carries_disclosure(self, model_bundle) -> None:
        """Even a non-result must say what model produced it and disclaim itself."""
        result = get_prediction_details(dict.fromkeys(FEATURES_V1, 0.0), model_bundle)
        assert result["insufficient_data"] is True
        assert result["prediction"] is None
        assert result["data_source"] == "synthetic"
        assert result["disclaimer"] == DISCLAIMER

    def test_labels_are_indicator_framed_not_diagnostic(self) -> None:
        """HARD RULE 2: labels describe an indicator, never assert a condition."""
        for label in BURNOUT_LABELS.values():
            assert "indicator" in label.lower()
            assert "burnout" not in label.lower()

    def test_probabilities_sum_to_one(self, model_bundle) -> None:
        features = extract_typing_features(process_keystroke_data(make_events()))
        result = get_prediction_details(features, model_bundle)
        assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-6)

    def test_feature_vector_shape_and_column_order(self) -> None:
        vector = feature_vector({"avg_typing_speed": 1.0, "backspace_ratio": 0.5})
        assert vector.shape == (1, len(FEATURES_V1))
        assert vector[0, FEATURES_V1.index("avg_typing_speed")] == 1.0
        assert vector[0, FEATURES_V1.index("backspace_ratio")] == 0.5
        assert vector[0, FEATURES_V1.index("max_pause_duration")] == 0.0, (
            "a missing feature must default to 0.0, never raise"
        )

    def test_disclosure_fields_without_a_bundle(self) -> None:
        """With no model loaded, disclosure still says what produced nothing."""
        fields = disclosure_fields(None)
        assert fields["data_source"] == SHIPPED_DATA_SOURCE
        assert fields["model_version"] == "unknown"
        assert fields["feature_set"] == FEATURE_SET_VERSION
        assert fields["disclaimer"] == DISCLAIMER

    def test_predict_falls_back_for_an_out_of_range_class(self) -> None:
        """A model returning a class index the label table does not know must not crash."""

        class _FixedEstimator:
            def predict(self, _x):  # type: ignore[no-untyped-def]
                return np.array([7])

            def predict_proba(self, _x):  # type: ignore[no-untyped-def]
                return np.array([[0.2, 0.3, 0.5]])

        class _IdentityScaler:
            def transform(self, _x):  # type: ignore[no-untyped-def]
                return _x

        bundle = ModelBundle(
            estimator=_FixedEstimator(), scaler=_IdentityScaler(),
            metadata={"model_version": "x", "data_source": "synthetic"},
        )
        features = extract_typing_features(process_keystroke_data(make_events()))
        level, label, description, confidence = predict_burnout(features, bundle)

        assert level == 7
        assert label == "Unknown"
        assert description == ""
        expected_confidence = 0.5
        assert confidence == pytest.approx(expected_confidence)


# --------------------------------------------------------------------------------------
# Disclosure helpers
# --------------------------------------------------------------------------------------


class TestDisclosure:
    """The formatting helpers that make unqualified metrics hard to produce."""

    def test_qualifier_for_known_sources(self) -> None:
        assert qualifier_for("synthetic") == "on synthetic data"
        assert qualifier_for("real") == "on real validated data"

    def test_unknown_source_raises_rather_than_defaulting(self) -> None:
        """An unrecognised source must fail loudly, not silently emit an unqualified metric."""
        with pytest.raises(ValueError):
            qualifier_for("made-up")

    def test_format_metric_includes_the_source(self) -> None:
        rendered = format_metric("Accuracy", 0.9013, "synthetic")
        assert "0.9013" in rendered
        assert "on synthetic data" in rendered

    def test_format_percentage_includes_the_source(self) -> None:
        rendered = format_percentage("Confidence", 0.62, "synthetic")
        assert "62%" in rendered
        assert "on synthetic data" in rendered

    def test_disclaimer_is_non_diagnostic(self) -> None:
        lowered = DISCLAIMER.lower()
        assert "not a medical" in lowered
        assert "diagnostic" in lowered


class TestFormatPredictionOutput:
    """CLI rendering — another surface where an unqualified metric could escape (F1)."""

    @pytest.fixture
    def result(self, model_bundle):
        features = extract_typing_features(process_keystroke_data(make_events()))
        return get_prediction_details(features, model_bundle)

    def test_every_percentage_is_qualified(self, result) -> None:
        from keystress.core.inference import format_prediction_output

        rendered = format_prediction_output(result)
        for line in rendered.splitlines():
            if "%" in line and "Correction rate" not in line:
                assert "synthetic" in line.lower(), f"unqualified metric line: {line!r}"

    def test_output_names_the_model_and_source(self, result) -> None:
        from keystress.core.inference import format_prediction_output

        rendered = format_prediction_output(result)
        assert result["model_version"] in rendered
        assert "synthetic" in rendered.lower()

    def test_output_is_non_diagnostic(self, result) -> None:
        from keystress.core.inference import format_prediction_output

        rendered = format_prediction_output(result).lower()
        assert "not a diagnosis" in rendered
        assert "not a medical" in rendered

    def test_output_carries_the_synthetic_caveat(self, result) -> None:
        from keystress.core.inference import format_prediction_output

        assert "hand" in format_prediction_output(result).lower()

    def test_real_source_output_omits_the_synthetic_caveat(self, result) -> None:
        from keystress.core.inference import format_prediction_output

        result["data_source"] = "real"
        result["model_version"] = "rf-v1-real-s42-n1500"
        rendered = format_prediction_output(result).lower()

        assert "on real validated data" in rendered
        assert "hand" not in rendered, "the synthetic caveat must not follow a real source"
        assert "synthetic" not in rendered

    def test_abstention_output_reports_no_score(self, model_bundle) -> None:
        from keystress.core.inference import format_prediction_output

        result = get_prediction_details(dict.fromkeys(FEATURES_V1, 0.0), model_bundle)
        rendered = format_prediction_output(result)

        assert "Insufficient data" in rendered
        assert "%" not in rendered, "an abstention must not display a percentage"
