"""
API-layer tests: validation, error handling, and the app factory (F12).

Complements `test_characterization.py`, which pins the contract. These cover the boundary
conditions and failure modes: bad payloads, missing models, and the configuration surface.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from keystress.api.predict import (
    MAX_KEYSTROKE_EVENTS,
    MIN_KEYSTROKE_EVENTS,
    validate_payload,
)
from keystress.app import create_app
from keystress.config import LOOPBACK_HOSTS, Settings, load_settings
from keystress.core.model import ModelRegistry
from tests.conftest import make_events


class TestPayloadValidation:
    """Validation is pure and unit-testable, independent of Flask."""

    def test_accepts_a_well_formed_payload(self) -> None:
        events, message = validate_payload({"keystroke_events": make_events()})
        assert events is not None
        assert message == ""

    @pytest.mark.parametrize("payload,fragment", [
        ("not-a-dict", "JSON object"),
        (None, "JSON object"),
        ({}, "No keystroke data"),
        ({"keystroke_events": None}, "No keystroke data"),
        ({"keystroke_events": {}}, "must be a list"),
        ({"keystroke_events": "abc"}, "must be a list"),
        ({"keystroke_events": []}, "Insufficient"),
    ])
    def test_rejects_bad_payloads_with_a_useful_message(self, payload, fragment: str) -> None:
        events, message = validate_payload(payload)
        assert events is None
        assert fragment.lower() in message.lower()

    def test_rejects_a_payload_just_below_the_minimum(self) -> None:
        events = make_events(count=MIN_KEYSTROKE_EVENTS - 1)
        assert validate_payload({"keystroke_events": events})[0] is None

    def test_accepts_a_payload_at_the_minimum(self) -> None:
        events = make_events(count=MIN_KEYSTROKE_EVENTS)
        assert validate_payload({"keystroke_events": events})[0] is not None

    def test_rejects_an_oversized_payload_without_processing_it(self) -> None:
        """The cap exists so a huge body is refused cheaply, not parsed then refused."""
        events = [{"timestamp": 0.0}] * (MAX_KEYSTROKE_EVENTS + 1)
        result, message = validate_payload({"keystroke_events": events})
        assert result is None
        assert "Too many" in message


class TestPredictEndpoint:
    """HTTP behaviour of `/api/predict`."""

    def test_rejects_a_non_json_body(self, client) -> None:
        response = client.post("/api/predict", data="not json",
                               content_type="application/json")
        assert response.status_code == 400

    def test_rejects_get(self, client) -> None:
        assert client.get("/api/predict").status_code == 405

    def test_error_bodies_have_an_error_key_and_no_prediction(self, client) -> None:
        body = client.post("/api/predict", json={}).get_json()
        assert "error" in body
        assert "prediction" not in body, "an error must not carry a partial result"

    def test_unknown_event_fields_do_not_break_scoring(self, client) -> None:
        events = [
            {"timestamp": i * 0.2, "is_backspace": False, "unexpected_field": i}
            for i in range(30)
        ]
        assert client.post("/api/predict", json={"keystroke_events": events}).status_code == 200

    def test_missing_model_returns_503_with_actionable_message(self, empty_client) -> None:
        response = empty_client.post(
            "/api/predict", json={"keystroke_events": make_events()}
        )
        assert response.status_code == 503
        assert "error" in response.get_json()

    def test_a_long_session_scores(self, client) -> None:
        events = make_events(count=5000, interval=0.15)
        assert client.post("/api/predict", json={"keystroke_events": events}).status_code == 200


class TestAppFactory:
    """The factory that replaced import-time global setup."""

    def test_creates_independent_apps(self, registry) -> None:
        first = create_app(registry=registry, load_model=False)
        second = create_app(registry=ModelRegistry(), load_model=False)
        assert first is not second
        assert first.extensions["keystress_registry"] is not \
            second.extensions["keystress_registry"]

    def test_registry_is_injectable(self, registry) -> None:
        """The property that makes the API testable without touching disk."""
        app = create_app(registry=registry, load_model=False)
        assert app.extensions["keystress_registry"] is registry

    def test_load_model_false_leaves_registry_empty(self) -> None:
        app = create_app(registry=ModelRegistry(), load_model=False)
        assert not app.extensions["keystress_registry"].is_loaded

    def test_settings_are_attached(self, registry) -> None:
        settings = Settings(port=1234)
        app = create_app(settings=settings, registry=registry, load_model=False)
        assert app.config["KEYSTRESS_SETTINGS"].port == 1234

    def test_routes_are_registered(self, app) -> None:
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert {"/", "/api/predict", "/api/health", "/readyz"} <= rules

    def test_starting_without_a_model_does_not_crash(self, tmp_path) -> None:
        """
        HARD RULE 6: the app must still start so health can report the degraded state.

        A process that refuses to boot cannot deliver the clear message the rule requires.

        Artifact paths are pointed at an empty temp directory rather than left at their
        defaults: the defaults resolve to the real ``models/`` directory, so this test
        would otherwise pass or fail depending on whether the developer had run training —
        the kind of hidden disk dependency the injectable registry exists to remove.
        """
        settings = Settings(
            auto_train=False,
            model_path=tmp_path / "model.pkl",
            scaler_path=tmp_path / "scaler.pkl",
            metadata_path=tmp_path / "meta.json",
        )
        app = create_app(settings=settings, registry=ModelRegistry(), load_model=True)

        assert app.test_client().get("/api/health").status_code == 200
        assert app.test_client().get("/api/health").get_json()["model_loaded"] is False
        assert app.test_client().get("/readyz").status_code == 503


class TestConfiguration:
    """Local-first defaults (HARD RULE 5)."""

    def test_default_host_is_loopback(self) -> None:
        assert Settings().host == "127.0.0.1"
        assert Settings().is_loopback

    def test_default_debug_is_off(self) -> None:
        assert Settings().debug is False

    def test_non_loopback_host_is_flagged(self) -> None:
        assert not Settings(host="0.0.0.0").is_loopback

    @pytest.mark.parametrize("host", sorted(LOOPBACK_HOSTS))
    def test_loopback_hosts_recognised(self, host: str) -> None:
        assert Settings(host=host).is_loopback

    def test_environment_overrides_are_read(self, monkeypatch) -> None:
        monkeypatch.setenv("KEYSTRESS_HOST", "192.168.1.5")
        monkeypatch.setenv("KEYSTRESS_PORT", "8080")
        monkeypatch.setenv("KEYSTRESS_DEBUG", "true")

        settings = load_settings()
        assert settings.host == "192.168.1.5"
        assert settings.port == 8080
        assert settings.debug is True
        assert not settings.is_loopback

    def test_defaults_apply_when_environment_is_empty(self, monkeypatch) -> None:
        for name in ("KEYSTRESS_HOST", "KEYSTRESS_PORT", "KEYSTRESS_DEBUG", "FLASK_DEBUG"):
            monkeypatch.delenv(name, raising=False)

        settings = load_settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 5000
        assert settings.debug is False

    def test_unparseable_port_falls_back_rather_than_crashing(self, monkeypatch) -> None:
        monkeypatch.setenv("KEYSTRESS_PORT", "not-a-number")
        assert load_settings().port == 5000

    def test_settings_are_immutable(self) -> None:
        with pytest.raises(FrozenInstanceError):
            Settings().host = "0.0.0.0"  # type: ignore[misc]


class TestStaticAssets:
    """The extracted frontend must actually be served (F10)."""

    @pytest.mark.parametrize("path,content_type", [
        ("/static/styles.css", "text/css"),
        ("/static/app.js", "javascript"),
    ])
    def test_assets_serve_with_the_right_type(self, client, path: str, content_type: str) -> None:
        response = client.get(path)
        assert response.status_code == 200
        assert content_type in response.headers["Content-Type"]
        assert len(response.data) > 100

    def test_unknown_asset_is_404(self, client) -> None:
        assert client.get("/static/nope.css").status_code == 404


class TestEnsureModel:
    """Startup model resolution, including the auto-train fallback."""

    def test_auto_train_builds_a_model_when_none_exists(self, tmp_path) -> None:
        """First run with no artifacts must produce a working, labelled model."""
        from keystress.app import ensure_model

        settings = Settings(
            auto_train=True,
            data_path=tmp_path / "data.csv",
            model_path=tmp_path / "model.pkl",
            scaler_path=tmp_path / "scaler.pkl",
            metadata_path=tmp_path / "meta.json",
        )
        registry = ModelRegistry()
        ensure_model(registry, settings)

        assert registry.is_loaded
        assert registry.get().data_source == "synthetic"
        assert settings.model_path.exists()

    def test_auto_train_disabled_leaves_registry_empty(self, tmp_path) -> None:
        from keystress.app import ensure_model

        settings = Settings(
            auto_train=False,
            model_path=tmp_path / "model.pkl",
            scaler_path=tmp_path / "scaler.pkl",
            metadata_path=tmp_path / "meta.json",
        )
        registry = ModelRegistry()
        ensure_model(registry, settings)

        assert not registry.is_loaded, "must not train when auto-training is disabled"

    def test_existing_model_is_loaded_without_retraining(self, tmp_path) -> None:
        from keystress.app import ensure_model
        from keystress.ml.synthetic import generate_synthetic_typing_data, save_synthetic_data
        from keystress.ml.train import train_and_evaluate

        data_path = save_synthetic_data(
            generate_synthetic_typing_data(n_samples=300, random_state=3),
            tmp_path / "data.csv",
        )
        paths = (tmp_path / "model.pkl", tmp_path / "scaler.pkl", tmp_path / "meta.json")
        train_and_evaluate(data_path=data_path, model_path=paths[0],
                           scaler_path=paths[1], metadata_path=paths[2], random_state=3)
        mtime = paths[0].stat().st_mtime_ns

        settings = Settings(auto_train=True, data_path=data_path, model_path=paths[0],
                            scaler_path=paths[1], metadata_path=paths[2])
        registry = ModelRegistry()
        ensure_model(registry, settings)

        assert registry.is_loaded
        assert paths[0].stat().st_mtime_ns == mtime, "existing model was needlessly retrained"

    def test_configure_logging_is_tolerant_of_a_bad_level(self) -> None:
        from keystress.app import configure_logging

        configure_logging("NOT_A_LEVEL")  # must not raise
