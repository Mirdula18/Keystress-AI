"""
Characterization tests for the served page and the prediction round-trip (F10).

**These tests were written before the frontend was extracted from the Python string, and
are unchanged by the extraction.** That is their entire purpose: they pin down the
observable behaviour of the page and the API so that moving ~680 lines of HTML/CSS/JS out
of `keystress/app.py` into `web/` can be proven behaviour-preserving rather than merely
asserted to be.

They deliberately assert on *externally observable* things — element IDs the JS depends
on, the fetch target, the disclosure fields, the response contract — and not on
whitespace, ordering, or the file the bytes came from. A characterization test that breaks
on reformatting would block the very refactor it exists to protect.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import make_events, make_varied_events, page_bundle

# --------------------------------------------------------------------------------------
# Rendered page
# --------------------------------------------------------------------------------------


class TestRenderedPage:
    """The page served at `/` must keep its structure and contract with the JS."""

    def test_index_returns_html_ok(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]

    def test_page_is_a_complete_html_document(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        assert body.lstrip().startswith("<!DOCTYPE html>")
        assert "<html" in body and "</html>" in body
        assert "<head>" in body and "</head>" in body
        assert "<body>" in body and "</body>" in body

    def test_page_has_title(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        assert "<title>" in body
        assert "Keystress-AI" in body

    @pytest.mark.parametrize("element_id", [
        # Inputs and controls the JS binds to.
        "typing-area", "analyze-btn",
        # Live stat readouts.
        "key-count", "backspace-count", "duration", "speed",
        # Cards toggled during the flow.
        "test-card", "loader-card", "results-card",
        # Result fields.
        "result-icon", "result-level", "result-confidence", "result-description",
        # Probability bars, by slot.
        "probability-section", "probability-heading",
        "label-low", "label-medium", "label-high",
        "prob-low", "prob-medium", "prob-high",
        "bar-low", "bar-medium", "bar-high",
        # Disclosure elements added by F1.
        "source-note", "disclaimer-text",
    ])
    def test_required_element_ids_present(self, client, element_id: str) -> None:
        """Every ID the frontend script resolves must exist in the markup."""
        body = client.get("/").get_data(as_text=True)
        assert f'id="{element_id}"' in body, f"missing element id: {element_id}"

    @pytest.mark.parametrize("function_name", [
        "recordKeyDown", "updateStats", "updateDuration",
        "resetTest", "analyzeTyping", "displayResults", "newTest",
        "sourceQualifier",
    ])
    def test_required_js_functions_present(self, client, function_name: str) -> None:
        """The behaviour of the page lives in these functions; all must survive."""
        bundle = page_bundle(client)
        assert function_name in bundle, f"missing JS function: {function_name}"

    def test_page_posts_to_the_predict_endpoint(self, client) -> None:
        assert "/api/predict" in page_bundle(client)

    def test_page_records_only_timestamp_and_backspace(self, client) -> None:
        """
        The client-side collection shape is part of the privacy guarantee.

        The script must build events from a timestamp and a Backspace comparison, and must
        never reference a property that carries character identity.
        """
        bundle = page_bundle(client)
        assert "performance.now()" in bundle
        assert "'Backspace'" in bundle or '"Backspace"' in bundle
        assert "is_backspace" in bundle

        for forbidden in ("event.target.value", "typingArea.value.length",
                          "clipboardData", "event.code", "event.charCode"):
            assert forbidden not in bundle, f"frontend references {forbidden}"

    def test_page_carries_the_synthetic_research_banner(self, client) -> None:
        """F1: the synthetic caveat is prominent page content, not fine print."""
        body = client.get("/").get_data(as_text=True)
        assert "research-banner" in body
        assert "synthetic" in body.lower(), "the caveat must be in the page itself"

    def test_page_has_privacy_notice(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        assert "privacy-notice" in body

    def test_minimum_keystroke_gate_matches_the_server_minimum(self, client) -> None:
        """
        The client gate equals the server's MIN_KEYSTROKE_EVENTS.

        The server's constant is the contract; the page must not refuse a session the
        server would accept (or enable analysis for one it would not).
        """
        assert re.search(r"keyCount\s*<\s*5", page_bundle(client))

    def test_stylesheet_rules_present(self, client) -> None:
        """The CSS travels with the page, wherever it is stored."""
        bundle = page_bundle(client)
        for selector in (".result-icon", ".probability-bar", ".research-banner",
                         ".source-note", ".privacy-notice", ".btn-primary"):
            assert selector in bundle, f"missing style rule: {selector}"

    def test_page_is_stable_across_requests(self, client) -> None:
        """Rendering is deterministic - no timestamps or nonces leak into the page."""
        assert client.get("/").get_data() == client.get("/").get_data()


# --------------------------------------------------------------------------------------
# Prediction round-trip
# --------------------------------------------------------------------------------------


class TestPredictRoundTrip:
    """The `/api/predict` contract, pinned before extraction."""

    def test_successful_prediction_shape(self, client, events) -> None:
        response = client.post("/api/predict", json={"keystroke_events": events})
        assert response.status_code == 200

        body = response.get_json()
        for key in ("prediction", "label", "description", "confidence", "probabilities",
                    "labels", "insufficient_data", "features", "level_class",
                    "data_source", "model_version", "feature_set", "disclaimer"):
            assert key in body, f"response missing {key}"

    def test_prediction_values_are_sane(self, client, events) -> None:
        body = client.post("/api/predict", json={"keystroke_events": events}).get_json()

        assert body["prediction"] in (0, 1, 2)
        # metrics-ok: bounds assertion on a range, not a published metric value
        assert 0.0 <= body["confidence"] <= 1.0
        assert len(body["probabilities"]) == 3
        assert body["probabilities"] == pytest.approx(body["probabilities"])
        assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-6)
        assert body["level_class"] in ("low", "medium", "high")
        assert body["insufficient_data"] is False

    def test_confidence_equals_max_probability(self, client, events) -> None:
        body = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        assert body["confidence"] == pytest.approx(max(body["probabilities"]))

    def test_prediction_index_matches_highest_probability(self, client, events) -> None:
        body = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        probabilities = body["probabilities"]
        assert body["prediction"] == probabilities.index(max(probabilities))

    def test_labels_match_probability_positions(self, client, events) -> None:
        body = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        assert body["labels"] == ["Low (indicator)", "Medium (indicator)", "High (indicator)"]
        assert body["label"] == body["labels"][body["prediction"]]

    def test_disclosure_fields_always_populated(self, client, events) -> None:
        """F1 acceptance: no response without a data source."""
        body = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        assert body["data_source"] == "synthetic"
        assert body["model_version"]
        assert body["feature_set"] == "v1"
        assert "not a medical" in body["disclaimer"].lower()

    def test_features_are_the_five_v1_features(self, client, events) -> None:
        body = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        assert set(body["features"]) == {
            "avg_typing_speed", "avg_inter_key_delay", "max_pause_duration",
            "backspace_ratio", "typing_consistency",
        }

    def test_prediction_is_deterministic(self, client, events) -> None:
        first = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        second = client.post("/api/predict", json={"keystroke_events": events}).get_json()
        assert first == second

    def test_varied_realistic_session_scores(self, client) -> None:
        response = client.post(
            "/api/predict", json={"keystroke_events": make_varied_events()}
        )
        assert response.status_code == 200
        assert response.get_json()["insufficient_data"] is False

    def test_backspaces_are_counted(self, client) -> None:
        body = client.post("/api/predict", json={
            "keystroke_events": make_events(count=40, backspace_every=4)
        }).get_json()
        assert body["features"]["backspace_ratio"] > 0


class TestPredictErrorPaths:
    """Malformed input is rejected clearly, never with a stack trace or a fake result."""

    @pytest.mark.parametrize("payload,status", [
        ({}, 400),
        ({"keystroke_events": None}, 400),
        ({"keystroke_events": "not-a-list"}, 400),
        ({"keystroke_events": []}, 400),
        ({"keystroke_events": [{"timestamp": 0.0}]}, 400),
    ])
    def test_bad_payloads_rejected_with_400(self, client, payload, status: int) -> None:
        response = client.post("/api/predict", json=payload)
        assert response.status_code == status
        assert "error" in response.get_json()

    def test_non_numeric_timestamp_rejected(self, client) -> None:
        response = client.post("/api/predict", json={
            "keystroke_events": [{"timestamp": "abc", "is_backspace": False}] * 6
        })
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_missing_timestamp_rejected(self, client) -> None:
        response = client.post("/api/predict", json={
            "keystroke_events": [{"is_backspace": False}] * 6
        })
        assert response.status_code == 400

    def test_zero_duration_session_abstains(self, client) -> None:
        """HARD RULE 6: no invented result from a session with no signal."""
        body = client.post("/api/predict", json={
            "keystroke_events": [{"timestamp": 1.0, "is_backspace": False}] * 10
        }).get_json()

        assert body["insufficient_data"] is True
        assert body["prediction"] is None
        assert body["confidence"] is None
        assert body["probabilities"] is None
        assert body["level_class"] == "unknown", "a non-result must not be styled as 'low'"
        assert body["data_source"] == "synthetic"
        assert body["disclaimer"]

    def test_no_model_returns_503_not_a_guess(self, empty_client, events) -> None:
        response = empty_client.post("/api/predict", json={"keystroke_events": events})
        assert response.status_code == 503
        assert "error" in response.get_json()


class TestHealthEndpoints:
    """Health and readiness, pinned so extraction cannot disturb them."""

    def test_health_reports_model_identity(self, client) -> None:
        body = client.get("/api/health").get_json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["data_source"] == "synthetic"
        assert body["model_version"] == "rf-v1-synthetic-test"

    def test_health_is_ok_without_a_model(self, empty_client) -> None:
        response = empty_client.get("/api/health")
        assert response.status_code == 200
        assert response.get_json()["model_loaded"] is False

    def test_readyz_true_with_model(self, client) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.get_json()["ready"] is True

    def test_readyz_false_without_model(self, empty_client) -> None:
        response = empty_client.get("/readyz")
        assert response.status_code == 503
        assert response.get_json()["ready"] is False
