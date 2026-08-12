"""
Consent-flow API tests (F2).

Covers the ethical acceptance criteria end to end: no prediction without consent, the
disclaimer/policy is retrievable, nothing is stored without opt-in, and a user can delete
their data and have it be genuinely gone.

The shared ``client`` fixture disables the consent gate so unrelated prediction tests need
no token. Here we build ``consent_client``, which keeps the gate on, because consent is
exactly what these tests exercise.
"""

from __future__ import annotations

import pytest

from keystress.app import create_app
from keystress.core.consent import CONSENT_VERSION
from keystress.core.model import ModelRegistry
from keystress.core.storage import Store
from tests.conftest import make_events, page_bundle


@pytest.fixture
def consent_client(registry: ModelRegistry, store: Store):
    """A client for an app with the consent gate ON (rate limiting off for determinism)."""
    app = create_app(registry=registry, store=store, load_model=False)
    app.config.update(TESTING=True, RATELIMIT_ENABLED=False, KEYSTRESS_REQUIRE_CONSENT=True)
    return app.test_client()


def _consent(client, *, analysis: bool = True, donate: bool = False) -> str:
    """Record consent and return the participant id."""
    response = client.post("/api/consent", json={"analysis": analysis, "donate": donate})
    assert response.status_code == 201
    return response.get_json()["participant_id"]


class TestConsentPolicy:
    def test_policy_is_public_and_versioned(self, consent_client) -> None:
        body = consent_client.get("/api/consent/policy").get_json()
        assert body["consent_version"] == CONSENT_VERSION
        assert "timing" in body["summary"].lower()
        assert body["disclaimer"]


class TestRecordingConsent:
    def test_analysis_consent_is_required(self, consent_client) -> None:
        response = consent_client.post("/api/consent", json={"analysis": False})
        assert response.status_code == 400
        assert "consent" in response.get_json()["error"].lower()

    def test_records_consent_and_returns_a_token(self, consent_client) -> None:
        body = consent_client.post(
            "/api/consent", json={"analysis": True, "donate": True}
        ).get_json()
        assert body["participant_id"]
        assert body["donate"] is True
        assert body["consent_version"] == CONSENT_VERSION

    def test_non_object_body_is_rejected(self, consent_client) -> None:
        assert consent_client.post("/api/consent", json="nope").status_code == 400


class TestPredictionConsentGate:
    """No prediction runs without consent (HARD RULE 4)."""

    def test_prediction_without_consent_is_forbidden(self, consent_client) -> None:
        response = consent_client.post(
            "/api/predict", json={"keystroke_events": make_events()}
        )
        assert response.status_code == 403
        assert "consent" in response.get_json()["error"].lower()

    def test_prediction_with_a_bad_token_is_forbidden(self, consent_client) -> None:
        response = consent_client.post(
            "/api/predict",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": "not-a-real-id"},
        )
        assert response.status_code == 403

    def test_prediction_with_consent_succeeds(self, consent_client) -> None:
        pid = _consent(consent_client)
        response = consent_client.post(
            "/api/predict",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        assert response.status_code == 200
        assert response.get_json()["data_source"] == "synthetic"

    def test_token_may_travel_in_the_body(self, consent_client) -> None:
        pid = _consent(consent_client)
        response = consent_client.post(
            "/api/predict", json={"keystroke_events": make_events(), "consent_id": pid}
        )
        assert response.status_code == 200


class TestDonationOptIn:
    """Nothing is stored without an explicit donate opt-in."""

    def test_donation_without_consent_is_forbidden(self, consent_client) -> None:
        response = consent_client.post(
            "/api/donate", json={"keystroke_events": make_events()}
        )
        assert response.status_code == 403

    def test_analysis_only_consent_cannot_donate(self, consent_client) -> None:
        pid = _consent(consent_client, donate=False)
        response = consent_client.post(
            "/api/donate",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        assert response.status_code == 403

    def test_donation_with_opt_in_is_stored(self, consent_client, store) -> None:
        pid = _consent(consent_client, donate=True)
        response = consent_client.post(
            "/api/donate",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        assert response.status_code == 201
        assert store.list_donations(pid), "the donation should have been persisted"


class TestWithdrawingConsent:
    """
    A participant can change their mind without deleting everything (HARD RULE 4).

    Deletion is the nuclear option; withdrawal is the everyday one. Both must work.
    """

    def test_donate_consent_can_be_withdrawn(self, consent_client) -> None:
        pid = _consent(consent_client, donate=True)
        response = consent_client.patch(
            f"/api/consent/{pid}", json={"analysis": True, "donate": False}
        )
        assert response.status_code == 200
        assert response.get_json()["donate"] is False

        # Storage stops immediately; analysis still works.
        assert consent_client.post(
            "/api/donate",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        ).status_code == 403
        assert consent_client.post(
            "/api/predict",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        ).status_code == 200

    def test_withdrawing_analysis_consent_stops_prediction(self, consent_client) -> None:
        pid = _consent(consent_client)
        consent_client.patch(f"/api/consent/{pid}", json={"analysis": False, "donate": False})
        response = consent_client.post(
            "/api/predict",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        assert response.status_code == 403

    def test_withdrawal_does_not_delete_existing_donations(self, consent_client, store) -> None:
        """Withdrawal stops new storage; erasing what exists is a separate, explicit act."""
        pid = _consent(consent_client, donate=True)
        consent_client.post(
            "/api/donate",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        consent_client.patch(f"/api/consent/{pid}", json={"analysis": True, "donate": False})
        assert len(store.list_donations(pid)) == 1

    def test_both_fields_must_be_stated(self, consent_client) -> None:
        """A partial update would make 'omitted' ambiguous between unchanged and withdrawn."""
        pid = _consent(consent_client, donate=True)
        response = consent_client.patch(f"/api/consent/{pid}", json={"donate": False})
        assert response.status_code == 400
        assert "explicitly" in response.get_json()["error"]

    def test_non_object_body_is_rejected(self, consent_client) -> None:
        pid = _consent(consent_client)
        assert consent_client.patch(f"/api/consent/{pid}", json="nope").status_code == 400

    def test_unknown_participant_is_404(self, consent_client) -> None:
        response = consent_client.patch(
            "/api/consent/nope", json={"analysis": True, "donate": True}
        )
        assert response.status_code == 404


class TestViewAndDelete:
    def test_view_returns_stored_data(self, consent_client) -> None:
        pid = _consent(consent_client, donate=True)
        consent_client.post(
            "/api/donate",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        summary = consent_client.get(f"/api/data/{pid}").get_json()
        assert summary["participant_id"] == pid
        assert len(summary["donations"]) == 1

    def test_view_unknown_is_404(self, consent_client) -> None:
        assert consent_client.get("/api/data/nope").status_code == 404

    def test_delete_makes_data_gone(self, consent_client) -> None:
        pid = _consent(consent_client, donate=True)
        consent_client.post(
            "/api/donate",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )

        assert consent_client.delete(f"/api/data/{pid}").status_code == 200
        # It is actually gone: view 404s, and the token no longer authorises a prediction.
        assert consent_client.get(f"/api/data/{pid}").status_code == 404
        forbidden = consent_client.post(
            "/api/predict",
            json={"keystroke_events": make_events()},
            headers={"X-Consent-Id": pid},
        )
        assert forbidden.status_code == 403

    def test_delete_unknown_is_404(self, consent_client) -> None:
        assert consent_client.delete("/api/data/nope").status_code == 404


class TestConsentUI:
    """
    The page carries the gate too (F2 acceptance: "no prediction runs without consent").

    The server is the enforcement point — these tests do not pretend otherwise. What they
    pin is that the page *asks*: a build that shipped the typing box without the consent
    card would technically still be safe, but would put people in front of a tool that
    403s and explains nothing. That is a real regression and it should fail here.
    """

    @pytest.mark.parametrize("element_id", [
        # The gate itself.
        "consent-card", "consent-summary", "consent-disclaimer",
        "consent-analysis", "consent-donate", "consent-btn", "consent-version",
        # Transparency and deletion controls.
        "data-card", "data-status", "donate-toggle", "delete-btn", "data-output",
        # Says whether a session was stored.
        "donation-note",
    ])
    def test_consent_element_ids_present(self, consent_client, element_id: str) -> None:
        body = consent_client.get("/").get_data(as_text=True)
        assert f'id="{element_id}"' in body, f"missing element id: {element_id}"

    @pytest.mark.parametrize("function_name", [
        "grantConsent", "updateConsentButton", "showConsentedView", "showConsentGate",
        "restoreConsent", "changeDonateConsent", "viewMyData", "deleteMyData",
        "donateSession", "consentHeaders",
    ])
    def test_consent_js_functions_present(self, consent_client, function_name: str) -> None:
        assert function_name in page_bundle(consent_client), f"missing JS: {function_name}"

    def test_typing_card_starts_hidden(self, consent_client) -> None:
        """
        The tool is not usable until the gate is passed — the card ships hidden.

        Hiding moved from a `style="display: none"` attribute to the `is-hidden` class
        when F16 made the CSP strict, so this looks for the class. The claim is the same
        one: the typing card is not visible until app.js reveals it.
        """
        body = consent_client.get("/").get_data(as_text=True)
        marker = body.index('id="test-card"')
        opening_tag = body[body.rindex("<section", 0, marker):marker]
        assert "is-hidden" in opening_tag, (
            "the typing card must start hidden so consent precedes use"
        )

    def test_page_sends_the_consent_token(self, consent_client) -> None:
        assert "X-Consent-Id" in page_bundle(consent_client)

    @pytest.mark.parametrize("endpoint", [
        "/api/consent/policy", "/api/consent", "/api/donate", "/api/data/",
    ])
    def test_page_uses_the_consent_endpoints(self, consent_client, endpoint: str) -> None:
        assert endpoint in page_bundle(consent_client)

    def test_page_offers_deletion(self, consent_client) -> None:
        """Acceptance: a user can delete their data — from the UI, not only the API."""
        bundle = page_bundle(consent_client)
        assert "method: 'DELETE'" in bundle
        assert "Delete everything" in bundle

    def test_donation_is_opt_in_not_default(self, consent_client) -> None:
        """Neither consent checkbox may ship pre-ticked; consent must be an action."""
        body = consent_client.get("/").get_data(as_text=True)
        for element_id in ("consent-analysis", "consent-donate"):
            marker = body.index(f'id="{element_id}"')
            assert "checked" not in body[marker:marker + 100], (
                f"{element_id} must not be pre-ticked"
            )

    def test_page_still_records_only_timing(self, consent_client) -> None:
        """
        The consent flow must not have introduced a content-bearing field.

        Same forbidden list as the F10 characterization test: these are *reads* of typed
        content. ``typingArea.value = ''`` is absent from it deliberately — clearing the
        box is a write, and writes cannot leak anything.
        """
        bundle = page_bundle(consent_client)
        for forbidden in ("event.target.value", "typingArea.value.length", "clipboardData",
                          "event.code", "event.charCode"):
            assert forbidden not in bundle, f"frontend references {forbidden}"

    def test_donated_payload_carries_only_timing_fields(self, consent_client) -> None:
        """The donation request body is built from the same two-field event shape."""
        bundle = page_bundle(consent_client)
        donate_call = bundle[bundle.index("function donateSession"):]
        donate_call = donate_call[:donate_call.index("\n}")]
        assert "timestamp: k.timestamp" in donate_call
        assert "is_backspace: k.is_backspace" in donate_call
