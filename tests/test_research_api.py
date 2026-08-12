"""
Research API tests (F4): the instrument endpoint and questionnaire submission.

The acceptance criterion this module exists for is "can collect and store consented,
labelled real sessions with no content capture". Each half is tested separately: that a
labelled pair is genuinely produced end to end, and that the paths which must *not* store
anything genuinely store nothing.

Like `test_consent_api.py`, these build a client with the consent gate ON, because consent
is half of what is under test.
"""

from __future__ import annotations

import pytest

from keystress.app import create_app
from keystress.core.disclosure import FEATURES_V1
from keystress.core.model import ModelRegistry
from keystress.core.storage import Store
from keystress.research import instrument as inst
from tests.conftest import make_events


@pytest.fixture
def consent_client(registry: ModelRegistry, store: Store):
    """A client for an app with the consent gate ON (rate limiting off for determinism)."""
    app = create_app(registry=registry, store=store, load_model=False)
    app.config.update(TESTING=True, RATELIMIT_ENABLED=False, KEYSTRESS_REQUIRE_CONSENT=True)
    return app.test_client()


def _consent(client, *, analysis: bool = True, donate: bool = False) -> str:
    response = client.post("/api/consent", json={"analysis": analysis, "donate": donate})
    assert response.status_code == 201
    return response.get_json()["participant_id"]


def _answers(value: int = 50) -> dict[str, int]:
    return {item.id: value for item in inst.ITEMS}


class TestInstrumentEndpoint:
    """The questionnaire is readable before anyone consents to anything."""

    def test_instrument_is_public(self, consent_client) -> None:
        # Reading what you are about to be asked is part of informed consent, not a leak.
        response = consent_client.get("/api/instrument")
        assert response.status_code == 200

    def test_instrument_carries_items_and_scales(self, consent_client) -> None:
        body = consent_client.get("/api/instrument").get_json()
        assert len(body["items"]) == len(inst.ITEMS)
        assert set(body["scales"]) == {"frequency", "degree"}

    @pytest.mark.parametrize("field", ["citation", "adaptation_note", "disclaimer"])
    def test_instrument_carries_its_provenance(self, consent_client, field: str) -> None:
        assert consent_client.get("/api/instrument").get_json()[field]


class TestSubmission:
    """Scoring a submission."""

    def test_a_complete_submission_returns_the_participants_scores(
        self, consent_client
    ) -> None:
        participant = _consent(consent_client)
        response = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers(50)},
            headers={"X-Consent-Id": participant},
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["overall_score"] == 50.0
        assert set(body["subscale_scores"]) == set(inst.SUBSCALES)
        assert body["band"]

    def test_the_response_carries_the_banding_caveat(self, consent_client) -> None:
        participant = _consent(consent_client)
        body = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers(100)},
            headers={"X-Consent-Id": participant},
        ).get_json()

        assert "not a diagnostic threshold" in body["caveat"]

    def test_no_typing_verdict_is_returned_alongside_the_score(self, consent_client) -> None:
        """
        D-019: the visitor sees their questionnaire score, never the typing model's guess.

        The model is synthetic-trained and unvalidated, so showing its output beside a
        real self-report invites precisely the inference the project refuses to support —
        that the two measured the same thing.
        """
        participant = _consent(consent_client)
        body = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers(75)},
            headers={"X-Consent-Id": participant},
        ).get_json()

        for forbidden in ("prediction", "probabilities", "confidence", "level_class"):
            assert forbidden not in body, f"questionnaire result leaks a model field: {forbidden}"

    @pytest.mark.parametrize(("payload", "fragment"), [
        ({"responses": {}}, "incomplete"),
        ({"responses": {"p1": 50}}, "incomplete"),
        ({"responses": "everything"}, "must be an object"),
        ({}, "must be an object"),
    ])
    def test_malformed_submissions_are_rejected(
        self, consent_client, payload: dict, fragment: str
    ) -> None:
        participant = _consent(consent_client)
        response = consent_client.post(
            "/api/questionnaire", json=payload, headers={"X-Consent-Id": participant}
        )
        assert response.status_code == 400
        assert fragment in response.get_json()["error"].lower()

    def test_an_off_scale_value_is_rejected(self, consent_client) -> None:
        participant = _consent(consent_client)
        answers = _answers(50)
        answers["p1"] = 42  # not a scale anchor

        response = consent_client.post(
            "/api/questionnaire",
            json={"responses": answers},
            headers={"X-Consent-Id": participant},
        )
        assert response.status_code == 400


class TestConsentGate:
    """A questionnaire is processing someone's data, so it needs consent too."""

    def test_without_a_token_it_refuses(self, consent_client) -> None:
        response = consent_client.post("/api/questionnaire", json={"responses": _answers()})
        assert response.status_code == 403

    def test_with_a_withdrawn_token_it_refuses(self, consent_client) -> None:
        participant = _consent(consent_client)
        consent_client.patch(
            f"/api/consent/{participant}", json={"analysis": False, "donate": False}
        )

        response = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers()},
            headers={"X-Consent-Id": participant},
        )
        assert response.status_code == 403


class TestStorageIsOptIn:
    """Answers are only kept with the separate donate opt-in."""

    def test_analysis_only_participant_gets_a_score_but_stores_nothing(
        self, consent_client, store: Store
    ) -> None:
        participant = _consent(consent_client, donate=False)
        body = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers()},
            headers={"X-Consent-Id": participant},
        ).get_json()

        assert body["stored"] is False
        assert "not stored" in body["storage_note"]
        assert store.list_responses(participant) == []

    def test_a_donor_has_their_answers_kept(self, consent_client, store: Store) -> None:
        participant = _consent(consent_client, donate=True)
        body = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers()},
            headers={"X-Consent-Id": participant},
        ).get_json()

        assert body["stored"] is True
        assert len(store.list_responses(participant)) == 1

    def test_withdrawing_donation_stops_further_storage(
        self, consent_client, store: Store
    ) -> None:
        participant = _consent(consent_client, donate=True)
        headers = {"X-Consent-Id": participant}
        consent_client.post("/api/questionnaire", json={"responses": _answers()},
                            headers=headers)
        consent_client.patch(
            f"/api/consent/{participant}", json={"analysis": True, "donate": False}
        )
        body = consent_client.post(
            "/api/questionnaire", json={"responses": _answers(75)}, headers=headers
        ).get_json()

        assert body["stored"] is False
        assert len(store.list_responses(participant)) == 1, "the earlier answer stays"


class TestLabelledPair:
    """The whole point: a real typing session with a real label."""

    def _donate(self, client, participant: str) -> int:
        response = client.post(
            "/api/donate",
            json={"keystroke_events": make_events(count=40)},
            headers={"X-Consent-Id": participant},
        )
        assert response.status_code == 201
        return response.get_json()["donation_id"]

    def test_a_full_journey_produces_one_labelled_record(
        self, consent_client, store: Store
    ) -> None:
        participant = _consent(consent_client, donate=True)
        donation_id = self._donate(consent_client, participant)

        response = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers(100), "donation_id": donation_id},
            headers={"X-Consent-Id": participant},
        )
        assert response.status_code == 200

        records = store.labelled_records()
        assert len(records) == 1
        assert records[0]["participant_id"] == participant
        assert records[0]["label"] == 2
        assert set(records[0]["features"]) == set(FEATURES_V1)

    def test_a_donation_id_from_another_participant_is_refused(
        self, consent_client, store: Store
    ) -> None:
        mine = _consent(consent_client, donate=True)
        theirs = _consent(consent_client, donate=True)
        their_donation = self._donate(consent_client, theirs)

        response = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers(), "donation_id": their_donation},
            headers={"X-Consent-Id": mine},
        )

        assert response.status_code == 400
        assert store.list_responses(mine) == [], (
            "a confused client must not have its answers stored against the wrong session"
        )

    def test_a_non_integer_donation_id_is_rejected(self, consent_client) -> None:
        participant = _consent(consent_client, donate=True)
        response = consent_client.post(
            "/api/questionnaire",
            json={"responses": _answers(), "donation_id": "12"},
            headers={"X-Consent-Id": participant},
        )
        assert response.status_code == 400


class TestTransparency:
    """Stored answers must be visible and deletable like everything else."""

    def test_responses_appear_in_view_my_data(self, consent_client) -> None:
        participant = _consent(consent_client, donate=True)
        consent_client.post("/api/questionnaire", json={"responses": _answers()},
                            headers={"X-Consent-Id": participant})

        body = consent_client.get(f"/api/data/{participant}").get_json()
        assert len(body["responses"]) == 1
        assert body["responses"][0]["instrument_version"] == inst.INSTRUMENT_VERSION

    def test_deletion_removes_responses_too(self, consent_client, store: Store) -> None:
        participant = _consent(consent_client, donate=True)
        consent_client.post("/api/questionnaire", json={"responses": _answers()},
                            headers={"X-Consent-Id": participant})

        assert consent_client.delete(f"/api/data/{participant}").status_code == 200
        assert store.list_responses(participant) == []
