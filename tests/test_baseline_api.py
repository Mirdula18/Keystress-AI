"""
Personal baseline through the API and the store (F6).

`test_baseline.py` owns the statistics. This module owns the wiring, where the mistakes
would be about *whose* data is used and *when* a baseline may appear at all:

- a baseline is built from the requesting participant's rows and nobody else's,
- a session is never compared against a baseline it is itself part of,
- and a participant who has deliberately stored nothing is not shown a cold-start
  progress meter, which would read as a nudge to opt in.
"""

from __future__ import annotations

import pytest

from keystress.app import create_app
from keystress.core.baseline import MIN_BASELINE_SESSIONS
from keystress.core.disclosure import FEATURES_V1
from keystress.core.model import ModelRegistry
from keystress.core.storage import Store
from tests.conftest import make_events


@pytest.fixture
def consent_client(registry: ModelRegistry, store: Store):
    """A client with the consent gate ON, since a baseline needs a token."""
    app = create_app(registry=registry, store=store, load_model=False)
    app.config.update(TESTING=True, RATELIMIT_ENABLED=False, KEYSTRESS_REQUIRE_CONSENT=True)
    return app.test_client()


def _consent(client, *, donate: bool = True) -> str:
    response = client.post("/api/consent", json={"analysis": True, "donate": donate})
    assert response.status_code == 201
    return response.get_json()["participant_id"]


def _features(speed: float) -> dict[str, float]:
    row = dict.fromkeys(FEATURES_V1, 0.2)
    row["avg_typing_speed"] = speed
    return row


def _seed_history(store: Store, participant: str, n: int, speed: float = 4.0) -> None:
    """Give a participant `n` donated sessions around a stable pace."""
    for index in range(n):
        store.save_donation(participant, _features(speed + (index % 3 - 1) * 0.2))


def _predict(client, participant: str):
    return client.post(
        "/api/predict",
        json={"keystroke_events": make_events(count=40)},
        headers={"X-Consent-Id": participant},
    )


class TestFeatureHistory:
    """The store side."""

    def test_history_is_newest_first(self, store: Store) -> None:
        participant = store.create_participant(analysis=True, donate=True)["participant_id"]
        for speed in (1.0, 2.0, 3.0):
            store.save_donation(participant, _features(speed))

        speeds = [row["avg_typing_speed"] for row in store.feature_history(participant)]
        assert speeds == [3.0, 2.0, 1.0]

    def test_history_is_limited_to_the_window(self, store: Store) -> None:
        participant = store.create_participant(analysis=True, donate=True)["participant_id"]
        _seed_history(store, participant, 10)
        assert len(store.feature_history(participant, limit=4)) == 4

    def test_one_participant_never_sees_another(self, store: Store) -> None:
        """
        A personal baseline built from someone else's typing would be worse than none:
        it would confidently describe a person against a stranger's norm.
        """
        mine = store.create_participant(analysis=True, donate=True)["participant_id"]
        theirs = store.create_participant(analysis=True, donate=True)["participant_id"]
        _seed_history(store, theirs, 8, speed=40.0)

        assert store.feature_history(mine) == []

    def test_an_unknown_participant_has_no_history(self, store: Store) -> None:
        assert store.feature_history("nobody") == []

    def test_deleting_a_participant_removes_their_baseline_history(self, store: Store) -> None:
        # There is no separate cache to forget: the baseline is derived from rows that go.
        participant = store.create_participant(analysis=True, donate=True)["participant_id"]
        _seed_history(store, participant, 6)
        store.delete_participant(participant)

        assert store.feature_history(participant) == []


class TestPredictResponse:
    """When the block appears, and what it says."""

    def test_a_donor_with_no_history_gets_the_cold_start_state(self, consent_client) -> None:
        participant = _consent(consent_client)
        body = _predict(consent_client, participant).get_json()

        assert body["personal"]["available"] is False
        assert f"0 of {MIN_BASELINE_SESSIONS}" in body["personal"]["summary"]

    def test_a_donor_with_enough_history_gets_a_comparison(
        self, consent_client, store: Store
    ) -> None:
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS + 2)

        body = _predict(consent_client, participant).get_json()
        assert body["personal"]["available"] is True
        assert "deviations" in body["personal"]

    def test_no_block_without_a_token(self, client) -> None:
        """No participant, no personal history, nothing to compare — omit it entirely."""
        body = client.post(
            "/api/predict", json={"keystroke_events": make_events(count=40)}
        ).get_json()
        assert "personal" not in body

    def test_no_block_for_an_analysis_only_participant(self, consent_client) -> None:
        """
        Someone who deliberately stores nothing must not be shown a progress meter
        counting sessions they have not agreed to store. That reads as a nudge to opt in,
        on a page whose entire design is that opting in is optional.
        """
        participant = _consent(consent_client, donate=False)
        body = _predict(consent_client, participant).get_json()
        assert "personal" not in body

    def test_the_current_session_is_not_part_of_its_own_baseline(
        self, consent_client, store: Store
    ) -> None:
        """
        The session is only stored when the client calls /api/donate afterwards, so
        history read during a prediction holds previous sessions only. Comparing a
        session with a baseline containing it would drag the norm toward it and shrink
        every deviation.
        """
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS)

        _predict(consent_client, participant)

        assert len(store.feature_history(participant)) == MIN_BASELINE_SESSIONS

    def test_the_block_carries_its_interpretation_note(self, consent_client, store: Store) -> None:
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS)

        note = _predict(consent_client, participant).get_json()["personal"]["interpretation_note"]
        assert "no link between these changes and burnout" in note

    def test_the_disclosure_fields_still_travel(self, consent_client, store: Store) -> None:
        # F1's contract is additive: a personal block does not displace data_source.
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS)

        body = _predict(consent_client, participant).get_json()
        assert body["data_source"] == "synthetic"
        assert body["disclaimer"]

    def test_withdrawing_donation_withdraws_the_baseline(
        self, consent_client, store: Store
    ) -> None:
        """
        A baseline is built from stored history, so withdrawing storage consent stops it
        being used — the earlier rows stay (withdrawal is not deletion, D-022), but they
        are no longer read to describe the person.
        """
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS)
        consent_client.patch(
            f"/api/consent/{participant}", json={"analysis": True, "donate": False}
        )

        body = _predict(consent_client, participant).get_json()
        assert "personal" not in body
        assert len(store.list_donations(participant)) == MIN_BASELINE_SESSIONS


class TestNoWellbeingClaim:
    """The block describes typing, never the person (HARD RULE 2)."""

    def test_the_block_contains_no_risk_or_diagnostic_language(
        self, consent_client, store: Store
    ) -> None:
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS + 3)

        block = str(_predict(consent_client, participant).get_json()["personal"]).lower()
        for forbidden in ("burnout risk", "risk score", "diagnos", "elevated risk"):
            assert forbidden not in block

    def test_the_model_verdict_and_the_personal_block_stay_separate_fields(
        self, consent_client, store: Store
    ) -> None:
        """
        They answer different questions — "does this look like the synthetic high class"
        and "does this look like your own other sessions" — and merging them would let
        one borrow the other's authority.
        """
        participant = _consent(consent_client)
        _seed_history(store, participant, MIN_BASELINE_SESSIONS)

        body = _predict(consent_client, participant).get_json()
        assert "prediction" in body
        assert "prediction" not in body["personal"]
        assert "confidence" not in body["personal"]


class TestPageRendering:
    """The results card shows the comparison, and its caveat with it."""

    @pytest.mark.parametrize("element_id", [
        "personal-note", "personal-summary", "personal-caveat",
    ])
    def test_the_elements_exist(self, client, element_id: str) -> None:
        assert f'id="{element_id}"' in client.get("/").get_data(as_text=True)

    def test_the_note_starts_hidden(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        marker = body.index('id="personal-note"')
        assert "is-hidden" in body[body.rindex("<div", 0, marker):marker]

    def test_the_script_renders_summary_and_caveat_together(self) -> None:
        """
        The caveat is what stops a comparison being read as a verdict, so it is rendered
        in the same function as the summary rather than left to a separate call that
        could be forgotten.
        """
        from pathlib import Path

        script = Path("keystress/web/static/app.js").read_text(encoding="utf-8")
        block = script[script.index("function showPersonalNote("):]
        block = block[:block.index("\n}")]

        assert "personal.summary" in block
        assert "personal.interpretation_note" in block

    def test_an_absent_block_hides_the_note(self) -> None:
        from pathlib import Path

        script = Path("keystress/web/static/app.js").read_text(encoding="utf-8")
        block = script[script.index("function showPersonalNote("):]
        block = block[:block.index("\n}")]
        assert "if (!personal)" in block

    def test_the_page_renders_no_personal_verdict(self) -> None:
        """The note may show what changed; it may not show a level, score, or risk."""
        from pathlib import Path

        script = Path("keystress/web/static/app.js").read_text(encoding="utf-8")
        block = script[script.index("function showPersonalNote("):]
        block = block[:block.index("\n}")]
        for forbidden in ("level_class", "confidence", "probabilities", "prediction"):
            assert forbidden not in block
