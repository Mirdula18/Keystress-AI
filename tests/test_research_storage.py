"""
Storage of questionnaire responses (F4).

`tests/test_storage.py` owns the consent and donation tables; this module owns the
`responses` table those two now sit alongside. The properties worth pinning are the ones
that would corrupt a research dataset rather than crash anything:

- a response is gated on the same donate consent a donation is,
- a response can only label *its own* participant's typing session,
- deletion still takes everything, now that "everything" is three tables,
- and the transparency endpoint shows responses too, or it stops being transparent.
"""

from __future__ import annotations

import pytest

from keystress.core.disclosure import FEATURES_V1
from keystress.core.storage import Store
from keystress.research import instrument as inst
from keystress.research.scoring import score_responses


def make_result(value: int = 50):
    """A scored questionnaire with every item answered the same way."""
    return score_responses({item.id: value for item in inst.ITEMS})


def make_features(speed: float = 3.5) -> dict[str, float]:
    """A plausible feature vector."""
    return {name: speed for name in FEATURES_V1}


@pytest.fixture
def donor(store: Store) -> str:
    """A participant who has consented to both analysis and donation."""
    return store.create_participant(analysis=True, donate=True)["participant_id"]


class TestSavingResponses:
    """The write path."""

    def test_a_response_is_stored_and_listed(self, store: Store, donor: str) -> None:
        response_id = store.save_response(donor, make_result(75))
        assert response_id > 0

        responses = store.list_responses(donor)
        assert len(responses) == 1
        # Twelve items score 75; the reverse-scored energy item scores 25.
        assert responses[0]["overall_score"] == pytest.approx((12 * 75 + 25) / 13)

    def test_subscale_scores_are_stored_separately(self, store: Store, donor: str) -> None:
        """
        Both subscales are kept, not just the overall score. Which subscale carries the
        signal is one of the first questions the real data can answer, and an overall
        score alone would throw that away permanently.
        """
        store.save_response(donor, make_result(100))
        stored = store.list_responses(donor)[0]
        assert stored["personal_score"] == 100.0
        assert stored["studies_score"] == pytest.approx(600 / 7)

    def test_item_scores_survive_the_round_trip(self, store: Store, donor: str) -> None:
        store.save_response(donor, make_result(25))
        stored = store.list_responses(donor)[0]
        assert set(stored["item_scores"]) == inst.REQUIRED_ITEM_IDS

    def test_the_instrument_version_is_recorded(self, store: Store, donor: str) -> None:
        # Without this, answers to two different questionnaires would pool silently.
        store.save_response(donor, make_result())
        assert store.list_responses(donor)[0]["instrument_version"] == inst.INSTRUMENT_VERSION

    def test_without_donate_consent_nothing_is_stored(self, store: Store) -> None:
        participant = store.create_participant(analysis=True, donate=False)["participant_id"]
        with pytest.raises(PermissionError):
            store.save_response(participant, make_result())
        assert store.list_responses(participant) == []

    def test_a_withdrawn_participant_can_no_longer_store(
        self, store: Store, donor: str
    ) -> None:
        store.update_consent(donor, analysis=True, donate=False)
        with pytest.raises(PermissionError):
            store.save_response(donor, make_result())


class TestPairing:
    """A response labels one specific typing session."""

    def test_a_response_can_be_paired_with_a_donation(self, store: Store, donor: str) -> None:
        donation_id = store.save_donation(donor, make_features())
        response_id = store.save_response(donor, make_result(), donation_id=donation_id)

        assert response_id > 0
        assert store.list_responses(donor)[0]["donation_id"] == donation_id

    def test_a_response_may_stand_alone(self, store: Store, donor: str) -> None:
        """
        An unpaired response is a valid questionnaire that simply has no typing session
        attached. It is kept — it is still the participant's data, and it still belongs in
        their "view my data" — but it never reaches the labelled dataset.
        """
        store.save_response(donor, make_result())
        assert store.list_responses(donor)[0]["donation_id"] is None
        assert store.labelled_records() == []

    def test_cannot_label_another_participants_session(self, store: Store, donor: str) -> None:
        """
        Pairing one person's typing with another's questionnaire would produce a
        confidently mislabelled training row — worse than no row at all, because nothing
        downstream could tell it was wrong.
        """
        other = store.create_participant(analysis=True, donate=True)["participant_id"]
        their_donation = store.save_donation(other, make_features())

        with pytest.raises(ValueError, match="does not belong"):
            store.save_response(donor, make_result(), donation_id=their_donation)

        assert store.list_responses(donor) == []


class TestLabelledRecords:
    """The join that becomes the dataset."""

    def test_only_paired_rows_are_returned(self, store: Store, donor: str) -> None:
        paired = store.save_donation(donor, make_features(2.0))
        store.save_donation(donor, make_features(9.0))  # no questionnaire
        store.save_response(donor, make_result(100), donation_id=paired)
        store.save_response(donor, make_result(0))      # no typing session

        records = store.labelled_records()
        assert len(records) == 1
        assert records[0]["donation_id"] == paired
        assert records[0]["features"]["avg_typing_speed"] == 2.0

    def test_records_carry_the_grouping_key(self, store: Store, donor: str) -> None:
        """
        F5 splits by participant, so the export must say which rows came from the same
        person. Without it, one participant's sessions land on both sides of a split and
        every metric is optimistic.
        """
        donation_id = store.save_donation(donor, make_features())
        store.save_response(donor, make_result(), donation_id=donation_id)

        assert store.labelled_records()[0]["participant_id"] == donor

    def test_records_are_ordered_oldest_first(self, store: Store, donor: str) -> None:
        for speed in (1.0, 2.0, 3.0):
            donation_id = store.save_donation(donor, make_features(speed))
            store.save_response(donor, make_result(), donation_id=donation_id)

        speeds = [r["features"]["avg_typing_speed"] for r in store.labelled_records()]
        assert speeds == [1.0, 2.0, 3.0], "export order must be stable across runs"

    def test_records_span_participants(self, store: Store, donor: str) -> None:
        other = store.create_participant(analysis=True, donate=True)["participant_id"]
        for participant in (donor, other):
            donation_id = store.save_donation(participant, make_features())
            store.save_response(participant, make_result(), donation_id=donation_id)

        assert {r["participant_id"] for r in store.labelled_records()} == {donor, other}


class TestTransparencyAndDeletion:
    """Responses are the participant's data too."""

    def test_summary_includes_responses(self, store: Store, donor: str) -> None:
        store.save_response(donor, make_result())
        summary = store.participant_summary(donor)
        assert len(summary["responses"]) == 1
        assert "donations" in summary

    def test_deleting_a_participant_removes_their_responses(
        self, store: Store, donor: str
    ) -> None:
        donation_id = store.save_donation(donor, make_features())
        store.save_response(donor, make_result(), donation_id=donation_id)

        assert store.delete_participant(donor) is True

        assert store.list_responses(donor) == []
        assert store.labelled_records() == []
        assert store.participant_summary(donor) is None

    def test_deletion_leaves_other_participants_alone(self, store: Store, donor: str) -> None:
        other = store.create_participant(analysis=True, donate=True)["participant_id"]
        store.save_response(other, make_result())

        store.delete_participant(donor)

        assert len(store.list_responses(other)) == 1


class TestSchemaMigration:
    """An existing database gains the table without losing anything."""

    def test_reopening_an_older_database_adds_the_table(self, tmp_path) -> None:
        path = tmp_path / "existing.db"
        first = Store(path)
        participant = first.create_participant(analysis=True, donate=True)["participant_id"]
        first.save_donation(participant, make_features())

        # Reopening runs the schema script again; it is idempotent by construction.
        second = Store(path)
        second.save_response(participant, make_result())

        assert len(second.list_donations(participant)) == 1
        assert len(second.list_responses(participant)) == 1
