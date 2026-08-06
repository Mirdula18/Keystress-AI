"""
Unit tests for the consent/donation store (F2).

These exercise the store directly, independent of Flask: consent records round-trip,
consent can be changed and withdrawn, donation is gated on opt-in, and deletion is real.
"""

from __future__ import annotations

import pytest

from keystress.core.consent import CONSENT_VERSION
from keystress.core.disclosure import FEATURES_V1
from keystress.core.storage import Store


@pytest.fixture
def store(tmp_path) -> Store:
    """A store on a temp database."""
    return Store(tmp_path / "unit.db")


def _features(**overrides: float) -> dict[str, float]:
    """A full v1 feature dict, defaulting to 1.0, with optional overrides."""
    base = dict.fromkeys(FEATURES_V1, 1.0)
    base.update(overrides)
    return base


class TestConsentRecords:
    def test_create_and_read_back(self, store: Store) -> None:
        record = store.create_participant(analysis=True, donate=False)
        assert record["analysis"] is True
        assert record["donate"] is False
        assert record["consent_version"] == CONSENT_VERSION

        fetched = store.get_participant(record["participant_id"])
        assert fetched == record

    def test_unknown_participant_is_none(self, store: Store) -> None:
        assert store.get_participant("does-not-exist") is None

    def test_ids_are_unique(self, store: Store) -> None:
        a = store.create_participant(analysis=True, donate=True)["participant_id"]
        b = store.create_participant(analysis=True, donate=True)["participant_id"]
        assert a != b

    def test_consent_can_be_changed_and_withdrawn(self, store: Store) -> None:
        pid = store.create_participant(analysis=True, donate=True)["participant_id"]
        updated = store.update_consent(pid, analysis=True, donate=False)
        assert updated is not None and updated["donate"] is False
        assert store.has_donate_consent(pid) is False
        assert store.has_analysis_consent(pid) is True

    def test_update_unknown_participant_returns_none(self, store: Store) -> None:
        assert store.update_consent("nope", analysis=True, donate=True) is None

    def test_consent_predicates_reject_missing_ids(self, store: Store) -> None:
        assert store.has_analysis_consent(None) is False
        assert store.has_analysis_consent("") is False
        assert store.has_donate_consent(None) is False


class TestDonations:
    def test_donation_requires_donate_consent(self, store: Store) -> None:
        pid = store.create_participant(analysis=True, donate=False)["participant_id"]
        with pytest.raises(PermissionError):
            store.save_donation(pid, _features())

    def test_donation_is_stored_and_listed(self, store: Store) -> None:
        pid = store.create_participant(analysis=True, donate=True)["participant_id"]
        donation_id = store.save_donation(
            pid, _features(avg_typing_speed=2.5), data_source="synthetic", prediction=1
        )
        assert donation_id > 0

        donations = store.list_donations(pid)
        assert len(donations) == 1
        assert donations[0]["features"]["avg_typing_speed"] == 2.5
        assert donations[0]["data_source"] == "synthetic"
        assert donations[0]["prediction"] == 1

    def test_features_are_coerced_and_whitelisted(self, store: Store) -> None:
        pid = store.create_participant(analysis=True, donate=True)["participant_id"]
        store.save_donation(pid, {"avg_typing_speed": 3, "junk": "content"})
        stored = store.list_donations(pid)[0]["features"]
        assert set(stored) == set(FEATURES_V1)
        assert isinstance(stored["avg_typing_speed"], float)
        assert stored["avg_typing_speed"] == 3.0


class TestTransparencyAndDeletion:
    def test_summary_includes_consent_and_donations(self, store: Store) -> None:
        pid = store.create_participant(analysis=True, donate=True)["participant_id"]
        store.save_donation(pid, _features())
        summary = store.participant_summary(pid)
        assert summary is not None
        assert summary["analysis"] is True
        assert len(summary["donations"]) == 1

    def test_summary_unknown_is_none(self, store: Store) -> None:
        assert store.participant_summary("nope") is None

    def test_delete_removes_participant_and_donations(self, store: Store) -> None:
        pid = store.create_participant(analysis=True, donate=True)["participant_id"]
        store.save_donation(pid, _features())

        assert store.delete_participant(pid) is True
        assert store.get_participant(pid) is None
        # Cascade: the donations must be gone too, not orphaned.
        assert store.list_donations(pid) == []

    def test_delete_unknown_returns_false(self, store: Store) -> None:
        assert store.delete_participant("nope") is False


def test_store_persists_across_connections(tmp_path) -> None:
    """A second Store on the same file sees the first one's data (real persistence)."""
    path = tmp_path / "persist.db"
    pid = Store(path).create_participant(analysis=True, donate=True)["participant_id"]
    assert Store(path).get_participant(pid) is not None
