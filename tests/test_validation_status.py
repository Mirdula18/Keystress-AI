"""
Validation status tests (F5): derivation, and how it reaches an operator.

The status answers "has this model ever been tested against real people?", which is a
different question from "is a model loaded" — and the one an operator is most likely to
assume the answer to. These tests cover the derivation rules and the two endpoints that
report them.
"""

from __future__ import annotations

import json

import pytest

from keystress.ml.validation import (
    EVALUATED_NO_SKILL,
    NOT_VALIDATED,
    VALIDATED_ON_REAL,
    disclosure_for,
    explain,
    find_report,
    status_for,
)

METADATA = {"model_version": "rf-v1-synthetic-g2-s42-n1500", "data_source": "synthetic"}


def make_report(**overrides) -> dict:
    """A real-data report that beat the baselines, unless overridden."""
    report = {
        "report_id": "rf-v1-synthetic-g2-s42-n1500__real__seed42",
        "model_version": METADATA["model_version"],
        "dataset": {"data_source": "real"},
        "beats_all_baselines": True,
    }
    report.update(overrides)
    return report


class TestDerivation:
    """The status is a function of two facts and nothing else."""

    def test_no_report_means_not_validated(self) -> None:
        assert status_for(METADATA, None) == NOT_VALIDATED

    def test_a_real_report_that_beat_the_baselines_validates(self) -> None:
        assert status_for(METADATA, make_report()) == VALIDATED_ON_REAL

    def test_a_real_report_with_no_skill_gets_its_own_status(self) -> None:
        status = status_for(METADATA, make_report(beats_all_baselines=False))
        assert status == EVALUATED_NO_SKILL
        assert "did NOT beat" in explain(status)

    def test_a_synthetic_report_never_validates(self) -> None:
        """
        However good its numbers. This is the conflation `CLAUDE.md` §1 exists to
        prevent: separability of authored classes is not evidence about people.
        """
        report = make_report(dataset={"data_source": "synthetic"})
        assert status_for(METADATA, report) == NOT_VALIDATED

    def test_a_report_for_a_different_model_is_ignored(self) -> None:
        """
        Model versions are deterministic (D-017), so a mismatch means the artifact
        changed — and a report about a different artifact says nothing about this one.
        """
        report = make_report(model_version="rf-v1-real-s7-n900")
        assert status_for(METADATA, report) == NOT_VALIDATED

    def test_an_unknown_status_explains_conservatively(self) -> None:
        # "We do not know" must read as "no evidence", never as "probably fine".
        assert explain("something-new") == explain(NOT_VALIDATED)


class TestDisclosureBlock:
    """What gets attached wherever a model is described."""

    def test_an_unvalidated_synthetic_model_carries_the_standing_notice(self) -> None:
        block = disclosure_for(METADATA, None)
        assert block["status"] == NOT_VALIDATED
        assert "synthetic" in block["notice"].lower()

    def test_a_validated_model_names_what_it_was_evaluated_on(self) -> None:
        block = disclosure_for(METADATA, make_report())
        assert block["evaluated_on"] == "real"
        assert block["evaluation_report"]

    def test_the_block_always_explains_itself(self) -> None:
        # A caller that renders only this block must still tell the truth unaided.
        assert disclosure_for(METADATA, None)["explanation"]


class TestFindingReports:
    """Reading reports off disk."""

    def test_no_directory_means_no_report(self, tmp_path) -> None:
        assert find_report("anything", tmp_path / "absent") is None

    def test_a_matching_report_is_found(self, tmp_path) -> None:
        path = tmp_path / f"{METADATA['model_version']}__real__seed42.json"
        path.write_text(json.dumps(make_report()), encoding="utf-8")
        assert find_report(METADATA["model_version"], tmp_path)["dataset"]["data_source"] == "real"

    def test_a_report_for_another_model_is_not_returned(self, tmp_path) -> None:
        (tmp_path / "other-model__real__seed42.json").write_text(
            json.dumps(make_report()), encoding="utf-8"
        )
        assert find_report(METADATA["model_version"], tmp_path) is None

    def test_a_corrupt_report_is_treated_as_absent(self, tmp_path) -> None:
        """
        The conservative direction: the alternative is claiming validation on the
        strength of a file nobody can parse.
        """
        (tmp_path / f"{METADATA['model_version']}__real__seed42.json").write_text(
            "{not json", encoding="utf-8"
        )
        assert find_report(METADATA["model_version"], tmp_path) is None


class TestHealthEndpoints:
    """An operator reading a green health check must not have to infer this."""

    def test_health_reports_the_validation_block(self, client) -> None:
        body = client.get("/api/health").get_json()
        assert body["validation"]["status"] == NOT_VALIDATED
        assert body["validation"]["explanation"]

    def test_health_without_a_model_omits_the_block(self, empty_client) -> None:
        body = empty_client.get("/api/health").get_json()
        assert body["model_loaded"] is False
        assert "validation" not in body

    def test_readyz_separates_ready_from_believable(self, client) -> None:
        """
        "Ready to serve" and "fit to be believed" are different claims, reported
        separately rather than collapsed into one green tick.
        """
        body = client.get("/readyz").get_json()
        assert body["ready"] is True
        assert body["validation_status"] == NOT_VALIDATED

    def test_a_stored_report_promotes_the_status(self, client, app, tmp_path) -> None:
        """End to end: run an evaluation, and the health endpoint stops saying untested."""
        bundle = app.extensions["keystress_registry"].get()
        report_dir = tmp_path / "eval"
        report_dir.mkdir()
        (report_dir / f"{bundle.model_version}__real__seed42.json").write_text(
            json.dumps(make_report(model_version=bundle.model_version)), encoding="utf-8"
        )
        app.config["KEYSTRESS_EVAL_REPORT_DIR"] = str(report_dir)

        body = client.get("/api/health").get_json()
        assert body["validation"]["status"] == VALIDATED_ON_REAL

    @pytest.mark.parametrize("endpoint", ["/api/health", "/readyz"])
    def test_the_data_source_is_still_reported(self, client, endpoint: str) -> None:
        # F1's guarantee must survive the addition: knowing a model is unvalidated does
        # not replace knowing what it was trained on.
        assert client.get(endpoint).get_json()["data_source"] == "synthetic"
