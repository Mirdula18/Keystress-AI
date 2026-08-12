"""
Tests for the labelling instrument (F4).

The instrument is *content*, not logic, so most of what can go wrong here is a quiet
editing mistake: an item that drifts into the wrong subscale, a reverse flag lost in a
re-word, a version that stops being bumped. Each of those silently corrupts every label
collected afterwards, and none of them would fail anything else in the suite.
"""

from __future__ import annotations

import pytest

from keystress.research import instrument as inst


class TestStructure:
    """The shape of the questionnaire."""

    def test_two_subscales_and_thirteen_items(self) -> None:
        assert inst.SUBSCALES == ("personal", "studies")
        assert len(inst.ITEMS) == 13

    def test_subscale_sizes_match_the_adaptation(self) -> None:
        # 6 personal + 7 studies-related; the 6 client-related items are dropped (D-020).
        assert len(inst.items_for("personal")) == 6
        assert len(inst.items_for("studies")) == 7

    def test_item_ids_are_unique(self) -> None:
        ids = [item.id for item in inst.ITEMS]
        assert len(ids) == len(set(ids))

    def test_every_item_belongs_to_a_known_subscale(self) -> None:
        for item in inst.ITEMS:
            assert item.subscale in inst.SUBSCALES

    def test_unknown_subscale_is_rejected(self) -> None:
        # Returning an empty tuple would produce a mean over nothing further downstream.
        with pytest.raises(ValueError, match="Unknown subscale"):
            inst.items_for("client")

    def test_required_ids_cover_every_item(self) -> None:
        assert {item.id for item in inst.ITEMS} == inst.REQUIRED_ITEM_IDS


class TestScales:
    """Both response scales must land on the same anchors."""

    @pytest.mark.parametrize("scale", [inst.FREQUENCY_SCALE, inst.DEGREE_SCALE])
    def test_scale_uses_the_cbi_anchors(self, scale: inst.ResponseScale) -> None:
        assert scale.values == {0, 25, 50, 75, 100}

    @pytest.mark.parametrize("scale", [inst.FREQUENCY_SCALE, inst.DEGREE_SCALE])
    def test_options_run_from_most_to_least(self, scale: inst.ResponseScale) -> None:
        values = [value for _, value in scale.options]
        assert values == sorted(values, reverse=True)

    def test_the_two_scales_are_interchangeable_numerically(self) -> None:
        """
        A subscale mixes both scales, so they must share anchors — otherwise the mean
        would weight items by which wording they happened to use.
        """
        assert inst.FREQUENCY_SCALE.values == inst.DEGREE_SCALE.values

    def test_every_item_uses_a_known_scale(self) -> None:
        known = {inst.FREQUENCY_SCALE, inst.DEGREE_SCALE}
        for item in inst.ITEMS:
            assert item.scale in known


class TestReverseScoring:
    """Exactly one item is reversed, and it is the energy item."""

    def test_only_one_item_is_reverse_scored(self) -> None:
        reversed_items = [item for item in inst.ITEMS if item.reverse]
        assert len(reversed_items) == 1

    def test_the_reversed_item_is_the_energy_question(self) -> None:
        item = next(item for item in inst.ITEMS if item.reverse)
        assert "energy" in item.text.lower(), (
            "the reverse flag has drifted onto the wrong item; every stored score after "
            "such a change would be wrong in a way nothing else here would catch"
        )


class TestProvenance:
    """An instrument used without its provenance is being misused."""

    def test_citation_names_the_authors_and_year(self) -> None:
        assert "Kristensen" in inst.CITATION
        assert "2005" in inst.CITATION

    def test_the_adaptation_is_stated_not_hidden(self) -> None:
        note = inst.ADAPTATION_NOTE.lower()
        assert "adaptation" in note
        assert "client-related" in note
        assert "does not automatically transfer" in note

    def test_the_disclaimer_refuses_the_diagnostic_reading(self) -> None:
        disclaimer = inst.INSTRUMENT_DISCLAIMER.lower()
        assert "not a medical assessment" in disclaimer
        assert "cannot diagnose" in disclaimer

    def test_version_is_set(self) -> None:
        assert inst.INSTRUMENT_VERSION


class TestPayload:
    """What the API hands the page."""

    def test_payload_is_json_serialisable(self) -> None:
        import json

        json.dumps(inst.as_payload())  # must not raise

    def test_payload_carries_every_item(self) -> None:
        payload = inst.as_payload()
        assert len(payload["items"]) == len(inst.ITEMS)
        assert {item["id"] for item in payload["items"]} == inst.REQUIRED_ITEM_IDS

    @pytest.mark.parametrize("field", [
        "citation", "licence_note", "adaptation_note", "disclaimer", "instrument_version",
    ])
    def test_provenance_travels_with_the_items(self, field: str) -> None:
        """
        The frontend must not have to remember to show these. A page that renders the
        questions without the citation and the adaptation note is the misuse this design
        is meant to make awkward.
        """
        assert inst.as_payload()[field]

    def test_payload_carries_both_scales_with_their_options(self) -> None:
        scales = inst.as_payload()["scales"]
        assert set(scales) == {"frequency", "degree"}
        for scale in scales.values():
            assert len(scale["options"]) == 5
            assert {option["value"] for option in scale["options"]} == {0, 25, 50, 75, 100}

    def test_payload_contains_no_free_text_field(self) -> None:
        """
        There is no free-text item, by design: it is the one field type that could carry
        content, and the project's entire promise is that content is never collected.
        """
        for item in inst.as_payload()["items"]:
            assert item["scale"] in ("frequency", "degree")
