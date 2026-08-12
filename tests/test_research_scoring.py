"""
Tests for questionnaire scoring (F4).

The arithmetic is simple enough that the risk is not a wrong formula but a wrong
*convention* applied confidently: reverse scoring on the wrong side, an overall score that
averages subscale means instead of items, a band boundary off by one. Each produces
plausible numbers, so the tests below use worked examples where the right answer is known
by construction.
"""

from __future__ import annotations

import pytest

from keystress.research import instrument as inst
from keystress.research.scoring import (
    BAND_LOWER_BOUNDS,
    LABEL_CAVEAT,
    ScoreResult,
    label_for_score,
    score_item,
    score_responses,
)


def responses_all(value: int) -> dict[str, int]:
    """Every item answered with the same anchor."""
    return {item.id: value for item in inst.ITEMS}


class TestScoreItem:
    """One answer at a time."""

    def test_a_normal_item_passes_its_value_through(self) -> None:
        item = inst.ITEMS_BY_ID["p1"]
        assert score_item(item, 75) == 75

    def test_the_reverse_item_is_inverted(self) -> None:
        """"Plenty of energy" is the *absence* of burnout, so 100 must score 0."""
        item = inst.ITEMS_BY_ID["s4"]
        assert item.reverse
        assert score_item(item, 100) == 0
        assert score_item(item, 0) == 100
        assert score_item(item, 50) == 50  # the midpoint is its own reflection

    @pytest.mark.parametrize("bad_value", [10, 99, -25, 125])
    def test_a_value_off_the_scale_is_rejected(self, bad_value: int) -> None:
        # Clamping or rounding would put an uninterpretable number in a research dataset.
        with pytest.raises(ValueError, match="not one of"):
            score_item(inst.ITEMS_BY_ID["p1"], bad_value)

    @pytest.mark.parametrize("bad_value", ["75", 75.0, None, True])
    def test_a_non_integer_value_is_rejected(self, bad_value: object) -> None:
        with pytest.raises(ValueError, match="integer scale value"):
            score_item(inst.ITEMS_BY_ID["p1"], bad_value)


class TestScoreResponses:
    """A whole questionnaire."""

    def test_all_zeros_scores_zero_except_the_reverse_item(self) -> None:
        """
        Answering "never" to everything includes "never enough energy", which *is* a
        burnout signal — so the overall score is not zero. This is the case that catches
        a reverse flag applied backwards.
        """
        result = score_responses(responses_all(0))

        assert result.subscale_scores["personal"] == 0.0
        # Six studies items at 0, one reversed item at 100 → 100/7.
        assert result.subscale_scores["studies"] == pytest.approx(100 / 7)
        assert result.overall_score == pytest.approx(100 / 13)

    def test_all_hundreds_is_not_a_perfect_hundred(self) -> None:
        """The mirror image: "always exhausted" plus "always plenty of energy"."""
        result = score_responses(responses_all(100))

        assert result.subscale_scores["personal"] == 100.0
        assert result.subscale_scores["studies"] == pytest.approx(600 / 7)
        assert result.overall_score == pytest.approx(1200 / 13)

    def test_midpoint_answers_score_the_midpoint(self) -> None:
        """50 reverses to 50, so this one is clean whichever way the flag points."""
        result = score_responses(responses_all(50))
        assert result.overall_score == 50.0
        assert all(score == 50.0 for score in result.subscale_scores.values())

    def test_overall_is_the_item_mean_not_the_mean_of_subscale_means(self) -> None:
        """
        The subscales have 6 and 7 items, so the two conventions differ. Averaging the
        means would silently up-weight every personal item.
        """
        responses = {item.id: (100 if item.subscale == "personal" else 0)
                     for item in inst.ITEMS}
        result = score_responses(responses)

        mean_of_means = sum(result.subscale_scores.values()) / 2
        assert result.overall_score != pytest.approx(mean_of_means)
        assert result.overall_score == pytest.approx((6 * 100 + 100) / 13)

    def test_item_scores_are_kept_for_rescoring(self) -> None:
        """
        Post-reversal item values are stored so a later change to the banding can be
        applied to existing responses instead of invalidating them.
        """
        result = score_responses(responses_all(25))
        assert set(result.item_scores) == inst.REQUIRED_ITEM_IDS
        assert result.item_scores["s4"] == 75  # reversed

    def test_a_missing_item_is_rejected(self) -> None:
        responses = responses_all(50)
        del responses["p3"]
        with pytest.raises(ValueError, match="incomplete.*p3"):
            score_responses(responses)

    def test_an_unknown_item_is_rejected(self) -> None:
        # A client on a different instrument version, or one inventing fields.
        responses = responses_all(50)
        responses["notes"] = 50
        with pytest.raises(ValueError, match="Unknown item"):
            score_responses(responses)

    def test_scoring_is_deterministic(self) -> None:
        responses = responses_all(75)
        assert score_responses(responses) == score_responses(responses)


class TestBanding:
    """The project's own convention, and the one place it is defined."""

    @pytest.mark.parametrize(("score", "expected"), [
        (0.0, 0), (49.9, 0),
        (50.0, 1), (74.9, 1),
        (75.0, 2), (100.0, 2),
    ])
    def test_band_boundaries_are_inclusive_at_the_lower_edge(
        self, score: float, expected: int
    ) -> None:
        assert label_for_score(score) == expected

    @pytest.mark.parametrize("score", [-0.1, 100.1, 1000.0])
    def test_a_score_outside_the_range_is_rejected(self, score: float) -> None:
        with pytest.raises(ValueError, match="outside"):
            label_for_score(score)

    def test_labels_line_up_with_the_models_three_classes(self) -> None:
        """
        The label is a class index, so it can be compared with the model's output in F5
        without a second mapping that could drift.
        """
        assert len(BAND_LOWER_BOUNDS) == 3
        assert {label_for_score(s) for s in (0.0, 60.0, 90.0)} == {0, 1, 2}


class TestPayload:
    """What a participant is shown."""

    def test_payload_carries_the_caveat(self) -> None:
        payload = score_responses(responses_all(100)).as_payload()
        assert payload["caveat"] == LABEL_CAVEAT
        assert "not a diagnostic threshold" in payload["caveat"]

    def test_payload_names_the_subscales_for_display(self) -> None:
        payload = score_responses(responses_all(50)).as_payload()
        assert payload["subscale_labels"]["personal"] == "Personal burnout"
        assert set(payload["subscale_scores"]) == set(inst.SUBSCALES)

    def test_payload_records_the_instrument_version(self) -> None:
        payload = score_responses(responses_all(50)).as_payload()
        assert payload["instrument_version"] == inst.INSTRUMENT_VERSION

    def test_payload_is_json_serialisable(self) -> None:
        import json

        json.dumps(score_responses(responses_all(25)).as_payload())

    def test_scores_are_rounded_for_display_only(self) -> None:
        """The stored value keeps full precision; only the payload rounds."""
        result: ScoreResult = score_responses(responses_all(0))
        assert result.overall_score != round(result.overall_score, 2)
        assert result.as_payload()["overall_score"] == round(result.overall_score, 2)
