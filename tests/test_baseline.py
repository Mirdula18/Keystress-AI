"""
Personal baseline tests (F6).

Two things need pinning. The statistics: a robust centre and spread that survive the one
bad session a five-session baseline will inevitably contain. And the honesty: a deviation
must stay a description of typing, never become a claim about wellbeing, and the
cold-start state must be a plain "not enough history yet" rather than a confident guess.
"""

from __future__ import annotations

import pytest

from keystress.core.baseline import (
    MIN_BASELINE_SESSIONS,
    NOTABLE_DEVIATION,
    build_baseline,
    describe_deviation,
    deviation_from,
    personal_summary,
)
from keystress.core.disclosure import FEATURES_V1


def session(speed: float = 4.0, **overrides: float) -> dict[str, float]:
    """A feature row, varying only what a test cares about."""
    row = {
        "avg_typing_speed": speed,
        "avg_inter_key_delay": 0.25,
        "max_pause_duration": 1.2,
        "backspace_ratio": 0.08,
        "typing_consistency": 0.15,
    }
    row.update(overrides)
    return row


def steady_history(n: int = 8, speed: float = 4.0) -> list[dict[str, float]]:
    """A participant whose sessions vary slightly around a stable norm."""
    return [session(speed=speed + (index % 3 - 1) * 0.2) for index in range(n)]


class TestBuildingABaseline:
    """The statistics."""

    def test_a_baseline_covers_every_feature(self) -> None:
        baseline = build_baseline(steady_history())
        assert set(baseline.centre) == set(FEATURES_V1)
        assert set(baseline.spread) == set(FEATURES_V1)

    def test_the_centre_is_the_median_of_the_history(self) -> None:
        history = [session(speed=speed) for speed in (2.0, 4.0, 6.0, 8.0, 10.0)]
        assert build_baseline(history).centre["avg_typing_speed"] == 6.0

    def test_one_disastrous_session_does_not_move_the_centre(self) -> None:
        """
        The reason for a median rather than a mean. An interrupted session — a phone
        call mid-typing — is not a rare event in this data, and with five sessions a mean
        would let it redefine the person's "normal".
        """
        normal = [session(speed=4.0) for _ in range(5)]
        with_outlier = [*normal, session(speed=40.0)]

        assert build_baseline(with_outlier).centre["avg_typing_speed"] == pytest.approx(
            build_baseline(normal).centre["avg_typing_speed"], abs=0.5
        )

    def test_one_disastrous_session_does_not_inflate_the_spread(self) -> None:
        """
        And the reason for a MAD rather than a standard deviation: an inflated spread
        would silently mask every genuine change afterwards.
        """
        normal = [session(speed=4.0 + index * 0.1) for index in range(6)]
        with_outlier = [*normal, session(speed=40.0)]

        assert build_baseline(with_outlier).spread["avg_typing_speed"] < 1.0

    def test_only_the_window_of_recent_sessions_is_used(self) -> None:
        """A baseline should track a person as they change, not average their whole past."""
        history = [session(speed=4.0)] * 3 + [session(speed=100.0)] * 20
        baseline = build_baseline(history, window_sessions=3)

        assert baseline.n_sessions == 3
        assert baseline.centre["avg_typing_speed"] == 4.0

    def test_an_empty_history_yields_an_unusable_baseline_not_an_error(self) -> None:
        baseline = build_baseline([])
        assert baseline.n_sessions == 0
        assert not baseline.is_usable

    def test_a_baseline_becomes_usable_at_the_threshold(self) -> None:
        assert not build_baseline(steady_history(MIN_BASELINE_SESSIONS - 1)).is_usable
        assert build_baseline(steady_history(MIN_BASELINE_SESSIONS)).is_usable


class TestDeviation:
    """Comparing a session with the person's own norm."""

    def test_a_typical_session_deviates_little(self) -> None:
        baseline = build_baseline(steady_history())
        deviations = deviation_from(baseline, session(speed=4.0))
        assert abs(deviations["avg_typing_speed"]) < 1.0

    def test_a_much_slower_session_reads_negative(self) -> None:
        baseline = build_baseline(steady_history(speed=4.0))
        deviations = deviation_from(baseline, session(speed=1.0))
        assert deviations["avg_typing_speed"] < -NOTABLE_DEVIATION

    def test_a_much_faster_session_reads_positive(self) -> None:
        baseline = build_baseline(steady_history(speed=4.0))
        deviations = deviation_from(baseline, session(speed=9.0))
        assert deviations["avg_typing_speed"] > NOTABLE_DEVIATION

    def test_a_feature_with_no_personal_variation_is_none_not_infinity(self) -> None:
        """
        Dividing by a spread of zero would turn a rounding difference into a dramatic
        finding. "You have never varied in this, so there is nothing to compare against"
        is the honest answer.
        """
        history = [session(speed=4.0) for _ in range(6)]  # identical every time
        deviations = deviation_from(build_baseline(history), session(speed=4.001))
        assert deviations["avg_typing_speed"] is None

    def test_an_unusable_baseline_refuses_to_produce_a_number(self) -> None:
        # Producing one anyway would defeat the entire point of the threshold.
        baseline = build_baseline(steady_history(2))
        with pytest.raises(ValueError, match="are needed"):
            deviation_from(baseline, session())


class TestDescription:
    """Turning z-scores into something a person can read."""

    def test_only_notable_deviations_are_described(self) -> None:
        phrases = describe_deviation({"avg_typing_speed": 0.4, "backspace_ratio": 3.1})
        assert phrases == ["more corrections than usual"]

    def test_the_direction_is_reflected_in_the_words(self) -> None:
        assert describe_deviation({"avg_typing_speed": -3.0}) == ["slower than your usual pace"]
        assert describe_deviation({"avg_typing_speed": 3.0}) == ["faster than your usual pace"]

    def test_the_largest_change_is_described_first(self) -> None:
        phrases = describe_deviation({"avg_typing_speed": -2.5, "backspace_ratio": 5.0})
        assert phrases[0] == "more corrections than usual"

    def test_unmeasurable_features_are_skipped(self) -> None:
        assert describe_deviation({"avg_typing_speed": None, "backspace_ratio": 2.5}) == [
            "more corrections than usual"
        ]

    def test_no_feature_name_leaks_into_the_wording(self) -> None:
        """`avg_inter_key_delay is +2.3` tells a participant nothing."""
        phrases = describe_deviation(dict.fromkeys(FEATURES_V1, 3.0))
        for phrase in phrases:
            assert "_" not in phrase


class TestPersonalSummary:
    """The block that reaches the participant."""

    def test_cold_start_says_how_many_sessions_remain(self) -> None:
        block = personal_summary(build_baseline(steady_history(2)))
        assert block["available"] is False
        assert "2 of 5" in block["summary"]
        assert block["baseline"]["sessions_needed"] == 3

    def test_a_usable_baseline_describes_the_session(self) -> None:
        block = personal_summary(build_baseline(steady_history()), session(speed=1.0))
        assert block["available"] is True
        assert "slower than your usual pace" in block["summary"]

    def test_an_unremarkable_session_is_said_to_be_unremarkable(self) -> None:
        block = personal_summary(build_baseline(steady_history()), session(speed=4.0))
        assert "much like your own recent sessions" in block["summary"]

    def test_the_interpretation_note_is_present_in_every_state(self) -> None:
        """
        A client that renders only this block must still decline to draw the conclusion
        the data does not support.
        """
        for block in (
            personal_summary(build_baseline([])),
            personal_summary(build_baseline(steady_history())),
            personal_summary(build_baseline(steady_history()), session(speed=1.0)),
        ):
            assert "no link between these changes and burnout" in block["interpretation_note"]

    def test_the_block_carries_no_wellbeing_verdict(self) -> None:
        """
        F6's goal says "score burnout as deviation from the personal norm". This stops
        one step short of that on purpose: a deviation has not been shown to mean
        anything about wellbeing, and F5 is the test that would show it. Until then, the
        truthful reading of "you typed more slowly than usual" is exactly that.
        """
        block = personal_summary(build_baseline(steady_history()), session(speed=1.0))
        serialised = str(block).lower()
        for forbidden in ("burnout risk", "risk score", "probability", "level_class",
                          "elevated", "diagnos"):
            assert forbidden not in serialised

    def test_the_block_is_json_serialisable(self) -> None:
        import json

        json.dumps(personal_summary(build_baseline(steady_history()), session()))
