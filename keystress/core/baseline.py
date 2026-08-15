"""
Personal typing baselines (F6): change from *your own* normal.

The population model asks "does this session look like the sessions labelled high?" That
question is dominated by between-person variation: people type at wildly different speeds
for reasons — keyboard, language, age, practice, injury — that have nothing to do with
wellbeing. A fast typist having their worst week may still out-type a slow typist at their
best, so a population comparison mostly measures *who someone is*.

This module asks the better question: **does this session look like your own other
sessions?** Within-person change is the plausible signal, and it is also the more private
one — a personal baseline is meaningless to anyone but its owner, and cannot be used to
rank people against each other.

What this deliberately does not do
----------------------------------
It does not produce a burnout score. A deviation is a *description* — "you typed more
slowly than your own normal, with more corrections" — and that is all it is allowed to be
until the F4/F5 pipeline shows a deviation means anything about wellbeing. Turning "0.8
standard deviations slower" into "elevated burnout risk" would invent exactly the
validated relationship this project does not have (HARD RULE 2/3).

Robust statistics, because the samples are tiny
-----------------------------------------------
A baseline may rest on five sessions. With samples that small, one bad night — an
interrupted session, a new keyboard — moves a mean substantially and inflates a standard
deviation enough to hide everything afterwards. So the centre is the **median** and the
spread is the **median absolute deviation**, scaled to be comparable with a standard
deviation for normally-distributed data. Both are unmoved by a single outlier, which is
the failure mode that actually occurs here.

Privacy
-------
A baseline is computed from the participant's own donated feature rows and nothing else.
It requires the donate opt-in, because it requires history, and history only exists if
they chose to store it. Deleting their data deletes their baseline with it — there is no
separate cache to forget.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .disclosure import FEATURES_V1

logger = logging.getLogger(__name__)

#: Sessions required before any deviation is reported.
#:
#: Five is a judgement, not a derivation: below it the median and MAD are estimated from
#: so little that "unusual for you" is indistinguishable from "your second Tuesday". It is
#: deliberately a named constant rather than a literal, because the honest value is an
#: empirical question the F4 dataset can eventually answer.
MIN_BASELINE_SESSIONS: Final[int] = 5

#: Sessions beyond which older ones are dropped. A baseline should track a person as they
#: change — a new keyboard, a new term — rather than average over their entire history.
BASELINE_WINDOW: Final[int] = 30

#: Scale factor turning a median absolute deviation into a standard-deviation-comparable
#: number for normally-distributed data (1 / 0.6745).
MAD_TO_SIGMA: Final[float] = 1.4826

#: Below this spread, a feature is treated as having no usable variation: dividing by it
#: would turn rounding noise into a huge deviation.
MIN_SPREAD: Final[float] = 1e-9

#: |z| above which a feature is called out as unusual for this person. Two robust
#: deviations is uncommon without being rare, which suits a reflective prompt rather than
#: an alarm.
NOTABLE_DEVIATION: Final[float] = 2.0

#: How each feature reads when it moves. Used to describe a deviation in words, since
#: "avg_inter_key_delay is +2.3" tells a participant nothing.
FEATURE_PHRASING: Final[dict[str, tuple[str, str]]] = {
    "avg_typing_speed": ("faster than your usual pace", "slower than your usual pace"),
    "avg_inter_key_delay": ("longer gaps between keys than usual",
                            "shorter gaps between keys than usual"),
    "max_pause_duration": ("a longer pause than you usually take",
                           "no long pauses, unlike your usual sessions"),
    "backspace_ratio": ("more corrections than usual", "fewer corrections than usual"),
    "typing_consistency": ("a more uneven rhythm than usual",
                           "a steadier rhythm than usual"),
}


@dataclass(frozen=True)
class PersonalBaseline:
    """
    A participant's own typing norm.

    Attributes:
        n_sessions: Sessions the baseline was computed from.
        centre: Median value per feature.
        spread: Scaled median absolute deviation per feature.
        window: Maximum sessions considered.
    """

    n_sessions: int
    centre: dict[str, float]
    spread: dict[str, float]
    window: int = BASELINE_WINDOW

    @property
    def is_usable(self) -> bool:
        """Whether there is enough history to compare a session against."""
        return self.n_sessions >= MIN_BASELINE_SESSIONS

    def as_payload(self) -> dict[str, Any]:
        """Render for the API, without the raw per-feature numbers a caller cannot use."""
        return {
            "n_sessions": self.n_sessions,
            "sessions_needed": max(0, MIN_BASELINE_SESSIONS - self.n_sessions),
            "usable": self.is_usable,
            "window": self.window,
        }


def build_baseline(
    sessions: Sequence[dict[str, Any]],
    *,
    window: int = BASELINE_WINDOW,
) -> PersonalBaseline:
    """
    Build a baseline from a participant's stored feature rows.

    Parameters:
        sessions: Feature dictionaries, newest first (the order
            :meth:`keystress.core.storage.Store.list_donations` returns).
        window: Maximum number of recent sessions to use.

    Returns:
        PersonalBaseline: The participant's norm. A baseline built from too few sessions
        is still returned — with ``is_usable`` false — because "you have 2 of 5 sessions"
        is more useful to a person than an error.
    """
    recent = list(sessions)[:window]

    if not recent:
        return PersonalBaseline(n_sessions=0, centre={}, spread={}, window=window)

    centre: dict[str, float] = {}
    spread: dict[str, float] = {}
    for name in FEATURES_V1:
        values = np.array([float(session.get(name, 0.0)) for session in recent])
        median = float(np.median(values))
        centre[name] = median
        # MAD, not standard deviation: with five sessions one interrupted night would
        # otherwise both shift the centre and inflate the spread enough to mask
        # everything after it.
        spread[name] = float(np.median(np.abs(values - median)) * MAD_TO_SIGMA)

    return PersonalBaseline(
        n_sessions=len(recent), centre=centre, spread=spread, window=window
    )


def deviation_from(
    baseline: PersonalBaseline,
    features: dict[str, float],
) -> dict[str, float | None]:
    """
    Express a session as robust z-scores against a personal baseline.

    Parameters:
        baseline: The participant's norm.
        features: This session's features.

    Returns:
        dict: Per feature, how many robust deviations this session sits from that
        person's median. ``None`` where the participant's own history shows no variation
        in that feature — dividing by a spread of zero would turn rounding noise into a
        dramatic finding, and "no variation to compare against" is the honest answer.

    Raises:
        ValueError: If the baseline is not usable. Callers must check
            :attr:`PersonalBaseline.is_usable` and show the cold-start state instead;
            producing a number here would defeat the point of having the threshold.
    """
    if not baseline.is_usable:
        raise ValueError(
            f"Baseline rests on {baseline.n_sessions} session(s); "
            f"{MIN_BASELINE_SESSIONS} are needed before a deviation means anything."
        )

    deviations: dict[str, float | None] = {}
    for name in FEATURES_V1:
        spread = baseline.spread.get(name, 0.0)
        if spread < MIN_SPREAD:
            deviations[name] = None
            continue
        deviations[name] = float((features.get(name, 0.0) - baseline.centre[name]) / spread)
    return deviations


def describe_deviation(deviations: dict[str, float | None]) -> list[str]:
    """
    Put the notable deviations into plain language.

    Parameters:
        deviations: A :func:`deviation_from` result.

    Returns:
        list[str]: One phrase per feature that moved notably, newest wording first by
        magnitude. Empty when nothing stood out — which is itself worth saying, and the
        caller does.
    """
    notable = [
        (name, value) for name, value in deviations.items()
        if value is not None and abs(value) >= NOTABLE_DEVIATION
    ]
    notable.sort(key=lambda item: abs(item[1]), reverse=True)

    phrases: list[str] = []
    for name, value in notable:
        higher, lower = FEATURE_PHRASING.get(name, (f"a higher {name}", f"a lower {name}"))
        phrases.append(higher if value > 0 else lower)
    return phrases


def personal_summary(
    baseline: PersonalBaseline,
    features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Build the response block describing a session against a personal baseline.

    This is the whole feature's honesty surface, so it is worth being explicit about what
    the block does and does not contain. It carries deviations and a description of them.
    It carries no burnout level, no risk score, and no probability, because a deviation
    has not been shown to mean anything about wellbeing — F4 collects the data that could
    show it and F5 is the test. Until then the truthful reading of "you typed more slowly
    than usual" is "you typed more slowly than usual".

    Parameters:
        baseline: The participant's norm.
        features: This session's features, when there is a session to compare.

    Returns:
        dict: ``available``, the baseline summary, and — when usable — ``deviations``,
        ``notable``, and a plain-language ``summary``.
    """
    block: dict[str, Any] = {
        "available": baseline.is_usable,
        "baseline": baseline.as_payload(),
        # Repeated in every state so a client that renders only this block still declines
        # to draw a conclusion the data does not support.
        "interpretation_note": (
            "This compares the session with your own previous sessions. It describes what "
            "changed in your typing, not how you are - no link between these changes and "
            "burnout has been established."
        ),
    }

    if not baseline.is_usable:
        needed = MIN_BASELINE_SESSIONS - baseline.n_sessions
        block["summary"] = (
            f"Not enough history yet: {baseline.n_sessions} of {MIN_BASELINE_SESSIONS} "
            f"sessions stored. {needed} more and this will show how a session compares "
            "with your own normal."
        )
        return block

    if features is None:
        block["summary"] = (
            f"Your baseline is built from {baseline.n_sessions} of your own sessions."
        )
        return block

    deviations = deviation_from(baseline, features)
    notable = describe_deviation(deviations)

    block["deviations"] = deviations
    block["notable"] = notable
    block["summary"] = (
        "Compared with your own recent sessions, this one showed " + ", ".join(notable) + "."
        if notable else
        "This session looked much like your own recent sessions."
    )
    return block
