"""
Scoring the questionnaire (F4): raw answers to subscale scores and a dataset label.

Two separate jobs live here, and the distinction matters:

**Scoring** is the instrument's own arithmetic. Each answer is already on the CBI's 0-100
anchor scale, one item is inverted, and a subscale score is the mean of its items. Nothing
about that is this project's invention, so it is implemented exactly as published and
tested against worked examples.

**Labelling** is this project's choice. Turning a continuous score into one of three
classes needs cut-points, and the ones used here (below 50, 50-74, 75+) are the bands
commonly reported in the CBI literature. They are a reporting convention, not a diagnostic
threshold, and :data:`LABEL_CAVEAT` says so wherever a label is shown. The reason the
project needs a class label at all is that the existing model is a three-class classifier
(``ARCHITECTURE.md`` §4.3); F5 compares its predictions against these labels, so they must
be on the same footing.

The banding is deliberately kept in one place. A future decision to model the continuous
score directly - which would be better science, since it discards no information - only
has to change what the dataset carries, not find every threshold scattered through the
code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from .instrument import (
    INSTRUMENT_VERSION,
    ITEMS_BY_ID,
    REQUIRED_ITEM_IDS,
    SUBSCALE_LABELS,
    SUBSCALES,
    Item,
    items_for,
)

# --------------------------------------------------------------------------------------
# Bands and labels
# --------------------------------------------------------------------------------------

#: Lower bound of each band, ascending. The class index is the position in this tuple, so
#: it lines up with the model's three-class output without a second mapping to keep in
#: step.
BAND_LOWER_BOUNDS: Final[tuple[float, ...]] = (0.0, 50.0, 75.0)

#: Human-readable band names, by class index.
BAND_NAMES: Final[tuple[str, ...]] = ("below the burnout range", "moderate", "high")

#: Attached wherever a band is reported.
LABEL_CAVEAT: Final[str] = (
    "These bands (below 50, 50-74, 75 and above) are a reporting convention used with "
    "this questionnaire, not a diagnostic threshold. A score in the moderate or high "
    "band does not mean you have been diagnosed with anything."
)

#: Scores are means of 0-100 anchors, so this is the full possible range.
SCORE_MIN: Final[float] = 0.0
SCORE_MAX: Final[float] = 100.0


@dataclass(frozen=True)
class ScoreResult:
    """
    A scored questionnaire.

    Attributes:
        subscale_scores: Mean score per subscale, each 0-100.
        overall_score: Mean across all items, 0-100. Computed over items rather than by
            averaging the two subscale means, so a subscale with more items carries
            proportionally more weight - the same convention the instrument uses.
        label: Class index from :data:`BAND_LOWER_BOUNDS`, derived from the overall score.
        band: Human-readable name of that class.
        instrument_version: The instrument the answers were given against.
        item_scores: Each item's post-reversal value, kept so a stored response can be
            re-scored later if the banding changes.
    """

    subscale_scores: dict[str, float]
    overall_score: float
    label: int
    band: str
    instrument_version: str = INSTRUMENT_VERSION
    item_scores: dict[str, int] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        """
        Render for the API, with the caveat attached.

        The caveat is part of the payload, not decoration the frontend may forget: a band
        shown without it reads as a verdict.
        """
        return {
            "instrument_version": self.instrument_version,
            "subscale_scores": {
                subscale: round(score, 2) for subscale, score in self.subscale_scores.items()
            },
            "subscale_labels": dict(SUBSCALE_LABELS),
            "overall_score": round(self.overall_score, 2),
            "label": self.label,
            "band": self.band,
            "caveat": LABEL_CAVEAT,
        }


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def score_item(item: Item, raw_value: Any) -> int:
    """
    Validate one answer and apply reverse scoring.

    Parameters:
        item: The item answered.
        raw_value: The submitted value, expected to be one of the item's scale anchors.

    Returns:
        int: The value that enters the mean - inverted for a reverse-scored item.

    Raises:
        ValueError: If the value is not one of the item's permitted anchors. Rejected
            rather than clamped or rounded: a value the scale does not define is a broken
            client or a hand-made request, and quietly coercing it would put an
            uninterpretable number into a research dataset.
    """
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(
            f"Item {item.id!r} expects an integer scale value, got {type(raw_value).__name__}"
        )

    if raw_value not in item.scale.values:
        raise ValueError(
            f"Item {item.id!r} has value {raw_value}, which is not one of "
            f"{sorted(item.scale.values)}"
        )

    # High energy means low burnout, so the one reverse item is inverted about the scale
    # midpoint before it joins the mean.
    return int(SCORE_MAX - raw_value) if item.reverse else raw_value


def label_for_score(score: float) -> int:
    """
    Map an overall score to a class index.

    Parameters:
        score: Overall score, 0-100.

    Returns:
        int: Index into :data:`BAND_LOWER_BOUNDS` - 0 below the burnout range, 1 moderate,
        2 high.

    Raises:
        ValueError: If the score is outside 0-100, which means the caller computed it
            some other way and the band would be meaningless.
    """
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"Score {score} is outside the {SCORE_MIN:g}-{SCORE_MAX:g} range")

    label = 0
    for index, lower_bound in enumerate(BAND_LOWER_BOUNDS):
        if score >= lower_bound:
            label = index
    return label


def score_responses(responses: Mapping[str, Any]) -> ScoreResult:
    """
    Score a complete set of answers.

    Parameters:
        responses: Item id to submitted value.

    Returns:
        ScoreResult: Subscale means, overall score, and the derived band.

    Raises:
        ValueError: If any item is missing, unknown, or carries a value the scale does not
            define.

    Note:
        Every item is required. The instrument's published guidance allows scoring a
        partly-complete response by averaging the answered items when fewer than half are
        missing, but that rule exists for paper questionnaires handed out in a room. Here
        the form can simply require all thirteen, and a complete response is worth more to
        the dataset than a permissive one - a partial record would silently be a different
        measurement from a full one while looking identical in the exported CSV.
    """
    submitted = set(responses)

    missing = REQUIRED_ITEM_IDS - submitted
    if missing:
        raise ValueError(f"Questionnaire is incomplete; missing item(s): {sorted(missing)}")

    unknown = submitted - REQUIRED_ITEM_IDS
    if unknown:
        # An unknown id means the client is on a different instrument version, or is
        # sending fields of its own. Neither should end up in a research dataset.
        raise ValueError(f"Unknown item id(s): {sorted(unknown)}")

    item_scores = {
        item_id: score_item(ITEMS_BY_ID[item_id], value)
        for item_id, value in responses.items()
    }

    subscale_scores = {
        subscale: _mean(item_scores[item.id] for item in items_for(subscale))
        for subscale in SUBSCALES
    }
    overall_score = _mean(item_scores.values())

    return ScoreResult(
        subscale_scores=subscale_scores,
        overall_score=overall_score,
        label=label_for_score(overall_score),
        band=BAND_NAMES[label_for_score(overall_score)],
        item_scores=item_scores,
    )


def _mean(values) -> float:
    """Arithmetic mean of an iterable of numbers, as a float."""
    values = list(values)
    return float(sum(values)) / len(values)
