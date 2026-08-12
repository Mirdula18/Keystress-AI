"""
The labelling instrument (F4, D-020): a studies-adapted Copenhagen Burnout Inventory.

The label in the real dataset is a self-report from a recognised instrument rather than
anything this project invents. D-020 chose the **Copenhagen Burnout Inventory** (CBI) for
one decisive reason: it is free to use, so anyone can fork this repository and run the
site without buying a licence. The obvious student alternative, the Maslach Burnout
Inventory - Student Survey, is licensed per administration, which is incompatible with an
open, self-hostable tool.

Citation
--------
Kristensen, T. S., Borritz, M., Villadsen, E., & Christensen, K. B. (2005). *The
Copenhagen Burnout Inventory: A new tool for the assessment of burnout.* Work & Stress,
19(3), 192-207. https://doi.org/10.1080/02678370500297720

The CBI is published as a free instrument; the authors ask that it be cited. Any UI that
shows these items must show the citation with them, which is why :data:`CITATION` travels
in the API payload rather than living only in this docstring.

**This is an adaptation, and adaptations are not the validated original.**
-------------------------------------------------------------------------
Two deliberate changes, both required by D-020 and both recorded in
:data:`ADAPTATION_NOTE` so a participant and a future reader see them:

1. **The client-related subscale is dropped.** Its six items ask about working with
   clients, which a student answering this site has none of. Keeping them would collect
   six items of noise and imply the tool is about care work.
2. **"Work" is re-worded as "studies".** The work-related subscale becomes
   studies-related: "your work" → "your studies", "the working day" → "the study day".

The consequence is honest but real: **the psychometric properties established for the CBI
were established for the CBI, not for this shortened re-wording.** No claim is made that
this adaptation is validated. It is a defensible label source for a research prototype,
and the protocol says so in those words.

**Verify the wording before fielding a study.** The item text below was checked against
public reference descriptions of the instrument, but published versions differ in one
known respect: whether the last three studies-related items take the frequency options or
the degree options. Both option sets map onto the identical 0-100 anchors, so a subscale
mean is unaffected either way — but anyone running this with participants should confirm
the wording against Kristensen et al. (2005) directly and bump
:data:`INSTRUMENT_VERSION` if they change anything.

Versioning
----------
:data:`INSTRUMENT_VERSION` is stored on every response row. A change to any item's text,
its scale, or its reverse flag changes what a stored score *means*, so it must bump the
version — otherwise old and new answers would be silently pooled in the dataset as if
they came from the same questionnaire. This is the same rule
:mod:`keystress.core.consent` applies to the consent wording, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

#: Bump on any change to item text, scale, ordering, or reverse-scoring.
INSTRUMENT_VERSION: Final[str] = "cbi-studies-v1"

INSTRUMENT_NAME: Final[str] = "Copenhagen Burnout Inventory (studies-adapted, short form)"

CITATION: Final[str] = (
    "Kristensen, T. S., Borritz, M., Villadsen, E., & Christensen, K. B. (2005). "
    "The Copenhagen Burnout Inventory: A new tool for the assessment of burnout. "
    "Work & Stress, 19(3), 192-207."
)

LICENCE_NOTE: Final[str] = (
    "The Copenhagen Burnout Inventory is free to use and adapt, with citation. That is "
    "why it was chosen over licensed alternatives: anyone can run this site without "
    "paying per response."
)

ADAPTATION_NOTE: Final[str] = (
    "This is an adaptation, not the original questionnaire: the client-related subscale "
    "is omitted, and the work-related items are re-worded to ask about studies. The "
    "published evidence for the Copenhagen Burnout Inventory was established for the "
    "original instrument, so it does not automatically transfer to this shortened, "
    "re-worded version."
)

#: Shown above the questionnaire. A burnout score is not a diagnosis, and a site that
#: collects one has an obligation to say so before the person answers, not after.
INSTRUMENT_DISCLAIMER: Final[str] = (
    "This questionnaire is a research measure, not a medical assessment, and it cannot "
    "diagnose anything. It gives you a score describing how you have been feeling "
    "recently. If your answers worry you, please talk to a person you trust or a health "
    "professional."
)

# --------------------------------------------------------------------------------------
# Response scales
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseScale:
    """
    A set of answer options and the value each carries.

    Attributes:
        name: Identifier used in the API payload.
        options: ``(label, value)`` pairs, ordered as they should be displayed. Values
            are the CBI's 0-100 anchors, so a subscale mean is directly on that scale.
    """

    name: str
    options: tuple[tuple[str, int], ...]

    @property
    def values(self) -> frozenset[int]:
        """The accepted values, for validating a submitted response."""
        return frozenset(value for _, value in self.options)

    def as_payload(self) -> dict[str, Any]:
        """
        Render for the API, as a list of ``{label, score}`` objects.

        The key is ``score`` rather than the obvious ``value`` because ``value`` is on the
        project's forbidden-field list (:data:`keystress.core.collect.FORBIDDEN_EVENT_FIELDS`)
        — it is one of the names a leaked input field would arrive under. Using it here for
        something harmless would make every content scan of this payload a false positive,
        and a scan that cries wolf is one someone eventually weakens.
        """
        return {
            "name": self.name,
            "options": [{"label": label, "score": score} for label, score in self.options],
        }


#: "How often …" items.
FREQUENCY_SCALE: Final[ResponseScale] = ResponseScale(
    name="frequency",
    options=(
        ("Always", 100),
        ("Often", 75),
        ("Sometimes", 50),
        ("Seldom", 25),
        ("Never / almost never", 0),
    ),
)

#: "To what degree …" items. Different words, identical anchors, so the two scales mix
#: within a subscale without distorting its mean.
DEGREE_SCALE: Final[ResponseScale] = ResponseScale(
    name="degree",
    options=(
        ("To a very high degree", 100),
        ("To a high degree", 75),
        ("Somewhat", 50),
        ("To a low degree", 25),
        ("To a very low degree", 0),
    ),
)

# --------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------

#: The two subscales this adaptation keeps. Client-related is dropped (see module docs).
SUBSCALE_PERSONAL: Final[str] = "personal"
SUBSCALE_STUDIES: Final[str] = "studies"

SUBSCALES: Final[tuple[str, ...]] = (SUBSCALE_PERSONAL, SUBSCALE_STUDIES)

SUBSCALE_LABELS: Final[dict[str, str]] = {
    SUBSCALE_PERSONAL: "Personal burnout",
    SUBSCALE_STUDIES: "Studies-related burnout",
}

SUBSCALE_DESCRIPTIONS: Final[dict[str, str]] = {
    SUBSCALE_PERSONAL: (
        "How physically and emotionally exhausted you feel in general - whatever the "
        "cause."
    ),
    SUBSCALE_STUDIES: (
        "How much of that exhaustion you attribute to your studies specifically."
    ),
}


@dataclass(frozen=True)
class Item:
    """
    One questionnaire item.

    Attributes:
        id: Stable identifier stored with the response. Never renumber an existing id -
            bump :data:`INSTRUMENT_VERSION` and add a new one instead, or stored answers
            will silently change meaning.
        text: The question as shown.
        subscale: Which subscale it contributes to.
        scale: Which response scale it uses.
        reverse: Whether a high answer means *less* burnout, so the value is inverted
            before averaging.
    """

    id: str
    text: str
    subscale: str
    scale: ResponseScale
    reverse: bool = False

    def as_payload(self) -> dict[str, Any]:
        """
        Render for the API.

        ``reverse`` is deliberately included: the scoring rule is not a secret, and hiding
        it would make the score harder to audit, not safer. The question is sent as
        ``question``, not ``text``, for the same reason the scale sends ``score`` rather
        than ``value`` — see :meth:`ResponseScale.as_payload`.
        """
        return {
            "id": self.id,
            "question": self.text,
            "subscale": self.subscale,
            "scale": self.scale.name,
            "reverse": self.reverse,
        }


#: The 13 items: 6 personal + 7 studies-related.
#:
#: Order is the display order and is part of the versioned content — questionnaire
#: responses are known to be sensitive to item order, so it is fixed here rather than
#: left to whatever the UI happens to do.
ITEMS: Final[tuple[Item, ...]] = (
    Item("p1", "How often do you feel tired?", SUBSCALE_PERSONAL, FREQUENCY_SCALE),
    Item("p2", "How often are you physically exhausted?", SUBSCALE_PERSONAL, FREQUENCY_SCALE),
    Item("p3", "How often are you emotionally exhausted?", SUBSCALE_PERSONAL, FREQUENCY_SCALE),
    Item("p4", "How often do you think: “I can’t take it anymore”?",
         SUBSCALE_PERSONAL, FREQUENCY_SCALE),
    Item("p5", "How often do you feel worn out?", SUBSCALE_PERSONAL, FREQUENCY_SCALE),
    Item("p6", "How often do you feel weak and susceptible to illness?",
         SUBSCALE_PERSONAL, FREQUENCY_SCALE),

    Item("s1", "Do you feel worn out at the end of a day of studying?",
         SUBSCALE_STUDIES, FREQUENCY_SCALE),
    Item("s2", "Are you exhausted in the morning at the thought of another day of study?",
         SUBSCALE_STUDIES, FREQUENCY_SCALE),
    Item("s3", "Do you feel that every hour of study is tiring for you?",
         SUBSCALE_STUDIES, FREQUENCY_SCALE),
    # The one reverse-scored item in the instrument: having energy is the *absence* of
    # burnout, so the raw value is inverted before it joins the subscale mean.
    Item("s4", "Do you have enough energy for family and friends during your free time?",
         SUBSCALE_STUDIES, FREQUENCY_SCALE, reverse=True),
    Item("s5", "Are your studies emotionally exhausting?", SUBSCALE_STUDIES, DEGREE_SCALE),
    Item("s6", "Do your studies frustrate you?", SUBSCALE_STUDIES, DEGREE_SCALE),
    Item("s7", "Do you feel burnt out because of your studies?",
         SUBSCALE_STUDIES, DEGREE_SCALE),
)

#: Lookup by id, built once. Used by scoring to reject an unknown item id.
ITEMS_BY_ID: Final[dict[str, Item]] = {item.id: item for item in ITEMS}

#: The ids a complete submission must carry, as a set for cheap comparison.
REQUIRED_ITEM_IDS: Final[frozenset[str]] = frozenset(ITEMS_BY_ID)


def items_for(subscale: str) -> tuple[Item, ...]:
    """
    Return the items belonging to a subscale, in display order.

    Parameters:
        subscale: One of :data:`SUBSCALES`.

    Returns:
        tuple[Item, ...]: The subscale's items.

    Raises:
        ValueError: If the subscale is unknown. An unknown subscale would otherwise
            silently return no items and produce a mean of nothing.
    """
    if subscale not in SUBSCALES:
        raise ValueError(f"Unknown subscale {subscale!r}; expected one of {list(SUBSCALES)}")
    return tuple(item for item in ITEMS if item.subscale == subscale)


def as_payload() -> dict[str, Any]:
    """
    Render the whole instrument for the API.

    Everything a page needs to display the questionnaire *honestly* is in here: the items,
    the scales, and the provenance. The citation, licence note, adaptation note, and
    disclaimer travel with the items rather than being left to the frontend to remember,
    because a UI that shows the questions without them is the failure mode this design is
    trying to prevent.

    Returns:
        dict: JSON-serialisable instrument payload.
    """
    return {
        "instrument_version": INSTRUMENT_VERSION,
        "name": INSTRUMENT_NAME,
        "citation": CITATION,
        "licence_note": LICENCE_NOTE,
        "adaptation_note": ADAPTATION_NOTE,
        "disclaimer": INSTRUMENT_DISCLAIMER,
        "scales": {
            FREQUENCY_SCALE.name: FREQUENCY_SCALE.as_payload(),
            DEGREE_SCALE.name: DEGREE_SCALE.as_payload(),
        },
        "subscales": [
            {
                "id": subscale,
                "label": SUBSCALE_LABELS[subscale],
                "description": SUBSCALE_DESCRIPTIONS[subscale],
            }
            for subscale in SUBSCALES
        ],
        "items": [item.as_payload() for item in ITEMS],
    }
