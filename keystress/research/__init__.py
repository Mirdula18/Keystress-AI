"""
Research harness (F4) — the only path that produces *real*, labelled data.

Everything the model currently knows came from :mod:`keystress.ml.synthetic`, whose
burnout classes were authored by hand. Training a classifier to recover authored classes
and reporting the result as performance is circular; this package exists to break that
circle by pairing consented, content-free typing metadata with a self-reported score from
a recognised burnout instrument.

Modules
-------
:mod:`keystress.research.instrument`
    The questionnaire as versioned content: items, response scales, citation, licence,
    and an explicit record of how it was adapted.
:mod:`keystress.research.scoring`
    Turning raw responses into subscale scores and the label the dataset carries.
:mod:`keystress.research.dataset`
    The labelled dataset schema and its export.

What this package must never do
-------------------------------
Collect content. The typing side of a record is the same five aggregate features the
serving path uses (:data:`keystress.core.disclosure.FEATURES_V1`); the questionnaire side
is a fixed set of integer scale values. Neither can carry a character, and no free-text
field exists anywhere in the schema — deliberately, because a free-text box is exactly
where content would eventually arrive.
"""

from __future__ import annotations
