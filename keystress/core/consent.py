"""
Consent policy text and versioning (F2).

The wording a participant agrees to is content, not code, so it lives in one place and
carries a version. Every consent record stores the version in force when it was given, so
that if the policy later changes we can tell exactly what each person agreed to — a
requirement for honest, auditable consent (HARD RULE 4).

Bumping the policy
------------------
Any change to :data:`CONSENT_SUMMARY` that alters what a participant is agreeing to must
also bump :data:`CONSENT_VERSION`. Cosmetic edits (typos) need not, but when in doubt,
bump: an unversioned change to the meaning of consent is exactly the failure this guards.
"""

from __future__ import annotations

#: Version of the consent wording below. Date-stamped so it is human-legible and ordered.
CONSENT_VERSION = "v1-2026-08-05"

#: Plain-language explanation shown before anything is analysed. No jargon, no fine print.
CONSENT_SUMMARY = (
    "Keystress-AI records only the timing of your keystrokes - how quickly you type, the "
    "pauses between keys, and how often you correct yourself. It never records the "
    "characters you type. Your typed text is never captured, stored, or sent anywhere. "
    "Analysis runs only after you agree, and it produces a research indicator for "
    "reflection - not a medical or diagnostic assessment. You can view or permanently "
    "delete everything stored about you at any time."
)

#: The two things a participant can agree to, independently.
#: ``ANALYSIS`` is required to run a prediction at all; ``DONATE`` is a separate opt-in to
#: store the session's timing features for research. Neither implies the other.
CONSENT_ANALYSIS = "analysis"
CONSENT_DONATE = "donate"
