"""
Validation status (F5): whether a model has ever been tested against real people.

The distinction this module encodes is the one the whole project turns on. A model trained
on :mod:`keystress.ml.synthetic` has been measured only against classes the generator
authored; a model that has been evaluated on the F4 dataset has been measured against
self-reported scores from people. Those are not two points on a scale of quality, they are
different kinds of claim, and conflating them is the failure `docs/CLAUDE.md` §1 is written
to prevent.

So the status is derived, never set by hand. It is a function of two facts — what the model
was trained on, and whether a real-data evaluation report exists for this exact model
version — and nothing else. There is no field an optimistic future contributor can flip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from ..core.disclosure import SYNTHETIC_MODEL_NOTICE

#: Where evaluation reports are written. One file per model version and dataset.
DEFAULT_REPORT_DIR: Final[Path] = Path("models") / "eval"

#: The three states a model can be in.
NOT_VALIDATED: Final[str] = "not-validated"
VALIDATED_ON_REAL: Final[str] = "validated-on-real-data"
EVALUATED_NO_SKILL: Final[str] = "evaluated-no-skill-demonstrated"

#: Plain-language explanation of each status. These are what a UI or a report shows; the
#: status string itself is for code.
STATUS_EXPLANATIONS: Final[dict[str, str]] = {
    NOT_VALIDATED: (
        "This model has never been evaluated against real, self-reported data. Its scores "
        "describe how separable the synthetic generator made its own classes, not any "
        "ability to detect burnout in a person."
    ),
    VALIDATED_ON_REAL: (
        "This model has been evaluated on real, consented, self-reported sessions using "
        "participant-grouped splits, and it beat every trivial baseline on that data. "
        "That makes its reported performance a real measurement - read it with the "
        "dataset's own limitations in mind."
    ),
    EVALUATED_NO_SKILL: (
        "This model has been evaluated on real, consented sessions and did NOT beat the "
        "trivial baselines. On the evidence available, it has not demonstrated any "
        "ability to detect burnout from typing. That is a legitimate result and it is "
        "reported rather than hidden."
    ),
}


def status_for(
    metadata: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> str:
    """
    Derive a model's validation status.

    Parameters:
        metadata: Model metadata, carrying ``data_source`` and ``model_version``.
        report: The evaluation report for this model, if one exists.

    Returns:
        str: One of :data:`NOT_VALIDATED`, :data:`VALIDATED_ON_REAL`,
        :data:`EVALUATED_NO_SKILL`.

    Note:
        A synthetic-data evaluation never confers validation, however good its numbers.
        The report must be against ``data_source == "real"`` *and* the model must have
        beaten every trivial baseline — a model that cannot outperform "always guess the
        most common class" has been evaluated, not validated, and saying so is the point
        of the third state.
    """
    if report is None:
        return NOT_VALIDATED

    if report.get("dataset", {}).get("data_source") != "real":
        return NOT_VALIDATED

    if report.get("model_version") != metadata.get("model_version"):
        # A report for a different model says nothing about this one. Model versions are
        # deterministic (F13), so a mismatch means the artifact changed.
        return NOT_VALIDATED

    return (
        VALIDATED_ON_REAL if report.get("beats_all_baselines")
        else EVALUATED_NO_SKILL
    )


def explain(status: str) -> str:
    """
    Return the plain-language explanation of a status.

    Parameters:
        status: One of the status constants.

    Returns:
        str: The explanation, or a conservative fallback for an unknown status — an
        unrecognised status is treated as unvalidated rather than as an unknown quantity,
        because the safe reading of "we do not know" is "no evidence".
    """
    return STATUS_EXPLANATIONS.get(status, STATUS_EXPLANATIONS[NOT_VALIDATED])


def find_report(
    model_version: str,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any] | None:
    """
    Load the most recent evaluation report for a model version, if any.

    Parameters:
        model_version: The model's version identifier.
        report_dir: Directory holding evaluation reports.

    Returns:
        dict | None: The report, or ``None`` when there is none or it cannot be read.
        A corrupt report is treated as absent — the conservative direction, since the
        alternative is claiming validation on the strength of a file nobody can parse.
    """
    import json

    directory = Path(report_dir)
    if not directory.is_dir():
        return None

    candidates = sorted(directory.glob(f"{model_version}__*.json"))
    if not candidates:
        return None

    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def disclosure_for(metadata: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build the validation block that accompanies a model wherever it is described.

    Parameters:
        metadata: Model metadata.
        report: The model's evaluation report, if any.

    Returns:
        dict: ``status``, ``explanation``, ``evaluated_on``, and — for an unvalidated
        synthetic model — the standing synthetic-data notice, so a caller that shows only
        this block still tells the truth.
    """
    status = status_for(metadata, report)
    block: dict[str, Any] = {
        "status": status,
        "explanation": explain(status),
        "evaluated_on": (report or {}).get("dataset", {}).get("data_source"),
        "evaluation_report": (report or {}).get("report_id"),
    }
    if status == NOT_VALIDATED and metadata.get("data_source") == "synthetic":
        block["notice"] = SYNTHETIC_MODEL_NOTICE
    return block
