"""
Disclosure Contract for Keystress-AI (F1 — honest metrics).

Every number this system emits must state where it came from. This module holds the
single source of truth for that contract: the disclaimer text, the data-source vocabulary,
the feature-set version, and helpers that make it hard to format a metric without its
source attached.

Why this exists
---------------
The shipped model is trained on synthetic data whose burnout classes were hand-authored by
``generate_synthetic_data``. Any accuracy measured against those labels describes how
separable the generator made its own classes — it says nothing about detecting real
burnout. Presenting such a figure unqualified would be the project's single largest
credibility failure, so the qualifier travels with the number by construction rather than
by author discipline.

See ``docs/CLAUDE.md`` §2 rule 3 and ``docs/FEATURES.md`` F1.
"""

from __future__ import annotations

from typing import Final, Literal

# --------------------------------------------------------------------------------------
# Data-source vocabulary
# --------------------------------------------------------------------------------------

DataSource = Literal["synthetic", "real"]

#: Data source of the model currently shipped with the project.
#:
#: This stays ``"synthetic"`` until a model trained on consented, labelled, real-world
#: sessions is promoted through the validation harness (F4 -> F5). Changing this constant
#: without that evidence is a hard-rule violation, not a configuration tweak.
SHIPPED_DATA_SOURCE: Final[DataSource] = "synthetic"

#: Human-readable qualifier appended to any metric derived from the shipped model.
DATA_SOURCE_QUALIFIERS: Final[dict[str, str]] = {
    "synthetic": "on synthetic data",
    "real": "on real validated data",
}

# --------------------------------------------------------------------------------------
# Disclaimers
# --------------------------------------------------------------------------------------

#: Attached to every prediction response. Required by the §5 data contract.
DISCLAIMER: Final[str] = (
    "Research indicator only - not a medical or diagnostic assessment. "
    "This reflects one typing session and cannot tell you whether you are burned out. "
    "If you are struggling, please talk to a person you trust or a health professional."
)

#: Longer form for the UI and CLI, explaining *why* the number is not what it looks like.
SYNTHETIC_MODEL_NOTICE: Final[str] = (
    "This model was trained on synthetic data whose burnout classes were defined by hand "
    "in the data generator. Its scores measure how separable those authored classes are, "
    "not any demonstrated ability to detect real burnout. No real-world performance has "
    "been established for this system."
)

# --------------------------------------------------------------------------------------
# Feature set (see ARCHITECTURE.md §4.2)
# --------------------------------------------------------------------------------------

FEATURE_SET_VERSION: Final[str] = "v1"

FEATURES_V1: Final[tuple[str, ...]] = (
    "avg_typing_speed",
    "avg_inter_key_delay",
    "max_pause_duration",
    "backspace_ratio",
    "typing_consistency",
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def qualifier_for(data_source: str) -> str:
    """
    Return the human-readable source qualifier for a data source.

    Parameters:
        data_source: One of the values in :data:`DATA_SOURCE_QUALIFIERS`.

    Returns:
        str: A phrase such as ``"on synthetic data"``.

    Raises:
        ValueError: If the data source is not a recognised value. Unknown sources fail
            loudly rather than silently producing an unqualified metric.
    """
    try:
        return DATA_SOURCE_QUALIFIERS[data_source]
    except KeyError:
        raise ValueError(
            f"Unknown data_source {data_source!r}; "
            f"expected one of {sorted(DATA_SOURCE_QUALIFIERS)}"
        ) from None


def format_metric(name: str, value: float, data_source: str, precision: int = 4) -> str:
    """
    Format a metric with its data source attached.

    Use this anywhere a metric is rendered for a human. It exists so that emitting an
    unqualified number requires deliberately bypassing the helper, which the
    metric-qualifier check (``tools/check_metric_qualifiers.py``) then catches.

    Parameters:
        name: Metric name, e.g. ``"Accuracy"``.
        value: Metric value, e.g. ``0.9013``.
        data_source: Source of the data the metric was measured on.
        precision: Decimal places to render.

    Returns:
        str: e.g. ``"Accuracy: 0.9013 (on synthetic data)"``.
    """
    return f"{name}: {value:.{precision}f} ({qualifier_for(data_source)})"


def format_percentage(name: str, value: float, data_source: str) -> str:
    """
    Format a 0-1 metric as a percentage with its data source attached.

    Parameters:
        name: Metric name, e.g. ``"Confidence"``.
        value: Metric value in the range 0-1.
        data_source: Source of the data the metric was measured on.

    Returns:
        str: e.g. ``"Confidence: 62% (on synthetic data)"``.
    """
    return f"{name}: {value:.0%} ({qualifier_for(data_source)})"
