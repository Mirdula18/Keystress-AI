"""
Feature Engineering for Keystress-AI.

Transforms session timing metadata into the versioned feature set the model consumes.

Every feature here is an *aggregate* over the session — a mean, a max, a standard
deviation, a ratio. None of them retains per-keystroke structure, so none can reconstruct
what was typed. Any new feature must clear the same bar (``docs/FEATURES.md`` F8).

Feature set ``v1`` (see :data:`keystress.core.disclosure.FEATURES_V1`):
    ``avg_typing_speed``     keys per second over the session
    ``avg_inter_key_delay``  mean time between consecutive keystrokes
    ``max_pause_duration``   longest gap between consecutive keystrokes
    ``backspace_ratio``      corrections divided by total keystrokes
    ``typing_consistency``   standard deviation of inter-key delays

Naming note: ``typing_consistency`` is a standard deviation, so a **higher** value means
**less** consistent typing. The name reads the wrong way round and is inherited; renaming
it would invalidate saved models, so it is corrected as part of the versioned feature-set
work in F8 rather than silently here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .disclosure import FEATURES_V1

logger = logging.getLogger(__name__)

#: Feature values returned when a session carries no usable timing signal. Downstream,
#: `inference.has_sufficient_signal` recognises this all-zero vector and abstains rather
#: than scoring it (HARD RULE 6).
ZERO_FEATURES: dict[str, float] = dict.fromkeys(FEATURES_V1, 0.0)


def extract_typing_features(session_data: Mapping[str, Any]) -> dict[str, float]:
    """
    Extract the ``v1`` feature set from session metadata.

    Parameters:
        session_data: Metadata from
            :func:`keystress.core.collect.process_keystroke_data`, carrying
            ``total_keys``, ``backspace_count``, ``duration``, and ``inter_key_delays``.

    Returns:
        dict: The five ``FEATURES_V1`` values. Returns all zeros when the session has no
        keys or no measurable duration, which callers treat as "cannot be scored".
    """
    total_keys = session_data.get("total_keys", 0)
    backspace_count = session_data.get("backspace_count", 0)
    duration = session_data.get("duration", 0)
    inter_key_delays = session_data.get("inter_key_delays", []) or []

    if total_keys <= 0 or duration <= 0:
        return dict(ZERO_FEATURES)

    if inter_key_delays:
        delays = np.asarray(inter_key_delays, dtype=float)
        avg_inter_key_delay = float(np.mean(delays))
        max_pause_duration = float(np.max(delays))
        typing_consistency = float(np.std(delays))
    else:
        avg_inter_key_delay = 0.0
        max_pause_duration = 0.0
        typing_consistency = 0.0

    return {
        "avg_typing_speed": float(total_keys / duration),
        "avg_inter_key_delay": avg_inter_key_delay,
        "max_pause_duration": max_pause_duration,
        "backspace_ratio": float(backspace_count / total_keys),
        "typing_consistency": typing_consistency,
    }


def features_to_dataframe(features: Mapping[str, float]) -> pd.DataFrame:
    """
    Convert a feature dictionary to a single-row DataFrame.

    Parameters:
        features: Feature values.

    Returns:
        pd.DataFrame: One row, columns in ``FEATURES_V1`` order.
    """
    return pd.DataFrame([{name: features.get(name, 0.0) for name in FEATURES_V1}])


def batch_extract_features(sessions: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Extract features from multiple sessions.

    Parameters:
        sessions: Session metadata records.

    Returns:
        pd.DataFrame: One row per session.
    """
    if not sessions:
        return pd.DataFrame(columns=list(FEATURES_V1))
    return pd.DataFrame([extract_typing_features(session) for session in sessions])


def normalize_features(df: pd.DataFrame,
                       feature_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Min-max normalise feature columns.

    Parameters:
        df: DataFrame of features.
        feature_columns: Columns to normalise; defaults to all numeric columns.

    Returns:
        pd.DataFrame: A copy with the selected columns scaled to ``[0, 1]``. Constant
        columns become ``0.0`` rather than dividing by zero.
    """
    columns = (
        feature_columns
        if feature_columns is not None
        else df.select_dtypes(include=[np.number]).columns.tolist()
    )

    normalized = df.copy()
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        spread = max_val - min_val
        normalized[column] = (df[column] - min_val) / spread if spread > 0 else 0.0

    return normalized


def get_feature_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Summarise feature distributions.

    Parameters:
        df: DataFrame of features.

    Returns:
        dict: Per-feature ``mean``, ``std``, ``min``, ``max`` for the columns present.
    """
    return {
        column: {
            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
        }
        for column in FEATURES_V1
        if column in df.columns
    }
