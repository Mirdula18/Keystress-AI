"""
Synthetic Data Generation for Keystress-AI.

Generates typing-behaviour data with burnout labels **written by hand**. This module is
the source of the project's central scientific problem, and it says so in its own
docstring so nobody downstream can be unaware:

    The three "burnout classes" below are statistical distributions chosen by the author.
    A classifier trained on them learns to recover distributions this file defined. Any
    accuracy measured against these labels describes how separable those authored classes
    are — it is **not** evidence that typing dynamics reveal burnout.

The generator is kept because it is genuinely useful for tests, for exercising the
pipeline, and for augmentation once real data exists (``docs/FEATURES.md`` F4). It is not
kept as a source of truth.

Burnout levels:
    0 = lower indicators, 1 = mixed indicators, 2 = higher indicators
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Version of the data-generating process. Bump whenever the distributions change, so a
#: model's version string records which generator produced its training data.
#:
#: g1 - inherited generator; clamped out-of-range draws with ``max(floor, x)``, which
#:      collapsed them onto a single value and created an artificial point mass.
#: g2 - resamples out-of-range draws instead of clamping (see :func:`_truncated_normal`).
SYNTHETIC_GENERATOR_VERSION = "g2"

DEFAULT_OUTPUT_PATH = Path("data") / "synthetic_typing_data.csv"

#: Hand-authored class parameters. Every number here is an authorial choice, not a
#: measurement. Format: (mean, std, floor).
CLASS_PARAMETERS: dict[int, dict[str, tuple[float, float, float]]] = {
    0: {  # lower indicators: fast, consistent, few corrections
        "avg_typing_speed": (5.0, 0.8, 0.5),
        "avg_inter_key_delay": (0.2, 0.05, 0.05),
        "typing_consistency": (0.05, 0.02, 0.01),
    },
    1: {  # mixed indicators
        "avg_typing_speed": (3.5, 1.0, 0.5),
        "avg_inter_key_delay": (0.35, 0.1, 0.05),
        "typing_consistency": (0.12, 0.04, 0.01),
    },
    2: {  # higher indicators: slow, variable, more corrections
        "avg_typing_speed": (2.0, 0.8, 0.5),
        "avg_inter_key_delay": (0.6, 0.15, 0.05),
        "typing_consistency": (0.25, 0.08, 0.01),
    },
}

#: Pause distributions per class: (exponential scale, offset).
PAUSE_PARAMETERS: dict[int, tuple[float, float]] = {
    0: (1.0, 0.5),
    1: (2.0, 1.0),
    2: (3.5, 2.0),
}

#: Backspace-ratio Beta parameters per class.
BACKSPACE_PARAMETERS: dict[int, tuple[float, float]] = {
    0: (2.0, 20.0),
    1: (3.0, 15.0),
    2: (4.0, 10.0),
}

#: Upper bound on backspace ratio. A session cannot plausibly be mostly corrections.
MAX_BACKSPACE_RATIO = 0.5

#: Safety valve for rejection sampling.
_MAX_RESAMPLE_ROUNDS = 100


def _truncated_normal(rng: np.random.Generator, mean: float, std: float,
                      floor: float, size: int) -> np.ndarray:
    """
    Draw from a normal distribution truncated below at ``floor``.

    **Why this is not ``max(floor, x)``.** The inherited generator clamped out-of-range
    draws to the floor, which does not truncate a distribution — it piles every rejected
    draw onto one exact value. For ``typing_consistency`` in class 0
    (``normal(0.05, 0.02)``, floor ``0.01``) that put roughly 2.3% of the class on exactly
    ``0.01``, and for ``avg_typing_speed`` in class 2 about 3% on exactly ``0.5``. A
    tree-based model can split on such a spike and score well by detecting an artifact of
    the generator rather than any pattern. No real typing session produces a value
    repeated to the bit.

    Rejection sampling instead redraws out-of-range values, giving a genuinely truncated
    distribution with no point mass.

    Parameters:
        rng: Seeded random generator.
        mean: Distribution mean.
        std: Distribution standard deviation.
        floor: Exclusive lower bound; draws at or below this are redrawn.
        size: Number of samples.

    Returns:
        np.ndarray: ``size`` samples, all strictly above ``floor``.

    Raises:
        RuntimeError: If the bound is so extreme that sampling cannot converge, which
            would otherwise loop forever.
    """
    samples = rng.normal(mean, std, size=size)
    for _ in range(_MAX_RESAMPLE_ROUNDS):
        below = samples <= floor
        if not below.any():
            return samples
        samples[below] = rng.normal(mean, std, size=int(below.sum()))

    raise RuntimeError(
        f"Rejection sampling failed to converge for normal(mean={mean}, std={std}) "
        f"truncated at {floor}: the floor is too far into the distribution's tail."
    )


def generate_synthetic_typing_data(n_samples: int = 1000,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic typing dataset with hand-authored burnout labels.

    The labels are *defined here*, not observed. See the module docstring.

    Parameters:
        n_samples: Total number of samples, split as evenly as possible across classes.
        random_state: Seed. Identical seeds produce identical datasets, which F13 relies
            on for reproducible model builds.

    Returns:
        pd.DataFrame: Columns are the five ``FEATURES_V1`` values plus ``burnout_level``.
    """
    rng = np.random.default_rng(random_state)

    per_class = n_samples // 3
    class_sizes = {0: per_class, 1: per_class, 2: n_samples - 2 * per_class}

    frames: list[pd.DataFrame] = []
    for level, size in class_sizes.items():
        if size <= 0:
            continue

        params = CLASS_PARAMETERS[level]
        pause_scale, pause_offset = PAUSE_PARAMETERS[level]
        beta_a, beta_b = BACKSPACE_PARAMETERS[level]

        speed_mean, speed_std, speed_floor = params["avg_typing_speed"]
        delay_mean, delay_std, delay_floor = params["avg_inter_key_delay"]
        consistency_mean, consistency_std, consistency_floor = params["typing_consistency"]

        frames.append(pd.DataFrame({
            "avg_typing_speed": _truncated_normal(
                rng, speed_mean, speed_std, speed_floor, size
            ),
            "avg_inter_key_delay": _truncated_normal(
                rng, delay_mean, delay_std, delay_floor, size
            ),
            # Exponential draws are non-negative by construction, so the offset alone
            # keeps them in range - no clamping needed and no point mass created.
            "max_pause_duration": rng.exponential(pause_scale, size=size) + pause_offset,
            # Beta draws are already bounded to [0, 1]; scaling to the plausible ceiling
            # preserves the shape instead of flattening its upper tail onto the bound.
            "backspace_ratio": rng.beta(beta_a, beta_b, size=size) * MAX_BACKSPACE_RATIO,
            "typing_consistency": _truncated_normal(
                rng, consistency_mean, consistency_std, consistency_floor, size
            ),
            "burnout_level": level,
        }))

    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def save_synthetic_data(df: pd.DataFrame,
                        filepath: Path | None = None) -> Path:
    """
    Save a synthetic dataset to CSV.

    Parameters:
        df: The dataset.
        filepath: Destination path; defaults to ``data/synthetic_typing_data.csv``.

    Returns:
        Path: The path written.
    """
    path = Path(filepath) if filepath is not None else DEFAULT_OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    logger.info("Synthetic dataset saved to %s (%d samples)", path, len(df))
    return path


def main() -> int:
    """
    CLI entrypoint: generate and save the default synthetic dataset.

    Returns:
        int: Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("Generating SYNTHETIC typing dataset with hand-authored labels.")
    print("These labels are authored, not observed - see module docstring.")

    df = generate_synthetic_typing_data(n_samples=1500)
    path = save_synthetic_data(df)

    print(f"\nWrote {len(df)} samples to {path}")
    print(f"Generator version: {SYNTHETIC_GENERATOR_VERSION}")
    print("\nClass distribution:")
    print(df["burnout_level"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
