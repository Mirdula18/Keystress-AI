"""
Keystress-AI — a privacy-preserving research prototype for typing dynamics.

Examines whether typing *rhythm* (speed, pauses, corrections) relates to academic
wellbeing, while never capturing the characters typed. Outputs are research indicators
with their data source attached — not medical or diagnostic assessments.

The shipped model is trained on synthetic data whose burnout classes were authored by hand
in the generator, so every metric it produces is labelled ``data_source: "synthetic"`` and
means only that. See ``docs/CLAUDE.md`` §1.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
