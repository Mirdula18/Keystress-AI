"""
Offline machine-learning pipeline: synthetic data generation and model training.

Kept separate from the serving path (``ARCHITECTURE.md`` §6). The serving model is only
ever updated by loading a saved artifact - never trained inline during a request.
"""
