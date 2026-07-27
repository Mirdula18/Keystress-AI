#!/usr/bin/env python3
"""
Reproducible-build check (F13).

Builds the synthetic dataset and model twice from scratch, in separate temporary
directories, and verifies the two runs agree.

What "reproducible" means here
------------------------------
Not byte-identical pickles. A joblib artifact can differ between runs for reasons that
have nothing to do with the model — memory layout, dict iteration details, library build
flags — so byte comparison produces both false alarms and, worse, a green light that says
nothing about behaviour.

This checks the properties that actually matter:

1. The generated **dataset** is byte-identical. This is the real determinism claim, and it
   is where a seeding mistake shows up first.
2. The two models produce **identical predictions** on a fixed probe set, to full float
   precision — including class probabilities, not just argmax.
3. The **metrics** are identical.
4. The **model version string** is identical, which is what lets a version identifier be
   trusted as a reference to a specific model.

A model that predicts identically and reports identical metrics is reproducible in every
sense a user of this project cares about.

Usage
-----
    python tools/check_reproducible_build.py            # two runs, seed 42
    python tools/check_reproducible_build.py --seed 7
    python tools/check_reproducible_build.py --samples 300   # faster, for local use

Exit codes: 0 reproducible, 1 divergence found.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from keystress.core.model import load_bundle
from keystress.ml.synthetic import generate_synthetic_typing_data, save_synthetic_data
from keystress.ml.train import train_and_evaluate

#: Fixed probe sessions used to compare model behaviour. Spans the feature space so a
#: divergence anywhere in the decision surface is visible, not just near one class.
PROBE_FEATURES: list[list[float]] = [
    [5.2, 0.19, 0.8, 0.04, 0.05],
    [3.4, 0.36, 2.1, 0.16, 0.13],
    [1.9, 0.62, 4.5, 0.29, 0.26],
    [4.1, 0.25, 1.2, 0.09, 0.08],
    [2.6, 0.48, 3.0, 0.22, 0.19],
    [0.6, 0.90, 8.0, 0.45, 0.40],
    [7.0, 0.10, 0.4, 0.01, 0.02],
]


def file_digest(path: Path) -> str:
    """
    Return the SHA-256 digest of a file.

    Parameters:
        path: File to hash.

    Returns:
        str: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_once(workdir: Path, seed: int, samples: int) -> dict[str, Any]:
    """
    Perform one complete build: generate data, train, and probe the model.

    Parameters:
        workdir: Directory to build in.
        seed: Random seed.
        samples: Number of synthetic samples.

    Returns:
        dict: Observable outputs of the build.
    """
    data_path = workdir / "synthetic.csv"
    model_path = workdir / "model.pkl"
    scaler_path = workdir / "scaler.pkl"
    metadata_path = workdir / "metadata.json"

    save_synthetic_data(
        generate_synthetic_typing_data(n_samples=samples, random_state=seed), data_path
    )

    results = train_and_evaluate(
        data_path=data_path,
        model_path=model_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        random_state=seed,
    )

    bundle = load_bundle(model_path, scaler_path, metadata_path)
    probe = np.array(PROBE_FEATURES)
    scaled = bundle.scaler.transform(probe)

    return {
        "dataset_digest": file_digest(data_path),
        "model_version": results["metadata"]["model_version"],
        "predictions": bundle.estimator.predict(scaled).tolist(),
        "probabilities": bundle.estimator.predict_proba(scaled).tolist(),
        "metrics": results["metadata"]["metrics"],
    }


def compare(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """
    Compare two builds and describe any divergence.

    Parameters:
        first: First build's outputs.
        second: Second build's outputs.

    Returns:
        list[str]: Human-readable failure descriptions; empty when reproducible.
    """
    failures: list[str] = []

    if first["dataset_digest"] != second["dataset_digest"]:
        failures.append(
            "Generated dataset differs between runs.\n"
            f"    run 1: {first['dataset_digest']}\n"
            f"    run 2: {second['dataset_digest']}\n"
            "    The synthetic generator is not deterministic for a fixed seed."
        )

    if first["model_version"] != second["model_version"]:
        failures.append(
            "Model version differs between runs.\n"
            f"    run 1: {first['model_version']}\n"
            f"    run 2: {second['model_version']}\n"
            "    A version identifier must reference exactly one model."
        )

    if first["predictions"] != second["predictions"]:
        failures.append(
            "Models disagree on the probe set.\n"
            f"    run 1: {first['predictions']}\n"
            f"    run 2: {second['predictions']}"
        )

    if first["probabilities"] != second["probabilities"]:
        differing = [
            i for i, (a, b) in enumerate(
                zip(first["probabilities"], second["probabilities"])
            ) if a != b
        ]
        failures.append(
            "Models produce different class probabilities.\n"
            f"    diverging probe rows: {differing}\n"
            "    Predictions may still agree, but the models are not identical."
        )

    if first["metrics"] != second["metrics"]:
        failures.append(
            "Reported metrics differ between runs.\n"
            f"    run 1: {first['metrics']}\n"
            f"    run 2: {second['metrics']}"
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    """
    Run the reproducibility check.

    Parameters:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 when reproducible, 1 on divergence.
    """
    parser = argparse.ArgumentParser(description="Verify reproducible synthetic model builds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--samples", type=int, default=1500,
                        help="Synthetic samples per build (default: 1500)")
    args = parser.parse_args(argv)

    print(f"Building twice from scratch: seed={args.seed}, samples={args.samples}")

    builds = []
    for run in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"keystress-repro-{run}-") as tmp:
            print(f"  run {run} ...", end="", flush=True)
            builds.append(build_once(Path(tmp), args.seed, args.samples))
            print(" done")

    failures = compare(builds[0], builds[1])

    if not failures:
        print("\nReproducibility check passed.")
        print(f"  dataset sha256: {builds[0]['dataset_digest'][:16]}...")
        print(f"  model version:  {builds[0]['model_version']}")
        print("  Two independent builds produced an identical dataset, identical")
        print("  predictions and probabilities on the probe set, and identical metrics.")
        return 0

    print(f"\nReproducibility check FAILED: {len(failures)} divergence(s).\n")
    for failure in failures:
        print(f"  - {failure}\n")
    print("A clean checkout must reproduce the same synthetic model (F13).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
