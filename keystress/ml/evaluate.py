"""
The validation harness (F5): evaluate a model honestly, and persist what was found.

This module answers one question — *does this model do anything useful?* — in a way that
cannot be quietly answered "yes". Four properties are structural rather than optional:

1. **Splits are by participant** (:mod:`keystress.ml.splits`). A random split lets the
   model score well by recognising a person across their own sessions.
2. **Trivial baselines are computed every time** (:mod:`keystress.ml.baselines`) and
   printed beside the model, never in an appendix. The report states in words whether the
   model beat them.
3. **Every number carries its data source.** A report over synthetic data says so on every
   line, and :mod:`keystress.ml.validation` refuses to call such a run validation
   regardless of the figures.
4. **The report keeps its own warnings.** Small test set, single-participant fold, a class
   never predicted, baselines not beaten — these travel inside the persisted JSON, so a
   report read later cannot be read without them.

The report is written per model version, so a claim can always be traced back to the exact
artifact it was made about (model versions are deterministic — F13/D-017).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.disclosure import (
    FEATURES_V1,
    SYNTHETIC_MODEL_NOTICE,
    format_metric,
    qualifier_for,
)
from ..core.model import ModelBundle
from .baselines import BASELINE_DESCRIPTIONS, baseline_predictions, beats_every_baseline
from .metrics import calibration_metrics, classification_metrics, never_predicted_classes
from .splits import InsufficientDataError, Split, grouped_split
from .validation import DEFAULT_REPORT_DIR, explain, status_for

logger = logging.getLogger(__name__)

#: Column holding the label in a labelled dataset.
LABEL_COLUMN = "label"

#: Display names for the three indicator classes.
CLASS_NAMES = ["Low", "Medium", "High"]

#: The classes the task defines, whether or not a given dataset contains them all.
TASK_CLASSES = [0, 1, 2]

#: The headline metric. Named once so the report, the baseline comparison, and the
#: validation verdict cannot end up using different ones.
HEADLINE_METRIC = "f1_macro"


def load_labelled_dataset(path: Path) -> pd.DataFrame:
    """
    Load a labelled dataset produced by ``keystress-export``.

    Parameters:
        path: Path to the CSV.

    Returns:
        pd.DataFrame: The dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        InsufficientDataError: If required columns are missing or it holds no rows. An
            empty dataset is not an error to be worked around — it means nobody has
            contributed yet, and there is nothing to evaluate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No labelled dataset at {path}. Export one with `keystress-export`."
        )

    frame = pd.read_csv(path)

    required = {"participant_id", LABEL_COLUMN, *FEATURES_V1}
    missing = required - set(frame.columns)
    if missing:
        raise InsufficientDataError(
            f"Dataset at {path} is missing required column(s): {sorted(missing)}"
        )
    if frame.empty:
        raise InsufficientDataError(f"Dataset at {path} contains no rows")

    return frame


def _predict(bundle: ModelBundle, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a bundle over a feature frame.

    Parameters:
        bundle: The model.
        features: Rows of :data:`keystress.core.disclosure.FEATURES_V1`.

    Returns:
        tuple: ``(predictions, probabilities)``.
    """
    scaled = bundle.scaler.transform(features[list(FEATURES_V1)].to_numpy())
    return bundle.estimator.predict(scaled), bundle.estimator.predict_proba(scaled)


def evaluate_split(
    bundle: ModelBundle,
    split: Split,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a model on one split, against the trivial baselines.

    Parameters:
        bundle: The model to evaluate.
        split: A participant-grouped split.
        seed: Seed for the random baselines.

    Returns:
        dict: ``model``, ``baselines``, ``calibration``, and the comparison verdict.
    """
    y_train = split.train[LABEL_COLUMN].to_numpy(dtype=int)
    y_test = split.test[LABEL_COLUMN].to_numpy(dtype=int)

    predictions, probabilities = _predict(bundle, split.test)

    model_metrics = classification_metrics(
        y_test, predictions, classes=TASK_CLASSES, class_names=CLASS_NAMES
    )

    baselines: dict[str, Any] = {}
    for name, baseline_pred in baseline_predictions(
        y_train, len(y_test), classes=np.array(TASK_CLASSES), seed=seed
    ).items():
        baselines[name] = {
            "description": BASELINE_DESCRIPTIONS[name],
            **classification_metrics(
                y_test, baseline_pred, classes=TASK_CLASSES, class_names=CLASS_NAMES
            ),
        }

    headline = model_metrics[HEADLINE_METRIC]
    baseline_headlines = {name: stats[HEADLINE_METRIC] for name, stats in baselines.items()}

    return {
        "model": model_metrics,
        "baselines": baselines,
        "calibration": calibration_metrics(y_test, probabilities),
        "headline_metric": HEADLINE_METRIC,
        "headline_score": headline,
        "baseline_headline_scores": baseline_headlines,
        "beats_all_baselines": beats_every_baseline(headline, baseline_headlines),
        "never_predicted": never_predicted_classes(model_metrics),
        "split": split.summary(),
    }


def evaluate(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    data_source: str,
    test_size: float = 0.3,
    seed: int = 42,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a model on a labelled dataset and build a full report.

    Parameters:
        bundle: The model to evaluate.
        frame: The labelled dataset.
        data_source: ``"real"`` or ``"synthetic"`` — what this dataset *is*. Required and
            unguessed: it determines whether the run can confer validation at all, and a
            default would eventually be wrong in the dangerous direction.
        test_size: Proportion of rows to hold out.
        seed: Seed for the split and the random baselines.
        dataset_path: Recorded in the report for traceability.

    Returns:
        dict: The report.

    Raises:
        InsufficientDataError: If the dataset cannot support a participant-grouped split.
        ValueError: If ``data_source`` is not a recognised value.
    """
    qualifier_for(data_source)  # raises on an unknown source rather than reporting one

    split = grouped_split(frame, test_size=test_size, seed=seed)
    results = evaluate_split(bundle, split, seed=seed)

    label_counts = frame[LABEL_COLUMN].value_counts().to_dict()
    dataset_block = {
        "data_source": data_source,
        "path": dataset_path,
        "n_records": int(len(frame)),
        "n_participants": int(frame["participant_id"].nunique()),
        "label_counts": {CLASS_NAMES[int(k)]: int(v) for k, v in sorted(label_counts.items())},
    }

    report: dict[str, Any] = {
        "report_id": f"{bundle.model_version}__{data_source}__seed{seed}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": bundle.model_version,
        "model_trained_on": bundle.data_source,
        "feature_set": bundle.feature_set,
        "dataset": dataset_block,
        "split_strategy": "participant-grouped",
        "seed": seed,
        **results,
        # Repeated at the top level so a reader who scrolls no further still gets it.
        "metrics_data_source": data_source,
    }
    report["warnings"] = _collect_warnings(report)
    report["validation_status"] = status_for(bundle.metadata, report)
    report["validation_explanation"] = explain(report["validation_status"])
    if bundle.data_source == "synthetic":
        report["model_notice"] = SYNTHETIC_MODEL_NOTICE

    logger.info(
        "Evaluated %s on %s data: %s %.4f (%s)",
        bundle.model_version, data_source, HEADLINE_METRIC,
        report["headline_score"], report["validation_status"],
    )
    return report


def _collect_warnings(report: dict[str, Any]) -> list[str]:
    """
    Gather every reason this report should be read cautiously.

    Kept in the persisted JSON rather than printed and forgotten: a report read six months
    later must carry its own caveats, because the person reading it will not have the
    context that produced it.
    """
    warnings: list[str] = list(report["split"]["warnings"])

    if report["dataset"]["data_source"] != "real":
        warnings.append(
            "This evaluation used synthetic data, so it measures how separable the "
            "generator's authored classes are. It is not evidence about people and "
            "cannot validate the model."
        )
    if report["dataset"]["n_participants"] < 10:
        warnings.append(
            f"Only {report['dataset']['n_participants']} participant(s) in the dataset; "
            "typing style varies enormously between people, so this measures those "
            "individuals more than the hypothesis."
        )
    if not report["beats_all_baselines"]:
        losers = [
            name for name, score in report["baseline_headline_scores"].items()
            if score >= report["headline_score"]
        ]
        warnings.append(
            "The model did not beat these trivial baselines: " + ", ".join(losers)
            + ". On this evidence it has demonstrated no useful skill."
        )
    if report["never_predicted"]:
        warnings.append(
            "The model never predicted: " + ", ".join(report["never_predicted"])
            + ". A class it cannot produce is a class it cannot detect."
        )
    if abs(report["calibration"]["confidence_minus_accuracy"]) >= 0.1:
        warnings.append(
            "Confidence is not calibrated: the model is "
            + report["calibration"]["verdict"] + "."
        )
    return warnings


def save_report(report: dict[str, Any], report_dir: Path = DEFAULT_REPORT_DIR) -> Path:
    """
    Persist a report, one file per model version and dataset.

    Parameters:
        report: A :func:`evaluate` result.
        report_dir: Destination directory.

    Returns:
        Path: The file written.
    """
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['report_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", path)
    return path


def format_report(report: dict[str, Any]) -> str:
    """
    Render a report for a terminal.

    Structured so the two things most easily forgotten come first and last: what data
    this was measured on, and whether the model beat a coin flip.

    Parameters:
        report: A :func:`evaluate` result.

    Returns:
        str: The formatted report.
    """
    source = report["dataset"]["data_source"]
    lines = [
        "=" * 78,
        f"MODEL EVALUATION - measured on {source.upper()} DATA",
        "=" * 78,
        f"Model:        {report['model_version']} (trained on {report['model_trained_on']} data)",
        f"Dataset:      {report['dataset']['n_records']} sessions from "
        f"{report['dataset']['n_participants']} participant(s)",
        f"Split:        {report['split_strategy']} "
        f"({report['split']['n_train']} train / {report['split']['n_test']} test rows, "
        f"seed {report['seed']})",
        f"Status:       {report['validation_status']}",
        "",
        report["validation_explanation"],
        "",
        "-" * 78,
        f"MODEL vs TRIVIAL BASELINES ({report['headline_metric']}, {qualifier_for(source)})",
        "-" * 78,
        f"  {'model':<14} {report['headline_score']:.4f}",
    ]
    for name, score in report["baseline_headline_scores"].items():
        lines.append(f"  {name:<14} {score:.4f}   ({BASELINE_DESCRIPTIONS[name]})")

    lines.extend([
        "",
        ("  => The model beat every trivial baseline on this data."
         if report["beats_all_baselines"] else
         "  => The model did NOT beat every trivial baseline. It has demonstrated no "
         "useful skill on this data."),
        "",
        "-" * 78,
        f"PER-CLASS PERFORMANCE ({qualifier_for(source)})",
        "-" * 78,
        # metrics-ok: column widths in a table whose heading carries the qualifier
        f"  {'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} "
        f"{'support':>9} {'predicted':>10}",
    ])
    for name, stats in report["model"]["per_class"].items():
        # metrics-ok: rows under the PER-CLASS heading, which states the data source
        lines.append(
            f"  {name:<10} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
            f"{stats['f1']:>10.4f} {stats['support']:>9d} {stats['predicted_count']:>10d}"
        )

    lines.extend([
        "",
        f"  Confusion matrix ({source} data), rows = actual, columns = predicted:",
    ])
    for name, row in zip(CLASS_NAMES, report["model"]["confusion_matrix"]):
        lines.append(f"    {name:<8} " + " ".join(f"{value:5d}" for value in row))

    calibration = report["calibration"]
    lines.extend([
        "",
        "-" * 78,
        f"CALIBRATION ({qualifier_for(source)})",
        "-" * 78,
        f"  Verdict: {calibration['verdict']}",
        format_metric("  Expected calibration error",
                      calibration["expected_calibration_error"], source),
        format_metric("  Brier score               ", calibration["brier_score"], source),
        "",
        # metrics-ok: column widths under the CALIBRATION heading, which is qualified
        f"  {'confidence bin':<18} {'n':>6} {'mean conf':>11} {'accuracy':>10}",
    ])
    for row in calibration["reliability_table"]:
        if not row["count"]:
            continue
        lines.append(
            f"  {row['bin_lower']:.1f} - {row['bin_upper']:.1f}       "
            f"{row['count']:>6d} {row['mean_confidence']:>11.4f} {row['accuracy']:>10.4f}"
        )

    if report["warnings"]:
        lines.extend(["", "=" * 78, "READ THIS BEFORE QUOTING ANY NUMBER ABOVE:"])
        lines.extend(f"  - {warning}" for warning in report["warnings"])
        lines.append("=" * 78)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint: evaluate the current model against a labelled dataset.

    ``--data-source`` is required rather than inferred from the filename. Inferring it
    would mean a mislabelled file could silently promote a model to "validated", which is
    the one mistake this harness exists to make impossible.

    Parameters:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        int: ``0`` when a report was produced — including one that says the model has no
        skill, which is a successful evaluation with a negative result. ``1`` only when
        the evaluation could not be run at all.
    """
    import argparse

    from ..config import load_settings
    from ..core.model import ModelRegistry, ModelUnavailableError

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="keystress-evaluate",
        description="Evaluate a model against a labelled dataset, with baselines (F5).",
    )
    parser.add_argument("--dataset", type=Path,
                        default=Path("data") / "labelled_sessions.csv",
                        help="Labelled dataset from `keystress-export`.")
    parser.add_argument("--data-source", required=True, choices=("real", "synthetic"),
                        help="What this dataset is. Only 'real' can confer validation.")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Proportion of rows held out.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split and baselines.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
                        help="Where to write the report.")
    parser.add_argument("--no-save", action="store_true",
                        help="Print the report without persisting it.")
    args = parser.parse_args(argv)

    settings = load_settings()
    registry = ModelRegistry()
    try:
        bundle = registry.load(
            settings.model_path, settings.scaler_path, settings.metadata_path
        )
    except ModelUnavailableError as exc:
        # `load_bundle` already normalises every artifact failure — missing, corrupt,
        # unpicklable — into this one exception with a message that says what to do.
        logger.error("Could not load a model to evaluate: %s", exc)
        return 1

    try:
        frame = load_labelled_dataset(args.dataset)
        report = evaluate(
            bundle, frame,
            data_source=args.data_source,
            test_size=args.test_size,
            seed=args.seed,
            dataset_path=str(args.dataset),
        )
    except (FileNotFoundError, InsufficientDataError, ValueError) as exc:
        logger.error("Evaluation could not run: %s", exc)
        return 1

    print(format_report(report))

    if not args.no_save:
        path = save_report(report, args.report_dir)
        print(f"\nReport written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
