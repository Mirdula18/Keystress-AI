"""
The labelled dataset (F4): schema, export, and a description of what came out.

This is the artifact the whole research harness exists to produce — one row per typing
session that has a real, self-reported label attached. It is what F5 evaluates against,
and it is the only thing that can turn "how separable did the generator make its own
classes" into a statement about people.

Schema (``labelled-v1``)
------------------------
=========================  ====================================================
``participant_id``         Anonymous UUID. **The grouping key**: F5 must keep one
                           person out of both sides of a split, so this column is
                           not optional and must not be dropped when sharing.
``donation_id``            The typing session.
``response_id``            The questionnaire that labels it.
``session_created_at``     When the typing session was donated (UTC, ISO-8601).
``response_created_at``    When the questionnaire was submitted.
``feature_set``            Feature-set version of the five feature columns.
``avg_typing_speed`` …     The five ``FEATURES_V1`` values.
``instrument_version``     Which questionnaire wording produced the label.
``personal_score``         Personal-burnout subscale, 0-100.
``studies_score``          Studies-related subscale, 0-100.
``overall_score``          Mean across items, 0-100. **The label to prefer.**
``label``                  Banded class index (0/1/2) derived from the overall
                           score. A convenience for the existing three-class
                           model, not a better measurement than the score.
=========================  ====================================================

Two columns are deliberately **not** exported
---------------------------------------------
The store also holds the synthetic model's ``prediction`` and ``data_source`` for each
donation. They stay out of the dataset on purpose: a file containing both the label and
the current model's guess at the label is one careless join away from training on its own
output, which would manufacture exactly the circularity this project is trying to escape.
They remain in the database for auditing, where being examined deliberately is the point.

Privacy
-------
Every column above is either an aggregate over timing, a questionnaire score, an opaque
identifier, or a timestamp. There is no free-text column anywhere in the schema, and no
per-keystroke data survives the feature extraction that happens before storage. A
participant who deletes their data disappears from every future export, because the export
is generated from the live store rather than accumulated in a file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Final

from ..core.disclosure import FEATURES_V1
from ..core.storage import Store
from .instrument import SUBSCALES
from .scoring import BAND_NAMES

logger = logging.getLogger(__name__)

#: Bump when a column is added, removed, or changes meaning.
DATASET_VERSION: Final[str] = "labelled-v1"

#: Columns, in export order. The identifier and provenance columns come first so a person
#: opening the CSV sees what a row *is* before they see numbers.
DATASET_COLUMNS: Final[tuple[str, ...]] = (
    "participant_id",
    "donation_id",
    "response_id",
    "session_created_at",
    "response_created_at",
    "feature_set",
    *FEATURES_V1,
    "instrument_version",
    "personal_score",
    "studies_score",
    "overall_score",
    "label",
)

#: Below this, an evaluation is not worth reporting as a number. Not a magic threshold —
#: a deliberately conservative floor that :func:`describe` warns about, so a five-row
#: dataset never quietly becomes a claim.
MIN_USEFUL_RECORDS: Final[int] = 30

#: Likewise for participants. Thirty sessions from three people is three people's typing.
MIN_USEFUL_PARTICIPANTS: Final[int] = 10


def build_rows(store: Store) -> list[dict[str, Any]]:
    """
    Flatten the store's labelled pairs into dataset rows.

    Parameters:
        store: The consent/donation store.

    Returns:
        list[dict]: One dict per row, keyed by :data:`DATASET_COLUMNS`, oldest first.
    """
    rows: list[dict[str, Any]] = []
    for record in store.labelled_records():
        features = record["features"]
        row: dict[str, Any] = {
            "participant_id": record["participant_id"],
            "donation_id": record["donation_id"],
            "response_id": record["response_id"],
            "session_created_at": record["session_created_at"],
            "response_created_at": record["response_created_at"],
            "feature_set": record["feature_set"],
            "instrument_version": record["instrument_version"],
            "personal_score": record["personal_score"],
            "studies_score": record["studies_score"],
            "overall_score": record["overall_score"],
            "label": record["label"],
        }
        # Missing features default to 0.0 rather than raising: the store already
        # guarantees the whitelist, so a gap here would mean a feature-set change, which
        # `feature_set` records and the exporter should not silently fail on.
        for name in FEATURES_V1:
            row[name] = float(features.get(name, 0.0))
        rows.append(row)
    return rows


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarise a dataset, including the reasons it might not be usable yet.

    Parameters:
        rows: Rows from :func:`build_rows`.

    Returns:
        dict: Counts, per-class and per-participant breakdowns, and ``warnings`` — a list
        of plain-language reasons this dataset should not yet be treated as evidence.
        The warnings are part of the return value rather than log output because the
        caller that writes a file should be able to write them next to it.
    """
    participants = {row["participant_id"] for row in rows}
    label_counts = Counter(row["label"] for row in rows)
    per_participant = Counter(row["participant_id"] for row in rows)

    warnings: list[str] = []
    if len(rows) < MIN_USEFUL_RECORDS:
        warnings.append(
            f"Only {len(rows)} labelled session(s); fewer than {MIN_USEFUL_RECORDS} is too "
            "few to evaluate a model on. Collect more before reporting any number."
        )
    if len(participants) < MIN_USEFUL_PARTICIPANTS:
        warnings.append(
            f"Only {len(participants)} participant(s). Typing style varies enormously "
            "between people, so a handful of participants measures those individuals, "
            "not the hypothesis."
        )
    missing_classes = [
        BAND_NAMES[label] for label in range(len(BAND_NAMES)) if label not in label_counts
    ]
    if missing_classes:
        warnings.append(
            "No sessions in class(es): " + ", ".join(missing_classes)
            + ". A model cannot be evaluated on a class that never appears."
        )
    if per_participant and max(per_participant.values()) > max(1, len(rows) // 2):
        warnings.append(
            "One participant contributes more than half the rows, so aggregate results "
            "would mostly describe that person."
        )

    return {
        "dataset_version": DATASET_VERSION,
        "n_records": len(rows),
        "n_participants": len(participants),
        "label_counts": {BAND_NAMES[label]: count for label, count in sorted(label_counts.items())},
        "sessions_per_participant": {
            "min": min(per_participant.values()) if per_participant else 0,
            "max": max(per_participant.values()) if per_participant else 0,
        },
        "instrument_versions": sorted({row["instrument_version"] for row in rows}),
        "feature_sets": sorted({row["feature_set"] for row in rows}),
        "subscales": list(SUBSCALES),
        "warnings": warnings,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """
    Write rows as CSV.

    Parameters:
        rows: Rows from :func:`build_rows`.
        path: Destination file; parent directories are created.

    Returns:
        Path: The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DATASET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    """
    Write rows as JSON Lines.

    Parameters:
        rows: Rows from :func:`build_rows`.
        path: Destination file; parent directories are created.

    Returns:
        Path: The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            ordered = {column: row[column] for column in DATASET_COLUMNS}
            handle.write(json.dumps(ordered, sort_keys=False) + "\n")
    return path


def export(store: Store, path: Path, fmt: str = "csv") -> dict[str, Any]:
    """
    Build and write the labelled dataset, returning its description.

    Parameters:
        store: The store to export from.
        path: Destination file.
        fmt: ``"csv"`` or ``"jsonl"``.

    Returns:
        dict: The :func:`describe` summary, with ``path`` added.

    Raises:
        ValueError: If the format is unknown.
    """
    if fmt not in ("csv", "jsonl"):
        raise ValueError(f"Unknown format {fmt!r}; expected 'csv' or 'jsonl'")

    rows = build_rows(store)
    written = write_csv(rows, path) if fmt == "csv" else write_jsonl(rows, path)

    summary = describe(rows)
    summary["path"] = str(written)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """
    Render a dataset description for a terminal.

    Every warning is printed. An export that is too small to evaluate should say so at
    the moment it is produced, not in a footnote nobody reads later.

    Parameters:
        summary: A :func:`describe` result.

    Returns:
        str: The formatted report.
    """
    lines = [
        "=" * 72,
        f"LABELLED DATASET ({summary['dataset_version']}) - real, consented, self-reported",
        "=" * 72,
        f"Records:            {summary['n_records']}",
        f"Participants:       {summary['n_participants']}",
        f"Instrument:         {', '.join(summary['instrument_versions']) or '-'}",
        f"Feature set:        {', '.join(summary['feature_sets']) or '-'}",
        "",
        "Label distribution:",
    ]
    for band, count in summary["label_counts"].items():
        lines.append(f"  {band:<28} {count}")

    if not summary["label_counts"]:
        lines.append("  (none)")

    if summary.get("path"):
        lines.extend(["", f"Written to: {summary['path']}"])

    if summary["warnings"]:
        lines.extend(["", "-" * 72, "THIS DATASET IS NOT YET EVIDENCE:"])
        lines.extend(f"  - {warning}" for warning in summary["warnings"])
        lines.append("-" * 72)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint: export the labelled dataset from a store.

    Parameters:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code. ``0`` even for an empty dataset — an empty export is a
        true statement about a site nobody has used yet, not a failure.
    """
    from ..config import load_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="keystress-export",
        description="Export the labelled research dataset (F4).",
    )
    parser.add_argument("--store", type=Path, default=None,
                        help="Path to the SQLite store (default: the configured store).")
    parser.add_argument("--out", type=Path, default=Path("data") / "labelled_sessions.csv",
                        help="Destination file.")
    parser.add_argument("--format", dest="fmt", choices=("csv", "jsonl"), default="csv",
                        help="Output format.")
    args = parser.parse_args(argv)

    store_path = args.store if args.store is not None else load_settings().store_path
    if not Path(store_path).exists():
        logger.error("No store at %s; nothing has been collected yet.", store_path)
        return 1

    summary = export(Store(store_path), args.out, args.fmt)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
