#!/usr/bin/env python3
"""
Metric-qualifier check (F1).

Fails the build if a performance metric appears anywhere in the repository without a
data-source qualifier nearby.

Why
---
``docs/CLAUDE.md`` HARD RULE 3 forbids publishing an unqualified accuracy figure. The
project's biggest credibility risk is a number like "~90% accuracy" that reads as
real-world performance while actually measuring how separable a hand-authored synthetic
generator made its own classes. Author discipline is not a control; this is.

What counts as a violation
--------------------------
A line containing a metric keyword (accuracy, precision, recall, f1, auc, ...) *and* a
metric-shaped number (a percentage, or a bare decimal/integer), where no data-source
qualifier (synthetic, real, data_source, ...) appears in the surrounding context window.

Deliberately narrow: it flags numbers next to metric words, not every number. A checker
that cries wolf gets switched off, and a switched-off checker protects nothing.

Escape hatch
------------
Append ``metrics-ok: <reason>`` on the line, or on the line above it, to record a
reviewed exception. The reason is mandatory - a bare marker is itself a violation, so
silencing the check always leaves a written justification behind.

Usage
-----
    python tools/check_metric_qualifiers.py            # scan the repository
    python tools/check_metric_qualifiers.py path ...   # scan specific paths

Exit codes: 0 clean, 1 violations found.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

#: Words that mean "this is a performance number".
#:
#: ``f1`` is matched only in its scored forms (``f1-score``, ``f1_score``, ``f1 score``).
#: A bare ``F1`` in this project is a *feature identifier* from ``docs/FEATURES.md``, not
#: a metric, and matching it would flood the output with false positives.
METRIC_KEYWORDS: tuple[str, ...] = (
    "accuracy", "accurate",
    "precision", "recall",
    "f1-score", "f1_score", "f1 score",
    "auc", "roc",
    "confidence",
    "specificity", "sensitivity",
)

#: Tokens that satisfy the qualifier requirement.
QUALIFIER_TOKENS: tuple[str, ...] = (
    "synthetic",
    "real data", "real validated", "real-world data",
    "data_source", "data source",
    "hand-authored", "generator",
    "uncalibrated",
    "not validated", "unvalidated",
    "no real-world", "never been tested",
)

#: Lines matching these are structurally incapable of publishing a claim.
IGNORE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(#|//|\*|<!--)?\s*metrics-ok:", re.IGNORECASE),
)

#: Numbers carrying a CSS or typographic unit are layout values, not claims.
_UNIT_SUFFIX = r"(?!\s*(?:rem|px|em|ex|ch|vh|vw|vmin|vmax|pt|pc|cm|mm|in|deg|s\b|ms\b))"

#: Unambiguous metric shapes: a percentage, or a decimal in the 0-1 range.
METRIC_NUMBER = re.compile(
    r"(?<![\w.-])"
    r"(?:"
    r"\d{1,3}(?:\.\d+)?\s*%"           # 90%, 85.5 %
    r"|0?\.\d+"                         # 0.9013, .90
    r")"
    + _UNIT_SUFFIX
)

#: A bare integer is only a metric when it is plausibly a score (0-100) *and* sits right
#: next to the metric word. Without this restriction the checker reads the line numbers in
#: documentation tables ("| app.py | 715 | ... confidence ... |") as accuracy figures.
BARE_SCORE = re.compile(r"(?<![\w.%-])(\d{1,3})(?![\w.%])" + _UNIT_SUFFIX)

#: Maximum characters between a metric word and a bare integer for it to count as that
#: metric's value.
BARE_SCORE_PROXIMITY: int = 12

#: Files worth scanning.
SCANNED_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".md", ".txt", ".html", ".js", ".json", ".yml", ".yaml", ".toml", ".cfg", ".rst",
})

#: Directories never scanned.
EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env",
    "node_modules", "build", "dist", ".mypy_cache", ".ruff_cache",
    "models", "data", ".eggs", "htmlcov",
})

#: Files never scanned. The checker's own source and its tests necessarily contain
#: unqualified examples — that is precisely what they assert against — so scanning them
#: would make the tool flag itself.
EXCLUDED_FILES: frozenset[str] = frozenset({
    "check_metric_qualifiers.py",
    "test_metric_qualifiers.py",
})

#: Path fragments never scanned (fixtures holding deliberately-bad samples).
EXCLUDED_PATH_PARTS: tuple[str, ...] = (
    "tests/fixtures/metric_check",
)

#: How many lines either side of a hit may supply the qualifier. A qualifier in the same
#: sentence or the adjacent table row counts; one three paragraphs away does not.
CONTEXT_LINES: int = 2


@dataclass(frozen=True)
class Violation:
    """A metric published without a data-source qualifier."""

    path: Path
    line_number: int
    line: str
    keyword: str

    def render(self, root: Path) -> str:
        """Format the violation as an editor-clickable message."""
        try:
            shown = self.path.relative_to(root)
        except ValueError:
            shown = self.path
        return (
            f"{shown}:{self.line_number}: metric '{self.keyword}' has a number but no "
            f"data-source qualifier\n"
            f"    {self.line.strip()}"
        )


# --------------------------------------------------------------------------------------
# Core checks
# --------------------------------------------------------------------------------------


def find_metric_keyword(line: str) -> tuple[str, int, int] | None:
    """
    Return the metric keyword present on a line, with its position.

    Parameters:
        line: A single line of text.

    Returns:
        ``(keyword, start, end)`` for the first match, or ``None``.
    """
    lowered = line.lower()
    for keyword in METRIC_KEYWORDS:
        match = re.search(rf"(?<![\w-]){re.escape(keyword)}(?![\w-])", lowered)
        if match:
            return keyword, match.start(), match.end()
    return None


def has_metric_number(line: str) -> bool:
    """
    Report whether a line contains an unambiguously metric-shaped number.

    Only percentages and 0-1 decimals qualify. Bare integers are handled separately by
    :func:`has_adjacent_score`, because on their own they are far more often line numbers,
    sample counts, or version numbers than they are claims.

    Parameters:
        line: A single line of text.

    Returns:
        bool: ``True`` if a percentage or 0-1 decimal is present.
    """
    return METRIC_NUMBER.search(line) is not None


def has_adjacent_score(line: str, keyword_start: int, keyword_end: int) -> bool:
    """
    Report whether a plausible score sits immediately beside the metric word.

    Parameters:
        line: A single line of text.
        keyword_start: Start offset of the metric keyword.
        keyword_end: End offset of the metric keyword.

    Returns:
        bool: ``True`` if an integer in 0-100 appears within
        :data:`BARE_SCORE_PROXIMITY` characters of the keyword.
    """
    for match in BARE_SCORE.finditer(line):
        if not 0 <= int(match.group(1)) <= 100:
            continue
        distance = (
            keyword_start - match.end() if match.end() <= keyword_start
            else match.start() - keyword_end
        )
        if 0 <= distance <= BARE_SCORE_PROXIMITY:
            return True
    return False


def has_qualifier(context: str) -> bool:
    """
    Report whether a context window contains a data-source qualifier.

    Parameters:
        context: Joined lines around a candidate line.

    Returns:
        bool: ``True`` if any qualifier token appears.
    """
    lowered = context.lower()
    return any(token in lowered for token in QUALIFIER_TOKENS)


def is_ignored_line(line: str) -> bool:
    """
    Report whether a line carries a reviewed-exception marker.

    A marker with no stated reason does not count, so the escape hatch cannot be used
    to silence the check anonymously.

    Parameters:
        line: A single line of text.

    Returns:
        bool: ``True`` if the line is an accepted exception.
    """
    match = re.search(r"metrics-ok:\s*(\S.*)$", line, re.IGNORECASE)
    return bool(match and match.group(1).strip())


def check_text(text: str, path: Path) -> list[Violation]:
    """
    Find unqualified metrics in a block of text.

    Parameters:
        text: Full file contents.
        path: Path used for reporting.

    Returns:
        list[Violation]: Violations in line order.
    """
    lines = text.splitlines()
    violations: list[Violation] = []

    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in IGNORE_LINE_PATTERNS):
            continue
        if is_ignored_line(line):
            continue
        # An exception marker on the preceding line covers this line too.
        if index > 0 and is_ignored_line(lines[index - 1]):
            continue

        found = find_metric_keyword(line)
        if found is None:
            continue
        keyword, keyword_start, keyword_end = found
        if not has_metric_number(line) and not has_adjacent_score(line, keyword_start, keyword_end):
            continue

        start = max(0, index - CONTEXT_LINES)
        end = min(len(lines), index + CONTEXT_LINES + 1)
        if has_qualifier("\n".join(lines[start:end])):
            continue

        violations.append(
            Violation(path=path, line_number=index + 1, line=line, keyword=keyword)
        )

    return violations


def check_file(path: Path) -> list[Violation]:
    """
    Scan a single file.

    Parameters:
        path: File to scan.

    Returns:
        list[Violation]: Violations found; empty if the file is unreadable as text.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return check_text(text, path)


def iter_files(roots: Sequence[Path]) -> Iterable[Path]:
    """
    Yield every scannable file under the given roots.

    Parameters:
        roots: Files or directories to walk.

    Yields:
        Path: Files eligible for scanning.
    """
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in SCANNED_SUFFIXES and root.name not in EXCLUDED_FILES:
                yield root
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SCANNED_SUFFIXES:
                continue
            if path.name in EXCLUDED_FILES:
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            posix = path.as_posix()
            if any(fragment in posix for fragment in EXCLUDED_PATH_PARTS):
                continue
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the check.

    Parameters:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 when clean, 1 when violations were found.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "paths", nargs="*", type=Path, default=None,
        help="Files or directories to scan (default: repository root)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    roots = args.paths if args.paths else [repo_root]

    violations: list[Violation] = []
    for path in iter_files(roots):
        violations.extend(check_file(path))

    if not violations:
        print("Metric-qualifier check passed: no unqualified metrics found.")
        return 0

    print(f"Metric-qualifier check FAILED: {len(violations)} unqualified metric(s).\n")
    for violation in violations:
        print(violation.render(repo_root))
        print()
    print(
        "Every published metric must state its data source (HARD RULE 3).\n"
        "Fix by naming the source near the number - for example 'on synthetic data' -\n"
        "or, if this is a false positive, append 'metrics-ok: <reason>' to the line."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
