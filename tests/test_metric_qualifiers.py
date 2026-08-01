"""
Tests for the metric-qualifier check (F1).

The point of these tests is that the checker must be *able to fail*. A guard that always
passes is indistinguishable from no guard at all, so the first thing asserted here is
that a known-bad fixture is rejected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_metric_qualifiers import (  # noqa: E402
    check_file,
    check_text,
    has_metric_number,
    has_qualifier,
    is_ignored_line,
    iter_files,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "metric_check"


class TestCheckerCanFail:
    """The guard must reject unqualified metrics."""

    def test_bad_fixture_produces_violations(self) -> None:
        violations = check_file(FIXTURES / "bad_unqualified.md")
        assert violations, "checker failed to flag the deliberately-bad fixture"

    def test_bad_fixture_flags_every_unqualified_line(self) -> None:
        violations = check_file(FIXTURES / "bad_unqualified.md")
        flagged = {v.line_number for v in violations}
        # Four distinct unqualified claims live in the fixture.
        assert len(flagged) >= 4, f"expected >= 4 violations, got {len(flagged)}"

    def test_good_fixture_is_clean(self) -> None:
        violations = check_file(FIXTURES / "good_qualified.md")
        assert violations == [], (
            "checker flagged properly qualified metrics: "
            + "; ".join(v.render(REPO_ROOT) for v in violations)
        )


class TestQualifierDetection:
    """Qualifier recognition."""

    @pytest.mark.parametrize("text", [
        "Accuracy: 90% on synthetic data",
        "accuracy 0.90 (data_source: synthetic)",
        "90% accuracy measured on real validated data",
        "confidence 85%, uncalibrated",
    ])
    def test_qualified_metrics_pass(self, text: str) -> None:
        assert check_text(text, Path("x.md")) == []

    @pytest.mark.parametrize("text", [
        "Accuracy: 90%",
        "The model achieves 0.90 precision.",
        "| F1-Score | ~90% |",
        "Confidence: 85%",
    ])
    def test_unqualified_metrics_fail(self, text: str) -> None:
        assert check_text(text, Path("x.md")), f"missed unqualified metric: {text!r}"

    def test_qualifier_found_in_nearby_lines(self) -> None:
        text = "All figures below were measured on synthetic data.\n\nAccuracy: 90%"
        assert check_text(text, Path("x.md")) == []

    def test_qualifier_too_far_away_does_not_count(self) -> None:
        text = "Measured on synthetic data.\n\n\n\n\nAccuracy: 90%"
        assert check_text(text, Path("x.md")), "distant qualifier should not satisfy the check"

    def test_has_qualifier_is_case_insensitive(self) -> None:
        assert has_qualifier("On SYNTHETIC Data")

    def test_bare_integer_score_next_to_the_keyword_is_flagged(self) -> None:
        """`accuracy 90` is a claim even though 90 is not a percentage or decimal."""
        assert check_text("accuracy 90", Path("x.md"))

    def test_bare_integer_far_from_the_keyword_is_ignored(self) -> None:
        """Line numbers and the like must not be read as scores."""
        assert check_text("the line number 90 contains the accuracy table",
                          Path("x.md")) == []


class TestFalsePositiveResistance:
    """A noisy checker gets disabled, so layout values must not trip it."""

    @pytest.mark.parametrize("line", [
        ".result-confidence { font-size: 0.9rem; }",
        ".result-confidence { margin-top: 12px; }",
        ".confidence-bar { transition: width 0.5s ease; }",
    ])
    def test_css_units_are_not_metrics(self, line: str) -> None:
        assert check_text(line, Path("x.css")) == [], f"CSS value flagged: {line!r}"

    def test_metric_word_without_number_is_ignored(self) -> None:
        assert check_text("We report accuracy in the eval report.", Path("x.md")) == []

    def test_number_without_metric_word_is_ignored(self) -> None:
        assert check_text("There are 1500 samples in the dataset.", Path("x.md")) == []

    def test_has_metric_number_rejects_unit_values(self) -> None:
        assert not has_metric_number("font-size: 0.9rem")
        assert has_metric_number("accuracy 0.90")

    @pytest.mark.parametrize("line", [
        "measured cost of 0.371s, and the confidence figure stops wobbling",
        "the fit took 1.250s so confidence in the timing is high",
        "transition: width 0.5s ease;  /* .result-confidence */",
    ])
    def test_multi_digit_decimals_with_units_are_not_metrics(self, line: str) -> None:
        """
        Regression: the regex used to backtrack around the unit guard.

        Given "0.371s" the engine matched "0.371", saw the forbidden "s", then gave back a
        digit to match "0.37" - whose next character is "1", not a unit - and reported a
        metric. Anchoring the fraction so it cannot be shortened fixes it.
        """
        assert check_text(line, Path("x.md")) == [], f"unit value flagged: {line!r}"

    def test_genuine_decimals_are_still_caught(self) -> None:
        """The fix must not blind the checker to real unqualified metrics."""
        assert check_text("accuracy 0.9013", Path("x.md"))
        assert check_text("recall was 0.87 overall", Path("x.md"))


class TestExceptionMarker:
    """The escape hatch must require a written reason."""

    def test_marker_with_reason_silences_the_line(self) -> None:
        text = "Accuracy: 90%  metrics-ok: threshold config, not a claim"
        assert check_text(text, Path("x.md")) == []

    def test_marker_without_reason_does_not_silence(self) -> None:
        text = "Accuracy: 90%  metrics-ok:"
        assert check_text(text, Path("x.md")), "bare marker must not silence the check"

    def test_marker_on_preceding_line_covers_next_line(self) -> None:
        text = "# metrics-ok: fixture value used by the calibration test\nAccuracy: 90%"
        assert check_text(text, Path("x.py")) == []

    def test_is_ignored_line_requires_reason(self) -> None:
        assert is_ignored_line("x  metrics-ok: because reasons")
        assert not is_ignored_line("x  metrics-ok:   ")


class TestRepositoryIsClean:
    """The live repository must satisfy its own rule (F1 acceptance criterion)."""

    def test_repository_has_no_unqualified_metrics(self) -> None:
        assert main([str(REPO_ROOT)]) == 0, (
            "the repository contains an unqualified metric; "
            "run `python tools/check_metric_qualifiers.py` for details"
        )

    def test_cli_exits_nonzero_on_bad_input(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "check_metric_qualifiers.py"),
             str(FIXTURES / "bad_unqualified.md")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1, "CLI must exit 1 on violations"
        assert "FAILED" in result.stdout


class TestFileScanning:
    """Walk logic: exclusions, suffixes, and unreadable files."""

    def test_iter_files_respects_exclusions(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        (tmp_path / "b.md").write_text("x", encoding="utf-8")
        (tmp_path / "c.csv").write_text("x", encoding="utf-8")
        (tmp_path / "check_metric_qualifiers.py").write_text("x", encoding="utf-8")

        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "d.py").write_text("x", encoding="utf-8")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "e.py").write_text("x", encoding="utf-8")

        found = sorted(p.name for p in iter_files([tmp_path]))
        assert found == ["a.py", "b.md"], (
            "expected only scannable, non-excluded files, got " + repr(found)
        )

    def test_check_file_is_blank_on_binary_content(self, tmp_path) -> None:
        """A file that cannot be decoded as text contributes no violations."""
        path = tmp_path / "binary.md"
        path.write_bytes(b"\xff\xfe\x00\x80\x81\xff")
        assert check_file(path) == []

    def test_check_text_on_empty_string(self) -> None:
        assert check_text("", Path("x.md")) == []
