"""Tests for nightly.pytest_parser — parse_junit_xml, NightlyRunStats."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.helpers.nightly.pytest_parser import NightlyRunStats, parse_junit_xml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SAMPLE_JUNIT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="6" failures="2" errors="1" skipped="0" time="45.3">
    <testcase classname="tests.smoke.test_a" name="test_x" time="1.0"/>
    <testcase classname="tests.smoke.test_b" name="test_y" time="1.0"/>
    <testcase classname="tests.regression.cli.test_run" name="test_z" time="1.0"/>
    <testcase classname="tests.smoke.test_fail" name="test_broken" time="0.5">
      <failure message="AssertionError: 1 != 2">E   AssertionError: 1 != 2</failure>
    </testcase>
    <testcase classname="tests.smoke.test_fail" name="test_other" time="0.5">
      <failure message="ValueError: bad">E   ValueError: bad</failure>
    </testcase>
    <testcase classname="tests.smoke.test_err" name="test_crash" time="0.3">
      <error message="RuntimeError: boom">E   RuntimeError: boom</error>
    </testcase>
  </testsuite>
</testsuites>
"""


@pytest.fixture
def sample_junit_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.xml"
    path.write_text(_SAMPLE_JUNIT, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parse_junit_xml — happy path
# ---------------------------------------------------------------------------


def test_parse_junit_xml_extracts_all_counts(sample_junit_path: Path) -> None:
    stats = parse_junit_xml((sample_junit_path,))
    assert stats.passed == 3
    assert stats.failed == 2
    assert stats.errors == 1


def test_parse_junit_xml_extracts_duration(sample_junit_path: Path) -> None:
    stats = parse_junit_xml((sample_junit_path,))
    assert stats.duration_sec == pytest.approx(4.3)


def test_parse_junit_xml_extracts_failed_cases(sample_junit_path: Path) -> None:
    stats = parse_junit_xml((sample_junit_path,))
    assert len(stats.failed_cases) == 2
    assert "tests/smoke/test_fail.py::test_broken" in stats.failed_cases
    assert "tests/smoke/test_fail.py::test_other" in stats.failed_cases
    assert isinstance(stats.failed_cases, tuple)


def test_parse_junit_xml_extracts_first_error(sample_junit_path: Path) -> None:
    stats = parse_junit_xml((sample_junit_path,))
    assert "AssertionError: 1 != 2" in stats.first_error


def test_parse_junit_xml_sums_duration_across_files(tmp_path: Path) -> None:
    first = tmp_path / "phase1.xml"
    second = tmp_path / "phase2.xml"
    first.write_text(
        '<testsuite><testcase classname="pkg.mod" name="test_a" time="10.0"/></testsuite>',
        encoding="utf-8",
    )
    second.write_text(
        '<testsuite><testcase classname="pkg.mod" name="test_b" time="5.5"/></testsuite>',
        encoding="utf-8",
    )

    stats = parse_junit_xml((first, second))
    assert stats.passed == 2
    assert stats.duration_sec == pytest.approx(15.5)


# ---------------------------------------------------------------------------
# parse_junit_xml — edge cases
# ---------------------------------------------------------------------------


def test_parse_missing_paths_returns_all_zeros() -> None:
    stats = parse_junit_xml((Path("/nonexistent/a.xml"), Path("/nonexistent/b.xml")))
    assert stats.passed == 0
    assert stats.failed == 0
    assert stats.errors == 0
    assert stats.duration_sec == -1.0
    assert stats.failed_cases == ()
    assert stats.first_error == ""


def test_parse_empty_path_list_returns_all_zeros() -> None:
    stats = parse_junit_xml(())
    assert stats.passed == 0
    assert stats.failed == 0
    assert stats.errors == 0
    assert stats.duration_sec == -1.0


def test_parse_skipped_testcase_not_counted_as_passed(tmp_path: Path) -> None:
    path = tmp_path / "skipped.xml"
    path.write_text(
        """\
<testsuite>
  <testcase classname="pkg.mod" name="test_skip" time="0.1">
    <skipped message="not now"/>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )
    stats = parse_junit_xml((path,))
    assert stats.passed == 0
    assert stats.failed == 0
    assert stats.errors == 0


# ---------------------------------------------------------------------------
# NightlyRunStats immutability
# ---------------------------------------------------------------------------


def test_nightly_run_stats_is_frozen() -> None:
    stats = NightlyRunStats(
        passed=1,
        failed=0,
        errors=0,
        duration_sec=1.0,
        failed_cases=(),
        first_error="",
    )
    with pytest.raises(Exception):
        stats.passed = 2  # type: ignore[misc]
