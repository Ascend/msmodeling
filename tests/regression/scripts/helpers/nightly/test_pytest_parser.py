"""Tests for nightly.pytest_parser — parse_junit_xml, NightlyRunStats."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from scripts.helpers.nightly.pytest_parser import (
    NightlyRunStats,
    extract_pytest_log_snippet,
    parse_junit_file,
    parse_junit_xml,
    slowest_testcases,
)
from tests.helpers.junit_xml import (
    SAMPLE_MULTI_CASE_JUNIT,
    TIMED_RANKING_JUNIT,
    write_junit_xml_content,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_sample_junit(tmp_path: Path) -> Path:
    return write_junit_xml_content(tmp_path / "sample.xml", SAMPLE_MULTI_CASE_JUNIT)


# ---------------------------------------------------------------------------
# parse_junit_xml — happy path
# ---------------------------------------------------------------------------


def test_parse_junit_xml_extracts_all_counts(tmp_path: Path) -> None:
    stats = parse_junit_xml((_write_sample_junit(tmp_path),))
    assert stats.passed == 3
    assert stats.failed == 2
    assert stats.errors == 1


def test_parse_junit_xml_extracts_duration(tmp_path: Path) -> None:
    stats = parse_junit_xml((_write_sample_junit(tmp_path),))
    assert stats.duration_sec == pytest.approx(4.3)


def test_parse_junit_xml_extracts_failed_cases(tmp_path: Path) -> None:
    stats = parse_junit_xml((_write_sample_junit(tmp_path),))
    assert len(stats.failed_cases) == 2
    assert "tests/smoke/test_fail.py::test_broken" in stats.failed_cases
    assert "tests/smoke/test_fail.py::test_other" in stats.failed_cases
    assert isinstance(stats.failed_cases, tuple)


def test_parse_junit_xml_extracts_first_error(tmp_path: Path) -> None:
    stats = parse_junit_xml((_write_sample_junit(tmp_path),))
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


def test_parse_testsuite_level_error_counts_as_error(tmp_path: Path) -> None:
    path = tmp_path / "suite_error.xml"
    path.write_text(
        """\
<testsuite errors="1">
  <error message="ValueError: collection failed">E   ValueError: collection failed</error>
</testsuite>
""",
        encoding="utf-8",
    )
    stats = parse_junit_xml((path,))
    assert stats.errors == 1
    assert stats.passed == 0
    assert "collection failed" in stats.first_error


def test_extract_pytest_log_snippet_reads_short_traceback_line(tmp_path: Path) -> None:
    log_path = tmp_path / "phase1.log"
    log_path.write_text(
        "collecting ...\nE   ValueError: 'deepseek_v4' is already used\n",
        encoding="utf-8",
    )
    snippet = extract_pytest_log_snippet(log_path)
    assert "deepseek_v4" in snippet


def test_extract_pytest_log_snippet_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    assert extract_pytest_log_snippet(tmp_path / "missing.log") == ""


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
# parse_junit_file — single file
# ---------------------------------------------------------------------------


def test_parse_junit_file_returns_stats_for_valid_file(tmp_path: Path) -> None:
    stats = parse_junit_file(_write_sample_junit(tmp_path))
    assert stats is not None
    assert stats.passed == 3
    assert stats.failed == 2
    assert stats.errors == 1


def test_parse_junit_file_returns_none_for_missing_file() -> None:
    assert parse_junit_file(Path("/nonexistent/missing.xml")) is None


# ---------------------------------------------------------------------------
# slowest_testcases
# ---------------------------------------------------------------------------


def _write_timed_junit(tmp_path: Path) -> Path:
    return write_junit_xml_content(tmp_path / "timed.xml", TIMED_RANKING_JUNIT)


def test_slowest_testcases_sorted_descending(tmp_path: Path) -> None:
    slowest = slowest_testcases((_write_timed_junit(tmp_path),))
    seconds = [s for _node, s in slowest]
    assert seconds == sorted(seconds, reverse=True)
    assert slowest[0] == ("pkg/mod.py::test_slow", pytest.approx(9.0))


def test_slowest_testcases_respects_top_n(tmp_path: Path) -> None:
    slowest = slowest_testcases((_write_timed_junit(tmp_path),), top_n=2)
    assert len(slowest) == 2
    assert [s for _node, s in slowest] == [pytest.approx(9.0), pytest.approx(3.0)]


def test_slowest_testcases_skips_missing_files(tmp_path: Path) -> None:
    slowest = slowest_testcases((Path("/nonexistent/a.xml"), _write_timed_junit(tmp_path)))
    assert len(slowest) == 3


def test_slowest_testcases_skips_testcases_without_time_attr(tmp_path: Path) -> None:
    path = tmp_path / "untimed.xml"
    path.write_text(
        """\
<testsuite>
  <testcase classname="pkg.mod" name="test_timed" time="2.0"/>
  <testcase classname="pkg.mod" name="test_untimed"/>
</testsuite>
""",
        encoding="utf-8",
    )
    slowest = slowest_testcases((path,))
    assert slowest == (("pkg/mod.py::test_timed", pytest.approx(2.0)),)


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
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.passed = 2
