"""Tests for nightly.pytest_parser — parse_pytest_stdout, NightlyRunStats."""

from __future__ import annotations

import dataclasses

import pytest

from scripts.helpers.nightly.pytest_parser import (
    NightlyRunStats,
    normalize_stdout_node_id,
    parse_pytest_stdout,
)

_REAL_UNITTEST_FAILED_LINE = (
    "tests/regression/cli/test_throughput_optimizer/"
    "TestThroughputOptimizerNightly.py::test_vl_model_aggregation_with_output_validation"
)
_REAL_UNITTEST_NODE_ID = (
    "tests/regression/cli/test_throughput_optimizer.py::"
    "TestThroughputOptimizerNightly::test_vl_model_aggregation_with_output_validation"
)
_REAL_PLAIN_FUNCTION_FAILED_LINE = "tests/smoke/test_fail.py::test_broken"


def test_normalize_stdout_node_id_rewrites_unittest_class_path() -> None:
    assert normalize_stdout_node_id(_REAL_UNITTEST_FAILED_LINE) == _REAL_UNITTEST_NODE_ID


def test_normalize_stdout_node_id_leaves_plain_function_unchanged() -> None:
    assert normalize_stdout_node_id(_REAL_PLAIN_FUNCTION_FAILED_LINE) == _REAL_PLAIN_FUNCTION_FAILED_LINE


def test_normalize_stdout_node_id_leaves_canonical_class_node_unchanged() -> None:
    assert normalize_stdout_node_id(_REAL_UNITTEST_NODE_ID) == _REAL_UNITTEST_NODE_ID


def test_normalize_stdout_node_id_preserves_parametrize_suffix() -> None:
    raw = (
        "tests/regression/cli/test_throughput_optimizer/"
        "TestThroughputOptimizerNightly.py::test_vl_model_aggregation_with_output_validation[case0]"
    )
    expected = (
        "tests/regression/cli/test_throughput_optimizer.py::"
        "TestThroughputOptimizerNightly::test_vl_model_aggregation_with_output_validation[case0]"
    )
    assert normalize_stdout_node_id(raw) == expected


def test_parse_pytest_stdout_extracts_failed_and_error_nodes() -> None:
    stdout = """\
tests/smoke/test_fail.py::test_broken FAILED                                  [ 33%]
tests/smoke/test_err.py::test_crash ERROR                                     [ 66%]
tests/smoke/test_ok.py::test_pass PASSED                                       [100%]

========================= 1 failed, 1 error, 1 passed in 4.30s =========================
"""
    stats = parse_pytest_stdout(stdout, exit_code=1)
    assert stats.passed == 1
    assert stats.failed == 1
    assert stats.errors == 1
    assert stats.duration_sec == pytest.approx(4.3)
    assert stats.failed_cases == (
        "tests/smoke/test_fail.py::test_broken",
        "tests/smoke/test_err.py::test_crash",
    )


def test_parse_pytest_stdout_normalizes_unittest_class_failed_line() -> None:
    stdout = (
        f"{_REAL_UNITTEST_FAILED_LINE} FAILED                                  [ 12%]\n"
        "========================= 1 failed in 4.30s =========================\n"
    )
    stats = parse_pytest_stdout(stdout, exit_code=1)
    assert stats.failed_cases == (_REAL_UNITTEST_NODE_ID,)


def test_parse_pytest_stdout_extracts_xdist_failed_nodes() -> None:
    stdout = """\
[gw0] [100%] FAILED tests/smoke/test_fail.py::test_broken
========================= 1 failed in 4.30s =========================
"""
    stats = parse_pytest_stdout(stdout, exit_code=1)
    assert stats.failed_cases == ("tests/smoke/test_fail.py::test_broken",)


def test_parse_pytest_stdout_xdist_normalizes_unittest_class_failed_line() -> None:
    stdout = (
        f"[gw1] [ 42%] FAILED {_REAL_UNITTEST_FAILED_LINE}\n"
        "========================= 1 failed in 4.30s =========================\n"
    )
    stats = parse_pytest_stdout(stdout, exit_code=1)
    assert stats.failed_cases == (_REAL_UNITTEST_NODE_ID,)


def test_parse_pytest_stdout_infra_failure_sets_first_error() -> None:
    stdout = "collecting ...\nE   ValueError: 'deepseek_v4' is already used\n"
    stats = parse_pytest_stdout(stdout, exit_code=2)
    assert stats.failed_cases == ()
    assert "deepseek_v4" in stats.first_error


def test_nightly_run_stats_is_frozen() -> None:
    stats = NightlyRunStats(
        passed=1,
        failed=0,
        errors=0,
        duration_sec=1.0,
        failed_cases=(),
        first_error="",
        failure_reasons={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.passed = 2  # type: ignore[misc]


def test_parse_pytest_stdout_extracts_tb_line_reason() -> None:
    stdout = """\
tests/smoke/test_fail.py::test_broken FAILED                                  [100%]
FAILED tests/smoke/test_fail.py::test_broken - AssertionError: expected 1

========================= 1 failed in 1.00s =========================
"""
    stats = parse_pytest_stdout(stdout, exit_code=1)
    assert stats.failed_cases == ("tests/smoke/test_fail.py::test_broken",)
    assert stats.failure_reasons["tests/smoke/test_fail.py::test_broken"] == "AssertionError: expected 1"
