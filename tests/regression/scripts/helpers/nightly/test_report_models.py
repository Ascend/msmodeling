"""Tests for nightly.report_models — CoverageSummary, MapCoverageSummary, EnvInfo."""

from __future__ import annotations

from scripts.helpers.nightly.report_models import (
    CoverageSummary,
    EnvInfo,
    FeishuReportInput,
    MapCoverageSummary,
    PhaseBreakdownEntry,
)

# ---------------------------------------------------------------------------
# EnvInfo
# ---------------------------------------------------------------------------


def test_env_info_fields() -> None:
    e = EnvInfo(commit="abc123", branch="main", timestamp="2026-01-01T00:00:00Z")
    assert e.commit == "abc123"
    assert e.branch == "main"
    assert e.timestamp == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# MapCoverageSummary
# ---------------------------------------------------------------------------


def test_map_coverage_summary() -> None:
    m = MapCoverageSummary(source_files=10, symbols=42)
    assert m.source_files == 10
    assert m.symbols == 42


# ---------------------------------------------------------------------------
# CoverageSummary
# ---------------------------------------------------------------------------


def test_coverage_summary() -> None:
    c = CoverageSummary(
        line_percent=85.5,
        branch_percent=72.3,
        line_threshold=70.0,
        branch_threshold=50.0,
        gate_passed=True,
        message="passed",
    )
    assert c.line_percent == 85.5
    assert c.gate_passed is True


def test_phase_breakdown_entry_defaults() -> None:
    entry = PhaseBreakdownEntry(label="phase1", passed=1, failed=0, duration_sec=1.0)
    assert entry.exit_code == 0
    assert entry.infra_failure is False


def test_feishu_report_input_accepts_phase_breakdown() -> None:
    report = FeishuReportInput(
        timestamp="2026-01-01T00:00:00Z",
        branch="main",
        commit="abc",
        passed=0,
        failed=0,
        errors=0,
        duration_sec=-1.0,
        overall_exit=1,
        coverage_line_percent=None,
        coverage_branch_percent=None,
        coverage_line_threshold=None,
        coverage_branch_threshold=None,
        coverage_gate_passed=None,
        test_map_source_files=0,
        test_map_symbols=0,
        test_map_written=False,
        failed_cases=(),
        first_error="",
        phase_breakdown=(
            PhaseBreakdownEntry(
                label="phase1",
                passed=0,
                failed=0,
                duration_sec=-1.0,
                exit_code=1,
                infra_failure=True,
            ),
        ),
    )
    assert report.phase_breakdown[0].infra_failure is True
