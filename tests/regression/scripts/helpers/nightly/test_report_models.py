"""Tests for nightly.report_models — CoverageSummary, MapCoverageSummary, EnvInfo."""

from __future__ import annotations

from scripts.helpers.nightly.report_models import CoverageSummary, EnvInfo, MapCoverageSummary

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
