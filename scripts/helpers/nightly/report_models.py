"""Report domain models for nightly pipeline.

CoverageSummary, MapCoverageSummary, EnvInfo, PhaseBreakdownEntry, FeishuReportInput.
All frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvInfo:
    commit: str
    branch: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class MapCoverageSummary:
    test_nodes: int
    symbol_refs: int


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    line_percent: float
    branch_percent: float
    line_threshold: float
    branch_threshold: float
    gate_passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class PhaseBreakdownEntry:
    label: str
    passed: int
    failed: int
    duration_sec: float
    exit_code: int = 0
    infra_failure: bool = False


@dataclass(frozen=True, slots=True)
class FeishuReportInput:
    timestamp: str
    branch: str
    commit: str
    passed: int
    failed: int
    errors: int
    duration_sec: float
    overall_exit: int
    coverage_line_percent: float | None
    coverage_branch_percent: float | None
    coverage_line_threshold: float | None
    coverage_branch_threshold: float | None
    coverage_gate_passed: bool | None
    test_map_test_nodes: int
    test_map_symbol_refs: int
    test_map_written: bool
    failed_cases: tuple[str, ...]
    first_error: str
    weak_coverage_symbols: tuple[str, ...] = ()
    redundancy_warnings: tuple[dict[str, object], ...] = ()
    expired_exemption_section: str = ""
    phase_breakdown: tuple[PhaseBreakdownEntry, ...] = ()
    slowest_tests: tuple[tuple[str, float], ...] = ()
    drift_warnings: tuple[str, ...] = ()
