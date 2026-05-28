"""Report domain models for nightly pipeline.

NightlyReport, CoverageSummary, MapCoverageSummary, EnvInfo.
All frozen dataclasses. NightlyReport.to_dict handles JSON serialization.
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
    source_files: int
    symbols: int


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    line_percent: float
    branch_percent: float
    line_threshold: float
    branch_threshold: float
    gate_passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class NightlyReport:
    pytest_exit_code: int
    passed: int
    failed: int
    errors: int
    duration_sec: float
    failed_cases: list[str]
    first_error: str
    commit: str
    branch: str
    timestamp: str
    test_map_source_files: int
    test_map_symbols: int
    test_map_path: str
    test_map_written: bool
    coverage: CoverageSummary | None
    weak_coverage_symbols: tuple[str, ...] = ()
    redundancy_warnings: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "pytest_exit_code": self.pytest_exit_code,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_sec": self.duration_sec,
            "failed_cases": self.failed_cases,
            "first_error": self.first_error,
            "commit": self.commit,
            "branch": self.branch,
            "timestamp": self.timestamp,
            "test_map_source_files": self.test_map_source_files,
            "test_map_symbols": self.test_map_symbols,
            "test_map_path": self.test_map_path,
            "test_map_written": self.test_map_written,
        }
        if self.coverage is not None:
            result.update(
                {
                    "coverage_line_percent": self.coverage.line_percent,
                    "coverage_branch_percent": self.coverage.branch_percent,
                    "coverage_line_threshold": self.coverage.line_threshold,
                    "coverage_branch_threshold": self.coverage.branch_threshold,
                    "coverage_gate_passed": self.coverage.gate_passed,
                    "coverage_message": self.coverage.message,
                }
            )
        else:
            result.update(
                {
                    "coverage_line_percent": None,
                    "coverage_branch_percent": None,
                    "coverage_line_threshold": None,
                    "coverage_branch_threshold": None,
                    "coverage_gate_passed": None,
                    "coverage_message": "",
                }
            )
        result["weak_coverage_symbols"] = list(self.weak_coverage_symbols)
        result["redundancy_warnings"] = list(self.redundancy_warnings)
        return result
