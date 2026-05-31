"""Report domain models for nightly pipeline.

CoverageSummary, MapCoverageSummary, EnvInfo. All frozen dataclasses.
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
