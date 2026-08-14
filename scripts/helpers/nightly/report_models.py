"""Report domain models for nightly pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttributionConclusion(str, Enum):
    """Per-failure conclusion for Feishu + exit-code policy."""

    FIRST_BAD = "first_bad"
    NEED_HUMAN = "need_human"
    CANNOT_REPRODUCE = "cannot_reproduce"
    UNCOLLECTIBLE = "uncollectible"


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
class FailureBlame:
    node_id: str
    commit_id: str
    author: str
    subject: str
    conclusion: AttributionConclusion
    last_reason: str = ""

    @property
    def attributed(self) -> bool:
        return self.conclusion == AttributionConclusion.FIRST_BAD

    @property
    def needs_human(self) -> bool:
        return self.conclusion in {
            AttributionConclusion.NEED_HUMAN,
            AttributionConclusion.UNCOLLECTIBLE,
        }


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
    failure_blames: tuple[FailureBlame, ...] = ()
    drift_warnings: tuple[str, ...] = ()
    pipeline_log_url: str = ""
    infra_message: str = ""
    timed_out: bool = False
    status_note: str = ""
