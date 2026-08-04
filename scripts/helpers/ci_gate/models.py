"""Domain models for CI incremental gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date


@dataclass(frozen=True, slots=True)
class PathPatterns:
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestDiscovery:
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceExemption:
    file: str
    symbol: str
    reason: str
    applicant: str
    approver: str
    deadline: date
    ticket: str | None = None

    @property
    def symbol_key(self) -> str:
        return f"{self.file}::{self.symbol}"


@dataclass(frozen=True, slots=True)
class TestExemption:
    test_id: str
    reason: str
    applicant: str
    approver: str
    deadline: date
    ticket: str | None = None


@dataclass(frozen=True, slots=True)
class ExpiredExemptionReport:
    symbol_key: str
    deadline: date
    reason: str
    applicant: str
    approver: str
    ticket: str | None


@dataclass(frozen=True, slots=True)
class CiGatePolicy:
    sources: PathPatterns
    tests: PathPatterns
    configs: PathPatterns
    source_exemptions: tuple[SourceExemption, ...]
    test_exemptions: tuple[TestExemption, ...]
    approvers: frozenset[str]

    @property
    def roots(self) -> tuple[str, ...]:
        return self.sources.include_patterns

    @property
    def discovery(self) -> TestDiscovery:
        return TestDiscovery(
            include_patterns=self.tests.include_patterns,
            exclude_patterns=self.tests.exclude_patterns,
        )


GatePolicy = CiGatePolicy


@dataclass(frozen=True, slots=True)
class GateError:
    category: str
    path: str
    symbol: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ChangeSet:
    config: tuple[str, ...]
    new_test: tuple[str, ...]
    del_test: tuple[str, ...]
    new_source: tuple[str, ...]
    del_source: tuple[str, ...]
    modified_source: tuple[tuple[str, frozenset[int]], ...]
    modified_test: tuple[str, ...] = ()
    unscoped_python: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        config: tuple[str, ...] = (),
        new_test: tuple[str, ...] = (),
        del_test: tuple[str, ...] = (),
        new_source: tuple[str, ...] = (),
        del_source: tuple[str, ...] = (),
        modified_source: dict[str, frozenset[int]] | None = None,
        modified_test: tuple[str, ...] = (),
        unscoped_python: tuple[str, ...] = (),
    ) -> ChangeSet:
        mod = tuple(sorted((path, lines) for path, lines in (modified_source or {}).items()))
        return cls(
            config=config,
            new_test=new_test,
            del_test=del_test,
            new_source=new_source,
            del_source=del_source,
            modified_source=mod,
            modified_test=modified_test,
            unscoped_python=unscoped_python,
        )


@dataclass(frozen=True, slots=True)
class GateStepResult:
    errors: tuple[GateError, ...] = ()
    tests: frozenset[str] = frozenset()
    all_exempt_test_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Baseline:
    test_map: dict[str, dict[str, list[str]]]
    policy: CiGatePolicy

    @property
    def roots(self) -> tuple[str, ...]:
        return self.policy.roots

    @property
    def exemptions(self) -> tuple[SourceExemption, ...]:
        return self.policy.source_exemptions

    @property
    def test_exemptions(self) -> tuple[TestExemption, ...]:
        return self.policy.test_exemptions

    @property
    def discovery(self) -> TestDiscovery:
        return self.policy.discovery


@dataclass(frozen=True, slots=True)
class CiGatePlan:
    deleted_source_tests: frozenset[str]
    changed_test_nodes: frozenset[str]
    regression_tests: frozenset[str]
    full_suite: bool
    all_exempt_test_files: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class TestRunWave:
    targets: tuple[str, ...]
    marker: str | None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    full_suite: bool
    waves: tuple[TestRunWave, ...]
    reasons: dict[str, str]

    @property
    def has_work(self) -> bool:
        return bool(self.waves)
