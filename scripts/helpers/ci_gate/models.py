"""Domain models for CI incremental gate.

ChangeSet, GateStepResult, Baseline, CiGatePlan, GateError.
All frozen dataclasses — no I/O, no logic beyond property accessors.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.helpers.ci_gate.gate_policy import SourceExemption, TestDiscovery, TestExemption


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
    # (old_path, new_path, similarity_score) for product-source renames (git -M).
    renames: tuple[tuple[str, str, int], ...] = ()
    # Existing test files edited in place (status M/C). Re-run and remapped like
    # new tests, since their edits may change which source symbols they cover.
    modified_test: tuple[str, ...] = ()

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
        renames: tuple[tuple[str, str, int], ...] = (),
        modified_test: tuple[str, ...] = (),
    ) -> ChangeSet:
        mod = tuple(sorted((path, lines) for path, lines in (modified_source or {}).items()))
        return cls(
            config=config,
            new_test=new_test,
            del_test=del_test,
            new_source=new_source,
            del_source=del_source,
            modified_source=mod,
            renames=renames,
            modified_test=modified_test,
        )


@dataclass(frozen=True, slots=True)
class GateStepResult:
    errors: tuple[GateError, ...] = ()
    tests: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Baseline:
    test_map: dict[str, dict[str, list[str]]]
    exemptions: tuple[SourceExemption, ...]
    test_exemptions: tuple[TestExemption, ...]
    discovery: TestDiscovery
    roots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CiGatePlan:
    blocking_errors: tuple[GateError, ...]
    deleted_source_tests: frozenset[str]
    changed_test_nodes: frozenset[str]
    regression_tests: frozenset[str]
    full_suite: bool


@dataclass(frozen=True, slots=True)
class TestRunWave:
    """One pytest invocation: targets (node ids or directories) share a marker."""

    targets: tuple[str, ...]
    marker: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Deduplicated pytest schedule produced after policy checks pass."""

    full_suite: bool
    waves: tuple[TestRunWave, ...]
    reasons: dict[str, str]

    @property
    def has_work(self) -> bool:
        return bool(self.waves)
