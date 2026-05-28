"""Domain models for CI incremental gate.

Exemption, ChangeSet, GateStepResult, Baseline, CiGatePlan.
All frozen dataclasses — no I/O, no logic beyond property accessors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Exemption:
    file: str
    symbol: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChangeSet:
    config: tuple[str, ...]
    new_test: tuple[str, ...]
    del_test: tuple[str, ...]
    new_source: tuple[str, ...]
    del_source: tuple[str, ...]
    modified_source: tuple[tuple[str, frozenset[int]], ...]

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
    ) -> ChangeSet:
        mod = tuple(sorted((path, lines) for path, lines in (modified_source or {}).items()))
        return cls(
            config=config,
            new_test=new_test,
            del_test=del_test,
            new_source=new_source,
            del_source=del_source,
            modified_source=mod,
        )

    def modified_source_map(self) -> dict[str, frozenset[int]]:
        return dict(self.modified_source)


@dataclass(frozen=True, slots=True)
class GateStepResult:
    errors: tuple[str, ...] = ()
    tests: frozenset[str] = frozenset()
    cross_layer_deferred: frozenset[str] = frozenset()
    full_suite: bool = False

    @property
    def all_tests(self) -> frozenset[str]:
        return self.tests | self.cross_layer_deferred


@dataclass(frozen=True, slots=True)
class Baseline:
    test_map: dict[str, dict[str, list[str]]]
    exemptions: tuple[Exemption, ...]
    product_prefixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CiGatePlan:
    blocking_errors: tuple[str, ...]
    deleted_source_tests: frozenset[str]
    incremental_tests: frozenset[str]
    full_suite: bool
    symbol_warnings: tuple[str, ...] = ()
