"""Domain models for CI incremental gate.

ChangeSet, GateStepResult, Baseline, CiGatePlan, GateError.
All frozen dataclasses — no I/O, no logic beyond property accessors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from scripts.helpers.ci_gate.gate_policy import SourceExemption, TestDiscovery

# Map product source prefix → regression layer directory.
# Used by _split_cross_layer_tests (in rules.py) to decide whether a source
# change targets a specific regression layer or all layers.
_SOURCE_PREFIX_TO_LAYER: Final[dict[str, str]] = {
    "tensor_cast/": "tests/regression/tensor_cast/",
    "serving_cast/": "tests/regression/serving_cast/",
}

_REGRESSION_LAYERS: Final[tuple[str, ...]] = (
    "tests/regression/tensor_cast/",
    "tests/regression/serving_cast/",
    "tests/regression/cli/",
    "tests/regression/web_ui/",
)


def regression_layer_for_source(source_path: str) -> str | None:
    """Return the regression layer directory for a source prefix, or None."""
    for prefix, layer in _SOURCE_PREFIX_TO_LAYER.items():
        if source_path.startswith(prefix):
            return layer
    return None


def layer_of_test(test_id: str) -> str | None:
    for layer in _REGRESSION_LAYERS:
        if test_id.startswith(layer):
            return layer
    return None


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
        )

    def modified_source_map(self) -> dict[str, frozenset[int]]:
        return dict(self.modified_source)


@dataclass(frozen=True, slots=True)
class GateStepResult:
    errors: tuple[GateError, ...] = ()
    tests: frozenset[str] = frozenset()
    cross_layer_deferred: frozenset[str] = frozenset()
    full_suite: bool = False

    @property
    def all_tests(self) -> frozenset[str]:
        return self.tests | self.cross_layer_deferred


@dataclass(frozen=True, slots=True)
class Baseline:
    test_map: dict[str, dict[str, list[str]]]
    exemptions: tuple[SourceExemption, ...]
    discovery: TestDiscovery
    product_prefixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CiGatePlan:
    blocking_errors: tuple[GateError, ...]
    deleted_source_tests: frozenset[str]
    incremental_tests: frozenset[str]
    full_suite: bool
