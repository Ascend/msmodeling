"""Gate rules: individual policy checks that produce GateStepResult.

Each _gate_* function is a self-contained rule. No orchestration here —
build_ci_gate_plan in gate.py decides which rules to run and merges results.
"""

from __future__ import annotations

from pathlib import Path

from scripts.helpers.ci_gate.gate_policy import SourceExemption, is_exempt
from scripts.helpers.ci_gate.models import (
    ChangeSet,
    GateError,
    GateStepResult,
    layer_of_test,
    regression_layer_for_source,
)
from scripts.helpers.common import ast_utils
from scripts.helpers.common.test_map_loader import is_product_source

# ---------------------------------------------------------------------------
# Cross-layer splitting
# ---------------------------------------------------------------------------


def _split_cross_layer_tests(source_path: str, test_ids: set[str]) -> tuple[set[str], set[str]]:
    """Split test_ids into (immediate, deferred) when source spans regression layers.

    If source_path maps to a specific regression layer, tests in that layer run
    immediately; tests in other layers are deferred. If source_path maps to no
    known layer, or tests are already single-layer, no splitting occurs.
    """
    preferred = regression_layer_for_source(source_path)
    layers = {layer_of_test(tid) for tid in test_ids} - {None}
    if len(layers) <= 1:
        return test_ids, set()
    if preferred is None:
        return test_ids, set()
    immediate = {tid for tid in test_ids if tid.startswith(preferred)}
    deferred = test_ids - immediate
    if immediate and deferred:
        return immediate, deferred
    return test_ids, set()


def _merge_step_results(*steps: GateStepResult) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    deferred: set[str] = set()
    full_suite = False
    for step in steps:
        errors.extend(step.errors)
        tests.update(step.tests)
        deferred.update(step.cross_layer_deferred)
        full_suite = full_suite or step.full_suite
    return GateStepResult(
        errors=tuple(errors),
        tests=frozenset(tests),
        cross_layer_deferred=frozenset(deferred),
        full_suite=full_suite,
    )


def _product_paths(paths: tuple[str, ...], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if is_product_source(path, prefixes))


# ---------------------------------------------------------------------------
# Gate: config changes
# ---------------------------------------------------------------------------


def gate_config() -> GateStepResult:
    return GateStepResult(full_suite=True)


# ---------------------------------------------------------------------------
# Gate: new source files
# ---------------------------------------------------------------------------


def gate_new_source(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    prefixes: tuple[str, ...],
) -> GateStepResult:
    errors: list[GateError] = []
    for path in changes.new_source:
        if not path.endswith(".py"):
            continue
        if not is_product_source(path, prefixes):
            continue
        if test_map.get(path):
            continue
        source_path = repo_root / path
        if not source_path.is_file():
            continue
        symbols = ast_utils.top_level_definitions(source_path)
        if not symbols:
            errors.append(GateError(category="new_source", path=path))
            continue
        unmapped = [sym for sym in symbols if not is_exempt(exemptions, path, sym)]
        for sym in unmapped:
            errors.append(GateError(category="new_source", path=path, symbol=sym))
    return GateStepResult(errors=tuple(errors))


# ---------------------------------------------------------------------------
# Gate: new tests
# ---------------------------------------------------------------------------


def gate_new_tests(changes: ChangeSet) -> GateStepResult:
    return GateStepResult(tests=frozenset(changes.new_test))


# ---------------------------------------------------------------------------
# Gate: deleted source files
# ---------------------------------------------------------------------------


def gate_deleted_source(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    prefixes: tuple[str, ...],
) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    deferred: set[str] = set()
    deleted_test_files = set(changes.del_test)
    for path in changes.del_source:
        if not is_product_source(path, prefixes):
            continue
        file_map = test_map.get(path)
        if not file_map:
            errors.append(GateError(category="deleted_source", path=path))
            continue
        all_tests = {
            tid for tids in file_map.values() for tid in tids if tid.split("::", 1)[0] not in deleted_test_files
        }
        if not all_tests:
            continue
        immediate, cross = _split_cross_layer_tests(path, all_tests)
        tests.update(immediate)
        deferred.update(cross)
    return GateStepResult(
        errors=tuple(errors),
        tests=frozenset(tests),
        cross_layer_deferred=frozenset(deferred),
    )


# ---------------------------------------------------------------------------
# Gate: deleted tests
# ---------------------------------------------------------------------------


def gate_deleted_tests(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
) -> GateStepResult:
    errors: list[GateError] = []
    deleted_source_set = set(changes.del_source)
    for deleted_path in changes.del_test:
        sole_coverage: list[str] = []
        normalized_deleted = deleted_path.split("::", 1)[0]
        for src_file, symbols in test_map.items():
            if src_file in deleted_source_set:
                continue
            for symbol, test_ids in symbols.items():
                normalized_paths = {test_id.split("::", 1)[0] for test_id in test_ids}
                if len(normalized_paths) == 1 and normalized_deleted in normalized_paths:
                    sole_coverage.append(f"{src_file}::{symbol}")
        if sole_coverage:
            detail = "\n".join(sole_coverage)
            errors.append(GateError(category="deleted_test", path=deleted_path, detail=detail))
    return GateStepResult(errors=tuple(errors))


# ---------------------------------------------------------------------------
# Gate: modified source files
# ---------------------------------------------------------------------------


def gate_modified_source(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    prefixes: tuple[str, ...],
) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    for path, raw_lines in changes.modified_source:
        if not is_product_source(path, prefixes):
            continue
        source_path = repo_root / path
        executable = ast_utils.filter_executable_lines(source_path, set(raw_lines))
        if not executable:
            continue
        symbols = ast_utils.symbols_for_lines(source_path, executable)
        file_map = test_map.get(path, {})
        for symbol in symbols:
            mapped = file_map.get(symbol)
            if mapped:
                tests.update(mapped)
            elif is_exempt(exemptions, path, symbol):
                continue
            else:
                errors.append(GateError(category="modified_source", path=path, symbol=symbol))
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))
