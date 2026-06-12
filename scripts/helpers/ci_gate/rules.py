"""Gate rules: individual policy checks that produce GateStepResult.

Each _gate_* function is a self-contained rule. No orchestration here —
build_ci_gate_plan in gate.py decides which rules to run and merges results.
"""

from __future__ import annotations

from pathlib import Path

from scripts.helpers.ci_gate.gate_policy import SourceExemption, is_exempt
from scripts.helpers.ci_gate.models import ChangeSet, GateError, GateStepResult
from scripts.helpers.common import ast_utils
from scripts.helpers.common.coverage_omit import is_coverage_omitted_source
from scripts.helpers.common.coverage_symbol_check import symbol_lines_covered_in_data
from scripts.helpers.common.test_map_loader import is_product_source


def _merge_step_results(*steps: GateStepResult) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    for step in steps:
        errors.extend(step.errors)
        tests.update(step.tests)
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))


def _product_paths(paths: tuple[str, ...], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if is_product_source(path, prefixes))


def _executable_lines_for_symbol(source_path: Path, symbol: str) -> set[int]:
    spans = ast_utils.iter_qualified_definition_spans(source_path)
    symbol_lines: set[int] = set()
    for span in spans:
        if span.qualified_name == symbol:
            symbol_lines = set(range(span.start_line, span.end_line + 1))
            break
    if not symbol_lines:
        return set()
    return ast_utils.filter_executable_lines(source_path, symbol_lines)


def _symbol_lines_from_diff(source_path: Path, symbol: str, executable: set[int]) -> set[int]:
    spans = ast_utils.iter_qualified_definition_spans(source_path)
    return {line_no for line_no in executable if ast_utils.symbol_for_line(spans, line_no) == symbol}


def _coverage_fallback_passes(
    repo_root: Path,
    file_path: str,
    symbol: str,
    lines: set[int],
    *,
    coverage_path: Path | None,
) -> bool:
    if not coverage_path or not lines:
        return False
    return symbol_lines_covered_in_data(repo_root, file_path, symbol, lines, coverage_path)


# ---------------------------------------------------------------------------
# Gate: new source files
# ---------------------------------------------------------------------------


def gate_new_source(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path | None = None,
) -> GateStepResult:
    errors: list[GateError] = []
    for path in changes.new_source:
        if not path.endswith(".py"):
            continue
        if not is_product_source(path, roots):
            continue
        if is_coverage_omitted_source(path, roots):
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
            if _coverage_fallback_passes(
                repo_root,
                path,
                sym,
                _executable_lines_for_symbol(source_path, sym),
                coverage_path=coverage_path,
            ):
                continue
            errors.append(GateError(category="new_source", path=path, symbol=sym))
    return GateStepResult(errors=tuple(errors))


# ---------------------------------------------------------------------------
# Gate: new tests
# ---------------------------------------------------------------------------


def gate_new_tests(changes: ChangeSet) -> GateStepResult:
    """Run added and in-place-modified test files directly (their edits may change coverage)."""
    candidates = changes.new_test + changes.modified_test
    return GateStepResult(tests=frozenset(candidates))


# ---------------------------------------------------------------------------
# Gate: deleted source files
# ---------------------------------------------------------------------------


def gate_deleted_source(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    roots: tuple[str, ...],
) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    deleted_test_files = set(changes.del_test)
    for path in changes.del_source:
        if not is_product_source(path, roots):
            continue
        file_map = test_map.get(path)
        if not file_map:
            errors.append(GateError(category="deleted_source", path=path))
            continue
        guard_tests = {
            tid for tids in file_map.values() for tid in tids if tid.split("::", 1)[0] not in deleted_test_files
        }
        tests.update(guard_tests)
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))


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
    roots: tuple[str, ...],
    *,
    coverage_path: Path | None = None,
) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    for path, raw_lines in changes.modified_source:
        if not is_product_source(path, roots):
            continue
        if is_coverage_omitted_source(path, roots):
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
            elif is_exempt(exemptions, path, symbol) or _coverage_fallback_passes(
                repo_root,
                path,
                symbol,
                _symbol_lines_from_diff(source_path, symbol, executable),
                coverage_path=coverage_path,
            ):
                continue
            else:
                errors.append(GateError(category="modified_source", path=path, symbol=symbol))
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))
