"""Gate rules: individual policy checks that produce GateStepResult."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scripts.helpers.ci_gate.models import (
    ChangeSet,
    GateError,
    GateStepResult,
    SourceExemption,
    TestExemption,
)
from scripts.helpers.ci_gate.policy import is_exempt, is_test_exempt
from scripts.helpers.ci_gate.test_map_query import (
    TestMapIndex,
    build_test_map_index,
    is_symbol_mapped,
    nodes_for_test_file,
    source_watchers,
    symbol_watchers,
)
from scripts.helpers.common import ast_utils
from scripts.helpers.common.ast_utils import (
    MODULE_SYMBOL,
    CoverageChecks,
    coverage_checks_for_definition,
    coverage_measurable_lines,
    executable_lines_for_canonical_symbol,
    gated_coverage_symbols,
    import_symbol_for_definition,
    touched_definition_symbols,
)
from scripts.helpers.common.coverage_omit import is_coverage_omitted_source
from scripts.helpers.common.coverage_symbol_check import (
    CoverageDataReader,
    symbol_lines_covered_in_data,
)
from scripts.helpers.common.pytest_runner import collect_all_test_node_ids
from scripts.helpers.common.test_map_loader import is_product_source

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _product_paths(paths: tuple[str, ...], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if is_product_source(path, prefixes))


def _symbol_lines_from_diff(source_path: Path, canonical_symbol: str, executable: set[int]) -> set[int]:
    return executable & executable_lines_for_canonical_symbol(source_path, canonical_symbol)


def _is_import_scope_symbol(symbol: str) -> bool:
    return symbol == MODULE_SYMBOL or symbol.endswith(f"::{MODULE_SYMBOL}")


def _skip_import_scope_via_decorator_coverage(
    symbol: str,
    executable: set[int],
    sym_lines: set[int],
    decorator_import_lines: set[int],
) -> bool:
    if not _is_import_scope_symbol(symbol) or not decorator_import_lines:
        return False
    return executable <= decorator_import_lines or (bool(sym_lines) and sym_lines <= decorator_import_lines)


def _coverage_checks_pass(
    repo_root: Path,
    file_path: str,
    source_path: Path,
    symbol: str,
    checks: CoverageChecks,
    *,
    coverage_path: Path | None,
    coverage_data: CoverageDataReader | None = None,
) -> bool:
    if checks.import_lines:
        import_symbol = import_symbol_for_definition(source_path, symbol)
        if not _coverage_fallback_passes(
            repo_root,
            file_path,
            source_path,
            import_symbol,
            set(checks.import_lines),
            coverage_path=coverage_path,
            coverage_data=coverage_data,
        ):
            return False
    if checks.strict_lines:
        if not _coverage_fallback_passes(
            repo_root,
            file_path,
            source_path,
            symbol,
            set(checks.strict_lines),
            coverage_path=coverage_path,
            coverage_data=coverage_data,
        ):
            return False
    elif checks.proxy_lines and not _coverage_fallback_passes(
        repo_root,
        file_path,
        source_path,
        symbol,
        set(checks.proxy_lines),
        coverage_path=coverage_path,
        coverage_data=coverage_data,
    ):
        return False
    return True


def _coverage_fallback_passes(
    repo_root: Path,
    file_path: str,
    source_path: Path,
    symbol: str,
    lines: set[int],
    *,
    coverage_path: Path | None,
    coverage_data: CoverageDataReader | None = None,
) -> bool:
    if not lines or (coverage_data is None and not coverage_path):
        return False
    check_lines = set(lines) | coverage_measurable_lines(source_path, lines)
    require_test_context = not (symbol == MODULE_SYMBOL or symbol.endswith(f"::{MODULE_SYMBOL}"))
    if symbol_lines_covered_in_data(
        repo_root,
        file_path,
        symbol,
        check_lines,
        coverage_path,
        coverage_data=coverage_data,
        require_test_context=require_test_context,
    ):
        logger.info(
            "Coverage fallback accepted %s::%s (%d executable line(s))",
            file_path,
            symbol,
            len(check_lines),
        )
        return True
    return False


def _modified_source_mapping_error(
    repo_root: Path,
    path: str,
    source_path: Path,
    symbol: str,
    executable: set[int],
    *,
    touched: frozenset[str],
    touched_checks: dict[str, CoverageChecks],
    decorator_import_lines: set[int],
    coverage_path: Path | None,
    coverage_data: CoverageDataReader | None,
) -> GateError | None:
    if symbol in touched:
        checks = touched_checks[symbol]
        if not checks.import_lines and not checks.strict_lines and not checks.proxy_lines:
            return None
        if _coverage_checks_pass(
            repo_root,
            path,
            source_path,
            symbol,
            checks,
            coverage_path=coverage_path,
            coverage_data=coverage_data,
        ):
            return None
        return GateError(category="modified_source", path=path, symbol=symbol)

    sym_lines = _symbol_lines_from_diff(source_path, symbol, executable)
    # Module/class import-scope symbols are attributed from decorator lines;
    # skip when decorator import coverage already satisfied the touched path.
    if _skip_import_scope_via_decorator_coverage(symbol, executable, sym_lines, decorator_import_lines):
        return None
    if not sym_lines:
        return None
    if _coverage_fallback_passes(
        repo_root,
        path,
        source_path,
        symbol,
        sym_lines,
        coverage_path=coverage_path,
        coverage_data=coverage_data,
    ):
        return None
    return GateError(category="modified_source", path=path, symbol=symbol)


def _symbol_error_for_unmapped_source(
    repo_root: Path,
    path: str,
    source_path: Path,
    canonical_symbol: str,
    *,
    coverage_path: Path | None,
    coverage_data: CoverageDataReader | None = None,
) -> list[GateError]:
    lines = executable_lines_for_canonical_symbol(source_path, canonical_symbol)
    if not lines:
        return []
    if _coverage_fallback_passes(
        repo_root,
        path,
        source_path,
        canonical_symbol,
        lines,
        coverage_path=coverage_path,
        coverage_data=coverage_data,
    ):
        return []
    return [
        GateError(
            category="new_source",
            path=path,
            symbol=None if canonical_symbol == MODULE_SYMBOL else canonical_symbol,
        )
    ]


def _new_source_errors_for_path(
    repo_root: Path,
    path: str,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path | None,
    coverage_data: CoverageDataReader | None = None,
) -> list[GateError]:
    if not path.endswith(".py"):
        return []
    if is_coverage_omitted_source(path, roots):
        return []
    source_path = repo_root / path
    if not source_path.is_file():
        return []
    required_symbols = gated_coverage_symbols(source_path)
    if not required_symbols:
        return []
    errors: list[GateError] = []
    for canonical_symbol in sorted(required_symbols):
        if is_exempt(exemptions, path, canonical_symbol) or is_symbol_mapped(test_map, path, canonical_symbol):
            continue
        errors.extend(
            _symbol_error_for_unmapped_source(
                repo_root,
                path,
                source_path,
                canonical_symbol,
                coverage_path=coverage_path,
                coverage_data=coverage_data,
            )
        )
    return errors


def gate_new_source(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path | None = None,
    coverage_data: CoverageDataReader | None = None,
    check_mapping: bool = True,
) -> GateStepResult:
    if not check_mapping:
        return GateStepResult()
    errors: list[GateError] = []
    for path in changes.new_source:
        errors.extend(
            _new_source_errors_for_path(
                repo_root,
                path,
                test_map,
                exemptions,
                roots,
                coverage_path=coverage_path,
                coverage_data=coverage_data,
            )
        )
    return GateStepResult(errors=tuple(errors))


def gate_new_tests(
    changes: ChangeSet,
    test_exemptions: tuple[TestExemption, ...],
    *,
    full_suite: bool = False,
) -> GateStepResult:
    """Collect pytest node ids from new or modified test files without marker filtering."""
    if full_suite:
        return GateStepResult()
    test_files = changes.new_test + changes.modified_test
    if not test_files:
        return GateStepResult()
    collected = collect_all_test_node_ids(list(test_files))
    nodes: list[str] = []
    all_exempt: list[str] = []
    for test_file in test_files:
        file_nodes = [node_id for node_id in collected if node_id.split("::", 1)[0] == test_file]
        runnable = tuple(node_id for node_id in file_nodes if not is_test_exempt(test_exemptions, node_id))
        if file_nodes and not runnable:
            all_exempt.append(test_file)
            continue
        nodes.extend(runnable)
    return GateStepResult(tests=frozenset(nodes), all_exempt_test_files=tuple(all_exempt))


def gate_deleted_source(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    roots: tuple[str, ...],
) -> GateStepResult:
    tests: set[str] = set()
    errors: list[GateError] = []
    deleted_test_files = set(changes.del_test)
    index = build_test_map_index(test_map)
    for path in changes.del_source:
        if not is_product_source(path, roots):
            continue
        sole_coverage: list[str] = []
        guard_tests: set[str] = set()
        for node in source_watchers(test_map, path, index=index):
            if node.split("::", 1)[0] in deleted_test_files:
                continue
            guard_tests.add(node)
            for symbol in test_map.get(node, {}).get(path, []):
                other_watchers = symbol_watchers(test_map, path, symbol, index=index) - {node}
                other_watchers -= {n for n in other_watchers if n.split("::", 1)[0] in deleted_test_files}
                if not other_watchers:
                    sole_coverage.append(f"{path}::{symbol}")
        tests.update(guard_tests)
        if sole_coverage:
            detail = "\n".join(sorted(set(sole_coverage)))
            errors.append(GateError(category="deleted_source", path=path, detail=detail))
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))


def gate_deleted_tests(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
) -> GateStepResult:
    errors: list[GateError] = []
    deleted_source_set = set(changes.del_source)
    index = build_test_map_index(test_map)
    for deleted_path in changes.del_test:
        deleted_nodes = nodes_for_test_file(test_map, deleted_path)
        sole_coverage: list[str] = []
        for node in deleted_nodes:
            for src_file, symbols in test_map.get(node, {}).items():
                if src_file in deleted_source_set:
                    continue
                for symbol in symbols:
                    watchers = symbol_watchers(test_map, src_file, symbol, index=index) - {node}
                    if not watchers:
                        sole_coverage.append(f"{src_file}::{symbol}")
        if sole_coverage:
            detail = "\n".join(sorted(set(sole_coverage)))
            errors.append(GateError(category="deleted_test", path=deleted_path, detail=detail))
    return GateStepResult(errors=tuple(errors))


def _modified_source_path_gate(
    repo_root: Path,
    path: str,
    raw_lines: frozenset[int],
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    index: TestMapIndex,
    *,
    coverage_path: Path | None,
    coverage_data: CoverageDataReader | None,
    check_mapping: bool,
    collect_tests: bool,
) -> GateStepResult:
    if not is_product_source(path, roots):
        return GateStepResult()
    if is_coverage_omitted_source(path, roots):
        return GateStepResult()
    source_path = repo_root / path
    executable = ast_utils.filter_executable_lines(source_path, set(raw_lines))
    if not executable:
        return GateStepResult()
    line_symbols = ast_utils.canonical_symbols_for_lines(source_path, executable)
    touched = touched_definition_symbols(source_path, executable)
    all_symbols = line_symbols | touched
    touched_checks: dict[str, CoverageChecks] = {
        touched_symbol: coverage_checks_for_definition(source_path, touched_symbol, executable)
        for touched_symbol in touched
    }
    decorator_import_lines = set().union(*(checks.import_lines for checks in touched_checks.values()))
    errors: list[GateError] = []
    tests: set[str] = set()
    for symbol in all_symbols:
        mapped = symbol_watchers(test_map, path, symbol, index=index)
        if not mapped and _is_import_scope_symbol(symbol):
            mapped = source_watchers(test_map, path, index=index)
        if mapped:
            if collect_tests:
                tests.update(mapped)
        elif not check_mapping or is_exempt(exemptions, path, symbol):
            continue
        else:
            mapping_error = _modified_source_mapping_error(
                repo_root,
                path,
                source_path,
                symbol,
                executable,
                touched=touched,
                touched_checks=touched_checks,
                decorator_import_lines=decorator_import_lines,
                coverage_path=coverage_path,
                coverage_data=coverage_data,
            )
            if mapping_error is not None:
                errors.append(mapping_error)
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))


def _run_modified_source_gate(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path | None = None,
    coverage_data: CoverageDataReader | None = None,
    check_mapping: bool,
    collect_tests: bool,
) -> GateStepResult:
    errors: list[GateError] = []
    tests: set[str] = set()
    index = build_test_map_index(test_map)
    for path, raw_lines in changes.modified_source:
        path_step = _modified_source_path_gate(
            repo_root,
            path,
            raw_lines,
            test_map,
            exemptions,
            roots,
            index,
            coverage_path=coverage_path,
            coverage_data=coverage_data,
            check_mapping=check_mapping,
            collect_tests=collect_tests,
        )
        errors.extend(path_step.errors)
        tests.update(path_step.tests)
    return GateStepResult(errors=tuple(errors), tests=frozenset(tests))


def gate_modified_source(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path | None = None,
    coverage_data: CoverageDataReader | None = None,
    check_mapping: bool = True,
) -> GateStepResult:
    return _run_modified_source_gate(
        repo_root,
        changes,
        test_map,
        exemptions,
        roots,
        coverage_path=coverage_path,
        coverage_data=coverage_data,
        check_mapping=check_mapping,
        collect_tests=True,
    )


def collect_modified_source_mapping_errors(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path,
    coverage_data: CoverageDataReader | None,
) -> tuple[GateError, ...]:
    return _run_modified_source_gate(
        repo_root,
        changes,
        test_map,
        exemptions,
        roots,
        coverage_path=coverage_path,
        coverage_data=coverage_data,
        check_mapping=True,
        collect_tests=False,
    ).errors
