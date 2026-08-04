#!/usr/bin/env python3
"""Build test_map JSON from Coverage.py dynamic contexts (pytest --cov-context=test)."""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final, Protocol

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.diff import resolve_head_commit
from scripts.helpers.common.ast_utils import (
    MODULE_SYMBOL,
    canonical_symbol_for_line,
    collect_file_symbols,
)
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.coverage_omit import is_coverage_omitted_source
from scripts.helpers.common.pytest_runner import PYTEST_IGNORE_ADDOPTS

logger = logging.getLogger(__name__)

TEST_MAP_SCHEMA_VERSION: Final = 1
# test_node -> source_rel_path -> list[canonical_symbol]
TestMap = dict[str, dict[str, list[str]]]

# Deprecated alias kept for callers that still import the old unclassified token.
UNCLASSIFIED_SYMBOL: Final = MODULE_SYMBOL


class _CoverageDataReader(Protocol):
    def measured_files(self) -> list[str]: ...

    def contexts_by_lineno(self, filename: str) -> dict[int, list[str]]: ...

    def read(self) -> None: ...


def normalize_test_node_id(node_id: str) -> str:
    """Strip parametrized suffix ``[param]`` and pytest-cov phase suffix ``|run``."""
    base = node_id.split("|", 1)[0].strip() if node_id else ""
    return base.split("[", 1)[0] if base else ""


def _relative_repo_key(abs_path: str, roots: tuple[str, ...]) -> str | None:
    try:
        rel = Path(abs_path).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    key = rel.as_posix()
    if key.startswith(roots):
        return key
    return None


def _normalize_pytest_context(ctx: str) -> str:
    """Strip pytest-cov phase suffix and parametrized suffix from a coverage context."""
    return normalize_test_node_id(ctx.split("|", 1)[0].strip() if ctx else "")


def _collect_allowed_node_ids(
    marker_expr: str,
    pytest_args: list[str] | None = None,
) -> frozenset[str]:
    """Return node ids from smoke/regression directories matching marker_expr."""
    if pytest_args is None:
        pytest_args = [
            sys.executable,
            "-m",
            "pytest",
            *PYTEST_IGNORE_ADDOPTS,
            str(REPO_ROOT / "tests" / "smoke"),
            str(REPO_ROOT / "tests" / "regression"),
            "-m",
            marker_expr,
            "--collect-only",
            "-q",
            "--no-header",
        ]
    proc = subprocess.run(
        pytest_args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        logger.error("pytest collect-only failed:\n%s", proc.stderr or proc.stdout)
        raise SystemExit(proc.returncode)

    node_ids: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped and stripped.startswith("tests/"):
            node_ids.add(normalize_test_node_id(stripped))
    return frozenset(node_ids)


def _accumulate_measured_file(
    measured: str,
    *,
    data: _CoverageDataReader,
    allowed_node_ids: frozenset[str],
    resolved_roots: tuple[str, ...],
    by_test: dict[str, dict[str, set[str]]],
) -> None:
    key = _relative_repo_key(measured, resolved_roots)
    if key is None or is_coverage_omitted_source(key, resolved_roots):
        return
    ctxmap = data.contexts_by_lineno(measured)
    if not ctxmap:
        return

    file_symbols = collect_file_symbols(Path(measured).resolve())
    for line_no, ctxs in ctxmap.items():
        sym = canonical_symbol_for_line(file_symbols, line_no)
        for ctx in ctxs:
            nid = _normalize_pytest_context(ctx)
            if nid and nid in allowed_node_ids:
                by_test[nid][key].add(sym)


def _finalize_test_map(by_test: dict[str, dict[str, set[str]]]) -> TestMap:
    result: TestMap = {}
    for test_node, sources in sorted(by_test.items()):
        filtered: dict[str, list[str]] = {}
        for src_file, symbols in sorted(sources.items()):
            kept = sorted(symbols)
            if kept:
                filtered[src_file] = kept
        if filtered:
            result[test_node] = filtered
    return result


def collect_from_coverage(
    allowed_node_ids: frozenset[str],
    *,
    coverage_path: Path | None = None,
    roots: tuple[str, ...] | None = None,
) -> TestMap:
    """Read .coverage data and return ``test_node -> source_file -> [symbols]``."""
    try:
        coverage_data_mod = importlib.import_module("coverage.data")
        coverage_misc_mod = importlib.import_module("coverage.misc")
    except ImportError:
        logger.warning("coverage package not installed")
        return {}

    coverage_data_cls = coverage_data_mod.CoverageData
    coverage_exception_cls = coverage_misc_mod.CoverageException

    if coverage_path is None:
        coverage_path = REPO_ROOT / ".coverage"
    resolved_roots = roots if roots is not None else product_roots()

    if not coverage_path.is_file():
        logger.warning("Coverage data not found: %s", coverage_path)
        return {}

    data: _CoverageDataReader = coverage_data_cls(str(coverage_path))
    try:
        data.read()
    except coverage_exception_cls as exc:
        logger.warning("Failed to read coverage data: %s", exc)
        return {}
    except OSError as exc:
        logger.warning("Coverage data file error: %s", exc)
        return {}

    by_test: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for measured in data.measured_files():
        _accumulate_measured_file(
            measured,
            data=data,
            allowed_node_ids=allowed_node_ids,
            resolved_roots=resolved_roots,
            by_test=by_test,
        )

    return _finalize_test_map(by_test)


def _prune_missing_source_keys(mapping: TestMap) -> TestMap:
    """Drop source paths that no longer exist on disk."""
    pruned: TestMap = {}
    for test_node, sources in mapping.items():
        kept = {src: syms for src, syms in sources.items() if (REPO_ROOT / src).is_file()}
        if kept:
            pruned[test_node] = kept
    return pruned


def write_test_map(
    output_path: Path,
    mapping: TestMap,
    *,
    built_from_commit: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    commit = built_from_commit if built_from_commit is not None else resolve_head_commit(REPO_ROOT)
    payload = {
        "schema_version": TEST_MAP_SCHEMA_VERSION,
        "built_from_commit": commit,
        "map": mapping,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source_files = {src for sources in mapping.values() for src in sources}
    symbol_count = sum(len(syms) for sources in mapping.values() for syms in sources.values())
    logger.info(
        "test_map written: %s (%d test nodes, %d source files, %d symbol entries)",
        output_path,
        len(mapping),
        len(source_files),
        symbol_count,
    )


def build_test_map(
    output_path: Path,
    *,
    marker_expr: str,
    coverage_path: Path | None = None,
    roots: tuple[str, ...] | None = None,
    allowed_node_ids: frozenset[str] | None = None,
) -> None:
    allowed = allowed_node_ids if allowed_node_ids is not None else _collect_allowed_node_ids(marker_expr)
    mapping = collect_from_coverage(allowed, coverage_path=coverage_path, roots=roots)
    write_test_map(output_path, _prune_missing_source_keys(mapping))


def collect_test_map(
    *,
    marker_expr: str,
    coverage_path: Path | None = None,
    roots: tuple[str, ...] | None = None,
    allowed_node_ids: frozenset[str] | None = None,
) -> TestMap:
    """Return test_map dict in memory — no file I/O."""
    allowed = allowed_node_ids if allowed_node_ids is not None else _collect_allowed_node_ids(marker_expr)
    mapping = collect_from_coverage(allowed, coverage_path=coverage_path, roots=roots)
    return _prune_missing_source_keys(mapping)


def _qualified_symbol(src_file: str, symbol: str) -> str:
    return f"{src_file}::{symbol}"


def _index_redundancy_mapping(
    mapping: TestMap,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    test_to_symbols: dict[str, set[str]] = defaultdict(set)
    symbol_to_tests: dict[str, set[str]] = defaultdict(set)
    for test_node, sources in mapping.items():
        for src_file, symbols in sources.items():
            for sym in symbols:
                if sym == MODULE_SYMBOL:
                    continue
                qualified = _qualified_symbol(src_file, sym)
                test_to_symbols[test_node].add(qualified)
                symbol_to_tests[qualified].add(test_node)
    return test_to_symbols, symbol_to_tests


def _over_covered_symbol_warnings(
    symbol_to_tests: dict[str, set[str]],
    *,
    max_per_symbol: int,
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for qualified, test_nodes in sorted(symbol_to_tests.items()):
        if len(test_nodes) <= max_per_symbol:
            continue
        warnings.append(
            {
                "type": "over_covered_symbol",
                "symbol": qualified,
                "test_count": len(test_nodes),
                "threshold": max_per_symbol,
                "tests": sorted(test_nodes),
            }
        )
    return warnings


def _redundant_pair_warnings(
    test_to_symbols: dict[str, set[str]],
    symbol_to_tests: dict[str, set[str]],
    *,
    jaccard_threshold: float,
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    compared_pairs: set[tuple[str, str]] = set()
    for tests in symbol_to_tests.values():
        test_list = sorted(tests)
        for i, a_id in enumerate(test_list):
            for b_id in test_list[i + 1 :]:
                pair = (a_id, b_id)
                if pair in compared_pairs:
                    continue
                compared_pairs.add(pair)
                a_syms = test_to_symbols[a_id]
                b_syms = test_to_symbols[b_id]
                intersection = a_syms & b_syms
                union = a_syms | b_syms
                if not union:
                    continue
                jaccard = len(intersection) / len(union)
                if jaccard >= jaccard_threshold:
                    warnings.append(
                        {
                            "type": "redundant_pair",
                            "test_a": a_id,
                            "test_b": b_id,
                            "jaccard": round(jaccard, 3),
                            "shared_symbols": sorted(intersection),
                        }
                    )
    return warnings


def detect_redundant_cases(
    mapping: TestMap,
    *,
    jaccard_threshold: float = 0.85,
    max_per_symbol: int = 5,
) -> list[dict[str, object]]:
    """Return redundancy warnings from a test-node keyed test_map."""
    test_to_symbols, symbol_to_tests = _index_redundancy_mapping(mapping)
    warnings = _over_covered_symbol_warnings(symbol_to_tests, max_per_symbol=max_per_symbol)
    warnings.extend(
        _redundant_pair_warnings(
            test_to_symbols,
            symbol_to_tests,
            jaccard_threshold=jaccard_threshold,
        )
    )
    return warnings


collect_allowed_node_ids = _collect_allowed_node_ids
prune_missing_source_keys = _prune_missing_source_keys
