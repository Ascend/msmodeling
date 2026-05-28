#!/usr/bin/env python3
"""Build test_map JSON from Coverage.py dynamic contexts (pytest --cov-context=test)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

from scripts.helpers.common.ast_utils import iter_qualified_definition_spans, symbol_for_line
from scripts.helpers.common.coverage_config import PRODUCT_SOURCE_PREFIXES

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
UNCLASSIFIED_SYMBOL: Final = "*"
MIN_LINES_PER_SYMBOL: Final[int] = 3


def _relative_repo_key(abs_path: str) -> str | None:
    try:
        rel = Path(abs_path).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    key = rel.as_posix()
    if key.startswith(PRODUCT_SOURCE_PREFIXES):
        return key
    return None


def _normalize_pytest_context(ctx: str) -> str:
    """Strip pytest-cov phase suffix ``|run``, ``|setup``, ``|teardown``."""
    return ctx.split("|", 1)[0].strip() if ctx else ""


def _collect_allowed_node_ids(
    marker_expr: str,
    pytest_args: list[str] | None = None,
) -> frozenset[str]:
    """Return node ids from smoke/regression directories matching marker_expr.

    Uses ``pytest --collect-only -q --no-header`` for stable machine-readable
    output. Strips parameterised suffixes ``[param]`` to match coverage context
    base node ids.
    """
    if pytest_args is None:
        pytest_args = [
            sys.executable,
            "-m",
            "pytest",
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
        print(proc.stderr or proc.stdout, file=sys.stderr)
        raise SystemExit(proc.returncode)

    node_ids: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped and stripped.startswith("tests/"):
            base_id = stripped.split("[", 1)[0]
            node_ids.add(base_id)
    return frozenset(node_ids)


def collect_from_coverage(
    allowed_node_ids: frozenset[str],
    *,
    coverage_path: Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Read .coverage data, resolve symbols, return {rel_path: {symbol: [test_ids]}}.

    Returns empty dict if coverage data is missing or unreadable — logs warning
    to stderr so caller can distinguish "no data" from "empty map".
    """
    from coverage.data import CoverageData
    from coverage.misc import CoverageException

    if coverage_path is None:
        coverage_path = REPO_ROOT / ".coverage"

    if not coverage_path.is_file():
        print(f"Coverage data not found: {coverage_path}", file=sys.stderr)
        return {}

    data = CoverageData(str(coverage_path))
    try:
        data.read()
    except CoverageException as exc:
        print(f"Failed to read coverage data: {exc}", file=sys.stderr)
        return {}
    except OSError as exc:
        print(f"Coverage data file error: {exc}", file=sys.stderr)
        return {}

    by_file: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for measured in data.measured_files():
        key = _relative_repo_key(measured)
        if key is None:
            continue
        ctxmap = data.contexts_by_lineno(measured)
        if not ctxmap:
            continue

        spans = iter_qualified_definition_spans(Path(measured).resolve())

        for line_no, ctxs in ctxmap.items():
            sym = symbol_for_line(spans, line_no)
            bucket = sym if sym is not None else UNCLASSIFIED_SYMBOL
            for ctx in ctxs:
                nid = _normalize_pytest_context(ctx)
                if nid and nid in allowed_node_ids:
                    by_file[key][bucket][nid] += 1

    result: dict[str, dict[str, list[str]]] = {}
    for fp, syms in sorted(by_file.items()):
        filtered: dict[str, list[str]] = {}
        for sym, test_lines in sorted(syms.items()):
            kept = sorted(nid for nid, count in test_lines.items() if count >= MIN_LINES_PER_SYMBOL)
            if kept:
                filtered[sym] = kept
        if filtered:
            result[fp] = filtered

    return result


def _prune_missing_source_keys(
    mapping: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    """Drop product paths that no longer exist on disk."""
    return {source_path: symbols for source_path, symbols in mapping.items() if (REPO_ROOT / source_path).is_file()}


def write_test_map(output_path: Path, mapping: dict[str, dict[str, list[str]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "map": mapping}
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    symbol_count = sum(len(syms) for syms in mapping.values())
    print(
        f"test_map written: {output_path} ({len(mapping)} source files, {symbol_count} symbols)",
        file=sys.stderr,
    )


def build_test_map(
    output_path: Path,
    *,
    marker_expr: str,
    coverage_path: Path | None = None,
) -> None:
    allowed = _collect_allowed_node_ids(marker_expr)
    mapping = collect_from_coverage(allowed, coverage_path=coverage_path)
    mapping = _prune_missing_source_keys(mapping)
    write_test_map(output_path, mapping)


def detect_redundant_cases(
    mapping: dict[str, dict[str, list[str]]],
    *,
    jaccard_threshold: float = 0.85,
    max_per_symbol: int = 5,
) -> list[dict[str, object]]:
    """Return redundancy warnings from a test_map.

    Two checks:
    1. Symbols covered by more than *max_per_symbol* test cases.
    2. Pairs of test cases whose covered-symbol sets have Jaccard similarity
       >= *jaccard_threshold*.

    Returns a list of warning dicts suitable for nightly report inclusion.
    """
    warnings: list[dict[str, object]] = []

    test_to_symbols: dict[str, set[str]] = defaultdict(set)
    for src_file, symbols in mapping.items():
        for sym, test_ids in symbols.items():
            qualified = f"{src_file}::{sym}"
            for tid in test_ids:
                test_to_symbols[tid].add(qualified)

    for src_file, symbols in mapping.items():
        for sym, test_ids in symbols.items():
            if len(test_ids) > max_per_symbol:
                warnings.append(
                    {
                        "type": "over_covered_symbol",
                        "symbol": f"{src_file}::{sym}",
                        "test_count": len(test_ids),
                        "threshold": max_per_symbol,
                        "tests": sorted(test_ids),
                    }
                )

    test_ids_sorted = sorted(test_to_symbols)
    for i in range(len(test_ids_sorted)):
        for j in range(i + 1, len(test_ids_sorted)):
            a_id = test_ids_sorted[i]
            b_id = test_ids_sorted[j]
            a_syms = test_to_symbols[a_id]
            b_syms = test_to_symbols[b_id]
            if not a_syms or not b_syms:
                continue
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
