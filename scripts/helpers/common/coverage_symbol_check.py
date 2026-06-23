"""Check whether changed source lines were executed in Coverage.py data."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol

from scripts.helpers.common.build_test_map import normalize_test_node_id


class CoverageDataReader(Protocol):
    def measured_files(self) -> list[str]: ...

    def contexts_by_lineno(self, filename: str) -> dict[int, list[str]]: ...

    def read(self) -> None: ...


def load_coverage_data(coverage_path: Path) -> CoverageDataReader | None:
    """Load repo-root coverage data once for repeated symbol checks."""
    if not coverage_path.is_file():
        return None

    try:
        coverage_data_mod = importlib.import_module("coverage.data")
        coverage_misc_mod = importlib.import_module("coverage.misc")
    except ImportError:
        return None

    coverage_data_cls = coverage_data_mod.CoverageData
    coverage_exception_cls = coverage_misc_mod.CoverageException

    data: CoverageDataReader = coverage_data_cls(str(coverage_path))
    try:
        data.read()
    except (coverage_exception_cls, OSError):
        return None
    return data


def _measured_path_for_source(data: CoverageDataReader, repo_root: Path, source_rel_path: str) -> str | None:
    target = (repo_root / source_rel_path).resolve()
    for measured in data.measured_files():
        if Path(measured).resolve() == target:
            return str(measured)
    return None


def _is_pytest_test_context(ctx: str) -> bool:
    """Return True when *ctx* names a pytest test node (not import-time or conftest)."""
    normalized = normalize_test_node_id(ctx.split("|", 1)[0].strip() if ctx else "")
    return normalized.startswith("tests/") and "::" in normalized


def symbol_lines_covered_in_data(
    repo_root: Path,
    source_rel_path: str,
    symbol: str,
    lines: set[int],
    coverage_path: Path | None,
    *,
    coverage_data: CoverageDataReader | None = None,
    require_test_context: bool = True,
) -> bool:
    """Return True when any *lines* was executed in coverage data.

    When *require_test_context* is True (default), only pytest test node contexts
    count; import-time, conftest-only, and empty contexts do not satisfy the gate.
    When False, any measured line key in *lines* is sufficient.
    *symbol* is part of the public API for call-site clarity.
    """
    _ = symbol
    if not lines:
        return False

    data = coverage_data
    if data is None:
        if coverage_path is None or not coverage_path.is_file():
            return False
        data = load_coverage_data(coverage_path)
        if data is None:
            return False

    measured = _measured_path_for_source(data, repo_root, source_rel_path)
    if measured is None:
        return False

    ctxmap = data.contexts_by_lineno(measured)
    if not ctxmap:
        return False

    for line_no in lines:
        if line_no not in ctxmap:
            continue
        if not require_test_context or any(_is_pytest_test_context(ctx) for ctx in ctxmap[line_no]):
            return True
    return False
