"""Check whether changed source lines were executed in Coverage.py data."""

from __future__ import annotations

from pathlib import Path


def _measured_path_for_source(data: object, repo_root: Path, source_rel_path: str) -> str | None:
    target = (repo_root / source_rel_path).resolve()
    measured_files = getattr(data, "measured_files", None)
    if measured_files is None:
        return None
    for measured in measured_files():
        if Path(measured).resolve() == target:
            return measured
    return None


def symbol_lines_covered_in_data(
    repo_root: Path,
    source_rel_path: str,
    symbol: str,
    lines: set[int],
    coverage_path: Path,
) -> bool:
    """Return True when any *lines* was executed (any coverage context).

    Context may be a pytest node id, empty string (import-time), or conftest.
    *symbol* is part of the public API for call-site clarity.
    """
    _ = symbol
    if not lines or not coverage_path.is_file():
        return False

    from coverage.data import CoverageData
    from coverage.misc import CoverageException

    data = CoverageData(str(coverage_path))
    try:
        data.read()
    except (CoverageException, OSError):
        return False

    measured = _measured_path_for_source(data, repo_root, source_rel_path)
    if measured is None:
        return False

    ctxmap = data.contexts_by_lineno(measured)
    if not ctxmap:
        return False

    for line_no in lines:
        contexts = ctxmap.get(line_no)
        if contexts:
            return True
    return False
