"""Nightly summary helpers: git env, test_map summary, weak-coverage detection.

Git env collection, test_map summary loading, and weak-coverage symbol
detection from coverage data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.nightly.report_models import EnvInfo, MapCoverageSummary

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(args: list[str], cwd: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found")
    result = subprocess.run(
        [git, *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    return result.stdout.strip()


def fetch_env_info() -> EnvInfo:
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd=REPO_ROOT) or "unknown"
    branch = _run_git(["branch", "--show-current"], cwd=REPO_ROOT) or "unknown"
    timestamp = datetime.now(UTC).isoformat()
    return EnvInfo(commit=commit, branch=branch, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Test map summary
# ---------------------------------------------------------------------------


def load_test_map_summary(path: Path | None) -> MapCoverageSummary:
    if path is None or not path.is_file():
        return MapCoverageSummary(source_files=0, symbols=0)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return MapCoverageSummary(source_files=0, symbols=0)
    mapping = data.get("map", data)
    if not isinstance(mapping, dict):
        return MapCoverageSummary(source_files=0, symbols=0)
    symbols = sum(len(syms) for syms in mapping.values())
    return MapCoverageSummary(source_files=len(mapping), symbols=symbols)


# ---------------------------------------------------------------------------
# Weak-coverage detection
# ---------------------------------------------------------------------------


def compute_weak_coverage_symbols(
    test_map_path: Path | None,
    coverage_path: Path,
    *,
    threshold: float = 0.50,
) -> tuple[str, ...]:
    """Return symbols with local coverage below *threshold*.

    Reads test_map to enumerate symbols, then checks .coverage data
    to compute per-symbol line hit rates. Symbols below threshold
    are returned as ``"src_file::symbol_name"`` strings.
    """
    if test_map_path is None or not test_map_path.is_file():
        return ()

    try:
        data = json.loads(test_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    mapping = data.get("map", data)
    if not isinstance(mapping, dict):
        return ()

    from coverage.data import CoverageData

    try:
        cov = CoverageData(str(coverage_path))
        cov.read()
        if not cov.measured_files():
            return ()
    except Exception:
        return ()

    weak: list[str] = []
    for src_file, symbols in mapping.items():
        abs_path = REPO_ROOT / src_file
        if not abs_path.is_file():
            continue
        from scripts.helpers.common import ast_utils

        spans = ast_utils.iter_qualified_definition_spans(abs_path)
        ctxmap = cov.contexts_by_lineno(str(abs_path))
        for span in spans:
            qualified = f"{src_file}::{span.qualified_name}"
            span_lines = set(range(span.start_line, span.end_line + 1))
            hit = sum(1 for ln in span_lines if ctxmap and ln in ctxmap and ctxmap[ln])
            total = len(span_lines)
            if total > 0 and hit / total < threshold:
                weak.append(qualified)

    return tuple(sorted(weak))
