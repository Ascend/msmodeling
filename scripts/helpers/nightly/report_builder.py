"""Nightly summary helpers: git env, test_map summary, weak-coverage detection.

Git env collection, test_map summary loading, and weak-coverage symbol
detection from coverage data.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pathlib

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.nightly.pytest_parser import (
    NightlyRunStats,
    extract_pytest_log_snippet,
    parse_junit_file,
)
from scripts.helpers.nightly.report_models import (
    EnvInfo,
    MapCoverageSummary,
    PhaseBreakdownEntry,
)

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(args: list[str], cwd: pathlib.Path) -> str:
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
    timestamp = datetime.now(timezone.utc).isoformat()
    return EnvInfo(commit=commit, branch=branch, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Test map summary
# ---------------------------------------------------------------------------


def load_test_map_summary(path: pathlib.Path | None) -> MapCoverageSummary:
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
    test_map_path: pathlib.Path | None,
    coverage_path: pathlib.Path,
    *,
    mapping: dict[str, dict[str, list[str]]] | None = None,
    threshold: float = 0.50,
) -> tuple[str, ...]:
    """Return symbols with local coverage below *threshold*.

    Reads test_map to enumerate symbols, then checks .coverage data
    to compute per-symbol line hit rates. Symbols below threshold
    are returned as ``"src_file::symbol_name"`` strings.
    """
    if mapping is None:
        if test_map_path is None or not test_map_path.is_file():
            return ()

        try:
            data = json.loads(test_map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ()

        mapping = data.get("map", data)
        if not isinstance(mapping, dict):
            return ()

    try:
        coverage_mod = importlib.import_module("coverage")
        cov = coverage_mod.Coverage(data_file=str(coverage_path))
        cov.load()
        coverage_data: Any = cov.get_data()
        if not coverage_data.measured_files():
            return ()
    except Exception:
        return ()

    weak: list[str] = []
    for src_file in mapping:
        abs_path = REPO_ROOT / src_file
        if not abs_path.is_file():
            continue
        from scripts.helpers.common import ast_utils

        spans = ast_utils.iter_qualified_definition_spans(abs_path)
        ctxmap = coverage_data.contexts_by_lineno(str(abs_path))
        for span in spans:
            qualified = f"{src_file}::{span.qualified_name}"
            span_lines = set(range(span.start_line, span.end_line + 1))
            hit = sum(1 for ln in span_lines if ctxmap and ln in ctxmap and ctxmap[ln])
            total = len(span_lines)
            if total > 0 and hit / total < threshold:
                weak.append(qualified)

    return tuple(sorted(weak))


# ---------------------------------------------------------------------------
# Phase breakdown and first-error resolution
# ---------------------------------------------------------------------------


def build_phase_breakdown(
    phase_labels: tuple[str, ...],
    junit_paths: tuple[pathlib.Path, ...],
    phase_exits: tuple[int, ...],
) -> tuple[PhaseBreakdownEntry, ...]:
    """Build per-phase stats from JUnit files and pytest exit codes."""
    if len(phase_labels) != len(junit_paths) or len(phase_labels) != len(phase_exits):
        msg = "phase_labels, junit_paths, and phase_exits must have equal length"
        raise ValueError(msg)

    entries: list[PhaseBreakdownEntry] = []
    for label, junit_path, exit_code in zip(phase_labels, junit_paths, phase_exits, strict=True):
        phase_stats = parse_junit_file(junit_path)
        if phase_stats is None:
            if exit_code != 0:
                entries.append(
                    PhaseBreakdownEntry(
                        label=label,
                        passed=0,
                        failed=0,
                        duration_sec=-1.0,
                        exit_code=exit_code,
                        infra_failure=True,
                    )
                )
            continue

        failed_count = phase_stats.failed + phase_stats.errors
        infra_failure = exit_code != 0 and failed_count == 0 and phase_stats.passed == 0
        entries.append(
            PhaseBreakdownEntry(
                label=label,
                passed=phase_stats.passed,
                failed=failed_count,
                duration_sec=phase_stats.duration_sec,
                exit_code=exit_code if exit_code != 0 else 0,
                infra_failure=infra_failure,
            )
        )
    return tuple(entries)


def resolve_first_error(
    stats: NightlyRunStats,
    phase_exits: tuple[int, ...],
    phase_log_paths: tuple[pathlib.Path | None, ...],
) -> str:
    """Return JUnit first error, or fall back to the first failing phase log."""
    if stats.first_error:
        return stats.first_error
    if len(phase_exits) != len(phase_log_paths):
        msg = "phase_exits and phase_log_paths must have equal length"
        raise ValueError(msg)

    for exit_code, log_path in zip(phase_exits, phase_log_paths, strict=True):
        if exit_code == 0 or log_path is None:
            continue
        snippet = extract_pytest_log_snippet(log_path)
        if snippet:
            return snippet
    return ""
