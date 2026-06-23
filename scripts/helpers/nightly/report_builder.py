"""Nightly summary helpers: git env, test_map summary, weak-coverage detection.

Git env collection, test_map summary loading, and weak-coverage symbol
detection from coverage data.
"""

from __future__ import annotations

import importlib
import json
from collections import defaultdict
from datetime import datetime

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pathlib

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.diff import git_stdout
from scripts.helpers.common.coverage_config import product_roots
from scripts.helpers.common.test_map_loader import parse_test_map_map_object
from scripts.helpers.common.test_map_report import (
    iter_unique_symbol_refs,
    summarize_test_map,
)
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


def fetch_env_info() -> EnvInfo:
    commit = git_stdout(REPO_ROOT, "rev-parse", "--short", "HEAD") or "unknown"
    branch = git_stdout(REPO_ROOT, "branch", "--show-current") or "unknown"
    timestamp = datetime.now(UTC).isoformat()
    return EnvInfo(commit=commit, branch=branch, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Test map summary
# ---------------------------------------------------------------------------


def load_test_map_summary(path: pathlib.Path | None) -> MapCoverageSummary:
    if path is None or not path.is_file():
        return MapCoverageSummary(test_nodes=0, symbol_refs=0)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return MapCoverageSummary(test_nodes=0, symbol_refs=0)
    if not isinstance(data, dict):
        return MapCoverageSummary(test_nodes=0, symbol_refs=0)
    inner = data.get("map", data)
    try:
        mapping = parse_test_map_map_object(inner, roots=product_roots())
    except Exception:
        return MapCoverageSummary(test_nodes=0, symbol_refs=0)
    if not mapping:
        return MapCoverageSummary(test_nodes=0, symbol_refs=0)
    return summarize_test_map(mapping)


# ---------------------------------------------------------------------------
# Weak-coverage detection
# ---------------------------------------------------------------------------


def _weak_symbols_for_file(
    src_file: str,
    abs_path: pathlib.Path,
    coverage_data: Any,
    *,
    symbols: set[str],
    threshold: float,
) -> list[str]:
    from scripts.helpers.common import ast_utils

    spans = ast_utils.iter_canonical_definition_spans(abs_path)
    ctxmap = coverage_data.contexts_by_lineno(str(abs_path))
    weak: list[str] = []
    for span in spans:
        if span.qualified_name not in symbols:
            continue
        span_lines = set(range(span.start_line, span.end_line + 1))
        hit = sum(1 for ln in span_lines if ctxmap and ln in ctxmap and ctxmap[ln])
        total = len(span_lines)
        if total > 0 and hit / total < threshold:
            weak.append(f"{src_file}::{span.qualified_name}")
    return weak


def _load_coverage_data(coverage_path: pathlib.Path) -> Any | None:
    try:
        coverage_mod = importlib.import_module("coverage")
        cov = coverage_mod.Coverage(data_file=str(coverage_path))
        cov.load()
        coverage_data: Any = cov.get_data()
    except Exception:
        return None
    if not coverage_data.measured_files():
        return None
    return coverage_data


def _resolve_node_map(
    test_map_path: pathlib.Path | None,
    mapping: dict[str, dict[str, list[str]]] | None,
) -> dict[str, dict[str, list[str]]] | None:
    if mapping is not None:
        return mapping
    if test_map_path is None or not test_map_path.is_file():
        return None
    try:
        data = json.loads(test_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    inner = data.get("map", data)
    try:
        loaded = parse_test_map_map_object(inner, roots=product_roots())
    except Exception:
        return None
    if not loaded:
        return None
    return loaded


def compute_weak_coverage_symbols(
    test_map_path: pathlib.Path | None,
    coverage_path: pathlib.Path,
    *,
    mapping: dict[str, dict[str, list[str]]] | None = None,
    threshold: float = 0.50,
) -> tuple[str, ...]:
    """Return symbols with local coverage below *threshold*.

    Enumerates unique ``source_file::symbol`` refs from the node-oriented
    test_map, then checks .coverage line hit rates per symbol span.
    """
    node_map = _resolve_node_map(test_map_path, mapping)
    if node_map is None:
        return ()

    refs_by_file: dict[str, set[str]] = defaultdict(set)
    for src_file, symbol in iter_unique_symbol_refs(node_map):
        refs_by_file[src_file].add(symbol)

    coverage_data = _load_coverage_data(coverage_path)
    if coverage_data is None:
        return ()

    weak: list[str] = []
    for src_file, symbols in sorted(refs_by_file.items()):
        abs_path = REPO_ROOT / src_file
        if not abs_path.is_file():
            continue
        weak.extend(
            _weak_symbols_for_file(
                src_file,
                abs_path.resolve(),
                coverage_data,
                symbols=symbols,
                threshold=threshold,
            )
        )

    return tuple(sorted(weak))


# ---------------------------------------------------------------------------
# Phase breakdown and first-error resolution
# ---------------------------------------------------------------------------


def build_phase_breakdown(
    phase_labels: tuple[str, ...],
    junit_paths: tuple[pathlib.Path, ...],
    phase_exits: tuple[int, ...],
    *,
    phase_stats: tuple[NightlyRunStats | None, ...] | None = None,
) -> tuple[PhaseBreakdownEntry, ...]:
    """Build per-phase stats from JUnit files and pytest exit codes."""
    if len(phase_labels) != len(junit_paths) or len(phase_labels) != len(phase_exits):
        msg = "phase_labels, junit_paths, and phase_exits must have equal length"
        raise ValueError(msg)

    if phase_stats is None:
        phase_stats = tuple(parse_junit_file(junit_path) for junit_path in junit_paths)
    elif len(phase_stats) != len(junit_paths):
        msg = "phase_stats length must match junit_paths"
        raise ValueError(msg)

    entries: list[PhaseBreakdownEntry] = []
    for label, junit_path, exit_code, parsed_stats in zip(
        phase_labels, junit_paths, phase_exits, phase_stats, strict=True
    ):
        if parsed_stats is None:
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

        failed_count = parsed_stats.failed + parsed_stats.errors
        infra_failure = exit_code != 0 and failed_count == 0 and parsed_stats.passed == 0
        entries.append(
            PhaseBreakdownEntry(
                label=label,
                passed=parsed_stats.passed,
                failed=failed_count,
                duration_sec=parsed_stats.duration_sec,
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
