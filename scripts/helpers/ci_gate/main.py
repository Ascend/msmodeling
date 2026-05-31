#!/usr/bin/env python3
"""CI incremental gate: diff analysis, test selection, pytest, coverage gate.

CLI entry point for run_ci_gate.sh. Orchestrates diff → gate plan → execution.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.helpers._config import Config, ConfigError
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.diff import classify_changes, fetch_diff_line_map, resolve_base_ref
from scripts.helpers.ci_gate.errors import format_blocking_errors
from scripts.helpers.ci_gate.gate_policy import validate_gate_policy_if_changed
from scripts.helpers.ci_gate.models import ChangeSet, CiGatePlan, GateError, GateStepResult
from scripts.helpers.ci_gate.rules import (
    _merge_step_results,
    _product_paths,
    gate_config,
    gate_deleted_source,
    gate_deleted_tests,
    gate_modified_source,
    gate_new_source,
    gate_new_tests,
)
from scripts.helpers.common import ast_utils
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.build_test_map import collect_test_map
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.coverage_gate import GateConfig, check_ut_gate
from scripts.helpers.common.test_map_loader import load_baseline, prune_deleted_sources

_PYTEST_MARKER = "not npu"
_SYMBOL_COVERAGE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Coverage data probe
# ---------------------------------------------------------------------------


def _has_coverage_data(coverage_path: Path) -> bool:
    """Check if coverage data exists (handles parallel mode .coverage.* files)."""
    from coverage.data import CoverageData

    try:
        data = CoverageData(str(coverage_path))
        data.read()
        return bool(data.measured_files())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-symbol coverage check
# ---------------------------------------------------------------------------


def _check_symbol_level_coverage(
    changes: ChangeSet,
    coverage_path: Path,
    *,
    threshold: float = _SYMBOL_COVERAGE_THRESHOLD,
) -> tuple[str, ...]:
    """Check per-symbol local coverage for changed symbols.

    Returns a tuple of warning strings (advisory, not blocking).
    """
    if not _has_coverage_data(coverage_path):
        return ()

    from coverage.data import CoverageData

    try:
        data = CoverageData(str(coverage_path))
        data.read()
    except Exception:
        return ()

    warnings: list[str] = []
    for path, raw_lines in changes.modified_source:
        abs_path = REPO_ROOT / path
        if not abs_path.is_file():
            continue
        spans = ast_utils.iter_qualified_definition_spans(abs_path)
        changed_lines = set(raw_lines)
        ctxmap = data.contexts_by_lineno(str(abs_path))
        for span in spans:
            span_lines = set(range(span.start_line, span.end_line + 1))
            relevant = changed_lines & span_lines
            if not relevant:
                continue
            hit = 0
            for line_no in relevant:
                if ctxmap and line_no in ctxmap and ctxmap[line_no]:
                    hit += 1
            local_pct = hit / len(relevant) if relevant else 1.0
            if local_pct < threshold:
                warnings.append(f"{path}::{span.qualified_name}: local coverage {local_pct:.0%} < {threshold:.0%}")

    return tuple(warnings)


# ---------------------------------------------------------------------------
# New test map: run new tests → collect map in memory → merge with baseline
# ---------------------------------------------------------------------------


def _remap_renamed_sources(
    test_map: dict[str, dict[str, list[str]]],
    renames: tuple[tuple[str, str, int], ...],
) -> dict[str, dict[str, list[str]]]:
    """Move test_map entries from old path to new path for renamed sources.

    Lets a pure rename pass with no test churn and a renamed-with-edits source
    resolve its unchanged symbols against the moved map entry.
    """
    remapped = dict(test_map)
    for old_path, new_path, _score in renames:
        if old_path in remapped:
            remapped[new_path] = remapped.pop(old_path)
    return remapped


def _merge_test_maps(
    baseline: dict[str, dict[str, list[str]]],
    new_map: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    merged: dict[str, dict[str, list[str]]] = {}
    for key, syms in baseline.items():
        merged[key] = dict(syms)
    for key, syms in new_map.items():
        if key in merged:
            merged[key].update(syms)
        else:
            merged[key] = dict(syms)
    return merged


def _run_new_tests_and_build_map(
    new_tests: tuple[str, ...],
    marker_expr: str,
) -> dict[str, dict[str, list[str]]]:
    logger = logging.getLogger("ci_gate")
    logger.info("Phase 0: running %d new test(s) to build new test_map ...", len(new_tests))

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *new_tests,
        "-m",
        marker_expr,
        "-n0",
        *cov_pytest_args(cov_context=True),
        "-q",
        "--no-header",
        "--tb=long",
        "--durations=20",
    ]
    logger.info("Running pytest: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        print("CI gate failed: new test(s) failed. Fix test failures before gate check.")
        raise SystemExit(proc.returncode)

    new_test_map = collect_test_map(marker_expr=marker_expr)
    logger.info(
        "Phase 0: new test_map — %d source files, %d symbols",
        len(new_test_map),
        sum(len(s) for s in new_test_map.values()),
    )
    return new_test_map


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def build_ci_gate_plan(repo_root: Path, changes: ChangeSet, baseline) -> CiGatePlan:
    """Apply all gate rules, return a single CiGatePlan.

    Single source of truth — no duplicate logic with apply_gates.
    """
    test_map = baseline.test_map
    exemptions = baseline.exemptions
    prefixes = baseline.product_prefixes

    blocking: list[GateError] = []
    results: list[GateStepResult] = []
    full_suite = False

    if changes.config:
        results.append(gate_config())
        full_suite = True

    has_del_source = bool(_product_paths(changes.del_source, prefixes))
    has_new_source = bool(_product_paths(changes.new_source, prefixes))

    deleted_source_tests: frozenset[str] = frozenset()

    if has_del_source:
        del_step = gate_deleted_source(changes, test_map, prefixes)
        blocking.extend(del_step.errors)
        deleted_source_tests = del_step.all_tests

    if has_new_source:
        effective_map = test_map
        if has_del_source:
            effective_map = prune_deleted_sources(test_map, changes.del_source)
        new_step = gate_new_source(repo_root, changes, effective_map, exemptions, prefixes)
        blocking.extend(new_step.errors)

    if changes.del_test:
        del_test_step = gate_deleted_tests(changes, test_map)
        blocking.extend(del_test_step.errors)

    if changes.new_test:
        results.append(gate_new_tests(changes))

    if changes.modified_source:
        mod_step = gate_modified_source(repo_root, changes, test_map, exemptions, prefixes)
        blocking.extend(mod_step.errors)
        results.append(mod_step)

    merged = _merge_step_results(*results)
    incremental_tests = merged.all_tests - deleted_source_tests

    return CiGatePlan(
        blocking_errors=tuple(blocking),
        deleted_source_tests=deleted_source_tests,
        incremental_tests=incremental_tests,
        full_suite=full_suite,
    )


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------


def _run_pytest(targets: list[str], *, coverage: bool, append: bool = False) -> int:
    if not targets:
        return 0
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-m",
        _PYTEST_MARKER,
        "-n",
        "auto",
        "-q",
        "--no-header",
        "--tb=long",
        "--durations=20",
    ]
    if coverage:
        cmd.extend(cov_pytest_args(append=append))
    logger = logging.getLogger("ci_gate")
    logger.info("Running pytest: %s", shlex.join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _log_blocking_errors(logger: logging.Logger, errors: tuple[GateError, ...]) -> None:
    logger.error("%s", format_blocking_errors(errors))


def _log_deleted_source_failure(logger: logging.Logger, tests: frozenset[str]) -> None:
    logger.error(
        "CI gate failed: deleted product source but corresponding tests still fail (delete or update orphaned tests)."
    )
    logger.error("Guard tests for deleted sources:")
    for test_id in sorted(tests):
        logger.error("  - %s", test_id)


def _log_source_change_failure(logger: logging.Logger) -> None:
    logger.error("CI gate failed: source change caused test failure(s). See pytest output above for all failed cases.")


def _log_coverage_failure(logger: logging.Logger, message: str) -> None:
    logger.error("CI gate failed: coverage below threshold.")
    logger.error("%s", message)


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------


def _check_coverage_gate(cfg: Config, logger: logging.Logger) -> tuple[int, str]:
    passed, message = check_ut_gate(config=GateConfig.from_config(cfg))
    if passed:
        logger.info("%s", message)
        return 0, message
    _log_coverage_failure(logger, message)
    return 1, message


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logger = setup_logger()
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    logger.info("Resolving merge-base against %s ...", cfg.base_branch)
    try:
        merge_base = resolve_base_ref(REPO_ROOT, cfg.base_branch)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Merge-base: %s", merge_base[:12])

    try:
        validate_gate_policy_if_changed(REPO_ROOT, merge_base)
        baseline = load_baseline(REPO_ROOT, cfg)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Fetching diff ...")
    diff = fetch_diff_line_map(REPO_ROOT, merge_base)
    logger.info("Diff: %d files changed", len(diff))

    logger.info("Classifying changes ...")
    changes = classify_changes(REPO_ROOT, merge_base, diff, baseline.discovery)
    logger.info(
        "Changes: config=%d new_test=%d del_test=%d new_source=%d del_source=%d modified=%d",
        len(changes.config),
        len(changes.new_test),
        len(changes.del_test),
        len(changes.new_source),
        len(changes.del_source),
        len(changes.modified_source),
    )

    if changes.renames:
        logger.info("Remapping test_map for %d renamed source(s) ...", len(changes.renames))
        baseline = baseline.__class__(
            test_map=_remap_renamed_sources(baseline.test_map, changes.renames),
            exemptions=baseline.exemptions,
            discovery=baseline.discovery,
            product_prefixes=baseline.product_prefixes,
        )

    if changes.new_test:
        new_test_map = _run_new_tests_and_build_map(changes.new_test, _PYTEST_MARKER)
        merged_map = _merge_test_maps(baseline.test_map, new_test_map)
        baseline = baseline.__class__(
            test_map=merged_map,
            exemptions=baseline.exemptions,
            discovery=baseline.discovery,
            product_prefixes=baseline.product_prefixes,
        )

    logger.info("Building gate plan ...")
    plan = build_ci_gate_plan(REPO_ROOT, changes, baseline)

    if plan.blocking_errors:
        _log_blocking_errors(logger, plan.blocking_errors)
        return 1

    ran_deleted_phase = False
    if plan.deleted_source_tests:
        ran_deleted_phase = True
        logger.info(
            "Phase 1: running %d deleted-source guard tests ...",
            len(plan.deleted_source_tests),
        )
        code = _run_pytest(sorted(plan.deleted_source_tests), coverage=True, append=False)
        if code != 0:
            _log_deleted_source_failure(logger, plan.deleted_source_tests)
            return code
        logger.info("Phase 1: passed")

    if plan.full_suite:
        phase2_targets = ["tests/smoke/", "tests/regression/"]
    elif plan.incremental_tests:
        phase2_targets = sorted(plan.incremental_tests)
    else:
        phase2_targets = []

    if phase2_targets:
        logger.info("Phase 2: running %d test targets ...", len(phase2_targets))
        code = _run_pytest(phase2_targets, coverage=True, append=ran_deleted_phase)
        if code != 0:
            _log_source_change_failure(logger)
            return code
        logger.info("Phase 2: passed")

    coverage_path = REPO_ROOT / ".coverage"
    if not _has_coverage_data(coverage_path):
        logger.warning("No coverage data produced; skipping coverage gate")
        print("CI gate passed: no coverage data to check")
        return 0

    logger.info("Checking per-symbol coverage ...")
    symbol_warnings = _check_symbol_level_coverage(changes, coverage_path)
    if symbol_warnings:
        logger.warning("Per-symbol coverage warnings (advisory, not blocking):")
        for warning in symbol_warnings:
            logger.warning("  - %s", warning)

    logger.info("Checking coverage gate ...")
    exit_code, message = _check_coverage_gate(cfg, logger)
    if exit_code == 0:
        print(message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
