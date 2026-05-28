#!/usr/bin/env python3
"""CI incremental gate: diff analysis, test selection, pytest, coverage gate.

CLI entry point for run_ci_gate.sh. Orchestrates diff → gate plan → execution.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.diff import classify_changes, fetch_diff_line_map, resolve_base_ref
from scripts.helpers.ci_gate.models import ChangeSet, CiGatePlan, GateStepResult
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
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.coverage_gate import GateConfig, check_ut_gate, load_totals
from scripts.helpers.common.test_map_loader import load_baseline, prune_deleted_sources

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

_PYTEST_MARKER = "not npu and not nightly"
_SYMBOL_COVERAGE_THRESHOLD = 0.50


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
    try:
        _ = load_totals(coverage_path)
    except (FileNotFoundError, RuntimeError):
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
# Plan building
# ---------------------------------------------------------------------------


def build_ci_gate_plan(repo_root: Path, changes: ChangeSet, baseline) -> CiGatePlan:
    """Apply all gate rules, return a single CiGatePlan.

    Single source of truth — no duplicate logic with apply_gates.
    """
    test_map = baseline.test_map
    exemptions = baseline.exemptions
    prefixes = baseline.product_prefixes

    blocking: list[str] = []
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


def apply_gates(repo_root: Path, changes: ChangeSet, baseline) -> GateStepResult:
    """Legacy wrapper — delegates to build_ci_gate_plan."""
    plan = build_ci_gate_plan(repo_root, changes, baseline)
    return GateStepResult(
        errors=plan.blocking_errors,
        tests=plan.incremental_tests,
        cross_layer_deferred=frozenset(),
        full_suite=plan.full_suite,
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
        "-vv",
        "--durations=20",
    ]
    if coverage:
        cmd.extend(cov_pytest_args(append=append))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def emit_selection_report(plan: CiGatePlan, merge_base: str) -> None:
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "merge_base": merge_base,
        "full_suite": plan.full_suite,
        "deleted_source_tests": sorted(plan.deleted_source_tests),
        "incremental_tests": sorted(plan.incremental_tests),
        "blocking_errors": list(plan.blocking_errors),
        "symbol_warnings": list(plan.symbol_warnings),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _print_blocking_errors(errors: tuple[str, ...]) -> None:
    print("CI gate failed: policy violation — no tests executed.", file=sys.stderr)
    print("Resolve the following before merge:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)


def _print_deleted_source_failure(tests: frozenset[str]) -> None:
    print(
        "CI gate failed: deleted product source but corresponding tests still fail (delete or update orphaned tests).",
        file=sys.stderr,
    )
    print("Guard tests for deleted sources:", file=sys.stderr)
    for test_id in sorted(tests):
        print(f"  - {test_id}", file=sys.stderr)


def _print_source_change_failure() -> None:
    print(
        "CI gate failed: source change caused test failure(s). See pytest output above for all failed cases.",
        file=sys.stderr,
    )


def _print_coverage_failure(message: str) -> None:
    print("CI gate failed: coverage below threshold.", file=sys.stderr)
    print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# Coverage gate
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


def _check_coverage_gate(cfg: Config) -> int:
    passed, message = check_ut_gate(config=GateConfig.from_config(cfg))
    if passed:
        print(message)
        return 0
    _print_coverage_failure(message)
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = Config.from_env()

    try:
        baseline = load_baseline(REPO_ROOT, cfg)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    merge_base = resolve_base_ref(REPO_ROOT, cfg.base_branch)
    diff = fetch_diff_line_map(REPO_ROOT, merge_base)
    changes = classify_changes(REPO_ROOT, merge_base, diff)
    plan = build_ci_gate_plan(REPO_ROOT, changes, baseline)

    emit_selection_report(plan, merge_base)

    if plan.blocking_errors:
        _print_blocking_errors(plan.blocking_errors)
        return 1

    ran_deleted_phase = False
    if plan.deleted_source_tests:
        ran_deleted_phase = True
        code = _run_pytest(sorted(plan.deleted_source_tests), coverage=True, append=False)
        if code != 0:
            _print_deleted_source_failure(plan.deleted_source_tests)
            return code

    if plan.full_suite:
        phase2_targets = ["tests/smoke/", "tests/regression/"]
    elif plan.incremental_tests:
        phase2_targets = sorted(plan.incremental_tests)
    else:
        phase2_targets = []

    if phase2_targets:
        code = _run_pytest(phase2_targets, coverage=True, append=ran_deleted_phase)
        if code != 0:
            _print_source_change_failure()
            return code

    coverage_path = REPO_ROOT / ".coverage"
    if not _has_coverage_data(coverage_path):
        print("No coverage data produced; skipping coverage gate", file=sys.stderr)
        return 0

    symbol_warnings = _check_symbol_level_coverage(changes, coverage_path)
    if symbol_warnings:
        print("Per-symbol coverage warnings (advisory, not blocking):", file=sys.stderr)
        for w in symbol_warnings:
            print(f"  - {w}", file=sys.stderr)
        plan = CiGatePlan(
            blocking_errors=plan.blocking_errors,
            deleted_source_tests=plan.deleted_source_tests,
            incremental_tests=plan.incremental_tests,
            full_suite=plan.full_suite,
            symbol_warnings=symbol_warnings,
        )

    return _check_coverage_gate(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
