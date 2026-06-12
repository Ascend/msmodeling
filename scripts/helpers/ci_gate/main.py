#!/usr/bin/env python3
"""CI incremental gate: diff analysis, test selection, pytest.

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
from scripts.helpers.ci_gate.errors import format_blocking_errors, format_phase0_failure_hint
from scripts.helpers.ci_gate.gate_policy import TestExemption, is_test_exempt, validate_gate_policy_if_changed
from scripts.helpers.ci_gate.models import Baseline, ChangeSet, CiGatePlan, GateError, GateStepResult
from scripts.helpers.ci_gate.rules import (
    _merge_step_results,
    _product_paths,
    gate_deleted_source,
    gate_deleted_tests,
    gate_modified_source,
    gate_new_source,
    gate_new_tests,
)
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.build_test_map import collect_test_map
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.pytest_runner import build_pytest_cmd, collect_test_node_ids, count_collected_tests
from scripts.helpers.common.test_map_loader import load_baseline, prune_deleted_sources

_INCREMENTAL_MARKER = "not npu and not nightly and not network"
_PHASE0_MARKER = "not npu"
_FULL_SUITE_MARKER = "not npu"
# Keep NPU-marked coverage contexts mappable when coverage data comes from an NPU-capable run.
_TEST_MAP_MARKER = "not nightly and not network"


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
        merged[key] = {symbol: list(test_ids) for symbol, test_ids in syms.items()}
    for key, syms in new_map.items():
        if key not in merged:
            merged[key] = {}
        for symbol, test_ids in syms.items():
            existing = merged[key].get(symbol, [])
            seen = set(existing)
            merged[key][symbol] = existing + [test_id for test_id in test_ids if test_id not in seen]
    return merged


def _run_new_tests_and_build_map(
    new_tests: tuple[str, ...],
    test_map_marker_expr: str,
    *,
    roots: tuple[str, ...],
    test_exemptions: tuple[TestExemption, ...] = (),
) -> tuple[dict[str, dict[str, list[str]]], bool]:
    logger = logging.getLogger("ci_gate")
    logger.info(
        "Phase 0 start: %d new/modified test file(s) in PR -> run non-exempt pytest node(s) with coverage -> refresh test_map",
        len(new_tests),
    )
    logger.info(
        "Phase 0 failure means: new/modified tests failed, or register exemptions.tests in tests/.ci/gate_policy.yaml",
    )

    nodes_to_run: list[str] = []
    for test_file in new_tests:
        collected = collect_test_node_ids([test_file], marker=_PHASE0_MARKER)
        runnable = tuple(node_id for node_id in collected if not is_test_exempt(test_exemptions, node_id))
        if collected and not runnable:
            logger.info(
                "Phase 0 skip %s: all %d node(s) listed in exemptions.tests (not a failure)",
                test_file,
                len(collected),
            )
        nodes_to_run.extend(runnable)

    if not nodes_to_run:
        logger.info(
            "Phase 0 skip pytest: no runnable nodes (all exempt or none collected); not a failure",
        )
        return {}, False

    collected_count = len(nodes_to_run)
    cmd = build_pytest_cmd(
        sys.executable,
        nodes_to_run,
        marker=_PHASE0_MARKER,
        collected_count=collected_count,
        extra_args=cov_pytest_args(cov_context=True),
    )
    logger.info("Running pytest: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        logger.error(
            "Phase 0 failed: new/modified test pytest run failed; fix tests or register exemptions.tests (see template below)",
        )
        print(format_phase0_failure_hint(tuple(nodes_to_run)))
        raise SystemExit(proc.returncode)

    new_test_map: dict[str, dict[str, list[str]]] = collect_test_map(
        marker_expr=test_map_marker_expr,
        coverage_path=REPO_ROOT / ".coverage",
        roots=roots,
    )
    logger.info(
        "Phase 0: new test_map — %d source files, %d symbols",
        len(new_test_map),
        sum(len(s) for s in new_test_map.values()),
    )
    return new_test_map, True


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def build_ci_gate_plan(
    repo_root: Path,
    changes: ChangeSet,
    baseline: Baseline,
    *,
    coverage_path: Path | None = None,
    phase0_ran_pytest: bool = False,
) -> CiGatePlan:
    """Apply all gate rules, return a single CiGatePlan.

    Single source of truth — no duplicate logic with apply_gates.
    """
    test_map = baseline.test_map
    exemptions = baseline.exemptions
    roots = baseline.roots

    blocking: list[GateError] = []
    results: list[GateStepResult] = []
    full_suite = False

    if changes.config:
        full_suite = True

    has_del_source = bool(_product_paths(changes.del_source, roots))
    has_new_source = bool(_product_paths(changes.new_source, roots))

    deleted_source_tests: frozenset[str] = frozenset()

    if has_del_source:
        del_step = gate_deleted_source(changes, test_map, roots)
        blocking.extend(del_step.errors)
        deleted_source_tests = del_step.tests

    if has_new_source:
        effective_map = test_map
        if has_del_source:
            effective_map = prune_deleted_sources(test_map, changes.del_source)
        new_step = gate_new_source(
            repo_root,
            changes,
            effective_map,
            exemptions,
            roots,
            coverage_path=coverage_path,
        )
        blocking.extend(new_step.errors)

    if changes.del_test:
        del_test_step = gate_deleted_tests(changes, test_map)
        blocking.extend(del_test_step.errors)

    if (changes.new_test or changes.modified_test) and not phase0_ran_pytest:
        results.append(gate_new_tests(changes))

    if changes.modified_source:
        mod_step = gate_modified_source(
            repo_root,
            changes,
            test_map,
            exemptions,
            roots,
            coverage_path=coverage_path,
        )
        blocking.extend(mod_step.errors)
        results.append(mod_step)

    merged = _merge_step_results(*results)
    incremental_tests = merged.tests - deleted_source_tests

    return CiGatePlan(
        blocking_errors=tuple(blocking),
        deleted_source_tests=deleted_source_tests,
        incremental_tests=incremental_tests,
        full_suite=full_suite,
    )


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------


def _run_pytest(targets: list[str], *, marker: str) -> int:
    if not targets:
        return 0
    collected = count_collected_tests(targets, marker=marker)
    cmd = build_pytest_cmd(
        sys.executable,
        targets,
        marker=marker,
        collected_count=collected,
    )
    logger = logging.getLogger("ci_gate")
    logger.info("Running pytest: %s", shlex.join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _log_blocking_errors(logger: logging.Logger, errors: tuple[GateError, ...]) -> None:
    counts: dict[str, int] = {}
    for err in errors:
        counts[err.category] = counts.get(err.category, 0) + 1
    summary = ", ".join(f"{category}={count}" for category, count in sorted(counts.items()))
    logger.error(
        "Policy validation failed; Phase 1/2 skipped (%d violation(s): %s)",
        len(errors),
        summary or "unknown",
    )
    logger.error(
        "Meaning: product change lacks test_map coverage, or sources exemption not listed in gate_policy.yaml; see list below",
    )
    logger.error("%s", format_blocking_errors(errors))


def _log_deleted_source_failure(logger: logging.Logger, tests: frozenset[str]) -> None:
    logger.error(
        "Phase 1 failed: product source deleted but guard tests did not pass (orphan tests must be removed or updated)",
    )
    logger.error("Guard test list:")
    for test_id in sorted(tests):
        logger.error("  - %s", test_id)


def _log_source_change_failure(logger: logging.Logger, *, full_suite: bool) -> None:
    if full_suite:
        logger.error(
            "Phase 2 full suite failed: full regression triggered by config/conftest/dependency changes did not pass; see pytest output above",
        )
    else:
        logger.error(
            "Phase 2 incremental failed: product change broke test_map mapped regression tests; see pytest output above",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logger = setup_logger()
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    logger.info(
        "CI gate flow: Phase0(new/mod tests->test_map) -> policy validation -> Phase1(deleted source guard) -> Phase2(incremental/full regression)",
    )

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
    changes = classify_changes(REPO_ROOT, merge_base, diff, baseline.discovery, baseline.roots)
    logger.info(
        "Changes: config=%d new_test=%d mod_test=%d del_test=%d new_source=%d del_source=%d modified=%d",
        len(changes.config),
        len(changes.new_test),
        len(changes.modified_test),
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
            test_exemptions=baseline.test_exemptions,
            discovery=baseline.discovery,
            roots=baseline.roots,
        )

    coverage_path: Path | None = None
    phase0_ran_pytest = False
    tests_to_remap = changes.new_test + changes.modified_test
    if tests_to_remap:
        logger.info(
            "Detected %d new/modified test file(s); entering Phase 0 to build test_map ...",
            len(tests_to_remap),
        )
        new_test_map, ran_phase0 = _run_new_tests_and_build_map(
            tests_to_remap,
            _TEST_MAP_MARKER,
            roots=baseline.roots,
            test_exemptions=baseline.test_exemptions,
        )
        if ran_phase0:
            phase0_ran_pytest = True
            merged_map = _merge_test_maps(baseline.test_map, new_test_map)
            coverage_path = REPO_ROOT / ".coverage"
            baseline = baseline.__class__(
                test_map=merged_map,
                exemptions=baseline.exemptions,
                test_exemptions=baseline.test_exemptions,
                discovery=baseline.discovery,
                roots=baseline.roots,
            )

    logger.info("Building gate plan ...")
    plan = build_ci_gate_plan(
        REPO_ROOT,
        changes,
        baseline,
        coverage_path=coverage_path,
        phase0_ran_pytest=phase0_ran_pytest,
    )

    if plan.blocking_errors:
        _log_blocking_errors(logger, plan.blocking_errors)
        return 1

    if plan.deleted_source_tests:
        logger.info(
            "Phase 1 start: %d product source file(s) deleted -> run test_map guard tests (expect failure to expose orphan tests)",
            len(plan.deleted_source_tests),
        )
        logger.info(
            "Phase 1 failure means: product source deleted but mapped guard tests still failed (delete or update orphan tests)"
        )
        code = _run_pytest(sorted(plan.deleted_source_tests), marker=_INCREMENTAL_MARKER)
        if code != 0:
            _log_deleted_source_failure(logger, plan.deleted_source_tests)
            return code
        logger.info("Phase 1 passed")

    if plan.full_suite:
        logger.info("Phase 2 full suite: config/conftest/dependency change -> run all non-NPU tests under tests/")
        logger.info("Phase 2 failure means: product change broke existing regression tests")
        code = _run_pytest(["tests"], marker=_FULL_SUITE_MARKER)
        if code != 0:
            _log_source_change_failure(logger, full_suite=True)
            return code
        logger.info("Phase 2 full suite passed")
    elif plan.incremental_tests:
        phase2_targets = sorted(
            test_id for test_id in plan.incremental_tests if not is_test_exempt(baseline.test_exemptions, test_id)
        )
        if not phase2_targets:
            logger.info("Phase 2 skip: all incremental targets listed in exemptions.tests (not a failure)")
        else:
            logger.info(
                "Phase 2 incremental: product code change -> run %d test_map selected test(s)",
                len(phase2_targets),
            )
            logger.info("Phase 2 failure means: product change broke mapped regression tests")
            code = _run_pytest(phase2_targets, marker=_INCREMENTAL_MARKER)
            if code != 0:
                _log_source_change_failure(logger, full_suite=False)
                return code
            logger.info("Phase 2 incremental passed")

    print("CI gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
