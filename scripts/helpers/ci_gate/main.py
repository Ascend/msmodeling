#!/usr/bin/env python3
"""CI incremental gate: diff analysis, test selection, pytest.

CLI entry point for run_ci_gate.sh. Orchestrates diff → gate plan → execution.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from collections import Counter
from typing import TYPE_CHECKING

from scripts.helpers._config import Config, ConfigError
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.diff import classify_changes, fetch_diff, resolve_base_ref
from scripts.helpers.ci_gate.errors import format_blocking_errors, format_pytest_failure_hint
from scripts.helpers.ci_gate.gate_policy import (
    SourceExemption,
    TestExemption,
    gate_policy_changed_in_diff,
    is_test_exempt,
    validate_gate_policy_if_changed,
)
from scripts.helpers.ci_gate.models import (
    Baseline,
    ChangeSet,
    CiGatePlan,
    ExecutionPlan,
    GateError,
    TestRunWave,
)
from scripts.helpers.ci_gate.rules import (
    _product_paths,
    gate_deleted_source,
    gate_deleted_tests,
    gate_modified_source,
    gate_new_source,
    gate_new_tests,
    gate_unscoped_source,
)
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.pytest_runner import (
    build_pytest_cmd,
    count_collected_tests,
    filter_collectable_node_ids,
)
from scripts.helpers.common.test_map_loader import load_baseline, prune_deleted_sources, validate_test_map_freshness

if TYPE_CHECKING:
    from pathlib import Path

_CHANGED_TEST_MARKER = "not npu"
_REGRESSION_MARKER = "not npu and not nightly and not network"
_FULL_SUITE_MARKER = "not npu"
_COVERAGE_DATA_PATH = REPO_ROOT / ".coverage"
_SAMPLE_NODE_LIMIT = 3

_REASON_CONFIG = "dependency or test configuration changed"
_REASON_CHANGED_TEST = "new or changed test file"
_REASON_REGRESSION = "changed product file mapped regression"
_REASON_DELETED_SOURCE = "deleted product file guard test"


def _remap_renamed_sources(
    test_map: dict[str, dict[str, list[str]]],
    renames: tuple[tuple[str, str, int], ...],
) -> dict[str, dict[str, list[str]]]:
    """Move coverage-mapping entries from old path to new path for renamed sources."""
    remapped = dict(test_map)
    for old_path, new_path, score in renames:
        if score < 100:
            continue
        if old_path in remapped:
            remapped[new_path] = remapped.pop(old_path)
    return remapped


def build_hard_blocking_plan(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    roots: tuple[str, ...],
) -> tuple[GateError, ...]:
    """Pre-run hard policy: deleted tests/sources and mapping gaps that block pytest."""
    blocking: list[GateError] = []

    if _product_paths(changes.del_source, roots):
        blocking.extend(gate_deleted_source(changes, test_map, roots).errors)

    if changes.del_test:
        blocking.extend(gate_deleted_tests(changes, test_map).errors)

    if changes.unscoped_source:
        blocking.extend(gate_unscoped_source(changes).errors)

    return tuple(blocking)


def build_coverage_mapping_errors(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path,
) -> tuple[GateError, ...]:
    """Post-run soft policy: new/modified source mapping with coverage fallback."""
    blocking: list[GateError] = []

    has_new_source = bool(_product_paths(changes.new_source, roots))
    if has_new_source:
        effective_map = test_map
        if _product_paths(changes.del_source, roots):
            effective_map = prune_deleted_sources(test_map, changes.del_source)
        blocking.extend(
            gate_new_source(
                repo_root,
                changes,
                effective_map,
                exemptions,
                roots,
                coverage_path=coverage_path,
                check_mapping=True,
            ).errors
        )

    if changes.modified_source:
        blocking.extend(
            gate_modified_source(
                repo_root,
                changes,
                test_map,
                exemptions,
                roots,
                coverage_path=coverage_path,
                check_mapping=True,
            ).errors
        )

    return tuple(blocking)


def build_ci_gate_plan(
    repo_root: Path,
    changes: ChangeSet,
    baseline: Baseline,
    *,
    gate_policy_changed: bool = False,
) -> CiGatePlan:
    """Build pytest schedule without pre-run new/modified source mapping checks."""
    test_map = baseline.test_map
    roots = baseline.roots

    full_suite = bool(changes.config) or gate_policy_changed

    deleted_source_tests: frozenset[str] = frozenset()
    changed_test_nodes: frozenset[str] = frozenset()
    regression_tests: frozenset[str] = frozenset()

    if _product_paths(changes.del_source, roots):
        deleted_source_tests = gate_deleted_source(changes, test_map, roots).tests

    if not full_suite and (changes.new_test or changes.modified_test):
        changed_test_nodes = gate_new_tests(
            changes,
            baseline.test_exemptions,
            marker=_CHANGED_TEST_MARKER,
            full_suite=full_suite,
        ).tests

    if changes.modified_source:
        mod_step = gate_modified_source(
            repo_root,
            changes,
            test_map,
            baseline.exemptions,
            roots,
            check_mapping=False,
        )
        regression_tests = mod_step.tests - deleted_source_tests

    return CiGatePlan(
        blocking_errors=(),
        deleted_source_tests=deleted_source_tests,
        changed_test_nodes=changed_test_nodes,
        regression_tests=regression_tests,
        full_suite=full_suite,
    )


def _needs_union_coverage(changes: ChangeSet, roots: tuple[str, ...]) -> bool:
    has_product = bool(
        _product_paths(changes.new_source, roots)
        or _product_paths(changes.del_source, roots)
        or changes.modified_source
        or changes.renames
    )
    has_test = bool(changes.new_test or changes.modified_test or changes.del_test)
    return has_product or has_test


def _needs_post_run_mapping_check(changes: ChangeSet, roots: tuple[str, ...]) -> bool:
    return bool(_product_paths(changes.new_source, roots) or changes.modified_source)


def compute_execution_plan(plan: CiGatePlan, test_exemptions: tuple[TestExemption, ...]) -> ExecutionPlan:
    """Build a deduplicated pytest schedule from a passing gate plan."""
    if plan.full_suite:
        return ExecutionPlan(
            full_suite=True,
            waves=(TestRunWave(targets=("tests",), marker=_FULL_SUITE_MARKER),),
            reasons={"tests/": _REASON_CONFIG},
        )

    scheduled: dict[str, str] = {}

    for node_id in plan.changed_test_nodes:
        scheduled[node_id] = _REASON_CHANGED_TEST

    for node_id in plan.regression_tests:
        if is_test_exempt(test_exemptions, node_id):
            continue
        scheduled.setdefault(node_id, _REASON_REGRESSION)

    for node_id in plan.deleted_source_tests:
        if is_test_exempt(test_exemptions, node_id):
            continue
        scheduled.setdefault(node_id, _REASON_DELETED_SOURCE)

    changed_nodes = tuple(sorted(node for node in scheduled if node in plan.changed_test_nodes))
    other_nodes = tuple(sorted(node for node in scheduled if node not in plan.changed_test_nodes))

    waves: list[TestRunWave] = []
    if changed_nodes:
        waves.append(TestRunWave(targets=changed_nodes, marker=_CHANGED_TEST_MARKER))
    if other_nodes:
        waves.append(TestRunWave(targets=other_nodes, marker=_REGRESSION_MARKER))

    return ExecutionPlan(full_suite=False, waves=tuple(waves), reasons=scheduled)


def _collected_count_for_targets(targets: list[str], *, marker: str) -> int:
    return count_collected_tests(targets, marker=marker)


def _run_pytest(targets: list[str], *, marker: str, use_cov: bool = False, cov_append: bool = False) -> int:
    if not targets:
        return 0

    logger = logging.getLogger("ci_gate")
    if all("::" in target for target in targets):
        collectable_set = frozenset(filter_collectable_node_ids(targets, marker=marker))
        collectable = [target for target in targets if target in collectable_set]
        skipped = [target for target in targets if target not in collectable_set]
        if skipped:
            logger.info("Skipping non-collectable pytest node(s): %s", ", ".join(skipped))
        if not collectable:
            return 0
        run_targets = collectable
        collected = len(collectable)
    else:
        run_targets = targets
        collected = _collected_count_for_targets(targets, marker=marker)

    extra_args = cov_pytest_args(cov_context=True, append=cov_append) if use_cov else ()
    cmd = build_pytest_cmd(
        sys.executable,
        run_targets,
        marker=marker,
        collected_count=collected,
        extra_args=extra_args,
    )
    logger.info("Running pytest: %s", shlex.join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def _sample_nodes(nodes: tuple[str, ...], limit: int = _SAMPLE_NODE_LIMIT) -> str:
    if not nodes:
        return ""
    sample = ", ".join(nodes[:limit])
    if len(nodes) > limit:
        sample = f"{sample}, ... (+{len(nodes) - limit} more)"
    return sample


def _log_execution_plan(logger: logging.Logger, execution: ExecutionPlan) -> None:
    if execution.full_suite:
        logger.info("Selected full test suite: %s", _REASON_CONFIG)
        return
    if not execution.has_work:
        logger.info("No pytest targets after policy checks; skipping test run")
        return

    counts = Counter(execution.reasons.values())
    for reason, count in sorted(counts.items()):
        logger.info("Scheduling %d test node(s): %s", count, reason)
    all_nodes = tuple(node for wave in execution.waves for node in wave.targets)
    logger.info("Sample node(s): %s", _sample_nodes(all_nodes))
    logger.info("Execution uses %d pytest wave(s) after deduplication", len(execution.waves))


def _log_blocking_errors(logger: logging.Logger, errors: tuple[GateError, ...]) -> None:
    counts: dict[str, int] = {}
    for err in errors:
        counts[err.category] = counts.get(err.category, 0) + 1
    summary = ", ".join(f"{category}={count}" for category, count in sorted(counts.items()))
    logger.error(
        "Policy validation failed; pytest skipped (%d issue(s): %s)",
        len(errors),
        summary or "unknown",
    )


def _log_pytest_failure(
    logger: logging.Logger,
    *,
    full_suite: bool,
    failed_nodes: tuple[str, ...],
) -> None:
    if full_suite:
        logger.error("Full test suite failed; see pytest output above")
        print("CI gate failed: full test suite did not pass. See pytest output above.")
        return
    if failed_nodes:
        logger.error("Selected tests failed; see pytest output above")
        print(format_pytest_failure_hint(failed_nodes))
        return
    logger.error("Selected tests failed; see pytest output above")
    print("CI gate failed: selected tests did not pass. See pytest output above.")


def _print_success_summary(execution: ExecutionPlan, changes: ChangeSet) -> None:
    if execution.full_suite:
        config_paths = ", ".join(changes.config) if changes.config else "tests/"
        print(f"CI gate passed: full test suite ({config_paths})")
        return

    node_count = sum(len(wave.targets) for wave in execution.waves)
    counts = Counter(execution.reasons.values())
    reason_parts = ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
    if reason_parts:
        print(f"CI gate passed: {node_count} test node(s) ({reason_parts})")
    else:
        print(f"CI gate passed: {node_count} test node(s)")


def _log_change_summary(logger: logging.Logger, changes: ChangeSet) -> None:
    if changes.config:
        logger.info("Config path(s): %s", ", ".join(changes.config))
    logger.info(
        "Changes: config=%d new_test=%d mod_test=%d del_test=%d new_source=%d del_source=%d modified=%d renames=%d",
        len(changes.config),
        len(changes.new_test),
        len(changes.modified_test),
        len(changes.del_test),
        len(changes.new_source),
        len(changes.del_source),
        len(changes.modified_source),
        len(changes.renames),
    )


def _baseline_with_renames(baseline: Baseline, renames: tuple[tuple[str, str, int], ...]) -> Baseline:
    return baseline.__class__(
        test_map=_remap_renamed_sources(baseline.test_map, renames),
        exemptions=baseline.exemptions,
        test_exemptions=baseline.test_exemptions,
        discovery=baseline.discovery,
        roots=baseline.roots,
    )


def _run_execution_waves(
    logger: logging.Logger,
    execution: ExecutionPlan,
    *,
    use_cov: bool,
) -> int:
    for wave_index, wave in enumerate(execution.waves):
        pytest_code = _run_pytest(
            list(wave.targets),
            marker=wave.marker,
            use_cov=use_cov,
            cov_append=use_cov and wave_index > 0,
        )
        if pytest_code != 0:
            failed_nodes = wave.targets if not execution.full_suite else ()
            _log_pytest_failure(logger, full_suite=execution.full_suite, failed_nodes=failed_nodes)
            return pytest_code
    return 0


def _soft_mapping_exit_code(
    changes: ChangeSet,
    baseline: Baseline,
    *,
    logger: logging.Logger,
) -> int:
    if not _needs_post_run_mapping_check(changes, baseline.roots):
        return 0
    logger.info("Checking new/modified source coverage mapping against collected data ...")
    soft_errors = build_coverage_mapping_errors(
        REPO_ROOT,
        changes,
        baseline.test_map,
        baseline.exemptions,
        baseline.roots,
        coverage_path=_COVERAGE_DATA_PATH,
    )
    if not soft_errors:
        return 0
    _log_blocking_errors(logger, soft_errors)
    print(format_blocking_errors(soft_errors, pytest_ran=True))
    return 1


def _prepare_gate_inputs(
    cfg: Config,
    logger: logging.Logger,
) -> tuple[Baseline, ChangeSet, bool] | int:
    logger.info("Resolving merge-base against %s ...", cfg.base_branch)
    try:
        merge_base = resolve_base_ref(REPO_ROOT, cfg.base_branch)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Merge-base: %s", merge_base[:12])

    try:
        validate_gate_policy_if_changed(REPO_ROOT, merge_base)
        baseline, test_map_commit = load_baseline(REPO_ROOT, cfg)
        validate_test_map_freshness(REPO_ROOT, test_map_commit, merge_base)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Fetching diff ...")
    diff_result = fetch_diff(REPO_ROOT, merge_base)
    logger.info("Diff: %d files changed", len(diff_result.line_map))

    logger.info("Classifying changes ...")
    changes = classify_changes(REPO_ROOT, merge_base, diff_result, baseline.discovery, baseline.roots)
    _log_change_summary(logger, changes)

    if changes.renames:
        logger.info("Remapping coverage mapping for %d renamed source(s) ...", len(changes.renames))
        baseline = _baseline_with_renames(baseline, changes.renames)

    logger.info("Validating hard-blocking policy ...")
    hard_errors = build_hard_blocking_plan(changes, baseline.test_map, baseline.roots)
    if hard_errors:
        _log_blocking_errors(logger, hard_errors)
        print(format_blocking_errors(hard_errors))
        return 1

    policy_changed = gate_policy_changed_in_diff(REPO_ROOT, merge_base)
    if policy_changed:
        logger.info("gate_policy.yaml changed; scheduling full test suite")
    return baseline, changes, policy_changed


def main() -> int:
    logger = setup_logger()
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    logger.info("CI gate: classify diff, validate policy, plan tests, run deduplicated selection")

    prepared = _prepare_gate_inputs(cfg, logger)
    if isinstance(prepared, int):
        return prepared
    baseline, changes, policy_changed = prepared

    logger.info("Building gate plan ...")
    plan = build_ci_gate_plan(REPO_ROOT, changes, baseline, gate_policy_changed=policy_changed)

    execution = compute_execution_plan(plan, baseline.test_exemptions)
    _log_execution_plan(logger, execution)

    use_cov = _needs_union_coverage(changes, baseline.roots)
    if use_cov and execution.has_work:
        logger.info("Union pytest run will collect branch coverage with per-test context")

    if not execution.has_work and not _needs_post_run_mapping_check(changes, baseline.roots):
        print("CI gate passed")
        return 0

    pytest_code = _run_execution_waves(logger, execution, use_cov=use_cov) if execution.has_work else 0
    if pytest_code != 0:
        return pytest_code

    mapping_code = _soft_mapping_exit_code(changes, baseline, logger=logger)
    if mapping_code != 0:
        return mapping_code

    if execution.has_work:
        _print_success_summary(execution, changes)
    else:
        print("CI gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
