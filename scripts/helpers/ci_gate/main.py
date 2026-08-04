#!/usr/bin/env python3
"""CI incremental gate: diff analysis, test selection, pytest."""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.helpers._config import Config, ConfigError
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.classifier import classify_changes
from scripts.helpers.ci_gate.comments import (
    maybe_post_all_exempt_tests_comment,
    maybe_post_exemption_drift_comment,
    maybe_post_shadowed_defs_comment,
    maybe_post_unscoped_python_comment,
)
from scripts.helpers.ci_gate.diff import fetch_diff, resolve_base_ref
from scripts.helpers.ci_gate.errors import (
    format_blocking_errors,
    format_pytest_failure_hint,
)
from scripts.helpers.ci_gate.models import (
    Baseline,
    ChangeSet,
    CiGatePlan,
    CiGatePolicy,
    ExecutionPlan,
    GateError,
    GateStepResult,
    SourceExemption,
    TestExemption,
    TestRunWave,
)
from scripts.helpers.ci_gate.policy import (
    is_test_exempt,
    validate_gate_policy_if_changed,
)
from scripts.helpers.ci_gate.policy_drift import gate_exemption_drift, iter_rename_pairs
from scripts.helpers.ci_gate.rules import (
    _product_paths,
    collect_modified_source_mapping_errors,
    gate_deleted_source,
    gate_deleted_tests,
    gate_modified_source,
    gate_new_source,
    gate_new_tests,
)
from scripts.helpers.ci_gate.test_map_query import prune_deleted_sources
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.ast_utils import ShadowWarning, collect_shadow_warnings
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.coverage_symbol_check import load_coverage_data
from scripts.helpers.common.pytest_runner import (
    build_pytest_cmd,
    count_collected_tests,
    filter_collectable_node_ids,
)
from scripts.helpers.common.test_map_loader import (
    assess_test_map_freshness,
    load_baseline,
)

if TYPE_CHECKING:
    from pathlib import Path

_CHANGED_TEST_MARKER: str | None = None
# Mapped/guard regression wave and config-triggered full suite share this marker.
_REGRESSION_MARKER = "not npu and not nightly and not network"
_FULL_SUITE_MARKER = _REGRESSION_MARKER
_COVERAGE_DATA_PATH = REPO_ROOT / ".coverage"
_SAMPLE_NODE_LIMIT = 3

_REASON_CONFIG = "dependency or test configuration changed"
_REASON_CHANGED_TEST = "new or changed test file"
_REASON_REGRESSION = "changed product file mapped regression"
_REASON_DELETED_SOURCE = "deleted product file guard test"


@dataclass(frozen=True, slots=True)
class _PrepareFailure:
    code: int
    message: str


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    baseline: Baseline
    changes: ChangeSet
    deleted_source_step: GateStepResult
    force_full_suite: bool = False


def build_hard_blocking_plan(
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    policy: CiGatePolicy,
    rename_pairs: tuple[tuple[str, str], ...] = (),
    *,
    deleted_source_step: GateStepResult | None = None,
) -> tuple[GateError, ...]:
    roots = policy.roots
    blocking: list[GateError] = []

    blocking.extend(gate_exemption_drift(policy, changes, rename_pairs))

    if changes.del_test:
        blocking.extend(gate_deleted_tests(changes, test_map).errors)

    if _product_paths(changes.del_source, roots):
        step = deleted_source_step if deleted_source_step is not None else gate_deleted_source(changes, test_map, roots)
        blocking.extend(step.errors)

    return tuple(blocking)


def collect_product_shadow_warnings(
    repo_root: Path,
    changes: ChangeSet,
    roots: tuple[str, ...],
) -> tuple[ShadowWarning, ...]:
    """Collect shadow warnings for new or modified product files in *changes*."""
    product_files: set[str] = set(_product_paths(changes.new_source, roots))
    product_files.update(_product_paths(tuple(path for path, _lines in changes.modified_source), roots))

    warnings: list[ShadowWarning] = []
    for rel_path in sorted(product_files):
        abs_path = repo_root / rel_path
        for warning in collect_shadow_warnings(abs_path):
            file_rel = rel_path if warning.file == str(abs_path) else warning.file
            warnings.append(
                ShadowWarning(
                    file=file_rel,
                    line=warning.line,
                    name=warning.name,
                    shadowed_by_line=warning.shadowed_by_line,
                )
            )
    return tuple(warnings)


def build_coverage_mapping_errors(
    repo_root: Path,
    changes: ChangeSet,
    test_map: dict[str, dict[str, list[str]]],
    exemptions: tuple[SourceExemption, ...],
    roots: tuple[str, ...],
    *,
    coverage_path: Path,
    modified_source_step: GateStepResult | None = None,
) -> tuple[GateError, ...]:
    blocking: list[GateError] = []
    coverage_data = load_coverage_data(coverage_path)

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
                coverage_data=coverage_data,
                check_mapping=True,
            ).errors
        )

    if changes.modified_source:
        if modified_source_step is not None:
            blocking.extend(
                collect_modified_source_mapping_errors(
                    repo_root,
                    changes,
                    test_map,
                    exemptions,
                    roots,
                    coverage_path=coverage_path,
                    coverage_data=coverage_data,
                )
            )
        else:
            blocking.extend(
                gate_modified_source(
                    repo_root,
                    changes,
                    test_map,
                    exemptions,
                    roots,
                    coverage_path=coverage_path,
                    coverage_data=coverage_data,
                    check_mapping=True,
                ).errors
            )

    return tuple(blocking)


def build_ci_gate_plan(
    repo_root: Path,
    changes: ChangeSet,
    baseline: Baseline,
    *,
    deleted_source_step: GateStepResult | None = None,
    modified_source_step: GateStepResult | None = None,
    force_full_suite: bool = False,
) -> CiGatePlan:
    test_map = baseline.test_map
    roots = baseline.roots
    full_suite = force_full_suite or bool(changes.config)

    deleted_source_tests: frozenset[str] = frozenset()
    changed_test_nodes: frozenset[str] = frozenset()
    regression_tests: frozenset[str] = frozenset()

    if _product_paths(changes.del_source, roots):
        step = deleted_source_step if deleted_source_step is not None else gate_deleted_source(changes, test_map, roots)
        deleted_source_tests = step.tests

    all_exempt_test_files: frozenset[str] = frozenset()
    if not full_suite and (changes.new_test or changes.modified_test):
        new_tests_step = gate_new_tests(
            changes,
            baseline.test_exemptions,
            full_suite=full_suite,
        )
        changed_test_nodes = new_tests_step.tests
        all_exempt_test_files = frozenset(new_tests_step.all_exempt_test_files)

    if not full_suite and changes.modified_source:
        mod_step = (
            modified_source_step
            if modified_source_step is not None
            else gate_modified_source(
                repo_root,
                changes,
                test_map,
                baseline.exemptions,
                roots,
                check_mapping=False,
            )
        )
        regression_tests = mod_step.tests - deleted_source_tests

    return CiGatePlan(
        deleted_source_tests=deleted_source_tests,
        changed_test_nodes=changed_test_nodes,
        regression_tests=regression_tests,
        full_suite=full_suite,
        all_exempt_test_files=all_exempt_test_files,
    )


def _needs_union_coverage(changes: ChangeSet, roots: tuple[str, ...]) -> bool:
    has_product = bool(
        _product_paths(changes.new_source, roots)
        or _product_paths(changes.del_source, roots)
        or changes.modified_source
    )
    has_test = bool(changes.new_test or changes.modified_test or changes.del_test)
    return has_product or has_test


def _needs_post_run_mapping_check(changes: ChangeSet, roots: tuple[str, ...]) -> bool:
    return bool(_product_paths(changes.new_source, roots) or changes.modified_source)


def compute_execution_plan(plan: CiGatePlan, test_exemptions: tuple[TestExemption, ...]) -> ExecutionPlan:
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


def _collected_count_for_targets(targets: list[str], *, marker: str | None) -> int:
    if marker is None:
        from scripts.helpers.common.pytest_runner import collect_all_test_node_ids

        return len(collect_all_test_node_ids(targets))
    return count_collected_tests(targets, marker=marker)


def _run_pytest(
    targets: list[str],
    *,
    marker: str | None,
    use_cov: bool = False,
    cov_append: bool = False,
) -> int:
    if not targets:
        return 0

    logger = logging.getLogger("ci_gate")
    if all("::" in target for target in targets):
        logger.info(
            "Filtering %d pytest node id(s) for collectability (marker=%r)",
            len(targets),
            marker,
        )
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


def _pytest_failure_user_message(
    *,
    full_suite: bool,
    failed_nodes: tuple[str, ...],
) -> str:
    if full_suite:
        return "CI gate failed: full test suite did not pass. See pytest output above."
    if failed_nodes:
        return format_pytest_failure_hint(failed_nodes)
    return "CI gate failed: selected tests did not pass. See pytest output above."


def _log_pytest_failure(
    logger: logging.Logger,
    *,
    full_suite: bool,
    failed_nodes: tuple[str, ...],
) -> None:
    if full_suite:
        logger.error("Full test suite failed; see pytest output above")
        return
    logger.error("Selected tests failed; see pytest output above")


def _success_user_message(execution: ExecutionPlan, changes: ChangeSet) -> str:
    if execution.full_suite:
        config_paths = ", ".join(changes.config) if changes.config else "tests/"
        return f"CI gate passed: full test suite ({config_paths})"

    node_count = sum(len(wave.targets) for wave in execution.waves)
    counts = Counter(execution.reasons.values())
    reason_parts = ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
    if reason_parts:
        return f"CI gate passed: {node_count} test node(s) ({reason_parts})"
    return f"CI gate passed: {node_count} test node(s)"


def _log_change_summary(logger: logging.Logger, changes: ChangeSet, cfg: Config) -> None:
    if changes.config:
        logger.info("Config path(s): %s", ", ".join(changes.config))
    if changes.unscoped_python:
        logger.warning(
            "Unscoped Python change(s) outside gate_policy.yaml roots/tests/configs: %s",
            ", ".join(changes.unscoped_python),
        )
        maybe_post_unscoped_python_comment(changes.unscoped_python, cfg=cfg)
    logger.info(
        "Changes: config=%d new_test=%d mod_test=%d del_test=%d new_source=%d del_source=%d modified=%d unscoped_py=%d",
        len(changes.config),
        len(changes.new_test),
        len(changes.modified_test),
        len(changes.del_test),
        len(changes.new_source),
        len(changes.del_source),
        len(changes.modified_source),
        len(changes.unscoped_python),
    )


def _baseline_without_test_map(baseline: Baseline) -> Baseline:
    return baseline.__class__(test_map={}, policy=baseline.policy)


def _run_execution_waves(
    logger: logging.Logger,
    execution: ExecutionPlan,
    *,
    use_cov: bool,
) -> tuple[int, str | None]:
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
            return pytest_code, _pytest_failure_user_message(
                full_suite=execution.full_suite,
                failed_nodes=failed_nodes,
            )
    return 0, None


def _soft_mapping_exit_code(
    changes: ChangeSet,
    baseline: Baseline,
    *,
    modified_source_step: GateStepResult | None = None,
    logger: logging.Logger,
) -> tuple[int, str | None]:
    if not _needs_post_run_mapping_check(changes, baseline.roots):
        return 0, None
    logger.info("Checking new/modified source coverage mapping against collected data ...")
    soft_errors = build_coverage_mapping_errors(
        REPO_ROOT,
        changes,
        baseline.test_map,
        baseline.exemptions,
        baseline.roots,
        coverage_path=_COVERAGE_DATA_PATH,
        modified_source_step=modified_source_step,
    )
    if not soft_errors:
        return 0, None
    _log_blocking_errors(logger, soft_errors)
    return 1, format_blocking_errors(soft_errors, pytest_ran=True)


def _prepare_gate_inputs(
    cfg: Config,
    logger: logging.Logger,
) -> _PreparedInputs | _PrepareFailure:
    logger.info("Resolving merge-base against %s ...", cfg.base_branch)
    try:
        merge_base = resolve_base_ref(REPO_ROOT, cfg.base_branch)
    except ConfigError as exc:
        logger.error("%s", exc)
        return _PrepareFailure(1, str(exc))
    logger.info("Merge-base: %s", merge_base[:12])

    force_full_suite = False
    try:
        validate_gate_policy_if_changed(REPO_ROOT, merge_base)
        baseline, test_map_commit = load_baseline(REPO_ROOT, cfg)
        freshness = assess_test_map_freshness(REPO_ROOT, test_map_commit, merge_base)
        if freshness.block_message:
            raise ConfigError(freshness.block_message)
        if freshness.warn_message:
            logger.warning(
                "%s; falling back to the full test suite without stale coverage mapping", freshness.warn_message
            )
            baseline = _baseline_without_test_map(baseline)
            force_full_suite = True
    except ConfigError as exc:
        logger.error("%s", exc)
        return _PrepareFailure(1, str(exc))

    logger.info("Fetching diff ...")
    diff_result = fetch_diff(REPO_ROOT, merge_base)
    logger.info("Diff: %d files changed", len(diff_result.line_map))

    logger.info("Classifying changes ...")
    try:
        changes = classify_changes(diff_result, baseline.policy)
    except ConfigError as exc:
        logger.error("%s", exc)
        return _PrepareFailure(1, str(exc))
    _log_change_summary(logger, changes, cfg)

    shadow_warnings = collect_product_shadow_warnings(REPO_ROOT, changes, baseline.roots)
    for warning in shadow_warnings:
        logger.warning(
            "Shadowed duplicate definition: %s:%d `%s` shadowed by line %d",
            warning.file,
            warning.line,
            warning.name,
            warning.shadowed_by_line,
        )
    maybe_post_shadowed_defs_comment(shadow_warnings, cfg=cfg)

    logger.info("Validating hard-blocking policy ...")
    rename_pairs = iter_rename_pairs(diff_result.entries)
    deleted_source_step = GateStepResult()
    if _product_paths(changes.del_source, baseline.roots):
        deleted_source_step = gate_deleted_source(changes, baseline.test_map, baseline.roots)
    hard_errors = build_hard_blocking_plan(
        changes,
        baseline.test_map,
        baseline.policy,
        rename_pairs,
        deleted_source_step=deleted_source_step,
    )
    if hard_errors:
        _log_blocking_errors(logger, hard_errors)
        drift_errors = tuple(err for err in hard_errors if err.category == "exemption_drift")
        if drift_errors:
            maybe_post_exemption_drift_comment(drift_errors, cfg=cfg)
        return _PrepareFailure(1, format_blocking_errors(hard_errors))

    if force_full_suite:
        logger.info("test_map is stale; scheduling full test suite")
    return _PreparedInputs(
        baseline=baseline,
        changes=changes,
        deleted_source_step=deleted_source_step,
        force_full_suite=force_full_suite,
    )


def _log_all_exempt_test_files(
    plan: CiGatePlan,
    cfg: Config,
    logger: logging.Logger,
) -> None:
    if not plan.all_exempt_test_files:
        return
    for path in sorted(plan.all_exempt_test_files):
        logger.warning("Changed test file has only exempt tests; no pytest scheduled: %s", path)
    maybe_post_all_exempt_tests_comment(tuple(sorted(plan.all_exempt_test_files)), cfg=cfg)


def _run_gate_finalize(
    changes: ChangeSet,
    baseline: Baseline,
    execution: ExecutionPlan,
    *,
    modified_source_step: GateStepResult | None = None,
    use_cov: bool,
    logger: logging.Logger,
) -> int:
    if not execution.has_work and not _needs_post_run_mapping_check(changes, baseline.roots):
        print("CI gate passed")
        return 0

    pytest_code = 0
    user_message: str | None = None
    if execution.has_work:
        pytest_code, user_message = _run_execution_waves(logger, execution, use_cov=use_cov)
    if pytest_code != 0:
        if user_message is not None:
            print(user_message)
        return pytest_code

    mapping_code, mapping_message = _soft_mapping_exit_code(
        changes,
        baseline,
        modified_source_step=modified_source_step,
        logger=logger,
    )
    if mapping_code != 0:
        if mapping_message is not None:
            print(mapping_message)
        return mapping_code

    if execution.has_work:
        print(_success_user_message(execution, changes))
    else:
        print("CI gate passed")
    return 0


def main() -> int:
    logger = setup_logger()
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    logger.info("CI gate: classify diff, validate policy, plan tests, run deduplicated selection")

    prepared = _prepare_gate_inputs(cfg, logger)
    if isinstance(prepared, _PrepareFailure):
        print(prepared.message)
        return prepared.code

    modified_source_step: GateStepResult | None = None
    if not prepared.changes.config and prepared.changes.modified_source:
        modified_source_step = gate_modified_source(
            REPO_ROOT,
            prepared.changes,
            prepared.baseline.test_map,
            prepared.baseline.exemptions,
            prepared.baseline.roots,
            check_mapping=False,
        )

    logger.info("Building gate plan ...")
    plan = build_ci_gate_plan(
        REPO_ROOT,
        prepared.changes,
        prepared.baseline,
        deleted_source_step=prepared.deleted_source_step,
        modified_source_step=modified_source_step,
        force_full_suite=prepared.force_full_suite,
    )
    _log_all_exempt_test_files(plan, cfg, logger)

    execution = compute_execution_plan(plan, prepared.baseline.test_exemptions)
    _log_execution_plan(logger, execution)

    use_cov = _needs_union_coverage(prepared.changes, prepared.baseline.roots)
    if use_cov and execution.has_work:
        logger.info("Union pytest run will collect branch coverage with per-test context")

    return _run_gate_finalize(
        prepared.changes,
        prepared.baseline,
        execution,
        modified_source_step=modified_source_step,
        use_cov=use_cov,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
