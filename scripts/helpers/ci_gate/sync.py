#!/usr/bin/env python3

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Maintain authoritative test_map against a target branch HEAD.

CLI entry for ``run_test_map_sync.sh``. Reads and writes only
``MSMODELING_TEST_MAP_PATH``; OBS upload/download stays outside the repo.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers._config import Config, ConfigError, format_expected_got
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.diff import (
    ephemeral_target_checkout,
    fetch_changed_paths,
    is_git_ancestor,
    resolve_target_head,
)
from scripts.helpers.ci_gate.gate_policy import load_gate_policy
from scripts.helpers.ci_gate.test_map_query import (
    classify_touched_paths,
    has_unmapped_existing_source,
    select_resync_targets,
)
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.build_test_map import (
    TestMap,
    collect_test_map,
    normalize_test_node_id,
    prune_missing_source_keys,
    write_test_map,
)
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.pytest_runner import build_pytest_cmd, collect_test_node_ids
from scripts.helpers.common.test_map_config import (
    TEST_MAP_EXECUTION_MARKER,
    resolve_test_map_path,
)
from scripts.helpers.common.test_map_loader import load_test_map_with_commit

_DEFAULT_SYNC_INTERVAL_SECONDS: Final = 60.0

_shutdown_flag: list[bool] = [False]


def _parse_sync_interval() -> float:
    raw = os.environ.get("MSMODELING_TEST_MAP_SYNC_INTERVAL", "").strip()
    if not raw:
        return _DEFAULT_SYNC_INTERVAL_SECONDS
    try:
        interval = float(raw)
    except ValueError as exc:
        raise ConfigError(format_expected_got("MSMODELING_TEST_MAP_SYNC_INTERVAL", "a number", raw)) from exc
    if interval <= 0:
        raise ConfigError(format_expected_got("MSMODELING_TEST_MAP_SYNC_INTERVAL", "positive", interval))
    return interval


def resolve_target_branch(*, cli_target: str | None, cfg: Config) -> str:
    if cli_target:
        return cli_target
    env_target = os.environ.get("MSMODELING_TEST_MAP_TARGET_BRANCH", "").strip()
    if env_target:
        return env_target
    return cfg.base_branch


def _test_file_for_node(test_node: str) -> str:
    return test_node.split("::", 1)[0]


def apply_incremental_test_map_update(
    existing_map: TestMap,
    fresh_map: TestMap,
    touched_paths: frozenset[str],
) -> TestMap:
    """Merge *fresh_map* into *existing_map* for git-touched product or test paths."""
    all_test_nodes = set(existing_map) | set(fresh_map)
    merged: TestMap = {}
    for test_node in all_test_nodes:
        test_file = _test_file_for_node(test_node)
        if test_file in touched_paths:
            fresh_sources = fresh_map.get(test_node)
            if fresh_sources:
                merged[test_node] = {src: list(syms) for src, syms in fresh_sources.items()}
            continue

        sources: dict[str, list[str]] = {}
        for src_path, symbols in existing_map.get(test_node, {}).items():
            if src_path not in touched_paths:
                sources[src_path] = list(symbols)
        for src_path, symbols in fresh_map.get(test_node, {}).items():
            if src_path in touched_paths:
                sources[src_path] = list(symbols)
        if sources:
            merged[test_node] = sources
    return prune_missing_source_keys(merged)


_FULL_TEST_TARGETS: Final = ("tests/smoke/", "tests/regression/")


def build_test_map_pytest_cmd(
    python_exe: str,
    targets: tuple[str, ...] = _FULL_TEST_TARGETS,
    *,
    collected_count: int | None = None,
) -> list[str]:
    if collected_count is None:
        collected_count = len(targets)
    return build_pytest_cmd(
        python_exe,
        targets,
        marker=TEST_MAP_EXECUTION_MARKER,
        collected_count=collected_count,
        extra_args=(*cov_pytest_args(cov_context=True), "-q", "--no-header"),
    )


def run_test_map_pytest(
    repo_root: Path,
    python_exe: str,
    targets: tuple[str, ...] = _FULL_TEST_TARGETS,
    *,
    collected_count: int | None = None,
) -> int:
    cmd = build_test_map_pytest_cmd(python_exe, targets, collected_count=collected_count)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    logging.getLogger(__name__).info("Running pytest: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return proc.returncode


def can_incremental_sync(repo_root: Path, built_from_commit: str, target_head: str) -> bool:
    """Return True when an incremental merge from *built_from_commit* to *target_head* is safe."""
    if built_from_commit == target_head:
        return True
    return is_git_ancestor(repo_root, built_from_commit, target_head)


def _try_load_existing_map(cfg: Config) -> tuple[TestMap | None, str | None]:
    map_path = resolve_test_map_path(cfg, must_exist=False)
    if not map_path.is_file():
        return None, None
    try:
        return load_test_map_with_commit(cfg)
    except ConfigError:
        return None, None


def _clear_coverage_artifacts(repo_root: Path) -> None:
    for path in (repo_root / ".coverage", *sorted(repo_root.glob(".coverage.*"))):
        if path.is_file():
            path.unlink()


def _normalized_node_ids(node_ids: tuple[str, ...]) -> frozenset[str]:
    return frozenset(filter(None, (normalize_test_node_id(node_id) for node_id in node_ids)))


def _collect_execution_node_ids(targets: tuple[str, ...]) -> tuple[str, ...]:
    """Collect execution ids, expanding normalized map keys to parametrized ids."""
    file_targets = tuple(target for target in targets if "::" not in target)
    node_targets = tuple(target for target in targets if "::" in target)
    wanted = {normalize_test_node_id(target) for target in node_targets}
    parent_files = tuple(
        dict.fromkeys(
            target.split("::", 1)[0] for target in node_targets if (REPO_ROOT / target.split("::", 1)[0]).is_file()
        )
    )
    collected: list[str] = []
    if file_targets:
        collected.extend(collect_test_node_ids(file_targets, marker=TEST_MAP_EXECUTION_MARKER))
    if parent_files:
        collected.extend(
            node_id
            for node_id in collect_test_node_ids(parent_files, marker=TEST_MAP_EXECUTION_MARKER)
            if normalize_test_node_id(node_id) in wanted
        )
    return tuple(dict.fromkeys(collected))


def _collect_fresh_map(allowed_node_ids: frozenset[str]) -> TestMap:
    gate_policy = load_gate_policy(REPO_ROOT)
    return collect_test_map(
        marker_expr=TEST_MAP_EXECUTION_MARKER,
        roots=gate_policy.roots,
        allowed_node_ids=allowed_node_ids,
    )


def _run_pytest_and_collect_fresh_map(
    repo_root: Path,
    python_exe: str,
    targets: tuple[str, ...] = _FULL_TEST_TARGETS,
) -> TestMap | None:
    collected_node_ids = _collect_execution_node_ids(targets)
    if not collected_node_ids:
        if targets == _FULL_TEST_TARGETS:
            logging.getLogger(__name__).error("test_map collection produced no node ids for the full suite")
            return None
        return {}
    _clear_coverage_artifacts(repo_root)
    run_targets = targets if targets == _FULL_TEST_TARGETS else collected_node_ids
    pytest_exit = run_test_map_pytest(
        repo_root,
        python_exe,
        run_targets,
        collected_count=len(collected_node_ids),
    )
    if pytest_exit != 0:
        return None
    return _collect_fresh_map(_normalized_node_ids(collected_node_ids))


def _full_rebuild_test_map(
    cfg: Config,
    *,
    target_branch: str,
    target_head: str,
    logger: logging.Logger,
    reason: str,
) -> int:
    map_path = resolve_test_map_path(cfg, must_exist=False)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    logger.warning("test_map full rebuild: %s", reason)
    with ephemeral_target_checkout(REPO_ROOT, target_branch):
        fresh_map = _run_pytest_and_collect_fresh_map(REPO_ROOT, sys.executable)
    if fresh_map is None:
        logger.error("test_map full rebuild aborted: pytest failed")
        return 1
    write_test_map(map_path, fresh_map, built_from_commit=target_head)
    logger.info(
        "test_map full rebuild wrote %s at built_from_commit=%s",
        map_path,
        target_head[:12],
    )
    return 0


def sync_test_map_once(
    cfg: Config,
    *,
    target_branch: str,
    logger: logging.Logger,
) -> int:
    map_path = resolve_test_map_path(cfg, must_exist=False)
    existing_map, built_from_commit = _try_load_existing_map(cfg)
    target_head = resolve_target_head(REPO_ROOT, target_branch)
    logger.info(
        "test_map sync: built_from_commit=%s target=%s (%s)",
        (built_from_commit or "(none)")[:12],
        target_head[:12],
        target_branch,
    )

    if existing_map is None or not built_from_commit:
        return _full_rebuild_test_map(
            cfg,
            target_branch=target_branch,
            target_head=target_head,
            logger=logger,
            reason="test_map file missing, unreadable, or built_from_commit absent",
        )

    if built_from_commit == target_head:
        logger.info("test_map is up to date with target HEAD")
        return 0

    if not can_incremental_sync(REPO_ROOT, built_from_commit, target_head):
        return _full_rebuild_test_map(
            cfg,
            target_branch=target_branch,
            target_head=target_head,
            logger=logger,
            reason=(f"built_from_commit {built_from_commit[:12]} is not an ancestor of target HEAD {target_head[:12]}"),
        )

    touched_paths = fetch_changed_paths(REPO_ROOT, built_from_commit, target_head)
    logger.info(
        "Incremental update range %s..%s (%d path(s) touched)",
        built_from_commit[:12],
        target_head[:12],
        len(touched_paths),
    )

    fallback_reason: str | None = None
    fresh_map: TestMap | None = None
    with ephemeral_target_checkout(REPO_ROOT, target_branch):
        gate_policy = load_gate_policy(REPO_ROOT)
        touched = classify_touched_paths(touched_paths, gate_policy)
        if touched.configs:
            fallback_reason = f"test collection/configuration changed: {', '.join(sorted(touched.configs)[:3])}"
        elif has_unmapped_existing_source(existing_map, touched, repo_root=REPO_ROOT):
            fallback_reason = "a changed source has no watcher in the existing test_map"
        elif not touched.tests and not touched.sources:
            write_test_map(map_path, existing_map, built_from_commit=target_head)
            logger.info("No map-relevant paths changed; advanced built_from_commit without pytest")
            return 0
        else:
            targets = select_resync_targets(existing_map, touched, repo_root=REPO_ROOT)
            fresh_map = _run_pytest_and_collect_fresh_map(REPO_ROOT, sys.executable, targets)
    if fallback_reason is not None:
        return _full_rebuild_test_map(
            cfg,
            target_branch=target_branch,
            target_head=target_head,
            logger=logger,
            reason=fallback_reason,
        )
    if fresh_map is None:
        logger.error("test_map sync aborted: pytest failed")
        return 1

    updated_map = apply_incremental_test_map_update(existing_map, fresh_map, touched_paths)
    write_test_map(map_path, updated_map, built_from_commit=target_head)
    logger.info("test_map sync wrote %s at built_from_commit=%s", map_path, target_head[:12])
    return 0


def _register_shutdown_handlers() -> None:
    def _handle_signal(_signum: int, _frame: object) -> None:
        _shutdown_flag[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def sync_test_map_watch(
    cfg: Config,
    *,
    target_branch: str,
    interval_seconds: float,
    logger: logging.Logger,
) -> int:
    _shutdown_flag[0] = False
    _register_shutdown_handlers()
    logger.info(
        "test_map sync watch: target=%s interval=%.0fs",
        target_branch,
        interval_seconds,
    )
    while not _shutdown_flag[0]:
        try:
            exit_code = sync_test_map_once(cfg, target_branch=target_branch, logger=logger)
        except ConfigError as exc:
            logger.error("%s", exc)
            return 1
        if exit_code != 0:
            logger.error(
                "test_map sync cycle failed (exit %d); retrying after interval",
                exit_code,
            )
        deadline = time.monotonic() + interval_seconds
        while not _shutdown_flag[0] and time.monotonic() < deadline:
            time.sleep(0.1)
    logger.info("test_map sync watch stopped")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync MSMODELING_TEST_MAP_PATH with a target branch HEAD.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one sync cycle and exit.")
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Poll target branch HEAD until interrupted.",
    )
    parser.add_argument(
        "--target-branch",
        default=None,
        help=(
            "Target branch or remote/branch "
            "(default: MSMODELING_TEST_MAP_TARGET_BRANCH or MSMODELING_TEST_BASE_BRANCH)."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Poll interval in seconds for --watch (default: MSMODELING_TEST_MAP_SYNC_INTERVAL or 60).",
    )
    return parser


def _log_sync_env(logger: logging.Logger, target_branch: str, interval_seconds: float | None) -> None:
    logger.info("  MSMODELING_TEST_MAP_TARGET_BRANCH = %s", target_branch)
    if interval_seconds is not None:
        logger.info("  MSMODELING_TEST_MAP_SYNC_INTERVAL = %s", interval_seconds)


def main(argv: list[str] | None = None) -> int:
    logger = setup_logger("test_map_sync")
    args = build_arg_parser().parse_args(argv)
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    if not cfg.test_map_path:
        logger.error("MSMODELING_TEST_MAP_PATH is required for test_map sync")
        return 1

    try:
        target_branch = resolve_target_branch(cli_target=args.target_branch, cfg=cfg)
        interval = args.interval if args.interval is not None else _parse_sync_interval()
        _log_sync_env(logger, target_branch, interval if args.watch else None)
        if args.once:
            return sync_test_map_once(cfg, target_branch=target_branch, logger=logger)
        return sync_test_map_watch(
            cfg,
            target_branch=target_branch,
            interval_seconds=interval,
            logger=logger,
        )
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
