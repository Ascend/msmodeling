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

"""Shared pytest subprocess helpers for CI gate and test_map tooling."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.helpers._config import ConfigError
from scripts.helpers._paths import REPO_ROOT

logger = logging.getLogger(__name__)

_NOT_FOUND_RE = re.compile(r"^ERROR: not found: ([^\n\r]+)", re.MULTILINE)

PYTEST_IGNORE_ADDOPTS: Final[list[str]] = ["-o", "addopts="]
_COLLECT_IDS_PLUGIN: Final = "scripts.helpers.common.pytest_collect_ids"
_COLLECT_ALL_MARKEXPR: Final = "npu or not npu"
_COLLECT_NODE_IDS_ENV: Final = "MSMODELING_COLLECT_NODE_IDS"
_COLLECT_ENV_KEYS: Final = frozenset(
    {
        "PYTEST_ADDOPTS",
        "PYTEST_CURRENT_TEST",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_XDIST_TESTRUNUID",
    }
)

_DEFAULT_RUN_ARGS: Final[tuple[str, ...]] = (
    "-vv",
    "--tb=short",
    "--durations=20",
    "--disable-warnings",
)


def _parse_plugin_node_ids(text: str) -> tuple[str, ...]:
    """Parse collect-ids plugin output, one complete node id per line.

    Parameter ids may contain spaces, so the line is kept whole.
    """
    node_ids: list[str] = []
    for line in text.splitlines():
        node_id = _normalize_node_id(line)
        if "::" in node_id and node_id.startswith("tests/"):
            node_ids.append(node_id)
    return tuple(node_ids)


def _parse_collect_only_node_ids(stdout: str) -> tuple[str, ...]:
    """Parse pytest stdout when the collect-ids file is empty.

    Reporter lines can carry trailing text, so only the first token is kept.
    Node ids with spaces come from :func:`_parse_plugin_node_ids`.
    """
    node_ids: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if "::" not in stripped or "tests/" not in stripped:
            continue
        node_id = _normalize_node_id(stripped).split()[0]
        if "::" in node_id and node_id.startswith("tests/"):
            node_ids.append(node_id)
    return tuple(node_ids)


def _node_ids_from_collect(proc: subprocess.CompletedProcess[str]) -> tuple[str, ...]:
    plugin_ids = getattr(proc, "_plugin_node_ids", None)
    if plugin_ids:
        return plugin_ids
    return _parse_collect_only_node_ids(proc.stdout or "")


def _normalize_node_id(node_id: str) -> str:
    """Map pytest node ids to repo-relative ``tests/...`` form for comparison."""
    normalized = node_id.strip()
    tests_idx = normalized.find("tests/")
    if tests_idx >= 0:
        return normalized[tests_idx:]
    return normalized


def _parse_not_found_node_ids(stderr: str) -> frozenset[str]:
    return frozenset(_normalize_node_id(node_id) for node_id in _NOT_FOUND_RE.findall(stderr))


def collect_test_node_ids(targets: Sequence[str], *, marker: str | None) -> tuple[str, ...]:
    """Collect pytest node ids matching *marker* from collect-only stdout."""
    return _collect_with_marker(targets, marker=marker)


def collect_all_test_node_ids(targets: Sequence[str]) -> tuple[str, ...]:
    """Collect pytest node ids from *targets*, including npu, nightly, and network."""
    return _collect_with_marker(targets, marker=None)


def _collect_subprocess_env(node_id_file: str) -> dict[str, str]:
    """Drop parent pytest, xdist, and coverage settings from a nested collect."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _COLLECT_ENV_KEYS and not key.startswith("COV_CORE_")
    }
    root = str(REPO_ROOT)
    parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    if root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([root, *parts])
    env[_COLLECT_NODE_IDS_ENV] = node_id_file
    return env


def _collect_pytest_cmd(targets: Sequence[str], *, marker: str | None) -> list[str]:
    """Build collect-only argv that records one node id per line.

    ``-qq`` makes pytest print ``path: count`` instead of node ids. The collect-ids
    plugin writes the ids. The trailing ``-m`` wins over pyproject ``addopts``.
    ``npu or not npu`` keeps exempted npu tests when no marker was requested.
    ``no:cov`` and ``no:xdist`` keep the nested collect off the parent session.
    """
    return [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        *PYTEST_IGNORE_ADDOPTS,
        "--collect-only",
        "-qq",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:cov",
        "-p",
        "no:xdist",
        "-p",
        _COLLECT_IDS_PLUGIN,
        "-m",
        _COLLECT_ALL_MARKEXPR if marker is None else marker,
    ]


def _run_collect_only(targets: Sequence[str], *, marker: str | None) -> subprocess.CompletedProcess[str]:
    fd, node_id_file = tempfile.mkstemp(prefix="msmodeling-collect-", suffix=".txt")
    os.close(fd)
    try:
        proc = subprocess.run(
            _collect_pytest_cmd(targets, marker=marker),
            cwd=REPO_ROOT,
            env=_collect_subprocess_env(node_id_file),
            capture_output=True,
            text=True,
            check=False,
        )
        recorded = ""
        if os.path.isfile(node_id_file):
            with open(node_id_file, encoding="utf-8") as handle:
                recorded = handle.read()
    finally:
        try:
            os.unlink(node_id_file)
        except OSError:
            pass
    recorded_ids = _parse_plugin_node_ids(recorded)
    if recorded_ids:
        proc.stdout = "".join(f"{node_id}\n" for node_id in recorded_ids)
        setattr(proc, "_plugin_node_ids", recorded_ids)
    return proc


def _collect_with_marker(targets: Sequence[str], *, marker: str | None) -> tuple[str, ...]:
    if not targets:
        return ()

    proc = _run_collect_only(targets, marker=marker)
    if proc.returncode not in (0, 5):
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ConfigError(f"pytest collect-only failed (exit {proc.returncode})" + (f": {detail}" if detail else ""))

    return _node_ids_from_collect(proc)


def filter_collectable_node_ids(targets: Sequence[str], *, marker: str | None) -> tuple[str, ...]:
    """Return collectable node ids; drop stale ids instead of failing the batch."""
    if not targets:
        return ()
    if not all("::" in target for target in targets):
        if marker is None:
            return collect_all_test_node_ids(targets)
        return collect_test_node_ids(targets, marker=marker)

    proc = _run_collect_only(targets, marker=marker)
    if proc.returncode not in (0, 4, 5):
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ConfigError(f"pytest collect-only failed (exit {proc.returncode})" + (f": {detail}" if detail else ""))

    batch_ids = _node_ids_from_collect(proc)
    if batch_ids:
        target_set = frozenset(targets)
        return tuple(node_id for node_id in batch_ids if node_id in target_set)

    not_found = _parse_not_found_node_ids(proc.stderr or "")
    if not_found:
        remaining = tuple(target for target in targets if _normalize_node_id(target) not in not_found)
        dropped = len(targets) - len(remaining)
        if dropped:
            sample = ", ".join(sorted(not_found)[:3])
            logger.info("Dropped %d stale pytest node id(s); sample: %s", dropped, sample)
            return remaining

    if proc.returncode == 0:
        return tuple(targets)

    if proc.returncode == 5:
        return ()

    detail = (proc.stderr or proc.stdout or "").strip()
    raise ConfigError("pytest collect-only exit 4 with unparseable stderr" + (f": {detail[:500]}" if detail else ""))


def count_collected_tests(targets: Sequence[str], *, marker: str) -> int:
    """Collect pytest node ids matching *marker* and return the count."""
    return len(collect_test_node_ids(targets, marker=marker))


def xdist_worker_args(collected_count: int) -> list[str]:
    """Return pytest-xdist flags sized to the collected test count."""
    if collected_count == 0:
        return []
    worker_count = min(os.cpu_count() or 1, max(collected_count, 1))
    return ["-n", str(worker_count), "--dist", "worksteal"]


def build_pytest_cmd(
    python: str,
    targets: Sequence[str],
    *,
    marker: str | None,
    collected_count: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Assemble a pytest command with optional marker and collect-first xdist sizing."""
    cmd = [
        python,
        "-m",
        "pytest",
        *targets,
        *PYTEST_IGNORE_ADDOPTS,
    ]
    if marker is not None:
        cmd.extend(["-m", marker])
    cmd.extend([*xdist_worker_args(collected_count), *_DEFAULT_RUN_ARGS, *extra_args])
    return cmd
