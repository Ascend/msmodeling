"""Shared pytest subprocess helpers for CI gate and test_map tooling."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.helpers._config import ConfigError
from scripts.helpers._paths import REPO_ROOT

logger = logging.getLogger(__name__)

_NOT_FOUND_RE = re.compile(r"^ERROR: not found: ([^\n\r]+)", re.MULTILINE)

PYTEST_IGNORE_ADDOPTS: Final[list[str]] = ["-o", "addopts="]

_DEFAULT_RUN_ARGS: Final[tuple[str, ...]] = (
    "-vv",
    "--tb=short",
    "--durations=20",
    "--disable-warnings",
)


def _parse_collect_only_node_ids(stdout: str) -> tuple[str, ...]:
    node_ids: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped and stripped.startswith("tests/"):
            node_ids.append(stripped)
    return tuple(node_ids)


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
    """Collect pytest node ids from *targets* without applying a marker filter."""
    return _collect_with_marker(targets, marker=None)


def _collect_with_marker(targets: Sequence[str], *, marker: str | None) -> tuple[str, ...]:
    if not targets:
        return ()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        *PYTEST_IGNORE_ADDOPTS,
        "--collect-only",
        "-q",
        "--no-header",
    ]
    if marker is not None:
        cmd.extend(["-m", marker])
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 5):
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ConfigError(f"pytest collect-only failed (exit {proc.returncode})" + (f": {detail}" if detail else ""))

    return _parse_collect_only_node_ids(proc.stdout)


def _run_collect_only(targets: Sequence[str], *, marker: str | None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        *PYTEST_IGNORE_ADDOPTS,
        "--collect-only",
        "-q",
        "--no-header",
    ]
    if marker is not None:
        cmd.extend(["-m", marker])
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


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

    batch_ids = _parse_collect_only_node_ids(proc.stdout)
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
