"""Shared pytest subprocess helpers for CI gate and test_map tooling."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.helpers._config import ConfigError
from scripts.helpers._paths import REPO_ROOT

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


def collect_test_node_ids(targets: Sequence[str], *, marker: str) -> tuple[str, ...]:
    """Collect pytest node ids matching *marker* from collect-only stdout."""
    if not targets:
        return ()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        *PYTEST_IGNORE_ADDOPTS,
        "-m",
        marker,
        "--collect-only",
        "-q",
        "--no-header",
    ]
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
    marker: str,
    collected_count: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Assemble a pytest command with explicit marker and collect-first xdist sizing."""
    cmd = [
        python,
        "-m",
        "pytest",
        *targets,
        *PYTEST_IGNORE_ADDOPTS,
        "-m",
        marker,
        *xdist_worker_args(collected_count),
        *_DEFAULT_RUN_ARGS,
        *extra_args,
    ]
    return cmd
