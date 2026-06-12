"""Shared product source roots for coverage, test_map, and testguard."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.gate_policy import load_gate_policy
from scripts.helpers.common.pytest_runner import xdist_worker_args


def product_roots(repo_root: Path | None = None) -> tuple[str, ...]:
    """Product source path prefixes from tests/.ci/gate_policy.yaml ``roots``."""
    return load_gate_policy(repo_root or REPO_ROOT).roots


def pytest_xdist_args(*, collected_count: int | None = None) -> list[str]:
    """pytest-xdist parallelism; safe with pytest-cov when ``[tool.coverage.run] parallel`` is set.

    When *collected_count* is omitted, worker count defaults to ``os.cpu_count()`` (legacy callers).
    """
    if collected_count is None:
        collected_count = os.cpu_count() or 1
    return xdist_worker_args(collected_count)


def cov_pytest_args(*, cov_context: bool = False, append: bool = False) -> list[str]:
    """Build --cov flags for pytest. Always uses --cov-branch, no terminal report.

    Requires ``parallel = true`` under ``[tool.coverage.run]`` when used with ``pytest_xdist_args()``.
    """
    cov_packages = tuple(p.rstrip("/") for p in product_roots())
    args: list[str] = [f"--cov={pkg}" for pkg in cov_packages]
    args.append("--cov-branch")
    if cov_context:
        args.append("--cov-context=test")
    if append:
        args.append("--cov-append")
    args.append("--cov-report=")
    return args
