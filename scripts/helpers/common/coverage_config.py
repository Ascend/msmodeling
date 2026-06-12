"""Shared product source roots for coverage, test_map, and testguard."""

from __future__ import annotations

import os
from typing import Final

from scripts.helpers.common.pytest_runner import xdist_worker_args

PRODUCT_SOURCE_PREFIXES: Final[tuple[str, ...]] = (
    "cli/",
    "serving_cast/",
    "tensor_cast/",
    "web_ui/",
    "scripts/",
    "tools/",
)

COV_PACKAGES: Final[tuple[str, ...]] = tuple(p.rstrip("/") for p in PRODUCT_SOURCE_PREFIXES)


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
    args: list[str] = [f"--cov={pkg}" for pkg in COV_PACKAGES]
    args.append("--cov-branch")
    if cov_context:
        args.append("--cov-context=test")
    if append:
        args.append("--cov-append")
    args.append("--cov-report=")
    return args
