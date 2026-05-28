"""Shared product source roots for coverage, test_map, and testguard."""

from __future__ import annotations

from typing import Final

PRODUCT_SOURCE_PREFIXES: Final[tuple[str, ...]] = (
    "cli/",
    "serving_cast/",
    "tensor_cast/",
    "web_ui/",
    "scripts/",
    "tools/",
)

COV_PACKAGES: Final[tuple[str, ...]] = tuple(p.rstrip("/") for p in PRODUCT_SOURCE_PREFIXES)


def cov_pytest_args(*, cov_context: bool = False, append: bool = False) -> list[str]:
    """Build --cov flags for pytest. Always uses --cov-branch, no terminal report."""
    args: list[str] = [f"--cov={pkg}" for pkg in COV_PACKAGES]
    args.append("--cov-branch")
    if cov_context:
        args.append("--cov-context=test")
    if append:
        args.append("--cov-append")
    args.append("--cov-report=")
    return args
