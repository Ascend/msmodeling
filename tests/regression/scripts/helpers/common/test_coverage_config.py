"""Tests for common.coverage_config."""

from __future__ import annotations

from scripts.helpers.common.coverage_config import (
    COV_PACKAGES,
    PRODUCT_SOURCE_PREFIXES,
    cov_pytest_args,
    pytest_xdist_args,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_product_source_prefixes_not_empty() -> None:
    assert len(PRODUCT_SOURCE_PREFIXES) > 0


def test_cov_packages_match_prefixes() -> None:
    assert tuple(p.rstrip("/") for p in PRODUCT_SOURCE_PREFIXES) == COV_PACKAGES


# ---------------------------------------------------------------------------
# cov_pytest_args
# ---------------------------------------------------------------------------


def test_cov_args_default_no_context_no_append() -> None:
    args = cov_pytest_args()
    for pkg in COV_PACKAGES:
        assert f"--cov={pkg}" in args
    assert "--cov-branch" in args
    assert "--cov-context=test" not in args
    assert "--cov-append" not in args
    assert "--cov-report=" in args


def test_cov_args_with_context() -> None:
    args = cov_pytest_args(cov_context=True)
    assert "--cov-context=test" in args


def test_cov_args_with_append() -> None:
    args = cov_pytest_args(append=True)
    assert "--cov-append" in args


def test_cov_args_both_context_and_append() -> None:
    args = cov_pytest_args(cov_context=True, append=True)
    assert "--cov-context=test" in args
    assert "--cov-append" in args


def test_pytest_xdist_args() -> None:
    assert pytest_xdist_args() == ["-n", "auto"]
