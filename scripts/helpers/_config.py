"""Centralised env-var configuration for all helpers.

Thin wrapper over os.environ with validation — avoids pydantic-settings
dependency while giving single-source-of-truth for every config key used
across ci_gate, nightly, and common modules.

Shell entry scripts (run_ci_gate.sh, run_nightly.sh, etc.) set env-vars
with defaults. Python reads them here; missing optional keys fall back to
built-in defaults so that callers need not set every variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Raised when a required config key is missing or invalid."""


def _parse_float(key: str, *, default: float | None = None) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        if default is not None:
            return default
        raise ConfigError(f"{key} is required (set it in the shell entry script)")
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _parse_bool(key: str, *, default: bool | None = None) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        if default is not None:
            return default
        raise ConfigError(f"{key} is required (set it in the shell entry script)")
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ConfigError(f"{key} must be a boolean (0/1/true/false), got {raw!r}")


def _parse_str(key: str, *, default: str | None = None) -> str:
    raw = os.environ.get(key, "").strip()
    if not raw:
        if default is not None:
            return default
        raise ConfigError(f"{key} is required (set it in the shell entry script)")
    return raw


def _validate_threshold(key: str, value: float) -> float:
    if not (0 <= value <= 100):
        raise ConfigError(f"{key} must be in [0, 100], got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Config:
    # --- test_map ---
    test_map_path: str | None  # MSMODELING_TEST_MAP_PATH

    # --- git ---
    base_branch: str  # MSMODELING_TEST_BASE_BRANCH

    # --- nightly coverage report ---
    line_threshold: float  # MSMODELING_TEST_LINE_THRESHOLD
    branch_threshold: float  # MSMODELING_TEST_BRANCH_THRESHOLD

    # --- nightly ---
    benchmark_parallel: bool  # MSMODELING_BENCHMARK_PARALLEL
    feishu_webhook_url: str  # FEISHU_WEBHOOK_URL

    # --- cache ---
    msmodeling_cache: str  # MSMODELING_CACHE

    # --- pytest session (consumed by conftest.py, not helpers) ---
    weights_prune: bool  # MSMODELING_TEST_WEIGHTS_PRUNE

    @classmethod
    def from_env(cls) -> Config:
        test_map_path = os.environ.get("MSMODELING_TEST_MAP_PATH") or None
        line_threshold = _validate_threshold(
            "MSMODELING_TEST_LINE_THRESHOLD",
            _parse_float("MSMODELING_TEST_LINE_THRESHOLD", default=60.0),
        )
        branch_threshold = _validate_threshold(
            "MSMODELING_TEST_BRANCH_THRESHOLD",
            _parse_float("MSMODELING_TEST_BRANCH_THRESHOLD", default=40.0),
        )
        return cls(
            test_map_path=test_map_path,
            base_branch=_parse_str("MSMODELING_TEST_BASE_BRANCH", default="master"),
            line_threshold=line_threshold,
            branch_threshold=branch_threshold,
            benchmark_parallel=_parse_bool("MSMODELING_BENCHMARK_PARALLEL", default=False),
            feishu_webhook_url=os.environ.get("FEISHU_WEBHOOK_URL", "").strip(),
            msmodeling_cache=_parse_str("MSMODELING_CACHE", default=".msmodeling_cache"),
            weights_prune=_parse_bool("MSMODELING_TEST_WEIGHTS_PRUNE", default=False),
        )
