"""Centralised env-var configuration for all helpers.

Thin wrapper over os.environ with validation — avoids pydantic-settings
dependency while giving single-source-of-truth for every config key used
across ci_gate, nightly, and common modules.

Default values are managed by shell entry scripts (run_ci_gate.sh, run_nightly.sh);
Python reads env-vars without hard-coded fallbacks so that defaults live in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Raised when a required config key is missing or invalid."""


def _require(key: str) -> str:
    raw = os.environ.get(key, "").strip()
    if not raw:
        raise ConfigError(f"{key} is required (set it in the shell entry script)")
    return raw


@dataclass(frozen=True, slots=True)
class Config:
    # --- test_map ---
    test_map_path: str | None  # MSMODELING_TEST_MAP_PATH

    # --- git ---
    base_branch: str  # MSMODELING_TEST_BASE_BRANCH

    # --- coverage gate ---
    line_threshold: float  # MSMODELING_TEST_LINE_THRESHOLD
    branch_threshold: float  # MSMODELING_TEST_BRANCH_THRESHOLD

    # --- nightly ---
    benchmark_parallel: bool  # MSMODELING_BENCHMARK_PARALLEL
    test_map_marker: str  # MSMODELING_TEST_MAP_MARKER
    feishu_webhook_url: str  # FEISHU_WEBHOOK_URL

    # --- pytest session (consumed by conftest.py, not helpers) ---
    weights_prune: bool  # MSMODELING_TEST_WEIGHTS_PRUNE

    @classmethod
    def from_env(cls) -> Config:
        test_map_path = os.environ.get("MSMODELING_TEST_MAP_PATH") or None
        if not test_map_path:
            raise ConfigError("MSMODELING_TEST_MAP_PATH is required and must not be empty")
        return cls(
            test_map_path=test_map_path,
            base_branch=_require("MSMODELING_TEST_BASE_BRANCH"),
            line_threshold=_parse_float("MSMODELING_TEST_LINE_THRESHOLD"),
            branch_threshold=_parse_float("MSMODELING_TEST_BRANCH_THRESHOLD"),
            benchmark_parallel=_parse_bool("MSMODELING_BENCHMARK_PARALLEL"),
            test_map_marker=_require("MSMODELING_TEST_MAP_MARKER"),
            feishu_webhook_url=os.environ.get("FEISHU_WEBHOOK_URL", "").strip(),
            weights_prune=_parse_bool("MSMODELING_TEST_WEIGHTS_PRUNE"),
        )


def _parse_float(key: str) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        raise ConfigError(f"{key} is required (set it in the shell entry script)")
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _parse_bool(key: str) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        raise ConfigError(f"{key} is required (set it in the shell entry script)")
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ConfigError(f"{key} must be a boolean (0/1/true/false), got {raw!r}")
