"""Tests for helpers._config.Config."""

from __future__ import annotations

import os

import pytest
from scripts.helpers._config import Config, ConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove config-related env vars before each test."""
    for key in tuple(os.environ):
        if key.startswith("MSMODELING_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)


# ---------------------------------------------------------------------------
# from_env — defaults
# ---------------------------------------------------------------------------


def test_from_env_all_optional_vars_use_fallbacks() -> None:
    cfg = Config.from_env()
    assert cfg.base_branch == "master"
    assert cfg.line_threshold == 60.0
    assert cfg.branch_threshold == 40.0
    assert cfg.benchmark_parallel is False
    assert cfg.feishu_webhook_url == ""
    assert cfg.msmodeling_cache == ".msmodeling_cache"
    assert cfg.weights_prune is False


def test_from_env_test_map_path_none_when_not_set() -> None:
    cfg = Config.from_env()
    assert cfg.test_map_path is None


# ---------------------------------------------------------------------------
# from_env — set values
# ---------------------------------------------------------------------------


def test_from_env_all_values_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSMODELING_TEST_MAP_PATH", "/tmp/map.json")
    monkeypatch.setenv("MSMODELING_TEST_BASE_BRANCH", "develop")
    monkeypatch.setenv("MSMODELING_TEST_LINE_THRESHOLD", "85.5")
    monkeypatch.setenv("MSMODELING_TEST_BRANCH_THRESHOLD", "60.0")
    monkeypatch.setenv("MSMODELING_BENCHMARK_PARALLEL", "1")
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setenv("MSMODELING_CACHE", "/tmp/cache")
    monkeypatch.setenv("MSMODELING_TEST_WEIGHTS_PRUNE", "0")

    cfg = Config.from_env()
    assert cfg.test_map_path == "/tmp/map.json"
    assert cfg.base_branch == "develop"
    assert cfg.line_threshold == 85.5
    assert cfg.branch_threshold == 60.0
    assert cfg.benchmark_parallel is True
    assert cfg.feishu_webhook_url == "https://example.com/hook"
    assert cfg.msmodeling_cache == "/tmp/cache"
    assert cfg.weights_prune is False


# ---------------------------------------------------------------------------
# _parse_float — error
# ---------------------------------------------------------------------------


def test_from_env_line_threshold_invalid_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_TEST_LINE_THRESHOLD", "abc")
    with pytest.raises(ConfigError, match="MSMODELING_TEST_LINE_THRESHOLD must be a number, got 'abc'"):
        Config.from_env()


# ---------------------------------------------------------------------------
# _validate_threshold — out of range
# ---------------------------------------------------------------------------


def test_from_env_line_threshold_negative_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_TEST_LINE_THRESHOLD", "-1")
    with pytest.raises(ConfigError, match="MSMODELING_TEST_LINE_THRESHOLD must be in"):
        Config.from_env()


def test_from_env_line_threshold_over_100_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_TEST_LINE_THRESHOLD", "101")
    with pytest.raises(ConfigError, match="MSMODELING_TEST_LINE_THRESHOLD must be in"):
        Config.from_env()


def test_from_env_branch_threshold_negative_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_TEST_BRANCH_THRESHOLD", "-5")
    with pytest.raises(ConfigError, match="MSMODELING_TEST_BRANCH_THRESHOLD must be in"):
        Config.from_env()


def test_from_env_branch_threshold_over_100_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_TEST_BRANCH_THRESHOLD", "150")
    with pytest.raises(ConfigError, match="MSMODELING_TEST_BRANCH_THRESHOLD must be in"):
        Config.from_env()


# ---------------------------------------------------------------------------
# _parse_bool — true variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on"])
def test_from_env_benchmark_parallel_true_variants(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSMODELING_BENCHMARK_PARALLEL", value)
    cfg = Config.from_env()
    assert cfg.benchmark_parallel is True


# ---------------------------------------------------------------------------
# _parse_bool — false variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "off"])
def test_from_env_benchmark_parallel_false_variants(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSMODELING_BENCHMARK_PARALLEL", value)
    cfg = Config.from_env()
    assert cfg.benchmark_parallel is False


# ---------------------------------------------------------------------------
# _parse_bool — error
# ---------------------------------------------------------------------------


def test_from_env_weights_prune_invalid_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_TEST_WEIGHTS_PRUNE", "maybe")
    with pytest.raises(
        ConfigError,
        match="MSMODELING_TEST_WEIGHTS_PRUNE must be a boolean.*got 'maybe'",
    ):
        Config.from_env()
