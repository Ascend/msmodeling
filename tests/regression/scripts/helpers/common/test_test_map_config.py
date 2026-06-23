"""Tests for common.test_map_config — resolve_test_map_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.common.test_map_config import (
    TEST_MAP_COLLECTION_MARKER,
    TEST_MAP_EXECUTION_MARKER,
    resolve_test_map_path,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg_with_valid_path(tmp_path_factory: pytest.TempPathFactory) -> Config:
    path = tmp_path_factory.mktemp("test_map") / "map.json"
    path.write_text("{}", encoding="utf-8")
    return Config(
        test_map_path=str(path),
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )


# ---------------------------------------------------------------------------
# resolve_test_map_path — success
# ---------------------------------------------------------------------------


def test_resolve_must_exist_returns_path_when_file_exists(
    cfg_with_valid_path: Config,
) -> None:
    result = resolve_test_map_path(cfg_with_valid_path, must_exist=True)
    assert result.is_file()


def test_resolve_must_exist_false_returns_path_even_when_missing() -> None:
    cfg = Config(
        test_map_path="/nonexistent/path.json",
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )
    result = resolve_test_map_path(cfg, must_exist=False)
    assert result == Path("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# resolve_test_map_path — errors
# ---------------------------------------------------------------------------


def test_resolve_empty_path_raises_config_error() -> None:
    cfg = Config(
        test_map_path="",
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )
    with pytest.raises(ConfigError, match="MSMODELING_TEST_MAP_PATH is not set"):
        resolve_test_map_path(cfg, must_exist=False)


def test_resolve_directory_path_raises_config_error(tmp_path: Path) -> None:
    cfg = Config(
        test_map_path=str(tmp_path),
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )
    with pytest.raises(ConfigError, match=f"must be a file, got directory: {tmp_path}"):
        resolve_test_map_path(cfg, must_exist=False)


def test_resolve_must_exist_missing_file_raises_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    cfg = Config(
        test_map_path=str(missing),
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )
    with pytest.raises(ConfigError, match=f"not found: {missing}"):
        resolve_test_map_path(cfg, must_exist=True)


def test_test_map_markers_match_ci_gate_scope() -> None:
    assert TEST_MAP_EXECUTION_MARKER == "not npu and not nightly and not network"
    assert TEST_MAP_COLLECTION_MARKER == "not nightly and not network"
