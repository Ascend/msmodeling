"""Tests for common.test_map_config — resolve_test_map_path, is_config_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.common.test_map_config import CONFIG_FILE_NAMES, is_config_path, resolve_test_map_path

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


# ---------------------------------------------------------------------------
# is_config_path
# ---------------------------------------------------------------------------


def test_is_config_path_dot_ci_directory_returns_true() -> None:
    assert is_config_path("tests/.ci/some_config.yaml") is True


def test_is_config_path_conftest_returns_true() -> None:
    assert is_config_path("tests/conftest.py") is True


def test_is_config_path_subdirectory_conftest_returns_true() -> None:
    assert is_config_path("tests/regression/web_ui/conftest.py") is True


@pytest.mark.parametrize("fname", sorted(CONFIG_FILE_NAMES))
def test_is_config_path_config_filenames_return_true(fname: str) -> None:
    assert is_config_path(fname) is True


def test_is_config_path_source_file_returns_false() -> None:
    assert is_config_path("cli/main.py") is False


def test_is_config_path_test_file_returns_false() -> None:
    assert is_config_path("tests/smoke/test_x.py") is False


def test_is_config_path_tests_prefix_without_conftest_returns_false() -> None:
    assert is_config_path("tests/helpers/foo.py") is False
