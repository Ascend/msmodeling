"""Test map path resolution and config-file detection.

Extracted from scripts.helpers.ci_gate.py — single authority for MSMODELING_TEST_MAP_PATH
resolution and CONFIG_FILE_NAMES used by diff classification.
"""

from __future__ import annotations

from pathlib import Path

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.gate_policy import GATE_POLICY_REL

CONFIG_FILE_NAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        ".coveragerc",
        "requirements.txt",
        "uv.lock",
    }
)


def resolve_test_map_path(cfg: Config, *, must_exist: bool) -> Path:
    """Return test_map Path from config, validating existence when required.

    Args:
        cfg: Application config.
        must_exist: If True, raise ConfigError when file not found.

    Raises:
        ConfigError: If path not set, is a directory, or must_exist but missing.
    """
    raw: str | None = cfg.test_map_path
    if not raw:
        raise ConfigError("MSMODELING_TEST_MAP_PATH is not set")
    path = Path(raw)
    if path.is_dir():
        raise ConfigError(f"MSMODELING_TEST_MAP_PATH must be a file, got directory: {path}")
    if must_exist and not path.is_file():
        raise ConfigError(f"test_map not found: {path}")
    return path


def is_config_path(path: str) -> bool:
    """Return True if *path* is a CI config file that triggers full-suite."""
    if path.startswith("tests/") and path.endswith("/conftest.py"):
        return True
    return path.rsplit("/", 1)[-1] in CONFIG_FILE_NAMES


def is_full_suite_trigger_path(path: str) -> bool:
    """Return True when a change should run the full PR gate test suite."""
    if is_config_path(path):
        return True
    if path.startswith("scripts/helpers/"):
        return True
    return path == GATE_POLICY_REL.as_posix()
