"""Test map path resolution.

Extracted from scripts.helpers.ci_gate.py — single authority for MSMODELING_TEST_MAP_PATH
resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from scripts.helpers._config import Config, ConfigError

# Hardcoded collection scope for test_map build/sync (not env-configurable).
TEST_MAP_EXECUTION_MARKER: Final = "not npu and not nightly and not network"
TEST_MAP_COLLECTION_MARKER: Final = "not nightly and not network"


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
