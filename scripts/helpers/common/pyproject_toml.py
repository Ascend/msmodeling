"""Load and parse pyproject.toml from the repository root."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final, cast

from scripts.helpers._errors import ConfigError, format_expected_got
from scripts.helpers._paths import REPO_ROOT

_PYPROJECT_REL: Final = Path("pyproject.toml")


def _load_tomllib() -> Any:
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError:
        try:
            return importlib.import_module("tomli")
        except ImportError as exc:
            raise ConfigError(
                "tomli required to parse pyproject.toml on Python < 3.11. "
                "Run: uv sync --frozen --group build  (or --group ci for tests)"
            ) from exc


def decode_pyproject_bytes(raw: bytes) -> dict[str, object]:
    """Decode ``pyproject.toml`` bytes into a nested dict.

    Args:
        raw: UTF-8 encoded TOML content.

    Returns:
        Parsed TOML tables as a dict.

    Raises:
        ConfigError: If the TOML is invalid.
    """
    tomllib = _load_tomllib()

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: invalid TOML: {exc}") from exc
    return cast("dict[str, object]", data)


def read_pyproject_data(*, repo_root: Path | None = None) -> dict[str, object]:
    """Read and parse ``pyproject.toml`` from *repo_root* (or the repo root).

    Args:
        repo_root: Repository root containing ``pyproject.toml``. Defaults to
            :data:`scripts.helpers._paths.REPO_ROOT`.

    Returns:
        Parsed TOML tables as a dict.

    Raises:
        ConfigError: If the file is missing, unreadable, or invalid.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    pyproject_path = root / _PYPROJECT_REL
    if not pyproject_path.is_file():
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: file not found")

    try:
        raw = pyproject_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: cannot read file: {exc}") from exc
    return decode_pyproject_bytes(raw)


def read_project_version(*, repo_root: Path | None = None) -> str:
    """Return ``project.version`` from ``pyproject.toml``.

    Args:
        repo_root: Repository root containing ``pyproject.toml``. Defaults to
            :data:`scripts.helpers._paths.REPO_ROOT`.

    Returns:
        The project version string.

    Raises:
        ConfigError: If ``project`` / ``project.version`` is missing or invalid.
    """
    data = read_pyproject_data(repo_root=repo_root)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: {format_expected_got('project', 'a table', project)}")
    version = project.get("version")
    if not isinstance(version, str):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: {format_expected_got('project.version', 'a string', version)}")
    return version
