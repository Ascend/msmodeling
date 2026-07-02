"""Load and parse pyproject.toml from the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from scripts.helpers._config import ConfigError, format_expected_got
from scripts.helpers._paths import REPO_ROOT

_PYPROJECT_REL: Final = Path("pyproject.toml")


def _load_tomllib():
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ConfigError(
                "tomli required to parse pyproject.toml on Python < 3.11. Run: uv sync --frozen --group ci"
            ) from exc
    return tomllib


def decode_pyproject_bytes(raw: bytes) -> dict[str, object]:
    tomllib = _load_tomllib()

    try:
        return tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: invalid TOML: {exc}") from exc


def read_pyproject_data(*, repo_root: Path | None = None) -> dict[str, object]:
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
    data = read_pyproject_data(repo_root=repo_root)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: {format_expected_got('project', 'a table', project)}")
    version = project.get("version")
    if not isinstance(version, str):
        raise ConfigError(f"{_PYPROJECT_REL.as_posix()}: {format_expected_got('project.version', 'a string', version)}")
    return version
