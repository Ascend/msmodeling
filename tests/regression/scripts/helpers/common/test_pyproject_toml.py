"""Tests for common.pyproject_toml."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from scripts.helpers._config import ConfigError
from scripts.helpers.common import pyproject_toml


def test_decode_pyproject_bytes_valid() -> None:
    raw = b'[project]\nname = "msmodeling"\nversion = "1.2.3"\n'
    data = pyproject_toml.decode_pyproject_bytes(raw)
    assert data["project"] == {"name": "msmodeling", "version": "1.2.3"}


def test_decode_pyproject_bytes_invalid_toml_raises_config_error() -> None:
    with pytest.raises(ConfigError, match=r"pyproject\.toml: invalid TOML"):
        pyproject_toml.decode_pyproject_bytes(b"[project\nversion = bad")


def test_read_pyproject_data_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"pyproject\.toml: file not found"):
        pyproject_toml.read_pyproject_data(repo_root=tmp_path)


def test_read_pyproject_data_oserror_on_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def fail_read_bytes(self: Path) -> bytes:
        if self == pyproject_path:
            msg = "permission denied"
            raise OSError(msg)
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    with pytest.raises(ConfigError, match=r"pyproject\.toml: cannot read file"):
        pyproject_toml.read_pyproject_data(repo_root=tmp_path)


def test_read_project_version_normal(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "msmodeling"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    assert pyproject_toml.read_project_version(repo_root=tmp_path) == "0.2.0"


def test_read_project_version_missing_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "msmodeling"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"Expected 'project\.version' to be a string"):
        pyproject_toml.read_project_version(repo_root=tmp_path)


def test_read_project_version_missing_project_table_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('name = "msmodeling"\nversion = "0.2.0"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"Expected 'project' to be a table"):
        pyproject_toml.read_project_version(repo_root=tmp_path)


def test_read_project_version_non_table_project_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('project = "not-a-table"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"Expected 'project' to be a table. Got 'not-a-table'"):
        pyproject_toml.read_project_version(repo_root=tmp_path)


def test_read_project_version_non_str_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "msmodeling"\nversion = 42\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"Expected 'project\.version' to be a string"):
        pyproject_toml.read_project_version(repo_root=tmp_path)


def test_decode_uses_tomli_when_tomllib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    class FakeTomli:
        @staticmethod
        def loads(text: str) -> dict[str, object]:
            return {"parsed_by": "tomli", "text": text}

        class TOMLDecodeError(ValueError):
            pass

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "tomllib":
            msg = "No module named 'tomllib'"
            raise ModuleNotFoundError(msg)
        if name == "tomli":
            return FakeTomli()
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    data = pyproject_toml.decode_pyproject_bytes(b'[project]\nversion = "1.0.0"\n')
    assert data["parsed_by"] == "tomli"


def test_decode_raises_when_tomllib_and_tomli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name in {"tomllib", "tomli"}:
            msg = f"No module named '{name}'"
            raise ModuleNotFoundError(msg)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ConfigError, match=r"tomli required to parse pyproject\.toml on Python < 3\.11"):
        pyproject_toml.decode_pyproject_bytes(b'[project]\nversion = "1.0.0"\n')
