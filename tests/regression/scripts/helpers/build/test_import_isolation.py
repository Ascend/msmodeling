"""Import-chain isolation: build.main must not require pydantic at import time."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.helpers import _config, _errors

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter with repo root on PYTHONPATH."""
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_import_build_main_does_not_require_pydantic() -> None:
    """Normal: importing scripts.helpers.build.main succeeds with pydantic blocked."""
    code = r"""
import builtins
import sys

blocked = ("pydantic", "pydantic_settings")
real_import = builtins.__import__

def fake_import(name, *args, **kwargs):
    if name in blocked or name.startswith("pydantic.") or name.startswith("pydantic_settings."):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = fake_import
# Ensure a clean import of the build chain.
for key in list(sys.modules):
    if key == "scripts" or key.startswith("scripts."):
        del sys.modules[key]
    if key in blocked or key.startswith("pydantic.") or key.startswith("pydantic_settings."):
        del sys.modules[key]

import scripts.helpers.build.main as m
assert hasattr(m, "main") and hasattr(m, "run_build")
assert "pydantic" not in sys.modules
assert "pydantic_settings" not in sys.modules
print("ok")
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_import_pyproject_toml_does_not_require_pydantic() -> None:
    """Normal: pyproject_toml import chain does not need pydantic."""
    code = r"""
import builtins
import sys

blocked = ("pydantic", "pydantic_settings")
real_import = builtins.__import__

def fake_import(name, *args, **kwargs):
    if name in blocked or name.startswith("pydantic.") or name.startswith("pydantic_settings."):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = fake_import
for key in list(sys.modules):
    if key == "scripts" or key.startswith("scripts."):
        del sys.modules[key]
    if key in blocked or key.startswith("pydantic.") or key.startswith("pydantic_settings."):
        del sys.modules[key]

import scripts.helpers.common.pyproject_toml as m
assert hasattr(m, "read_project_version")
assert "pydantic" not in sys.modules
print("ok")
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_errors_module_is_stdlib_only() -> None:
    """Edge: _errors itself must not pull third-party packages."""
    code = r"""
import ast
from pathlib import Path

path = Path("scripts/helpers/_errors.py")
tree = ast.parse(path.read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
        else:
            names = [node.module.split(".", 1)[0]] if node.module else []
        for name in names:
            assert name in {"__future__", "typing", "dataclasses", "collections", "abc"}, name
print("ok")
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_config_reexports_same_config_error() -> None:
    """Normal: ConfigError has a single definition, re-exported from _config."""
    assert _config.ConfigError is _errors.ConfigError
    assert _config.format_expected_got is _errors.format_expected_got
