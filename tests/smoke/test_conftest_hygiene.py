"""Guard against conftest modules polluting ``sys.modules`` (e.g. tensor_cast mocks)."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTEST_PLUGIN_MODULES = (
    "tests.regression.tensor_cast.conftest",
    "tests.regression.serving_cast.conftest",
)


def _discover_conftest_modules() -> list[str]:
    return [
        ".".join(path.relative_to(_REPO_ROOT).with_suffix("").parts)
        for path in sorted(_REPO_ROOT.glob("tests/**/conftest.py"))
    ]


def _tensor_cast_module_keys() -> list[str]:
    return [key for key in sys.modules if key == "tensor_cast" or key.startswith("tensor_cast.")]


def _snapshot_tensor_cast_modules() -> dict[str, types.ModuleType]:
    return {key: sys.modules[key] for key in _tensor_cast_module_keys()}


def _restore_tensor_cast_modules(saved: dict[str, types.ModuleType]) -> None:
    for key in _tensor_cast_module_keys():
        if key not in saved:
            del sys.modules[key]
    sys.modules.update(saved)


def _load_conftest_modules_like_pytest() -> None:
    """Mirror pytest conftest discovery without re-importing already-loaded plugins."""
    if "tests.conftest" not in sys.modules:
        importlib.import_module("tests.conftest")
    for name in _PYTEST_PLUGIN_MODULES:
        if name not in sys.modules:
            importlib.import_module(name)
    for name in _discover_conftest_modules():
        if name in _PYTEST_PLUGIN_MODULES or name == "tests.conftest":
            continue
        if name not in sys.modules:
            importlib.import_module(name)


def test_tensor_cast_spec_intact_after_conftest_plugins_loaded() -> None:
    """Conftest load order must not replace real tensor_cast entries in sys.modules."""
    saved = _snapshot_tensor_cast_modules()
    try:
        _load_conftest_modules_like_pytest()
        tensor_cast = importlib.import_module("tensor_cast")
        assert isinstance(tensor_cast, types.ModuleType)
        assert not isinstance(tensor_cast, MagicMock)
        assert getattr(tensor_cast, "__spec__", None) is not None

        from tensor_cast.device import DeviceProfile

        assert inspect.isclass(DeviceProfile)  # pylint: disable=no-member
        assert not isinstance(DeviceProfile, MagicMock)
        assert DeviceProfile.__module__ == "tensor_cast.device"
    finally:
        _restore_tensor_cast_modules(saved)
