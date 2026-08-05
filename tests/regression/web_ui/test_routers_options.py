"""Real unit tests for api/routers/options.py.

Single test file for the options router. ``tensor_cast`` IS importable in this
env, so the happy path is exercised for real; import-failure / bad-file
branches use fixture-scoped ``patch.dict(sys.modules)`` (reverts after the
test) and ``patch`` of os/importlib helpers — NOT global conftest mocking.
Per tests/SKILL.md.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

from api.routers.options import _load_new_device_profile_modules, list_devices, router


class TestRouterConfiguration:
    """Tests for router setup."""

    def test_router_prefix(self):
        assert router.prefix == "/api/options"

    def test_router_tag(self):
        assert router.tags == ["options"]


class TestListDevices:
    """Tests for the list_devices endpoint (live DeviceProfile registry)."""

    def test_returns_real_device_profiles(self):
        """Happy path: tensor_cast is importable → real profiles are listed."""
        result = list_devices()
        assert len(result) > 0
        # Each entry is an OptionItem(value=name, label=name).
        assert all(item.value == item.label for item in result)
        assert all(isinstance(item.value, str) for item in result)

    def test_returns_empty_when_tensor_cast_unavailable(self):
        """When tensor_cast import fails, the endpoint returns []."""
        with patch.dict(sys.modules, {"tensor_cast": None}):
            assert list_devices() == []


class TestLoadNewDeviceProfileModules:
    """Tests for the directory re-scan that picks up custom profiles."""

    def test_returns_early_when_pkg_import_fails(self):
        """If tensor_cast.device_profiles can't import, scan is skipped."""
        with (
            patch.dict(sys.modules, {"tensor_cast.device_profiles": None}),
            patch("api.routers.options.logger") as mock_logger,
        ):
            _load_new_device_profile_modules()
        mock_logger.debug.assert_called_once()

    def test_returns_early_when_pkg_has_no_file(self):
        """A package with no __file__ is skipped via patch.object on the real module."""
        import tensor_cast.device_profiles

        with patch.object(tensor_cast.device_profiles, "__file__", None):
            _load_new_device_profile_modules()

    def test_returns_early_when_dir_missing(self):
        """A __file__ whose dir doesn't exist is skipped."""
        mock_pkg = MagicMock()
        mock_pkg.__file__ = "/nonexistent/pkg/__init__.py"
        with (
            patch.dict(sys.modules, {"tensor_cast.device_profiles": mock_pkg}),
            patch("os.path.isdir", return_value=False),
        ):
            _load_new_device_profile_modules()

    def test_imports_new_module_file(self):
        """A new .py file (not yet imported) is imported via importlib."""
        mock_pkg = MagicMock()
        mock_pkg.__file__ = "/fake/pkg/__init__.py"
        with (
            patch.dict(sys.modules, {"tensor_cast.device_profiles": mock_pkg}),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["new_profile.py", "_skip.py", "readme.md"]),
            patch("importlib.import_module") as mock_import,
        ):
            _load_new_device_profile_modules()
        # Only new_profile.py qualifies (.py, not _-prefixed); the others skip.
        mock_import.assert_called_once_with("tensor_cast.device_profiles.new_profile")

    def test_skips_already_loaded_module(self):
        """A .py file already in sys.modules is not re-imported."""
        mock_pkg = MagicMock()
        mock_pkg.__file__ = "/fake/pkg/__init__.py"
        modname = "tensor_cast.device_profiles.already_loaded"
        with (
            patch.dict(sys.modules, {"tensor_cast.device_profiles": mock_pkg, modname: MagicMock()}),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["already_loaded.py"]),
            patch("importlib.import_module") as mock_import,
        ):
            _load_new_device_profile_modules()
        mock_import.assert_not_called()

    def test_bad_file_logs_warning_and_continues(self):
        """One failing import logs a warning but doesn't abort the scan."""
        mock_pkg = MagicMock()
        mock_pkg.__file__ = "/fake/pkg/__init__.py"
        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            # Delegate to the real importer for everything except the bad profile,
            # so patch's own module-resolution (pkgutil.resolve_name) keeps working.
            if name.endswith(".bad"):
                raise RuntimeError("boom")
            if name.startswith("tensor_cast.device_profiles."):
                return MagicMock()
            return real_import(name, *args, **kwargs)

        # NOTE: patch logger BEFORE importlib.import_module — a later patch's
        # __enter__ resolves its target via importlib, which would otherwise hit
        # the mocked importer.
        with (
            patch.dict(sys.modules, {"tensor_cast.device_profiles": mock_pkg}),
            patch("api.routers.options.logger") as mock_logger,
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["good.py", "bad.py"]),
            patch("importlib.import_module", side_effect=fake_import),
        ):
            _load_new_device_profile_modules()
        mock_logger.warning.assert_called_once()
        # call_args: ("Failed to import device profile %s", modname)
        assert mock_logger.warning.call_args[0][1].endswith(".bad")
