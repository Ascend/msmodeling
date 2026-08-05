"""Unit tests for plugin_discovery.collect_plugin_migration_paths.

Pure-logic module extracted from migrations/env.py so it can be tested without
the alembic runtime (env.py's top-level ``config = context.config`` only exists
while alembic is running a migration). Every branch is covered here.
"""

from __future__ import annotations

import logging


from plugin_discovery import collect_plugin_migration_paths


class _FakePlugin:
    """A plugin object exposing ``migrations_path``."""

    def __init__(self, migrations_path=None):
        self.migrations_path = migrations_path


class _FakeEntryPoint:
    """A minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, plugin=None, load_exc=None):
        self.name = name
        self._plugin = plugin
        self._load_exc = load_exc

    def load(self):
        if self._load_exc is not None:
            raise self._load_exc
        return lambda: self._plugin


def _patch_entry_points(monkeypatch, eps):
    """Patch importlib.metadata.entry_points(group=...) to return ``eps``."""

    def modern(group=None):
        return eps

    import importlib.metadata as im

    monkeypatch.setattr(im, "entry_points", modern)


class TestCollectPluginMigrationPaths:
    def test_empty_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "")
        assert collect_plugin_migration_paths() == []

    def test_whitespace_only_env_returns_empty(self, monkeypatch):
        """A value that splits to no real names (e.g. ',,') -> []."""
        monkeypatch.setenv("MSMODELING_PLUGINS", " , , ")
        assert collect_plugin_migration_paths() == []

    def test_no_matching_entry_points_returns_empty(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")
        _patch_entry_points(monkeypatch, [])  # no EPs registered
        assert collect_plugin_migration_paths() == []

    def test_plugin_not_in_allow_list_skipped(self, monkeypatch):
        """An EP whose name isn't in the allow-list is skipped."""
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")
        _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="other_plugin", plugin=_FakePlugin("/x"))])
        assert collect_plugin_migration_paths() == []

    def test_plugin_with_valid_dir_is_collected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")
        mig_dir = tmp_path / "plugin_versions"
        mig_dir.mkdir()
        _patch_entry_points(
            monkeypatch,
            [_FakeEntryPoint(name="demo_plugin", plugin=_FakePlugin(str(mig_dir)))],
        )
        assert collect_plugin_migration_paths() == [mig_dir]

    def test_plugin_with_nonexistent_dir_skipped(self, monkeypatch, tmp_path):
        """migrations_path pointing at a non-directory is skipped (path.is_dir() False)."""
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")
        _patch_entry_points(
            monkeypatch,
            [_FakeEntryPoint(name="demo_plugin", plugin=_FakePlugin(str(tmp_path / "nope")))],
        )
        assert collect_plugin_migration_paths() == []

    def test_plugin_without_migrations_path_skipped(self, monkeypatch):
        """A plugin lacking the migrations_path attribute contributes nothing."""
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")

        class _Bare:
            pass

        _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="demo_plugin", plugin=_Bare())])
        assert collect_plugin_migration_paths() == []

    def test_plugin_with_falsy_migrations_path_skipped(self, monkeypatch):
        """migrations_path that is None/empty is skipped (truthiness check)."""
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")
        _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="demo_plugin", plugin=_FakePlugin(None))])
        assert collect_plugin_migration_paths() == []

    def test_plugin_loader_raising_is_swallowed(self, monkeypatch, tmp_path, caplog):
        """A plugin whose .load() raises is silently skipped AND logged at ERROR."""
        # The broken plugin must be in the allowed list (MSMODELING_PLUGINS)
        # so it reaches the try/except in collect_plugin_migration_paths.
        monkeypatch.setenv("MSMODELING_PLUGINS", "broken,demo_plugin")
        mig_dir = tmp_path / "pv"
        mig_dir.mkdir()
        _patch_entry_points(
            monkeypatch,
            [
                _FakeEntryPoint(name="broken", load_exc=RuntimeError("boom")),
                _FakeEntryPoint(name="demo_plugin", plugin=_FakePlugin(str(mig_dir))),
            ],
        )
        # Broken plugin swallowed; the good one still collected.
        with caplog.at_level(logging.ERROR):
            result = collect_plugin_migration_paths()
        assert result == [mig_dir]
        # The broken plugin's load failure must have been logged (not silently dropped).
        assert any("Failed to load plugin" in r.message and "broken" in r.message for r in caplog.records), (
            f"Expected 'Failed to load plugin broken' log; got: {[r.message for r in caplog.records]}"
        )

    def test_plugin_factory_call_raising_is_swallowed(self, monkeypatch):
        """An exception from calling plugin_factory() (after .load() succeeds) is
        swallowed by the except — covers the try-body failure path.
        """
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo_plugin")

        class _FactoryExplodesEntryPoint(_FakeEntryPoint):
            def load(self):
                # Factory whose invocation raises -> caught inside the try body.
                def _factory():
                    raise RuntimeError("factory blew up")

                return _factory

        _patch_entry_points(monkeypatch, [_FactoryExplodesEntryPoint(name="demo_plugin")])
        assert collect_plugin_migration_paths() == []
