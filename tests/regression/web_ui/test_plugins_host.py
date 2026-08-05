"""Unit tests for the plugin host: contract, loader, manager (web_ui/backend/plugins/)."""

from __future__ import annotations

import sys
from unittest.mock import patch

import plugins.loader as _loader
import pytest
from fastapi import APIRouter, FastAPI
from plugins.contract import API_VERSION, MsmdPlugin, check_api_version
from plugins.loader import discover_plugins, get_allowed_plugins, topological_sort
from plugins.manager import PluginManager

# Construct plugin instances with the SAME MsmdPlugin class the loader will
# isinstance-check against. The editable msmodeling install can make
# plugins.contract (flat) and web_ui.backend.plugins.contract (qualified) resolve
# to two distinct class objects; loader.MsmdPlugin is exactly what load_plugins
# uses, so instances built from it always pass the check.
_PluginCls = _loader.MsmdPlugin


# --- contract ---------------------------------------------------------------


def _plugin(**over):
    base = {
        "id": "p",
        "version": "1.0.0",
        "api_version": "1",
        "router": None,
        "mount_path": None,
        "migrations_path": None,
        "startup": None,
        "shutdown": None,
        "depends": (),
        "extension_points": {},
    }
    base.update(over)
    return _PluginCls(**base)


class TestContract:
    def test_check_api_version_match(self):
        assert check_api_version(API_VERSION) is True

    def test_check_api_version_mismatch(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="plugins.contract"):
            assert check_api_version("999") is False
        assert any("does not match" in r.message for r in caplog.records)

    def test_msmdplugin_defaults(self):
        p = MsmdPlugin(id="x", version="1", api_version="1")
        assert p.router is None and p.mount_path is None and p.migrations_path is None
        assert p.startup is None and p.shutdown is None
        assert p.depends == () and p.extension_points == {}


# --- loader.discover / allow-list -------------------------------------------


def _ep(name, factory, *, raises=False):
    """Build a fake entry point whose .load() yields factory (or raises)."""

    class _FakeEP:
        def __init__(self):
            self.name = name

        def load(self):
            if raises:
                raise RuntimeError("boom")
            return factory

    return _FakeEP()


class TestAllowedPlugins:
    def test_unset_env(self, monkeypatch):
        monkeypatch.delenv("MSMODELING_PLUGINS", raising=False)
        assert get_allowed_plugins() == set()

    def test_empty_env(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "")
        assert get_allowed_plugins() == set()

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "a,b,c")
        assert get_allowed_plugins() == {"a", "b", "c"}

    def test_whitespace_and_empties(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", " a , ,b ")
        assert get_allowed_plugins() == {"a", "b"}


class TestDiscover:
    def test_no_entry_points(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            assert discover_plugins() == {}

    def test_discovers_and_dedups(self):
        with patch(
            "importlib.metadata.entry_points",
            return_value=[_ep("a", lambda: _plugin(id="a")), _ep("b", lambda: _plugin(id="b"))],
        ):
            found = discover_plugins()
        assert set(found) == {"a", "b"}

    def test_duplicate_id_raises(self):
        with (
            patch(
                "importlib.metadata.entry_points",
                return_value=[_ep("a", lambda: _plugin(id="a")), _ep("a", lambda: _plugin(id="a"))],
            ),
            pytest.raises(RuntimeError, match="Duplicate plugin ID"),
        ):
            discover_plugins()


# --- loader.topological_sort ------------------------------------------------


class TestTopoSort:
    def test_no_deps(self):
        plugins = {"a": _plugin(id="a"), "b": _plugin(id="b")}
        # Both have in-degree 0; order is insertion-stable among them.
        assert set(topological_sort(plugins)) == {"a", "b"}

    def test_dependency_ordered_before_dependent(self):
        plugins = {
            "a": _plugin(id="a", depends=("b",)),
            "b": _plugin(id="b"),
        }
        order = topological_sort(plugins)
        assert order.index("b") < order.index("a")

    def test_missing_dependency_raises(self):
        plugins = {"a": _plugin(id="a", depends=("ghost",))}
        with pytest.raises(ValueError, match="unknown plugin"):
            topological_sort(plugins)

    def test_cycle_raises(self):
        plugins = {
            "a": _plugin(id="a", depends=("b",)),
            "b": _plugin(id="b", depends=("a",)),
        }
        with pytest.raises(ValueError, match="Cyclic dependency"):
            topological_sort(plugins)

    def test_multi_dependency_decrement(self):
        # c depends on (a, b): processing a decrements c's in-degree 2->1
        # (the != 0 branch), then processing b decrements 1->0 and queues c.
        plugins = {
            "a": _plugin(id="a"),
            "b": _plugin(id="b"),
            "c": _plugin(id="c", depends=("a", "b")),
        }
        order = topological_sort(plugins)
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")


# --- loader.load_plugins ----------------------------------------------------


def _load_with(eps, allowed):
    import os

    key = "MSMODELING_PLUGINS"
    old = os.environ.get(key)
    os.environ[key] = allowed
    try:
        with patch("importlib.metadata.entry_points", return_value=eps):
            return _loader.load_plugins(FastAPI())
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class TestLoadPlugins:
    def test_empty_whitelist_returns_empty(self):
        # No env whitelist -> nothing loaded (regardless of discovery).
        assert _load_with([_ep("a", lambda: _plugin(id="a"))], "") == {}

    def test_whitelist_filters_undiscovered(self):
        # Whitelist set but nothing on disk -> empty.
        assert _load_with([], "a") == {}

    def test_no_whitelisted_match_returns_empty(self):
        # Discovered plugin exists but none match the whitelist -> to_load empty.
        loaded = _load_with([_ep("a", lambda: _plugin(id="a"))], "other")
        assert loaded == {}

    def test_skips_not_in_whitelist(self):
        loaded = _load_with(
            [_ep("a", lambda: _plugin(id="a")), _ep("b", lambda: _plugin(id="b"))],
            "a",
        )
        assert set(loaded) == {"a"}

    def test_skips_incompatible_api_version(self):
        loaded = _load_with(
            [_ep("a", lambda: _plugin(id="a", api_version="999"))],
            "a",
        )
        assert loaded == {}

    def test_factory_raise_is_skipped(self):
        loaded = _load_with([_ep("a", lambda: _plugin(id="a"), raises=True)], "a")
        assert loaded == {}

    def test_factory_returns_non_plugin_is_skipped(self):
        loaded = _load_with([_ep("a", lambda: "not a plugin")], "a")
        assert loaded == {}

    def test_loads_in_dependency_order(self):
        eps = [
            _ep("a", lambda: _plugin(id="a", depends=("b",))),
            _ep("b", lambda: _plugin(id="b")),
        ]
        loaded = _load_with(eps, "a,b")
        assert list(loaded.keys()) == ["b", "a"]  # dependency first

    def test_cyclic_dependency_caught_in_load(self):
        """A cyclic dependency among loaded plugins is caught -> returns empty."""
        loaded = _load_with(
            [_ep("a", lambda: _plugin(id="a", depends=("b",))), _ep("b", lambda: _plugin(id="b", depends=("a",)))],
            "a,b",
        )
        assert loaded == {}


# --- manager.register / apply ----------------------------------------------


def _router(prefix):
    r = APIRouter(prefix=prefix)

    @r.get("/x")
    def _x():
        return {}

    return r


class TestManagerRegister:
    def test_publishes_extension_points(self):
        app = FastAPI()
        p = _plugin(id="p", extension_points={"telemetry_sink": object()})
        PluginManager({"p": p}, app).register()
        assert app.state.plugins_extension_points == p.extension_points

    def test_skips_when_no_extension_points(self):
        app = FastAPI()
        PluginManager({"p": _plugin(id="p")}, app).register()
        # register() always publishes the aggregate; with no extension points
        # declared, the aggregate is an empty dict (not absent).
        assert app.state.plugins_extension_points == {}

    def test_extension_point_override_last_wins(self):
        """Two plugins declare the same extension point — last-wins override."""
        app = FastAPI()
        ep_a = object()
        ep_b = object()
        p_a = _plugin(id="a", extension_points={"sink": ep_a})
        p_b = _plugin(id="b", extension_points={"sink": ep_b})
        PluginManager({"a": p_a, "b": p_b}, app).register()
        # Last plugin wins on conflict.
        assert app.state.plugins_extension_points["sink"] is ep_b


class TestManagerApply:
    def test_mount_path_mounts_bare(self):
        app = FastAPI()
        p = _plugin(id="p", router=_router("/api/p"), mount_path="/api/p")
        PluginManager({"p": p}, app).apply()
        paths = {getattr(r, "path", "") for r in app.routes}
        assert "/api/p/x" in paths
        assert not any("/plugins/p" in getattr(r, "path", "") for r in app.routes)

    def test_no_mount_path_namespaced(self):
        app = FastAPI()
        p = _plugin(id="p", router=_router("/inner"))
        PluginManager({"p": p}, app).apply()
        paths = {getattr(r, "path", "") for r in app.routes}
        assert "/plugins/p/inner/x" in paths

    def test_plugin_without_router_skipped(self):
        app = FastAPI()
        before = len(app.routes)
        PluginManager({"p": _plugin(id="p")}, app).apply()
        assert len(app.routes) == before  # nothing mounted


# --- manager.bootstrap / destroy -------------------------------------------


class TestBootstrap:
    def test_sync_startup_called(self):
        app = FastAPI()
        called = []

        def startup(a):
            called.append("sync")

        p = _plugin(id="p", startup=startup)
        import asyncio

        asyncio.run(PluginManager({"p": p}, app).bootstrap())
        assert called == ["sync"]

    def test_async_startup_awaited(self):
        app = FastAPI()
        called = []

        async def startup(a):
            called.append("async")

        import asyncio

        p = _plugin(id="p", startup=startup)
        asyncio.run(PluginManager({"p": p}, app).bootstrap())
        assert called == ["async"]

    def test_startup_failure_is_swallowed(self):
        app = FastAPI()

        def startup(a):
            raise RuntimeError("nope")

        import asyncio

        p = _plugin(id="p", startup=startup)
        # Must not raise.
        asyncio.run(PluginManager({"p": p}, app).bootstrap())

    def test_plugin_without_startup_skipped(self):
        app = FastAPI()
        import asyncio

        asyncio.run(PluginManager({"p": _plugin(id="p")}, app).bootstrap())  # no error


class TestDestroy:
    def test_sync_shutdown_in_reverse_order(self):
        import asyncio

        app = FastAPI()
        order = []
        asyncio.run(
            PluginManager(
                {
                    "a": _plugin(id="a", shutdown=lambda a: order.append("a")),
                    "b": _plugin(id="b", shutdown=lambda a: order.append("b")),
                },
                app,
            ).destroy()
        )
        assert order == ["b", "a"]  # reverse

    def test_async_shutdown_awaited(self):
        app = FastAPI()
        called = []

        async def shutdown(a):
            called.append("async")

        import asyncio

        asyncio.run(PluginManager({"p": _plugin(id="p", shutdown=shutdown)}, app).destroy())
        assert called == ["async"]

    def test_shutdown_failure_is_swallowed(self):
        app = FastAPI()

        def shutdown(a):
            raise RuntimeError("nope")

        import asyncio

        asyncio.run(PluginManager({"p": _plugin(id="p", shutdown=shutdown)}, app).destroy())

    def test_plugin_without_shutdown_skipped(self):
        app = FastAPI()
        import asyncio

        asyncio.run(PluginManager({"p": _plugin(id="p")}, app).destroy())


class TestAbsolutePathAlias:
    """Regression guard for the MsmdPlugin "twin-class" trap.

    The backend runs from ``web_ui/backend`` (cwd), so its modules are imported
    as TOP-LEVEL packages (``plugins``, ``plugins.contract``). Installed plugin
    packages, however, import the same files via ABSOLUTE path
    (``web_ui.backend.plugins.contract``). Without the sys.modules alias wired
    in ``plugins/__init__.py``, Python loads contract.py twice → two distinct
    MsmdPlugin classes → the loader's isinstance() check rejects EVERY real
    plugin ("entry point did not return MsmdPlugin instance").

    These tests pin the alias so the trap cannot silently regress. The other
    tests in this file build plugin instances from ``_loader.MsmdPlugin``
    (same class the loader checks) and thus never exercise this path.
    """

    def test_absolute_path_aliased_to_top_level(self):
        # ``import plugins`` triggers __init__'s alias registration.
        import plugins  # noqa: F401  (already imported, but explicit)

        for abs_name, top_name in [
            ("web_ui.backend.plugins", "plugins"),
            ("web_ui.backend.plugins.contract", "plugins.contract"),
            ("web_ui.backend.plugins.loader", "plugins.loader"),
            ("web_ui.backend.plugins.manager", "plugins.manager"),
        ]:
            assert abs_name in sys.modules, f"{abs_name} not registered as alias — twin-class bug will resurface"
            assert sys.modules[abs_name] is sys.modules[top_name], (
                f"{abs_name} is a distinct module object from {top_name}"
            )

    def test_msmdplugin_class_identity_shared(self):
        """A plugin importing MsmdPlugin via the absolute path must get the SAME
        class object the loader isinstance-checks — otherwise real plugins are
        rejected at load time.
        """
        import plugins  # noqa: F401
        from plugins.contract import MsmdPlugin as top_cls

        abs_cls = sys.modules["web_ui.backend.plugins.contract"].MsmdPlugin
        assert abs_cls is top_cls, (
            "MsmdPlugin loaded via web_ui.backend.plugins.contract differs from "
            "the one via plugins.contract — installed plugins would be rejected"
        )

    def test_real_plugin_instance_passes_isinstance(self):
        """End-to-end: a plugin instance built from the ABSOLUTE-path class
        (as a real installed plugin would) must pass the loader's isinstance
        check (which uses the top-level class).
        """
        import plugins  # noqa: F401

        # Simulate an installed plugin: it imports MsmdPlugin via the absolute
        # path because it depends on the msmodeling distribution.
        abs_contract = sys.modules["web_ui.backend.plugins.contract"]
        plugin_instance = abs_contract.MsmdPlugin(
            id="real-plugin",
            version="1.0.0",
            api_version="1",
        )
        # The loader isinstance-checks against _loader.MsmdPlugin (top-level).
        assert isinstance(plugin_instance, _loader.MsmdPlugin)
