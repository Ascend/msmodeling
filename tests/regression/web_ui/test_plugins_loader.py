"""Unit tests for ``plugins.loader``.

Covers ``get_allowed_plugins`` (env parsing), ``discover_plugins`` (entry-point
discovery + duplicate detection), ``topological_sort`` (Kahn's algorithm +
cycle/missing-dep errors), and the ``load_plugins`` orchestration pipeline
(allow-list filter, instantiation, api_version check, dependency resolution).
``importlib.metadata.entry_points`` is monkeypatched so no real plugin needs to
be installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.contract import API_VERSION, MsmdPlugin
from plugins.loader import (
    discover_plugins,
    get_allowed_plugins,
    load_plugins,
    topological_sort,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEP:
    """Minimal stand-in for importlib.metadata.EntryPoint.

    ``load()`` returns a factory; calling the factory returns ``plugin`` (or
    raises / returns a custom value via ``factory``/``load_exc``).
    """

    def __init__(self, name: str, plugin: Any = None, *, load_exc: BaseException | None = None, factory: Any = None):
        self.name = name
        self._plugin = plugin
        self._load_exc = load_exc
        self._factory = factory

    def load(self):
        if self._load_exc is not None:
            raise self._load_exc
        if self._factory is not None:
            return self._factory
        return lambda: self._plugin


def _plugin(pid: str = "p", *, api_version: str = API_VERSION, depends: tuple[str, ...] = ()) -> MsmdPlugin:
    return MsmdPlugin(id=pid, version="1.0.0", api_version=api_version, depends=depends)


def _patch_eps(monkeypatch, eps):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: eps)


# ---------------------------------------------------------------------------
# get_allowed_plugins
# ---------------------------------------------------------------------------


class TestGetAllowedPlugins:
    def test_unset_env(self, monkeypatch):
        monkeypatch.delenv("MSMODELING_PLUGINS", raising=False)
        assert get_allowed_plugins() == set()

    def test_empty_env(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "")
        assert get_allowed_plugins() == set()

    def test_whitespace_only(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", " , , ")
        assert get_allowed_plugins() == set()

    def test_single(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")
        assert get_allowed_plugins() == {"demo"}

    def test_multiple(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "a,b,c")
        assert get_allowed_plugins() == {"a", "b", "c"}

    def test_dedup_and_trim(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", " a , a ,b ")
        assert get_allowed_plugins() == {"a", "b"}


# ---------------------------------------------------------------------------
# discover_plugins
# ---------------------------------------------------------------------------


class TestDiscoverPlugins:
    def test_no_entry_points_returns_empty(self, monkeypatch):
        _patch_eps(monkeypatch, [])
        assert discover_plugins() == {}

    def test_single_entry_point(self, monkeypatch):
        _patch_eps(monkeypatch, [_FakeEP(name="demo")])
        assert set(discover_plugins().keys()) == {"demo"}

    def test_multiple_entry_points(self, monkeypatch):
        _patch_eps(monkeypatch, [_FakeEP(name="a"), _FakeEP(name="b"), _FakeEP(name="c")])
        assert set(discover_plugins().keys()) == {"a", "b", "c"}

    def test_duplicate_id_raises(self, monkeypatch):
        _patch_eps(monkeypatch, [_FakeEP(name="dup"), _FakeEP(name="dup")])
        with pytest.raises(RuntimeError, match="Duplicate plugin ID"):
            discover_plugins()


# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_empty(self):
        assert topological_sort({}) == []

    def test_no_dependencies(self):
        plugins = {"a": _plugin("a"), "b": _plugin("b")}
        result = topological_sort(plugins)
        assert set(result) == {"a", "b"}

    def test_linear_chain_dependency_before_dependent(self):
        plugins = {"a": _plugin("a"), "b": _plugin("b", depends=("a",))}
        assert topological_sort(plugins) == ["a", "b"]

    def test_diamond_order(self):
        plugins = {
            "a": _plugin("a"),
            "b": _plugin("b", depends=("a",)),
            "c": _plugin("c", depends=("a",)),
            "d": _plugin("d", depends=("b", "c")),
        }
        order = topological_sort(plugins)
        # a before b,c ; b,c before d
        assert order.index("a") < order.index("b") < order.index("d")
        assert order.index("a") < order.index("c") < order.index("d")

    def test_missing_dependency_raises(self):
        plugins = {"a": _plugin("a", depends=("ghost",))}
        with pytest.raises(ValueError, match="unknown plugin"):
            topological_sort(plugins)

    def test_cycle_raises(self):
        plugins = {
            "a": _plugin("a", depends=("b",)),
            "b": _plugin("b", depends=("a",)),
        }
        with pytest.raises(ValueError, match="Cyclic dependency"):
            topological_sort(plugins)


# ---------------------------------------------------------------------------
# load_plugins (app is unused by the current implementation)
# ---------------------------------------------------------------------------


class TestLoadPlugins:
    def test_empty_allowlist_returns_empty(self, monkeypatch):
        monkeypatch.delenv("MSMODELING_PLUGINS", raising=False)
        _patch_eps(monkeypatch, [_FakeEP(name="demo", plugin=_plugin("demo"))])
        assert load_plugins(None) == {}

    def test_nothing_discovered_returns_empty(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")
        _patch_eps(monkeypatch, [])
        assert load_plugins(None) == {}

    def test_none_whitelisted_returns_empty(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")
        _patch_eps(monkeypatch, [_FakeEP(name="other", plugin=_plugin("other"))])
        assert load_plugins(None) == {}

    def test_factory_returns_non_plugin_skipped(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")
        _patch_eps(monkeypatch, [_FakeEP(name="demo", plugin="not-a-plugin")])
        assert load_plugins(None) == {}

    def test_load_raising_is_skipped(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")
        _patch_eps(monkeypatch, [_FakeEP(name="demo", load_exc=RuntimeError("boom"))])
        assert load_plugins(None) == {}

    def test_factory_call_raising_is_skipped(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")

        def _exploding_factory():
            raise RuntimeError("factory blew up")

        _patch_eps(monkeypatch, [_FakeEP(name="demo", factory=_exploding_factory)])
        assert load_plugins(None) == {}

    def test_api_version_mismatch_skipped(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "demo")
        _patch_eps(monkeypatch, [_FakeEP(name="demo", plugin=_plugin("demo", api_version="999"))])
        assert load_plugins(None) == {}

    def test_dependency_cycle_returns_empty(self, monkeypatch):
        # Two mutually-dependent valid plugins -> topological_sort raises ->
        # load_plugins swallows and returns {}.
        monkeypatch.setenv("MSMODELING_PLUGINS", "a,b")
        _patch_eps(
            monkeypatch,
            [
                _FakeEP(name="a", plugin=_plugin("a", depends=("b",))),
                _FakeEP(name="b", plugin=_plugin("b", depends=("a",))),
            ],
        )
        assert load_plugins(None) == {}

    def test_happy_path_returns_dependency_ordered(self, monkeypatch):
        monkeypatch.setenv("MSMODELING_PLUGINS", "a,b")
        _patch_eps(
            monkeypatch,
            [
                _FakeEP(name="b", plugin=_plugin("b", depends=("a",))),
                _FakeEP(name="a", plugin=_plugin("a")),
            ],
        )
        result = load_plugins(None)
        assert list(result.keys()) == ["a", "b"]
        assert all(isinstance(p, MsmdPlugin) for p in result.values())
