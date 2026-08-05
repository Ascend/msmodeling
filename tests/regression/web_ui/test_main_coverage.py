"""Coverage for web_ui/backend/main.py: bind-address, create_app plugin branches, lifespan plugin hooks."""

from __future__ import annotations

from unittest.mock import patch

import main
import plugins.loader as _loader
from fastapi import FastAPI
from fastapi.testclient import TestClient

_PluginCls = _loader.MsmdPlugin


def _fake_plugin(*, startup=None, shutdown=None):
    return _PluginCls(
        id="fake",
        version="1.0.0",
        api_version="1",
        router=None,
        mount_path=None,
        migrations_path=None,
        startup=startup,
        shutdown=shutdown,
        depends=(),
        extension_points={},
    )


class TestGetBindAddress:
    def test_returns_localhost(self):
        host = main.get_bind_address()
        assert host in ("127.0.0.1", "::1")

    def test_ipv6_fallback(self):
        """When IPv4 bind fails, falls through to IPv6 (or 127.0.0.1 if both fail)."""
        import socket as _sock

        class FakeSocket:
            def __init__(self, *args):
                self._family = args[0] if args else _sock.AF_INET

            def bind(self, addr):
                if self._family == _sock.AF_INET:
                    raise OSError("no IPv4")

            def close(self):
                pass

        with patch("socket.socket", side_effect=FakeSocket):
            host = main.get_bind_address()
        assert host in ("::1", "127.0.0.1")

    def test_no_working_stack(self):
        """When both IPv4 and IPv6 fail, falls back to 127.0.0.1."""

        class FailingSocket:
            def __init__(self, *args):
                pass

            def bind(self, addr):
                raise OSError("no stack")

            def close(self):
                pass

        with patch("socket.socket", side_effect=FailingSocket):
            assert main.get_bind_address() == "127.0.0.1"


class TestCreateAppPluginBranches:
    def test_loads_plugins_into_state(self):
        p = _fake_plugin()
        with patch.object(main, "load_plugins", return_value={"fake": p}):
            app = main.create_app()
        assert app.state.plugins == {"fake": p}

    def test_plugin_load_failure_falls_back_to_empty(self):
        with patch.object(main, "load_plugins", side_effect=RuntimeError("boom")):
            app = main.create_app()
        assert app.state.plugins == {}

    def test_lifespan_no_new_schemas(self, monkeypatch, tmp_path):
        """Covers the `if registered:` False branch (no new schemas to register)."""
        monkeypatch.setenv("MSMODELING_UI_DIR", str(tmp_path))
        with patch.object(main, "_upsert_schema_snapshots", return_value=[]), TestClient(main.create_app()):
            pass


class TestLifespanPluginHooks:
    def test_bootstrap_and_destroy_run(self, monkeypatch, tmp_path):
        """The lifespan calls plugin startup on boot + shutdown on teardown."""
        monkeypatch.setenv("MSMODELING_UI_DIR", str(tmp_path))
        events: list[str] = []

        def startup(app):
            events.append("startup")

        def shutdown(app):
            events.append("shutdown")

        p = _fake_plugin(startup=startup, shutdown=shutdown)
        with patch.object(main, "load_plugins", return_value={"fake": p}):
            app = main.create_app()
        # Drive the lifespan: entering the context runs startup hooks;
        # leaving runs shutdown hooks.
        with TestClient(app):
            assert "startup" in events
        assert "shutdown" in events


class TestMain:
    """Tests for main() (the CLI entry extracted from ``if __name__ == "__main__"``)."""

    def _call_main(self, monkeypatch):
        """Invoke main.main() with a mocked uvicorn module; return the mock."""
        from unittest.mock import MagicMock
        import sys

        mock_uvicorn = MagicMock()
        monkeypatch.setitem(sys.modules, "uvicorn", mock_uvicorn)
        # Re-import uvicorn inside main() picks up the mocked module.
        main.main()
        return mock_uvicorn

    def test_main_calls_uvicorn_with_default_port(self, monkeypatch):
        """main() binds to localhost and runs uvicorn with port=8000 by default."""
        monkeypatch.delenv("MSMODELING_PORT", raising=False)
        mock_uvicorn = self._call_main(monkeypatch)
        mock_uvicorn.run.assert_called_once()
        call_args, call_kwargs = mock_uvicorn.run.call_args
        # app is created by create_app() and passed positionally.
        assert isinstance(call_args[0], FastAPI)
        assert call_kwargs["port"] == 8000
        assert call_kwargs["log_level"] == "info"
        assert "host" in call_kwargs

    def test_main_respects_port_env(self, monkeypatch):
        """MSMODELING_PORT env var overrides the default port."""
        monkeypatch.setenv("MSMODELING_PORT", "9999")
        mock_uvicorn = self._call_main(monkeypatch)
        _, call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs["port"] == 9999

    def test_main_invalid_port_raises(self, monkeypatch):
        """Non-numeric MSMODELING_PORT propagates ValueError (fail-fast)."""
        monkeypatch.setenv("MSMODELING_PORT", "not-a-port")
        from unittest.mock import MagicMock
        import sys

        monkeypatch.setitem(sys.modules, "uvicorn", MagicMock())
        import pytest

        with pytest.raises(ValueError):
            main.main()
