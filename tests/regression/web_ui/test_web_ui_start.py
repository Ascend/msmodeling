"""Tests for web_ui.web_ui_start module."""

from __future__ import annotations

import os
import socket

from unittest.mock import patch

import web_ui.web_ui_start as web_ui_start
from web_ui.web_ui_start import ensure_localhost_bypass_proxy, get_bind_address, main

# Socket address family constants for consistent mocking
AF_INET = socket.AF_INET
AF_INET6 = socket.AF_INET6


class TestEnsureLocalhostBypassProxy:
    """Tests for ensure_localhost_bypass_proxy function."""

    def test_ensure_localhost_adds_to_empty_env(self) -> None:
        """Test adding bypass items to empty environment."""
        # Clear the environment variables
        for key in ("NO_PROXY", "no_proxy"):
            os.environ.pop(key, None)

        ensure_localhost_bypass_proxy()

        no_proxy = os.environ.get("NO_PROXY", "")
        assert "localhost" in no_proxy
        assert "127.0.0.1" in no_proxy
        assert "::1" in no_proxy

    def test_ensure_localhost_preserves_existing(self) -> None:
        """Test that existing bypass items are preserved."""
        # Clear any existing values first
        os.environ.pop("NO_PROXY", None)
        os.environ.pop("no_proxy", None)

        # On Windows, NO_PROXY and no_proxy are case-insensitive and refer to the same var
        # So only test with one key to avoid confusion
        os.environ["NO_PROXY"] = "example.com,another.org"

        ensure_localhost_bypass_proxy()

        no_proxy = os.environ.get("NO_PROXY", "")
        assert "example.com" in no_proxy
        assert "another.org" in no_proxy
        assert "localhost" in no_proxy
        assert "127.0.0.1" in no_proxy

    def test_ensure_localhost_does_not_duplicate(self) -> None:
        """Test that items are not duplicated."""
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"

        ensure_localhost_bypass_proxy()

        no_proxy = os.environ.get("NO_PROXY", "")
        parts = no_proxy.split(",")
        assert parts.count("localhost") == 1
        assert parts.count("127.0.0.1") == 1

    def test_ensure_localhost_handles_whitespace(self) -> None:
        """Test handling of whitespace in environment variable."""
        os.environ["NO_PROXY"] = "example.com , another.org"

        ensure_localhost_bypass_proxy()

        no_proxy = os.environ.get("NO_PROXY", "")
        assert "example.com" in no_proxy
        assert "localhost" in no_proxy


class TestGetBindAddress:
    """Tests for get_bind_address function."""

    def test_get_bind_address_integration(self) -> None:
        """Integration test: call get_bind_address without mocks."""
        # This test calls the actual function to ensure it works in real environment
        result = get_bind_address()
        # Should return either IPv4 or IPv6 address based on system support
        assert result in ["127.0.0.1", "[::1]"]

    def test_get_bind_address_ipv4_fallback_via_monkeypatch(self, monkeypatch) -> None:
        """Test IPv6 fallback when IPv4 fails using monkeypatch for real coverage."""

        # Create a fake socket class that raises OSError for IPv4, works for IPv6
        class FakeSocket:
            def __init__(self, family, *args):
                self.family = family
                self.closed = False

            def bind(self, address):
                if self.family == 2:  # AF_INET
                    # Simulate IPv4 bind failure
                    raise OSError("IPv4 unavailable")
                # IPv6 bind succeeds

            def close(self):
                self.closed = True

        # Monkeypatch socket.socket to return our fake socket
        import socket as socket_module

        monkeypatch.setattr(socket_module, "socket", FakeSocket)
        # Also patch at the module level where it's used
        monkeypatch.setattr(web_ui_start.socket, "socket", FakeSocket)

        result = get_bind_address()

        assert result == "[::1]"

    def test_get_bind_address_both_fail_via_monkeypatch(self, monkeypatch) -> None:
        """Test default return when both IPv4 and IPv6 fail using monkeypatch."""

        # Create a fake socket class that always raises OSError
        class AlwaysFailSocket:
            def __init__(self, family, *args):
                pass

            def bind(self, address):
                # Always fail
                raise OSError("Network unavailable")

            def close(self):
                pass

        # Monkeypatch socket.socket
        import socket as socket_module

        monkeypatch.setattr(socket_module, "socket", AlwaysFailSocket)
        monkeypatch.setattr(web_ui_start.socket, "socket", AlwaysFailSocket)

        result = get_bind_address()

        assert result == "127.0.0.1"


class TestMain:
    """Tests for main function."""

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("web_ui.web_ui_start.get_bind_address", return_value="127.0.0.1")
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_default_args(self, mock_get_addr, mock_ensure, mock_launch) -> None:
        """Test main with default arguments."""
        main()

        mock_ensure.assert_called_once()
        mock_launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=2345,
            share=False,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("web_ui.web_ui_start.get_bind_address", return_value="127.0.0.1")
    @patch("sys.argv", ["web_ui_start.py", "--port", "8080"])
    def test_main_custom_args(self, mock_get_addr, mock_ensure, mock_launch) -> None:
        """Test main with custom arguments."""
        main()

        mock_launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=8080,
            share=False,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("web_ui.web_ui_start.get_bind_address", return_value="127.0.0.1")
    @patch("sys.argv", ["web_ui_start.py", "--share"])
    def test_main_with_share(self, mock_get_addr, mock_ensure, mock_launch) -> None:
        """Test main with share flag."""
        main()

        mock_launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=2345,
            share=True,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("web_ui.web_ui_start.get_bind_address", return_value="127.0.0.1")
    @patch.dict(os.environ, {"GRADIO_SERVER_PORT": "9000"})
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_with_env_vars(self, mock_get_addr, mock_ensure, mock_launch) -> None:
        """Test main with GRADIO_SERVER_PORT environment variable."""
        main()

        mock_launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=9000,
            share=False,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("web_ui.web_ui_start.get_bind_address", return_value="[::1]")
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_with_ipv6_detection(self, mock_get_addr, mock_ensure, mock_launch) -> None:
        """Test main when get_bind_address detects IPv6 only."""
        main()

        mock_launch.assert_called_once_with(
            server_name="[::1]",
            server_port=2345,
            share=False,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_calls_ensure_bypass(self, mock_ensure, mock_launch) -> None:
        """Test that main calls ensure_localhost_bypass_proxy."""
        main()

        mock_ensure.assert_called_once()
