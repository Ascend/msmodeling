"""Tests for web_ui.web_ui_start module."""

from __future__ import annotations

import os

from unittest.mock import patch


from web_ui.web_ui_start import ensure_localhost_bypass_proxy, main


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


class TestMain:
    """Tests for main function."""

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_default_args(self, mock_ensure, mock_launch) -> None:
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
    @patch("sys.argv", ["web_ui_start.py", "--host", "127.0.0.1", "--port", "8080"])
    def test_main_custom_args(self, mock_ensure, mock_launch) -> None:
        """Test main with custom arguments."""
        main()

        mock_launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=8080,
            share=False,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("sys.argv", ["web_ui_start.py", "--share"])
    def test_main_with_share(self, mock_ensure, mock_launch) -> None:
        """Test main with share flag."""
        main()

        mock_launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=2345,
            share=True,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch.dict(os.environ, {"GRADIO_SERVER_NAME": "192.168.1.1", "GRADIO_SERVER_PORT": "9000"})
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_with_env_vars(self, mock_ensure, mock_launch) -> None:
        """Test main with environment variables."""
        main()

        mock_launch.assert_called_once_with(
            server_name="192.168.1.1",
            server_port=9000,
            share=False,
        )

    @patch("web_ui.web_ui_start.launch_app")
    @patch("web_ui.web_ui_start.ensure_localhost_bypass_proxy")
    @patch("sys.argv", ["web_ui_start.py"])
    def test_main_calls_ensure_bypass(self, mock_ensure, mock_launch) -> None:
        """Test that main calls ensure_localhost_bypass_proxy."""
        main()

        mock_ensure.assert_called_once()
