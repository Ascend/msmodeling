"""Tests for web_ui.__init__ module."""

from __future__ import annotations

from unittest.mock import Mock, patch


import web_ui


class TestInitModule:
    """Tests for web_ui.__init__ module."""

    def test_launch_app_exists(self) -> None:
        """Test that launch_app function exists."""
        assert hasattr(web_ui, "launch_app")
        assert callable(web_ui.launch_app)

    @patch("web_ui.app.launch_app")
    def test_launch_app_forwards_args(self, mock_launch) -> None:
        """Test that launch_app forwards arguments to app.launch_app."""
        mock_launch.return_value = "test_result"
        result = web_ui.launch_app(server_name="127.0.0.1", server_port=8080, share=True)

        mock_launch.assert_called_once_with(server_name="127.0.0.1", server_port=8080, share=True)
        assert result == "test_result"

    @patch("web_ui.app.launch_app")
    def test_launch_app_with_kwargs(self, mock_launch) -> None:
        """Test launch_app with keyword arguments."""
        mock_launch.return_value = Mock()
        web_ui.launch_app(server_name="0.0.0.0", server_port=2345)

        mock_launch.assert_called_once()

    @patch("web_ui.app.launch_app")
    def test_launch_app_without_args(self, mock_launch) -> None:
        """Test launch_app without arguments."""
        mock_launch.return_value = Mock()
        web_ui.launch_app()

        mock_launch.assert_called_once()

    def test_module_all(self) -> None:
        """Test __all__ export list."""
        assert hasattr(web_ui, "__all__")
        assert "launch_app" in web_ui.__all__
        assert len(web_ui.__all__) == 1
