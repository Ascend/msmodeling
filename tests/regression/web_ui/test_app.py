"""Tests for web_ui.app module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import web_ui.app as app_module


class TestAppModuleConstants:
    """Tests for app module constants."""

    def test_quant_linear_options(self) -> None:
        """Test quantization linear options."""
        assert app_module.QUANT_LINEAR_OPTIONS
        assert "DISABLED" in app_module.QUANT_LINEAR_OPTIONS
        assert "W8A16_STATIC" in app_module.QUANT_LINEAR_OPTIONS
        assert len(app_module.QUANT_LINEAR_OPTIONS) == 9

    def test_quant_attention_options(self) -> None:
        """Test quantization attention options."""
        assert app_module.QUANT_ATTENTION_OPTIONS
        assert "DISABLED" in app_module.QUANT_ATTENTION_OPTIONS
        assert "INT8" in app_module.QUANT_ATTENTION_OPTIONS
        assert len(app_module.QUANT_ATTENTION_OPTIONS) == 3

    def test_app_title(self) -> None:
        """Test app title."""
        assert app_module.APP_TITLE == "Modeling Compass"

    def test_app_icon(self) -> None:
        """Test app icon."""
        assert app_module.APP_ICON
        assert "data:image/svg+xml" in app_module.APP_ICON

    def test_app_head(self) -> None:
        """Test app head."""
        assert app_module.APP_HEAD
        assert "<meta" in app_module.APP_HEAD
        assert "<style>" in app_module.APP_HEAD


class TestBuildApp:
    """Tests for build_app function."""

    def test_build_app_requires_gradio(self) -> None:
        """Test that build_app raises RuntimeError without gradio."""
        # Mock gr as None
        original_gr = app_module.gr
        app_module.gr = None
        with pytest.raises(RuntimeError, match="gradio is not installed"):
            app_module.build_app()
        app_module.gr = original_gr


class TestLaunchApp:
    """Tests for launch_app function."""

    @patch("web_ui.app.gr")
    @patch("web_ui.app.build_app")
    def test_launch_app_default_params(self, mock_build, mock_gr) -> None:
        """Test launch_app with default parameters."""
        mock_demo = Mock()
        mock_build.return_value = mock_demo
        mock_demo.launch = Mock(return_value=Mock())

        app_module.launch_app()

        mock_build.assert_called_once()
        mock_demo.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=2345,
            share=False,
            inbrowser=False,
            show_error=True,
        )

    @patch("web_ui.app.gr")
    @patch("web_ui.app.build_app")
    def test_launch_app_custom_params(self, mock_build, mock_gr) -> None:
        """Test launch_app with custom parameters."""
        mock_demo = Mock()
        mock_build.return_value = mock_demo
        mock_demo.launch = Mock(return_value=Mock())

        app_module.launch_app(server_name="127.0.0.1", server_port=8080, share=True)

        mock_demo.launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=8080,
            share=True,
            inbrowser=False,
            show_error=True,
        )

    @patch("web_ui.app.gr")
    @patch("web_ui.app.build_app")
    def test_launch_app_returns_launch_result(self, mock_build, mock_gr) -> None:
        """Test that launch_app returns the launch result."""
        mock_demo = Mock()
        mock_build.return_value = mock_demo
        expected_result = Mock(server_name="test", port=8080)
        mock_demo.launch = Mock(return_value=expected_result)

        result = app_module.launch_app()

        assert result == expected_result


class TestHeroHtml:
    """Tests for HERO_HTML constant if accessible."""

    def test_hero_html_exists(self) -> None:
        """Test that HERO_HTML constant exists."""
        assert hasattr(app_module, "HERO_HTML")
        hero_html = getattr(app_module, "HERO_HTML")
        assert isinstance(hero_html, str)
        assert len(hero_html) > 0
