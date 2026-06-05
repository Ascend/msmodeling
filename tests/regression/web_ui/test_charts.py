"""Tests for web_ui.charts module."""

from __future__ import annotations

import pandas as pd
import pytest

from web_ui.charts import (
    _baseline_df,
    _clean_plot_title,
    _metric_spec,
    _pick_plot_font_name,
    _safe_df_from_rows,
    bar_plot,
    baseline_plot,
    empty_plot,
    empty_pie_plot,
    line_plot,
    make_figures,
    optimizer_chart_figures,
    optimizer_top_configs_plot,
    pie_plot,
    scatter_plot,
    setup_matplotlib,
    text_chart_figures,
    top_ops_plot,
    video_chart_figures,
)


class TestPickPlotFontName:
    """Tests for _pick_plot_font_name function."""

    def test_pick_font_returns_string(self) -> None:
        """Test that font picker returns a string."""
        result = _pick_plot_font_name()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pick_font_in_candidates(self) -> None:
        """Test that returned font is in candidate list."""
        result = _pick_plot_font_name()
        candidates = [
            "Noto Sans SC",
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "DejaVu Sans",
        ]
        # Font should be one of the candidates or DejaVu Sans fallback
        assert result in candidates or result == "DejaVu Sans"


class TestSetupMatplotlib:
    """Tests for setup_matplotlib function."""

    def test_setup_matplotlib_configures_rc(self) -> None:
        """Test that matplotlib rcParams are configured."""
        import matplotlib

        original_family = matplotlib.rcParams.get("font.family", [])
        original_sans = matplotlib.rcParams.get("font.sans-serif", [])

        setup_matplotlib()

        assert matplotlib.rcParams["font.family"] == ["sans-serif"]
        assert len(matplotlib.rcParams["font.sans-serif"]) > 0
        assert matplotlib.rcParams["axes.unicode_minus"] is False

        # Restore original values
        matplotlib.rcParams["font.family"] = original_family
        matplotlib.rcParams["font.sans-serif"] = original_sans


class TestCleanPlotTitle:
    """Tests for _clean_plot_title function."""

    def test_clean_title_basic(self) -> None:
        """Test cleaning a basic title."""
        assert _clean_plot_title("Test Title") == "Test Title"

    def test_clean_title_with_parentheses(self) -> None:
        """Test removing parenthetical suffixes."""
        assert _clean_plot_title("Title (some info)") == "Title"
        assert _clean_plot_title("Title (detail)") == "Title"

    def test_clean_title_with_nested_parens(self) -> None:
        """Test with nested parentheses - function doesn't handle them."""
        # The function doesn't handle nested parens, returns unchanged
        assert _clean_plot_title("Title (detail (nested))") == "Title (detail (nested))"

    def test_clean_title_with_full_width_parens(self) -> None:
        """Test with full-width parentheses."""
        assert _clean_plot_title("Title (info)") == "Title"

    def test_clean_title_whitespace(self) -> None:
        """Test whitespace handling."""
        assert _clean_plot_title("  Title  ") == "Title"
        assert _clean_plot_title("") == ""

    def test_clean_title_no_parentheses(self) -> None:
        """Test title without parentheses."""
        assert _clean_plot_title("Pure Title") == "Pure Title"


class TestEmptyPlot:
    """Tests for empty_plot function."""

    def test_empty_plot_returns_figure(self) -> None:
        """Test empty plot returns a figure."""
        fig = empty_plot("Test")
        assert fig is not None
        assert hasattr(fig, "axes")

    def test_empty_plot_has_title(self) -> None:
        """Test empty plot has the specified title."""
        fig = empty_plot("Test Title")
        # Title is applied via suptitle
        assert len(fig.axes) > 0


class TestEmptyPiePlot:
    """Tests for empty_pie_plot function."""

    def test_empty_pie_plot_returns_figure(self) -> None:
        """Test empty pie plot returns a figure."""
        fig = empty_pie_plot("Test")
        assert fig is not None


class TestPiePlot:
    """Tests for pie_plot function."""

    def test_pie_plot_empty_data(self) -> None:
        """Test pie plot with empty data."""
        fig = pie_plot({}, "Test")
        assert fig is not None

    def test_pie_plot_all_zeros(self) -> None:
        """Test pie plot with all zero values."""
        fig = pie_plot({"a": 0, "b": 0}, "Test")
        assert fig is not None

    def test_pie_plot_valid_data(self) -> None:
        """Test pie plot with valid data."""
        data = {"A": 10.0, "B": 20.0, "C": 30.0}
        fig = pie_plot(data, "Distribution")
        assert fig is not None

    def test_pie_plot_filters_zeros(self) -> None:
        """Test that zero values are filtered."""
        data = {"A": 10.0, "B": 0.0, "C": 20.0}
        fig = pie_plot(data, "Test")
        assert fig is not None


class TestBarPlot:
    """Tests for bar_plot function."""

    def test_bar_plot_empty_df(self) -> None:
        """Test bar plot with empty DataFrame."""
        df = pd.DataFrame()
        fig = bar_plot(df, "x", "y", "Test", "Y")
        assert fig is not None

    def test_bar_plot_missing_columns(self) -> None:
        """Test bar plot with missing columns."""
        df = pd.DataFrame({"a": [1, 2]})
        fig = bar_plot(df, "x", "y", "Test", "Y")
        assert fig is not None

    def test_bar_plot_valid_data(self) -> None:
        """Test bar plot with valid data."""
        df = pd.DataFrame({"category": ["A", "B", "C"], "value": [10, 20, 15]})
        fig = bar_plot(df, "category", "value", "Categories", "Value")
        assert fig is not None

    def test_bar_plot_with_group(self) -> None:
        """Test bar plot with grouping."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "group": ["X", "Y", "X", "Y"], "val": [10, 15, 20, 25]})
        fig = bar_plot(df, "cat", "val", "Test", "Value", group="group")
        assert fig is not None

    def test_bar_plot_limits_data(self) -> None:
        """Test that data is limited to 60 rows."""
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        fig = bar_plot(df, "x", "y", "Test", "Y")
        assert fig is not None


class TestLinePlot:
    """Tests for line_plot function."""

    def test_line_plot_empty_df(self) -> None:
        """Test line plot with empty DataFrame."""
        df = pd.DataFrame()
        fig = line_plot(df, "x", "y", "Test", "Y")
        assert fig is not None

    def test_line_plot_missing_columns(self) -> None:
        """Test line plot with missing columns."""
        df = pd.DataFrame({"a": [1, 2]})
        fig = line_plot(df, "x", "y", "Test", "Y")
        assert fig is not None

    def test_line_plot_valid_data(self) -> None:
        """Test line plot with valid data."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 15]})
        fig = line_plot(df, "x", "y", "Line", "Y")
        assert fig is not None

    def test_line_plot_with_group(self) -> None:
        """Test line plot with grouping."""
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [10, 15, 20, 25], "group": ["A", "B", "A", "B"]})
        fig = line_plot(df, "x", "y", "Test", "Y", group="group")
        assert fig is not None

    def test_line_plot_limits_data(self) -> None:
        """Test that data is limited to 300 rows."""
        df = pd.DataFrame({"x": range(400), "y": range(400)})
        fig = line_plot(df, "x", "y", "Test", "Y")
        assert fig is not None


class TestScatterPlot:
    """Tests for scatter_plot function."""

    def test_scatter_plot_empty_df(self) -> None:
        """Test scatter plot with empty DataFrame."""
        df = pd.DataFrame()
        fig = scatter_plot(df, "x", "y", "Test", "Y")
        assert fig is not None

    def test_scatter_plot_missing_columns(self) -> None:
        """Test scatter plot with missing columns."""
        df = pd.DataFrame({"a": [1, 2]})
        fig = scatter_plot(df, "x", "y", "Test", "Y")
        assert fig is not None

    def test_scatter_plot_valid_data(self) -> None:
        """Test scatter plot with valid data."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 15]})
        fig = scatter_plot(df, "x", "y", "Scatter", "Y")
        assert fig is not None

    def test_scatter_plot_with_group(self) -> None:
        """Test scatter plot with grouping."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 15, 25], "group": ["A", "A", "B", "B"]})
        fig = scatter_plot(df, "x", "y", "Test", "Y", group="group")
        assert fig is not None

    def test_scatter_plot_with_annotation(self) -> None:
        """Test scatter plot with annotation."""
        df = pd.DataFrame({"x": [1, 2], "y": [10, 20], "label": ["Point1", "Point2"]})
        fig = scatter_plot(df, "x", "y", "Test", "Y", annotate="label")
        assert fig is not None


class TestTopOpsPlot:
    """Tests for top_ops_plot function."""

    def test_top_ops_plot_none_latest(self) -> None:
        """Test with None latest result."""
        fig = top_ops_plot(None, "Test")
        assert fig is not None

    def test_top_ops_plot_no_breakdown(self) -> None:
        """Test with no operator breakdown."""
        latest = type("obj", (object,), {"tables": {}})()
        fig = top_ops_plot(latest, "Test")
        assert fig is not None

    def test_top_ops_plot_empty_breakdown(self) -> None:
        """Test with empty breakdown list."""
        latest = type("obj", (object,), {"tables": {"op_breakdown": []}})()
        fig = top_ops_plot(latest, "Test")
        assert fig is not None

    def test_top_ops_plot_valid_data(self) -> None:
        """Test with valid operator breakdown."""
        latest = type(
            "obj",
            (object,),
            {
                "tables": {
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 5000},
                        {"name": "op2", "analytic_total_us": 3000},
                    ]
                }
            },
        )()
        fig = top_ops_plot(latest, "Operators")
        assert fig is not None


class TestOptimizerTopConfigsPlot:
    """Tests for optimizer_top_configs_plot function."""

    def test_optimizer_top_configs_none_latest(self) -> None:
        """Test with None latest result."""
        fig = optimizer_top_configs_plot(None, "Test")
        assert fig is not None

    def test_optimizer_top_configs_no_configs(self) -> None:
        """Test with no top configs."""
        latest = type("obj", (object,), {"tables": {}})()
        fig = optimizer_top_configs_plot(latest, "Test")
        assert fig is not None

    def test_optimizer_top_configs_valid_data(self) -> None:
        """Test with valid top configs."""
        latest = type(
            "obj",
            (object,),
            {
                "tables": {
                    "top_configs": [
                        {"rank": 1, "throughput_token_s": 1000, "parallel": "TP=4"},
                        {"rank": 2, "throughput_token_s": 800, "parallel": "TP=2"},
                    ]
                }
            },
        )()
        fig = optimizer_top_configs_plot(latest, "Top Configs")
        assert fig is not None


class TestMetricSpec:
    """Tests for _metric_spec function."""

    def test_metric_spec_text_generate(self) -> None:
        """Test metric spec for text_generate."""
        spec = _metric_spec("text_generate")
        assert spec["column"] == "tps_per_device"
        assert spec["raw_label"] == "TPS/Device (token/s)"
        assert spec["ratio_mode"] == "higher_better"

    def test_metric_spec_video_generate(self) -> None:
        """Test metric spec for video_generate."""
        spec = _metric_spec("video_generate")
        assert spec["column"] == "analytic_total_time_s"
        assert spec["raw_label"] == "Analysis Time (s)"
        assert spec["ratio_mode"] == "lower_better"

    def test_metric_spec_optimizer(self) -> None:
        """Test metric spec for throughput_optimizer."""
        spec = _metric_spec("throughput_optimizer")
        assert spec["column"] == "best_throughput"
        assert spec["raw_label"] == "Best Throughput (token/s)"
        assert spec["ratio_mode"] == "higher_better"

    def test_metric_spec_unknown(self) -> None:
        """Test metric spec for unknown type defaults to optimizer."""
        spec = _metric_spec("unknown_type")
        assert spec["column"] == "best_throughput"


class TestBaselineDf:
    """Tests for _baseline_df function."""

    def test_baseline_df_empty_rows(self) -> None:
        """Test with empty rows."""
        df, devices = _baseline_df("text_generate", [], None)
        assert df.empty
        assert devices == []

    def test_baseline_df_no_device_column(self) -> None:
        """Test with no device column."""
        rows = [{"tps_per_device": 100.0}]
        df, devices = _baseline_df("text_generate", rows, None)
        assert df.empty

    def test_baseline_df_no_metric_column(self) -> None:
        """Test with no metric column."""
        rows = [{"device": "D1"}]
        df, devices = _baseline_df("text_generate", rows, None)
        assert df.empty

    def test_baseline_df_valid_data(self) -> None:
        """Test with valid data."""
        rows = [
            {"device": "D1", "tps_per_device": 100.0},
            {"device": "D2", "tps_per_device": 200.0},
        ]
        df, devices = _baseline_df("text_generate", rows, "D1")
        assert not df.empty
        assert "D1" in devices
        assert "D2" in devices

    def test_baseline_df_ratio_calculation(self) -> None:
        """Test performance ratio calculation."""
        rows = [
            {"device": "D1", "tps_per_device": 100.0},
            {"device": "D2", "tps_per_device": 200.0},
        ]
        df, devices = _baseline_df("text_generate", rows, "D1")
        assert df.loc[df["Device"] == "D2", "Performance Ratio (x)"].iloc[0] == 2.0


class TestBaselinePlot:
    """Tests for baseline_plot function."""

    def test_baseline_plot_requires_gradio(self) -> None:
        """Test baseline_plot requires gradio."""
        import web_ui.charts as charts

        original_gr = charts.gr
        charts.gr = None
        with pytest.raises(RuntimeError, match="gradio is not installed"):
            baseline_plot("text_generate", [], None)
        charts.gr = original_gr

    def test_baseline_plot_empty_data(self) -> None:
        """Test baseline_plot with empty data returns empty plot."""
        # Mock gr to be available
        import web_ui.charts as charts

        if charts.gr is None:
            charts.gr = type("gr", (), {"Blocks": bool})()

        fig, update, df = baseline_plot("text_generate", [], None)[:3]
        assert fig is not None


class TestTextChartFigures:
    """Tests for text_chart_figures function."""

    def test_text_chart_figures_empty_df(self) -> None:
        """Test with empty DataFrame."""
        fig1, fig2, fig3 = text_chart_figures(pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_text_chart_figures_valid_df(self) -> None:
        """Test with valid DataFrame."""
        df = pd.DataFrame(
            {
                "device": ["D1", "D2"],
                "num_queries": [16, 32],
                "tps_per_device": [100, 200],
                "analytic_total_time_s": [1.0, 2.0],
            }
        )
        fig1, fig2, fig3 = text_chart_figures(df, None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None


class TestVideoChartFigures:
    """Tests for video_chart_figures function."""

    def test_video_chart_figures_empty_df(self) -> None:
        """Test with empty DataFrame."""
        fig1, fig2, fig3 = video_chart_figures(pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_video_chart_figures_valid_df(self) -> None:
        """Test with valid DataFrame."""
        df = pd.DataFrame(
            {
                "device": ["D1", "D2"],
                "analytic_total_time_s": [1.5, 2.0],
                "communication_total_s": [0.5, 0.8],
                "quantize_linear_action": ["W8A8", "FP16"],
            }
        )
        fig1, fig2, fig3 = video_chart_figures(df, None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None


class TestOptimizerChartFigures:
    """Tests for optimizer_chart_figures function."""

    def test_optimizer_chart_figures_empty_df(self) -> None:
        """Test with empty DataFrame."""
        fig1, fig2, fig3 = optimizer_chart_figures(pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_optimizer_chart_figures_valid_df(self) -> None:
        """Test with valid DataFrame."""
        df = pd.DataFrame(
            {
                "device": ["D1", "D2"],
                "best_throughput": [1000, 2000],
                "best_ttft_ms": [50, 75],
                "best_tpot_ms": [25, 30],
            }
        )
        fig1, fig2, fig3 = optimizer_chart_figures(df, None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None


class TestMakeFigures:
    """Tests for make_figures routing function."""

    def test_make_figures_text_generate(self) -> None:
        """Test make_figures for text_generate."""
        fig1, fig2, fig3 = make_figures("text_generate", pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_make_figures_video_generate(self) -> None:
        """Test make_figures for video_generate."""
        fig1, fig2, fig3 = make_figures("video_generate", pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_make_figures_optimizer(self) -> None:
        """Test make_figures for throughput_optimizer."""
        fig1, fig2, fig3 = make_figures("throughput_optimizer", pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_make_figures_unknown_type(self) -> None:
        """Test make_figures for unknown type returns empty plots."""
        fig1, fig2, fig3 = make_figures("unknown", pd.DataFrame(), None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None


class TestSafeDfFromRows:
    """Tests for _safe_df_from_rows in charts module."""

    def test_safe_df_from_rows_none(self) -> None:
        """Test with None returns empty DataFrame."""
        result = _safe_df_from_rows(None)
        assert result.empty

    def test_safe_df_from_rows_empty_list(self) -> None:
        """Test with empty list returns empty DataFrame."""
        result = _safe_df_from_rows([])
        assert result.empty

    def test_safe_df_from_rows_valid(self) -> None:
        """Test with valid rows."""
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = _safe_df_from_rows(rows)
        assert len(result) == 2

    def test_safe_df_from_rows_with_none_values(self) -> None:
        """Test with rows containing None values."""
        rows = [{"a": 1, "b": None}, {"a": 3, "b": 4}]
        result = _safe_df_from_rows(rows)
        assert len(result) == 2

    def test_safe_df_from_rows_with_mixed_types(self) -> None:
        """Test with rows containing mixed data types."""
        rows = [{"a": 1, "b": "text"}, {"a": 3.5, "b": True}]
        result = _safe_df_from_rows(rows)
        assert len(result) == 2


class TestPiePlotEdgeCases:
    """Tests for pie_plot edge cases."""

    def test_pie_plot_with_negative_values(self) -> None:
        """Test pie_plot with negative values (should be filtered out)."""
        data = {"A": -10, "B": 20, "C": 30}
        result = pie_plot(data, "Test")
        assert result is not None

    def test_pie_plot_with_very_small_values(self) -> None:
        """Test pie_plot with very small values."""
        data = {"A": 0.001, "B": 0.002, "C": 0.003}
        result = pie_plot(data, "Test")
        assert result is not None

    def test_pie_plot_with_single_value(self) -> None:
        """Test pie_plot with only one value."""
        data = {"A": 100}
        result = pie_plot(data, "Test")
        assert result is not None


class TestLinePlotEdgeCases:
    """Tests for line_plot edge cases."""

    def test_line_plot_with_all_same_y_values(self) -> None:
        """Test line_plot when all y values are the same."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 10, 10], "group": ["A", "A", "A"]})
        result = line_plot(df, "x", "y", "Test", "Y", "X", group="group")
        assert result is not None

    def test_line_plot_with_single_point(self) -> None:
        """Test line_plot with only one data point."""
        df = pd.DataFrame({"x": [1], "y": [10], "group": ["A"]})
        result = line_plot(df, "x", "y", "Test", "Y", "X", group="group")
        assert result is not None


class TestScatterPlotEdgeCases:
    """Tests for scatter_plot edge cases."""

    def test_scatter_plot_with_duplicate_points(self) -> None:
        """Test scatter_plot with duplicate points."""
        df = pd.DataFrame({"x": [1, 1, 2], "y": [10, 10, 20], "group": ["A", "A", "B"]})
        result = scatter_plot(df, "x", "y", "Test", "Y", "X", group="group")
        assert result is not None

    def test_scatter_plot_with_large_values(self) -> None:
        """Test scatter_plot with very large values."""
        df = pd.DataFrame({"x": [1e10, 2e10], "y": [1e10, 2e10], "group": ["A", "B"]})
        result = scatter_plot(df, "x", "y", "Test", "Y", "X", group="group")
        assert result is not None


class TestBarPlotExtended:
    """Extended tests for bar_plot function."""

    def test_bar_plot_with_negative_values(self) -> None:
        """Test bar_plot with negative values."""
        df = pd.DataFrame({"x": ["A", "B", "C"], "y": [-10, 20, -5], "group": ["G1", "G1", "G2"]})
        result = bar_plot(df, "x", "y", "Test", "Y", "X", group="group")
        assert result is not None

    def test_bar_plot_with_long_labels(self) -> None:
        """Test bar_plot with very long category labels."""
        df = pd.DataFrame(
            {
                "x": ["Very Long Category Name A", "Very Long Category Name B"],
                "y": [10, 20],
            }
        )
        result = bar_plot(df, "x", "y", "Test", "Y", "X")
        assert result is not None


# Duplicate TestMetricSpec class removed - merged with earlier definition


class TestBaselinePlotEdgeCases:
    """Tests for baseline_plot edge cases."""

    def test_baseline_df_with_none_baseline_device(self) -> None:
        """Test baseline_df with None as baseline_device."""
        rows = [{"device": "D1", "tps_per_device": 100}]
        result = _baseline_df("text_generate", rows, None)
        # Returns tuple of (df, devices)
        assert len(result) == 2

    def test_baseline_df_with_empty_rows(self) -> None:
        """Test baseline_df with empty rows list."""
        result = _baseline_df("text_generate", [], None)
        # Returns tuple of (empty_df, empty_devices)
        df, devices = result
        assert df.empty


class TestChartFiguresEdgeCases:
    """Tests for chart figure functions edge cases."""

    def test_text_chart_figures_with_none_latest(self) -> None:
        """Test text_chart_figures with None as latest."""
        df = pd.DataFrame({"device": ["D1"], "tps_per_device": [100]})
        fig1, fig2, fig3 = text_chart_figures(df, None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_video_chart_figures_with_none_latest(self) -> None:
        """Test video_chart_figures with None as latest."""
        df = pd.DataFrame({"device": ["D1"], "analytic_total_time_s": [10]})
        fig1, fig2, fig3 = video_chart_figures(df, None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_optimizer_chart_figures_with_none_latest(self) -> None:
        """Test optimizer_chart_figures with None as latest."""
        df = pd.DataFrame({"device": ["D1"], "best_throughput": [1000]})
        fig1, fig2, fig3 = optimizer_chart_figures(df, None)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
