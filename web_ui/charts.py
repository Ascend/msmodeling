"""Chart generation helpers."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None


# -----------------------------
# Matplotlib / font setup
# -----------------------------
def _pick_plot_font_name() -> str:
    candidates = [
        "Noto Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


_PLOT_FONT_NAME = _pick_plot_font_name()


def setup_matplotlib():
    """Initialize matplotlib font settings."""
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [_PLOT_FONT_NAME, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# Initialize
setup_matplotlib()


# -----------------------------
# Plot helpers
# -----------------------------
def _clean_plot_title(title: str) -> str:
    """Remove UI-only parenthetical suffixes from chart titles."""
    text = str(title or "").strip()
    return re.sub(r"\s*[\uFF08(][^\uFF09)]*[\uFF09)]\s*$", "", text).strip()


def _apply_figure_title(fig, title: str, *, fontsize: int = 17, y: float = 0.975):
    clean_title = _clean_plot_title(title)
    if clean_title:
        fig.suptitle(
            clean_title,
            x=0.5,
            y=y,
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            fontname=_PLOT_FONT_NAME,
            color="#12203d",
        )


def _tight_layout_with_title(fig, *, top: float = 0.91):
    fig.tight_layout(rect=[0.02, 0.02, 0.98, top])


def empty_plot(title: str):
    """Create an empty chart."""
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    ax.text(
        0.5,
        0.5,
        "No data available",
        ha="center",
        va="center",
        fontsize=16,
        fontname=_PLOT_FONT_NAME,
    )
    _apply_figure_title(fig, title, fontsize=17)
    ax.axis("off")
    _tight_layout_with_title(fig)
    return fig


def empty_pie_plot(title: str):
    """Create an empty pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.text(
        0.5,
        0.5,
        "No data available",
        ha="center",
        va="center",
        fontsize=16,
        fontname=_PLOT_FONT_NAME,
    )
    _apply_figure_title(fig, title, fontsize=14)
    ax.axis("off")
    _tight_layout_with_title(fig)
    return fig


def pie_plot(data: dict[str, float], title: str):
    """Create a pie chart with a separate legend to avoid overlap."""
    if not data:
        return empty_pie_plot(title)

    filtered_data = {k: float(v) for k, v in data.items() if v and float(v) > 0}
    if not filtered_data:
        return empty_pie_plot(title)

    labels = list(filtered_data.keys())
    values = list(filtered_data.values())

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, _texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.72,
        textprops={"fontname": _PLOT_FONT_NAME, "fontsize": 8},
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for autotext in autotexts:
        autotext.set_fontname(_PLOT_FONT_NAME)
        autotext.set_fontsize(8)

    _apply_figure_title(fig, title, fontsize=13)
    legend_labels = [f"{label}: {value:.2f} GB" for label, value in zip(labels, values)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
        prop={"family": _PLOT_FONT_NAME, "size": 8},
        labelspacing=0.7,
        handlelength=1.0,
        handletextpad=0.5,
    )
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.02, right=0.74, top=0.84, bottom=0.08)
    return fig


def _apply_font_to_axes(ax):
    """Apply the selected font to the axes."""
    ax.title.set_fontname(_PLOT_FONT_NAME)
    ax.xaxis.label.set_fontname(_PLOT_FONT_NAME)
    ax.yaxis.label.set_fontname(_PLOT_FONT_NAME)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(_PLOT_FONT_NAME)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_title().set_fontname(_PLOT_FONT_NAME)
        for txt in legend.get_texts():
            txt.set_fontname(_PLOT_FONT_NAME)


def _apply_axis_style(ax, title: str, xlabel: str, ylabel: str):
    """Apply the standard axis style."""
    ax.set_title("")
    _apply_figure_title(ax.figure, title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=12, fontname=_PLOT_FONT_NAME)
    ax.set_ylabel(ylabel, fontsize=12, fontname=_PLOT_FONT_NAME)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.25)
    ax.spines["bottom"].set_alpha(0.25)
    _apply_font_to_axes(ax)


def bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    group: str | None = None,
    value_fontsize: int = 8,
):
    """Create a bar chart."""
    if df.empty or x not in df.columns or y not in df.columns:
        return empty_plot(title)

    use_cols = [c for c in [x, y, group] if c and c in df.columns]
    plot_df = df[use_cols].dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return empty_plot(title)
    if len(plot_df) > 60:
        plot_df = plot_df.head(60)

    fig, ax = plt.subplots(figsize=(14.5, 6.8))

    if group and group in plot_df.columns and plot_df[group].nunique() > 1:
        pivot = (
            plot_df.groupby([x, group], dropna=False)[y]
            .mean()
            .reset_index()
            .pivot(index=x, columns=group, values=y)
            .fillna(0)
        )
        categories = [str(v) for v in pivot.index.tolist()]
        group_names = [str(v) for v in pivot.columns.tolist()]
        n_cat = len(categories)
        n_group = max(1, len(group_names))
        total_span = 0.66
        bar_w = min(0.16, total_span / n_group)
        xs = np.arange(n_cat)
        offsets = np.array([(i - (n_group - 1) / 2) * bar_w for i in range(n_group)])
        color_map = plt.get_cmap("tab10")

        for idx, gname in enumerate(group_names):
            vals = pivot[gname].tolist()
            bars = ax.bar(
                xs + offsets[idx],
                vals,
                width=bar_w * 0.86,
                label=gname,
                alpha=0.92,
                color=color_map(idx % 10),
            )
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_fontsize,
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(categories, rotation=28, ha="right")
        ax.legend(title=group, frameon=False)
        ax.set_xlim(-0.55, n_cat - 0.45)
    else:
        plot_df = plot_df.sort_values(by=y, ascending=False, kind="stable")
        categories = plot_df[x].astype(str).tolist()
        values = plot_df[y].tolist()
        n = len(categories)
        xs = np.arange(n)
        if n <= 3:
            bar_w = 0.28
        elif n <= 6:
            bar_w = 0.40
        else:
            bar_w = 0.56
        bars = ax.bar(xs, values, width=bar_w, alpha=0.92, color="#4568e6")
        ax.set_xticks(xs)
        ax.set_xticklabels(categories, rotation=28, ha="right")
        ax.set_xlim(-0.5, n - 0.5)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    _apply_axis_style(ax, title, xlabel or x, ylabel)
    _tight_layout_with_title(fig)
    return fig


def line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    group: str | None = None,
):
    """Create a line chart."""
    if df.empty or x not in df.columns or y not in df.columns:
        return empty_plot(title)

    use_cols = [c for c in [x, y, group] if c and c in df.columns]
    plot_df = df[use_cols].dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return empty_plot(title)
    if len(plot_df) > 300:
        plot_df = plot_df.head(300)

    fig, ax = plt.subplots(figsize=(14.5, 6.8))

    if group and group in plot_df.columns and plot_df[group].nunique() > 1:
        grouped = plot_df.groupby(group)
        color_map = plt.get_cmap("tab10")
        for idx, (name, sub) in enumerate(grouped):
            with suppress(Exception):
                sub = sub.sort_values(by=x, kind="stable")
            ax.plot(
                sub[x],
                sub[y],
                marker="o",
                linewidth=2.4,
                markersize=6,
                label=str(name),
                alpha=0.95,
                color=color_map(idx % 10),
            )
        ax.legend(title=group, frameon=False)
    else:
        with suppress(Exception):
            plot_df = plot_df.sort_values(by=x, kind="stable")
        ax.plot(
            plot_df[x],
            plot_df[y],
            marker="o",
            linewidth=2.4,
            markersize=6,
            alpha=0.95,
            color="#355cde",
        )

    _apply_axis_style(ax, title, xlabel or x, ylabel)
    _tight_layout_with_title(fig)
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    group: str | None = None,
    annotate: str | None = None,
):
    """?????"""
    if df.empty or x not in df.columns or y not in df.columns:
        return empty_plot(title)

    use_cols = [c for c in [x, y, group, annotate] if c and c in df.columns]
    plot_df = df[use_cols].dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return empty_plot(title)
    if len(plot_df) > 300:
        plot_df = plot_df.head(300)

    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    color_map = plt.get_cmap("tab10")

    if group and group in plot_df.columns and plot_df[group].nunique() > 1:
        grouped = plot_df.groupby(group)
        for idx, (name, sub) in enumerate(grouped):
            ax.scatter(
                sub[x],
                sub[y],
                s=70 if "Pareto" in str(name) else 42,
                alpha=0.92,
                label=str(name),
                color=color_map(idx % 10),
                edgecolors="white",
                linewidths=0.6,
            )
            if "Pareto" in str(name):
                try:
                    sub = sub.sort_values(by=x, kind="stable")
                    ax.plot(
                        sub[x],
                        sub[y],
                        color=color_map(idx % 10),
                        linewidth=2.0,
                        alpha=0.9,
                    )
                except Exception:
                    pass  # nosec B110
        ax.legend(title=group, frameon=False)
    else:
        ax.scatter(
            plot_df[x],
            plot_df[y],
            s=46,
            alpha=0.92,
            color="#355cde",
            edgecolors="white",
            linewidths=0.6,
        )

    if annotate and annotate in plot_df.columns and len(plot_df) <= 24:
        for _, row in plot_df.iterrows():
            ax.annotate(
                str(row[annotate]),
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    _apply_axis_style(ax, title, xlabel or x, ylabel)
    _tight_layout_with_title(fig)
    return fig


def top_ops_plot(latest, title: str = "Top 10 Operator Time"):
    """Create the operator-time chart."""
    if latest is None:
        return empty_plot(title)
    rows = latest.tables.get("op_breakdown", [])
    if not rows:
        return empty_plot(title)
    op_df = pd.DataFrame(rows)
    if op_df.empty or "name" not in op_df.columns or "analytic_total_us" not in op_df.columns:
        return empty_plot(title)
    op_df = op_df.sort_values(by="analytic_total_us", ascending=False).head(10).copy()
    op_df["analytic_total_ms"] = op_df["analytic_total_us"] / 1000.0
    return bar_plot(op_df, "name", "analytic_total_ms", title, "Time (ms)", xlabel="Operator")


def optimizer_top_configs_plot(latest, title: str = "Latest Run Top Config Throughput"):
    """Create the optimizer throughput chart."""
    if latest is None:
        return empty_plot(title)
    rows = latest.tables.get("top_configs", [])
    if not rows:
        return empty_plot(title)
    df = pd.DataFrame(rows)
    if df.empty:
        return empty_plot(title)
    df["rank"] = df["rank"].astype(str)
    return bar_plot(
        df,
        "rank",
        "throughput_token_s",
        title,
        "Throughput (token/s)",
        xlabel="Top Rank",
        group="parallel",
    )


def _metric_spec(sim_type: str):
    """Return the metric specification."""
    if sim_type == "text_generate":
        return {
            "column": "tps_per_device",
            "raw_label": "TPS/Device (token/s)",
            "ratio_mode": "higher_better",
        }
    if sim_type == "video_generate":
        return {
            "column": "analytic_total_time_s",
            "raw_label": "Analysis Time (s)",
            "ratio_mode": "lower_better",
        }
    return {
        "column": "best_throughput",
        "raw_label": "Best Throughput (token/s)",
        "ratio_mode": "higher_better",
    }


def _baseline_df(
    sim_type: str,
    rows: list[dict[str, Any]] | None,
    baseline_device: str | None,
):
    """Build the baseline comparison table."""
    spec = _metric_spec(sim_type)
    column = spec["column"]
    mode = spec["ratio_mode"]

    if gr is None:
        raise RuntimeError("gradio is not installed")

    df = _safe_df_from_rows(rows)
    if df.empty or "device" not in df.columns or column not in df.columns:
        return pd.DataFrame(), []

    base_df = df[[c for c in ["device", column] if c in df.columns]].dropna().copy()
    if base_df.empty:
        return pd.DataFrame(), []

    agg = base_df.groupby("device", dropna=False)[column].mean().reset_index()
    devices = agg["device"].astype(str).tolist()
    baseline = baseline_device if baseline_device in devices else (devices[0] if devices else None)
    if baseline is None:
        return pd.DataFrame(), []

    baseline_value = float(agg.loc[agg["device"] == baseline, column].iloc[0])
    if baseline_value == 0:
        return pd.DataFrame(), devices

    if mode == "higher_better":
        agg["performance_ratio"] = agg[column] / baseline_value
    else:
        agg["performance_ratio"] = baseline_value / agg[column]

    agg["baseline_device"] = baseline
    agg = agg.rename(columns={column: spec["raw_label"]})
    agg = agg[["device", "baseline_device", spec["raw_label"], "performance_ratio"]]
    agg = agg.rename(
        columns={
            "device": "Device",
            "baseline_device": "Baseline Device",
            "performance_ratio": "Performance Ratio (x)",
        }
    )
    return agg, devices


def baseline_plot(
    sim_type: str,
    rows: list[dict[str, Any]] | None,
    baseline_device: str | None,
):
    """Create the baseline comparison chart."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    base_df, devices = _baseline_df(sim_type, rows, baseline_device)
    if base_df.empty:
        return (
            empty_plot("Baseline-Normalized Performance"),
            gr.update(choices=devices, value=None),
            pd.DataFrame(),
        )

    fig = line_plot(
        base_df.rename(columns={"Device": "device", "Performance Ratio (x)": "ratio"}),
        "device",
        "ratio",
        "Baseline-Normalized Performance",
        "Relative Performance Ratio (x)",
        xlabel="Device",
        group=None,
    )
    ax = fig.axes[0]
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.4)
    _tight_layout_with_title(fig)
    value = baseline_device if baseline_device in devices else devices[0]
    return fig, gr.update(choices=devices, value=value), base_df


def text_chart_figures(df: pd.DataFrame, latest):
    """Create charts for text generation."""
    fig1 = line_plot(
        df,
        "num_queries",
        "tps_per_device",
        "TPS/Device vs num-queries",
        "Throughput (token/s)",
        xlabel="num-queries",
        group="device",
    )
    group_col = "quantize_linear_action" if "quantize_linear_action" in df.columns else None
    fig2 = bar_plot(
        df,
        "device",
        "analytic_total_time_s",
        "Analysis Time by Device",
        "Analysis Time (s)",
        xlabel="Device",
        group=group_col,
    )
    fig3 = top_ops_plot(latest, "Latest Run Top 10 Operator Time")
    return fig1, fig2, fig3


def video_chart_figures(df: pd.DataFrame, latest):
    """Create charts for video generation."""
    fig1 = bar_plot(
        df,
        "device",
        "analytic_total_time_s",
        "Total Analysis Time by Device",
        "Analysis Time (s)",
        xlabel="Device",
        group="quantize_linear_action",
    )
    if "communication_total_s" in df.columns and not df["communication_total_s"].dropna().empty:
        fig2 = bar_plot(
            df,
            "device",
            "communication_total_s",
            "Communication Time by Device",
            "Communication Time (s)",
            xlabel="Device",
            group=None,
        )
    else:
        fig2 = empty_plot("Communication Time by Device")
    fig3 = top_ops_plot(latest, "Latest Run Top 10 Operator Time")
    return fig1, fig2, fig3


def optimizer_chart_figures(df: pd.DataFrame, latest):
    """Create charts for the optimizer."""
    fig1 = bar_plot(
        df,
        "device",
        "best_throughput",
        "Best Throughput by Device",
        "Throughput (token/s)",
        xlabel="Device",
        group=None,
    )
    fig2 = bar_plot(
        df,
        "device",
        "best_ttft_ms",
        "Best TTFT by Device",
        "TTFT (ms)",
        xlabel="Device",
        group=None,
    )
    fig3 = bar_plot(
        df,
        "device",
        "best_tpot_ms",
        "Best TPOT by Device",
        "TPOT (ms)",
        xlabel="Device",
        group=None,
    )
    return fig1, fig2, fig3


def make_figures(sim_type: str, df: pd.DataFrame, latest):
    """Create charts for the given simulation type."""
    if sim_type == "text_generate":
        return text_chart_figures(df, latest)
    if sim_type == "video_generate":
        return video_chart_figures(df, latest)
    if sim_type == "throughput_optimizer":
        return optimizer_chart_figures(df, latest)
    return empty_plot("No Chart"), empty_plot("No Chart"), empty_plot("No Chart")


# -----------------------------
# Data helpers
# -----------------------------
def _safe_df_from_rows(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Create a DataFrame from row data."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
