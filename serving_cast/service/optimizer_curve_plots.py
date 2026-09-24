# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Optimizer curve plots: terminal ASCII throughput/QPS curves (plotext).

The terminal path relies on the optional ``plotext`` package and a shared canvas.
plotext 5.x exposes that canvas through module-level functions (``scatter``,
``xlim``, ``build``, ...). plotext 6.x removed those names; the same chart is
drawn on ``plotext.figure``. See ``_emit_terminal_optimizer_curve_ascii`` for
concurrency/thread-safety notes.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field

import pandas as pd

from serving_cast.service.optimizer_summary import (
    render_cross_device_comparison,
    render_cross_hardware_disagg_decode,
    render_cross_hardware_disagg_prefill,
    render_cross_hardware_pd_ratio,
    render_hardware_profile_comparison,
)

logger = logging.getLogger(__name__)

_PALETTE = [
    (144, 238, 144),
    (200, 200, 200),
    (135, 206, 235),
    (255, 182, 193),
    (255, 160, 122),
    (221, 160, 221),
]

# Terminal canvas (plotext): shared internal canvas (module-level in 5.x, plotext.figure
# in 6.x); not safe across overlapping calls / threads unless serialized externally
# (see _emit_terminal_optimizer_curve_ascii).
_TERMINAL_PLOT_COLS = 128
_TERMINAL_PLOT_ROWS = 38
_TERMINAL_MARKER = "●"
_AXIS_PADDING_RATIO = 0.08
_BASE_CURVE_COLUMNS = ("concurrency", "token/s")
_PD_TPS_RENAME = {
    "parallel_d": "parallel",
    "concurrency_d": "concurrency",
    "tpot_d": "tpot",
}
_PREFILL_EMIT_KWARGS = {"chart2_x_col": "ttft", "chart2_x_label": "TTFT (ms)"}
_DECODE_EMIT_KWARGS = {"chart2_x_col": "tpot", "chart2_x_label": "TPOT (ms)"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _axis_metric_name(axis_label: str) -> str:
    return axis_label.split(" (", 1)[0].strip() or axis_label


def _parallel_label(parallel: str) -> str:
    s = str(parallel)
    return s if len(s) < 80 else s[:77] + "..."


def _padded_axis_limits(values: list[float]) -> tuple[float, float] | None:
    nums = [float(v) for v in values if math.isfinite(float(v))]
    if not nums:
        return None

    lower = min(nums)
    upper = max(nums)
    span = upper - lower
    padding = span * _AXIS_PADDING_RATIO if span else max(abs(lower) * 0.1, 1.0)
    padded_lower = lower - padding
    return (max(0.0, padded_lower) if lower >= 0 else padded_lower, upper + padding)


def _compact_scatter_legend(buf: str, labels: list[str]) -> str:
    def _visible_len(text: str) -> int:
        return len(_ANSI_RE.sub("", text))

    def _pad_right_border(line: str, width_delta: int) -> str:
        if width_delta <= 0:
            return line
        border_idx = line.rfind("│")
        if border_idx < 0:
            return line + " " * width_delta
        return line[:border_idx] + (" " * width_delta) + line[border_idx:]

    lines = []
    for line in buf.splitlines():
        original = line
        for label in labels:
            line = line.replace(
                f"{_TERMINAL_MARKER}{_TERMINAL_MARKER} {label}",
                f"{_TERMINAL_MARKER}{label}",
            )
            line = line.replace(
                f"{_TERMINAL_MARKER}{_TERMINAL_MARKER}\x1b[0m {label}",
                f"{_TERMINAL_MARKER}\x1b[0m{label}",
            )
        lines.append(_pad_right_border(line, _visible_len(original) - _visible_len(line)))
    return "\n".join(lines)


def _jitter_overlapping_points(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """Slightly offset identical coordinates so every sweep result stays visible."""
    group_counts: dict[tuple[float, float], int] = {}
    group_sizes: dict[tuple[float, float], int] = {}
    for x, y in zip(xs, ys):
        key = (round(float(x), 6), round(float(y), 6))
        group_sizes[key] = group_sizes.get(key, 0) + 1

    if not group_sizes:
        return []

    xspan = max(xs) - min(xs) if xs else 0.0
    yspan = max(ys) - min(ys) if ys else 0.0
    x_step = max(xspan * 0.003, max((abs(x) for x in xs), default=1.0) * 0.001, 1e-3)
    y_step = max(yspan * 0.003, max((abs(y) for y in ys), default=1.0) * 0.001, 1e-3)

    jittered: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        key = (round(float(x), 6), round(float(y), 6))
        idx = group_counts.get(key, 0)
        group_counts[key] = idx + 1
        group_n = group_sizes[key]
        if group_n <= 1:
            jittered.append((x, y))
            continue
        offset = idx - (group_n - 1) / 2.0
        jittered.append((x + offset * x_step, y + offset * y_step))
    return jittered


def _set_plotext_canvas_size(plx: object, width: int, height: int) -> None:
    """Set terminal canvas size across plotext API spellings.

    plotext 5.3+ exposes ``plot_size``; 5.2 and earlier only have ``plotsize``.
    Skip silently if neither exists so optimizer output still prints.
    """
    setter = getattr(plx, "plot_size", None) or getattr(plx, "plotsize", None)
    if not callable(setter):
        logger.warning("plotext has neither plot_size nor plotsize; using default canvas size.")
        return
    setter(width, height)


def _sorted_curve_subset(curve_df: pd.DataFrame, parallel: str, sort_cols: list[str]) -> pd.DataFrame:
    sub = curve_df.loc[curve_df["parallel"].astype(str) == parallel]
    if "batch_size" in sub.columns:
        sub = sub.assign(_batch_sort=pd.to_numeric(sub["batch_size"], errors="coerce"))
        sort_cols = ["_batch_sort" if col == "batch_size" else col for col in sort_cols]
    else:
        sort_cols = [col for col in sort_cols if col != "batch_size"]
    sub = sub.sort_values(sort_cols) if sort_cols else sub
    return sub.drop(columns=["_batch_sort"], errors="ignore")


def _plotext_figure_api(plx) -> bool:
    """True for plotext 6+, which dropped the module-level ``scatter`` function."""
    return hasattr(plx, "figure") and not hasattr(plx, "scatter")


def _terminal_canvas_text(canvas) -> str:
    if isinstance(canvas, str):
        return canvas
    return str(canvas)


def _print_terminal_chart(buf: str, parallels: list[str]) -> None:
    if not buf:
        return
    buf = _compact_scatter_legend(buf, [_parallel_label(p) for p in parallels])
    print("\n" + buf + "\n")


def _emit_plotext6_chart(
    plx,
    plotted: list[tuple[int, str, list[float], list[float]]],
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    *,
    title: str,
    title_prefix: str,
    x_label: str,
    y_axis_label: str,
    parallels: list[str],
) -> None:
    """Draw one scatter chart with the plotext 6 figure API."""
    fig = plx.figure
    terminal = getattr(plx, "terminal", None)
    saved_limit = None
    if terminal is not None and hasattr(terminal, "limit"):
        raw_limit = getattr(terminal, "_limit", None)
        if isinstance(raw_limit, (list, tuple)) and len(raw_limit) >= 2:
            saved_limit = [bool(raw_limit[0]), bool(raw_limit[1])]
        # plot_size is clamped to the terminal unless this limit is off.
        terminal.limit(False, False)
    try:
        fig.clear()
        fig.plot_size(_TERMINAL_PLOT_COLS, _TERMINAL_PLOT_ROWS)
        # "clear" was a 5.x theme name. "colorless" keeps explicit marker RGB colors.
        fig.theme("colorless")
        for idx, parallel, jx, jy in plotted:
            point_marker = plx.marker(
                _TERMINAL_MARKER,
                pixel=plx.pixel(foreground=_PALETTE[idx % len(_PALETTE)]),
            )
            fig.draw(fig.signal(jx, jy, marker=point_marker).label(_parallel_label(parallel)))
        if xlim is not None:
            fig.ruler("x").lim(*xlim)
        if ylim is not None:
            fig.ruler("y").lim(*ylim)
        fig.title(f"{title_prefix}: {title}")
        fig.label(x_label, axis="x")
        fig.label(y_axis_label, axis="y")
        fig.ruler().grid(False)
        try:
            buf = _terminal_canvas_text(fig.build())
        except Exception:
            logger.exception("plotext failed to build chart: %s", title)
            buf = ""
        finally:
            fig.clear.data()
        _print_terminal_chart(buf, parallels)
    finally:
        if terminal is not None and saved_limit is not None:
            terminal.limit(*saved_limit)


def _emit_terminal_optimizer_curve_ascii(
    curve_df: pd.DataFrame,
    title_prefix: str,
    *,
    chart2_x_col: str = "tpot",
    chart2_x_label: str = "TPOT (ms)",
    y_axis_label: str = "Throughput (token/s)",
) -> None:
    """Print throughput (or QPS) point plots as terminal ASCII using plotext."""
    try:
        import plotext as plx
    except ImportError:
        logger.warning("plotext is not installed; skipping terminal curve plots.")
        return

    parallels = sorted(curve_df["parallel"].astype(str).unique())
    if not parallels:
        return
    y_metric = _axis_metric_name(y_axis_label)

    def _draw_chart(
        title: str,
        x_col: str,
        x_label: str,
        sort_cols: list[str],
    ) -> None:
        x_all: list[float] = []
        y_all: list[float] = []
        series: list[tuple[int, str, list[float], list[float]]] = []
        for idx, parallel in enumerate(parallels):
            sub = _sorted_curve_subset(curve_df, parallel, sort_cols)
            if sub.empty:
                continue
            points = pd.DataFrame(
                {
                    "x": pd.to_numeric(sub[x_col], errors="coerce"),
                    "y": pd.to_numeric(sub["token/s"], errors="coerce"),
                }
            ).dropna()
            if points.empty:
                continue
            xv = points["x"].tolist()
            yv = points["y"].tolist()
            x_all.extend(xv)
            y_all.extend(yv)
            series.append((idx, parallel, xv, yv))

        jittered_points = _jitter_overlapping_points(x_all, y_all)
        cursor = 0
        jittered_x_all: list[float] = []
        jittered_y_all: list[float] = []
        plotted: list[tuple[int, str, list[float], list[float]]] = []
        for idx, parallel, xv, yv in series:
            n_points = len(xv)
            jittered = jittered_points[cursor : cursor + n_points]
            cursor += n_points
            jx = [x for x, _ in jittered]
            jy = [y for _, y in jittered]
            jittered_x_all.extend(jx)
            jittered_y_all.extend(jy)
            plotted.append((idx, parallel, jx, jy))
        xlim = _padded_axis_limits(x_all + jittered_x_all)
        ylim = _padded_axis_limits(y_all + jittered_y_all)
        if _plotext_figure_api(plx):
            _emit_plotext6_chart(
                plx,
                plotted,
                xlim,
                ylim,
                title=title,
                title_prefix=title_prefix,
                x_label=x_label,
                y_axis_label=y_axis_label,
                parallels=parallels,
            )
            return

        _set_plotext_canvas_size(plx, _TERMINAL_PLOT_COLS, _TERMINAL_PLOT_ROWS)
        theme = getattr(plx, "theme", None)
        if callable(theme):
            theme("clear")
        for idx, parallel, jx, jy in plotted:
            plx.scatter(
                jx,
                jy,
                label=_parallel_label(parallel),
                color=_PALETTE[idx % len(_PALETTE)],
                marker=_TERMINAL_MARKER,
            )
        if xlim is not None:
            plx.xlim(*xlim)
        if ylim is not None:
            plx.ylim(*ylim)
        plx.title(f"{title_prefix}: {title}")
        plx.xlabel(x_label)
        plx.ylabel(y_axis_label)
        plx.grid(False)
        try:
            buf = plx.build()
        except Exception:
            logger.exception("plotext failed to build chart: %s", title)
            buf = ""
        finally:
            plx.clear_data()
        _print_terminal_chart(buf, parallels)

    try:
        chart_specs = (
            (
                f"{y_metric} vs concurrency",
                "concurrency",
                "Concurrency",
                ["concurrency", "batch_size", chart2_x_col],
            ),
            (
                f"{y_metric} vs {chart2_x_label.split()[0]}",
                chart2_x_col,
                chart2_x_label,
                [chart2_x_col, "batch_size", "concurrency"],
            ),
        )
        for title, x_col, x_label, sort_cols in chart_specs:
            _draw_chart(title, x_col, x_label, sort_cols)
    except Exception:
        logger.exception("Terminal ASCII optimizer curves failed.")


def _memory_filter(work: pd.DataFrame) -> pd.DataFrame:
    for mem_col in ("memory_left_gb", "device_memory_available_gb"):
        if mem_col in work.columns:
            mem = pd.to_numeric(work[mem_col], errors="coerce")
            work = work.loc[mem.isna() | (mem > 0)]
            break
    return work


def _require_columns(df: pd.DataFrame, required: set[str], message: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{message}: {sorted(missing)}")


def _sort_curve_df(work: pd.DataFrame) -> pd.DataFrame:
    if work.empty:
        return work

    sort_keys = ["parallel", "concurrency"]
    if "batch_size" in work.columns:
        work = work.assign(_batch_sort=pd.to_numeric(work["batch_size"], errors="coerce"))
        sort_keys.append("_batch_sort")
    sort_keys.append("token/s")
    return work.sort_values(sort_keys).reset_index(drop=True).drop(columns=["_batch_sort"], errors="ignore")


def _prepare_base_curve_df(
    df: pd.DataFrame,
    *,
    latency_col: str,
    missing_message: str,
) -> pd.DataFrame:
    required = {"parallel", latency_col, *_BASE_CURVE_COLUMNS}
    _require_columns(df, required, missing_message)

    work = df.copy()
    for col in (*_BASE_CURVE_COLUMNS, latency_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna(subset=["parallel", *_BASE_CURVE_COLUMNS, latency_col])


def _prepare_curve_df(
    df: pd.DataFrame,
    _ttft_limit: float | None,
    _tpot_limit: float | None,
) -> pd.DataFrame:
    """Aggregation rows: token/s vs concurrency / TPOT (latency column ``tpot``)."""
    return _prepare_latency_curve_df(
        df,
        latency_col="tpot",
        missing_message="DataFrame missing columns for curve plots",
    )


def _prepare_disagg_prefill_curve_df(
    df: pd.DataFrame,
    _ttft_limit: float | None,
) -> pd.DataFrame:
    """Disagg Prefill sweep: token/s vs concurrency / TTFT."""
    return _prepare_latency_curve_df(
        df,
        latency_col="ttft",
        missing_message="Prefill curve plot missing columns",
    )


def _prepare_latency_curve_df(
    df: pd.DataFrame,
    *,
    latency_col: str,
    missing_message: str,
) -> pd.DataFrame:
    work = _prepare_base_curve_df(
        df,
        latency_col=latency_col,
        missing_message=missing_message,
    )
    work = _memory_filter(work)
    return _sort_curve_df(work)


def plot_concurrency_curves_from_optimizer_summaries(
    results: list,
    *,
    basename_prefix: str,
    ttft_limit: float | None = None,
    tpot_limit: float | None = None,
) -> bool:
    """Merge aggregation summary frames and print terminal curves."""
    dfs = [df for r in results if (df := r.get_summary_df()) is not None and not df.empty]
    if not dfs:
        return False
    merged = pd.concat(dfs, ignore_index=True)
    return plot_concurrency_optimizer_curves(
        merged,
        basename_prefix=basename_prefix,
        ttft_limit=ttft_limit,
        tpot_limit=tpot_limit,
    )


def plot_concurrency_optimizer_curves(
    df: pd.DataFrame,
    *,
    basename_prefix: str,
    ttft_limit: float | None = None,
    tpot_limit: float | None = None,
) -> bool:
    """Aggregation mode terminal curves (token/s vs concurrency / TPOT)."""
    try:
        curve_df = _prepare_curve_df(df, ttft_limit, tpot_limit)
    except ValueError as exc:
        logger.warning("Skipping concurrency curve plots: %s", exc)
        return False

    return _emit_curve_df(
        curve_df,
        title_prefix=str(basename_prefix).strip()[:160] or "optimizer",
        skip_label="concurrency curve plots",
    )


def _emit_curve_df(
    curve_df: pd.DataFrame,
    *,
    title_prefix: str,
    skip_label: str,
    emit_kwargs: dict[str, str] | None = None,
) -> bool:
    if curve_df.empty:
        logger.warning("Skipping %s: no rows after filtering.", skip_label)
        return False

    _emit_terminal_optimizer_curve_ascii(
        curve_df,
        title_prefix=title_prefix,
        **(emit_kwargs or {}),
    )
    return True


def _emit_prepared_curve(
    prepare_curve: Callable[[], pd.DataFrame],
    *,
    title_prefix: str,
    skip_label: str,
    emit_kwargs: dict[str, str],
) -> bool:
    try:
        curve_df = prepare_curve()
    except ValueError as exc:
        logger.warning("Skipping %s: %s", skip_label, exc)
        return False
    return _emit_curve_df(
        curve_df,
        title_prefix=title_prefix,
        skip_label=skip_label,
        emit_kwargs=emit_kwargs,
    )


def plot_disagg_terminal_curves(
    results: list,
    *,
    basename_prefix: str,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> bool:
    """Terminal curves for disaggregation Prefill (TTFT x-axis) and/or Decode (TPOT)."""
    any_ok = False
    base = str(basename_prefix).strip()[:140] or "optimizer"

    for idx, res in enumerate(results):
        df = res.get_summary_df()
        if df is None or df.empty:
            continue
        dc = getattr(res, "data_config", None)
        if dc is None:
            continue

        prefill = dc.ttft_limits is not None and dc.tpot_limits is None
        decode = dc.tpot_limits is not None and dc.ttft_limits is None
        if not (prefill or decode):
            continue

        prepare_curve = (
            (lambda df=df: _prepare_disagg_prefill_curve_df(df, ttft_limit))
            if prefill
            else (lambda df=df: _prepare_curve_df(df, ttft_limit, tpot_limit))
        )
        phase = "prefill" if prefill else "decode"
        any_ok |= _emit_prepared_curve(
            prepare_curve,
            title_prefix=f"{base}_disagg_{phase}_{idx}",
            skip_label=f"{phase} concurrency",
            emit_kwargs=_PREFILL_EMIT_KWARGS if prefill else _DECODE_EMIT_KWARGS,
        )

    return any_ok


def _pd_tps_curve_df(
    pd_df: pd.DataFrame,
) -> pd.DataFrame:
    source_cols = tuple(_PD_TPS_RENAME)
    _require_columns(pd_df, set(source_cols), "PD TPS curve plot missing columns")
    work = pd_df[list(source_cols)].drop_duplicates().rename(columns=_PD_TPS_RENAME)
    work["tpot"] = pd.to_numeric(work["tpot"], errors="coerce")
    work["concurrency"] = pd.to_numeric(work["concurrency"], errors="coerce")
    work = work.loc[work["tpot"] > 0]
    work["token/s"] = pd.to_numeric(work["concurrency"], errors="coerce") / work["tpot"] * 1000
    return work


def plot_pd_ratio_terminal_curves(
    pd_df: pd.DataFrame,
    *,
    basename_prefix: str,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> bool:
    """Terminal curves for PD-ratio grid: TPS vs concurrency and TPOT."""
    if pd_df.empty:
        return False

    return _emit_prepared_curve(
        lambda: _prepare_curve_df(_pd_tps_curve_df(pd_df), ttft_limit, tpot_limit),
        title_prefix=f"{str(basename_prefix).strip()[:120] or 'optimizer'}_pd_decode_tps",
        skip_label="PD TPS",
        emit_kwargs=_DECODE_EMIT_KWARGS,
    )


@dataclass
class MultiDeviceComparisonRows:
    aggregation: list[dict] = field(default_factory=list)
    pd_ratio: list[dict] = field(default_factory=list)
    disagg_prefill: list[dict] = field(default_factory=list)
    disagg_decode: list[dict] = field(default_factory=list)


def _first_non_empty_summary_df(results: list):
    for res in results:
        summary_df = res.get_summary_df()
        if summary_df is not None and not summary_df.empty:
            return summary_df
    return None


def _plot_single_device_optimizer_curves(
    results: list,
    args,
    *,
    basename_prefix: str,
) -> None:
    """Dispatch terminal curve plotting for the active optimizer mode."""
    plot_kwargs = {
        "basename_prefix": basename_prefix,
        "ttft_limit": args.ttft_limits,
        "tpot_limit": args.tpot_limits,
    }

    if args.enable_optimize_prefill_decode_ratio:
        summary_df = _first_non_empty_summary_df(results)
        if summary_df is not None:
            plot_pd_ratio_terminal_curves(summary_df, **plot_kwargs)
        return

    if args.disagg:
        plot_disagg_terminal_curves(results, **plot_kwargs)
        return

    plot_concurrency_curves_from_optimizer_summaries(results, **plot_kwargs)


def _collect_cross_hardware_row(
    rows: MultiDeviceComparisonRows,
    res,
    profile_name: str,
    args,
) -> None:
    if args.disagg:
        collectors = (
            (res.collect_disagg_prefill_row, rows.disagg_prefill),
            (res.collect_disagg_decode_row, rows.disagg_decode),
        )
    elif args.enable_optimize_prefill_decode_ratio:
        collectors = ((res.collect_pd_ratio_comparison_row, rows.pd_ratio),)
    else:
        collectors = ((res.collect_comparison_row, rows.aggregation),)

    for collect, target in collectors:
        row = collect(profile_name)
        if row:
            target.append(row)


def run_multi_device_loop(
    args,
    device_targets: list[str],
    *,
    plot_curves_allowed: bool,
    logger: logging.Logger,
) -> MultiDeviceComparisonRows:
    """Run each device profile and collect cross-hardware rows.

    A manager-hosted coordinator shares frozen workload traces and keyed
    compile shape-mode decisions between existing process-pool workers. Device
    profiles intentionally share the first calibration for the same key.
    The optimizer's ``--jobs`` parallelism is deliberately unchanged.
    """
    from serving_cast.parallel_runner import ParallelRunner
    from serving_cast.service.workload_cache import create_workload_cache_manager

    rows = MultiDeviceComparisonRows()
    multi_hw = len(device_targets) > 1
    cache_manager = None
    workload_cache = None
    if multi_hw:
        cache_manager, workload_cache = create_workload_cache_manager(estimate_jobs=args.jobs)

    try:
        for profile_name in device_targets:
            run_args = copy(args)
            run_args.device = profile_name
            logger.info("Hardware profile: %s", profile_name)
            tasks = ParallelRunner(run_args, workload_cache=workload_cache)

            results = (
                tasks.run_agg()
                if not run_args.enable_optimize_prefill_decode_ratio and not run_args.disagg
                else tasks.run_disagg()
            )

            for res in results:
                res.report_final_result(run_args, silent=False)
                if multi_hw:
                    _collect_cross_hardware_row(rows, res, profile_name, run_args)

            if plot_curves_allowed:
                _plot_single_device_optimizer_curves(
                    results,
                    run_args,
                    basename_prefix=f"{profile_name}_{run_args.model_id}",
                )

        if workload_cache is not None:
            logger.info(workload_cache.summary())
    finally:
        if cache_manager is not None:
            cache_manager.shutdown()

    return rows


def render_cross_hardware_summary(
    args,
    device_targets: list[str],
    rows: MultiDeviceComparisonRows,
    *,
    logger: logging.Logger,
) -> None:
    """Print cross-hardware comparison tables for multi-device runs."""
    if len(device_targets) <= 1:
        return

    hw_profile_txt = render_hardware_profile_comparison(device_targets)
    if hw_profile_txt:
        print(hw_profile_txt)

    if args.disagg:
        for rendered in (
            render_cross_hardware_disagg_prefill(rows.disagg_prefill),
            render_cross_hardware_disagg_decode(rows.disagg_decode),
        ):
            if rendered:
                print(rendered)
        if not rows.disagg_prefill and not rows.disagg_decode:
            logger.warning(
                "No rows available for cross-hardware disaggregation comparison (all runs empty or limits omitted)."
            )
        return

    render_fn, table_rows, warning = (
        (
            render_cross_hardware_pd_ratio,
            rows.pd_ratio,
            "No rows available for cross-hardware PD ratio comparison (all runs empty or filtered out).",
        )
        if args.enable_optimize_prefill_decode_ratio
        else (
            render_cross_device_comparison,
            rows.aggregation,
            "No rows available for cross-hardware comparison (all runs empty).",
        )
    )
    rendered = render_fn(table_rows)
    if rendered:
        print(rendered)
    elif not table_rows:
        logger.warning(warning)
