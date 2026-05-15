"""Reusable UI components."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from tensor_cast.device import DeviceProfile


# -----------------------------
# Device helpers
# -----------------------------
def get_vendor_device_map() -> dict[str, list[str]]:
    """Return the vendor-to-device mapping."""
    vendor_map: dict[str, list[str]] = {}
    for profile in DeviceProfile.all_device_profiles.values():
        vendor = getattr(profile, "vendor", None)
        name = getattr(profile, "name", None)
        if not vendor or not name:
            continue
        if str(vendor).upper() == "TEST_VENDOR":
            continue
        vendor_map.setdefault(str(vendor), []).append(str(name))
    for vendor, names in vendor_map.items():
        vendor_map[vendor] = sorted(set(names))
    return {vendor: vendor_map[vendor] for vendor in sorted(vendor_map.keys())}


# -----------------------------
# Progress components
# -----------------------------
def progress_html(completed: int, total: int, latest: str = "", status: str = "") -> str:
    """Create the progress-bar HTML."""
    total = max(1, int(total or 1))
    completed = max(0, min(int(completed or 0), total))
    pct = completed / total * 100.0
    latest_text = latest or "Waiting for the first task"
    status_text = status or "Preparing"
    return f"""
    <div class="progress-shell">
      <div class="progress-title">
        <strong>Run Progress</strong>
        <span>{completed}/{total} | {pct:.1f}%</span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%"></div></div>
      <div class="progress-caption">Current Status: {status_text}<br/>Latest Task: {latest_text}</div>
    </div>
    """


# -----------------------------
# Section components
# -----------------------------
def result_plot(label: str | None = None, **kwargs):
    """Create a plot output without Gradio's overlay label."""
    if gr is None:
        raise RuntimeError("gradio is not installed")
    return gr.Plot(label=label, show_label=False, **kwargs)


def result_dataframe(
    label: str,
    *,
    max_height: int = 420,
    column_widths: list[str | int] | None = None,
    elem_classes: list[str] | str | None = None,
):
    """Create a searchable, copyable, fullscreen-capable result table."""
    if gr is None:
        raise RuntimeError("gradio is not installed")
    return gr.Dataframe(
        label=label,
        interactive=False,
        wrap=True,
        show_search="filter",
        show_copy_button=True,
        show_fullscreen_button=True,
        max_height=max_height,
        column_widths=column_widths,
        elem_classes=elem_classes,
    )


def render_section_card(title: str, subtitle: str = ""):
    """Render a styled section card."""
    if gr is None:
        raise RuntimeError("gradio is not installed")
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    gr.HTML(f'<div class="section-card"><h2>{title}</h2>{subtitle_html}</div>')


# -----------------------------
# Result section
# -----------------------------
def result_section(sim_type: str):
    """Create the result area with charts, tables, and export controls."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    progress = gr.HTML(progress_html(0, 1, "", "Waiting to run"))
    summary_md = gr.Markdown("### Summary\nWaiting to run.")
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Simulation Charts")
        chart1 = result_plot(label="Main Chart")
        chart2 = result_plot(label="Secondary Chart")
        chart3 = result_plot(label="Comparison Chart")
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Baseline-Normalized Performance")
        baseline_device = gr.Dropdown(choices=[], label="Baseline Device", allow_custom_value=False)
        baseline_plot = result_plot(label="Baseline Curve")
        baseline_df = result_dataframe("Baseline Table")
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Summary Table")
        with gr.Row():
            export_btn = gr.Button("Export Current Table to Excel")
        export_file = gr.File(label="Excel Export File", interactive=False)
        results_df = result_dataframe("Results Table")
    display_rows_state = gr.State([])
    full_rows_state = gr.State([])

    baseline_device.change(
        lambda base, rows: _refresh_baseline_view(sim_type, rows, base),
        inputs=[baseline_device, full_rows_state],
        outputs=[baseline_plot, baseline_df],
    )

    return (
        progress,
        summary_md,
        chart1,
        chart2,
        chart3,
        baseline_device,
        baseline_plot,
        baseline_df,
        results_df,
        display_rows_state,
        full_rows_state,
        export_btn,
        export_file,
    )


def _refresh_baseline_view(sim_type: str, rows: list[dict[str, Any]] | None, baseline_device: str | None):
    """Refresh the baseline view."""
    from .charts import baseline_plot

    fig, df = baseline_plot(sim_type, rows, baseline_device)[:2]
    # The dropdown update for the third return value is handled by the caller.
    base_df, devices = baseline_plot(sim_type, rows, baseline_device)[2:]
    return fig, df


def wire_export(button, state, file_output, prefix: str):
    """Bind the export button action."""
    button.click(
        lambda rows: export_current_rows(rows, prefix),
        inputs=[state],
        outputs=[file_output],
    )


# -----------------------------
# Data helpers
# -----------------------------
EXPORT_DIR = Path(".msmodeling_ui/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_df_from_rows(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Create a DataFrame from row data."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def export_current_rows(rows: list[dict[str, Any]] | None, prefix: str) -> str | None:
    """Export the current rows to Excel."""
    df = _safe_df_from_rows(rows)
    if df.empty:
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{prefix}_{uuid.uuid4().hex[:8]}.xlsx"
    file_path = EXPORT_DIR / file_name
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return str(file_path)


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into a list of records."""
    if df is None or df.empty:
        return []
    clean_df = df.copy()
    clean_df = clean_df.where(pd.notnull(clean_df), None)
    return clean_df.to_dict(orient="records")


# -----------------------------
# Text Generate result workspace
# -----------------------------
def text_generate_result_section():
    """Create the Text Generate result workspace with operator-time analysis."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    progress = gr.HTML(progress_html(0, 1, "", "Waiting to run"))
    with gr.Group(elem_classes=["recommendation-card"]):
        summary_md = gr.Markdown("### Recommendations\nWaiting to run.")

    # Core metrics: num-queries versus runtime
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Core Metrics")
        tg_tps_chart = result_plot(label="TPS/Device Comparison (Hidden)", visible=False)
        tg_time_chart = result_plot(label="num-queries vs Runtime")
        # Show the TPOT metric when MTP > 0.
        tg_tpot_metric = gr.Markdown("", visible=False)
    # Memory usage analysis with device filtering
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Memory Usage Analysis")
        with gr.Row():
            tg_memory_device = gr.Dropdown(choices=[], label="Device", allow_custom_value=False)
            tg_memory_case = gr.Dropdown(choices=[], label="Case (queries/tp-size)", allow_custom_value=False)
        with gr.Row(elem_classes=["memory-analysis-row"]):
            tg_memory_pie = result_plot(
                label="Memory Usage Breakdown",
                elem_classes=["memory-pie-plot"],
                scale=1,
            )
            tg_memory_table = result_dataframe(
                "Memory Details",
                max_height=360,
                column_widths=[150, 120],
                elem_classes=["memory-table"],
            )
    # Bandwidth and bottleneck details with device filtering
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Bandwidth and Bottleneck Analysis")
        with gr.Row():
            tg_bandwidth_device = gr.Dropdown(choices=[], label="Device", allow_custom_value=False)
            tg_bandwidth_case = gr.Dropdown(choices=[], label="Case (queries/tp-size)", allow_custom_value=False)
        tg_bandwidth_table = result_dataframe("Bandwidth and Bottleneck Details", max_height=300)

    # Operator-time analysis table
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Operator Time Analysis")
        with gr.Row():
            tg_op_device = gr.Dropdown(
                choices=[],
                label="Device for Operator Details",
                allow_custom_value=False,
            )
            tg_op_case = gr.Dropdown(choices=[], label="Case (queries/tp-size)", allow_custom_value=False)
            tg_op_top_n = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Top N Operators")
            tg_op_sort = gr.Radio(
                ["Total Time (ms)", "Avg Time (ms)", "Calls", "Operator"],
                value="Total Time (ms)",
                label="Sort By",
            )
        tg_op_columns = gr.CheckboxGroup(
            choices=[
                "Operator",
                "Category",
                "Total Time (ms)",
                "Avg Time (ms)",
                "Calls",
                "Device",
            ],
            value=[
                "Operator",
                "Category",
                "Total Time (ms)",
                "Avg Time (ms)",
                "Calls",
                "Device",
            ],
            label="Visible Columns",
        )
        tg_op_table = result_dataframe(
            "Operator Time Details",
            max_height=520,
            column_widths=[260, 140, 120, 120, 100, 170],
        )

    # Operator category statistics with device filtering
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Operator Category Statistics")
        with gr.Row():
            tg_op_category_device = gr.Dropdown(choices=[], label="Device", allow_custom_value=False)
            tg_op_category_case = gr.Dropdown(choices=[], label="Case (queries/tp-size)", allow_custom_value=False)
        tg_op_category_chart = result_plot(label="By Category")
        tg_op_category_table = result_dataframe("Category Statistics", max_height=300)

    # Cross-device operator comparison
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Cross-Device Operator Comparison")
        with gr.Row():
            tg_op_compare_mode = gr.Radio(
                choices=["Total Time", "Avg Time"],
                value="Total Time",
                label="Comparison Metric",
            )
        tg_op_compare_table = result_dataframe("Device Operator Comparison", max_height=460)

    # Summary table and export
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Run Summary")
        with gr.Row():
            tg_export_btn = gr.Button("Export Results to Excel")
        tg_export_file = gr.File(label="Excel Export File", interactive=False)
        tg_results_df = result_dataframe("Run Summary Table", max_height=360)

    tg_display_state = gr.State([])
    tg_full_state = gr.State([])
    tg_op_breakdown_state = gr.State([])  # Store operator-detail rows
    tg_current_model_state = gr.State("")  # Store the current model id
    tg_mtp_acceptance_state = gr.State([])  # Store the MTP acceptance-rate data

    return (
        progress,
        summary_md,
        tg_tps_chart,
        tg_time_chart,
        tg_tpot_metric,
        tg_memory_device,
        tg_memory_case,
        tg_memory_pie,
        tg_memory_table,
        tg_bandwidth_device,
        tg_bandwidth_case,
        tg_bandwidth_table,
        tg_op_device,
        tg_op_case,
        tg_op_top_n,
        tg_op_sort,
        tg_op_columns,
        tg_op_table,
        tg_op_category_device,
        tg_op_category_case,
        tg_op_category_chart,
        tg_op_category_table,
        tg_op_compare_mode,
        tg_op_compare_table,
        tg_export_btn,
        tg_export_file,
        tg_results_df,
        tg_display_state,
        tg_full_state,
        tg_op_breakdown_state,
        tg_current_model_state,
        tg_mtp_acceptance_state,
    )


# -----------------------------
# Video Generate result workspace
# -----------------------------
def video_generate_result_section():
    """Create the Video Generate result workspace with operator-time analysis."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    progress = gr.HTML(progress_html(0, 1, "", "Waiting to run"))
    with gr.Group(elem_classes=["recommendation-card"]):
        summary_md = gr.Markdown("### Recommendations\nWaiting to run.")

    # Core metrics
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Core Metrics")
        with gr.Row():
            vg_time_chart = result_plot(label="Total Analysis Time")
            vg_comm_chart = result_plot(label="Communication Time")

    # Operator-time analysis table
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Operator Time Analysis")
        with gr.Row():
            vg_op_device = gr.Dropdown(
                choices=[],
                label="Device for Operator Details",
                allow_custom_value=False,
            )
            vg_op_top_n = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Top N Operators")
            vg_op_sort = gr.Radio(
                ["Total Time (ms)", "Avg Time (ms)", "Calls", "Operator"],
                value="Total Time (ms)",
                label="Sort By",
            )
        vg_op_columns = gr.CheckboxGroup(
            choices=[
                "Operator",
                "Category",
                "Total Time (ms)",
                "Avg Time (ms)",
                "Calls",
                "Device",
            ],
            value=[
                "Operator",
                "Category",
                "Total Time (ms)",
                "Avg Time (ms)",
                "Calls",
                "Device",
            ],
            label="Visible Columns",
        )
        vg_op_table = result_dataframe(
            "Operator Time Details",
            max_height=520,
            column_widths=[260, 140, 120, 120, 100, 170],
        )

    # Operator category statistics
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Operator Category Statistics")
        vg_op_category_chart = result_plot(label="By Category")
        vg_op_category_table = result_dataframe("Category Statistics", max_height=300)

    # Cross-device operator comparison
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Cross-Device Operator Comparison")
        vg_op_compare_table = result_dataframe("Device Operator Comparison", max_height=460)

    # Summary table and export
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Run Summary")
        with gr.Row():
            vg_export_btn = gr.Button("Export Results to Excel")
        vg_export_file = gr.File(label="Excel Export File", interactive=False)
        vg_results_df = result_dataframe("Run Summary Table", max_height=360)

    vg_display_state = gr.State([])
    vg_full_state = gr.State([])
    vg_op_breakdown_state = gr.State([])  # Store operator-detail rows

    return (
        progress,
        summary_md,
        vg_time_chart,
        vg_comm_chart,
        vg_op_device,
        vg_op_top_n,
        vg_op_sort,
        vg_op_columns,
        vg_op_table,
        vg_op_category_chart,
        vg_op_category_table,
        vg_op_compare_table,
        vg_export_btn,
        vg_export_file,
        vg_results_df,
        vg_display_state,
        vg_full_state,
        vg_op_breakdown_state,
    )


# -----------------------------
# Throughput Optimizer result workspace
# -----------------------------


def optimizer_result_section():
    """Create the Throughput Optimizer result workspace."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    progress = gr.HTML(progress_html(0, 1, "", "Waiting to run"))
    with gr.Group(elem_classes=["recommendation-card"]):
        summary_md = gr.Markdown("\n".join(["### Recommendations", "Waiting to run."]))

    with gr.Tabs():
        with gr.Tab("Best Result by Device"), gr.Group(elem_classes=["section-card"]):
            gr.Markdown(
                "Review the best result for each device under the current "
                "limits before comparing a fixed configuration."
            )
            with gr.Row():
                op_throughput_chart = result_plot(label="Best Throughput by Device")
                op_ttft_chart = result_plot(label="Best TTFT by Device")
            with gr.Row():
                op_tpot_chart = result_plot(label="Best TPOT by Device")
                op_batch_chart = result_plot(label="Secondary Comparison Metric")
            op_results_df = result_dataframe("Best Results by Device", max_height=420)

        with gr.Tab("Fixed-Config Comparison"), gr.Group(elem_classes=["section-card"]):
            gr.Markdown("Compare devices under the same configuration for a fair side-by-side evaluation.")
            with gr.Row():
                op_fixed_config = gr.Dropdown(choices=[], label="Fixed Configuration", allow_custom_value=False)
                op_fixed_metric = gr.Radio(["Throughput", "TTFT", "TPOT"], value="Throughput", label="Metric")
            op_fixed_md = gr.Markdown("\n".join(["### Fixed-Config Comparison", "Waiting to run."]))
            op_fixed_chart = result_plot(label="Fixed-Config Metric Comparison")
            op_fixed_df = result_dataframe("Fixed-Config Table", max_height=420)

        with gr.Tab("PD Ratio"), gr.Group(elem_classes=["section-card"]):
            gr.Markdown("Use this tab to inspect the Prefill/Decode balance when deployment mode is PD Ratio.")
            op_pd_chart = result_plot(label="Prefill / Decode QPS Comparison")
            op_pd_df = result_dataframe("PD Ratio Key Metrics", max_height=320)

        with gr.Tab("Single-Device Details"), gr.Group(elem_classes=["section-card"]):
            gr.Markdown(
                "Inspect candidate configurations, the Pareto frontier, and detailed search results for one device."
            )
            with gr.Row():
                op_detail_device = gr.Dropdown(
                    choices=[],
                    label="Device for Search Details",
                    allow_custom_value=False,
                )
            op_detail_md = gr.Markdown("\n".join(["### Single-Device Search Details", "Waiting to run."]))
            op_detail_pareto_chart = result_plot(label="Single-Device Pareto Frontier")
            op_detail_df = result_dataframe("Single-Device Top Results", max_height=420)
            op_detail_output = gr.Textbox(
                label="Single-Device Full Output",
                interactive=False,
                lines=18,
                max_lines=36,
            )

    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Candidates and Export")
        with gr.Row():
            op_export_btn = gr.Button("Export Results to Excel")
        gr.Markdown(
            "Repeated runs for the same case reuse cached logs automatically; no manual history loading is required."
        )
        op_export_file = gr.File(label="Excel Export File", interactive=False)

    op_display_state = gr.State([])
    op_full_state = gr.State([])
    op_candidate_state = gr.State([])

    return (
        progress,
        summary_md,
        op_throughput_chart,
        op_ttft_chart,
        op_tpot_chart,
        op_batch_chart,
        op_pd_chart,
        op_pd_df,
        op_fixed_config,
        op_fixed_metric,
        op_fixed_md,
        op_fixed_chart,
        op_fixed_df,
        op_detail_device,
        op_detail_md,
        op_detail_pareto_chart,
        op_detail_df,
        op_detail_output,
        op_export_btn,
        op_export_file,
        op_results_df,
        op_display_state,
        op_full_state,
        op_candidate_state,
    )
