"""Web UI entry module."""

from __future__ import annotations

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from .callbacks import (
    preview_optimizer,
    preview_text_generate,
    preview_video_generate,
    refresh_optimizer_detail_v2,
    refresh_optimizer_fixed_compare,
    run_optimizer_v2,
    run_text_generate_v2,
    run_video_generate_v2,
    update_bandwidth_analysis_by_device,
    update_category_stats_by_device,
    update_compare_table_by_mode,
    update_memory_analysis_by_device,
    update_op_table_from_breakdown,
    update_video_op_table_from_breakdown,
)
from .charts import setup_matplotlib
from .components import (
    get_vendor_device_map,
    optimizer_result_section,
    render_section_card,
    text_generate_result_section,
    video_generate_result_section,
    wire_export,
)
from .styles import APP_CSS, COLORS

# Initialize matplotlib
setup_matplotlib()

# Quantization options
QUANT_LINEAR_OPTIONS = [
    "DISABLED",
    "W8A16_STATIC",
    "W8A8_STATIC",
    "W4A8_STATIC",
    "W8A16_DYNAMIC",
    "W8A8_DYNAMIC",
    "W4A8_DYNAMIC",
    "FP8",
    "MXFP4",
]
QUANT_ATTENTION_OPTIONS = ["DISABLED", "INT8", "FP8"]
APP_TITLE = "Modeling Compass"
APP_ICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
    "%3Cdefs%3E%3ClinearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E"
    "%3Cstop offset='0%25' stop-color='%23ffffff'/%3E"
    "%3Cstop offset='100%25' stop-color='%23dbe7ff'/%3E"
    "%3C/linearGradient%3E%3C/defs%3E"
    "%3Crect x='6' y='6' width='52' height='52' rx='16' fill='url(%23g)' stroke='%2389a7ff'/%3E"
    "%3Ccircle cx='32' cy='32' r='18' fill='none' stroke='%2321409a' stroke-opacity='.28'/%3E"
    "%3Cpath d='M32 13l4 11-4-2-4 2z' fill='%2321409a'/%3E"
    "%3Cg%3E%3Cpath d='M32 16l6 16-6-3-6 3z' fill='%23d94a2d'/%3E"
    "%3Cpath d='M32 48l6-16-6 3-6-3z' fill='%2321409a'/%3E%3C/g%3E"
    "%3Ccircle cx='32' cy='32' r='4' fill='%2312203d'/%3E%3C/svg%3E"
)
APP_HEAD = """
<meta charset="utf-8" />
<link rel="icon" type="image/svg+xml" href="__APP_ICON__" />
<style>
  .table-wrap[role="grid"] thead th:not(.row-number) {
    position: relative;
  }
  .mc-col-resizer {
    position: absolute;
    top: 0;
    right: -4px;
    width: 10px;
    height: 100%;
    cursor: col-resize;
    z-index: 12;
    touch-action: none;
  }
  .mc-col-resizer::after {
    content: "";
    position: absolute;
    top: 18%;
    bottom: 18%;
    left: 50%;
    width: 2px;
    transform: translateX(-50%);
    border-radius: 999px;
    background: rgba(33, 64, 154, 0.18);
    transition: background 0.18s ease;
  }
  .table-wrap[role="grid"] thead th:not(.row-number):hover .mc-col-resizer::after,
  .table-wrap[role="grid"].mc-column-resizing .mc-col-resizer::after {
    background: rgba(33, 64, 154, 0.48);
  }
  .table-wrap[role="grid"].mc-column-resizing,
  .table-wrap[role="grid"].mc-column-resizing * {
    cursor: col-resize !important;
    user-select: none !important;
  }
</style>
<script>
(() => {
  const MIN_WIDTH = 80;

  function bindResizableColumns(root = document) {
    const wraps = root.querySelectorAll('.table-wrap[role="grid"]');
    wraps.forEach((wrap) => {
      const headers = Array.from(wrap.querySelectorAll('thead th:not(.row-number)'));
      headers.forEach((th, index) => {
        th.dataset.mcColIndex = String(index);
        if (th.querySelector(':scope > .mc-col-resizer')) {
          return;
        }
        const handle = document.createElement('div');
        handle.className = 'mc-col-resizer';
        handle.setAttribute('aria-hidden', 'true');
        handle.addEventListener('mousedown', (event) => startResize(event, wrap, th, index));
        th.appendChild(handle);
      });
    });
  }

  function startResize(event, wrap, th, columnIndex) {
    event.preventDefault();
    event.stopPropagation();

    const startX = event.clientX;
    const startWidth = th.getBoundingClientRect().width;
    wrap.classList.add('mc-column-resizing');

    const onMove = (moveEvent) => {
      const nextWidth = Math.max(MIN_WIDTH, Math.round(startWidth + moveEvent.clientX - startX));
      wrap.style.setProperty(`--cell-width-${columnIndex}`, `${nextWidth}px`);
    };

    const onUp = () => {
      wrap.classList.remove('mc-column-resizing');
      document.removeEventListener('mousemove', onMove, true);
      document.removeEventListener('mouseup', onUp, true);
    };

    document.addEventListener('mousemove', onMove, true);
    document.addEventListener('mouseup', onUp, true);
  }

  function initResizableColumns() {
    bindResizableColumns(document);
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'childList' && (mutation.addedNodes.length || mutation.removedNodes.length)) {
          bindResizableColumns(document);
          break;
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initResizableColumns, { once: true });
  } else {
    initResizableColumns();
  }
})();
</script>
"""
APP_HEAD = APP_HEAD.replace("__APP_ICON__", APP_ICON)
HERO_HTML = """
<section class="hero">
  <div class="hero-brand">
    <div class="hero-logo" aria-hidden="true">
      <svg viewBox="0 0 88 88" role="img">
        <defs>
          <linearGradient id="compassShell" x1="12%" y1="12%" x2="88%" y2="88%">
            <stop offset="0%" stop-color="#ffffff" stop-opacity="0.98" />
            <stop offset="100%" stop-color="#d9e7ff" stop-opacity="0.92" />
          </linearGradient>
          <linearGradient id="needleNorth" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stop-color="#ff8f6b" />
            <stop offset="100%" stop-color="#d94a2d" />
          </linearGradient>
          <linearGradient id="needleSouth" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stop-color="#263f8f" />
            <stop offset="100%" stop-color="#6d8df7" />
          </linearGradient>
        </defs>
        <circle cx="44" cy="44" r="34" fill="url(#compassShell)" stroke="#89a7ff" stroke-width="2.5" />
        <circle cx="44" cy="44" r="25" fill="none" stroke="#89a7ff" stroke-opacity="0.35" stroke-width="1.6" />
        <path d="M44 16 L47 24 L44 22 L41 24 Z" fill="#355cde" />
        <path d="M72 44 L64 47 L66 44 L64 41 Z" fill="#355cde" opacity="0.8" />
        <path d="M44 72 L41 64 L44 66 L47 64 Z" fill="#355cde" opacity="0.55" />
        <path d="M16 44 L24 41 L22 44 L24 47 Z" fill="#355cde" opacity="0.8" />
        <g class="compass-needle">
          <path d="M44 18 L52 44 L44 40 L36 44 Z" fill="url(#needleNorth)" />
          <path d="M44 70 L52 44 L44 48 L36 44 Z" fill="url(#needleSouth)" />
        </g>
        <circle cx="44" cy="44" r="5.5" fill="#0f1b40" />
        <circle cx="44" cy="44" r="2.2" fill="#ffffff" />
      </svg>
    </div>
    <div class="hero-copy">
      <div class="hero-kicker">Large Model Inference Simulation Suite</div>
      <h1>Modeling Compass</h1>
      <p>
        A unified workspace for large-model simulation, video generation
        simulation, and inference deployment optimization. Supports LLM,
        VL, and video generation models, with parameter sweeps, baseline
        analysis, cross-device comparison,
        bottleneck analysis, and result export for both baseline evaluation and peak-performance exploration.
      </p>
    </div>
  </div>
</section>
"""


def build_theme():
    """Build the Gradio theme with web UI CSS loaded after theme variables."""
    if gr is None:
        raise RuntimeError("gradio is not installed")
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate").set(
        body_background_fill=COLORS["page_bg"],
        body_text_color=COLORS["text_main"],
        body_text_color_subdued=COLORS["text_sub"],
        block_background_fill="rgba(255,255,255,0.78)",
        block_background_fill_dark="rgba(255,255,255,0.78)",
        block_border_color="rgba(166, 184, 224, 0.58)",
        block_border_color_dark="rgba(166, 184, 224, 0.58)",
        block_border_width="1px",
        block_label_background_fill="rgba(248,251,255,0.96)",
        block_label_background_fill_dark="rgba(248,251,255,0.96)",
        block_label_border_color="rgba(166, 184, 224, 0.44)",
        block_label_border_color_dark="rgba(166, 184, 224, 0.44)",
        block_label_text_color=COLORS["text_main"],
        block_label_text_color_dark=COLORS["text_main"],
        block_radius="12px",
        block_shadow="none",
        block_shadow_dark="none",
        block_title_background_fill="rgba(248,251,255,0.96)",
        block_title_background_fill_dark="rgba(248,251,255,0.96)",
        block_title_border_color="rgba(166, 184, 224, 0.44)",
        block_title_border_color_dark="rgba(166, 184, 224, 0.44)",
        border_color_primary="rgba(166, 184, 224, 0.44)",
        border_color_primary_dark="rgba(166, 184, 224, 0.44)",
        button_border_width="1px",
        button_large_radius="10px",
        button_medium_radius="10px",
        button_primary_background_fill=COLORS["primary"],
        button_primary_background_fill_hover="#152a6a",
        button_primary_border_color="rgba(33, 64, 154, 0.18)",
        button_primary_border_color_hover="rgba(33, 64, 154, 0.28)",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="rgba(255,255,255,0.92)",
        button_secondary_background_fill_hover="rgba(245,248,255,0.98)",
        button_secondary_border_color="rgba(166, 184, 224, 0.52)",
        button_secondary_border_color_hover="rgba(33, 64, 154, 0.30)",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="0 8px 18px rgba(25, 40, 78, 0.08)",
        button_secondary_text_color=COLORS["text_main"],
        checkbox_background_color="#ffffff",
        checkbox_background_color_focus="#ffffff",
        checkbox_background_color_hover="#ffffff",
        checkbox_background_color_selected=COLORS["primary"],
        checkbox_border_color="rgba(166, 184, 224, 0.78)",
        checkbox_border_color_focus="rgba(33, 64, 154, 0.58)",
        checkbox_border_color_hover="rgba(33, 64, 154, 0.34)",
        checkbox_border_color_selected=COLORS["primary"],
        checkbox_border_radius="6px",
        checkbox_shadow="none",
        input_background_fill="rgba(255,255,255,0.96)",
        input_background_fill_focus="#ffffff",
        input_background_fill_hover="#ffffff",
        input_border_color="rgba(166, 184, 224, 0.72)",
        input_border_color_focus="rgba(33, 64, 154, 0.52)",
        input_border_color_hover="rgba(33, 64, 154, 0.28)",
        input_border_width="1px",
        input_radius="10px",
        input_shadow="inset 0 1px 0 rgba(255,255,255,0.78)",
        input_shadow_focus="0 0 0 3px rgba(33, 64, 154, 0.10)",
        panel_border_color="rgba(166, 184, 224, 0.44)",
        panel_border_color_dark="rgba(166, 184, 224, 0.44)",
        shadow_drop="0 1px 4px 0 rgba(25, 40, 78, 0.08)",
        shadow_drop_lg="0 2px 6px 0 rgba(25, 40, 78, 0.10)",
        shadow_inset="rgba(25, 40, 78, 0.04) 0px 2px 4px 0px inset",
        table_border_color="rgba(166, 184, 224, 0.50)",
        table_border_color_dark="rgba(166, 184, 224, 0.50)",
    )
    return theme


def build_app() -> gr.Blocks:
    """Build the Gradio application."""
    if gr is None:
        raise RuntimeError("gradio is not installed. Run `uv sync` (see README) or `pip install gradio`.")

    vendor_map = get_vendor_device_map()
    vendors = list(vendor_map.keys())
    default_vendor = vendors[0] if vendors else ""
    default_devices = vendor_map.get(default_vendor, [])
    default_device = default_devices[0] if default_devices else None

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(HERO_HTML)

        def _build_text_generate_workspace(
            section_title: str,
            description: str,
            *,
            vl_mode: bool = False,
            default_model: str = "Qwen/Qwen3-32B",
        ):
            render_section_card(section_title, description)
            with gr.Group(elem_classes=["section-card"]):
                tg_model = gr.Textbox(label="model-id", value=default_model)
                with gr.Row():
                    tg_vendor = gr.Dropdown(vendors, value=default_vendor, label="Vendor")
                    tg_device = gr.Dropdown(
                        default_devices,
                        value=default_device,
                        label="Device",
                    )
                with gr.Row():
                    tg_comp_vendor = gr.Dropdown(vendors, multiselect=True, label="Compare Vendors")
                    tg_comp = gr.Dropdown([], multiselect=True, label="Compare Devices")
                with gr.Accordion("Concurrency and Length", open=True):
                    with gr.Row():
                        tg_num_devices = gr.Textbox(label="num-devices", value="1")
                        tg_num_queries = gr.Textbox(label="num-queries", value="32")
                        tg_num_queries_list = gr.Textbox(
                            label="num-queries list",
                            value="",
                            placeholder="e.g. [1,2,4,8,16,32]",
                        )
                    with gr.Row():
                        tg_query_len = gr.Textbox(label="query-length", value="1")
                        tg_context_len = gr.Textbox(label="context-length", value="4500")
                    with gr.Row():
                        tg_decode = gr.Checkbox(label="decode", value=True)
                        tg_mtp = gr.Textbox(label="num-mtp-tokens", value="0")
                        tg_mtp_acceptance_rate = gr.Textbox(
                            label="mtp-acceptance-rate",
                            value="0.9,0.6,0.4,0.2",
                            placeholder="e.g. 0.9,0.6,0.4,0.2",
                            visible=False,
                        )
                with gr.Accordion("Quantization", open=True):
                    with gr.Row():
                        tg_qlinear = gr.Dropdown(
                            QUANT_LINEAR_OPTIONS,
                            value="W8A8_DYNAMIC",
                            label="quantize-linear-action",
                            allow_custom_value=True,
                        )
                        tg_qlinear_list = gr.Textbox(
                            label="quantize-linear-action list",
                            value="",
                            placeholder="e.g. [DISABLED,W8A8_DYNAMIC,FP8]",
                        )
                    with gr.Row():
                        tg_qattn = gr.Dropdown(
                            QUANT_ATTENTION_OPTIONS,
                            value="DISABLED",
                            label="quantize-attention-action",
                            allow_custom_value=True,
                        )
                        tg_qattn_list = gr.Textbox(
                            label="quantize-attention-action list",
                            value="",
                            placeholder="e.g. [DISABLED,INT8]",
                        )
                with gr.Accordion("Parallel Settings", open=False):
                    with gr.Row():
                        tg_tp = gr.Textbox(label="tp-size", value="1")
                        tg_tp_list = gr.Textbox(label="TP List", value="", placeholder="e.g. [1,2,4,8]")
                        tg_dp = gr.Textbox(label="dp-size (auto/num)", value="auto")
                        tg_ep = gr.Textbox(label="ep-size", value="1")
                    tg_compile = gr.Checkbox(label="compile", value=True)
                tg_img_bs = gr.Textbox(visible=False, value="")
                tg_img_h = gr.Textbox(visible=False, value="")
                tg_img_w = gr.Textbox(visible=False, value="")
                if vl_mode:
                    with gr.Accordion("VL Parameters", open=True):
                        with gr.Row():
                            tg_img_bs = gr.Textbox(label="image-batch-size", value="1")
                            tg_img_h = gr.Textbox(label="image-height", value="1024")
                            tg_img_w = gr.Textbox(label="image-width", value="1024")
                        gr.Markdown(
                            "The VL workspace supports image-batch-size, image-height, and image-width. "
                            "These inputs flow directly into the cross-device forward comparison workflow."
                        )
                with gr.Accordion("Other Parameters", open=False):
                    gr.Markdown(
                        "The fields below keep the CLI defaults unless you override them. "
                        "Preview validates the inputs and shows the generated command.",
                        elem_classes=["field-hint"],
                    )
                    with gr.Row():
                        tg_prefix_cache_hit_rate = gr.Textbox(
                            label="prefix-cache-hit-rate",
                            value="0",
                            placeholder="0 <= value < 1",
                        )
                        tg_reserved_memory_gb = gr.Textbox(
                            label="reserved-memory-gb (GB)",
                            value="0.0",
                            placeholder="default 0.0",
                        )
                        tg_log_level = gr.Dropdown(
                            ["debug", "info", "warning", "error", "critical"],
                            value="error",
                            label="log-level",
                        )
                    with gr.Row():
                        tg_compile_allow_graph_break = gr.Checkbox(
                            label="compile-allow-graph-break",
                            value=False,
                        )
                        tg_disable_repetition = gr.Checkbox(
                            label="disable-repetition",
                            value=False,
                        )
                        tg_quantize_lmhead = gr.Checkbox(label="quantize-lmhead", value=False)
                    with gr.Row():
                        tg_mxfp4_group_size = gr.Textbox(label="mxfp4-group-size", value="32")
                        tg_num_hidden_layers_override = gr.Textbox(
                            label="num-hidden-layers-override",
                            value="0",
                        )
                    with gr.Row():
                        tg_graph_log_url = gr.Textbox(label="Graph log URL", value="")
                        tg_chrome_trace = gr.Textbox(label="chrome-trace", value="")
                    tg_dump_input_shapes = gr.Checkbox(
                        label="dump-input-shapes",
                        value=False,
                    )
                    with gr.Accordion("Advanced Parallel Overrides", open=False):
                        gr.Markdown(
                            "These overrides must remain consistent with "
                            "num-devices. For example, the TP/DP/EP product "
                            "must not exceed num-devices, and an explicit "
                            "tp-size should usually divide num-devices exactly.",
                            elem_classes=["field-hint"],
                        )
                        with gr.Row():
                            tg_o_proj_tp_size = gr.Textbox(label="O-Proj TP", value="")
                            tg_o_proj_dp_size = gr.Textbox(label="O-Proj DP", value="")
                            tg_mlp_tp_size = gr.Textbox(label="MLP TP", value="")
                            tg_mlp_dp_size = gr.Textbox(label="MLP DP", value="")
                        with gr.Row():
                            tg_lmhead_tp_size = gr.Textbox(label="LMHead TP", value="")
                            tg_lmhead_dp_size = gr.Textbox(label="LMHead DP", value="")
                            tg_moe_tp_size = gr.Textbox(label="MoE TP", value="")
                            tg_moe_dp_size = gr.Textbox(label="MoE DP", value="1")
                        with gr.Row():
                            tg_word_embedding_tp = gr.Dropdown(
                                ["", "col", "row"],
                                value="",
                                label="word-embedding-tp",
                            )
                            tg_enable_redundant_experts = gr.Checkbox(
                                label="enable-redundant-experts",
                                value=False,
                            )
                            tg_enable_external_shared_experts = gr.Checkbox(
                                label="enable-external-shared-experts",
                                value=False,
                            )
                            tg_host_external_shared_experts = gr.Checkbox(
                                label="host-external-shared-experts",
                                value=False,
                            )
                    with gr.Row():
                        tg_remote_source = gr.Dropdown(
                            ["huggingface", "modelscope"],
                            value="huggingface",
                            label="remote-source",
                        )
                        tg_performance_model = gr.CheckboxGroup(
                            ["analytic", "profiling"],
                            value=["analytic"],
                            label="performance-model",
                        )
                        tg_profiling_database = gr.Textbox(label="profiling-database", value="")
                with gr.Row():
                    tg_preview_btn = gr.Button("Preview Configuration")
                    tg_run = gr.Button("Run", variant="primary")
                tg_preview_summary = gr.Markdown(
                    "### Configuration Summary\n"
                    "Click Preview to review the model, device, device "
                    "count, concurrency, quantization, and estimated task "
                    "count.",
                    elem_classes=["preview-summary"],
                )
                with gr.Accordion("Command Preview", open=False):
                    tg_preview = gr.Textbox(label="Command", lines=4, interactive=False)
            (
                tg_progress,
                tg_summary,
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
            ) = text_generate_result_section()
            tg_vendor.change(
                lambda v: gr.update(choices=vendor_map.get(v, []), value=(vendor_map.get(v, [None])[0])),
                inputs=[tg_vendor],
                outputs=[tg_device],
            )
            tg_comp_vendor.change(
                lambda vs: gr.update(
                    choices=sorted({d for vendor in (vs or []) for d in vendor_map.get(vendor, [])}),
                    value=[],
                ),
                inputs=[tg_comp_vendor],
                outputs=[tg_comp],
            )
            tg_op_device.change(
                update_op_table_from_breakdown,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_device,
                    tg_op_case,
                    tg_op_top_n,
                    tg_op_columns,
                    tg_op_sort,
                ],
                outputs=[tg_op_table],
            )
            tg_op_top_n.change(
                update_op_table_from_breakdown,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_device,
                    tg_op_case,
                    tg_op_top_n,
                    tg_op_columns,
                    tg_op_sort,
                ],
                outputs=[tg_op_table],
            )
            tg_op_sort.change(
                update_op_table_from_breakdown,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_device,
                    tg_op_case,
                    tg_op_top_n,
                    tg_op_columns,
                    tg_op_sort,
                ],
                outputs=[tg_op_table],
            )
            tg_op_columns.change(
                update_op_table_from_breakdown,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_device,
                    tg_op_case,
                    tg_op_top_n,
                    tg_op_columns,
                    tg_op_sort,
                ],
                outputs=[tg_op_table],
            )
            tg_op_case.change(
                update_op_table_from_breakdown,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_device,
                    tg_op_case,
                    tg_op_top_n,
                    tg_op_columns,
                    tg_op_sort,
                ],
                outputs=[tg_op_table],
            )
            tg_inputs = [
                tg_model,
                tg_device,
                tg_comp,
                tg_num_devices,
                tg_num_queries,
                tg_num_queries_list,
                tg_query_len,
                tg_context_len,
                tg_decode,
                tg_mtp,
                tg_mtp_acceptance_rate,
                tg_compile,
                tg_qlinear,
                tg_qlinear_list,
                tg_qattn,
                tg_qattn_list,
                tg_tp,
                tg_tp_list,
                tg_dp,
                tg_ep,
                tg_img_bs,
                tg_img_h,
                tg_img_w,
                tg_prefix_cache_hit_rate,
                tg_reserved_memory_gb,
                tg_log_level,
                tg_compile_allow_graph_break,
                tg_disable_repetition,
                tg_quantize_lmhead,
                tg_mxfp4_group_size,
                tg_graph_log_url,
                tg_dump_input_shapes,
                tg_chrome_trace,
                tg_num_hidden_layers_override,
                tg_o_proj_tp_size,
                tg_o_proj_dp_size,
                tg_mlp_tp_size,
                tg_mlp_dp_size,
                tg_lmhead_tp_size,
                tg_lmhead_dp_size,
                tg_moe_tp_size,
                tg_moe_dp_size,
                tg_word_embedding_tp,
                tg_enable_redundant_experts,
                tg_enable_external_shared_experts,
                tg_host_external_shared_experts,
                tg_remote_source,
                tg_performance_model,
                tg_profiling_database,
            ]

            def _toggle_mtp_acceptance_rate(mtp_tokens):
                try:
                    mtp_val = int(mtp_tokens) if mtp_tokens else 0
                except (ValueError, TypeError):
                    mtp_val = 0
                return gr.update(visible=mtp_val > 0)

            def _validate_mtp_tokens(query_len, mtp_tokens):
                try:
                    q_len = int(query_len) if query_len else 0
                    m_val = int(mtp_tokens) if mtp_tokens else 0
                except (ValueError, TypeError):
                    q_len = 0
                    m_val = 0
                if m_val > 0 and q_len <= m_val:
                    return gr.update(value=str(m_val + 1))
                return gr.update(value=query_len)

            tg_mtp.change(
                _toggle_mtp_acceptance_rate,
                inputs=[tg_mtp],
                outputs=[tg_mtp_acceptance_rate],
            )
            tg_mtp.change(
                _validate_mtp_tokens,
                inputs=[tg_query_len, tg_mtp],
                outputs=[tg_query_len],
            )
            tg_preview_btn.click(
                preview_text_generate,
                inputs=tg_inputs,
                outputs=[tg_preview_summary, tg_preview],
            )
            tg_run.click(
                run_text_generate_v2,
                inputs=tg_inputs,
                outputs=[
                    tg_progress,
                    tg_summary,
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
                    tg_op_table,
                    tg_op_category_device,
                    tg_op_category_case,
                    tg_op_category_chart,
                    tg_op_category_table,
                    tg_op_compare_mode,
                    tg_op_compare_table,
                    tg_results_df,
                    tg_display_state,
                    tg_full_state,
                    tg_op_breakdown_state,
                    tg_current_model_state,
                    tg_mtp_acceptance_state,
                ],
            )
            wire_export(tg_export_btn, tg_display_state, tg_export_file, "text_generate_results")
            tg_memory_device.change(
                update_memory_analysis_by_device,
                inputs=[tg_full_state, tg_memory_device, tg_memory_case],
                outputs=[tg_memory_pie, tg_memory_table],
            )
            tg_memory_case.change(
                update_memory_analysis_by_device,
                inputs=[tg_full_state, tg_memory_device, tg_memory_case],
                outputs=[tg_memory_pie, tg_memory_table],
            )
            tg_bandwidth_device.change(
                update_bandwidth_analysis_by_device,
                inputs=[tg_full_state, tg_bandwidth_device, tg_bandwidth_case],
                outputs=[tg_bandwidth_table],
            )
            tg_bandwidth_case.change(
                update_bandwidth_analysis_by_device,
                inputs=[tg_full_state, tg_bandwidth_device, tg_bandwidth_case],
                outputs=[tg_bandwidth_table],
            )
            tg_op_category_device.change(
                update_category_stats_by_device,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_category_device,
                    tg_op_category_case,
                ],
                outputs=[tg_op_category_chart, tg_op_category_table],
            )
            tg_op_category_case.change(
                update_category_stats_by_device,
                inputs=[
                    tg_op_breakdown_state,
                    tg_op_category_device,
                    tg_op_category_case,
                ],
                outputs=[tg_op_category_chart, tg_op_category_table],
            )
            tg_op_compare_mode.change(
                update_compare_table_by_mode,
                inputs=[tg_op_breakdown_state, tg_op_compare_mode],
                outputs=[tg_op_compare_table],
            )

        def _build_video_generate_workspace():
            render_section_card(
                "Multimodal / Video Generation Simulation",
                "Supports video generation simulation, USP / CFG / DiT "
                "Cache combinations, and cross-device comparison.",
            )
            with gr.Group(elem_classes=["section-card"]):
                vg_model = gr.Textbox(label="model-id", value="Wan2.2-T2V-A14B-Diffusers")
                with gr.Row():
                    vg_vendor = gr.Dropdown(vendors, value=default_vendor, label="Vendor")
                    vg_device = gr.Dropdown(
                        default_devices,
                        value=default_device,
                        label="Device",
                    )
                with gr.Row():
                    vg_comp_vendor = gr.Dropdown(vendors, multiselect=True, label="Compare Vendors")
                    vg_comp = gr.Dropdown([], multiselect=True, label="Compare Devices")
                with gr.Row():
                    vg_batch = gr.Textbox(label="batch-size", value="1")
                    vg_seq = gr.Textbox(label="seq-len", value="128")
                    vg_dtype = gr.Dropdown(
                        ["float16", "float32", "bfloat16"],
                        value="float16",
                        label="dtype",
                        allow_custom_value=True,
                    )
                with gr.Row():
                    vg_h = gr.Textbox(label="height", value="1280")
                    vg_w = gr.Textbox(label="width", value="720")
                    vg_frame = gr.Textbox(label="frame-num", value="129")
                    vg_step = gr.Textbox(label="sample-step", value="50")
                with gr.Row():
                    vg_qlinear = gr.Dropdown(
                        QUANT_LINEAR_OPTIONS,
                        value="W8A8_DYNAMIC",
                        label="quantize-linear-action",
                        allow_custom_value=True,
                    )
                    vg_qlinear_list = gr.Textbox(
                        label="quantize-linear-action list",
                        value="",
                        placeholder="e.g. [DISABLED,W8A8_DYNAMIC]",
                    )
                with gr.Row():
                    vg_world = gr.Textbox(label="num-devices", value="8")
                    vg_ulysses = gr.Textbox(label="ulysses-size", value="4")
                    vg_ulysses_list = gr.Textbox(
                        label="ulysses-size list",
                        value="",
                        placeholder="\u5982 [1,2,4,8]",
                    )
                with gr.Row():
                    vg_cfg = gr.Checkbox(label="use-cfg", value=True)
                    vg_cfgp = gr.Checkbox(label="cfg-parallel", value=True)
                with gr.Accordion("DiT Cache", open=False):
                    vg_cache = gr.Checkbox(label="dit-cache", value=False)
                    with gr.Row():
                        vg_cache_range = gr.Textbox(label="cache-step-range", value="20,30")
                        vg_cache_interval = gr.Textbox(label="cache-step-interval", value="5")
                        vg_cache_block = gr.Textbox(label="cache-block-range", value="")
                with gr.Accordion("Other Parameters", open=False):
                    gr.Markdown(
                        "The fields below keep the `video_generate.py` "
                        "defaults unless you open this section and "
                        "override them.",
                        elem_classes=["field-hint"],
                    )
                    with gr.Row():
                        vg_chrome_trace = gr.Textbox(label="chrome-trace", value="")
                        vg_log_level = gr.Dropdown(
                            ["debug", "info", "warning", "error", "critical"],
                            value="info",
                            label="log-level",
                        )
                with gr.Row():
                    vg_preview_btn = gr.Button("Preview Configuration")
                    vg_run = gr.Button("Run", variant="primary")
                vg_preview_summary = gr.Markdown(
                    "### \u914d\u7f6e\u6458\u8981\n"
                    "\u70b9\u51fb\u9884\u89c8\u540e\u663e\u793a\u6a21\u578b\u3001"
                    "\u82af\u7247\u3001\u5361\u6570\u3001\u89c6\u9891\u89c4\u683c\u3001"
                    "\u91cf\u5316\u548c\u9884\u8ba1\u4efb\u52a1\u6570\u3002",
                    elem_classes=["preview-summary"],
                )
                with gr.Accordion("Command Preview", open=False):
                    vg_preview = gr.Textbox(label="Command", lines=4, interactive=False)
            (
                vg_progress,
                vg_summary,
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
            ) = video_generate_result_section()
            vg_vendor.change(
                lambda v: gr.update(choices=vendor_map.get(v, []), value=(vendor_map.get(v, [None])[0])),
                inputs=[vg_vendor],
                outputs=[vg_device],
            )
            vg_comp_vendor.change(
                lambda vs: gr.update(
                    choices=sorted({d for vendor in (vs or []) for d in vendor_map.get(vendor, [])}),
                    value=[],
                ),
                inputs=[vg_comp_vendor],
                outputs=[vg_comp],
            )
            vg_op_device.change(
                update_video_op_table_from_breakdown,
                inputs=[
                    vg_op_breakdown_state,
                    vg_op_device,
                    vg_op_top_n,
                    vg_op_columns,
                    vg_op_sort,
                ],
                outputs=[vg_op_table],
            )
            vg_op_top_n.change(
                update_video_op_table_from_breakdown,
                inputs=[
                    vg_op_breakdown_state,
                    vg_op_device,
                    vg_op_top_n,
                    vg_op_columns,
                    vg_op_sort,
                ],
                outputs=[vg_op_table],
            )
            vg_op_sort.change(
                update_video_op_table_from_breakdown,
                inputs=[
                    vg_op_breakdown_state,
                    vg_op_device,
                    vg_op_top_n,
                    vg_op_columns,
                    vg_op_sort,
                ],
                outputs=[vg_op_table],
            )
            vg_op_columns.change(
                update_video_op_table_from_breakdown,
                inputs=[
                    vg_op_breakdown_state,
                    vg_op_device,
                    vg_op_top_n,
                    vg_op_columns,
                    vg_op_sort,
                ],
                outputs=[vg_op_table],
            )
            vg_inputs = [
                vg_model,
                vg_device,
                vg_comp,
                vg_batch,
                vg_seq,
                vg_h,
                vg_w,
                vg_frame,
                vg_step,
                vg_dtype,
                vg_qlinear,
                vg_qlinear_list,
                vg_world,
                vg_ulysses,
                vg_ulysses_list,
                vg_cfg,
                vg_cfgp,
                vg_cache,
                vg_cache_range,
                vg_cache_interval,
                vg_cache_block,
                vg_chrome_trace,
                vg_log_level,
            ]
            vg_preview_btn.click(
                preview_video_generate,
                inputs=vg_inputs,
                outputs=[vg_preview_summary, vg_preview],
            )
            vg_run.click(
                run_video_generate_v2,
                inputs=vg_inputs,
                outputs=[
                    vg_progress,
                    vg_summary,
                    vg_time_chart,
                    vg_comm_chart,
                    vg_op_device,
                    vg_op_table,
                    vg_op_category_chart,
                    vg_op_category_table,
                    vg_op_compare_table,
                    vg_results_df,
                    vg_display_state,
                    vg_full_state,
                    vg_op_breakdown_state,
                ],
            )

        with gr.Tabs():
            with gr.Tab("Simulator"):
                render_section_card(
                    "Simulator",
                    "Configure devices, context length, request count, "
                    "output tokens, analytic latency, operator timing, and "
                    "memory usage.",
                )
                with gr.Tabs(elem_classes=["sim-mode-tabs"]):
                    with gr.Tab("LLM Models"):
                        _build_text_generate_workspace(
                            "LLM Forward Simulation",
                            "Run forward simulation for LLM models such as Qwen, DeepSeek, and GLM5.",
                            vl_mode=False,
                            default_model="Qwen/Qwen3-32B",
                        )
                    with gr.Tab("VL Models"):
                        _build_text_generate_workspace(
                            "VL Forward Simulation",
                            "Forward simulation for image-text inputs with "
                            "configurable image count, image height, and "
                            "image width, followed by cross-device "
                            "comparison.",
                            vl_mode=True,
                            default_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        )
                    with gr.Tab("Video Models"):
                        _build_video_generate_workspace()

            # ==================== Optimizer Tab ====================
            with gr.Tab("Optimizer"):
                render_section_card(
                    "LLM Inference Deployment Optimizer",
                    "Optimize deployment plans and compare multiple "
                    "devices, constraint modes, and quantization "
                    "strategies.",
                )
                with gr.Group(elem_classes=["section-card"]):
                    op_model = gr.Textbox(label="model-id", value="Qwen/Qwen3-32B")
                    with gr.Row():
                        op_vendor = gr.Dropdown(vendors, value=default_vendor, label="Vendor")
                        op_device = gr.Dropdown(default_devices, value=default_device, label="Device")
                    with gr.Row():
                        op_comp_vendor = gr.Dropdown(vendors, multiselect=True, label="Peer Vendors")
                        op_comp = gr.Dropdown([], multiselect=True, label="Peer Devices")
                    with gr.Row():
                        op_num_devices = gr.Textbox(label="Device Count", value="4")
                        op_input = gr.Textbox(label="Input Length", value="3500")
                        op_output = gr.Textbox(label="Output Length", value="1500")
                        op_compile = gr.Checkbox(label="Enable Compilation", value=True)

                    gr.Markdown(
                        "#### Optimizer Workspace\n"
                        "Define the deployment mode and scenario "
                        "constraints first, then drill down into the "
                        "search space. The result area keeps both "
                        "cross-device comparison and the best search "
                        "result for each device family."
                    )
                    with gr.Row():
                        op_mode = gr.Radio(
                            ["PD Aggregated", "PD Disaggregated", "PD Ratio"],
                            value="PD Aggregated",
                            label="Deployment Mode",
                        )
                        op_prefix_cache_hit_rate = gr.Textbox(
                            label="Prefix Cache Hit Rate",
                            value="0",
                            placeholder="Enter a decimal value between 0 and 1. Default: 0",
                        )
                    op_mode_hint = gr.Markdown(
                        "Current mode: **PD Aggregated**. Use it for "
                        "multi-device baseline comparison before reviewing "
                        "the best configuration for each device."
                    )
                    gr.Markdown(
                        "Recommended reading order: **Best by Device -> "
                        "Fixed-Config Comparison -> PD Ratio -> "
                        "Single-Device Pareto Details**."
                    )

                    # Scenario presets
                    with gr.Accordion("Scenario Presets", open=False), gr.Row():
                        op_preset_offline = gr.Button("Offline Batch", size="sm")
                        op_preset_online = gr.Button("Online Service", size="sm")
                        op_preset_deep = gr.Button("Long Output", size="sm")
                        op_preset_fast = gr.Button("Fast Response", size="sm")

                    with gr.Accordion("Targets and Search Space", open=True):
                        with gr.Row():
                            op_tpot = gr.Textbox(
                                label="TPOT (ms)",
                                value="",
                                placeholder="Leave empty for offline scenarios",
                            )
                            op_tpot_list = gr.Textbox(
                                label="TPOT List",
                                value="",
                                placeholder="e.g. [None,50]",
                            )
                            op_ttft = gr.Textbox(
                                label="TTFT (ms)",
                                value="",
                                placeholder="Leave empty for offline scenarios",
                            )
                            op_ttft_list = gr.Textbox(
                                label="TTFT List",
                                value="",
                                placeholder="e.g. [None,2000]",
                            )
                        with gr.Row():
                            op_qlinear = gr.Dropdown(
                                QUANT_LINEAR_OPTIONS,
                                value="W8A8_DYNAMIC",
                                label="MLP Quantization Mode",
                                allow_custom_value=True,
                            )
                            op_qlinear_list = gr.Textbox(
                                label="MLP Quantization List",
                                value="",
                                placeholder="e.g. [DISABLED,W8A8_DYNAMIC,FP8]",
                            )
                        with gr.Row():
                            op_qattn = gr.Dropdown(
                                QUANT_ATTENTION_OPTIONS,
                                value="INT8",
                                label="Attention Quantization Mode",
                                allow_custom_value=True,
                            )
                            op_qattn_list = gr.Textbox(
                                label="Attention Quantization List",
                                value="",
                                placeholder="e.g. [DISABLED,INT8,FP8]",
                            )
                        with gr.Row():
                            op_tp_sizes = gr.Textbox(
                                label="TP Parallel Size List",
                                value="",
                                placeholder="e.g. [1,2,4,8], leave empty for automatic calculation",
                            )
                            op_batch_range = gr.Textbox(
                                label="Batch Size Range",
                                value="",
                                placeholder="e.g. [1,256] or [256]",
                            )
                        with gr.Row():
                            op_jobs = gr.Textbox(
                                label="Parallel Jobs",
                                value="8",
                                placeholder="Default: 8",
                            )
                            op_mxfp4_group_size = gr.Textbox(
                                label="MXFP4 Group Size",
                                value="32",
                                placeholder="Only used for MXFP4 quantization. Default: 32",
                            )

                    with gr.Accordion("Advanced Deployment Options", open=False):
                        with gr.Row():
                            op_prefill_devices_per_instance = gr.Textbox(
                                label="Prefill Devices per Instance",
                                value="1",
                                visible=False,
                                placeholder="Required in PD Ratio mode",
                            )
                            op_decode_devices_per_instance = gr.Textbox(
                                label="Decode Devices per Instance",
                                value="1",
                                visible=False,
                                placeholder="Required in PD Ratio mode",
                            )
                        with gr.Row():
                            op_compile_break = gr.Checkbox(label="Allow Graph Breaks", value=False)
                            op_max_batched_tokens = gr.Textbox(
                                label="Max Batched Tokens (max-batched-tokens)",
                                value="8192",
                                placeholder="Default: 8192",
                            )
                        with gr.Row():
                            op_mtp_tokens = gr.Textbox(
                                label="Speculative Decoding Token Count (num-mtp-tokens)",
                                value="0",
                                placeholder="0 disables this option. Supported by MTP models such as DeepSeek",
                            )
                            op_mtp_acceptance_rate = gr.Textbox(
                                label="Speculative Decoding Acceptance Rate (mtp-acceptance-rate)",
                                value="0.9,0.6,0.4,0.2",
                                placeholder="e.g. 0.9,0.6,0.4,0.2",
                            )
                    with gr.Accordion("VL Parameters", open=False), gr.Row():
                        op_img_h = gr.Textbox(label="image-height", value="")
                        op_img_w = gr.Textbox(label="image-width", value="")
                    with gr.Accordion("Other Parameters", open=False):
                        gr.Markdown(
                            "\u4ee5\u4e0b\u53c2\u6570\u9ed8\u8ba4\u4fdd\u6301 "
                            "throughput_optimizer.py \u9ed8\u8ba4\u503c\uff0c"
                            "\u6253\u5f00\u540e\u624d\u9700\u8981\u624b\u52a8\u8bbe\u7f6e\u3002",
                            elem_classes=["field-hint"],
                        )
                        with gr.Row():
                            op_reserved_memory_gb = gr.Textbox(label="reserved-memory-gb (GB)", value="0.0")
                            op_log_level = gr.Dropdown(
                                ["debug", "info", "warning", "error", "critical"],
                                value="error",
                                label="log-level",
                            )
                        with gr.Row():
                            op_serving_cost = gr.Textbox(label="Serving cost", value="0")
                            op_dump_original_results = gr.Checkbox(
                                label="\u5bfc\u51fa\u539f\u59cb\u5bfb\u4f18\u7ed3\u679c",
                                value=False,
                            )
                    with gr.Row():
                        op_preview_btn = gr.Button("Preview Configuration")
                        op_run = gr.Button("Run", variant="primary")
                    op_preview_summary = gr.Markdown(
                        "### Configuration Summary\n"
                        "Click Preview to review the model, device, "
                        "deployment mode, constraints, quantization, and "
                        "estimated task count.",
                        elem_classes=["preview-summary"],
                    )
                    with gr.Accordion("Command Preview", open=False):
                        op_preview = gr.Textbox(label="Command", lines=4, interactive=False)
                # Optimizer result workspace
                (
                    op_progress,
                    op_summary,
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
                ) = optimizer_result_section()

                # fixed-config comparison wiring
                op_fixed_config.change(
                    refresh_optimizer_fixed_compare,
                    inputs=[op_candidate_state, op_fixed_config, op_fixed_metric],
                    outputs=[op_fixed_md, op_fixed_chart, op_fixed_df],
                )
                op_fixed_metric.change(
                    refresh_optimizer_fixed_compare,
                    inputs=[op_candidate_state, op_fixed_config, op_fixed_metric],
                    outputs=[op_fixed_md, op_fixed_chart, op_fixed_df],
                )

                # single-device detail wiring
                op_detail_device.change(
                    refresh_optimizer_detail_v2,
                    inputs=[op_full_state, op_candidate_state, op_detail_device],
                    outputs=[
                        op_detail_md,
                        op_detail_pareto_chart,
                        op_detail_df,
                        op_detail_output,
                    ],
                )
                # Event bindings
                op_vendor.change(
                    lambda v: gr.update(
                        choices=vendor_map.get(v, []),
                        value=(vendor_map.get(v, [None])[0]),
                    ),
                    inputs=[op_vendor],
                    outputs=[op_device],
                )
                op_comp_vendor.change(
                    lambda vs: gr.update(
                        choices=sorted({d for vendor in (vs or []) for d in vendor_map.get(vendor, [])}),
                        value=[],
                    ),
                    inputs=[op_comp_vendor],
                    outputs=[op_comp],
                )

                def update_optimizer_mode_ui(mode: str):
                    hints = {
                        "PD Aggregated": (
                            "Current mode: **PD Aggregated**. Start with "
                            "cross-device baselines and then inspect the "
                            "best configuration for each device."
                        ),
                        "PD Disaggregated": (
                            "Current mode: **PD Disaggregated**. Use "
                            "this mode for disaggregated serving and "
                            "compare best results across devices in "
                            "prefill/decode split scenarios."
                        ),
                        "PD Ratio": (
                            "Current mode: **PD Ratio**. Search "
                            "Prefill/Decode balance directly and compare "
                            "Balanced QPS across devices."
                        ),
                    }
                    is_pd_mode = mode == "PD Ratio"
                    return (
                        gr.update(visible=is_pd_mode),
                        gr.update(visible=is_pd_mode),
                        gr.update(value=hints.get(mode, hints["PD Aggregated"])),
                    )

                op_mode.change(
                    update_optimizer_mode_ui,
                    inputs=[op_mode],
                    outputs=[
                        op_prefill_devices_per_instance,
                        op_decode_devices_per_instance,
                        op_mode_hint,
                    ],
                )

                # Scenario preset helpers
                def preset_offline():
                    """Offline batch preset: highest throughput with no latency limits."""
                    return (
                        gr.update(value=""),  # TPOT
                        gr.update(value=""),  # TPOT list
                        gr.update(value=""),  # TTFT
                        gr.update(value=""),  # TTFT list
                        gr.update(value="[1,512]"),  # batch_range
                        gr.update(value="3500"),  # input_length
                        gr.update(value="1500"),  # output_length
                    )

                def preset_online():
                    """Online service preset: low-latency interactive serving."""
                    return (
                        gr.update(value="50"),  # TPOT
                        gr.update(value=""),  # TPOT list
                        gr.update(value="2000"),  # TTFT
                        gr.update(value=""),  # TTFT list
                        gr.update(value="[1,128]"),  # batch_range
                        gr.update(value="500"),  # input_length
                        gr.update(value="500"),  # output_length
                    )

                def preset_deep():
                    """Long output preset for deep inference workloads."""
                    return (
                        gr.update(value=""),  # TPOT
                        gr.update(value=""),  # TPOT list
                        gr.update(value=""),  # TTFT
                        gr.update(value=""),  # TTFT list
                        gr.update(value="[1,32]"),  # batch_range
                        gr.update(value="1000"),  # input_length
                        gr.update(value="8000"),  # output_length
                    )

                def preset_fast():
                    """Fast response preset for short interactive requests."""
                    return (
                        gr.update(value="30"),  # TPOT
                        gr.update(value=""),  # TPOT list
                        gr.update(value="1000"),  # TTFT
                        gr.update(value=""),  # TTFT list
                        gr.update(value="[1,128]"),  # batch_range
                        gr.update(value="200"),  # input_length
                        gr.update(value="200"),  # output_length
                    )

                op_preset_offline.click(
                    preset_offline,
                    outputs=[
                        op_tpot,
                        op_tpot_list,
                        op_ttft,
                        op_ttft_list,
                        op_batch_range,
                        op_input,
                        op_output,
                    ],
                )
                op_preset_online.click(
                    preset_online,
                    outputs=[
                        op_tpot,
                        op_tpot_list,
                        op_ttft,
                        op_ttft_list,
                        op_batch_range,
                        op_input,
                        op_output,
                    ],
                )
                op_preset_deep.click(
                    preset_deep,
                    outputs=[
                        op_tpot,
                        op_tpot_list,
                        op_ttft,
                        op_ttft_list,
                        op_batch_range,
                        op_input,
                        op_output,
                    ],
                )
                op_preset_fast.click(
                    preset_fast,
                    outputs=[
                        op_tpot,
                        op_tpot_list,
                        op_ttft,
                        op_ttft_list,
                        op_batch_range,
                        op_input,
                        op_output,
                    ],
                )

                op_inputs = [
                    op_model,
                    op_device,
                    op_comp,
                    op_num_devices,
                    op_input,
                    op_output,
                    op_compile,
                    op_qlinear,
                    op_qlinear_list,
                    op_qattn,
                    op_qattn_list,
                    op_tpot,
                    op_tpot_list,
                    op_ttft,
                    op_ttft_list,
                    op_mtp_tokens,
                    op_mtp_acceptance_rate,
                    op_max_batched_tokens,
                    op_img_h,
                    op_img_w,
                    op_tp_sizes,
                    op_batch_range,
                    op_jobs,
                    op_mode,
                    op_prefix_cache_hit_rate,
                    op_prefill_devices_per_instance,
                    op_decode_devices_per_instance,
                    op_compile_break,
                    op_mxfp4_group_size,
                    op_reserved_memory_gb,
                    op_log_level,
                    op_serving_cost,
                    op_dump_original_results,
                ]
                op_preview_btn.click(
                    preview_optimizer,
                    inputs=op_inputs,
                    outputs=[op_preview_summary, op_preview],
                )
                op_run.click(
                    run_optimizer_v2,
                    inputs=op_inputs,
                    outputs=[
                        op_progress,
                        op_summary,
                        op_throughput_chart,
                        op_ttft_chart,
                        op_tpot_chart,
                        op_batch_chart,
                        op_pd_chart,
                        op_pd_df,
                        op_fixed_config,
                        op_fixed_md,
                        op_fixed_chart,
                        op_fixed_df,
                        op_detail_device,
                        op_detail_md,
                        op_detail_pareto_chart,
                        op_detail_df,
                        op_detail_output,
                        op_results_df,
                        op_display_state,
                        op_full_state,
                        op_candidate_state,
                    ],
                )
                wire_export(
                    op_export_btn,
                    op_display_state,
                    op_export_file,
                    "throughput_optimizer_results",
                )

    return demo


def launch_app(server_name: str = "0.0.0.0", server_port: int = 2345, share: bool = False):
    """Launch the application

    Args:
        server_name: Bind address. Default `0.0.0.0` listens on all interfaces.
        server_port: Bind port.
        share: Whether to create a public sharing link.
    """
    demo = build_app()
    theme = build_theme()
    return demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=False,
        show_error=True,
        theme=theme,
        css=APP_CSS,
        head=APP_HEAD,
    )


if __name__ == "__main__":
    launch_app()
