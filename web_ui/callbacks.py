"""Callback functions for the web UI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import pandas as pd

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from .charts import (
    _safe_df_from_rows,
    bar_plot,
    baseline_plot,
    empty_plot,
    line_plot,
    make_figures,
    scatter_plot,
)
from .command_builder import (
    build_optimizer_tasks,
    build_text_generate_tasks,
    build_video_generate_tasks,
)
from .components import df_to_records, progress_html
from .result_store import ResultStore
from .runner import ExperimentRunner, PROJECT_ROOT
from .utils import parse_optional_number, parse_scalar_or_list

if TYPE_CHECKING:
    from .schemas import ExperimentResult

# Global instances
STORE = ResultStore()
RUNNER = ExperimentRunner(STORE, max_workers=2)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

OPT_DEPLOY_PD_MIXED = "PD Aggregated"
OPT_DEPLOY_PD_SPLIT = "PD Disaggregated"
OPT_DEPLOY_PD_RATIO = "PD Ratio"
OPT_DEPLOY_MODE_ALIASES = {
    "": OPT_DEPLOY_PD_MIXED,
    "Aggregation": OPT_DEPLOY_PD_MIXED,
    "aggregation": OPT_DEPLOY_PD_MIXED,
    "PD Mixed": OPT_DEPLOY_PD_MIXED,
    "pd mixed": OPT_DEPLOY_PD_MIXED,
    "PD Aggregated": OPT_DEPLOY_PD_MIXED,
    "pd aggregated": OPT_DEPLOY_PD_MIXED,
    "\u805a\u5408\u90e8\u7f72": OPT_DEPLOY_PD_MIXED,
    "PD \u6df7\u90e8": OPT_DEPLOY_PD_MIXED,
    OPT_DEPLOY_PD_MIXED: OPT_DEPLOY_PD_MIXED,
    "Disagg": OPT_DEPLOY_PD_SPLIT,
    "disagg": OPT_DEPLOY_PD_SPLIT,
    "PD Split": OPT_DEPLOY_PD_SPLIT,
    "pd split": OPT_DEPLOY_PD_SPLIT,
    "PD Disaggregated": OPT_DEPLOY_PD_SPLIT,
    "pd disaggregated": OPT_DEPLOY_PD_SPLIT,
    "PD \u5206\u79bb": OPT_DEPLOY_PD_SPLIT,
    OPT_DEPLOY_PD_SPLIT: OPT_DEPLOY_PD_SPLIT,
    OPT_DEPLOY_PD_RATIO: OPT_DEPLOY_PD_RATIO,
}


def _normalize_optimizer_deployment_mode(mode: Any) -> str:
    text = str(mode or "").strip()
    return OPT_DEPLOY_MODE_ALIASES.get(text, text)


OP_TABLE_COLUMNS = [
    "Operator",
    "Category",
    "Total Time (ms)",
    "Average Time (ms)",
    "Calls",
    "Device",
]
OP_TABLE_DEFAULT_COLUMNS = [
    "Operator",
    "Category",
    "Total Time (ms)",
    "Average Time (ms)",
    "Calls",
    "Device",
]
OP_TABLE_SORT_OPTIONS = ["Total Time (ms)", "Average Time (ms)", "Calls", "Operator"]


def _dedupe(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _case_label_from_mapping(row: dict[str, Any] | pd.Series) -> str:
    def pick(key: str, default: Any = "-") -> Any:
        try:
            value = row.get(key, default)
        except AttributeError:
            value = default
        if value is None or value == "":
            return default
        return value

    num_queries = pick("num_queries")
    tp_size = pick("tp_size", 1)
    return f"Concurrency={num_queries} | TP={tp_size}"


def _case_choices_from_rows(
    rows: list[dict[str, Any]] | pd.DataFrame | None,
) -> list[str]:
    if rows is None:
        return []
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    choices: list[str] = []
    seen: set[str] = set()
    for row in records:
        label = _case_label_from_mapping(row)
        if label not in seen:
            seen.add(label)
            choices.append(label)
    return choices


def _filter_df_by_case(df: pd.DataFrame, case_label: str | None) -> pd.DataFrame:
    if df.empty or not case_label:
        return df
    if "case_label" not in df.columns:
        df = df.copy()
        df["case_label"] = df.apply(_case_label_from_mapping, axis=1)
    return df[df["case_label"] == case_label]


def _format_preview_error(error: Exception) -> tuple[str, str]:
    return (
        "\n".join(
            [
                "### Parameter Validation Failed",
                f"- {error}",
                "- Review numeric inputs, list syntax, and quantization settings, then preview again.",
            ]
        ),
        "",
    )


def _preview_first_command(tasks) -> str:
    """Return the first generated command as a string."""
    if not tasks:
        return "No command generated."

    first = tasks[0]
    command = getattr(first, "command", None)
    if not command:
        return "No command generated."

    return " ".join(str(part) for part in command)


def _preview_summary_markdown(sim_type: str, form: dict[str, Any], tasks: list[Any]) -> str:
    task_count = len(tasks)
    devices = _dedupe(task.params.get("device") for task in tasks)
    qlinear = _dedupe(task.params.get("quantize_linear_action") for task in tasks)
    qattn = _dedupe(task.params.get("quantize_attention_action") for task in tasks)
    model_id = form.get("model_id") or "-"
    device_text = ", ".join(devices[:8]) if devices else form.get("device", "-")
    qlinear_text = ", ".join(qlinear[:8]) if qlinear else form.get("quantize_linear_action", "-")
    qattn_text = ", ".join(qattn[:8]) if qattn else form.get("quantize_attention_action", "-")

    lines = ["### Configuration Summary"]
    lines.append(f"- Model: **{model_id}**")
    lines.append(f"- Device: **{device_text}**")
    lines.append(f"- Estimated Tasks: **{task_count}**")

    if sim_type == "text_generate":
        num_queries = _dedupe(task.params.get("num_queries") for task in tasks)
        query_text = ", ".join(num_queries[:8]) or form.get("num_queries", "-")
        lines.append(f"- Device Count / Concurrency: **{form.get('num_devices', '-')} / {query_text}**")
        lines.append(
            f"- Sequence Length: **Context {form.get('context_length', '-')} / "
            f"Generate {form.get('query_length', '-')} token**"
        )
        lines.append(f"- Quantization: **MLP={qlinear_text} / Attention={qattn_text}**")
        mode_text = "Decode" if form.get("decode") else "Prefill"
        lines.append(
            f"- Mode: **{mode_text} / TP={form.get('tp_size', '-')} / "
            f"DP={form.get('dp_size', '-')} / EP={form.get('ep_size', '-')}**"
        )
        if form.get("image_height") or form.get("image_width"):
            image_count = form.get("image_batch_size") or "-"
            image_height = form.get("image_height") or "-"
            image_width = form.get("image_width") or "-"
            lines.append(f"- Images: **{image_count} / {image_height} x {image_width}**")
    elif sim_type == "video_generate":
        ulysses = _dedupe(task.params.get("ulysses_size") for task in tasks)
        ulysses_text = ", ".join(ulysses[:8]) or form.get("ulysses_size", "-")
        lines.append(f"- Device Count / Ulysses: **{form.get('world_size', '-')} / {ulysses_text}**")
        lines.append(
            f"- Video Shape: **{form.get('height', '-')} x {form.get('width', '-')} / "
            f"{form.get('frame_num', '-')} frames / {form.get('sample_step', '-')} steps**"
        )
        lines.append(f"- Batch / Prompt: **{form.get('batch_size', '-')} / {form.get('seq_len', '-')} tokens**")
        lines.append(f"- Quantization: **MLP={qlinear_text}**")
        cfg_text = "Enabled" if form.get("use_cfg") else "Disabled"
        cfg_parallel_text = "Enabled" if form.get("cfg_parallel") else "Disabled"
        dit_cache_text = "Enabled" if form.get("dit_cache") else "Disabled"
        lines.append(
            f"- CFG / Cache: **CFG={cfg_text} / CFG Parallel={cfg_parallel_text} / DiT Cache={dit_cache_text}**"
        )
    elif sim_type == "throughput_optimizer":
        modes = _dedupe(task.params.get("deployment_mode") for task in tasks)
        tpot = _dedupe(task.params.get("tpot_limits") for task in tasks)
        ttft = _dedupe(task.params.get("ttft_limits") for task in tasks)
        lines.append(f"- Deployment Mode: **{', '.join(modes[:4]) or form.get('deployment_mode', '-')}**")
        lines.append(f"- Device Count / Jobs: **{form.get('num_devices', '-')} / {form.get('jobs', '-')}**")
        lines.append(
            f"- Length: **Input {form.get('input_length', '-')} / Output {form.get('output_length', '-')} token**"
        )
        lines.append(
            f"- Constraints: **TTFT={', '.join(ttft[:8]) or 'unlimited'} ms / "
            f"TPOT={', '.join(tpot[:8]) or 'unlimited'} ms**"
        )
        lines.append(f"- Quantization: **MLP={qlinear_text} / Attention={qattn_text}**")
        if str(form.get("prefix_cache_hit_rate") or "0") not in {"", "0", "0.0"}:
            lines.append(f"- Prefix Cache Hit Rate: **{form.get('prefix_cache_hit_rate')}**")
    return "\n".join(lines)


def _round_numeric_columns(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if any(unit in str(col) for unit in ["(ms)", "(s)", "(GB)", "(%)", "token/s", "QPS", "Throughput"]):
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.notna().any():
                out[col] = numeric.round(digits)
    return out


def _normalize_op_columns(columns: Sequence[str] | None) -> list[str]:
    # Compatibility mapping for column name variations
    col_mapping = {
        "Avg Time (ms)": "Average Time (ms)",
    }
    normalized = []
    for col in columns or []:
        mapped = col_mapping.get(str(col), str(col))
        if mapped in OP_TABLE_COLUMNS:
            normalized.append(mapped)
    return normalized or OP_TABLE_DEFAULT_COLUMNS.copy()


def _op_table_from_records(
    op_records: list[dict[str, Any]] | None,
    device: str | None,
    top_n: int,
    columns: Sequence[str] | None = None,
    sort_by: str | None = None,
    case_label: str | None = None,
) -> pd.DataFrame:
    if not op_records:
        return pd.DataFrame()

    df = pd.DataFrame(op_records)
    if df.empty:
        return pd.DataFrame()
    if device and "device" in df.columns:
        df = df[df["device"] == device]
    if df.empty:
        return pd.DataFrame()
    if case_label:
        if "case_label" not in df.columns:
            df = df.copy()
            df["case_label"] = df.apply(_case_label_from_mapping, axis=1)
        df = df[df["case_label"] == case_label]
    if df.empty:
        return pd.DataFrame()

    if "category" not in df.columns:
        name_series = df["name"] if "name" in df.columns else pd.Series([""] * len(df), index=df.index)
        df["category"] = name_series.fillna("").astype(str).apply(_categorize_op)
    for raw_col in ["analytic_total_us", "analytic_avg_us", "num_calls"]:
        if raw_col in df.columns:
            df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")

    df["Total Time (ms)"] = df.get("analytic_total_us", 0) / 1000.0
    df["Average Time (ms)"] = df.get("analytic_avg_us", 0) / 1000.0
    df["Calls"] = df.get("num_calls", 0)
    df["Operator"] = df.get("name", "-")
    df["Category"] = df.get("category", "Other")
    df["Device"] = df.get("device", "-")

    # Compatibility mapping for sort_by
    sort_mapping = {
        "Avg Time (ms)": "Average Time (ms)",
    }
    normalized_sort_by = sort_mapping.get(sort_by, sort_by) if sort_by else None
    sort_col = normalized_sort_by if normalized_sort_by in OP_TABLE_SORT_OPTIONS else "Total Time (ms)"
    ascending = sort_col == "Operator"
    df = df.sort_values(by=sort_col, ascending=ascending, kind="stable").head(int(top_n or 20))
    display_df = df[OP_TABLE_COLUMNS].copy()
    display_df = _round_numeric_columns(display_df)
    return display_df[_normalize_op_columns(columns)].reset_index(drop=True)


# -----------------------------
# Form builders
# -----------------------------
def _build_text_form(*vals):
    """Build the text generation form payload."""
    keys = [
        "model_id",
        "device",
        "competitor_devices",
        "num_devices",
        "num_queries",
        "num_queries_list",
        "query_length",
        "context_length",
        "decode",
        "num_mtp_tokens",
        "mtp_acceptance_rate",
        "compile",
        "quantize_linear_action",
        "quant_linear_list",
        "quantize_attention_action",
        "quant_attention_list",
        "tp_size",
        "tp_list",
        "dp_size",
        "ep_size",
        "image_batch_size",
        "image_height",
        "image_width",
        "prefix_cache_hit_rate",
        "reserved_memory_gb",
        "log_level",
        "enable_multistream",
        "compile_allow_graph_break",
        "disable_repetition",
        "quantize_lmhead",
        "mxfp4_group_size",
        "graph_log_url",
        "dump_input_shapes",
        "chrome_trace",
        "num_hidden_layers_override",
        "o_proj_tp_size",
        "o_proj_dp_size",
        "mlp_tp_size",
        "mlp_dp_size",
        "lmhead_tp_size",
        "lmhead_dp_size",
        "moe_tp_size",
        "moe_dp_size",
        "word_embedding_tp",
        "enable_redundant_experts",
        "enable_external_shared_experts",
        "host_external_shared_experts",
        "enable_sequence_parallel",
        "enable_shared_expert_tp",
        "enable_dispatch_ffn_combine",
        "remote_source",
        "performance_model",
        "profiling_database",
        "export_empirical_metrics",
    ]
    data = dict(zip(keys, vals))
    data["num_queries_sweep"] = data.pop("num_queries_list")
    data["tp_sweep"] = data.pop("tp_list")
    data["quant_linear_sweep"] = data.pop("quant_linear_list")
    data["quant_attention_sweep"] = data.pop("quant_attention_list")
    return data


def _build_video_form(*vals):
    """Build the video generation form payload."""
    keys = [
        "model_id",
        "device",
        "competitor_devices",
        "batch_size",
        "seq_len",
        "height",
        "width",
        "frame_num",
        "sample_step",
        "dtype",
        "quantize_linear_action",
        "quant_linear_list",
        "world_size",
        "ulysses_size",
        "ulysses_list",
        "use_cfg",
        "cfg_parallel",
        "dit_cache",
        "cache_step_range",
        "cache_step_interval",
        "cache_block_range",
        "chrome_trace",
        "log_level",
    ]
    data = dict(zip(keys, vals))
    data["quant_linear_sweep"] = data.pop("quant_linear_list")
    data["ulysses_sweep"] = data.pop("ulysses_list")
    return data


def _build_opt_form(*vals):
    """?????????"""
    keys = [
        "model_id",
        "device",
        "competitor_devices",
        "num_devices",
        "input_length",
        "output_length",
        "compile",
        "quantize_linear_action",
        "quant_linear_list",
        "quantize_attention_action",
        "quant_attention_list",
        "tpot_limits",
        "tpot_list",
        "ttft_limits",
        "ttft_list",
        "num_mtp_tokens",
        "mtp_acceptance_rate",
        "max_batched_tokens",
        "image_batch_size",
        "image_height",
        "image_width",
        "tp_sizes",
        "ep_sizes",
        "moe_dp_sizes",
        "batch_range",
        "jobs",
        "deployment_mode",
        "prefix_cache_hit_rate",
        "prefill_devices_per_instance",
        "decode_devices_per_instance",
        "enable_multistream",
        "compile_allow_graph_break",
        "mxfp4_group_size",
        "reserved_memory_gb",
        "log_level",
        "serving_cost",
        "dump_original_results",
        "concurrency_search_strategy",
    ]
    data = dict(zip(keys, vals))
    data["quant_linear_sweep"] = data.pop("quant_linear_list")
    data["quant_attention_sweep"] = data.pop("quant_attention_list")
    data["tpot_sweep"] = data.pop("tpot_list")
    data["ttft_sweep"] = data.pop("ttft_list")
    mode = _normalize_optimizer_deployment_mode(data.get("deployment_mode"))
    data["deployment_mode"] = mode
    data["disagg"] = mode == OPT_DEPLOY_PD_SPLIT
    data["enable_optimize_prefill_decode_ratio"] = mode == OPT_DEPLOY_PD_RATIO
    return data


def _validate_text_form(form: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def positive_int(key: str, label: str) -> int | None:
        raw = form.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            errors.append(f"{label}\u5fc5\u987b\u662f\u6b63\u6574\u6570\u3002")
            return None
        if value <= 0:
            errors.append(f"{label}\u5fc5\u987b\u5927\u4e8e0\u3002")
            return None
        return value

    def non_negative_float(key: str, label: str) -> float | None:
        try:
            value = float(form.get(key) or 0.0)
        except (TypeError, ValueError):
            errors.append(f"{label}\u5fc5\u987b\u662f\u6570\u5b57\u3002")
            return None
        if value < 0:
            errors.append(f"{label}\u4e0d\u80fd\u5c0f\u4e8e0\u3002")
            return None
        return value

    num_devices = positive_int("num_devices", "\u90e8\u7f72\u5361\u6570")
    positive_int("num_queries", "\u8bf7\u6c42\u5e76\u53d1\u6570")
    query_length = positive_int("query_length", "\u751f\u6210token\u6570\u91cf")
    try:
        context_length = int(form.get("context_length") or 0)
        if context_length < 0:
            errors.append("\u4e0a\u4e0b\u6587\u957f\u5ea6\u4e0d\u80fd\u5c0f\u4e8e0\u3002")
    except (TypeError, ValueError):
        errors.append("\u4e0a\u4e0b\u6587\u957f\u5ea6\u5fc5\u987b\u662f\u6574\u6570\u3002")

    try:
        num_mtp_tokens = int(form.get("num_mtp_tokens") or 0)
    except (TypeError, ValueError):
        errors.append("MTP token \u6570\u91cf\u5fc5\u987b\u662f\u6574\u6570\u3002")
        num_mtp_tokens = 0
    if num_mtp_tokens < 0:
        errors.append("MTP token \u6570\u91cf\u4e0d\u80fd\u5c0f\u4e8e0\u3002")
    if query_length is not None and num_mtp_tokens > 0 and query_length <= num_mtp_tokens:
        errors.append("\u751f\u6210token\u6570\u91cf\u5fc5\u987b\u5927\u4e8e MTP token \u6570\u91cf\u3002")

    prefix_cache = non_negative_float("prefix_cache_hit_rate", "Prefix Cache \u547d\u4e2d\u7387")
    if prefix_cache is not None and prefix_cache >= 1:
        errors.append("Prefix Cache \u547d\u4e2d\u7387\u5fc5\u987b\u5728 [0, 1) \u8303\u56f4\u5185\u3002")
    non_negative_float("reserved_memory_gb", "\u9884\u7559\u663e\u5b58")

    for key, label in [
        ("mxfp4_group_size", "MXFP4 \u5206\u7ec4\u5927\u5c0f"),
        ("moe_dp_size", "MoE DP"),
    ]:
        positive_int(key, label)
    try:
        if int(form.get("num_hidden_layers_override") or 0) < 0:
            errors.append("\u9690\u85cf\u5c42\u6570\u91cf\u8986\u76d6\u4e0d\u80fd\u5c0f\u4e8e0\u3002")
    except (TypeError, ValueError):
        errors.append("\u9690\u85cf\u5c42\u6570\u91cf\u8986\u76d6\u5fc5\u987b\u662f\u6574\u6570\u3002")

    if num_devices:
        try:
            tp_values = parse_scalar_or_list(form.get("tp_sweep") or form.get("tp_size") or 1, int)
            ep = int(form.get("ep_size") or 1)
            dp = parse_optional_number(form.get("dp_size"), int) or 1
        except (TypeError, ValueError) as exc:
            errors.append(
                f"TP/DP/EP \u5e76\u884c\u6570\u5fc5\u987b\u4e3a\u6b63\u6574\u6570\u6216 DP=auto\uff1a{exc}\u3002"
            )
            tp_values = [1]
            ep = dp = 1
        for value, label in [(dp, "DP"), (ep, "EP")]:
            if value <= 0:
                errors.append(f"{label} \u5e76\u884c\u6570\u5fc5\u987b\u5927\u4e8e0\u3002")
            elif value > num_devices:
                errors.append(f"{label} \u5e76\u884c\u6570\u4e0d\u80fd\u5927\u4e8e\u90e8\u7f72\u5361\u6570\u3002")
            elif num_devices % value != 0:
                errors.append(
                    f"\u90e8\u7f72\u5361\u6570\u9700\u8981\u80fd\u88ab {label} \u5e76\u884c\u6570\u6574\u9664\u3002"
                )
        for tp in tp_values:
            if tp <= 0:
                errors.append("TP \u5e76\u884c\u6570\u5fc5\u987b\u5927\u4e8e0\u3002")
            elif tp > num_devices:
                errors.append("TP \u5e76\u884c\u6570\u4e0d\u80fd\u5927\u4e8e\u90e8\u7f72\u5361\u6570\u3002")
            elif num_devices % tp != 0:
                errors.append(
                    "\u90e8\u7f72\u5361\u6570\u9700\u8981\u80fd\u88ab TP \u5e76\u884c\u6570\u6574\u9664\u3002"
                )
            if tp * dp * ep > num_devices:
                errors.append(
                    f"TP={tp}, DP={dp}, EP={ep} \u7684\u7ec4\u5408"
                    "\u4e0d\u80fd\u8d85\u8fc7\u90e8\u7f72\u5361\u6570\u3002"
                )

        for key, label in [
            ("o_proj_tp_size", "O-Proj TP"),
            ("o_proj_dp_size", "O-Proj DP"),
            ("mlp_tp_size", "MLP TP"),
            ("mlp_dp_size", "MLP DP"),
            ("lmhead_tp_size", "LMHead TP"),
            ("lmhead_dp_size", "LMHead DP"),
            ("moe_tp_size", "MoE TP"),
        ]:
            raw = form.get(key)
            if raw in (None, ""):
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                errors.append(f"{label}\u5fc5\u987b\u662f\u6b63\u6574\u6570\u3002")
                continue
            if value <= 0:
                errors.append(f"{label}\u5fc5\u987b\u5927\u4e8e0\u3002")
            elif value > num_devices:
                errors.append(f"{label}\u4e0d\u80fd\u5927\u4e8e\u90e8\u7f72\u5361\u6570\u3002")
            elif num_devices % value != 0:
                errors.append(f"\u90e8\u7f72\u5361\u6570\u9700\u8981\u80fd\u88ab {label} \u6574\u9664\u3002")

    perf_models = form.get("performance_model") or []
    if isinstance(perf_models, str):
        perf_models = parse_scalar_or_list(perf_models, str)
    if "profiling" in perf_models and not form.get("profiling_database"):
        errors.append(
            "\u9009\u62e9 profiling \u6027\u80fd\u6a21\u578b\u65f6"
            "\u9700\u8981\u586b\u5199 Profiling \u6570\u636e\u5e93\u8def\u5f84\u3002"
        )

    return errors


def _validate_video_form(form: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def positive_int(key: str, label: str) -> int | None:
        try:
            value = int(form.get(key))
        except (TypeError, ValueError):
            errors.append(f"{label} must be a positive integer.")
            return None
        if value <= 0:
            errors.append(f"{label} must be greater than 0.")
            return None
        return value

    for key, label in [
        ("batch_size", "Batch Size"),
        ("seq_len", "Prompt Length"),
        ("height", "Height"),
        ("width", "Width"),
        ("frame_num", "Frame Count"),
        ("sample_step", "Sample Step"),
    ]:
        positive_int(key, label)

    world_size = positive_int("world_size", "Device Count")
    if world_size:
        try:
            ulysses_values = parse_scalar_or_list(form.get("ulysses_sweep") or form.get("ulysses_size"), int)
        except Exception as exc:
            errors.append(f"Failed to parse Ulysses list: {exc}")
            ulysses_values = []
        for ulysses in ulysses_values:
            if ulysses <= 0:
                errors.append("Ulysses parallel size must be greater than 0.")
            elif ulysses > world_size:
                errors.append("Ulysses parallel size cannot be greater than Device Count.")
            elif world_size % ulysses != 0:
                errors.append("Device Count must be divisible by Ulysses parallel size.")

    if form.get("cache_step_interval") not in (None, ""):
        positive_int("cache_step_interval", "Cache Step Interval")
    for key, label in [
        ("cache_step_range", "Cache Step Range"),
        ("cache_block_range", "Cache Block Range"),
    ]:
        raw = str(form.get(key) or "").strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            errors.append(f"{label} must use the format start,end.")
            continue
        try:
            start, end = [int(p) for p in parts]
        except ValueError:
            errors.append(f"{label} must contain integers only.")
            continue
        if start < 0 or end <= start:
            errors.append(f"{label} must satisfy 0 <= start < end.")

    model_id = str(form.get("model_id") or "").strip()
    if not model_id:
        errors.append("model-id cannot be empty.")
    else:
        model_dir = Path(model_id)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_id
        if not model_dir.is_dir():
            errors.append(
                "Video Models currently requires a local Diffusers model directory, for example "
                "tests/assets/model_config/Wan2.2-T2V-A14B-Diffusers."
            )
        elif not (model_dir / "transformer" / "config.json").is_file():
            errors.append("Video Models requires a Diffusers model directory that contains transformer/config.json.")

    return errors


def _validate_optimizer_form(form: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require_positive_int(key: str, label: str) -> int | None:
        raw = form.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            errors.append(f"{label}\u5fc5\u987b\u662f\u6b63\u6574\u6570\u3002")
            return None
        if value <= 0:
            errors.append(f"{label}\u5fc5\u987b\u5927\u4e8e0\u3002")
            return None
        return value

    def non_negative_float(key: str, label: str) -> float | None:
        try:
            value = float(form.get(key) or 0.0)
        except (TypeError, ValueError):
            errors.append(f"{label}\u5fc5\u987b\u662f\u6570\u5b57\u3002")
            return None
        if value < 0:
            errors.append(f"{label}\u4e0d\u80fd\u5c0f\u4e8e0\u3002")
            return None
        return value

    num_devices = require_positive_int("num_devices", "\u90e8\u7f72\u5361\u6570")
    for key, label in [
        ("input_length", "\u8f93\u5165\u957f\u5ea6"),
        ("output_length", "\u8f93\u51fa\u957f\u5ea6"),
        ("jobs", "\u5e76\u884c\u4efb\u52a1\u6570"),
        ("max_batched_tokens", "max-batched-tokens"),
        ("mxfp4_group_size", "MXFP4 \u5206\u7ec4\u5927\u5c0f"),
    ]:
        require_positive_int(key, label)

    prefix_cache_hit_rate = non_negative_float("prefix_cache_hit_rate", "Prefix Cache Hit Rate")
    if prefix_cache_hit_rate is not None and prefix_cache_hit_rate >= 1:
        errors.append("Prefix Cache Hit Rate must be in the range [0, 1).")
    non_negative_float("reserved_memory_gb", "Reserved Memory")
    non_negative_float("serving_cost", "Serving cost")

    mode = _normalize_optimizer_deployment_mode(form.get("deployment_mode"))
    if mode not in {OPT_DEPLOY_PD_MIXED, OPT_DEPLOY_PD_SPLIT, OPT_DEPLOY_PD_RATIO}:
        errors.append("Invalid deployment mode.")

    if num_devices:
        raw_tp = str(form.get("tp_sizes") or "").strip()
        if raw_tp:
            try:
                tp_sizes = parse_scalar_or_list(raw_tp, int)
            except Exception as exc:
                errors.append(f"TP\u5e76\u884c\u5927\u5c0f\u5217\u8868\u89e3\u6790\u5931\u8d25\uff1a{exc}")
                tp_sizes = []
            for tp in tp_sizes:
                if tp <= 0:
                    errors.append("TP\u5e76\u884c\u5927\u5c0f\u5fc5\u987b\u5927\u4e8e0\u3002")
                elif tp > num_devices:
                    errors.append("TP\u5e76\u884c\u5927\u5c0f\u4e0d\u80fd\u5927\u4e8e\u90e8\u7f72\u5361\u6570\u3002")
                elif num_devices % tp != 0:
                    errors.append(
                        "\u90e8\u7f72\u5361\u6570\u9700\u8981\u80fd\u88ab TP\u5e76\u884c\u5927\u5c0f\u6574\u9664\u3002"
                    )

    raw_batch = str(form.get("batch_range") or "").strip()
    if raw_batch:
        try:
            batch_range = parse_scalar_or_list(raw_batch, int)
        except Exception as exc:
            errors.append(f"\u6279\u5927\u5c0f\u8303\u56f4\u89e3\u6790\u5931\u8d25\uff1a{exc}")
            batch_range = []
        if len(batch_range) not in {1, 2}:
            errors.append("\u6279\u5927\u5c0f\u8303\u56f4\u9700\u8981\u662f [max] \u6216 [min,max]\u3002")
        if any(v <= 0 for v in batch_range):
            errors.append("\u6279\u5927\u5c0f\u8303\u56f4\u5fc5\u987b\u5927\u4e8e0\u3002")
        if len(batch_range) == 2 and batch_range[1] < batch_range[0]:
            errors.append(
                "\u6279\u5927\u5c0f\u8303\u56f4\u7684\u6700\u5927\u503c"
                "\u9700\u8981\u4e0d\u5c0f\u4e8e\u6700\u5c0f\u503c\u3002"
            )

    if mode == OPT_DEPLOY_PD_RATIO:
        for key, label in [
            ("prefill_devices_per_instance", "Prefill Devices per Instance"),
            ("decode_devices_per_instance", "Decode Devices per Instance"),
        ]:
            raw = form.get(key)
            if raw in (None, ""):
                errors.append(f"{label} is required in PD Ratio mode.")
                continue
            value = require_positive_int(key, label)
            if value and num_devices:
                if value > num_devices:
                    errors.append(f"{label} cannot be greater than Device Count.")
                elif num_devices % value != 0:
                    errors.append(f"Device Count must be divisible by {label}.")

    return errors


def _optimizer_validation_markdown(errors: list[str]) -> str:
    lines = ["### Parameter Validation Failed"]
    lines.extend(f"- {msg}" for msg in errors)
    return "\n".join(lines)


def _optimizer_empty_outputs(summary: str, detail_md: str | None = None):
    empty_df = pd.DataFrame()
    return (
        progress_html(0, 1, "Parameter Validation Failed", "validation"),
        summary,
        empty_plot("Best Throughput by Device"),
        empty_plot("Best TTFT by Device"),
        empty_plot("Best TPOT by Device"),
        empty_plot("Best Batch Size Comparison"),
        empty_plot("Prefill / Decode QPS Comparison"),
        empty_df,
        gr.update(choices=[], value=None),
        "\n".join(["### Fixed-Config Comparison", "No results available."]),
        empty_plot("Fixed-Config Throughput Comparison"),
        empty_df,
        gr.update(choices=[], value=None),
        detail_md or "\n".join(["### Single-Device Search Details", "No results available."]),
        empty_plot("Single-Device Pareto Frontier"),
        empty_df,
        "",
        empty_df,
        [],
        [],
        [],
    )


def _text_validation_empty_outputs(summary: str):
    from .charts import empty_pie_plot

    empty_df = pd.DataFrame()
    return (
        progress_html(0, 1, "\u53c2\u6570\u6821\u9a8c\u5931\u8d25", "validation"),
        summary,
        empty_plot("TPS/Device \u5bf9\u6bd4"),
        empty_plot("\u5e76\u53d1\u6570 vs \u63a8\u7406\u65f6\u95f4"),
        gr.update(value="", visible=False),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        empty_pie_plot("\u663e\u5b58\u5360\u7528\u5206\u5e03"),
        empty_df,
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        empty_df,
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        empty_df,
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        empty_plot("\u7b97\u5b50\u5206\u7c7b\u8017\u65f6\u5206\u5e03"),
        empty_df,
        gr.update(),
        empty_df,
        empty_df,
        [],
        [],
        [],
        "",
        [],
    )


def _video_validation_empty_outputs(summary: str):
    empty_df = pd.DataFrame()
    return (
        progress_html(0, 1, "\u53c2\u6570\u6821\u9a8c\u5931\u8d25", "validation"),
        summary,
        empty_plot("\u603b\u5206\u6790\u65f6\u95f4\u5bf9\u6bd4"),
        empty_plot("\u901a\u4fe1\u65f6\u95f4\u5bf9\u6bd4"),
        gr.update(choices=[], value=None),
        empty_df,
        empty_plot("\u7b97\u5b50\u5206\u7c7b\u8017\u65f6\u5206\u5e03"),
        empty_df,
        empty_df,
        empty_df,
        [],
        [],
        [],
    )


def _results_to_df(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convert result records to a dataframe."""
    rows = [r.to_row() for r in results]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# -----------------------------
# Summary helpers
# -----------------------------
def _summary_markdown(df: pd.DataFrame, latest: ExperimentResult | None, sim_type: str) -> str:
    """Build the summary markdown."""
    if df.empty:
        return "### Summary\nNo results available."

    lines = ["### Summary"]
    lines.append(f"- Completed Runs: **{len(df)}**")
    if "device" in df.columns:
        devices = sorted({str(v) for v in df["device"].dropna().tolist()})
        lines.append(f"- Devices Covered: **{', '.join(devices[:12])}**")
    if latest is not None:
        lines.append(f"- Latest Completed Task: **{latest.label}**")
        lines.append(f"- Latest Source: **{latest.source}**")
        if latest.error:
            lines.append(f"- Latest Error: `{latest.error}`")

    if sim_type == "text_generate" and "tps_per_device" in df.columns:
        best = df.dropna(subset=["tps_per_device"])
        if not best.empty:
            top = best.sort_values(by="tps_per_device", ascending=False).iloc[0]
            lines.append(
                f"- Highest TPS/Device: **{top['tps_per_device']:.2f} token/s**, "
                f"device **{top.get('device', '-')}**, concurrency **{top.get('num_queries', '-')}**"
            )
    elif sim_type == "video_generate" and "analytic_total_time_s" in df.columns:
        best = df.dropna(subset=["analytic_total_time_s"])
        if not best.empty:
            top = best.sort_values(by="analytic_total_time_s", ascending=True).iloc[0]
            lines.append(
                f"- Lowest Analytic Time: **{top['analytic_total_time_s']:.3f} s**, device **{top.get('device', '-')}**"
            )
    elif sim_type == "throughput_optimizer":
        if latest is not None and latest.summary.get("no_result_reason"):
            lines.append(f"- Current Result Note: **{latest.summary.get('no_result_reason')}**")
        best = df.dropna(subset=["best_throughput"]) if "best_throughput" in df.columns else pd.DataFrame()
        if not best.empty:
            top = best.sort_values(by="best_throughput", ascending=False).iloc[0]
            lines.append(
                f"- Best Throughput: **{top['best_throughput']:.2f} token/s**, "
                f"device **{top.get('device', '-')}**, "
                f"parallel mode **{top.get('best_parallel', '-')}**, "
                f"batch size **{top.get('best_batch_size', '-')}**, "
                f"concurrency **{top.get('best_concurrency', '-')}**"
            )
        else:
            no_result_series = (
                df["no_result_reason"].dropna() if "no_result_reason" in df.columns else pd.Series(dtype=object)
            )
            if not no_result_series.empty:
                lines.append(f"- Feasibility Note: **{no_result_series.iloc[0]}**")
            else:
                lines.append(
                    "- Feasibility Note: **No valid optimization result "
                    "was found. Check whether the constraints are too "
                    "strict or whether the log is empty.**"
                )
    return "\n".join(lines)


# -----------------------------
# Display helpers
# -----------------------------
def _optimizer_deployment_mode(row: pd.Series) -> str:
    mode = row.get("deployment_mode")
    if isinstance(mode, str) and mode.strip():
        return _normalize_optimizer_deployment_mode(mode)
    if bool(row.get("enable_optimize_prefill_decode_ratio", False)) or pd.notna(row.get("pd_ratio")):
        return OPT_DEPLOY_PD_RATIO
    if bool(row.get("disagg", False)):
        return OPT_DEPLOY_PD_SPLIT
    return OPT_DEPLOY_PD_MIXED


def _optimizer_primary_metric(df: pd.DataFrame) -> tuple[str, str, str]:
    if "balanced_qps" in df.columns and df["balanced_qps"].notna().any():
        return "balanced_qps", "Balanced QPS Comparison", "Balanced QPS"
    return "best_batch_size", "Best Batch Size Comparison", "Batch Size"


def _simplify_optimizer_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact optimizer summary table for cross-device comparison."""
    if df.empty:
        return df

    slim = df.copy()
    slim["deployment_mode"] = slim.apply(_optimizer_deployment_mode, axis=1)
    preferred_cols = [
        "model_id",
        "device",
        "deployment_mode",
        "input_length",
        "output_length",
        "prefix_cache_hit_rate",
        "ttft_limits_ms",
        "tpot_limits_ms",
        "quantize_linear_action",
        "quantize_attention_action",
        "best_parallel",
        "best_batch_size",
        "best_concurrency",
        "best_throughput",
        "best_ttft_ms",
        "best_tpot_ms",
        "balanced_qps",
        "pd_ratio",
        "prefill_devices_per_instance",
        "decode_devices_per_instance",
        "status",
        "no_result_reason",
    ]
    existing = [c for c in preferred_cols if c in slim.columns]
    slim = slim[existing].copy()
    rename_map = {
        "model_id": "Model",
        "device": "Device",
        "deployment_mode": "Deployment Mode",
        "input_length": "Input Length",
        "output_length": "Output Length",
        "prefix_cache_hit_rate": "Prefix Cache Hit Rate",
        "ttft_limits_ms": "TTFT Limit (ms)",
        "tpot_limits_ms": "TPOT Limit (ms)",
        "quantize_linear_action": "MLP Quantization Mode",
        "quantize_attention_action": "Attention Quantization Mode",
        "best_parallel": "Best Parallel Mode",
        "best_batch_size": "Best Batch Size",
        "best_concurrency": "Best Concurrency",
        "best_throughput": "Best Throughput (token/s)",
        "best_ttft_ms": "Best TTFT (ms)",
        "best_tpot_ms": "Best TPOT (ms)",
        "balanced_qps": "Balanced QPS",
        "pd_ratio": "PD Ratio",
        "prefill_devices_per_instance": "Prefill Devices per Instance",
        "decode_devices_per_instance": "Decode Devices per Instance",
        "status": "Status",
        "no_result_reason": "Result Note",
    }
    return _round_numeric_columns(slim.rename(columns=rename_map))


def _display_df_for_sim(sim_type: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return the display dataframe for the given simulation type."""
    if sim_type == "throughput_optimizer":
        return _simplify_optimizer_display_df(df)
    return df


def _optimizer_pareto_chart(candidate_df: pd.DataFrame, device: str) -> Any:
    if candidate_df.empty:
        return empty_plot("Single-Device Pareto Frontier")

    work_df = candidate_df.copy().dropna(subset=["ttft_ms", "throughput_token_s"])
    if work_df.empty:
        return empty_plot("Single-Device Pareto Frontier")

    work_df["label"] = work_df.apply(
        lambda row: (f"{row.get('parallel', '-')} | B{row.get('batch_size', '-')} | C{row.get('concurrency', '-')}"),
        axis=1,
    )

    # Find rank 1 (best candidate)
    if "rank" in work_df.columns:
        rank1_idx = work_df[work_df["rank"] == 1].index
    else:
        # Fallback: find best throughput
        best_idx = work_df["throughput_token_s"].idxmax()
        rank1_idx = [best_idx]

    # Mark rank 1 as Pareto Frontier, others as Candidate Configurations
    work_df["series"] = "Candidate Configurations"
    work_df.loc[rank1_idx, "series"] = "Pareto Frontier"

    return scatter_plot(
        work_df,
        "ttft_ms",
        "throughput_token_s",
        f"Single-Device Pareto Frontier - {device}",
        "Throughput (token/s)",
        xlabel="TTFT (ms)",
        group="series",
        annotate="label",
    )


def _optimizer_state_rows(results: list[ExperimentResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = result.to_row()
        row["top_configs"] = (result.tables or {}).get("top_configs") or []
        row["raw_log"] = result.raw_log or ""
        rows.append(row)
    return rows


def _format_metric_value(value: Any, digits: int = 2) -> str:
    if value in (None, "") or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_int_value(value: Any) -> str:
    if value in (None, "") or pd.isna(value):
        return "-"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _format_limit_value(value: Any) -> str:
    if value in (None, "") or pd.isna(value):
        return "None ms"
    try:
        return f"{float(value):.2f} ms"
    except (TypeError, ValueError):
        return f"{value} ms"


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def _sep() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def _line(values: list[str]) -> str:
        return "|" + "|".join(f" {str(value).ljust(widths[idx])} " for idx, value in enumerate(values)) + "|"

    parts = [_sep(), _line(headers), _sep()]
    parts.extend(_line(row) for row in rows)
    parts.append(_sep())
    return "\n".join(parts)


def _optimizer_cli_style_output(top: pd.Series | None, device_candidates: pd.DataFrame, current_device: str) -> str:
    if top is None:
        return ""

    deployment_mode = top.get("deployment_mode", "-")
    if deployment_mode == "-":
        deployment_mode = _optimizer_deployment_mode(top)
    section_name = {
        OPT_DEPLOY_PD_MIXED: "PD Aggregated",
        OPT_DEPLOY_PD_SPLIT: "PD Disaggregated",
        OPT_DEPLOY_PD_RATIO: "PD Ratio",
    }.get(deployment_mode, deployment_mode)

    lines = [
        "********************************************************************************",
        "  ---------------------------------------------------------------------------- ",
        "  Input Configuration:",
        f"    Model: {top.get('model_id', '-')}",
        f"    Quantize Linear action: {top.get('quantize_linear_action', '-')}",
        f"    Quantize Attention action: {top.get('quantize_attention_action', '-')}",
        f"    Devices: {_format_int_value(top.get('num_devices'))} {current_device}",
        f"    Input Length: {_format_int_value(top.get('input_length'))}",
        f"    Output Length: {_format_int_value(top.get('output_length'))}",
        f"    TTFT Limits: {_format_limit_value(top.get('ttft_limits_ms'))}",
        f"    TPOT Limits: {_format_limit_value(top.get('tpot_limits_ms'))}",
        "  ---------------------------------------------------------------------------- ",
        "  Overall Best Configuration:",
        "    Best Throughput: "
        f"{_format_metric_value(top.get('best_throughput', top.get('throughput_token_s')))} "
        "tokens/s",
        f"    TTFT: {_format_metric_value(top.get('best_ttft_ms', top.get('ttft_ms')))} ms",
        f"    TPOT: {_format_metric_value(top.get('best_tpot_ms', top.get('tpot_ms')))} ms",
        "  ---------------------------------------------------------------------------- ",
    ]

    display_rows: list[list[str]] = []
    if not device_candidates.empty:
        for rank_idx, (_, row) in enumerate(device_candidates.iterrows(), start=1):
            display_rows.append(
                [
                    _format_int_value(row.get("rank") or rank_idx),
                    _format_metric_value(row.get("throughput_token_s")),
                    _format_metric_value(row.get("ttft_ms")),
                    _format_metric_value(row.get("tpot_ms")),
                    _format_int_value(row.get("concurrency")),
                    _format_int_value(row.get("num_devices", top.get("num_devices"))),
                    str(row.get("parallel", "-")),
                    _format_int_value(row.get("batch_size")),
                ]
            )
        lines.append(f"Top {len(display_rows)} {section_name} Configurations:")
        lines.append(
            _ascii_table(
                [
                    "Top",
                    "Throughput (token/s)",
                    "TTFT (ms)",
                    "TPOT (ms)",
                    "concurrency",
                    "num_devices",
                    "parallel",
                    "batch_size",
                ],
                display_rows,
            )
        )

    raw_log = ANSI_RE.sub("", str(top.get("raw_log", "") or "")).strip()
    if raw_log:
        lines.extend(["", "Raw CLI Output:", raw_log])

    lines.append("********************************************************************************")
    return "\n".join(lines)


def _optimizer_detail_view(
    summary_rows: list[dict[str, Any]] | None,
    candidate_rows: list[dict[str, Any]] | None,
    device: str | None,
):
    """Render single-device optimizer drill-down details."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    summary_df = _safe_df_from_rows(summary_rows)
    candidate_df = _safe_df_from_rows(candidate_rows)

    device_pool: list[str] = []
    if not summary_df.empty and "device" in summary_df.columns:
        device_pool.extend(summary_df["device"].dropna().astype(str).tolist())
    if not candidate_df.empty and "device" in candidate_df.columns:
        device_pool.extend(candidate_df["device"].dropna().astype(str).tolist())
    devices = sorted({d for d in device_pool if d})
    if not devices:
        return (
            gr.update(choices=[], value=None),
            "### Single-Device Search Details\nNo results available.",
            empty_plot("Single-Device Pareto Frontier"),
            pd.DataFrame(),
            "",
        )

    current_device = device if device in devices else devices[0]
    device_summary = (
        summary_df[summary_df["device"].astype(str) == current_device].copy()
        if not summary_df.empty and "device" in summary_df.columns
        else pd.DataFrame()
    )
    device_candidates = (
        candidate_df[candidate_df["device"].astype(str) == current_device].copy()
        if not candidate_df.empty and "device" in candidate_df.columns
        else pd.DataFrame()
    )

    top = None
    if not device_summary.empty:
        device_summary["deployment_mode"] = device_summary.apply(_optimizer_deployment_mode, axis=1)
        sort_col = (
            "balanced_qps"
            if "balanced_qps" in device_summary.columns and device_summary["balanced_qps"].notna().any()
            else "best_throughput"
        )
        if sort_col in device_summary.columns:
            device_summary = device_summary.sort_values(by=sort_col, ascending=False, kind="stable")
        top = device_summary.iloc[0]
    elif not device_candidates.empty:
        device_candidates = device_candidates.sort_values(
            by=["throughput_token_s", "ttft_ms"], ascending=[False, True], kind="stable"
        )
        top = device_candidates.iloc[0]
    else:
        return (
            gr.update(choices=devices, value=current_device),
            "### Single-Device Search Details\nNo results available.",
            empty_plot("Single-Device Pareto Frontier"),
            pd.DataFrame(),
            "",
        )

    pareto_chart = empty_plot("Single-Device Pareto Frontier")
    detail_df = pd.DataFrame()
    if not device_candidates.empty:
        device_candidates = device_candidates.sort_values(
            by=["throughput_token_s", "ttft_ms"], ascending=[False, True], kind="stable"
        )
        pareto_chart = _optimizer_pareto_chart(device_candidates, current_device)
        detail_cols = [
            col
            for col in [
                "rank",
                "deployment_mode",
                "parallel",
                "batch_size",
                "concurrency",
                "throughput_token_s",
                "ttft_ms",
                "tpot_ms",
                "num_devices",
                "model_id",
                "input_length",
                "output_length",
                "prefix_cache_hit_rate",
                "quantize_linear_action",
                "quantize_attention_action",
            ]
            if col in device_candidates.columns
        ]
        detail_df = (
            device_candidates[detail_cols]
            .copy()
            .rename(
                columns={
                    "rank": "Rank",
                    "deployment_mode": "Deployment Mode",
                    "parallel": "Parallel Mode",
                    "batch_size": "Batch Size",
                    "concurrency": "Concurrency",
                    "throughput_token_s": "Throughput (token/s)",  # nosec B105
                    "ttft_ms": "TTFT(ms)",
                    "tpot_ms": "TPOT(ms)",
                    "num_devices": "Device Count",
                    "model_id": "Model",
                    "input_length": "Input Length",
                    "output_length": "Output Length",
                    "prefix_cache_hit_rate": "Prefix Cache Hit Rate",
                    "quantize_linear_action": "MLP Quantization Mode",
                    "quantize_attention_action": "Attention Quantization Mode",
                }
            )
        )

    raw_output = _optimizer_cli_style_output(top, device_candidates, current_device)

    if pd.isna(top.get("best_throughput")) and pd.isna(top.get("balanced_qps")) and top.get("no_result_reason"):
        md = (
            f"### Single-Device Search Details\n"
            f"- Current Device: **{current_device}**\n"
            f"- Deployment Mode: **{top.get('deployment_mode', '-')}**\n"
            f"- Current Result: **No feasible deployment plan found**\n"
            f"- Reason: **{top.get('no_result_reason')}**"
        )
    else:
        deployment_mode = top.get("deployment_mode", "-")
        if deployment_mode == "-" and not device_summary.empty:
            deployment_mode = _optimizer_deployment_mode(top)
        md_lines = [
            "### Single-Device Search Details",
            f"- Current Device: **{current_device}**",
            f"- Deployment Mode: **{deployment_mode}**",
            f"- Input / Output Length: **"
            f"{_format_int_value(top.get('input_length'))} / "
            f"{_format_int_value(top.get('output_length'))}**",
            f"- Best Parallel Mode: **{top.get('best_parallel', top.get('parallel', '-'))}**",
            f"- Best Batch Size: **{top.get('best_batch_size', top.get('batch_size', '-'))}**",
            f"- Best Concurrency: **{top.get('best_concurrency', top.get('concurrency', '-'))}**",
            f"- Best Throughput: **"
            f"{float(top.get('best_throughput', top.get('throughput_token_s', 0)) or 0):.2f} "
            f"token/s**",
            f"- Best TTFT: **{float(top.get('best_ttft_ms', top.get('ttft_ms', 0)) or 0):.2f} ms**",
            f"- Best TPOT: **{float(top.get('best_tpot_ms', top.get('tpot_ms', 0)) or 0):.2f} ms**",
        ]
        prefix_cache = top.get("prefix_cache_hit_rate")
        if prefix_cache is not None and not pd.isna(prefix_cache) and float(prefix_cache) > 0:
            md_lines.append(f"- Prefix Cache Hit Rate: **{float(prefix_cache):.2f}**")
        if pd.notna(top.get("balanced_qps")):
            md_lines.append(f"- Balanced QPS: **{float(top.get('balanced_qps', 0) or 0):.2f}**")
        if pd.notna(top.get("pd_ratio")):
            md_lines.append(f"- PD Ratio: **{float(top.get('pd_ratio', 0) or 0):.2f}**")
        if pd.notna(top.get("prefill_devices_per_instance")) and pd.notna(top.get("decode_devices_per_instance")):
            prefill_devices = int(top.get("prefill_devices_per_instance"))
            decode_devices = int(top.get("decode_devices_per_instance"))
            md_lines.append(f"- Prefill/Decode Devices per Instance: **{prefill_devices}:{decode_devices}**")
        if not device_candidates.empty:
            md_lines.append(f"- Top Candidate Count: **{len(device_candidates)}**")
            md_lines.append(
                "- The table below shows the top results for this device, "
                "and the textbox keeps a CLI-style raw output view."
            )
        md = "\n".join(md_lines)

    return (
        gr.update(choices=devices, value=current_device),
        md,
        pareto_chart,
        detail_df,
        raw_output,
    )


# -----------------------------
# Common outputs
# -----------------------------
# -----------------------------
def _common_outputs(sim_type: str, results: list[ExperimentResult], latest: ExperimentResult | None):
    """Generate shared outputs."""
    full_df = _results_to_df(results)
    display_df = _display_df_for_sim(sim_type, full_df)
    summary = _summary_markdown(full_df, latest, sim_type)
    fig1, fig2, fig3 = make_figures(sim_type, full_df, latest)
    base_fig, base_update, base_df = baseline_plot(sim_type, df_to_records(full_df), None)
    return (
        summary,
        fig1,
        fig2,
        fig3,
        base_update,
        base_fig,
        base_df,
        display_df,
        df_to_records(display_df),
        df_to_records(full_df),
    )


# -----------------------------
# Run tasks
# -----------------------------
def _run_tasks(sim_type: str, tasks):
    """Run tasks."""
    results: list[ExperimentResult] = []
    total = len(tasks)
    for completed, _, result in RUNNER.run_matrix(tasks):
        results.append(result)
        (
            summary,
            fig1,
            fig2,
            fig3,
            base_update,
            base_fig,
            base_df,
            display_df,
            display_rows,
            full_rows,
        ) = _common_outputs(sim_type, results, result)
        progress = progress_html(completed, total, result.label, f"{result.status} / {result.source}")
        yield (
            progress,
            summary,
            fig1,
            fig2,
            fig3,
            base_update,
            base_fig,
            base_df,
            display_df,
            display_rows,
            full_rows,
        )


def _stop_run_feedback(title: str) -> tuple[str, str]:
    stopped = RUNNER.stop_all()
    progress = progress_html(0, 1, title, f"cancelled {stopped} task(s)")
    summary = "\n".join(
        [
            "### Run Cancelled",
            f"- Requested stop for **{stopped}** active task(s).",
            "- You can adjust parameters and start a new run at any time.",
        ]
    )
    return progress, summary


def stop_text_generate_run():
    return _stop_run_feedback("Text Generate Cancelled")


def stop_video_generate_run():
    return _stop_run_feedback("Video Generate Cancelled")


def stop_optimizer_run():
    return _stop_run_feedback("Optimizer Cancelled")


# -----------------------------
# Text generate callbacks
# -----------------------------
def run_text_generate(*vals):
    """Run text generation simulation."""
    tasks = build_text_generate_tasks(_build_text_form(*vals))
    yield from _run_tasks("text_generate", tasks)


def preview_text_generate(*vals):
    """Preview the text generation summary and command."""
    try:
        form = _build_text_form(*vals)
        errors = _validate_text_form(form)
        if errors:
            return _optimizer_validation_markdown(errors), ""
        tasks = build_text_generate_tasks(form)
    except Exception as exc:
        return _format_preview_error(exc)
    return _preview_summary_markdown("text_generate", form, tasks), _preview_first_command(tasks)


# -----------------------------
# Video generate callbacks
# -----------------------------
def run_video_generate(*vals):
    """Run video generation simulation."""
    form = _build_video_form(*vals)
    errors = _validate_video_form(form)
    if errors:
        yield _video_validation_empty_outputs(_optimizer_validation_markdown(errors))
        return
    tasks = build_video_generate_tasks(form)
    yield from _run_tasks("video_generate", tasks)


def preview_video_generate(*vals):
    """Preview the video generation summary and command."""
    try:
        form = _build_video_form(*vals)
        errors = _validate_video_form(form)
        if errors:
            return _optimizer_validation_markdown(errors), ""
        tasks = build_video_generate_tasks(form)
    except Exception as exc:
        return _format_preview_error(exc)
    return _preview_summary_markdown("video_generate", form, tasks), _preview_first_command(tasks)


# -----------------------------
# Optimizer callbacks
# -----------------------------
def run_optimizer(*vals):
    """Run the optimizer."""
    tasks = build_optimizer_tasks(_build_opt_form(*vals))
    results: list[ExperimentResult] = []
    total = len(tasks)
    for completed, _, result in RUNNER.run_matrix(tasks):
        results.append(result)
        (
            summary,
            fig1,
            fig2,
            fig3,
            base_update,
            base_fig,
            base_df,
            display_df,
            display_rows,
            full_rows,
        ) = _common_outputs("throughput_optimizer", results, result)
        detail_update, detail_md, _detail_pareto_chart, detail_df, _detail_output = _optimizer_detail_view(
            full_rows, None, None
        )
        progress = progress_html(completed, total, result.label, f"{result.status} / {result.source}")
        yield (
            progress,
            summary,
            fig1,
            fig2,
            fig3,
            base_update,
            base_fig,
            base_df,
            detail_update,
            detail_md,
            detail_df,
            display_df,
            display_rows,
            full_rows,
        )


def preview_optimizer(*vals):
    """Preview the optimizer summary and command."""
    form = _build_opt_form(*vals)
    errors = _validate_optimizer_form(form)
    if errors:
        return _optimizer_validation_markdown(errors), ""
    try:
        tasks = build_optimizer_tasks(form)
    except Exception as exc:
        return _format_preview_error(exc)
    return _preview_summary_markdown("throughput_optimizer", form, tasks), _preview_first_command(tasks)


# -----------------------------
# History callbacks
# -----------------------------
def load_history_for_sim(sim_type: str):
    """Load history results for the given simulation type."""
    rows = STORE.query_rows(sim_type)
    full_df = _safe_df_from_rows(rows)
    display_df = _display_df_for_sim(sim_type, full_df)
    summary = _summary_markdown(full_df, None, sim_type)
    fig1, fig2, fig3 = make_figures(sim_type, full_df, None)
    base_fig, base_update, base_df = baseline_plot(sim_type, df_to_records(full_df), None)
    progress = progress_html(
        len(full_df),
        len(full_df) if len(full_df) > 0 else 1,
        "History Loaded",
        "history",
    )
    return (
        progress,
        summary,
        fig1,
        fig2,
        fig3,
        base_update,
        base_fig,
        base_df,
        display_df,
        df_to_records(display_df),
        df_to_records(full_df),
    )


def load_optimizer_history():
    """Load optimizer history results."""
    (
        progress,
        summary,
        fig1,
        fig2,
        fig3,
        base_update,
        base_fig,
        base_df,
        display_df,
        display_rows,
        full_rows,
    ) = load_history_for_sim("throughput_optimizer")
    detail_update, detail_md, _detail_pareto_chart, detail_df, _detail_output = _optimizer_detail_view(
        full_rows, None, None
    )
    return (
        progress,
        summary,
        fig1,
        fig2,
        fig3,
        base_update,
        base_fig,
        base_df,
        detail_update,
        detail_md,
        _detail_pareto_chart,
        detail_df,
        display_df,
        display_rows,
        full_rows,
    )


# -----------------------------
# Refresh callbacks
# -----------------------------
def refresh_baseline_view(sim_type: str, rows: list[dict[str, Any]] | None, baseline_device: str | None):
    """Refresh the baseline views."""
    fig, update, df = baseline_plot(sim_type, rows, baseline_device)
    return fig, df


def refresh_optimizer_detail(rows: list[dict[str, Any]] | None, device: str | None):
    """Refresh optimizer details."""
    _update, md, _pareto, df, _raw = _optimizer_detail_view(rows, None, device)
    return md, df


def load_compare_rows():
    """Load comparison rows."""
    return pd.DataFrame(STORE.query_rows())


# -----------------------------
# Text Generate callbacks
# -----------------------------
def _categorize_op(op_name: str) -> str:
    """Classify operators."""
    op_lower = op_name.lower()
    if "attention" in op_lower or "attn" in op_lower:
        return "Attention"
    elif "linear" in op_lower or "gemm" in op_lower or "matmul" in op_lower:
        return "Linear"
    elif "all_reduce" in op_lower or "all_gather" in op_lower or "all_to_all" in op_lower:
        return "Communication"
    elif "moe" in op_lower or "expert" in op_lower:
        return "MoE"
    elif "norm" in op_lower or "layernorm" in op_lower or "rmsnorm" in op_lower:
        return "Normalization"
    elif "embedding" in op_lower:
        return "Embedding"
    elif "softmax" in op_lower or "silu" in op_lower or "gelu" in op_lower or "activation" in op_lower:
        return "Activation"
    elif "add" in op_lower or "mul" in op_lower or "div" in op_lower:
        return "Elementwise"
    elif "copy" in op_lower or "index" in op_lower or "slice" in op_lower or "reshape" in op_lower:
        return "Memory"
    else:
        return "Other"


def _text_generate_op_summary(results: list[ExperimentResult]) -> list[dict[str, Any]]:
    """Collect operator data from all results."""
    all_ops = []
    for result in results:
        device = result.params.get("device", "unknown")
        case_label = _case_label_from_mapping(result.params)
        op_breakdown = result.tables.get("op_breakdown", [])
        for op in op_breakdown:
            op["device"] = device
            op["num_queries"] = result.params.get("num_queries")
            op["tp_size"] = result.params.get("tp_size")
            op["case_label"] = case_label
            op["category"] = _categorize_op(op.get("name", ""))
            all_ops.append(op)
    return all_ops


def _text_generate_op_table(
    results: list[ExperimentResult],
    device: str | None,
    top_n: int,
    columns: Sequence[str] | None = None,
    sort_by: str | None = None,
    case_label: str | None = None,
) -> pd.DataFrame:
    """Build the operator time table for a specific device."""
    return _op_table_from_records(_text_generate_op_summary(results), device, top_n, columns, sort_by, case_label)


def _text_generate_category_stats(
    results: list[ExperimentResult],
    device: str | None = None,
    case_label: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build operator category statistics."""
    all_ops = _text_generate_op_summary(results)
    if not all_ops:
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_ops)
    if device and "device" in df.columns:
        df = df[df["device"] == device]
    df = _filter_df_by_case(df, case_label)
    if df.empty:
        return pd.DataFrame(), {}

    # Aggregate by category
    category_stats = df.groupby("category").agg({"analytic_total_us": "sum", "name": "count"}).reset_index()
    category_stats.columns = ["Category", "Total Time (us)", "Operator Count"]
    category_stats["Total Time (ms)"] = category_stats["Total Time (us)"] / 1000.0
    category_stats["Ratio (%)"] = category_stats["Total Time (us)"] / category_stats["Total Time (us)"].sum() * 100
    category_stats = category_stats.sort_values(by="Total Time (us)", ascending=False)

    display_df = category_stats[["Category", "Total Time (ms)", "Operator Count", "Ratio (%)"]].copy()

    # Build chart data
    chart_data = {
        "categories": category_stats["Category"].tolist(),
        "times_ms": category_stats["Total Time (ms)"].tolist(),
        "percentages": category_stats["Ratio (%)"].tolist(),
    }

    return _round_numeric_columns(display_df.reset_index(drop=True)), chart_data


def _text_generate_compare_table(results: list[ExperimentResult], top_n: int = 15) -> pd.DataFrame:
    """Build the cross-device operator comparison table without a total column."""
    all_ops = _text_generate_op_summary(results)
    if not all_ops:
        return pd.DataFrame()

    df = pd.DataFrame(all_ops)

    # Select the Top N operators by total time for each device
    top_ops_per_device = []
    for device in df["device"].unique():
        device_df = df[df["device"] == device]
        top_ops = device_df.nlargest(top_n, "analytic_total_us")["name"].tolist()
        top_ops_per_device.extend(top_ops)

    # Build a deduplicated union of operators
    unique_ops = list(set(top_ops_per_device))[:top_n]

    # Build the pivot table
    pivot_df = (
        df[df["name"].isin(unique_ops)]
        .pivot_table(index="name", columns="device", values="analytic_total_us", aggfunc="sum")
        .fillna(0)
    )

    # Convert to milliseconds
    pivot_df = pivot_df / 1000.0

    # Do not add a total column; sort directly by the maximum value
    max_col = pivot_df.max(axis=1)
    pivot_df = pivot_df.loc[max_col.sort_values(ascending=False).index]

    # Reset the index
    pivot_df = pivot_df.reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={"name": "Operator"})

    return _round_numeric_columns(pivot_df)


def _text_generate_summary_markdown(results: list[ExperimentResult]) -> str:
    """Build recommendation markdown for Text Generate."""
    if not results:
        return "### Recommendation\nNo results available."

    full_df = _results_to_df(results)
    if full_df.empty:
        return "### Recommendation\nNo results available."

    work_df = full_df.copy()
    ranking_col = (
        "tps_per_device"
        if "tps_per_device" in work_df.columns and work_df["tps_per_device"].notna().any()
        else "analytic_total_time_s"
    )
    ranked = work_df.dropna(subset=[ranking_col]) if ranking_col in work_df.columns else pd.DataFrame()
    if not ranked.empty:
        ranked = ranked.sort_values(
            by=ranking_col,
            ascending=(ranking_col == "analytic_total_time_s"),
            kind="stable",
        )

    lines = ["### Recommendation"]
    if not ranked.empty:
        top = ranked.iloc[0]
        lines.append(f"- Recommended Device: **{top.get('device', '-')}**")
        if ranking_col == "tps_per_device":
            lines.append(f"- Key Metric: **{float(top.get('tps_per_device', 0) or 0):.2f} token/s/Device**")
        else:
            lines.append(f"- Key Metric: **{float(top.get('analytic_total_time_s', 0) or 0) * 1000:.2f} ms**")
        lines.append(
            f"- Constraint Profile: **{top.get('num_devices', '-')} "
            f"devices / Concurrency {top.get('num_queries', '-')} / "
            f"Context {top.get('context_length', '-')} / Generate {top.get('query_length', '-')} token**"
        )
        lines.append(
            f"- Quantization: **MLP={top.get('quantize_linear_action', '-')} / "
            f"Attention={top.get('quantize_attention_action', '-')}**"
        )
        if pd.notna(top.get("analytic_total_time_s")):
            lines.append(f"- Analytic Time: **{float(top.get('analytic_total_time_s', 0) or 0) * 1000:.2f} ms**")
        if len(ranked) > 1 and ranking_col in ranked.columns:
            runner = ranked.iloc[1]
            top_val = float(top.get(ranking_col, 0) or 0)
            runner_val = float(runner.get(ranking_col, 0) or 0)
            if runner_val > 0:
                if ranking_col == "tps_per_device":
                    gap = (top_val - runner_val) / runner_val * 100.0
                    lines.append(f"- Throughput lead over runner-up **{runner.get('device', '-')}**: **{gap:.2f}%**")
                else:
                    gap = (runner_val - top_val) / runner_val * 100.0
                    lines.append(f"- Time Saved vs Runner-up **{runner.get('device', '-')}**: **{gap:.2f}%**")
    else:
        lines.append("- No valid simulation result is available for recommendation.")
        exec_errors = (
            work_df["execution_error"].dropna() if "execution_error" in work_df.columns else pd.Series(dtype=object)
        )
        if not exec_errors.empty:
            lines.append(f"- Failure Reason: **{exec_errors.iloc[0]}**")
        errors = work_df["error"].dropna() if "error" in work_df.columns else pd.Series(dtype=object)
        if exec_errors.empty and not errors.empty:
            lines.append(f"- Failure Reason: **{errors.iloc[0]}**")

    lines.append("")
    lines.append("### Workspace Summary")
    lines.append(f"- Completed Runs: **{len(results)}**")
    devices = work_df["device"].dropna().astype(str).unique().tolist() if "device" in work_df.columns else []
    if devices:
        lines.append(f"- Compared Devices: **{', '.join(devices[:8])}**")
    failed_count = int((work_df["status"] == "failed").sum()) if "status" in work_df.columns else 0
    if failed_count:
        lines.append(f"- Failed Runs: **{failed_count}**")
    return "\n".join(lines)


def _memory_analysis_from_summary(
    summary: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Extract memory-usage values from the result summary."""
    if not summary:
        return {}, pd.DataFrame()

    def value_of(*keys: str) -> float:
        for key in keys:
            raw = summary.get(key)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    total_memory = value_of("total_device_memory_gb")
    items = [
        ("Model Weights", value_of("model_weight_size_gb")),
        ("KV Cache", value_of("kv_cache_gb")),
        ("Activations", value_of("model_activation_size_gb")),
        ("Reserved Memory", value_of("reserved_memory_gb")),
        (
            "Available Memory",
            value_of("memory_available_gb", "device_memory_available_gb"),
        ),
    ]

    memory_data = {name: val for name, val in items if val > 0}
    rows = [{"Item": name, "Size (GB)": f"{val:.3f}"} for name, val in items if val > 0]
    if total_memory > 0:
        rows.append({"Item": "Total Memory", "Size (GB)": f"{total_memory:.3f}"})
    return memory_data, pd.DataFrame(rows) if rows else pd.DataFrame()


def _text_generate_memory_analysis(
    results: list[ExperimentResult],
    device: str | None = None,
    case_label: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Generate memory analysis for a selected device and concurrency/TP case."""
    if not results:
        return {}, pd.DataFrame()

    candidates = results
    if device:
        candidates = [result for result in candidates if str(result.params.get("device", "")) == str(device)]
    if case_label:
        candidates = [result for result in candidates if _case_label_from_mapping(result.params) == case_label]
    if not candidates:
        return {}, pd.DataFrame()

    latest = candidates[-1]
    summary = latest.summary if hasattr(latest, "summary") else {}
    return _memory_analysis_from_summary(summary)


def _text_generate_bandwidth_analysis(
    results: list[ExperimentResult],
    device: str | None = None,
    case_label: str | None = None,
) -> pd.DataFrame:
    """Generate bandwidth and bottleneck rows for a selected device/case."""
    if not results:
        return pd.DataFrame()

    rows = []
    for result in results:
        params = result.params if hasattr(result, "params") else {}
        if device and str(params.get("device", "")) != str(device):
            continue
        if case_label and _case_label_from_mapping(params) != case_label:
            continue
        summary = result.summary if hasattr(result, "summary") else {}
        row = {
            "device": params.get("device", "-"),
            "concurrency": params.get("num_queries", "-"),
            "tp_size": params.get("tp_size", "-"),
            "case": _case_label_from_mapping(params),
            "bottleneck_type": summary.get("bottleneck_type", "-"),
        }
        for src, dst in [
            ("memory_bound", "memory_bound_pct"),
            ("communication_bound", "communication_bound_pct"),
            ("compute_bound_mma", "compute_mma_bound_pct"),
            ("compute_bound_gp", "compute_gp_bound_pct"),
        ]:
            value = summary.get(src)
            if value is not None and value != "":
                try:
                    row[dst] = round(float(value), 1)
                except (TypeError, ValueError):
                    row[dst] = value
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _text_generate_time_chart(full_df: pd.DataFrame, devices: list[str]):
    """Build the analytic-time chart for concurrency or TP sweeps."""
    if full_df.empty or "analytic_total_time_s" not in full_df.columns:
        return empty_plot("Analytic Time Comparison")

    plot_df = full_df.copy()
    plot_df["analytic_total_time_ms"] = pd.to_numeric(plot_df["analytic_total_time_s"], errors="coerce") * 1000
    plot_df["tp_size"] = pd.to_numeric(plot_df.get("tp_size", 1), errors="coerce").fillna(1).astype(int)
    plot_df["num_queries"] = pd.to_numeric(plot_df.get("num_queries", 0), errors="coerce")
    plot_df = plot_df.dropna(subset=["analytic_total_time_ms", "num_queries", "tp_size"])
    if plot_df.empty:
        return empty_plot("Analytic Time Comparison")

    tp_values = sorted(plot_df["tp_size"].dropna().unique().tolist())
    num_queries_values = sorted(plot_df["num_queries"].dropna().unique().tolist())
    has_tp_sweep = len(tp_values) > 1
    has_concurrency_sweep = len(num_queries_values) > 1

    if has_tp_sweep and has_concurrency_sweep:
        import matplotlib.pyplot as plt

        n_tp = len(tp_values)
        fig, axes = plt.subplots(n_tp, 1, figsize=(14.5, max(4.2, 3.2 * n_tp)), squeeze=False)
        for idx, tp in enumerate(tp_values):
            ax = axes[idx][0]
            tp_df = plot_df[plot_df["tp_size"] == tp]
            if len(devices) > 1 and "device" in tp_df.columns:
                for device, group_df in tp_df.groupby("device", sort=False):
                    group_df = group_df.sort_values("num_queries")
                    ax.plot(
                        group_df["num_queries"],
                        group_df["analytic_total_time_ms"],
                        marker="o",
                        linewidth=2,
                        label=str(device),
                    )
                ax.legend(fontsize=9)
            else:
                tp_df = tp_df.sort_values("num_queries")
                ax.plot(
                    tp_df["num_queries"],
                    tp_df["analytic_total_time_ms"],
                    marker="o",
                    linewidth=2,
                )
            ax.set_title(f"TP={int(tp)}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Concurrency")
            ax.set_ylabel("Analytic Time (ms)")
            ax.grid(axis="y", alpha=0.25, linestyle="--")
        fig.suptitle(
            "Concurrency vs Analytic Time",
            x=0.5,
            y=0.985,
            ha="center",
            va="top",
            fontsize=17,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        return fig

    if has_tp_sweep:
        return line_plot(
            plot_df.sort_values("tp_size"),
            "tp_size",
            "analytic_total_time_ms",
            "TP Size vs Analytic Time",
            "Analytic Time (ms)",
            xlabel="TP Size",
            group="device" if len(devices) > 1 else None,
        )

    if has_concurrency_sweep:
        return line_plot(
            plot_df.sort_values("num_queries"),
            "num_queries",
            "analytic_total_time_ms",
            "Concurrency vs Analytic Time",
            "Analytic Time (ms)",
            xlabel="Concurrency",
            group="device" if len(devices) > 1 else None,
        )

    return bar_plot(
        plot_df,
        "device",
        "analytic_total_time_ms",
        "Analytic Time Comparison",
        "Analytic Time (ms)",
        xlabel="Device",
        group=None,
    )


def _text_generate_common_outputs(
    results: list[ExperimentResult],
    latest: ExperimentResult | None,
    current_model: str = "",
    mtp_acceptance_rate: str = "",
):
    """Generate Text Generate outputs, including TP/concurrency case selectors."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    from .charts import empty_pie_plot, pie_plot

    full_df = _results_to_df(results)
    if not full_df.empty:
        full_df = full_df.copy()
        full_df["case_label"] = full_df.apply(_case_label_from_mapping, axis=1)

    summary = _text_generate_summary_markdown(results)
    devices = sorted(full_df["device"].dropna().astype(str).unique().tolist()) if "device" in full_df.columns else []
    default_device = devices[0] if devices else None
    case_choices = _case_choices_from_rows(full_df)
    default_case = case_choices[0] if case_choices else None

    tps_chart = gr.update(visible=False)
    time_chart = _text_generate_time_chart(full_df, devices)

    num_mtp_tokens = int(latest.params.get("num_mtp_tokens", 0) or 0) if latest and hasattr(latest, "params") else 0
    decode_mode = bool(latest.params.get("decode", False)) if latest and hasattr(latest, "params") else False

    tpot_metric_md = ""
    if decode_mode and num_mtp_tokens > 0 and latest and hasattr(latest, "summary"):
        total_time = latest.summary.get("analytic_total_time_s", 0) or 0
        rates = []
        if mtp_acceptance_rate:
            try:
                rates = [float(r.strip()) for r in mtp_acceptance_rate.split(",") if r.strip()]
            except (ValueError, TypeError):
                rates = []
        if rates and total_time > 0:
            total_tokens = 1 + sum(rates[:num_mtp_tokens])
            tpot = (total_time / total_tokens) * 1000
            tpot_metric_md = f"**TPOT**: {tpot:.2f} ms/token"

    memory_data, memory_table = _text_generate_memory_analysis(results, default_device, default_case)
    if memory_data:
        memory_pie = pie_plot(memory_data, f"Memory usage - {default_device} - {default_case}")
    else:
        memory_pie = empty_pie_plot("Memory usage")

    bandwidth_table = _text_generate_bandwidth_analysis(results, default_device, default_case)
    op_table = _text_generate_op_table(results, default_device, 20, case_label=default_case)
    category_table, category_data = _text_generate_category_stats(results, default_device, default_case)

    if category_data:
        category_df = pd.DataFrame(
            {
                "category": category_data["categories"],
                "time_ms": category_data["times_ms"],
            }
        )
        category_chart = bar_plot(
            category_df,
            "category",
            "time_ms",
            "Operator category time",
            "Time (ms)",
            xlabel="Category",
            group=None,
        )
    else:
        category_chart = empty_plot("Operator category time")

    compare_table = _text_generate_compare_table(results, 15)

    display_df = full_df.copy()
    if not display_df.empty:
        if "analytic_total_time_s" in display_df.columns:
            display_df["inference_time_ms"] = pd.to_numeric(display_df["analytic_total_time_s"], errors="coerce") * 1000
        key_cols = [
            "model_id",
            "device",
            "num_devices",
            "num_queries",
            "tp_size",
            "case_label",
            "query_length",
            "context_length",
            "stage",
            "tps_per_device",
            "inference_time_ms",
            "quantize_linear_action",
            "quantize_attention_action",
            "status",
            "error",
        ]
        existing_cols = [c for c in key_cols if c in display_df.columns]
        display_df = display_df[existing_cols].rename(
            columns={
                "num_queries": "concurrency",
                "case_label": "case",
                "query_length": "query_tokens",
                "context_length": "context_tokens",
            }
        )
        display_df = _round_numeric_columns(display_df)

    op_breakdown_data = _text_generate_op_summary(results)
    case_update = gr.update(choices=case_choices, value=default_case)

    return (
        summary,
        tps_chart,
        time_chart,
        gr.update(value=tpot_metric_md, visible=bool(tpot_metric_md)),
        gr.update(choices=devices, value=default_device),
        case_update,
        memory_pie,
        memory_table,
        gr.update(choices=devices, value=default_device),
        case_update,
        bandwidth_table,
        gr.update(choices=devices, value=default_device),
        case_update,
        op_table,
        gr.update(choices=devices, value=default_device),
        case_update,
        category_chart,
        category_table,
        gr.update(),
        compare_table,
        display_df,
        df_to_records(display_df),
        df_to_records(full_df),
        op_breakdown_data,
        current_model,
        mtp_acceptance_rate,
    )


def run_text_generate_v2(*vals):
    """???????????????"""
    form = _build_text_form(*vals)
    errors = _validate_text_form(form)
    if errors:
        yield _text_validation_empty_outputs(_optimizer_validation_markdown(errors))
        return
    current_model = form.get("model_id", "")
    mtp_acceptance_rate = form.get("mtp_acceptance_rate", "")
    tasks = build_text_generate_tasks(form)
    results: list[ExperimentResult] = []
    total = len(tasks)
    for completed, _, result in RUNNER.run_matrix(tasks):
        results.append(result)
        (
            summary,
            tps_chart,
            time_chart,
            tpot_metric,
            memory_device_update,
            memory_case_update,
            memory_pie,
            memory_table,
            bandwidth_device_update,
            bandwidth_case_update,
            bandwidth_table,
            op_device_update,
            op_case_update,
            op_table,
            category_device_update,
            category_case_update,
            category_chart,
            category_table,
            compare_mode_update,
            compare_table,
            display_df,
            display_rows,
            full_rows,
            op_breakdown,
            current_model,
            mtp_acceptance_data,
        ) = _text_generate_common_outputs(results, result, current_model, mtp_acceptance_rate)
        progress = progress_html(completed, total, result.label, f"{result.status} / {result.source}")
        yield (
            progress,
            summary,
            tps_chart,
            time_chart,
            tpot_metric,
            memory_device_update,
            memory_case_update,
            memory_pie,
            memory_table,
            bandwidth_device_update,
            bandwidth_case_update,
            bandwidth_table,
            op_device_update,
            op_case_update,
            op_table,
            category_device_update,
            category_case_update,
            category_chart,
            category_table,
            compare_mode_update,
            compare_table,
            display_df,
            display_rows,
            full_rows,
            op_breakdown,
            current_model,
            mtp_acceptance_data,
        )


def refresh_text_generate_op_table(results_data: list[dict], device: str, top_n: int):
    """Refresh the operator table."""
    if not results_data:
        return pd.DataFrame()

    # Rebuild result objects
    # Rebuild from stored data

    return _text_generate_op_table([], device, top_n)  # Simplified path


def refresh_text_generate_op_table_from_store(full_rows: list[dict], device: str, top_n: int):
    """Refresh the operator table from stored data."""
    # Simplified placeholder. In production, reload operator data from STORE.
    return pd.DataFrame()  # Placeholder


def load_text_generate_history(current_model: str = ""):
    """Load Text Generate history results, optionally filtered by model."""
    from .charts import empty_pie_plot, pie_plot

    rows = STORE.query_rows("text_generate")
    full_df = _safe_df_from_rows(rows)

    if full_df.empty:
        return (
            progress_html(0, 1, "No History Data", "history"),
            "### Performance Summary\nNo history results available.",
            empty_plot("TPS/Device Comparison"),
            empty_plot("Analytic Time Comparison"),
            gr.update(value="", visible=False),  # tpot_metric
            gr.update(choices=[], value=None),  # memory_device
            empty_pie_plot("Memory Usage Breakdown"),
            pd.DataFrame(),
            gr.update(choices=[], value=None),  # bandwidth_device
            pd.DataFrame(),
            gr.update(choices=[], value=None),  # op_device
            pd.DataFrame(),
            gr.update(choices=[], value=None),  # category_device
            empty_plot("Operator Category Time"),
            pd.DataFrame(),
            gr.update(),  # compare_mode
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            [],
            [],
            "",
            "",
            "No history data",
        )

    # Filter by model
    hint = ""
    if current_model and "model_id" in full_df.columns:
        filtered_df = full_df[full_df["model_id"] == current_model]
        if filtered_df.empty:
            hint = (
                f"Warning: no history results found for model **{current_model}**. Showing all history results instead."
            )
        else:
            hint = f"Loaded history results for model **{current_model}** ({len(filtered_df)} records)."
            full_df = filtered_df
    else:
        models = full_df["model_id"].unique().tolist() if "model_id" in full_df.columns else []
        hint = f"Loaded all history results ({len(full_df)} records). Models: {', '.join(str(m) for m in models[:5])}"

    # Distinguish stages
    if "stage" in full_df.columns:
        stages = full_df["stage"].unique().tolist()
        if len(stages) > 1:
            hint += f"\n\nStages: {', '.join(str(s) for s in stages)}"

    devices = sorted(full_df["device"].unique().tolist()) if "device" in full_df.columns else []

    # Convert analytic time to milliseconds
    plot_df = full_df.copy()
    if "analytic_total_time_s" in plot_df.columns:
        plot_df["analytic_total_time_ms"] = plot_df["analytic_total_time_s"] * 1000

    # TPS chart
    tps_chart = (
        bar_plot(
            full_df,
            "num_queries",
            "tps_per_device",
            "TPS/Device Comparison",
            "Throughput (token/s)",
            xlabel="Concurrency",
            group="device" if len(devices) > 1 else None,
        )
        if not full_df.empty
        else empty_plot("TPS/Device Comparison")
    )

    # Analytic time chart
    time_chart = (
        bar_plot(
            plot_df,
            "device",
            "analytic_total_time_ms",
            "Analytic Time Comparison",
            "Analytic Time (ms)",
            xlabel="Device",
        )
        if not plot_df.empty
        else empty_plot("Analytic Time Comparison")
    )

    # Memory analysis using the latest row
    memory_data = {}
    if not full_df.empty:
        last_row = full_df.iloc[-1].to_dict()
        if last_row.get("model_weight_size_gb"):
            memory_data["Model Weights"] = last_row["model_weight_size_gb"]
        if last_row.get("kv_cache_gb"):
            memory_data["KV Cache"] = last_row["kv_cache_gb"]
        if last_row.get("model_activation_size_gb"):
            memory_data["Activations"] = last_row["model_activation_size_gb"]

    memory_pie = (
        pie_plot(memory_data, "Memory Usage Breakdown") if memory_data else empty_pie_plot("Memory Usage Breakdown")
    )
    memory_table = pd.DataFrame([{"Item": k, "Size (GB)": f"{v:.3f}"} for k, v in memory_data.items()])

    # Summary table
    display_df = full_df.copy()
    if not display_df.empty and "analytic_total_time_s" in display_df.columns:
        display_df["Analytic Time (ms)"] = display_df["analytic_total_time_s"] * 1000

    summary = _text_generate_summary_markdown([])

    return (
        progress_html(len(full_df), len(full_df), "History Loaded", "history"),
        summary,
        tps_chart,
        time_chart,
        gr.update(value="", visible=False),  # tpot_metric
        gr.update(choices=devices, value=devices[0] if devices else None),  # memory_device
        memory_pie,
        memory_table,
        gr.update(choices=devices, value=devices[0] if devices else None),  # bandwidth_device
        pd.DataFrame(),  # Bandwidth analysis
        gr.update(choices=devices, value=devices[0] if devices else None),  # op_device
        pd.DataFrame(),  # Operator table
        gr.update(choices=devices, value=devices[0] if devices else None),  # category_device
        empty_plot("Operator Category Time"),
        pd.DataFrame(),
        gr.update(),  # compare_mode
        pd.DataFrame(),  # compare_table
        display_df,
        df_to_records(display_df),
        df_to_records(full_df),
        [],  # Operator details
        current_model,
        "",  # mtp_acceptance_data
        hint,
    )


def update_op_table_from_breakdown(
    op_breakdown: list[dict],
    device: str,
    case_or_top_n: str | int | None = None,
    top_n_or_columns: int | Sequence[str] | None = None,
    columns_or_sort: Sequence[str] | str | None = None,
    sort_by: str | None = None,
):
    """Refresh operator table; supports both old and case-aware callback signatures.

    The function signature is ambiguous to support two calling patterns:
    1. Case-aware: (op_breakdown, device, case_label, top_n, columns, sort_by)
    2. Simple: (op_breakdown, device, top_n, columns, sort_by, extra_param)

    We detect the pattern by checking the type of the 3rd parameter.
    """
    # Check if we're in case-aware mode by looking at the 3rd and 4th parameters
    # Case-aware: 3rd is str (case), 4th is int (top_n)
    # Simple: 3rd is int/None (top_n), 4th is list (columns)
    is_case_mode = (
        isinstance(case_or_top_n, str)
        and case_or_top_n  # 3rd is non-empty string
        and isinstance(top_n_or_columns, (int, float))  # 4th is numeric
    )

    if is_case_mode:
        # Case-aware mode
        case_label = str(case_or_top_n)
        top_n = int(top_n_or_columns or 20)
        columns = columns_or_sort if isinstance(columns_or_sort, (list, tuple)) else None
        effective_sort = sort_by
    else:
        # Simple mode: 3rd param is top_n (or None), 4th param is columns, 5th param is sort_by
        case_label = None
        # Extract top_n from 3rd or 4th parameter
        if isinstance(case_or_top_n, (int, float)):
            top_n = int(case_or_top_n)
        elif isinstance(top_n_or_columns, (int, float)):
            top_n = int(top_n_or_columns)
        else:
            top_n = 20
        # Extract columns from 4th or 5th parameter
        if isinstance(top_n_or_columns, (list, tuple)):
            columns = top_n_or_columns
        elif isinstance(columns_or_sort, (list, tuple)):
            columns = columns_or_sort
        else:
            columns = None
        # Extract sort_by from 5th or 6th parameter
        if isinstance(columns_or_sort, str) and not isinstance(columns_or_sort, (list, tuple)):
            effective_sort = columns_or_sort
        else:
            effective_sort = sort_by

    return _op_table_from_records(op_breakdown, device, top_n, columns, effective_sort, case_label)


# -----------------------------
# Video Generate callbacks
# -----------------------------
def _video_generate_op_summary(results: list[ExperimentResult]) -> list[dict[str, Any]]:
    """Collect operator data from all results."""
    all_ops = []
    for result in results:
        device = result.params.get("device", "unknown")
        case_label = _case_label_from_mapping(result.params)
        op_breakdown = result.tables.get("op_breakdown", [])
        for op in op_breakdown:
            op["device"] = device
            op["num_queries"] = result.params.get("num_queries")
            op["tp_size"] = result.params.get("tp_size")
            op["case_label"] = case_label
            op["category"] = _categorize_op(op.get("name", ""))
            all_ops.append(op)
    return all_ops


def _video_generate_op_table(
    results: list[ExperimentResult],
    device: str | None,
    top_n: int,
    columns: Sequence[str] | None = None,
    sort_by: str | None = None,
    case_label: str | None = None,
) -> pd.DataFrame:
    """Build the operator time table for a specific device."""
    return _op_table_from_records(_video_generate_op_summary(results), device, top_n, columns, sort_by)


def _video_generate_category_stats(
    results: list[ExperimentResult],
) -> tuple[pd.DataFrame, dict]:
    """Build operator category statistics."""
    all_ops = _video_generate_op_summary(results)
    if not all_ops:
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_ops)

    # Aggregate by category
    category_stats = df.groupby("category").agg({"analytic_total_us": "sum", "name": "count"}).reset_index()
    category_stats.columns = ["Category", "Total Time (us)", "Operator Count"]
    category_stats["Total Time (ms)"] = category_stats["Total Time (us)"] / 1000.0
    category_stats["Ratio (%)"] = category_stats["Total Time (us)"] / category_stats["Total Time (us)"].sum() * 100
    category_stats = category_stats.sort_values(by="Total Time (us)", ascending=False)

    display_df = category_stats[["Category", "Total Time (ms)", "Operator Count", "Ratio (%)"]].copy()

    # Build chart data
    chart_data = {
        "categories": category_stats["Category"].tolist(),
        "times_ms": category_stats["Total Time (ms)"].tolist(),
        "percentages": category_stats["Ratio (%)"].tolist(),
    }

    return _round_numeric_columns(display_df.reset_index(drop=True)), chart_data


def _video_generate_compare_table(results: list[ExperimentResult], top_n: int = 15) -> pd.DataFrame:
    """Build the cross-device operator comparison table."""
    all_ops = _video_generate_op_summary(results)
    if not all_ops:
        return pd.DataFrame()

    df = pd.DataFrame(all_ops)

    # Select the Top N operators by total time for each device
    top_ops_per_device = []
    for device in df["device"].unique():
        device_df = df[df["device"] == device]
        top_ops = device_df.nlargest(top_n, "analytic_total_us")["name"].tolist()
        top_ops_per_device.extend(top_ops)

    # Build a deduplicated union of operators
    unique_ops = list(set(top_ops_per_device))[:top_n]

    # Build the pivot table
    pivot_df = (
        df[df["name"].isin(unique_ops)]
        .pivot_table(index="name", columns="device", values="analytic_total_us", aggfunc="sum")
        .fillna(0)
    )

    # Convert to milliseconds
    pivot_df = pivot_df / 1000.0

    # Add the total column
    pivot_df["Total (ms)"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by="Total (ms)", ascending=False)

    # Reset the index
    pivot_df = pivot_df.reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={"name": "Operator"})

    return _round_numeric_columns(pivot_df)


def _video_generate_summary_markdown(results: list[ExperimentResult]) -> str:
    """Build recommendation markdown for Video Generate."""
    if not results:
        return "### Recommendation\nNo results available."

    full_df = _results_to_df(results)
    if full_df.empty:
        return "### Recommendation\nNo results available."

    work_df = full_df.copy()
    ranked = (
        work_df.dropna(subset=["analytic_total_time_s"])
        if "analytic_total_time_s" in work_df.columns
        else pd.DataFrame()
    )
    if not ranked.empty:
        ranked = ranked.sort_values(by="analytic_total_time_s", ascending=True, kind="stable")

    lines = ["### Recommendation"]
    if not ranked.empty:
        top = ranked.iloc[0]
        lines.append(f"- Recommended Device: **{top.get('device', '-')}**")
        lines.append(f"- Key Metric: **{float(top.get('analytic_total_time_s', 0) or 0):.4f} s total analytic time**")
        if pd.notna(top.get("communication_total_s")):
            lines.append(f"- Communication Time: **{float(top.get('communication_total_s', 0) or 0):.4f} s**")
        lines.append(
            f"- Scenario: **{top.get('batch_size', '-')} Batch / {top.get('height', '-')} x {top.get('width', '-')} / "
            f"{top.get('frame_num', '-')} frames / {top.get('sample_step', '-')} steps**"
        )
        lines.append(
            f"- Parallelism and Quantization: **"
            f"{top.get('world_size', '-')} devices / "
            f"Ulysses={top.get('ulysses_size', '-')} / "
            f"MLP={top.get('quantize_linear_action', '-')}**"
        )
        if len(ranked) > 1:
            runner = ranked.iloc[1]
            top_val = float(top.get("analytic_total_time_s", 0) or 0)
            runner_val = float(runner.get("analytic_total_time_s", 0) or 0)
            if runner_val > 0:
                gap = (runner_val - top_val) / runner_val * 100.0
                lines.append(f"- Time Saved vs Runner-up **{runner.get('device', '-')}**: **{gap:.2f}%**")
    else:
        lines.append("- No valid simulation result is available for recommendation.")
        errors = work_df["error"].dropna() if "error" in work_df.columns else pd.Series(dtype=object)
        if not errors.empty:
            lines.append(f"- Failure Reason: **{errors.iloc[0]}**")

    lines.append("")
    lines.append("### Workspace Summary")
    lines.append(f"- Completed Runs: **{len(results)}**")
    devices = work_df["device"].dropna().astype(str).unique().tolist() if "device" in work_df.columns else []
    if devices:
        lines.append(f"- Compared Devices: **{', '.join(devices[:8])}**")
    failed_count = int((work_df["status"] == "failed").sum()) if "status" in work_df.columns else 0
    if failed_count:
        lines.append(f"- Failed Runs: **{failed_count}**")
    return "\n".join(lines)


def _video_generate_common_outputs(results: list[ExperimentResult], latest: ExperimentResult | None):
    """Generate Video Generate outputs."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    full_df = _results_to_df(results)
    summary = _video_generate_summary_markdown(results)

    # Load the device list
    devices = sorted(full_df["device"].unique().tolist()) if "device" in full_df.columns else []

    # Total analytic time chart
    time_chart = (
        bar_plot(
            full_df,
            "device",
            "analytic_total_time_s",
            "Total Analytic Time Comparison",
            "Analytic Time (s)",
            xlabel="Device",
            group="quantize_linear_action" if "quantize_linear_action" in full_df.columns else None,
        )
        if not full_df.empty
        else empty_plot("Total Analytic Time Comparison")
    )

    # Communication time chart
    if "communication_total_s" in full_df.columns and not full_df["communication_total_s"].dropna().empty:
        comm_chart = bar_plot(
            full_df,
            "device",
            "communication_total_s",
            "Communication Time Comparison",
            "Communication Time (s)",
            xlabel="Device",
            group=None,
        )
    else:
        comm_chart = empty_plot("Communication Time Comparison")

    # Operator table using the first device by default, Top 20
    op_table = _video_generate_op_table(results, devices[0] if devices else None, 20)

    # Category statistics
    category_table, category_data = _video_generate_category_stats(results)

    # Category chart
    if category_data:
        category_df = pd.DataFrame(
            {
                "Category": category_data["categories"],
                "Time (ms)": category_data["times_ms"],
            }
        )
        category_chart = bar_plot(
            category_df,
            "Category",
            "Time (ms)",
            "Operator Category Time",
            "Time (ms)",
            xlabel="Category",
            group=None,
        )
    else:
        category_chart = empty_plot("Operator Category Time")

    # Cross-device comparison table
    compare_table = _video_generate_compare_table(results, 15)

    # Summary table
    display_df = full_df.copy()
    if not display_df.empty:
        key_cols = [
            "model_id",
            "device",
            "batch_size",
            "height",
            "width",
            "frame_num",
            "sample_step",
            "analytic_total_time_s",
            "communication_total_s",
            "quantize_linear_action",
            "ulysses_size",
            "status",
            "error",
        ]
        existing_cols = [c for c in key_cols if c in display_df.columns]
        display_df = display_df[existing_cols].rename(
            columns={
                "model_id": "Model",
                "device": "Device",
                "batch_size": "Batch",
                "height": "Height",
                "width": "Width",
                "frame_num": "Frames",
                "sample_step": "Sample Steps",
                "analytic_total_time_s": "Total Analytic Time (s)",
                "communication_total_s": "Communication Time (s)",
                "quantize_linear_action": "MLP Quantization",
                "ulysses_size": "Ulysses Parallelism",
                "status": "Status",
                "error": "Error",
            }
        )
        display_df = _round_numeric_columns(display_df)
    # Operator detail data
    op_breakdown_data = _video_generate_op_summary(results)

    return (
        summary,
        time_chart,
        comm_chart,
        gr.update(choices=devices, value=devices[0] if devices else None),
        op_table,
        category_chart,
        category_table,
        compare_table,
        display_df,
        df_to_records(display_df),
        df_to_records(full_df),
        op_breakdown_data,
    )


def run_video_generate_v2(*vals):
    """Run video generation simulation in the refactored workspace."""
    form = _build_video_form(*vals)
    errors = _validate_video_form(form)
    if errors:
        yield _video_validation_empty_outputs(_optimizer_validation_markdown(errors))
        return
    tasks = build_video_generate_tasks(form)
    results: list[ExperimentResult] = []
    total = len(tasks)
    for completed, _, result in RUNNER.run_matrix(tasks):
        results.append(result)
        (
            summary,
            time_chart,
            comm_chart,
            device_update,
            op_table,
            category_chart,
            category_table,
            compare_table,
            display_df,
            display_rows,
            full_rows,
            op_breakdown,
        ) = _video_generate_common_outputs(results, result)
        progress = progress_html(completed, total, result.label, f"{result.status} / {result.source}")
        yield (
            progress,
            summary,
            time_chart,
            comm_chart,
            device_update,
            op_table,
            category_chart,
            category_table,
            compare_table,
            display_df,
            display_rows,
            full_rows,
            op_breakdown,
        )


def load_video_generate_history():
    """Load Video Generate history results."""
    rows = STORE.query_rows("video_generate")
    full_df = _safe_df_from_rows(rows)

    if full_df.empty:
        return (
            progress_html(0, 1, "No History Data", "history"),
            "### Performance Summary\nNo history results available.",
            empty_plot("Total Analytic Time Comparison"),
            empty_plot("Communication Time Comparison"),
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            empty_plot("Operator Category Time"),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            [],
            [],
        )

    # Simplified path using stored row data
    devices = sorted(full_df["device"].unique().tolist()) if "device" in full_df.columns else []

    summary = _video_generate_summary_markdown([])
    time_chart = (
        bar_plot(
            full_df,
            "device",
            "analytic_total_time_s",
            "Total Analytic Time Comparison",
            "Analytic Time (s)",
            xlabel="Device",
            group="quantize_linear_action" if "quantize_linear_action" in full_df.columns else None,
        )
        if not full_df.empty
        else empty_plot("Total Analytic Time Comparison")
    )

    if "communication_total_s" in full_df.columns and not full_df["communication_total_s"].dropna().empty:
        comm_chart = bar_plot(
            full_df,
            "device",
            "communication_total_s",
            "Communication Time Comparison",
            "Communication Time (s)",
            xlabel="Device",
        )
    else:
        comm_chart = empty_plot("Communication Time Comparison")

    display_df = full_df.copy()

    return (
        progress_html(len(full_df), len(full_df), "History Loaded", "history"),
        summary,
        time_chart,
        comm_chart,
        gr.update(choices=devices, value=devices[0] if devices else None),
        pd.DataFrame(),  # Operator table requires raw data
        empty_plot("Operator Category Time"),
        pd.DataFrame(),
        pd.DataFrame(),
        display_df,
        df_to_records(display_df),
        df_to_records(full_df),
        [],  # Operator details require raw data
    )


def update_video_op_table_from_breakdown(
    op_breakdown: list[dict],
    device: str,
    top_n: int,
    columns: Sequence[str] | None = None,
    sort_by: str | None = None,
):
    """Refresh the Video Generate table using operator detail rows."""
    return _op_table_from_records(op_breakdown, device, top_n, columns, sort_by)


# -----------------------------
# Throughput Optimizer callbacks
# -----------------------------
def _optimizer_metric_plot(full_df: pd.DataFrame, metric_col: str, title: str, ylabel: str):
    if full_df.empty or metric_col not in full_df.columns or not full_df[metric_col].notna().any():
        return empty_plot(title)
    return bar_plot(
        full_df,
        "device",
        metric_col,
        title,
        ylabel,
        xlabel="Device",
        group=None,
        value_fontsize=11,
    )


_FIXED_COMPARE_METRICS = {
    "Throughput": (
        "throughput_token_s",
        "Fixed-Config Throughput Comparison",
        "Throughput (token/s)",
        False,
    ),
    "TTFT": ("ttft_ms", "Fixed-Config TTFT Comparison", "TTFT (ms)", True),
    "TPOT": ("tpot_ms", "Fixed-Config TPOT Comparison", "TPOT (ms)", True),
}


def _optimizer_candidate_rows_from_records(
    records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        top_configs = record.get("top_configs") or []
        if not isinstance(top_configs, list):
            continue
        deployment_mode = _optimizer_deployment_mode(pd.Series(record))
        for candidate in top_configs:
            if not isinstance(candidate, dict):
                continue
            # For PD Ratio mode, use d_parallel, d_batch_size, d_concurrency
            # For standard mode, use parallel, batch_size, concurrency
            is_pd_ratio = candidate.get("pd_ratio") is not None
            if is_pd_ratio:
                parallel = candidate.get("d_parallel")
                batch_size = candidate.get("d_batch_size")
                concurrency = candidate.get("d_concurrency")
            else:
                parallel = candidate.get("parallel")
                batch_size = candidate.get("batch_size")
                concurrency = candidate.get("concurrency")
            if parallel in (None, "") or batch_size in (None, "") or concurrency in (None, ""):
                continue
            candidates.append(
                {
                    "model_id": record.get("model_id"),
                    "device": record.get("device"),
                    "deployment_mode": deployment_mode,
                    "num_devices": record.get("num_devices"),
                    "input_length": record.get("input_length"),
                    "output_length": record.get("output_length"),
                    "prefix_cache_hit_rate": record.get("prefix_cache_hit_rate"),
                    "quantize_linear_action": record.get("quantize_linear_action"),
                    "quantize_attention_action": record.get("quantize_attention_action"),
                    "parallel": parallel,
                    "batch_size": batch_size,
                    "concurrency": concurrency,
                    "throughput_token_s": candidate.get("throughput_token_s"),
                    "ttft_ms": candidate.get("ttft_ms"),
                    "tpot_ms": candidate.get("tpot_ms"),
                    "rank": candidate.get("rank"),
                    "pd_ratio": candidate.get("pd_ratio"),
                }
            )
    return candidates


def _optimizer_candidate_rows_from_results(
    results: list[ExperimentResult],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        row = result.to_row()
        row["top_configs"] = (result.tables or {}).get("top_configs") or []
        records.append(row)
    return _optimizer_candidate_rows_from_records(records)


def _optimizer_fixed_config_key(row: pd.Series) -> str:
    parts = [
        row.get("model_id", "-"),
        row.get("deployment_mode", "-"),
        row.get("num_devices", "-"),
        row.get("input_length", "-"),
        row.get("output_length", "-"),
        row.get("prefix_cache_hit_rate", 0),
        row.get("quantize_linear_action", "-"),
        row.get("quantize_attention_action", "-"),
        row.get("parallel", "-"),
        row.get("batch_size", "-"),
        row.get("concurrency", "-"),
    ]
    return "||".join(str(part) for part in parts)


def _optimizer_fixed_config_label(row: pd.Series, device_count: int | None = None) -> str:
    pieces = [
        str(row.get("model_id", "-")),
        str(row.get("deployment_mode", "-")),
        str(row.get("parallel", "-")),
        f"Batch {row.get('batch_size', '-')}",
        f"Concurrency {row.get('concurrency', '-')}",
    ]
    prefix_cache = row.get("prefix_cache_hit_rate")
    if prefix_cache is not None and not pd.isna(prefix_cache) and float(prefix_cache) > 0:
        pieces.append(f"Prefix Cache {float(prefix_cache):.2f}")
    if device_count is not None:
        pieces.append(f"{device_count} devices")
    return " | ".join(pieces)


def _optimizer_fixed_compare_outputs(
    candidate_rows: list[dict[str, Any]] | None,
    selected_key: str | None = None,
    metric_name: str = "Throughput",
):
    if gr is None:
        raise RuntimeError("gradio is not installed")

    metric_col, title, ylabel, lower_is_better = _FIXED_COMPARE_METRICS.get(
        metric_name,
        _FIXED_COMPARE_METRICS["Throughput"],
    )
    empty_df = pd.DataFrame()
    candidate_df = _safe_df_from_rows(candidate_rows)
    if candidate_df.empty:
        return (
            gr.update(choices=[], value=None),
            "\n".join(
                [
                    "### Fixed-Config Comparison",
                    "No comparable candidate configurations are available.",
                ]
            ),
            empty_plot(title),
            empty_df,
        )

    required_cols = {"device", "parallel", "batch_size", "concurrency", metric_col}
    if not required_cols.issubset(candidate_df.columns):
        return (
            gr.update(choices=[], value=None),
            "\n".join(
                [
                    "### Fixed-Config Comparison",
                    "The current results do not include enough candidate configuration fields.",
                ]
            ),
            empty_plot(title),
            empty_df,
        )

    work_df = candidate_df.copy()
    work_df["config_key"] = work_df.apply(_optimizer_fixed_config_key, axis=1)
    work_df = work_df.drop_duplicates(subset=["device", "config_key"], keep="first")
    work_df = work_df.dropna(subset=["device", "parallel", "batch_size", "concurrency"])
    if work_df.empty:
        return (
            gr.update(choices=[], value=None),
            "\n".join(
                [
                    "### Fixed-Config Comparison",
                    "No fixed configuration is available for cross-device comparison.",
                ]
            ),
            empty_plot(title),
            empty_df,
        )

    config_df = (
        work_df.groupby("config_key", as_index=False)
        .agg(
            model_id=("model_id", "first"),
            deployment_mode=("deployment_mode", "first"),
            num_devices=("num_devices", "first"),
            input_length=("input_length", "first"),
            output_length=("output_length", "first"),
            prefix_cache_hit_rate=("prefix_cache_hit_rate", "first"),
            quantize_linear_action=("quantize_linear_action", "first"),
            quantize_attention_action=("quantize_attention_action", "first"),
            parallel=("parallel", "first"),
            batch_size=("batch_size", "first"),
            concurrency=("concurrency", "first"),
            device_count=("device", "nunique"),
            avg_throughput=("throughput_token_s", "mean"),
        )
        .sort_values(
            by=["device_count", "avg_throughput"],
            ascending=[False, False],
            kind="stable",
        )
    )
    config_df["label"] = config_df.apply(
        lambda row: _optimizer_fixed_config_label(row, int(row.get("device_count", 0) or 0)),
        axis=1,
    )
    choice_values = [(row["label"], row["config_key"]) for _, row in config_df.iterrows()]
    valid_keys = set(config_df["config_key"].tolist())
    current_key = selected_key if selected_key in valid_keys else config_df.iloc[0]["config_key"]
    selected_row = config_df.loc[config_df["config_key"] == current_key].iloc[0]

    compare_df = work_df[work_df["config_key"] == current_key].copy()
    compare_df = compare_df.dropna(subset=[metric_col])
    if metric_col in compare_df.columns:
        compare_df = compare_df.sort_values(by=metric_col, ascending=lower_is_better, kind="stable")

    chart = bar_plot(
        compare_df,
        "device",
        metric_col,
        title,
        ylabel,
        xlabel="Device",
        group=None,
        value_fontsize=11,
    )

    table_cols = [
        col
        for col in [
            "device",
            "deployment_mode",
            "parallel",
            "batch_size",
            "concurrency",
            "throughput_token_s",
            "ttft_ms",
            "tpot_ms",
            "input_length",
            "output_length",
            "prefix_cache_hit_rate",
            "quantize_linear_action",
            "quantize_attention_action",
            "rank",
        ]
        if col in compare_df.columns
    ]
    table_df = compare_df[table_cols].copy() if table_cols else pd.DataFrame()
    if not table_df.empty:
        table_df = table_df.rename(
            columns={
                "device": "Device",
                "deployment_mode": "Deployment Mode",
                "parallel": "Parallel Mode",
                "batch_size": "Batch Size",
                "concurrency": "Concurrency",
                "throughput_token_s": "Throughput (token/s)",  # nosec B105
                "ttft_ms": "TTFT(ms)",
                "tpot_ms": "TPOT(ms)",
                "input_length": "Input Length",
                "output_length": "Output Length",
                "prefix_cache_hit_rate": "Prefix Cache Hit Rate",
                "quantize_linear_action": "MLP Quantization Mode",
                "quantize_attention_action": "Attention Quantization Mode",
                "rank": "Candidate Rank",
            }
        )

    lines = [
        "### Fixed-Config Comparison",
        f"- Current Fixed Configuration: **{_optimizer_fixed_config_label(selected_row)}**",
        f"- Comparable Devices: **{int(selected_row.get('device_count', 0) or 0)}**",
        f"- Current Metric: **{metric_name}**",
    ]
    if pd.notna(selected_row.get("input_length")) and pd.notna(selected_row.get("output_length")):
        input_length = int(selected_row.get("input_length"))
        output_length = int(selected_row.get("output_length"))
        num_devices = int(selected_row.get("num_devices", 0) or 0)
        lines.append(f"- Constraint Profile: **Input {input_length} / Output {output_length} / {num_devices} devices**")
    qlinear = selected_row.get("quantize_linear_action")
    qattn = selected_row.get("quantize_attention_action")
    if qlinear or qattn:
        lines.append(f"- Quantization: **MLP={qlinear or '-'} / Attention={qattn or '-'}**")
    if not compare_df.empty and metric_col in compare_df.columns:
        leader = compare_df.iloc[0]
        lines.append(
            f"- Current Leading Device: **{leader.get('device', '-')}**, "
            f"{metric_name} **{float(leader.get(metric_col, 0) or 0):.2f}**"
        )
    if int(selected_row.get("device_count", 0) or 0) < 2:
        lines.append(
            "- The current fixed configuration matches only one device "
            "and is mainly useful for single-device reference."
        )
    else:
        lines.append("- This view compares devices under the same parallel mode, batch size, and concurrency.")

    return (
        gr.update(choices=choice_values, value=current_key),
        "\n".join(lines),
        chart,
        table_df,
    )


def _optimizer_pd_ratio_outputs(full_df: pd.DataFrame):
    if full_df.empty or not {"p_qps", "d_qps"}.issubset(full_df.columns):
        return empty_plot("Prefill / Decode QPS Comparison"), pd.DataFrame()

    plot_rows: list[dict[str, Any]] = []
    for _, row in full_df.iterrows():
        device = str(row.get("device", "-"))
        p_qps = row.get("p_qps")
        d_qps = row.get("d_qps")
        if pd.notna(p_qps):
            plot_rows.append({"device": device, "metric": "Prefill QPS", "value": float(p_qps)})
        if pd.notna(d_qps):
            plot_rows.append({"device": device, "metric": "Decode QPS", "value": float(d_qps)})

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return empty_plot("Prefill / Decode QPS Comparison"), pd.DataFrame()

    chart = bar_plot(
        plot_df,
        "device",
        "value",
        "Prefill / Decode QPS Comparison",
        "QPS",
        xlabel="Device",
        group="metric",
        value_fontsize=11,
    )
    preferred_cols = [
        c
        for c in [
            "device",
            "deployment_mode",
            "p_qps",
            "d_qps",
            "balanced_qps",
            "pd_ratio",
            "prefill_devices_per_instance",
            "decode_devices_per_instance",
        ]
        if c in full_df.columns
    ]
    pd_df = full_df[preferred_cols].copy() if preferred_cols else pd.DataFrame()
    if not pd_df.empty:
        pd_df = pd_df.rename(
            columns={
                "device": "Device",
                "deployment_mode": "Deployment Mode",
                "p_qps": "Prefill QPS",
                "d_qps": "Decode QPS",
                "balanced_qps": "Balanced QPS",
                "pd_ratio": "PD Ratio",
                "prefill_devices_per_instance": "Prefill Devices per Instance",
                "decode_devices_per_instance": "Decode Devices per Instance",
            }
        )
    return chart, pd_df


def _optimizer_summary_markdown_from_df(full_df: pd.DataFrame, completed_count: int | None = None) -> str:
    if full_df.empty:
        return "\n".join(["### Recommendation", "No results available."])

    work_df = full_df.copy()
    work_df["deployment_mode"] = work_df.apply(_optimizer_deployment_mode, axis=1)
    ranking_col = (
        "balanced_qps"
        if "balanced_qps" in work_df.columns and work_df["balanced_qps"].notna().any()
        else "best_throughput"
    )
    ranking_label = "Balanced QPS" if ranking_col == "balanced_qps" else "\u541e\u5410"
    if ranking_col in work_df.columns:
        ranked = work_df.dropna(subset=[ranking_col]).sort_values(by=ranking_col, ascending=False)
    else:
        ranked = pd.DataFrame()

    lines_out = ["### Recommendation"]
    if not ranked.empty:
        top = ranked.iloc[0]
        lines_out.append(f"- Recommended Device: **{top.get('device', '-')}**")
        lines_out.append(f"- Recommended Deployment Mode: **{top.get('deployment_mode', '-')}**")
        lines_out.append(
            f"- Recommended Configuration: **{top.get('best_parallel', '-')} / "
            f"Batch {top.get('best_batch_size', '-')} / "
            f"Concurrency {top.get('best_concurrency', '-')}**"
        )
        lines_out.append(f"- {ranking_label}: **{float(top.get(ranking_col, 0) or 0):.2f}**")
        lines_out.append(
            f"- Throughput / TTFT / TPOT: **"
            f"{float(top.get('best_throughput', 0) or 0):.2f} token/s / "
            f"{float(top.get('best_ttft_ms', 0) or 0):.2f} ms / "
            f"{float(top.get('best_tpot_ms', 0) or 0):.2f} ms**"
        )
        prefix_cache = top.get("prefix_cache_hit_rate")
        if prefix_cache is not None and not pd.isna(prefix_cache) and float(prefix_cache) > 0:
            lines_out.append(f"- Prefix Cache Hit Rate: **{float(prefix_cache):.2f}**")
        if pd.notna(top.get("pd_ratio")):
            lines_out.append(f"- PD Ratio: **{float(top.get('pd_ratio', 0) or 0):.2f}**")
        if pd.notna(top.get("prefill_devices_per_instance")) and pd.notna(top.get("decode_devices_per_instance")):
            prefill_devices = int(top.get("prefill_devices_per_instance"))
            decode_devices = int(top.get("decode_devices_per_instance"))
            lines_out.append(f"- Prefill/Decode Devices per Instance: **{prefill_devices}:{decode_devices}**")
        if len(ranked) > 1:
            runner_up = ranked.iloc[1]
            top_val = float(top.get(ranking_col, 0) or 0)
            second_val = float(runner_up.get(ranking_col, 0) or 0)
            if second_val > 0:
                gap = (top_val - second_val) / second_val * 100.0
                lines_out.append(f"- Lead over runner-up **{runner_up.get('device', '-')}**: **{gap:.2f}%**")
    else:
        reason_series = (
            work_df["no_result_reason"].dropna() if "no_result_reason" in work_df.columns else pd.Series(dtype=object)
        )
        error_series = work_df["error"].dropna() if "error" in work_df.columns else pd.Series(dtype=object)
        exec_error_series = (
            work_df["execution_error"].dropna() if "execution_error" in work_df.columns else pd.Series(dtype=object)
        )
        lines_out.append("- No valid configuration is currently available for recommendation.")
        if not exec_error_series.empty:
            lines_out.append(f"- Backend Execution Error: **{exec_error_series.iloc[0]}**")
        elif not error_series.empty:
            lines_out.append(f"- Backend Execution Error: **{error_series.iloc[0]}**")
        elif not reason_series.empty:
            lines_out.append(f"- Primary Reason: **{reason_series.iloc[0]}**")
            lines_out.append(
                "- \u5efa\u8bae\u653e\u5bbd\u65f6\u5ef6\u7ea6\u675f\u3001"
                "\u589e\u52a0\u5361\u6570\u6216\u7f29\u77ed"
                "\u8f93\u5165/\u8f93\u51fa\u957f\u5ea6\u540e\u91cd\u8bd5\u3002"
            )

    lines_out.append("")
    lines_out.append("### Workspace Summary")
    lines_out.append(f"- Completed Runs: **{completed_count or len(full_df)}**")
    devices = work_df["device"].dropna().astype(str).unique().tolist() if "device" in work_df.columns else []
    if devices:
        lines_out.append(f"- Compared Devices: **{', '.join(devices[:8])}**")
    no_result_count = int(work_df["no_result_reason"].notna().sum()) if "no_result_reason" in work_df.columns else 0
    if no_result_count > 0:
        lines_out.append(f"- Runs Without Feasible Plans: **{no_result_count}**")
    return "\n".join(lines_out)


def _optimizer_summary_markdown(results: list[ExperimentResult]) -> str:
    """Generate optimizer recommendation markdown from live results."""
    if not results:
        return "\n".join(["### Recommendation", "No results available."])
    return _optimizer_summary_markdown_from_df(_results_to_df(results), len(results))


def _optimizer_common_outputs(results: list[ExperimentResult], latest: ExperimentResult | None):
    """Generate optimizer workspace outputs without changing the external interface."""
    if gr is None:
        raise RuntimeError("gradio is not installed")

    full_df = _results_to_df(results)
    if not full_df.empty:
        full_df = full_df.copy()
        full_df["deployment_mode"] = full_df.apply(_optimizer_deployment_mode, axis=1)
    summary = _optimizer_summary_markdown_from_df(full_df, len(results))

    throughput_chart = _optimizer_metric_plot(
        full_df, "best_throughput", "Best Throughput by Device", "Throughput (token/s)"
    )
    ttft_chart = _optimizer_metric_plot(full_df, "best_ttft_ms", "Best TTFT by Device", "TTFT (ms)")
    tpot_chart = _optimizer_metric_plot(full_df, "best_tpot_ms", "Best TPOT by Device", "TPOT (ms)")
    compare_metric, compare_title, compare_ylabel = _optimizer_primary_metric(full_df)
    batch_chart = _optimizer_metric_plot(full_df, compare_metric, compare_title, compare_ylabel)
    pd_chart, pd_df = _optimizer_pd_ratio_outputs(full_df)

    candidate_rows = _optimizer_candidate_rows_from_results(results)
    state_rows = _optimizer_state_rows(results)
    fixed_update, fixed_md, fixed_chart, fixed_df = _optimizer_fixed_compare_outputs(candidate_rows)
    detail_update, detail_md, detail_pareto_chart, detail_df, detail_output = _optimizer_detail_view(
        state_rows, candidate_rows, None
    )
    display_df = _simplify_optimizer_display_df(full_df)

    return (
        summary,
        throughput_chart,
        ttft_chart,
        tpot_chart,
        batch_chart,
        pd_chart,
        pd_df,
        fixed_update,
        fixed_md,
        fixed_chart,
        fixed_df,
        detail_update,
        detail_md,
        detail_pareto_chart,
        detail_df,
        detail_output,
        display_df,
        df_to_records(display_df),
        state_rows,
        candidate_rows,
    )


def run_optimizer_v2(*vals):
    """Run optimizer tasks for the refactored workspace."""
    form = _build_opt_form(*vals)
    errors = _validate_optimizer_form(form)
    if errors:
        yield _optimizer_empty_outputs(_optimizer_validation_markdown(errors))
        return

    tasks = build_optimizer_tasks(form)
    if not tasks:
        yield _optimizer_empty_outputs(
            "\n".join(
                [
                    "### Parameter Validation Failed",
                    "- No executable tasks were generated.",
                ]
            )
        )
        return

    results: list[ExperimentResult] = []
    total = len(tasks)
    for completed, _, result in RUNNER.run_matrix(tasks):
        results.append(result)
        (
            summary,
            throughput_chart,
            ttft_chart,
            tpot_chart,
            batch_chart,
            pd_chart,
            pd_df,
            fixed_update,
            fixed_md,
            fixed_chart,
            fixed_df,
            detail_update,
            detail_md,
            detail_pareto_chart,
            detail_df,
            detail_output,
            display_df,
            display_rows,
            full_rows,
            candidate_rows,
        ) = _optimizer_common_outputs(results, result)
        progress = progress_html(completed, total, result.label, f"{result.status} / {result.source}")
        yield (
            progress,
            summary,
            throughput_chart,
            ttft_chart,
            tpot_chart,
            batch_chart,
            pd_chart,
            pd_df,
            fixed_update,
            fixed_md,
            fixed_chart,
            fixed_df,
            detail_update,
            detail_md,
            detail_pareto_chart,
            detail_df,
            detail_output,
            display_df,
            display_rows,
            full_rows,
            candidate_rows,
        )


def load_optimizer_history_v2():
    """Load optimizer history into the refactored workspace."""
    rows = STORE.query_rows("throughput_optimizer")
    full_df = _safe_df_from_rows(rows)

    if full_df.empty:
        return (
            progress_html(0, 1, "No History Data", "history"),
            "\n".join(["### Recommendation", "No history results available."]),
            empty_plot("Best Throughput by Device"),
            empty_plot("Best TTFT by Device"),
            empty_plot("Best TPOT by Device"),
            empty_plot("Best Batch Size Comparison"),
            empty_plot("Prefill / Decode QPS Comparison"),
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            "\n".join(["### Fixed-Config Comparison", "No results available."]),
            empty_plot("Fixed-Config Throughput Comparison"),
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            "\n".join(["### Single-Device Search Details", "No results available."]),
            empty_plot("Single-Device Pareto Frontier"),
            pd.DataFrame(),
            "",
            pd.DataFrame(),
            [],
            [],
            [],
        )

    full_df = full_df.copy()
    full_df["deployment_mode"] = full_df.apply(_optimizer_deployment_mode, axis=1)
    summary = _optimizer_summary_markdown_from_df(full_df, len(full_df))
    throughput_chart = _optimizer_metric_plot(
        full_df, "best_throughput", "Best Throughput by Device", "Throughput (token/s)"
    )
    ttft_chart = _optimizer_metric_plot(full_df, "best_ttft_ms", "Best TTFT by Device", "TTFT (ms)")
    tpot_chart = _optimizer_metric_plot(full_df, "best_tpot_ms", "Best TPOT by Device", "TPOT (ms)")
    compare_metric, compare_title, compare_ylabel = _optimizer_primary_metric(full_df)
    batch_chart = _optimizer_metric_plot(full_df, compare_metric, compare_title, compare_ylabel)
    pd_chart, pd_df = _optimizer_pd_ratio_outputs(full_df)
    candidate_rows = _optimizer_candidate_rows_from_records(rows)
    fixed_update, fixed_md, fixed_chart, fixed_df = _optimizer_fixed_compare_outputs(candidate_rows)
    detail_update, detail_md, detail_pareto_chart, detail_df, detail_output = _optimizer_detail_view(
        rows, candidate_rows, None
    )
    display_df = _simplify_optimizer_display_df(full_df)

    return (
        progress_html(len(full_df), len(full_df), "History Loaded", "history"),
        summary,
        throughput_chart,
        ttft_chart,
        tpot_chart,
        batch_chart,
        pd_chart,
        pd_df,
        fixed_update,
        fixed_md,
        fixed_chart,
        fixed_df,
        detail_update,
        detail_md,
        detail_pareto_chart,
        detail_df,
        detail_output,
        display_df,
        df_to_records(display_df),
        rows,
        candidate_rows,
    )


def refresh_optimizer_fixed_compare(candidate_rows: list[dict], config_key: str | None, metric_name: str):
    """Refresh the fixed-configuration comparison results."""
    _update, md, chart, df = _optimizer_fixed_compare_outputs(candidate_rows, config_key, metric_name)
    return md, chart, df


def refresh_optimizer_detail_v2(full_rows: list[dict], candidate_rows: list[dict], device: str):
    """Refresh optimizer details in the refactored workspace."""
    _update, md, pareto_chart, df, raw_output = _optimizer_detail_view(full_rows, candidate_rows, device)
    return md, pareto_chart, df, raw_output


def update_memory_analysis_by_device(full_rows: list[dict], device: str, case_label: str | None = None):
    """Refresh memory analysis for the selected chip and case."""
    from .charts import empty_pie_plot, pie_plot

    if not full_rows:
        return empty_pie_plot("Memory usage"), pd.DataFrame()

    df = _safe_df_from_rows(full_rows)
    if device and "device" in df.columns:
        df = df[df["device"].astype(str) == str(device)]
    df = _filter_df_by_case(df, case_label)

    if df.empty:
        return empty_pie_plot("Memory usage"), pd.DataFrame()

    summary = df.tail(1).iloc[0].to_dict()
    memory_data, memory_table = _memory_analysis_from_summary(summary)
    if memory_data:
        title = f"Memory usage - {device}" + (f" - {case_label}" if case_label else "")
        return pie_plot(memory_data, title), memory_table
    return empty_pie_plot("Memory usage"), memory_table


def update_bandwidth_analysis_by_device(full_rows: list[dict], device: str, case_label: str | None = None):
    """Refresh bandwidth/bottleneck details for the selected chip and case."""
    if not full_rows:
        return pd.DataFrame()

    df = _safe_df_from_rows(full_rows)
    if device and "device" in df.columns:
        df = df[df["device"].astype(str) == str(device)]
    df = _filter_df_by_case(df, case_label)

    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        row_data = {
            "device": row.get("device", "-"),
            "concurrency": row.get("num_queries", "-"),
            "tp_size": row.get("tp_size", "-"),
            "case": row.get("case_label", _case_label_from_mapping(row)),
            "bottleneck_type": row.get("bottleneck_type", "-"),
        }
        for src, dst in [
            ("memory_bound", "memory_bound_pct"),
            ("communication_bound", "communication_bound_pct"),
            ("compute_bound_mma", "compute_mma_bound_pct"),
            ("compute_bound_gp", "compute_gp_bound_pct"),
        ]:
            value = row.get(src)
            if value is not None and value != "":
                try:
                    row_data[dst] = round(float(value), 1)
                except (TypeError, ValueError):
                    row_data[dst] = value
        rows.append(row_data)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def update_category_stats_by_device(op_breakdown: list[dict], device: str, case_label: str | None = None):
    """Refresh operator category statistics for the selected chip and case."""
    if not op_breakdown:
        return empty_plot("Operator category time"), pd.DataFrame()

    df = pd.DataFrame(op_breakdown)
    if device and "device" in df.columns:
        df = df[df["device"].astype(str) == str(device)]
    df = _filter_df_by_case(df, case_label)

    if df.empty:
        return empty_plot("Operator category time"), pd.DataFrame()

    category_stats = (
        df.groupby("category")
        .agg(
            {
                "analytic_total_us": "sum",
                "name": "count",
            }
        )
        .reset_index()
    )
    category_stats.columns = ["category", "total_time_us", "op_count"]
    category_stats["total_time_ms"] = category_stats["total_time_us"] / 1000.0
    category_stats["ratio_pct"] = category_stats["total_time_us"] / category_stats["total_time_us"].sum() * 100
    category_stats = category_stats.sort_values(by="total_time_us", ascending=False)

    display_df = category_stats[["category", "total_time_ms", "op_count", "ratio_pct"]].copy()
    category_df = pd.DataFrame(
        {
            "category": category_stats["category"].tolist(),
            "time_ms": category_stats["total_time_ms"].tolist(),
        }
    )
    title = f"Operator category time - {device}" + (f" - {case_label}" if case_label else "")
    category_chart = bar_plot(
        category_df,
        "category",
        "time_ms",
        title,
        "Time (ms)",
        xlabel="Category",
        group=None,
    )

    return category_chart, _round_numeric_columns(display_df.reset_index(drop=True))


def update_compare_table_by_mode(op_breakdown: list[dict], mode: str, top_n: int = 15):
    """Refresh the operator comparison table according to the selected mode."""
    if not op_breakdown:
        return pd.DataFrame()

    df = pd.DataFrame(op_breakdown)

    # Select the Top N operators by total time for each device
    top_ops_per_device = []
    for device in df["device"].unique():
        device_df = df[df["device"] == device]
        top_ops = device_df.nlargest(top_n, "analytic_total_us")["name"].tolist()
        top_ops_per_device.extend(top_ops)

    # Build a deduplicated union of operators
    unique_ops = list(set(top_ops_per_device))[:top_n]

    # Select the metric column for the chosen mode
    # Note: Radio choices are "Total Time" and "Avg Time" (not "Average Time")
    if mode == "Avg Time" or mode == "Average Time":  # Support both for compatibility
        value_col = "analytic_avg_us"
    else:  # Total time
        value_col = "analytic_total_us"

    # Build the pivot table
    pivot_df = (
        df[df["name"].isin(unique_ops)]
        .pivot_table(index="name", columns="device", values=value_col, aggfunc="sum")
        .fillna(0)
    )

    # Convert to milliseconds
    pivot_df = pivot_df / 1000.0

    # Sort by the maximum value
    max_col = pivot_df.max(axis=1)
    pivot_df = pivot_df.loc[max_col.sort_values(ascending=False).index]

    # Reset the index
    pivot_df = pivot_df.reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={"name": "Operator"})

    return _round_numeric_columns(pivot_df)
