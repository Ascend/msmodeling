from __future__ import annotations

import csv
import logging
import math
import os
import re
import sys
import traceback
from argparse import Namespace
from typing import Any, Optional

from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)


LIST_SEP = ";"


DEFAULT_TPOT_LIMIT_MS = 50.0


FLUSH_BATCH_SIZE = 10


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}


CSV_CONFIG_HEADER = [
    "case_name",
    "device",
    "num_devices",
    "model_id",
    "input_length",
    "output_length",
    "ttft_limits",
    "tpot_limits",
    "tp_sizes",
    "quantize_linear_action",
    "quantize_attention_action",
    "ep_sizes",
    "num_mtp_tokens",
    "mtp_acceptance_rate",
    "compile",
    "mode",
    "max_batched_tokens",
    "batch_range",
    "serving_cost",
    "jobs",
    "log_level",
    "mxfp4_group_size",
    "reserved_memory_gb",
    "compile_allow_graph_break",
]


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=LOG_LEVELS.get(log_level.lower(), logging.INFO),
        format="[%(levelname)s] [%(name)s] %(message)s",
        force=True,
    )


def _parse_list_float(s: Optional[str]) -> list[float]:
    if s is None or not str(s).strip():
        return []
    return [float(x.strip()) for x in str(s).split(LIST_SEP) if x.strip()]


def _parse_list_int(s: Optional[str]) -> Optional[list[int]]:
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(LIST_SEP) if x.strip()]


def _parse_bool(s: Optional[str]) -> bool:
    if s is None or not str(s).strip():
        return False

    normalized = str(s).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value {s!r}; expected true/false, 1/0, or yes/no")


def _parse_mode(s: Optional[str]) -> str:
    normalized = str(s or "agg").strip().lower()

    if normalized not in {"agg", "disagg"}:
        raise ValueError(f"Invalid mode {s!r}; expected 'agg' or 'disagg'")
    return normalized


def _single_limit(values: list[float], name: str) -> Optional[float]:
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"{name} accepts at most one value, got {len(values)}: {values}")
    return values[0]


def _parse_parallel(s: str) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    if not s or not isinstance(s, str):
        return None, None, None, None
    s = s.strip()
    m = re.match(r"tp(\d+)pp(\d+)dp(\d+)", s.lower())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), None

    tp = pp = dp = ep = None
    for part in s.split("|"):
        part = part.strip().lower()
        kv = re.match(r"(tp|pp|dp|ep)\s*=\s*(\d+)", part)
        if kv:
            val = int(kv.group(2))
            if kv.group(1) == "tp":
                tp = val
            elif kv.group(1) == "pp":
                pp = val
            elif kv.group(1) == "dp":
                dp = val
            elif kv.group(1) == "ep":
                ep = val
    if tp is not None or pp is not None or dp is not None or ep is not None:
        return tp, pp, dp, ep
    return None, None, None, None


def _parse_breakdown(
    s: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not s or not isinstance(s, str):
        return None, None, None, None
    mem = comm = cube = vec = None
    for part in s.split("|"):
        part = part.strip()
        if part.startswith("Mem "):
            mem = part.replace("Mem ", "").strip()
        elif part.startswith("Comm "):
            comm = part.replace("Comm ", "").strip()
        elif part.startswith("Cube "):
            cube = part.replace("Cube ", "").strip()
        elif part.startswith("Vec "):
            vec = part.replace("Vec ", "").strip()
    return mem, comm, cube, vec


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _parse_case(row: dict, row_number: int) -> dict:
    case_name = (row.get("case_name") or "").strip()
    if not case_name:
        case_name = f"row_{row_number}"

    try:
        ttft_limits = _parse_list_float(row.get("ttft_limits"))
        tpot_limits = _parse_list_float(row.get("tpot_limits"))

        if not tpot_limits:
            tpot_limits = [DEFAULT_TPOT_LIMIT_MS]
    except ValueError as e:
        raise ValueError(f"Row case_name={case_name}: ttft_limits/tpot_limits parse error: {e}") from e

    q_linear = (row.get("quantize_linear_action") or "").strip()
    q_attn = (row.get("quantize_attention_action") or "").strip()
    if q_linear:
        try:
            linear_action = QuantizeLinearAction(q_linear)
        except ValueError:
            valid = ", ".join(e.value for e in QuantizeLinearAction)

            raise ValueError(
                f"Row case_name={case_name}: invalid quantize_linear_action '{q_linear}'. Valid options: {valid}"
            ) from None
    else:
        linear_action = None

    if q_attn:
        try:
            attn_action = QuantizeAttentionAction(q_attn)
        except ValueError:
            valid = ", ".join(e.value for e in QuantizeAttentionAction)
            raise ValueError(
                f"Row case_name={case_name}: invalid quantize_attention_action '{q_attn}'. Valid options: {valid}"
            ) from None
    else:
        attn_action = None

    mtp_rate = _parse_list_float(row.get("mtp_acceptance_rate"))
    if not mtp_rate:
        mtp_rate = None

    max_pf = (row.get("max_batched_tokens") or "").strip()
    batch_range_raw = _parse_list_int(row.get("batch_range"))
    serving_cost_val = (row.get("serving_cost") or "").strip()
    jobs_val = (row.get("jobs") or "").strip()
    log_level = (row.get("log_level") or "info").strip().lower() or "info"
    mxfp_val = (row.get("mxfp4_group_size") or "").strip()

    reserved_val = (row.get("reserved_memory_gb") or "").strip()

    compile_allow_graph_break = _parse_bool(row.get("compile_allow_graph_break"))

    try:
        num_devices = int((row.get("num_devices") or "1").strip())
        input_length = int((row.get("input_length") or "0").strip())
        output_length = int((row.get("output_length") or "0").strip())
        num_mtp_tokens = int((row.get("num_mtp_tokens") or "0").strip())
        max_batched_tokens = int(max_pf) if max_pf else 8192
        serving_cost = float(serving_cost_val) if serving_cost_val else 0.0
        jobs = int(jobs_val) if jobs_val else 8
        mxfp4_group_size = int(mxfp_val) if mxfp_val else 32

        reserved_memory_gb = float(reserved_val) if reserved_val else None
    except (ValueError, TypeError) as e:
        raise ValueError(f"Row case_name={case_name}: numeric field parse error: {e}") from e

    return {
        "case_name": case_name,
        "device": (row.get("device") or "").strip(),
        "num_devices": num_devices,
        "model_id": (row.get("model_id") or "").strip(),
        "input_length": input_length,
        "output_length": output_length,
        "ttft_limits": ttft_limits,
        "tpot_limits": tpot_limits,
        "tp_sizes": _parse_list_int(row.get("tp_sizes")),
        "quantize_linear_action": linear_action,
        "quantize_attention_action": attn_action,
        "ep_sizes": _parse_list_int(row.get("ep_sizes")),
        "num_mtp_tokens": num_mtp_tokens,
        "mtp_acceptance_rate": mtp_rate,
        "do_compile": _parse_bool(row.get("compile")),
        "mode": _parse_mode(row.get("mode")),
        "max_batched_tokens": max_batched_tokens,
        "batch_range": batch_range_raw,
        "serving_cost": serving_cost,
        "jobs": jobs,
        "log_level": log_level,
        "mxfp4_group_size": mxfp4_group_size,
        "reserved_memory_gb": reserved_memory_gb,
        "compile_allow_graph_break": compile_allow_graph_break,
    }


def load_cases_from_csv(csv_path: str) -> list[dict]:
    cases: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {csv_path}")

        required_columns = {
            "device",
            "num_devices",
            "model_id",
            "input_length",
            "output_length",
        }
        missing = [c for c in required_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            if not any((row.get(k) or "").strip() for k in CSV_CONFIG_HEADER):
                continue
            try:
                cases.append(_parse_case(row, row_number))
            except (TypeError, ValueError) as exc:
                case_name = (row.get("case_name") or "").strip() or f"row_{row_number}"
                cases.append(
                    {
                        "case_name": case_name,
                        "parse_error": str(exc),
                        "log_level": "info",
                        "device": (row.get("device") or "").strip(),
                        "num_devices": 0,
                        "model_id": (row.get("model_id") or "").strip(),
                        "input_length": 0,
                        "output_length": 0,
                    }
                )
    return cases


def _build_optimizer_args(case_dict: dict, base_args: Any) -> Namespace:
    ttft = _single_limit(case_dict["ttft_limits"], "ttft_limits")
    tpot = _single_limit(case_dict["tpot_limits"], "tpot_limits")
    disagg = case_dict["mode"] == "disagg"

    q_linear = case_dict["quantize_linear_action"] or QuantizeLinearAction.W8A8_DYNAMIC
    q_attn = case_dict["quantize_attention_action"] or QuantizeAttentionAction.DISABLED
    mtp_rate = case_dict["mtp_acceptance_rate"] or [0.9, 0.6, 0.4, 0.2]

    tp_sizes = case_dict["tp_sizes"]
    ep_sizes = case_dict["ep_sizes"]
    moe_dp_sizes = getattr(base_args, "moe_dp_sizes", None)
    if all(x is None for x in (tp_sizes, ep_sizes, moe_dp_sizes)):
        tp_sizes = []

    return Namespace(
        input_length=case_dict["input_length"],
        output_length=case_dict["output_length"],
        device=case_dict["device"],
        model_id=case_dict["model_id"],
        num_devices=case_dict["num_devices"],
        compile=case_dict["do_compile"],
        compile_allow_graph_break=case_dict["compile_allow_graph_break"],
        num_mtp_tokens=case_dict["num_mtp_tokens"],
        mtp_acceptance_rate=mtp_rate,
        quantize_linear_action=q_linear,
        mxfp4_group_size=case_dict["mxfp4_group_size"],
        quantize_attention_action=q_attn,
        reserved_memory_gb=(
            case_dict["reserved_memory_gb"]
            if case_dict["reserved_memory_gb"] is not None
            else float(getattr(base_args, "reserved_memory_gb", 10.0))
        ),
        tp_sizes=tp_sizes,
        ttft_limits=ttft,
        tpot_limits=tpot,
        max_batched_tokens=case_dict["max_batched_tokens"],
        batch_range=case_dict["batch_range"],
        serving_cost=case_dict["serving_cost"],
        disagg=disagg,
        jobs=case_dict["jobs"],
        log_level=case_dict["log_level"],
        dump_original_results=getattr(base_args, "dump_original_results", False),
        image_batch_size=getattr(base_args, "image_batch_size", None),
        image_height=getattr(base_args, "image_height", None),
        image_width=getattr(base_args, "image_width", None),
        prefill_devices_per_instance=getattr(base_args, "prefill_devices_per_instance", None),
        decode_devices_per_instance=getattr(base_args, "decode_devices_per_instance", None),
        prefix_cache_hit_rate=getattr(base_args, "prefix_cache_hit_rate", 0.0),
        enable_multistream=getattr(base_args, "enable_multistream", True),
        enable_optimize_prefill_decode_ratio=getattr(base_args, "enable_optimize_prefill_decode_ratio", False),
        ep_sizes=case_dict["ep_sizes"],
        moe_dp_sizes=getattr(base_args, "moe_dp_sizes", None),
        concurrency_search_strategy=getattr(base_args, "concurrency_search_strategy", "exponential"),
    )


def _filter_best_row(summary: Any):
    try:
        import pandas as pd
    except ImportError:
        return None
    df = summary.get_summary_df() if hasattr(summary, "get_summary_df") else None
    if df is None or df.empty:
        return None
    dc = summary.data_config if hasattr(summary, "data_config") else None
    tpot_limit = (dc.tpot_limits if dc is not None else None) or float("inf")
    ttft_limit = (dc.ttft_limits if dc is not None else None) or float("inf")
    mask = (pd.to_numeric(df["tpot"], errors="coerce").fillna(float("inf")) <= tpot_limit) & (
        pd.to_numeric(df["ttft"], errors="coerce").fillna(float("inf")) <= ttft_limit
    )
    filtered = (
        df[mask]
        .sort_values(by="token/s", ascending=False)
        .groupby("parallel")
        .first()
        .reset_index()
        .sort_values(by="token/s", ascending=False)
        .reset_index(drop=True)
    )
    if filtered.empty:
        return None
    return filtered.iloc[0]


def _csv_val(x: Any, fmt: Optional[str] = None) -> str:
    if x is None:
        return ""
    if fmt is not None and isinstance(fmt, str):
        try:
            return fmt.format(x)
        except (ValueError, TypeError):
            return str(x)
    return str(x)


def _summary_to_csv_row(case_dict: dict, summary_result: list, best_rows: Optional[list] = None) -> list:
    case_name = case_dict["case_name"]
    device = case_dict["device"]
    num_devices = case_dict["num_devices"]
    model_id = case_dict["model_id"]
    input_length = case_dict["input_length"]
    output_length = case_dict["output_length"]
    tpot_limit = _single_limit(case_dict["tpot_limits"], "tpot_limits")
    ttft_limit = _single_limit(case_dict["ttft_limits"], "ttft_limits")
    num_mtp_tokens = case_dict["num_mtp_tokens"]
    d_lin = d_attn = d_use_ep = d_mtp = None
    d_slo = d_conc = d_tpot = d_tps = d_tps_dev = None
    d_mem = d_comm = d_cube = d_vec = None
    d_tp = d_pp = d_dp = None
    p_lin = p_attn = p_use_ep = p_mtp = None
    p_slo = p_conc = p_ttft = p_tps = p_tps_dev = None
    p_mem = p_comm = p_cube = p_vec = None
    p_tp = p_pp = p_dp = None

    def _quant_str(v) -> str:
        if v is None:
            return ""
        return getattr(v, "value", str(v))

    def set_decode_from_row(row) -> None:
        nonlocal d_lin, d_attn, d_use_ep, d_mtp, d_slo, d_conc, d_tpot, d_tps, d_tps_dev
        nonlocal d_mem, d_comm, d_cube, d_vec, d_tp, d_pp, d_dp
        d_lin = _quant_str(row.get("quantize_linear_action"))
        d_attn = _quant_str(row.get("quantize_attention_action"))
        _tp, _pp, _dp, _ep = _parse_parallel(str(row.get("parallel", "")))
        d_tp, d_pp, d_dp = _tp, _pp, _dp
        d_slo = tpot_limit
        d_conc = _safe_int(row.get("concurrency"))
        d_tpot = _safe_float(row.get("tpot"))
        d_tps = _safe_float(row.get("token/s"))
        d_tps_dev = _safe_float(row.get("token/s/device"))
        d_use_ep = str(_ep) if _ep is not None and _ep > 1 else ""
        d_mtp = num_mtp_tokens
        pbd = row.get("percentage_breakdowns(d)") or row.get("percentage_breakdowns")
        if pbd is not None:
            d_mem, d_comm, d_cube, d_vec = _parse_breakdown(str(pbd))

    def set_prefill_from_row(row) -> None:
        nonlocal p_lin, p_attn, p_use_ep, p_mtp, p_slo, p_conc, p_ttft, p_tps, p_tps_dev
        nonlocal p_mem, p_comm, p_cube, p_vec, p_tp, p_pp, p_dp
        p_lin = _quant_str(row.get("quantize_linear_action"))
        p_attn = _quant_str(row.get("quantize_attention_action"))
        _tp, _pp, _dp, _ep = _parse_parallel(str(row.get("parallel", "")))
        p_tp, p_pp, p_dp = _tp, _pp, _dp
        p_slo = ttft_limit
        p_conc = _safe_int(row.get("concurrency"))
        p_ttft = _safe_float(row.get("ttft"))
        p_tps = _safe_float(row.get("token/s"))
        p_tps_dev = _safe_float(row.get("token/s/device"))
        p_use_ep = str(_ep) if _ep is not None and _ep > 1 else ""
        p_mtp = num_mtp_tokens
        pbd = row.get("percentage_breakdowns(p)") or row.get("percentage_breakdowns")
        if pbd is not None:
            p_mem, p_comm, p_cube, p_vec = _parse_breakdown(str(pbd))

    mode = case_dict.get("mode", "agg")
    for idx, summary in enumerate(summary_result):
        data_config = summary.data_config if hasattr(summary, "data_config") else None
        if data_config is None:
            continue
        if best_rows is not None and idx < len(best_rows):
            best_row = best_rows[idx]
        else:
            best_row = _filter_best_row(summary)
        if best_row is None:
            continue

        if mode == "agg":
            set_decode_from_row(best_row)
            set_prefill_from_row(best_row)
            continue

        is_prefill = data_config.ttft_limits is not None and data_config.tpot_limits is None
        is_decode = data_config.tpot_limits is not None and data_config.ttft_limits is None

        if is_decode:
            set_decode_from_row(best_row)
        elif is_prefill:
            set_prefill_from_row(best_row)
        else:
            row_tpot = _safe_float(best_row.get("tpot"))
            row_ttft = _safe_float(best_row.get("ttft"))
            if row_tpot is not None:
                set_decode_from_row(best_row)
            if row_ttft is not None:
                set_prefill_from_row(best_row)

    def _fmt2(v):
        return "{:.2f}" if v is not None else None

    def _fmt1(v):
        return "{:.1f}" if v is not None else None

    decode_specs = [
        (d_lin, None),
        (d_attn, None),
        (d_use_ep, None),
        (d_mtp, None),
        (d_slo, _fmt2(d_slo)),
        (d_conc, None),
        (d_tpot, _fmt2(d_tpot)),
        (d_tps, _fmt1(d_tps)),
        (d_tps_dev, _fmt1(d_tps_dev)),
        (d_mem, None),
        (d_comm, None),
        (d_cube, None),
        (d_vec, None),
        (d_tp, None),
        (d_pp, None),
        (d_dp, None),
    ]
    prefill_specs = [
        (p_lin, None),
        (p_attn, None),
        (p_use_ep, None),
        (p_mtp, None),
        (p_slo, _fmt2(p_slo)),
        (p_conc, None),
        (p_ttft, _fmt2(p_ttft)),
        (p_tps, _fmt1(p_tps)),
        (p_tps_dev, _fmt1(p_tps_dev)),
        (p_mem, None),
        (p_comm, None),
        (p_cube, None),
        (p_vec, None),
        (p_tp, None),
        (p_pp, None),
        (p_dp, None),
    ]
    return [
        case_name,
        device,
        num_devices,
        input_length,
        output_length,
        model_id,
        *[_csv_val(v, fmt) for v, fmt in decode_specs],
        *[_csv_val(v, fmt) for v, fmt in prefill_specs],
        "",
        "",
    ]


def _write_op_csv(filepath: str, phases: list) -> None:
    header = ["Phase", "Op_Name", "Perf_Model", "Perf_Total_s", "Perf_Avg_s", "Call_Times"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for phase_name, ops in phases:
            if ops is None or not hasattr(ops, "rows"):
                continue
            for r in ops.rows:
                writer.writerow(
                    [
                        phase_name,
                        r.get("name", ""),
                        r.get("perf_model", ""),
                        r.get("perf_total", 0.0),
                        r.get("perf_avg", 0.0),
                        r.get("call_times", 0),
                    ]
                )


def _export_op_profile_csv(
    case_dict: dict, summary_result: list, out_dir: str, best_rows: Optional[list] = None
) -> None:
    case_name = case_dict.get("case_name", "unknown_case")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in case_name)
    op_dir = os.path.join(out_dir, "op_profiles") if out_dir else "op_profiles"
    has_any = False

    for idx, summary in enumerate(summary_result):
        if summary is None or not hasattr(summary, "get_summary_df"):
            continue
        if best_rows is not None and idx < len(best_rows):
            best_row = best_rows[idx]
        else:
            best_row = _filter_best_row(summary)
        if best_row is None:
            continue
        parallel = str(best_row.get("parallel", ""))
        batch_size = _safe_int(best_row.get("batch_size"))
        if batch_size is None:
            continue
        op_map = None
        if hasattr(summary, "get_op_profile_for"):
            op_map = summary.get_op_profile_for(parallel, batch_size)
        if not op_map:
            continue

        has_prefill = op_map.get("prefill") is not None
        has_decode = op_map.get("decode") is not None

        if has_prefill and has_decode:
            phases = [("prefill", op_map["prefill"]), ("decode", op_map["decode"])]
            if not has_any:
                os.makedirs(op_dir, exist_ok=True)
                has_any = True
            _write_op_csv(os.path.join(op_dir, f"{safe_name}.csv"), phases)
        elif has_prefill:
            if not has_any:
                os.makedirs(op_dir, exist_ok=True)
                has_any = True
            _write_op_csv(os.path.join(op_dir, f"{safe_name}_prefill.csv"), [("prefill", op_map["prefill"])])
        elif has_decode:
            if not has_any:
                os.makedirs(op_dir, exist_ok=True)
                has_any = True
            _write_op_csv(os.path.join(op_dir, f"{safe_name}_decode.csv"), [("decode", op_map["decode"])])


def _csv_header_and_ref_row() -> tuple[list, list]:
    linear_quant_options = ", ".join(e.value for e in QuantizeLinearAction)
    attn_quant_options = ", ".join(e.value for e in QuantizeAttentionAction)
    header = [
        "Case_Name",
        "Device Type",
        "Number of Devices",
        "Input Length",
        "Output Length",
        "Model",
        "Decode_Linear Quant Type",
        "Decode_Attn Quant Type",
        "Decode_EP Size",
        "Decode_MTP Tokens",
        "Decode_TPOT Target(ms)",
        "Decode_Concurrency",
        "Decode_TPOT(ms)",
        "Decode_Total TPS",
        "Decode_TPS/Device",
        "Decode_Mem",
        "Decode_Comm",
        "Decode_Cube",
        "Decode_Vec",
        "Decode_TP Size",
        "Decode_PP Size",
        "Decode_DP Size",
        "Prefill_Linear Quant Type",
        "Prefill_Attn Quant Type",
        "Prefill_EP Size",
        "Prefill_MTP Tokens",
        "Prefill_TTFT Target(ms)",
        "Prefill_Concurrency",
        "Prefill_TTFT(ms)",
        "Prefill_Total TPS",
        "Prefill_TPS/Device",
        "Prefill_Mem",
        "Prefill_Comm",
        "Prefill_Cube",
        "Prefill_Vec",
        "Prefill_TP Size",
        "Prefill_PP Size",
        "Prefill_DP Size",
        "QuantizeLinearAction_options",
        "QuantizeAttentionAction_options",
    ]
    ref_row = [""] * (len(header) - 2) + [linear_quant_options, attn_quant_options]
    return header, ref_row


def _empty_row_for_case(case_dict: dict) -> list:
    header, _ = _csv_header_and_ref_row()
    row = [""] * len(header)
    row[0] = case_dict["case_name"]
    row[1] = case_dict["device"]
    row[2] = case_dict["num_devices"]
    row[3] = case_dict["input_length"]
    row[4] = case_dict["output_length"]
    row[5] = case_dict["model_id"]
    return row


def _print_case_summary(case_dict: dict, row: list) -> None:
    has_prefill = bool(row[29])  # Prefill_Total_TPS
    has_decode = bool(row[13])  # Decode_Total_TPS
    if has_prefill or has_decode:
        print("  " + "-" * 76)
        if case_dict["mode"] == "agg":
            tps = row[13] or row[29]
            if tps:
                print("  Overall Best Configuration: ")
                print(f"    Best Throughput: {tps} tokens/s")
                if row[28]:
                    print(f"    TTFT: {row[28]} ms")
                if row[12]:
                    print(f"    TPOT: {row[12]} ms")
        else:
            if has_prefill:
                print("  Overall Best Configuration (Prefill): ")
                print(f"    Best Throughput: {row[29]} tokens/s")
                if row[28]:
                    print(f"    TTFT: {row[28]} ms")
            if has_decode:
                if has_prefill:
                    print("  " + "-" * 76)
                print("  Overall Best Configuration (Decode): ")
                print(f"    Best Throughput: {row[13]} tokens/s")
                if row[12]:
                    print(f"    TPOT: {row[12]} ms")
        print("  " + "-" * 76)

    if row[14]:
        print(f"Best decode: TP={row[19]}, TPS/Device={row[14]}, TPOT={row[12]}ms")
    if row[30]:
        print(f"Best prefill: TP={row[35]}, TPS/Device={row[30]}, TTFT={row[28]}ms")


def _print_results_summary(all_rows: list, output_file: str) -> None:
    for case_dict, row in all_rows:
        print(f"\nCase {case_dict['case_name']} Results:")
        if row[14]:  # Decode_TPS/Device
            print(
                f"  Decode - TPOT: {row[12]}ms, "
                f"TPS/Device: {row[14]}, "
                f"TP={row[19]}, PP={row[20]}, DP={row[21]}, "
                f"Concurrency: {row[11]}"
            )
        if row[30]:  # Prefill_TPS/Device
            print(
                f"  Prefill - TTFT: {row[28]}ms, "
                f"TPS/Device: {row[30]}, "
                f"TP={row[35]}, PP={row[36]}, DP={row[37]}, "
                f"Concurrency: {row[27]}"
            )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for case_dict, row in all_rows:
        print(f"\n{case_dict['case_name']}:")
        if row[14]:  # Decode_TPS/Device
            print(f"  Best Decode TPS/Device: {row[14]}")
            print(f"  Best Decode TPOT: {row[12]}ms")
            print(f"  Best Decode Config: TP={row[19]}, PP={row[20]}, DP={row[21]}")
        if row[30]:  # Prefill_TPS/Device
            print(f"  Best Prefill TPS/Device: {row[30]}")
            print(f"  Best Prefill TTFT: {row[28]}ms")
            print(f"  Best Prefill Config: TP={row[35]}, PP={row[36]}, DP={row[37]}")
    print(f"\nAll results saved to: {output_file}")
    print("=" * 80)


def run_cases_and_save(input_csv: str, output_csv: str, base_args: Any, export_op_profile: bool = False) -> None:
    from serving_cast.parallel_runner import ParallelRunner

    cases = load_cases_from_csv(input_csv)
    if not cases:
        print("No cases to run.", file=sys.stderr)
        return

    log_levels_used = {c["log_level"] for c in cases}
    if len(log_levels_used) > 1:
        print(
            f"Warning: cases use multiple log_levels {log_levels_used}; "
            f"using '{cases[0]['log_level']}' for the whole batch.",
            file=sys.stderr,
        )
    _configure_logging(cases[0]["log_level"])

    print("=" * 80)
    print("Benchmark Cases Runner")
    print("=" * 80)
    print(f"Total cases: {len(cases)}")
    print("Mode: sequential (one case at a time, result written after each case)")
    print("=" * 80)

    header, ref_row = _csv_header_and_ref_row()
    all_rows: list = []

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(ref_row)
        f.flush()

        for idx, case_dict in enumerate(cases, 1):
            print(f"\n[{idx}/{len(cases)}] Processing case: {case_dict['case_name']}")

            if case_dict.get("parse_error"):
                print(
                    f"Skip invalid case {case_dict['case_name']}: {case_dict['parse_error']}",
                    file=sys.stderr,
                )

                row = _empty_row_for_case(case_dict)
                all_rows.append((case_dict, row))
                writer.writerow(row)
                if idx % FLUSH_BATCH_SIZE == 0:
                    f.flush()

                continue
            try:
                args = _build_optimizer_args(case_dict, base_args)

                print(f"\n{'=' * 80}")
                print(f"Running case: {case_dict['case_name']}")
                print(f"{'=' * 80}")
                print(f"Device: {case_dict['device']}, Num Devices: {case_dict['num_devices']}")
                print(f"Model: {case_dict['model_id']}")
                print(f"Input Length: {case_dict['input_length']}, Output Length: {case_dict['output_length']}")
                print(f"TTFT Limits: {case_dict['ttft_limits']}, TPOT Limits: {case_dict['tpot_limits']}")
                print(f"Mode: {case_dict['mode']}")
                print("=" * 80)

                runner = ParallelRunner(args)
                if args.disagg:
                    summary_result = runner.run_disagg()
                else:
                    summary_result = runner.run_agg()

                cached_best_rows = (
                    [_filter_best_row(summary) if summary is not None else None for summary in summary_result]
                    if summary_result
                    else []
                )

                row = _summary_to_csv_row(case_dict, summary_result, best_rows=cached_best_rows)
                if export_op_profile and summary_result:
                    _export_op_profile_csv(
                        case_dict, summary_result, os.path.dirname(output_csv), best_rows=cached_best_rows
                    )
                _print_case_summary(case_dict, row)

            except Exception as e:
                print(f"Case {case_dict['case_name']} failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                row = _empty_row_for_case(case_dict)

            all_rows.append((case_dict, row))
            writer.writerow(row)
            if idx % FLUSH_BATCH_SIZE == 0:
                f.flush()
        f.flush()

    _print_results_summary(all_rows, output_csv)
