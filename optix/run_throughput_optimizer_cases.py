#!/usr/bin/env python
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CSV-driven batch runner for optix.run_throughput_optimizer_cases.

Load benchmark cases from CSV, call throughput_optimizer sequentially per case,
and aggregate results into a single output CSV.

Usage:
    python -m optix.run_throughput_optimizer_cases --input-csv cases.csv --output-csv results.csv
    python -m optix.run_throughput_optimizer_cases --write-template cases_template.csv

Note: ttft_limits and tpot_limits use milliseconds (ms) as the unit, consistent with throughput_optimizer.
"""

import argparse
import csv
import logging
import math
import re
import sys
import traceback
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any

# Try importing tensor_cast directly; only fall back to sys.path hack if needed.
# This avoids polluting sys.path when this module is imported by another entry point
# in an already-installed environment (e.g., `python -m optix....`).
try:
    from tensor_cast.core.quantization.datatypes import (
        QuantizeLinearAction,
        QuantizeAttentionAction,
    )
except ImportError:
    # Fallback: assume we are run as a script from a repo checkout without install.
    # Add repo root: this_file -> optix -> experimental -> repo root
    _project_root = str(Path(__file__).resolve().parents[2])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from tensor_cast.core.quantization.datatypes import (
        QuantizeLinearAction,
        QuantizeAttentionAction,
    )

LIST_SEP = ";"

# Default TPOT SLO limit in milliseconds, consistent with throughput_optimizer's ms unit
DEFAULT_TPOT_LIMIT_MS = 50.0

# Number of cases to accumulate before flushing CSV to disk
FLUSH_BATCH_SIZE = 10

# Logging level name → level mapping (mirrors serving_cast.service.utils.LOG_LEVELS)
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}


def _configure_logging(log_level: str) -> None:
    """Configure global logging once per batch run.

    Note: this calls logging.basicConfig(force=True), which replaces the root
    handler. Callers (e.g., other libraries) that share this process should be
    aware of the global side effect. Intended to be called once at the start of
    a batch by run_cases_and_save (not per case).
    """
    logging.basicConfig(
        level=LOG_LEVELS.get(log_level.lower(), logging.INFO),
        format="[%(levelname)s] [%(name)s] %(message)s",
        force=True,
    )


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
    "max_prefill_tokens",
    "batch_range",
    "serving_cost",
    "jobs",
    "log_level",
    "mxfp4_group_size",
    "reserved_memory_gb",
    "compile_allow_graph_break",
]


@dataclass
class BenchmarkCase:
    """Single benchmark case config, aligned with throughput_optimizer arguments."""

    case_name: str
    device: str
    num_devices: int
    model_id: str
    input_length: int
    output_length: int
    ttft_limits: List[float]
    tpot_limits: List[float]
    tp_sizes: Optional[List[int]] = None
    quantize_linear_action: Optional[QuantizeLinearAction] = None
    quantize_attention_action: Optional[QuantizeAttentionAction] = None
    ep_sizes: Optional[List[int]] = None
    num_mtp_tokens: int = 0
    mtp_acceptance_rate: Optional[List[float]] = None
    do_compile: bool = False
    mode: str = "agg"
    max_prefill_tokens: int = 8192
    batch_range: Optional[List[int]] = None
    serving_cost: float = 0.0
    jobs: int = 8
    log_level: str = "info"
    mxfp4_group_size: int = 32
    reserved_memory_gb: float = 0.0
    compile_allow_graph_break: bool = False


@dataclass
class BenchmarkResult:
    """Benchmark result for one case (aligned with throughput_optimizer output and CSV header)."""

    case_name: str
    device: str
    num_devices: int
    model_id: str
    input_length: int
    output_length: int
    # Decode result columns
    best_decode_linear_quant_type: Optional[str] = None
    best_decode_attn_quant_type: Optional[str] = None
    best_decode_tp_size: Optional[int] = None
    best_decode_use_ep: Optional[str] = None
    best_decode_mtp_tokens: Optional[int] = None
    best_decode_slo_target_ms: Optional[float] = None
    best_decode_concurrency: Optional[int] = None
    best_decode_tpot_ms: Optional[float] = None
    best_decode_total_tps: Optional[float] = None
    best_decode_tps_per_device: Optional[float] = None
    best_decode_mem_pct: Optional[str] = None
    best_decode_comm_pct: Optional[str] = None
    best_decode_cube_pct: Optional[str] = None
    best_decode_vec_pct: Optional[str] = None
    best_decode_pp_size: Optional[int] = None
    best_decode_dp_size: Optional[int] = None
    # Prefill result columns
    best_prefill_linear_quant_type: Optional[str] = None
    best_prefill_attn_quant_type: Optional[str] = None
    best_prefill_tp_size: Optional[int] = None
    best_prefill_use_ep: Optional[str] = None
    best_prefill_mtp_tokens: Optional[int] = None
    best_prefill_slo_target_ms: Optional[float] = None
    best_prefill_concurrency: Optional[int] = None
    best_prefill_ttft_ms: Optional[float] = None
    best_prefill_total_tps: Optional[float] = None
    best_prefill_tps_per_device: Optional[float] = None
    best_prefill_mem_pct: Optional[str] = None
    best_prefill_comm_pct: Optional[str] = None
    best_prefill_cube_pct: Optional[str] = None
    best_prefill_vec_pct: Optional[str] = None
    best_prefill_pp_size: Optional[int] = None
    best_prefill_dp_size: Optional[int] = None


def _parse_list_float(s: Optional[str]) -> List[float]:
    """Parse list-of-float string with LIST_SEP; return [] for empty/blank."""
    if s is None or not str(s).strip():
        return []
    return [float(x.strip()) for x in str(s).split(LIST_SEP) if x.strip()]


def _parse_list_int(s: Optional[str]) -> Optional[List[int]]:
    """Parse list-of-int string; return None for empty/blank."""
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(LIST_SEP) if x.strip()]


def _parse_bool(s: Optional[str]) -> bool:
    """Parse bool: true/1/yes -> True, else (including empty) -> False."""
    if s is None:
        return False
    v = str(s).strip().lower()
    return v in ("true", "1", "yes")


def _parse_optional_bool(s: Optional[str]) -> Optional[bool]:
    """Parse optional bool: empty -> None, true/1/yes -> True, false/0/no -> False."""
    if s is None or not str(s).strip():
        return None
    v = str(s).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return None


def _parse_mode(s: Optional[str]) -> str:
    """Parse mode: agg or disagg; default agg if empty or invalid."""
    if s is None or not str(s).strip():
        return "agg"
    v = str(s).strip().lower()
    if v == "disagg":
        return "disagg"
    return "agg"


def _single_limit(values: List[float], name: str) -> Optional[float]:
    """Extract a single limit value; raise if more than one provided."""
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"{name} accepts at most one value, got {len(values)}: {values}")
    return values[0]


def load_cases_from_csv(csv_path: str) -> List[BenchmarkCase]:
    """Load case list from CSV. Header must match CSV_CONFIG_HEADER by name.
    List fields use LIST_SEP (;) in cells, e.g. ttft_limits=1.0;2.0;3.0.
    """
    cases = []
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
        for row in reader:
            if not any((row.get(k) or "").strip() for k in CSV_CONFIG_HEADER):
                continue
            case_name = (row.get("case_name") or "").strip()
            if not case_name:
                case_name = f"row_{len(cases) + 1}"
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
                        f"Row case_name={case_name}: invalid quantize_linear_action '{q_linear}'. "
                        f"Valid options: {valid}"
                    ) from None
            else:
                linear_action = None
            if q_attn:
                try:
                    attn_action = QuantizeAttentionAction(q_attn)
                except ValueError:
                    valid = ", ".join(e.value for e in QuantizeAttentionAction)
                    raise ValueError(
                        f"Row case_name={case_name}: invalid quantize_attention_action '{q_attn}'. "
                        f"Valid options: {valid}"
                    ) from None
            else:
                attn_action = None

            mtp_rate = _parse_list_float(row.get("mtp_acceptance_rate"))
            if not mtp_rate:
                mtp_rate = None

            max_pf = (row.get("max_prefill_tokens") or "").strip()
            max_prefill_tokens = int(max_pf) if max_pf else 8192
            batch_range_raw = _parse_list_int(row.get("batch_range"))
            serving_cost_val = (row.get("serving_cost") or "").strip()
            serving_cost = float(serving_cost_val) if serving_cost_val else 0.0
            jobs_val = (row.get("jobs") or "").strip()
            jobs = int(jobs_val) if jobs_val else 8
            log_level = (row.get("log_level") or "info").strip().lower() or "info"
            mxfp_val = (row.get("mxfp4_group_size") or "").strip()
            mxfp4_group_size = int(mxfp_val) if mxfp_val else 32
            reserved_val = (row.get("reserved_memory_gb") or "").strip()
            reserved_memory_gb = float(reserved_val) if reserved_val else 0.0
            compile_allow_graph_break = _parse_bool(row.get("compile_allow_graph_break"))

            cases.append(
                BenchmarkCase(
                    case_name=case_name,
                    device=(row.get("device") or "").strip(),
                    num_devices=int((row.get("num_devices") or "1").strip()),
                    model_id=(row.get("model_id") or "").strip(),
                    input_length=int((row.get("input_length") or "0").strip()),
                    output_length=int((row.get("output_length") or "0").strip()),
                    ttft_limits=ttft_limits,
                    tpot_limits=tpot_limits,
                    tp_sizes=_parse_list_int(row.get("tp_sizes")),
                    quantize_linear_action=linear_action,
                    quantize_attention_action=attn_action,
                    ep_sizes=_parse_list_int(row.get("ep_sizes")),
                    num_mtp_tokens=int((row.get("num_mtp_tokens") or "0").strip()),
                    mtp_acceptance_rate=mtp_rate,
                    do_compile=_parse_bool(row.get("compile")),
                    mode=_parse_mode(row.get("mode")),
                    max_prefill_tokens=max_prefill_tokens,
                    batch_range=batch_range_raw,
                    serving_cost=serving_cost,
                    jobs=jobs,
                    log_level=log_level,
                    mxfp4_group_size=mxfp4_group_size,
                    reserved_memory_gb=reserved_memory_gb,
                    compile_allow_graph_break=compile_allow_graph_break,
                )
            )
    return cases


def write_template_csv(csv_path: str) -> None:
    """Generate template CSV with all config headers and multiple example rows."""
    examples = [
        # 1-card agg mode
        [
            "1card_agg_w8a8",
            "ATLAS_800_A3_752T_128G_DIE",
            "1",
            "Qwen/Qwen3-32B",
            "16000",
            "1000",
            "",
            str(int(DEFAULT_TPOT_LIMIT_MS)),
            "",
            "W8A8_DYNAMIC",
            "DISABLED",
            "",
            "0",
            "",
            "true",
            "agg",
            "8192",
            "",
            "0",
            "8",
            "info",
            "32",
            "0",
            "false",
        ],
        # 8-card agg mode
        [
            "8card_agg_w8a8",
            "ATLAS_800_A3_752T_128G_DIE",
            "8",
            "Qwen/Qwen3-32B",
            "3500",
            "1500",
            "",
            str(int(DEFAULT_TPOT_LIMIT_MS)),
            "",
            "W8A8_DYNAMIC",
            "DISABLED",
            "",
            "0",
            "",
            "true",
            "agg",
            "8192",
            "",
            "0",
            "8",
            "info",
            "32",
            "0",
            "false",
        ],
        # 4-card disagg mode with MTP
        [
            "4card_disagg_mtp",
            "ATLAS_800_A3_752T_128G_DIE",
            "4",
            "Qwen/Qwen3-32B",
            "16000",
            "1000",
            "",
            str(int(DEFAULT_TPOT_LIMIT_MS)),
            "",
            "W8A8_DYNAMIC",
            "DISABLED",
            "",
            "3",
            "0.9;0.6;0.4",
            "true",
            "disagg",
            "16000",
            "",
            "0",
            "8",
            "critical",
            "32",
            "0",
            "false",
        ],
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_CONFIG_HEADER)
        for example in examples:
            writer.writerow(example)
    print(f"Template CSV written to {csv_path} ({len(examples)} example rows)")


def validate_csv(csv_path: str) -> None:
    """Load and validate CSV cases, printing a summary without executing."""
    cases = load_cases_from_csv(csv_path)
    if not cases:
        print("No valid cases found in CSV.")
        return
    print(f"Found {len(cases)} case(s):")
    for i, c in enumerate(cases, 1):
        print(f"\n  [{i}] {c.case_name}")
        print(f"      device={c.device}, num_devices={c.num_devices}, model_id={c.model_id}")
        print(f"      input_length={c.input_length}, output_length={c.output_length}")
        print(f"      ttft_limits={c.ttft_limits}, tpot_limits={c.tpot_limits}")
        print(f"      mode={c.mode}, compile={c.do_compile}")
        print(f"      tp_sizes={c.tp_sizes}, ep_sizes={c.ep_sizes}")
        print(f"      quantize_linear={c.quantize_linear_action}, quantize_attention={c.quantize_attention_action}")
        print(f"      num_mtp_tokens={c.num_mtp_tokens}, mtp_acceptance_rate={c.mtp_acceptance_rate}")
    print(f"\nAll {len(cases)} case(s) validated successfully.")


def _parse_parallel(s: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse tp_size, pp_size, dp_size from parallel string.

    Supports two formats:
      - 'tp1pp1dp1' (compact, from agg mode)
      - 'TP=4 | PP=1 | DP=1' (verbose, from disagg mode)
    """
    if not s or not isinstance(s, str):
        return None, None, None
    s = s.strip()
    m = re.match(r"tp(\d+)pp(\d+)dp(\d+)", s.lower())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Verbose format: 'TP=4 | PP=1 | DP=1'
    tp = pp = dp = None
    for part in s.split("|"):
        part = part.strip().lower()
        kv = re.match(r"(tp|pp|dp)\s*=\s*(\d+)", part)
        if kv:
            val = int(kv.group(2))
            if kv.group(1) == "tp":
                tp = val
            elif kv.group(1) == "pp":
                pp = val
            elif kv.group(1) == "dp":
                dp = val
    if tp is not None or pp is not None or dp is not None:
        return tp, pp, dp
    return None, None, None


def _parse_breakdown(
    s: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse four percentage strings from 'Mem ... | Comm ... | Cube ... | Vec ...'."""
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


def _build_optimizer_args(case: BenchmarkCase) -> Namespace:
    """Build throughput_optimizer args (Namespace) from BenchmarkCase."""
    ttft = _single_limit(case.ttft_limits, "ttft_limits")
    tpot = _single_limit(case.tpot_limits, "tpot_limits")
    disagg = case.mode == "disagg"

    q_linear = case.quantize_linear_action or QuantizeLinearAction.W8A8_DYNAMIC
    q_attn = case.quantize_attention_action or QuantizeAttentionAction.DISABLED
    mtp_rate = case.mtp_acceptance_rate or [0.9, 0.6, 0.4, 0.2]

    return Namespace(
        input_length=case.input_length,
        output_length=case.output_length,
        device=case.device,
        model_id=case.model_id,
        num_devices=case.num_devices,
        compile=case.do_compile,
        compile_allow_graph_break=case.compile_allow_graph_break,
        num_mtp_tokens=case.num_mtp_tokens,
        mtp_acceptance_rate=mtp_rate,
        quantize_linear_action=q_linear,
        mxfp4_group_size=case.mxfp4_group_size,
        quantize_attention_action=q_attn,
        reserved_memory_gb=case.reserved_memory_gb,
        tp_sizes=case.tp_sizes,
        ttft_limits=ttft,
        tpot_limits=tpot,
        max_prefill_tokens=case.max_prefill_tokens,
        batch_range=case.batch_range,
        serving_cost=case.serving_cost,
        disagg=disagg,
        jobs=case.jobs,
        log_level=case.log_level,
        dump_original_results=False,
        # Attributes required by ParallelRunner / OptimizerData
        image_batch_size=None,
        image_height=None,
        image_width=None,
        prefill_devices_per_instance=None,
        decode_devices_per_instance=None,
        prefix_cache_hit_rate=0.0,
        enable_optimize_prefill_decode_ratio=False,
        ep_sizes=case.ep_sizes,
        moe_dp_sizes=None,
    )


def _filter_best_row(summary):
    """Pick the best row from a summary using only public OptimizerSummary API.

    Uses summary.get_summary_df() (public) and summary.data_config (public attribute).
    Replicates the filter+sort+group-by-parallel logic. Filtering by data_config limits
    handles disagg phase separation automatically: prefill summaries have tpot_limits=None
    and decode summaries have ttft_limits=None, so each summary only filters by its own
    SLO metric (the other becomes float('inf')).

    Returns the top row (pandas.Series) or None if no row passes the filter.
    """
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


def _summary_results_to_benchmark_result(case: BenchmarkCase, summary_result: List[Any]) -> BenchmarkResult:
    """Convert summary_result to BenchmarkResult; keep alignment with output header."""
    out = BenchmarkResult(
        case_name=case.case_name,
        device=case.device,
        num_devices=case.num_devices,
        model_id=case.model_id,
        input_length=case.input_length,
        output_length=case.output_length,
    )
    tpot_limit = _single_limit(case.tpot_limits, "tpot_limits")
    ttft_limit = _single_limit(case.ttft_limits, "ttft_limits")

    def _quant_str(v) -> str:
        if v is None:
            return ""
        return getattr(v, "value", str(v))

    def set_decode_from_row(row) -> None:
        out.best_decode_linear_quant_type = _quant_str(row.get("quantize_linear_action"))
        out.best_decode_attn_quant_type = _quant_str(row.get("quantize_attention_action"))
        tp, pp, dp = _parse_parallel(str(row.get("parallel", "")))
        out.best_decode_tp_size = tp
        out.best_decode_pp_size = pp
        out.best_decode_dp_size = dp
        out.best_decode_slo_target_ms = tpot_limit
        out.best_decode_concurrency = _safe_int(row.get("concurrency"))
        out.best_decode_tpot_ms = _safe_float(row.get("tpot"))
        out.best_decode_total_tps = _safe_float(row.get("token/s"))
        out.best_decode_tps_per_device = _safe_float(row.get("token/s/device"))
        out.best_decode_use_ep = (
            str(int(row.get("ep_size", 1))) if row.get("ep_size") is not None and int(row.get("ep_size", 1)) > 1 else ""
        )
        out.best_decode_mtp_tokens = case.num_mtp_tokens
        pbd = row.get("percentage_breakdowns(d)") or row.get("percentage_breakdowns")
        if pbd is not None:
            mem, comm, cube, vec = _parse_breakdown(str(pbd))
            out.best_decode_mem_pct, out.best_decode_comm_pct = mem, comm
            out.best_decode_cube_pct, out.best_decode_vec_pct = cube, vec

    def set_prefill_from_row(row) -> None:
        out.best_prefill_linear_quant_type = _quant_str(row.get("quantize_linear_action"))
        out.best_prefill_attn_quant_type = _quant_str(row.get("quantize_attention_action"))
        tp, pp, dp = _parse_parallel(str(row.get("parallel", "")))
        out.best_prefill_tp_size = tp
        out.best_prefill_pp_size = pp
        out.best_prefill_dp_size = dp
        out.best_prefill_slo_target_ms = ttft_limit
        out.best_prefill_concurrency = _safe_int(row.get("concurrency"))
        out.best_prefill_ttft_ms = _safe_float(row.get("ttft"))
        out.best_prefill_total_tps = _safe_float(row.get("token/s"))
        out.best_prefill_tps_per_device = _safe_float(row.get("token/s/device"))
        out.best_prefill_use_ep = (
            str(int(row.get("ep_size", 1))) if row.get("ep_size") is not None and int(row.get("ep_size", 1)) > 1 else ""
        )
        out.best_prefill_mtp_tokens = case.num_mtp_tokens
        pbd = row.get("percentage_breakdowns(p)") or row.get("percentage_breakdowns")
        if pbd is not None:
            mem, comm, cube, vec = _parse_breakdown(str(pbd))
            out.best_prefill_mem_pct, out.best_prefill_comm_pct = mem, comm
            out.best_prefill_cube_pct, out.best_prefill_vec_pct = cube, vec

    for summary in summary_result:
        data_config = summary.data_config if hasattr(summary, "data_config") else None
        if data_config is None:
            continue
        best_row = _filter_best_row(summary)
        if best_row is None:
            continue

        # Determine phase: prefill has ttft_limits set + tpot_limits None; decode has tpot_limits set + ttft_limits None
        is_prefill = data_config.ttft_limits is not None and data_config.tpot_limits is None
        is_decode = data_config.tpot_limits is not None and data_config.ttft_limits is None

        if is_decode:
            set_decode_from_row(best_row)
        elif is_prefill:
            set_prefill_from_row(best_row)
        else:
            # Agg mode: same summary has both metrics; apply decode if TPOT present, prefill if TTFT present
            row_tpot = _safe_float(best_row.get("tpot"))
            row_ttft = _safe_float(best_row.get("ttft"))
            if row_tpot is not None:
                set_decode_from_row(best_row)
            if row_ttft is not None:
                set_prefill_from_row(best_row)

    return out


def run_benchmark_case(case: BenchmarkCase) -> BenchmarkResult:
    """Run one benchmark case: call throughput_optimizer in-process and convert to unified result format."""
    print(f"\n{'=' * 80}")
    print(f"Running case: {case.case_name}")
    print(f"{'=' * 80}")
    print(f"Device: {case.device}, Num Devices: {case.num_devices}")
    print(f"Model: {case.model_id}")
    print(f"Input Length: {case.input_length}, Output Length: {case.output_length}")
    print(f"TTFT Limits: {case.ttft_limits}, TPOT Limits: {case.tpot_limits}")
    print(f"Mode: {case.mode}")
    print("=" * 80)

    args = _build_optimizer_args(case)
    # Note: logging is configured once per batch by run_cases_and_save.
    # Direct callers of run_benchmark_case are responsible for their own logging setup.
    from serving_cast.parallel_runner import ParallelRunner

    runner = ParallelRunner(args)
    if args.disagg:
        summary_result = runner.run_disagg()
    else:
        summary_result = runner.run_agg()

    result = _summary_results_to_benchmark_result(case, summary_result)

    # Print best configuration summary
    has_prefill = result.best_prefill_total_tps is not None
    has_decode = result.best_decode_total_tps is not None
    if has_prefill or has_decode:
        print("  " + "-" * 76)
        if case.mode == "agg":
            tps = result.best_decode_total_tps or result.best_prefill_total_tps
            if tps is not None:
                print("  Overall Best Configuration: ")
                print(f"    Best Throughput: {tps:.2f} tokens/s")
                if result.best_prefill_ttft_ms is not None:
                    print(f"    TTFT: {result.best_prefill_ttft_ms:.2f} ms")
                if result.best_decode_tpot_ms is not None:
                    print(f"    TPOT: {result.best_decode_tpot_ms:.2f} ms")
        else:
            if has_prefill:
                print("  Overall Best Configuration (Prefill): ")
                print(f"    Best Throughput: {result.best_prefill_total_tps:.2f} tokens/s")
                if result.best_prefill_ttft_ms is not None:
                    print(f"    TTFT: {result.best_prefill_ttft_ms:.2f} ms")
            if has_decode:
                if has_prefill:
                    print("  " + "-" * 76)
                print("  Overall Best Configuration (Decode): ")
                print(f"    Best Throughput: {result.best_decode_total_tps:.2f} tokens/s")
                if result.best_decode_tpot_ms is not None:
                    print(f"    TPOT: {result.best_decode_tpot_ms:.2f} ms")
        print("  " + "-" * 76)

    if result.best_decode_tps_per_device is not None:
        print(
            f"Best decode: TP={result.best_decode_tp_size}, "
            f"TPS/Device={result.best_decode_tps_per_device:.2f}, "
            f"TPOT={result.best_decode_tpot_ms}ms"
        )
    if result.best_prefill_tps_per_device is not None:
        print(
            f"Best prefill: TP={result.best_prefill_tp_size}, "
            f"TPS/Device={result.best_prefill_tps_per_device:.2f}, "
            f"TTFT={result.best_prefill_ttft_ms}ms"
        )

    return result


def _csv_val(x, fmt=None):
    """Format CSV cell value."""
    if x is None:
        return ""
    if fmt is not None and isinstance(fmt, str):
        try:
            return fmt.format(x)
        except (ValueError, TypeError):
            return str(x)
    return str(x)


def _csv_header_and_ref_row():
    """Return result CSV header and reference row (quantization options)."""
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
        "Decode_Use EP",
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
        "Prefill_Use EP",
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


def _result_row(r: BenchmarkResult) -> List[Any]:
    """Convert one BenchmarkResult to CSV row."""

    def _fmt2(v):
        return "{:.2f}" if v is not None else None

    def _fmt1(v):
        return "{:.1f}" if v is not None else None

    decode_specs = [
        (r.best_decode_linear_quant_type, None),
        (r.best_decode_attn_quant_type, None),
        (r.best_decode_use_ep, None),
        (r.best_decode_mtp_tokens, None),
        (r.best_decode_slo_target_ms, _fmt2(r.best_decode_slo_target_ms)),
        (r.best_decode_concurrency, None),
        (r.best_decode_tpot_ms, _fmt2(r.best_decode_tpot_ms)),
        (r.best_decode_total_tps, _fmt1(r.best_decode_total_tps)),
        (r.best_decode_tps_per_device, _fmt1(r.best_decode_tps_per_device)),
        (r.best_decode_mem_pct, None),
        (r.best_decode_comm_pct, None),
        (r.best_decode_cube_pct, None),
        (r.best_decode_vec_pct, None),
        (r.best_decode_tp_size, None),
        (r.best_decode_pp_size, None),
        (r.best_decode_dp_size, None),
    ]
    prefill_specs = [
        (r.best_prefill_linear_quant_type, None),
        (r.best_prefill_attn_quant_type, None),
        (r.best_prefill_use_ep, None),
        (r.best_prefill_mtp_tokens, None),
        (r.best_prefill_slo_target_ms, _fmt2(r.best_prefill_slo_target_ms)),
        (r.best_prefill_concurrency, None),
        (r.best_prefill_ttft_ms, _fmt2(r.best_prefill_ttft_ms)),
        (r.best_prefill_total_tps, _fmt1(r.best_prefill_total_tps)),
        (r.best_prefill_tps_per_device, _fmt1(r.best_prefill_tps_per_device)),
        (r.best_prefill_mem_pct, None),
        (r.best_prefill_comm_pct, None),
        (r.best_prefill_cube_pct, None),
        (r.best_prefill_vec_pct, None),
        (r.best_prefill_tp_size, None),
        (r.best_prefill_pp_size, None),
        (r.best_prefill_dp_size, None),
    ]
    return [
        r.case_name,
        r.device,
        r.num_devices,
        r.input_length,
        r.output_length,
        r.model_id,
        *[_csv_val(v, fmt) for v, fmt in decode_specs],
        *[_csv_val(v, fmt) for v, fmt in prefill_specs],
        "",
        "",
    ]


def save_results_to_csv(results: List[BenchmarkResult], output_file: str):
    """Save all results to CSV (header matches original output)."""
    header, ref_row = _csv_header_and_ref_row()
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(ref_row)
        for r in results:
            writer.writerow(_result_row(r))
    print(f"\nAll results saved to {output_file}")


def run_cases_and_save(
    cases: List[BenchmarkCase],
    output_file: str = "benchmark_cases_results.csv",
) -> None:
    """Run case list and save to CSV; sequential, write one row per case, batch flush."""
    if not cases:
        print("No cases to run.", file=sys.stderr)
        return

    # Configure logging once for the entire batch (avoids per-case global mutation).
    # If cases specify different log_levels, use the first case's level and warn.
    log_levels_used = {c.log_level for c in cases}
    if len(log_levels_used) > 1:
        print(
            f"Warning: cases use multiple log_levels {log_levels_used}; "
            f"using '{cases[0].log_level}' for the whole batch.",
            file=sys.stderr,
        )
    _configure_logging(cases[0].log_level)

    print("=" * 80)
    print("Benchmark Cases Runner")
    print("=" * 80)
    print(f"Total cases: {len(cases)}")
    print("Mode: sequential (one case at a time, result written after each case)")
    print("=" * 80)

    header, ref_row = _csv_header_and_ref_row()
    all_results: List[BenchmarkResult] = []

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(ref_row)
        f.flush()

        for idx, case in enumerate(cases, 1):
            print(f"\n[{idx}/{len(cases)}] Processing case: {case.case_name}")
            try:
                result = run_benchmark_case(case)
            except Exception as e:
                print(f"Case {case.case_name} failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                result = BenchmarkResult(
                    case_name=case.case_name,
                    device=case.device,
                    num_devices=case.num_devices,
                    model_id=case.model_id,
                    input_length=case.input_length,
                    output_length=case.output_length,
                )
            all_results.append(result)
            writer.writerow(_result_row(result))
            if idx % FLUSH_BATCH_SIZE == 0:
                f.flush()
        f.flush()

    # Print per-case results
    for result in all_results:
        print(f"\nCase {result.case_name} Results:")
        if result.best_decode_tps_per_device is not None:
            print(
                f"  Decode - TPOT: {result.best_decode_tpot_ms:.3f}ms, "
                f"TPS/Device: {result.best_decode_tps_per_device:.2f}, "
                f"TP={result.best_decode_tp_size}, PP={result.best_decode_pp_size}, DP={result.best_decode_dp_size}, "
                f"Concurrency: {result.best_decode_concurrency}"
            )
        if result.best_prefill_tps_per_device is not None:
            print(
                f"  Prefill - TTFT: {result.best_prefill_ttft_ms:.3f}ms, "
                f"TPS/Device: {result.best_prefill_tps_per_device:.2f}, "
                f"TP={result.best_prefill_tp_size}, PP={result.best_prefill_pp_size}, DP={result.best_prefill_dp_size}, "
                f"Concurrency: {result.best_prefill_concurrency}"
            )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for result in all_results:
        print(f"\n{result.case_name}:")
        if result.best_decode_tps_per_device is not None:
            print(f"  Best Decode TPS/Device: {result.best_decode_tps_per_device:.2f}")
            print(f"  Best Decode TPOT: {result.best_decode_tpot_ms:.3f}ms")
            print(
                f"  Best Decode Config: TP={result.best_decode_tp_size}, PP={result.best_decode_pp_size}, DP={result.best_decode_dp_size}"
            )
        if result.best_prefill_tps_per_device is not None:
            print(f"  Best Prefill TPS/Device: {result.best_prefill_tps_per_device:.2f}")
            print(f"  Best Prefill TTFT: {result.best_prefill_ttft_ms:.3f}ms")
            print(
                f"  Best Prefill Config: TP={result.best_prefill_tp_size}, PP={result.best_prefill_pp_size}, DP={result.best_prefill_dp_size}"
            )
    print(f"\nAll results saved to: {output_file}")
    print("=" * 80)


def _test_result_conversion() -> bool:
    """Verify consistency with throughput_optimizer output using mock summary; requires pandas."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("--test-conversion requires pandas, which is not installed.")
    common_cols = [
        "device_name",
        "num_devices",
        "model_id",
        "quantize_linear_action",
        "quantize_attention_action",
        "input_length",
        "output_length",
        "concurrency",
        "ttft",
        "tpot",
        "token/s",
        "token/s/device",
        "parallel",
        "batch_size",
    ]
    row = {
        "device_name": "ATLAS_800_A3_752T_128G_DIE",
        "num_devices": 1,
        "model_id": "Qwen/Qwen3-32B",
        "quantize_linear_action": QuantizeLinearAction.W8A8_DYNAMIC,
        "quantize_attention_action": QuantizeAttentionAction.DISABLED,
        "input_length": 6000,
        "output_length": 1000,
        "concurrency": 4,
        "ttft": None,
        "tpot": 40,
        "token/s": 83.2,
        "token/s/device": 83.2,
        "parallel": "tp1pp1dp1",
        "batch_size": 4,
    }
    df = pd.DataFrame([row])
    for c in common_cols:
        if c not in df.columns:
            df[c] = None
    df = df[common_cols]

    class MockDataConfig:
        ttft_limits = None
        tpot_limits = DEFAULT_TPOT_LIMIT_MS

    class MockSummary:
        data_config = MockDataConfig()

        def get_summary_df(self):
            return df

    case = BenchmarkCase(
        case_name="test",
        device=row["device_name"],
        num_devices=1,
        model_id=row["model_id"],
        input_length=row["input_length"],
        output_length=row["output_length"],
        ttft_limits=[],
        tpot_limits=[DEFAULT_TPOT_LIMIT_MS],
        mode="disagg",
        num_mtp_tokens=0,
    )
    result = _summary_results_to_benchmark_result(case, [MockSummary()])
    assert result.best_decode_tps_per_device == row["token/s/device"]
    assert result.best_decode_total_tps == row["token/s"]
    assert result.best_decode_tpot_ms == row["tpot"]
    assert result.best_decode_tp_size == 1 and result.best_decode_pp_size == 1 and result.best_decode_dp_size == 1
    assert result.best_decode_concurrency == row["concurrency"]
    print("--test-conversion passed: key fields match throughput_optimizer output.")
    return True


def _parse_args():
    """Parse CLI arguments for run_throughput_optimizer_cases."""
    parser = argparse.ArgumentParser(
        prog="run_throughput_optimizer_cases",
        description="Batch runner for optix.throughput_optimizer. "
        "Runs the optimizer once per case from a CSV input, "
        "then aggregates all results into a single CSV.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to input CSV file with benchmark cases (one case per row).",
    )
    parser.add_argument(
        "--write-template",
        type=str,
        default=None,
        metavar="PATH",
        help="Write a template CSV with example row to PATH and exit.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file for results. Defaults to 'benchmark_cases_results.csv' when --input-csv is used.",
    )
    parser.add_argument(
        "--test-conversion",
        action="store_true",
        default=False,
        help="Run internal conversion test and exit.",
    )
    parser.add_argument(
        "--validate-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Validate the input CSV file at PATH and print a summary without executing.",
    )
    args = parser.parse_args()
    return (
        args.input_csv,
        args.write_template,
        args.output_csv,
        args.test_conversion,
        args.validate_csv,
    )


if __name__ == "__main__":
    input_csv, write_template, output_csv, test_conversion, validate_csv_path = _parse_args()

    if test_conversion:
        ok = _test_result_conversion()
        sys.exit(0 if ok else 1)

    if write_template:
        write_template_csv(write_template)
        sys.exit(0)

    if validate_csv_path:
        validate_csv(validate_csv_path)
        sys.exit(0)

    if input_csv:
        cases = load_cases_from_csv(input_csv)
        out_file = output_csv or "benchmark_cases_results.csv"
        run_cases_and_save(cases, output_file=out_file)
    else:
        print(
            "No --input-csv provided. Use --write-template to generate a template CSV.",
            file=sys.stderr,
        )
        sys.exit(1)
