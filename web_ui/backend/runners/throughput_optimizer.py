"""Throughput-optimizer runner adapter.

Wraps ``serving_cast.parallel_runner.ParallelRunner``. ALL heavy imports
(``torch``, ``pandas``, ``tensor_cast``, ``serving_cast``) are INSIDE ``run`` so
the FastAPI app boots without the simulation stack.

Mirrors ``serving_cast/service/optimizer_curve_plots.py:run_multi_device_loop``:
the form's ``device`` field is multi-select, so the adapter loops
``ParallelRunner`` per selected device profile, then builds:

* one ``result_records`` row per ``(device, OptimizerSummary)`` (NOT one per
  explored config — ParallelRunner collapses configs into summary rows);
* capture-time ``rank`` (max ``throughput_token_s``; ties -> lower ttft then
  tpot then device) via ``domain.services.ranking`` (best_config is a
  pure lookup of the rank=1 record);
* a mode-aware assembled ``result`` envelope (aggregation / disagg_prefill +
  disagg_decode / pd_ratio) for the result record.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from typing import Any, Callable

from models.entities import ResultRecord
from runners._multicase import as_list, expand_cases, parse_float_list, resolve_model_id_path

logger = logging.getLogger(__name__)


# Multi-case expansion fields (Phase D2): tpot/ttft are free-float comma-lists,
# the two quant actions are multi-select. (device is already multi-select /
# multi-device — the existing sweep dimension.)
_THROUGHPUT_MULTI_FIELDS = {
    "device": as_list,
    "tpot_limits": parse_float_list,
    "ttft_limits": parse_float_list,
    "quantize_linear_action": as_list,
    "quantize_attention_action": as_list,
}


def _enum_val(v: Any) -> Any:
    """StrEnum -> its string value (for tagging records by case)."""
    return getattr(v, "value", v)


def _run_throughput_sweep(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the device sweep for one (already concrete) throughput case; return
    its record dicts (one per device/config), each tagged with the case fields
    so ``_assemble_throughput_result`` can group multi-case records.
    """
    from serving_cast.parallel_runner import ParallelRunner
    from serving_cast.service.utils import count_search_combinations, resolve_parallel_search_candidates
    from tensor_cast import device_profiles  # noqa: F401  registers builtins
    from tensor_cast.core.quantization.datatypes import (
        QuantizeAttentionAction,
        QuantizeLinearAction,
    )

    args = _build_namespace(params)
    args.quantize_attention_action = QuantizeAttentionAction(args.quantize_attention_action or "DISABLED")
    args.quantize_linear_action = QuantizeLinearAction(args.quantize_linear_action or "W8A8_DYNAMIC")
    args.quantize_non_expert_linear_action = QuantizeLinearAction(args.quantize_non_expert_linear_action or "DISABLED")
    pd_ratio_mode = bool(args.enable_optimize_prefill_decode_ratio)
    disagg_mode = bool(args.disagg) and not pd_ratio_mode

    # Mirror the CLI's pre-flight combination-count warning (cli/inference/
    # throughput_optimizer.py). The CLI uses ``print(..., file=sys.stderr)``
    # which is always visible; ParallelRunner uses ``logger.warning`` which
    # the worker's log-level filter (default ``error``) would silently drop.
    # Emit here so the warning is captured into the job log regardless of
    # log_level, then flag ``search_combination_warning_emitted`` to prevent
    # ParallelRunner from repeating it.
    tp_cands, ep_cands, moe_dp_cands, mtp_cands = resolve_parallel_search_candidates(
        args.tp_sizes,
        args.ep_sizes,
        args.moe_dp_sizes,
        getattr(args, "num_mtp_token_sizes", None),
        args.num_mtp_tokens,
        args.num_devices,
    )
    total_combinations = count_search_combinations(tp_cands, ep_cands, moe_dp_cands, mtp_cands)
    args.search_combination_warning_emitted = False
    if args.max_search_combinations and total_combinations > args.max_search_combinations:
        args.search_combination_warning_emitted = True
        print(
            "[WARNING] Large number of parallel search combinations "
            f"({total_combinations} = TP:{len(tp_cands)} x EP:{len(ep_cands)} "
            f"x MOE-DP:{len(moe_dp_cands)} x MTP:{len(mtp_cands)}). "
            "Optimization may take a long time. Consider narrowing --tp-sizes, --ep-sizes, "
            "--moe-dp-sizes, or --num-mtp-tokens; or increase --max-search-combinations.",
            file=sys.stderr,
            flush=True,
        )

    # device is now expanded externally (multi-case). Each case has a single
    # device string; ParallelRunner runs once for that device (no internal loop).
    device_name = params.get("device")
    if isinstance(device_name, list):
        device_name = device_name[0] if device_name else None

    case_tag = {
        "device": device_name,
        "tpot_limits": args.tpot_limits,
        "ttft_limits": args.ttft_limits,
        "quantize_linear_action": _enum_val(args.quantize_linear_action),
        "quantize_attention_action": _enum_val(args.quantize_attention_action),
    }

    records: list[dict[str, Any]] = []
    seq = 0
    run_args = copy.copy(args)
    run_args.device = device_name
    runner = ParallelRunner(run_args)
    summaries = runner.run_disagg() if (pd_ratio_mode or disagg_mode) else runner.run_agg()
    mode = _infer_mode(run_args, pd_ratio_mode, disagg_mode)
    for summary in summaries:
        try:
            summary.report_final_result(run_args, silent=False)
        except Exception:
            # Log at ERROR level (not DEBUG) so the failure is visible in the job
            # log regardless of the worker's log_level setting. Also print the
            # traceback to stdout so it's captured into the job log file.
            import traceback

            logger.error("Failed to report final result", exc_info=True)
            traceback.print_exc()
        for row in _summary_to_rows(summary, device_name, mode, args):
            row["config"].update(case_tag)
            row["seq"] = seq
            seq += 1
            records.append(row)

    # rank is NOT computed here: the main-process job_runner computes the single
    # authoritative global rank across all cases (persisted as authoritative), and
    # result_view recomputes a per-case rank on demand when rendering multi_case.
    # A per-case rank computed here would just be overwritten by job_runner (wasted
    # work).
    return records


def execute(
    params: dict[str, Any],
    *,
    cached_hashes: set[str] | None = None,
    form_schema_version: str | None = None,
    job_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Worker-side entry (runs in the ``runners._worker`` subprocess).

    Multi-case (Phase D2): tpot/ttft/quant fields may carry multiple values; each
    case runs the full device sweep. Case-level dedup: a case whose hash is in
    ``cached_hashes`` is skipped (in the returned ``skipped`` list) for the main
    process to reuse. Returns ``(records, skipped_hashes)``.
    """
    import time
    import traceback

    from cli.logo import print_logo
    from runners._multicase import compute_case_hash

    print_logo()
    print(f"[case-dedup] throughput_optimizer cached_hashes={len(cached_hashes or [])}", flush=True)

    # Validate PD ratio parameters early so the error reaches job_runner
    # (not swallowed by _multicase's per-case exception handler).
    if params.get("enable_optimize_prefill_decode_ratio"):
        if params.get("prefill_devices_per_instance") is None or params.get("decode_devices_per_instance") is None:
            raise ValueError(
                "Both --prefill-devices-per-instance and --decode-devices-per-instance "
                "are required when PD ratio optimization is enabled."
            )

    t0 = time.time()
    # Resolve a relative model_id against the repo root (cwd is web/backend).
    params = {**params, "model_id": resolve_model_id_path(params.get("model_id"))}
    cases = expand_cases(params, _THROUGHPUT_MULTI_FIELDS)

    # Trace path synthesis when chrome_trace is enabled
    from services.trace_store import legacy_hash_path

    def _synth_trace_path(case_params: dict[str, Any], case_hash: str | None) -> None:
        """If chrome_trace is True, replace it with the computed path."""
        if case_params.get("chrome_trace") is True and job_id and case_hash:
            case_params["chrome_trace"] = str(legacy_hash_path(job_id, case_hash))

    def _ch(cp: dict[str, Any]) -> str | None:
        return compute_case_hash("throughput_optimizer", form_schema_version, cp)

    cached = cached_hashes or set()
    all_records: list[dict[str, Any]] = []
    skipped: list[str] = []
    n = len(cases)
    from services.capture import capture_case_log
    from runners._cli_command import build_cli_command_string

    # Case divider: a prominent separator line (80 `=` chars) between cases,
    # replacing the `[case i/n]` stamp (which had display issues in some log
    # viewers). Each case's CLI command is still logged, just without the stamp.
    _divider = "\n" + "=" * 80

    if n > 1:
        # Multi-case expansion summary: the parent process logged the reference
        # command for the ORIGINAL params (before expansion). List the actual
        # per-case commands here so the job log reflects what will really be
        # executed. No `[case i/n]` stamp — just the commands separated by blank
        # lines (the divider goes between cases in the per-case loop below).
        print(f"{_divider}", flush=True)
        print(f"[throughput_optimizer] Expanding into {n} case(s) based on multi-field values:", flush=True)
        for case_params in cases:
            print(f"  {build_cli_command_string('throughput_optimizer', case_params)}", flush=True)
        print(flush=True)

    for i, case_params in enumerate(cases, 1):
        ch = _ch(case_params)
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            continue
        # Synthesize trace path if chrome_trace is enabled
        _synth_trace_path(case_params, ch)
        try:
            with capture_case_log() as buf:
                # Divider + CLI command + case key values. No case index label
                # — the divider separates cases visually; the CLI + keys
                # identify each case unambiguously.
                print(f"{_divider}", flush=True)
                print(f"CLI: {build_cli_command_string('throughput_optimizer', case_params)}", flush=True)
                print(
                    " ".join(f"{k}={case_params.get(k)}" for k in _THROUGHPUT_MULTI_FIELDS),
                    flush=True,
                )
                print(flush=True)
                recs = _run_throughput_sweep(case_params)
            case_log_text = buf.getvalue()
            for r in recs:
                r["case_hash"] = ch
                r["case_log"] = case_log_text
            all_records.extend(recs)
        except Exception as e:
            traceback.print_exc()
            all_records.append(
                {
                    "config": {k: case_params.get(k) for k in _THROUGHPUT_MULTI_FIELDS},
                    "summary": {"error": str(e)},
                    "tables": {},
                    "case_hash": ch,
                }
            )
    print(f"All experiments completed in {time.time() - t0:.2f} seconds.")
    return all_records, skipped


class ThroughputOptimizerAdapter:
    """Adapter implementing ``application/ports/RunnerPort``.

    Thin spawner (Phase B): delegates to ``runners._subprocess.run_module_subprocess``.
    """

    def run(
        self,
        params: dict[str, Any],
        *,
        job_id: str | None = None,
        on_progress: Callable[[int | None, str | None], None] | None = None,
        cancel_flag: Callable[[], bool] | None = None,
        cached_hashes: set[str] | None = None,
        form_schema_version: str | None = None,
    ) -> tuple[list[ResultRecord], list[str]]:
        from runners._subprocess import run_module_subprocess

        return run_module_subprocess(
            "throughput_optimizer",
            params,
            job_id=job_id,
            on_progress=on_progress,
            cancel_flag=cancel_flag,
            cached_hashes=cached_hashes,
            form_schema_version=form_schema_version,
        )


# ---------------------------------------------------------------------------
# Param -> argparse.Namespace mapping.
# ---------------------------------------------------------------------------

_BOOL_KEYS = {
    "compile",
    "compile_allow_graph_break",
    "disagg",
    "enable_optimize_prefill_decode_ratio",
    "dump_original_results",
}


def _split_ints(value: Any) -> list[int] | None:
    """Normalize an int-list field. The form sends these as free-text
    comma/space-separated strings (e.g. "1,2,4"); parse to a list. Pass ``None``
    through (means "no search"), coerce a real list if already one.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    return [int(p) for p in parts]


def _split_floats(value: Any) -> list[float]:
    """Normalize a float-list field (mtp_acceptance_rate). None -> []; parse a
    comma/space-separated string, or coerce a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    return [float(p) for p in parts]


def _build_namespace(params: dict[str, Any]) -> argparse.Namespace:
    """Build the argparse.Namespace ParallelRunner expects.

    Defaults mirror the CLI so an omitted field behaves like the CLI default.
    """

    def _get(key: str, default: Any = None) -> Any:
        v = params.get(key, default)
        return v

    def _num_or_none(key: str, default: Any = None) -> float | None:
        """Coerce an optional numeric field to ``float | None``.

        The form sends empty limits (e.g. ``ttft_limits``) as ``""``; the CLI
        default is ``None``. ``base_throughput_optimizer`` compares these limits
        with ``>`` during the batch-size binary search, so a stray ``""`` raises
        ``TypeError: '>' not supported between instances of 'float' and 'str'``.
        ``expand_cases`` already collapses "" -> None upstream, but this keeps
        ``_build_namespace`` self-contained for direct callers.
        """
        v = _get(key, default)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _int(key: str, default: int) -> int:
        """Integer field that always has a value.

        Number controls send ``null`` (and text-derived ones may send ``""``)
        when cleared; ``_get(key, default)`` only applies the default when the
        key is ABSENT, so a present-but-null value slips through. Coerce
        null/"" -> default here so e.g. a cleared ``num_devices`` can't reach
        ParallelRunner as None (-> ``None.bit_length()`` in resolve_search_sizes).
        """
        v = _get(key, default)
        if v is None or v == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _num(key: str, default: float) -> float:
        """Float counterpart of ``_int`` (null/"" -> default)."""
        v = _get(key, default)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    ns = argparse.Namespace()
    ns.model_id = _get("model_id")
    ns.device = None  # set per-loop
    ns.num_devices = _int("num_devices", 1)
    ns.reserved_memory_gb = _num("reserved_memory_gb", 10.0)
    ns.remote_source = _get("remote_source", "huggingface")
    ns.input_length = _int("input_length", 3500)
    ns.output_length = _int("output_length", 1500)
    ns.image_batch_size = _get("image_batch_size")
    ns.image_height = _get("image_height")
    ns.image_width = _get("image_width")
    ns.ttft_limits = _num_or_none("ttft_limits")
    ns.tpot_limits = _num_or_none("tpot_limits")
    ns.max_batched_tokens = _int("max_batched_tokens", 8192)
    ns.serving_cost = _num("serving_cost", 0)
    ns.num_mtp_tokens = _int("num_mtp_tokens", 0)
    # List fields arrive as free-text strings from the form — normalize to lists.
    ns.mtp_acceptance_rate = _split_floats(_get("mtp_acceptance_rate", []))
    ns.prefix_cache_hit_rate = _num("prefix_cache_hit_rate", 0.0)
    ns.concurrency_search_strategy = _get("concurrency_search_strategy", "exponential")
    ns.tp_sizes = _split_ints(_get("tp_sizes"))
    ns.ep_sizes = _split_ints(_get("ep_sizes"))
    ns.moe_dp_sizes = _split_ints(_get("moe_dp_sizes"))
    # Mirror the CLI's backward-compatible default (cli/inference/throughput_optimizer.py):
    # when none of tp/ep/moe-dp sizes are specified, the CLI searches TP across the
    # default range (powers of 2: e.g. num_devices=4 -> [1,2,4]) instead of pinning it
    # to num_devices. resolve_search_sizes treats ``None`` as "fixed, do not search"
    # (-> only TP=num_devices) and ``[]`` as "search the default range". Without this,
    # ParallelRunner explored a single TP config and the result table collapsed to one
    # row ("TP=num_devices | PP=1 | DP=1").
    if ns.tp_sizes is None and ns.ep_sizes is None and ns.moe_dp_sizes is None:
        ns.tp_sizes = []
    ns.batch_range = _split_ints(_get("batch_range"))
    ns.quantize_linear_action = _get("quantize_linear_action", "W8A8_DYNAMIC")
    ns.quantize_non_expert_linear_action = _get("quantize_non_expert_linear_action") or "DISABLED"
    ns.quantize_attention_action = _get("quantize_attention_action", "DISABLED")
    ns.mxfp4_group_size = _int("mxfp4_group_size", 32)
    ns.jobs = _int("jobs", 8)
    ns.max_search_combinations = _int("max_search_combinations", 100)
    ns.search_combination_warning_emitted = False
    ns.prefill_devices_per_instance = _get("prefill_devices_per_instance")
    ns.decode_devices_per_instance = _get("decode_devices_per_instance")
    ns.enable_optimize_prefill_decode_ratio = bool(_get("enable_optimize_prefill_decode_ratio", False))
    ns.disagg = bool(_get("disagg", False))
    ns.compile = bool(_get("compile", False))
    ns.compile_allow_graph_break = bool(_get("compile_allow_graph_break", False))
    ns.dump_original_results = bool(_get("dump_original_results", False))
    ns.chrome_trace = _get("chrome_trace")  # String path (synthesized by _synth_trace_path) or None
    # Handle compilation_config: set individual boolean attributes so
    # ParallelRunner._apply_compilation_config can read them, mirroring
    # UserInputConfig.from_args' compilation_config expansion.
    compilation_config = _get("compilation_config", []) or []
    ns.compilation_config = compilation_config
    ns.enable_multistream = "enable_multistream" in compilation_config
    ns.enable_sequence_parallel = "enable_sequence_parallel" in compilation_config
    ns.enable_matmul_allreduce = "enable_matmul_allreduce" in compilation_config
    ns.enable_dispatch_ffn_combine = "enable_dispatch_ffn_combine" in compilation_config
    # ParallelRunner._init_worker reads args.log_level to configure each worker
    # process's logging — omitting it crashes every worker at startup.
    ns.log_level = _get("log_level", "error")
    return ns


def _infer_mode(args, pd_ratio_mode: bool, disagg_mode: bool) -> str:
    """Decide the optimizer mode: ``pd_ratio`` > ``disagg_{prefill,decode}`` > ``aggregation``."""
    if pd_ratio_mode:
        return "pd_ratio"
    if disagg_mode:
        # Prefill phase if ttft set; Decode phase if tpot set.
        if getattr(args, "tpot_limits", None) is not None and getattr(args, "ttft_limits", None) is None:
            return "disagg_decode"
        return "disagg_prefill"
    return "aggregation"


# ---------------------------------------------------------------------------
# OptimizerSummary -> normalized rows (capture-time reshape).
# ---------------------------------------------------------------------------


def _summary_to_rows(summary, device_name: str, mode: str, args) -> list[dict[str, Any]]:
    """Reshape one ``OptimizerSummary`` into 1..N normalized record dicts.

    Uses the summary's public ``get_summary_df()`` (a DataFrame) + its public
    cross-hardware collectors (``collect_comparison_row`` etc.) where possible,
    falling back to the raw df rows.
    """
    df = summary.get_summary_df() if hasattr(summary, "get_summary_df") else None
    if df is None or getattr(df, "empty", True):
        return []

    rows: list[dict[str, Any]] = []
    if mode == "pd_ratio":
        for _, r in df.iterrows():
            rows.append(_pd_ratio_row(r, device_name))
    else:
        for _, r in df.iterrows():
            rows.append(_agg_disagg_row(r, device_name, mode, args))
    return rows


def _fnum(value: Any) -> float | None:
    """Coerce ``value`` to float; missing/NaN/unparseable -> ``None``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return None if math.isnan(f) else f


def _agg_disagg_row(r, device_name: str, mode: str, args) -> dict[str, Any]:
    """Build one normalized record dict from an aggregation/disagg DataFrame row."""
    parallel = str(r.get("parallel", ""))
    concurrency = r.get("concurrency")
    num_devices = r.get("num_devices")
    batch_size = r.get("batch_size")
    throughput = _fnum(r.get("token/s"))
    ttft = _fnum(r.get("ttft"))
    tpot = _fnum(r.get("tpot"))
    # Derive the per-row phase BEFORE computing QPS: run_disagg emits BOTH
    # prefill rows (ttft set) and decode rows (tpot set), but the job-level
    # ``mode`` is a single value. Passing the job mode to ``_disagg_qps`` made
    # decode rows take the prefill branch (or vice versa) and return None.
    if mode.startswith("disagg"):
        actual_mode = "disagg_prefill" if ttft is not None else "disagg_decode"
    else:
        actual_mode = mode
    qps = _disagg_qps(r, getattr(args, "output_length", None), actual_mode)
    summary = {
        "throughput_token_s": throughput,
        "qps": qps,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "mode": actual_mode,
        # Memory columns (used by the frontend to display memory breakdown and
        # filter OOM rows from scatter plots, mirroring the CLI's _memory_filter).
        # Column names in summary_df come from MEMORY_KEY_TO_COLUMN in
        # serving_cast.service.utils.
        "device_memory_available_gb": _fnum(r.get("avail_GB")),
        "model_weight_size_gb": _fnum(r.get("weight_GB")),
        "kv_cache_size_gb": _fnum(r.get("kv_cache_GB")),
        "model_activation_size_gb": _fnum(r.get("activation_GB")),
    }
    config = {
        "device": device_name,
        "parallel": parallel,
        "concurrency": concurrency,
        "num_devices": num_devices,
        "batch_size": batch_size,
        "mode": mode,
    }
    return {
        "config": config,
        "summary": summary,
        "tables": {"sweep": True},
    }


def _disagg_qps(r, output_length, mode: str) -> float | None:
    """Compute QPS for a disagg row: ``concurrency / ttft`` (prefill) or
    ``concurrency / (tpot * output_length)`` (decode); ``None`` if not derivable.
    """
    conc = _fnum(r.get("concurrency"))
    if conc is None or conc <= 0:
        return None
    if mode == "disagg_prefill":
        ttft = _fnum(r.get("ttft"))
        if ttft and ttft > 0:
            return conc / ttft * 1000.0
    if mode == "disagg_decode":
        tpot = _fnum(r.get("tpot"))
        if tpot and tpot > 0 and output_length:
            return conc / (tpot * float(output_length)) * 1000.0
    return None


def _pd_ratio_row(r, device_name: str) -> dict[str, Any]:
    """Build one normalized record dict from a prefill/decode-ratio DataFrame row."""
    summary = {
        "balanced_qps": _fnum(r.get("balanced_qps")),
        "pd_ratio": _fnum(r.get("pd_ratio")),
        "p_qps": _fnum(r.get("p_qps")),
        "d_qps": _fnum(r.get("d_qps")),
        "ttft_ms": _fnum(r.get("ttft_p")),
        "tpot_ms": _fnum(r.get("tpot_d")),
        "mode": "pd_ratio",
    }
    config = {
        "device": device_name,
        "parallel_p": str(r.get("parallel_p", "")),
        "parallel_d": str(r.get("parallel_d", "")),
        "p_devices_per_instance": r.get("num_devices_p"),
        "d_devices_per_instance": r.get("num_devices_d"),
        "p_batch_size": r.get("batch_size_p"),
        "d_batch_size": r.get("batch_size_d"),
        "p_concurrency": r.get("concurrency_p"),
        "d_concurrency": r.get("concurrency_d"),
        "mode": "pd_ratio",
    }
    return {"config": config, "summary": summary, "tables": {"sweep": True}}
