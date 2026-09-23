# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Throughput-optimizer runner adapter.

Wraps ``serving_cast.parallel_runner.ParallelRunner``. Heavy imports are inside
``run`` so FastAPI boots without the simulation stack. The form's ``device``
field is multi-select; the adapter loops ParallelRunner per device profile.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from typing import Any, Callable

from models.entities import ResultRecord
from runners._multicase import as_list, expand_cases, parse_float_list, resolve_model_id_path
from services.case_validation import validate_case_for_module

logger = logging.getLogger(__name__)


def _get_multi_case_fields() -> tuple[tuple[str, Callable], ...]:
    """Collect multi-case fields from UIFieldProps.multi_values=True.

    Returns:
        Tuple of (field_id, parse_fn) pairs.
    """
    from cli.registry.modules import get_spec
    from web_ui.backend.services.ui_props.throughput_optimizer import UI

    spec = get_spec("throughput_optimizer")
    fields = []
    for p in spec.fields:
        ui = UI.get(p.name)
        if ui and ui.multi_values:
            # Infer parse function from data_type
            if p.data_type in ("integer",):
                fields.append((p.name, parse_int_list))
            elif p.data_type in ("number",):
                fields.append((p.name, parse_float_list))
            else:  # string, string[], etc.
                fields.append((p.name, as_list))
    return tuple(fields)


def parse_int_list(value: Any) -> list[int]:
    """Parse a free-text comma/space list into ints.

    Args:
        value: Input value.

    Returns:
        List of ints.

    Raises:
        ValueError: If value cannot be parsed.
    """
    if value is None:
        return []
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            raise ValueError(f"invalid int list: {value!r}") from None
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(",", " ").split() if p.strip()]
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"invalid int list: {value!r}") from None
    raise ValueError(f"invalid int list: {value!r}")


_THROUGHPUT_MULTI_FIELDS_WITH_PARSERS = _get_multi_case_fields()
_THROUGHPUT_MULTI_FIELDS = {fid: parser for fid, parser in _THROUGHPUT_MULTI_FIELDS_WITH_PARSERS}


def _enum_val(v: Any) -> Any:
    """Extract string value from StrEnum member."""
    return getattr(v, "value", v)


def _run_throughput_sweep(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Run device sweep for one throughput case.

    Args:
        params: Concrete case params.

    Returns:
        List of record dicts, one per device/config.
    """
    from serving_cast.parallel_runner import ParallelRunner
    from serving_cast.service.utils import count_search_combinations, resolve_parallel_search_candidates
    from tensor_cast import device_profiles  # noqa: F401  registers builtins
    from tensor_cast.core.quantization.datatypes import (
        QuantizeAttentionAction,
        QuantizeLinearAction,
    )
    from services.enum_utils import coerce_enum

    args = _build_namespace(params)
    # Use coerce_enum for consistent kebab/UPPER_SNAKE handling (RFC §3.8.1)
    args.quantize_attention_action = coerce_enum(QuantizeAttentionAction, args.quantize_attention_action or "disabled")
    args.quantize_linear_action = coerce_enum(QuantizeLinearAction, args.quantize_linear_action or "W8A8_DYNAMIC")
    args.quantize_non_expert_linear_action = coerce_enum(
        QuantizeLinearAction, args.quantize_non_expert_linear_action or "disabled"
    )
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
        "tpot-limits": args.tpot_limits,
        "ttft-limits": args.ttft_limits,
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
    provided: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Worker-side entry point (runs in subprocess).

    Args:
        params: Form params (may contain multi-value fields).
        cached_hashes: Set of already-computed case hashes for dedup.
        form_schema_version: Schema version for hash computation.
        job_id: Job ID for trace path synthesis.
        provided: Set of explicitly provided field names (for wants_provided validators).

    Returns:
        Tuple of (records, skipped_hashes).
    """
    import time
    import traceback

    from cli.logo import print_logo
    from runners._multicase import compute_case_hash

    print_logo()
    print(f"[case-dedup] throughput_optimizer cached_hashes={len(cached_hashes or [])}", flush=True)

    # Validate PD ratio parameters early so the error reaches job_runner
    # (not swallowed by _multicase's per-case exception handler).
    if params.get("enable-optimize-prefill-decode-ratio"):
        if params.get("prefill-devices-per-instance") is None or params.get("decode-devices-per-instance") is None:
            raise ValueError(
                "Both --prefill-devices-per-instance and --decode-devices-per-instance "
                "are required when PD ratio optimization is enabled."
            )

    t0 = time.time()
    # Resolve a relative model_id against the repo root (cwd is web/backend).
    params = {**params, "model-id": resolve_model_id_path(params.get("model-id"))}
    cases = expand_cases(params, _THROUGHPUT_MULTI_FIELDS)

    # Trace path synthesis when chrome_trace is enabled
    from services.trace_store import legacy_hash_path

    def _synth_trace_path(case_params: dict[str, Any], case_hash: str | None) -> None:
        """If chrome_trace is True, replace it with the computed path.
        If False, convert to None (CLI expects string or None, not boolean).
        """
        trace_val = case_params.get("chrome-trace-file")
        if trace_val is True and job_id and case_hash:
            case_params["chrome-trace-file"] = str(legacy_hash_path(job_id, case_hash))
        elif trace_val is False:
            # Frontend sends boolean False, but CLI expects string or None
            case_params["chrome-trace-file"] = None

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

    for case_params in cases:
        ch = _ch(case_params)
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            continue
        # Synthesize trace path if chrome_trace is enabled
        _synth_trace_path(case_params, ch)

        # === Case-level validation (RFC §3.6) ===
        error, error_fields = validate_case_for_module("throughput_optimizer", case_params, provided or set())
        if error:
            all_records.append(
                {
                    "config": {k: case_params.get(k) for k in _THROUGHPUT_MULTI_FIELDS},
                    "summary": {"error": error, "error_fields": error_fields, "validation_failed": True},
                    "tables": {},
                    "case_hash": ch,
                }
            )
            continue  # Skip execution, move to next case

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

    # CLI parity: if ALL cases failed (every record has an error), re-raise
    # so the job is marked as failed (matching CLI's crash-on-error behavior).
    # Partial failures (some cases succeeded) still return results for display.
    # Validation failures are excluded — they are not execution failures.
    real_execution_failures = [
        r for r in all_records if "error" in r.get("summary", {}) and not r["summary"].get("validation_failed")
    ]
    if all_records and len(real_execution_failures) == len(all_records):
        first_error = real_execution_failures[0].get("summary", {}).get("error", "Unknown error")
        raise RuntimeError(f"All {len(all_records)} case(s) failed. First error: {first_error}")

    return all_records, skipped


class ThroughputOptimizerAdapter:
    """Adapter implementing RunnerPort. Delegates to subprocess runner."""

    def run(
        self,
        params: dict[str, Any],
        *,
        job_id: str | None = None,
        on_progress: Callable[[int | None, str | None], None] | None = None,
        cancel_flag: Callable[[], bool] | None = None,
        cached_hashes: set[str] | None = None,
        form_schema_version: str | None = None,
        provided: set[str] | None = None,
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
            provided=provided,
        )


# ---------------------------------------------------------------------------
# Param -> argparse.Namespace mapping (registry-driven).
# ---------------------------------------------------------------------------


def _zero_for(data_type: str) -> Any:
    """Zero value for a registry data_type."""
    return {
        "integer": 0,
        "number": 0.0,
        "string": "",
        "boolean": False,
        "integer[]": [],
        "number[]": [],
        "string[]": [],
    }.get(data_type)


def _coerce_list(p, raw_list: list) -> list:
    """Normalize list input to match p.data_type.

    Args:
        p: Param definition.
        raw_list: Raw list from form.

    Returns:
        Normalized list.
    """
    if p.data_type == "integer[]":
        out: list[int] = []
        for v in raw_list:
            if isinstance(v, str):
                parts = [x.strip() for x in v.replace(",", " ").split() if x.strip()]
                out.extend(int(x) for x in parts)
            else:
                out.append(int(v))
        return out
    if p.data_type == "number[]":
        out_f: list[float] = []
        for v in raw_list:
            if isinstance(v, str):
                parts = [x.strip() for x in v.replace(",", " ").split() if x.strip()]
                out_f.extend(float(x) for x in parts)
            else:
                out_f.append(float(v))
        return out_f
    # string[] — pass through (choices already validated upstream)
    return [str(v) if not isinstance(v, str) else v for v in raw_list]


def _coerce_param(p, raw: Any) -> Any:
    """Registry-driven type coercion for one Param.

    Args:
        p: Param definition.
        raw: Raw value from form.

    Returns:
        Coerced value matching p.data_type.
    """

    default = p.default
    is_enum_defaulted = p.data_type == "string" and hasattr(default, "value") and hasattr(default, "name")

    # ── Missing / empty input: fall back to default ────────────────────
    # Treat None, "", and whitespace-only strings as "missing" so downstream
    # ``is None`` branches (e.g. speculative-method) see a consistent signal.
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        # number-or-None fields (default=None, no min/max bound): "" -> None, not 0.0
        if p.data_type == "number" and default is None:
            return None
        # Optional list fields (default=None): None stays None so downstream
        # ``if ns.tp_sizes is None`` branches still work (legacy CLI compat).
        if p.data_type.endswith("[]") and default is None:
            return None
        # integer fields with default=None: None stays None (not 0) so downstream
        # ``if ns.prefill_devices_per_instance is not None`` branches work correctly.
        # These are optional fields (e.g. PD ratio params), not required fields.
        if p.data_type == "integer" and default is None:
            return None
        # string fields with default=None: None stays None (not ""). Downstream
        # code branches on ``is None`` for optional string fields (e.g.
        # speculative-method: ``if spec_method is not None`` triggers the
        # speculative search path; an empty string would incorrectly activate it).
        # The previous cli_type-only guard (input-length YAML path dispatch) is
        # subsumed by this broader rule.
        if p.data_type == "string" and default is None:
            return None
        if default is not None:
            return default
        # default is None + no special case: return data_type zero
        return _zero_for(p.data_type)

    # ── List input ─────────────────────────────────────────────────────
    if isinstance(raw, list):
        if p.data_type.endswith("[]"):
            return _coerce_list(p, raw)
        # Scalar field got a list — take first element (defensive)
        return _coerce_param(p, raw[0] if raw else None)

    # ── Scalar coercion by data_type ───────────────────────────────────
    try:
        # Custom cli_type (e.g. check_positive_integer_and_string for input-length
        # which can be either a positive int or a YAML path). The registry's
        # cli_type is the authoritative conversion — reuse it verbatim so the
        # runner matches CLI behavior exactly.
        if getattr(p, "cli_type", None) is not None:
            return p.cli_type(raw)
        if p.data_type == "boolean":
            if isinstance(raw, str):
                return raw.lower() not in ("", "false", "0", "no")
            return bool(raw)
        if p.data_type == "integer":
            return int(raw)
        if p.data_type == "number":
            return float(raw)
        if p.data_type == "string":
            if is_enum_defaulted:
                # Enum-defaulted param: store the Enum member directly
                # (matches CLI argparse which also stores Enum members).
                # Accept both kebab formal values ("w8a8-dynamic") and
                # legacy UPPER_SNAKE ("W8A8_DYNAMIC").
                from services.enum_utils import coerce_enum

                enum_cls = type(default)
                # If raw is already the right Enum member, pass it through
                # directly (no str conversion, no deprecation warning).
                if isinstance(raw, enum_cls):
                    return raw
                return coerce_enum(enum_cls, str(raw))
            return str(raw)
        if p.data_type.endswith("[]"):
            # Single scalar value for a list field — wrap and recurse
            return _coerce_list(p, [raw])
    except (TypeError, ValueError):
        # Parse failure: for optional fields (default=None), return None so
        # downstream logic that branches on ``is None`` still works (matches
        # the old _num_or_none behavior for ttft_limit/tpot_limit etc).
        if default is None:
            return None
        return default
    return raw


def _build_namespace(params: dict[str, Any]) -> argparse.Namespace:
    """Build argparse.Namespace for ParallelRunner from params.

    Args:
        params: Form params dict.

    Returns:
        Configured Namespace with all fields coerced.
    """
    from cli.registry.modules import get_spec

    spec = get_spec("throughput_optimizer")
    ns = argparse.Namespace()
    for p in spec.fields:
        setattr(ns, p.dest_name, _coerce_param(p, params.get(p.name)))

    # ── Post-processing (NOT derivable from registry) ───────────────
    # device: set per-loop by the sweep (overwritten after _build_namespace).
    # The DEVICES param is a string[] but the sweep iterates device-by-device
    # and sets run_args.device = single device name; the namespace's initial
    # device value is irrelevant. Set to None to match the previous behavior
    # (and make test assertions about the initial state stable).
    ns.device = None

    # ── Backward-compat adjustments (legacy runner behavior vs registry) ──
    # The old hand-written _build_namespace used ad-hoc defaults that diverge
    # from the registry's declared defaults for several fields. ParallelRunner
    # depends on these specific values; align here until the registry/runner
    # contracts are reconciled. Each item is commented with the old behavior.
    #
    # mtp_acceptance_rate: old runner defaulted to [] (no MTP sweep) when
    # the field was absent. Registry default is [0.9, 0.6, 0.4, 0.2]. Keep
    # old behavior: when the field is absent in params, use []. (Note the
    # attribute is SINGULAR — the Param's cli_dest, matching ParallelRunner.)
    if "mtp-acceptance-rates" not in params:
        ns.mtp_acceptance_rate = []
    # image_batch_size / image_height / image_width: old runner defaulted to
    # None (not 0) when absent. ParallelRunner branches on `is not None`.
    if "image-batch-size" not in params:
        ns.image_batch_size = None
    if "image-height" not in params:
        ns.image_height = None
    if "image-width" not in params:
        ns.image_width = None
    # prefill/decode_devices_per_instance: old runner defaulted to None.
    if "prefill-devices-per-instance" not in params:
        ns.prefill_devices_per_instance = None
    if "decode-devices-per-instance" not in params:
        ns.decode_devices_per_instance = None
    # chrome_trace: old runner defaulted to None (not ''). _synth_trace_path
    # and ParallelRunner dispatch on `is not None` / `is True`.
    if "chrome-trace-file" not in params:
        ns.chrome_trace_file = None
    # num_mtp_tokens: registry declares integer[] (multi-value), but runner
    # reads args.num_mtp_tokens as a SINGLE int. Mirror the CLI's
    # _normalize_mtp_token_values: first candidate → num_mtp_tokens (int),
    # full deduped list → num_mtp_token_sizes (sweep candidates).
    mtp = ns.num_mtp_tokens
    if isinstance(mtp, list):
        normalized: list[int] = []
        for v in mtp:
            if v not in normalized:
                normalized.append(int(v))
        ns.num_mtp_tokens = normalized[0] if normalized else 0
        ns.num_mtp_token_sizes = normalized
    elif mtp is None:
        ns.num_mtp_tokens = 0
        ns.num_mtp_token_sizes = []
    else:
        ns.num_mtp_tokens = int(mtp)
        ns.num_mtp_token_sizes = [int(mtp)]
    # max_batched_tokens: keep None (CLI default) for consistency.
    # Removed: old runner defaulted to 8192, but CLI uses None.

    # Legacy CLI compat: when none of tp/ep/moe-dp sizes are specified,
    # the CLI searches TP across the default range (powers of 2 up to
    # num_devices). resolve_search_sizes treats ``None`` as "fixed, do not
    # search" and ``[]`` as "search the default range". Without this,
    # ParallelRunner explored only a single TP config.
    if ns.tp_sizes is None and ns.ep_sizes is None and ns.moe_dp_sizes is None:
        ns.tp_sizes = []
    # Unified speculative decoding (mirror of the CLI arg_parse post-steps):
    # When any speculative method is active (mtp/dflash/dspark), the legacy
    # MTP entry is forced off; the speculative search candidates reuse the
    # MTP search slot (num_mtp_token_sizes). G1 block/acceptance resolution
    # happens inside ParallelRunner (it re-resolves per-iteration for multi-N).
    spec_method = getattr(ns, "speculative_method", None)
    if spec_method is not None:
        # Legacy MTP entry forced off; search uses num_mtp_token_sizes.
        ns.num_mtp_tokens = 0
        ns.num_mtp_token_sizes = []

        n_raw = getattr(ns, "num_speculative_tokens", None)
        if isinstance(n_raw, list):
            n_candidates = []
            for v in n_raw:
                if v not in n_candidates:
                    n_candidates.append(int(v))
        elif n_raw is None:
            # Omitted: dflash/dspark → builtin block_size; mtp requires explicit n.
            if spec_method == "mtp":
                raise ValueError("--speculative-method mtp requires --num-speculative-tokens")
            from cli.utils import resolve_num_speculative_tokens_to_block

            builtin_n, _ = resolve_num_speculative_tokens_to_block(
                0,
                draft_model_config_path=getattr(ns, "draft_model_config_path", None),
                explicit=False,
            )
            n_candidates = [builtin_n]
        else:
            n_candidates = [int(n_raw)]

        if any(v == 0 for v in n_candidates):
            raise ValueError(
                "--speculative-method is set but --num-speculative-tokens contains 0. "
                "Omit --speculative-method for a baseline run, or pass n >= 1."
            )

        ns.num_speculative_token_sizes = n_candidates
        ns.num_mtp_token_sizes = list(n_candidates)
        ns.num_mtp_tokens = 0

        if len(n_candidates) == 1:
            ns.num_speculative_tokens = n_candidates[0]
            from cli.utils import resolve_draft_block_and_acceptance

            resolve_draft_block_and_acceptance(ns)
        else:
            ns.num_speculative_tokens = n_candidates[0]
    # Internal flag — set to True below when the warning fires
    ns.search_combination_warning_emitted = False
    # compilation_config: list of selected options → ParallelRunner reads 4
    # individual boolean attrs (not a registry concept, this is the runner's
    # API contract with ParallelRunner._apply_compilation_config).
    cfg = ns.compilation_config or []
    ns.enable_multistream = "enable_multistream" in cfg
    ns.enable_sequence_parallel = "enable_sequence_parallel" in cfg
    ns.enable_matmul_allreduce = "enable_matmul_allreduce" in cfg
    ns.enable_dispatch_ffn_combine = "enable_dispatch_ffn_combine" in cfg
    # log_level: convention field (RFC §3.5.1 — not in registry). ParallelRunner
    # reads args.log_level to configure each worker process's logging; omitting
    # it crashes every worker at startup.
    ns.log_level = params.get("log-level", "error") or "error"
    return ns


def _infer_mode(args, pd_ratio_mode: bool, disagg_mode: bool) -> str:
    """Decide optimizer mode.

    Args:
        args: Parsed namespace.
        pd_ratio_mode: Whether PD ratio optimization is enabled.
        disagg_mode: Whether disaggregation is enabled.

    Returns:
        Mode string: 'pd_ratio', 'disagg_prefill', 'disagg_decode', or 'aggregation'.
    """
    if pd_ratio_mode:
        return "pd_ratio"
    if disagg_mode:
        # Prefill phase if ttft set; Decode phase if tpot set.
        if getattr(args, "tpot_limits", None) is not None and getattr(args, "ttft_limits", None) is None:
            return "disagg_decode"
        return "disagg_prefill"
    return "aggregation"


# ---------------------------------------------------------------------------
# OptimizerSummary -> normalized rows
# ---------------------------------------------------------------------------


def _summary_to_rows(summary, device_name: str, mode: str, args) -> list[dict[str, Any]]:
    """Reshape OptimizerSummary into normalized record dicts.

    Args:
        summary: OptimizerSummary from ParallelRunner.
        device_name: Device profile name.
        mode: Optimizer mode.
        args: Parsed namespace.

    Returns:
        List of record dicts.
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
    """Coerce value to float; return None for missing/NaN/unparseable."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return None if math.isnan(f) else f


def _agg_disagg_row(r, device_name: str, mode: str, args) -> dict[str, Any]:
    """Build one normalized record dict from aggregation/disagg DataFrame row.

    Args:
        r: DataFrame row.
        device_name: Device profile name.
        mode: Optimizer mode.
        args: Parsed namespace.

    Returns:
        Record dict with config, summary, tables.
    """
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
    """Compute QPS for disagg row.

    Args:
        r: DataFrame row.
        output_length: Output length in tokens.
        mode: Either 'disagg_prefill' or 'disagg_decode'.

    Returns:
        QPS value or None if not computable.
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
    """Build one normalized record dict from prefill/decode-ratio DataFrame row.

    Args:
        r: DataFrame row.
        device_name: Device profile name.

    Returns:
        Record dict with config, summary, tables.
    """
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
