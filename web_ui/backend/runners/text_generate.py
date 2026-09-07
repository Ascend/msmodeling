"""Text-generate runner adapter.

The heavy simulation runs in a subprocess via ``runners._worker``. The worker
calls ``execute(params)`` which builds a ``ModelRunner``, runs inference, and
returns the serialized envelope. ``ModelRunner.run`` returns a
``ModelRunnerMetrics`` dataclass; ``execute`` serializes its fields.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from models.entities import ResultRecord
from runners._multicase import resolve_model_id_path
from services.case_validation import validate_case_for_module

logger = logging.getLogger(__name__)


def _metrics_to_envelope(metrics: Any) -> dict[str, Any]:
    """Serialize ModelRunnerMetrics fields into result envelope dict.

    Args:
        metrics: ModelRunnerMetrics dataclass from ModelRunner.run.

    Returns:
        Dict with batch_size, run_time_s, memory_gb, breakdowns, etc.
    """
    breakdowns = getattr(metrics, "breakdowns", {}) or {}
    breakdowns_percent: dict[str, dict[str, float]] = {}
    for name, breakdown in breakdowns.items():
        try:
            total = sum(breakdown.values())
        except (TypeError, ValueError):
            continue
        if total == 0:
            continue
        breakdowns_percent[name] = {k: round(v * 100 / total, 4) for k, v in breakdown.items()}

    return {
        "batch_size": getattr(metrics, "batch_size", None),
        "run_time_s": getattr(metrics, "run_time_s", None),
        "execution_time_s": dict(getattr(metrics, "execution_time_s", {}) or {}),
        "tps_per_model": dict(getattr(metrics, "tps_per_model", {}) or {}),
        "memory_gb": {
            "total_device": getattr(metrics, "total_device_memory_gb", None),
            "model_weight": getattr(metrics, "model_weight_size_gb", None),
            "peak_usage": getattr(metrics, "peak_memory_usage_gb", None),
            "kv_cache": getattr(metrics, "kv_cache_size_gb", None),
            "kv_cache_per_token": getattr(metrics, "kv_cache_per_token_gb", None),
            "model_activation": getattr(metrics, "model_activation_size_gb", None),
            "reserved": getattr(metrics, "reserved_memory_gb", None),
            "available": getattr(metrics, "device_memory_available_gb", None),
        },
        "breakdowns_raw": {k: dict(v) for k, v in breakdowns.items()},
        "breakdowns_percent": breakdowns_percent,
        "perf_model_name": getattr(metrics, "perf_model_name", None),
    }


def _as_list(value: Any) -> list:
    """Normalize a multi-select value (list | str | None) into a list.

    Args:
        value: Input value from form.

    Returns:
        List of non-None values.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v is not None]
    return [value]


def _parse_int_list(value: Any) -> list[int]:
    """Parse a free-text comma/space list into ints.

    Args:
        value: Input value (list, int, or comma-separated string).

    Returns:
        List of ints.

    Raises:
        ValueError: If value cannot be parsed as int list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            raise ValueError(f"invalid integer list: {value!r}") from None
    if isinstance(value, int):
        return [value]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    try:
        return [int(p) for p in parts]
    except (TypeError, ValueError):
        raise ValueError(f"invalid integer list: {value!r}") from None


def _extract_op_breakdown(
    rt: Any,
    perf_model: str,
    dump_input_shapes: bool,
    dump_op_bound_results: bool,
) -> list[dict[str, Any]]:
    """Extract operator-level data with bound and input_shapes.

    Args:
        rt: Runtime object from ModelRunner.
        perf_model: Performance model name.
        dump_input_shapes: Whether to include input shapes.
        dump_op_bound_results: Whether to include bound percentages.

    Returns:
        List of operator breakdown dicts.
    """
    if not rt:
        return []

    try:
        aggregated = rt._aggregate_average_table_data(
            first_model=perf_model,
            group_by_input_shapes=dump_input_shapes,
            dump_op_bound_results=dump_op_bound_results,
        )
        sorted_items = rt._sort_average_table_items(aggregated, perf_model)
        rows: list[dict[str, Any]] = []

        for key, data in sorted_items:
            total_s = data.total_runtimes.get(perf_model, 0)
            row: dict[str, Any] = {
                "name": key.op_name,
                "bound": key.bound or None,
                "input_shapes": key.input_shapes or None,
                "total_s": total_s,
                "avg_s": total_s / data.count if data.count > 0 else 0,
                "calls": data.count,
            }

            if dump_op_bound_results:
                components = data.bound_components.get(perf_model, {})
                total = sum(components.values()) or 1
                row["bound_pct"] = {
                    "memory": components.get("memory", 0) / total * 100,
                    "comm": components.get("communication", 0) / total * 100,
                    "mma": components.get("mma", 0) / total * 100,
                    "gp": components.get("gp", 0) / total * 100,
                }

            rows.append(row)

        return rows
    except Exception:
        return []


# Fields expanded into multi-case cartesian product.
def _get_multi_case_fields() -> tuple[tuple[str, Callable], ...]:
    """Collect multi-case fields from UIFieldProps.multi_values=True.

    Returns:
        Tuple of (field_id, parse_fn) pairs. Parse function inferred from
        Param.data_type.
    """
    from cli.registry.modules import get_spec
    from web_ui.backend.services.ui_props.text_generate import UI

    spec = get_spec("text_generate")
    fields = []
    for p in spec.fields:
        ui = UI.get(p.name)
        if ui and ui.multi_values:
            # Infer parse function from data_type
            if p.data_type in ("integer",):
                fields.append((p.name, _parse_int_list))
            elif p.data_type in ("number",):
                fields.append((p.name, _parse_float_list))
            else:  # string, string[], etc.
                fields.append((p.name, _as_list))
    return tuple(fields)


_MULTI_CASE_FIELDS_WITH_PARSERS = _get_multi_case_fields()
_MULTI_CASE_FIELDS = tuple(fid for fid, _ in _MULTI_CASE_FIELDS_WITH_PARSERS)


def _parse_float_list(value: Any) -> list[float]:
    """Parse a free-text comma/space list into floats.

    Args:
        value: Input value (list, number, or comma-separated string).

    Returns:
        List of floats.

    Raises:
        ValueError: If value cannot be parsed as float list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            raise ValueError(f"invalid float list: {value!r}") from None
    if isinstance(value, (int, float)):
        return [float(value)]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    try:
        return [float(p) for p in parts]
    except (TypeError, ValueError):
        raise ValueError(f"invalid float list: {value!r}") from None


def _expand_cases(params: dict[str, Any], *, max_cases: int | None = None) -> list[dict[str, Any]]:
    """Expand multi-case fields into cartesian product of concrete case param dicts.

    Args:
        params: Original form params with potentially multi-value fields.
        max_cases: Override for MAX_CASES limit.

    Returns:
        List of case param dicts, one per combination.
    """
    import itertools

    # Build sources dict: field_id → list of values (parsed via field-specific parser)
    sources = {}
    for field_id, parse_fn in _MULTI_CASE_FIELDS_WITH_PARSERS:
        parsed = parse_fn(params.get(field_id))
        sources[field_id] = parsed if parsed else [params.get(field_id)]

    # Determine the case limit: explicit arg > params > no limit
    limit = max_cases if max_cases is not None else params.get("max-search-combinations")
    # Cap the cartesian product incrementally so a runaway form can't fan out
    # into thousands of sequentially-run cases.
    if limit is not None:
        total = 1
        for key in _MULTI_CASE_FIELDS:
            total *= len(sources[key])
            if total > limit:
                raise ValueError(f"too many cases: {total} > {limit}")
    cases: list[dict[str, Any]] = []
    for combo in itertools.product(*(sources[k] for k in _MULTI_CASE_FIELDS)):
        case = dict(params)
        for key, val in zip(_MULTI_CASE_FIELDS, combo):
            case[key] = val
        cases.append(case)
    return cases


def _run_one_case(params: dict[str, Any]) -> dict[str, Any]:
    """Run a single text-gen case; return its record dict.

    Args:
        params: Concrete case params (single values, not lists).

    Returns:
        Record dict with config, summary, and tables.
    """
    from tensor_cast.core.compilation_config import apply_compilation_config
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner

    # Apply compilation config to global config (same as CLI does).
    # Without this, the compilation passes read stale/default values from
    # config.compilation.* even though UserInputConfig has the correct flags.
    compilation_config = params.get("compilation-config")
    apply_compilation_config(compilation_config)

    user_input = _build_user_input(params)

    runner = ModelRunner(user_input)
    # Capture the raw Runtime via the observer (per examples/
    # text_generate_api_example.py) so we can aggregate its event_list into the
    # structured per-op table — metrics.runtime_event_list can be empty.
    runtime_holder: dict[str, Any] = {}

    def _capture_runtime(rt: Any) -> None:
        runtime_holder["rt"] = rt

    # Use the SAME input generator as the CLI (`generate_inputs`) so the web UI
    # op breakdown matches `python -m cli.inference.text_generate` op-for-op.
    # The ModelRunner default (`generate_inputs_varlen`) has a minor asymmetry
    # with the CLI in pure-decode batches: it builds a `selected_token_indices`
    # tensor where the CLI passes `None`, which causes the sampler/lm_head to
    # emit an extra `aten.index.Tensor` op (~2 µs) that the CLI does not.
    metrics = runner.run_inference(
        generate_inputs_func=generate_inputs,
        runtime_observer=_capture_runtime,
    )

    metrics.print_info()  # CLI result table (stdout) -> captured

    envelope = _metrics_to_envelope(metrics)
    perf_model = next(iter(metrics.execution_time_s), None) or "analytic"
    rt = runtime_holder.get("rt")
    # Extract rich operator-level data (replaces the old simple format)
    dump_input_shapes = params.get("dump-input-shapes", False)
    dump_op_bound_results = params.get("dump-op-bound-results", False)
    op_breakdown = _extract_op_breakdown(rt, perf_model, dump_input_shapes, dump_op_bound_results)

    envelope["op_breakdown"] = op_breakdown
    envelope["dump_input_shapes"] = dump_input_shapes
    envelope["dump_op_bound_results"] = dump_op_bound_results
    return {
        "config": {
            "device": params.get("device"),
            "model_id": params.get("model-id"),
            "num_devices": params.get("num-devices", 1),
            "num_queries": params.get("num-queries"),
            "quantize_linear_action": params.get("quantize-linear-action"),
            "quantize_attention_action": params.get("quantize-attention-action"),
            "tp_size": params.get("tp-size"),
        },
        "summary": {
            "run_time_s": envelope.get("run_time_s"),
            "tps_per_model": envelope.get("tps_per_model"),
        },
        "tables": envelope,
    }


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
    import traceback

    from cli.logo import print_logo
    from runners._multicase import compute_case_hash

    print_logo()  # CLI banner (stderr) -> captured
    print(f"[case-dedup] text_generate cached_hashes={len(cached_hashes or [])}", flush=True)
    # Resolve a relative model_id against the repo root (cwd is web/backend).
    params = {**params, "model-id": resolve_model_id_path(params.get("model-id"))}
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

    cases = _expand_cases(params)

    def _ch(cp: dict[str, Any]) -> str | None:
        return compute_case_hash("text_generate", form_schema_version, cp)

    cached = cached_hashes or set()
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    from services.capture import capture_case_log
    from runners._cli_command import build_cli_command_string

    # Case divider: a prominent separator line (80 `=` chars) between cases,
    # replacing the `[case i/n]` stamp (which had display issues in some log
    # viewers). Each case's CLI command is still logged, just without the stamp.
    _divider = "\n" + "=" * 80

    # Multi-case expansion summary: the parent process logged the reference
    # command for the ORIGINAL params; list the actual per-case commands here
    # so the job log reflects what will really be executed. No `[case i/n]`
    # stamp.
    if len(cases) > 1:
        print(f"{_divider}", flush=True)
        print(
            f"[text_generate] Expanding into {len(cases)} case(s) based on multi-field values:",
            flush=True,
        )
        for case_params in cases:
            print(f"  {build_cli_command_string('text_generate', case_params)}", flush=True)
        print(flush=True)

    if len(cases) == 1:
        ch = _ch(cases[0])
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            return records, skipped
        case_params = cases[0]
        _synth_trace_path(case_params, ch)

        # === Case-level validation (RFC §3.6) ===
        error, error_fields = validate_case_for_module("text_generate", case_params, provided or set())
        if error:
            records.append(
                {
                    "config": {
                        "model_id": case_params.get("model-id"),
                        "num_devices": case_params.get("num-devices", 1),
                    },
                    "summary": {"error": error, "error_fields": error_fields, "validation_failed": True},
                    "tables": {},
                    "case_hash": ch,
                    "case_log": "",
                }
            )
            return records, skipped

        with capture_case_log() as buf:
            # Divider + CLI command. No case index label.
            print(f"{_divider}", flush=True)
            print(f"CLI: {build_cli_command_string('text_generate', case_params)}", flush=True)
            print(flush=True)
            rec = _run_one_case(case_params)
        rec["case_hash"] = ch
        rec["case_log"] = buf.getvalue()
        records.append(rec)
        return records, skipped

    for case_params in cases:
        ch = _ch(case_params)
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            continue
        _synth_trace_path(case_params, ch)
        cfg = {k: case_params.get(k) for k in _MULTI_CASE_FIELDS}

        # === Case-level validation (RFC §3.6) ===
        error, error_fields = validate_case_for_module("text_generate", case_params, provided or set())
        if error:
            records.append(
                {
                    "config": {
                        **cfg,
                        "model_id": case_params.get("model-id"),
                        "num_devices": case_params.get("num-devices", 1),
                    },
                    "summary": {"error": error, "error_fields": error_fields, "validation_failed": True},
                    "tables": {},
                    "case_hash": ch,
                    "case_log": "",
                }
            )
            continue  # Skip execution, move to next case

        buf = None
        try:
            with capture_case_log() as buf:
                # Divider + CLI command + key values. No case index label.
                print(f"{_divider}", flush=True)
                print(f"CLI: {build_cli_command_string('text_generate', case_params)}", flush=True)
                print(
                    f"device={cfg['device']} "
                    f"num_queries={cfg['num-queries']} quant={cfg['quantize-linear-action']} "
                    f"att={cfg['quantize-attention-action']} tp={cfg['tp-size']}",
                    flush=True,
                )
                print(flush=True)
                rec = _run_one_case(case_params)
            rec["case_hash"] = ch
            rec["case_log"] = buf.getvalue()
            records.append(rec)
        except Exception as e:
            traceback.print_exc()
            records.append(
                {
                    "config": {
                        **cfg,
                        "model_id": case_params.get("model-id"),
                        "num_devices": case_params.get("num-devices", 1),
                    },
                    "summary": {"error": str(e)},
                    "tables": {},
                    "case_hash": ch,
                    # Preserve the captured CLI echo + partial output and append the
                    # formatted traceback, so the failed case keeps a case_log
                    # (records without a case_log are dropped at persistence time).
                    "case_log": (buf.getvalue() + traceback.format_exc()) if buf is not None else "",
                }
            )
    return records, skipped


class TextGenerateRunnerAdapter:
    """Adapter implementing RunnerPort. Delegates to subprocess runner."""

    def run(
        self,
        params: dict[str, Any],
        *,
        job_id: str,
        on_progress: Callable[[int | None, str | None], None] | None = None,
        cancel_flag: Callable[[], bool] | None = None,
        cached_hashes: set[str] | None = None,
        form_schema_version: str | None = None,
        provided: set[str] | None = None,
    ) -> tuple[list[ResultRecord], list[str]]:
        from runners._subprocess import run_module_subprocess

        return run_module_subprocess(
            "text_generate",
            params,
            job_id=job_id,
            on_progress=on_progress,
            cancel_flag=cancel_flag,
            cached_hashes=cached_hashes,
            form_schema_version=form_schema_version,
            provided=provided,
        )


def _build_user_input(params: dict[str, Any]):
    """Build UserInputConfig from frontend params (kebab-case keys).

    Mirrors CLI's UserInputConfig.from_args. Applies defaults from Param
    definitions and handles unified speculative decoding resolution.

    Args:
        params: Frontend params dict with kebab-case keys.

    Returns:
        Configured UserInputConfig instance.
    """
    from types import SimpleNamespace

    from cli.registry.modules import get_spec
    from tensor_cast.core.quantization.datatypes import (
        QuantizeAttentionAction,
        QuantizeLinearAction,
    )
    from tensor_cast.core.user_config import UserInputConfig
    from services.enum_utils import coerce_enum

    # Apply defaults from Param definition for missing fields
    # This ensures that if the frontend doesn't send a field, we use the Param default
    spec = get_spec("text_generate")
    filled_params = dict(params)  # Copy to avoid mutating the original
    for field in spec.fields:
        if field.name not in filled_params and field.default is not None:
            filled_params[field.name] = field.default

    # Convert kebab-case registry keys → snake_case so from_args can read them
    # (from_args expects CLI-style snake_case attribute names and applies its
    # own semantic mappings: num_devices→world_size, etc.)
    # Respect cli_dest mapping: some params have a different destination name
    # (e.g., "no-repetition" → "disable_repetition")
    renamed = {}
    for field in spec.fields:
        if field.name in filled_params:
            renamed[field.dest_name] = filled_params[field.name]
    # Also include any extra fields not in spec (shouldn't happen, but be safe)
    for k, v in filled_params.items():
        if k not in {f.name for f in spec.fields}:
            renamed[k.replace("-", "_")] = v
    ns = SimpleNamespace(**renamed)

    # Unified speculative decoding (mirror of the CLI arg_parse post-steps):
    # resolve num_speculative_tokens → draft_block_size, bridge mtp → legacy
    # num_mtp_tokens, and align decode query_length to n+1. G2/G3 argv checks
    # are CLI-only (the form enforces condition visibility), but resolution
    # is shared.
    from cli.utils import draft_method, resolve_draft_block_and_acceptance

    resolve_draft_block_and_acceptance(ns)
    method = draft_method(ns)
    if method is not None:
        n = int(getattr(ns, "num_speculative_tokens", 0) or 0)
        # Bridge --speculative-method mtp --num-speculative-tokens N onto the
        # legacy num_mtp_tokens field so ConfigResolver/MtpWrapper match
        # --num-mtp-tokens N.
        if method == "mtp" and int(getattr(ns, "num_mtp_tokens", 0) or 0) == 0:
            ns.num_mtp_tokens = n
        if getattr(ns, "decode", False):
            decode_query_len = n + 1
            if decode_query_len >= 2 and int(getattr(ns, "query_length", 0) or 0) != decode_query_len:
                label = {"dspark": "DSpark", "dflash": "Dflash", "mtp": "MTP"}.get(method, method)
                logger.warning(
                    "%s decode sets query-length to num_speculative_tokens+1 (%d); was %s",
                    label,
                    decode_query_len,
                    ns.query_length,
                )
                ns.query_length = decode_query_len

    user_input = UserInputConfig.from_args(ns)

    # The frontend sends quantize enums as plain strings (kebab or UPPER_SNAKE);
    # the CLI path receives QuantizeLinearAction enums via argparse ``type=``.
    # Downstream code calls ``action.name``, so coerce any str values back to their
    # StrEnum. Use coerce_enum for consistent kebab/UPPER_SNAKE handling (RFC §3.8.1).
    for fname, enum_cls in (
        ("quantize_linear_action", QuantizeLinearAction),
        ("quantize_non_expert_linear_action", QuantizeLinearAction),
        ("quantize_attention_action", QuantizeAttentionAction),
    ):
        raw_value = getattr(user_input, fname)
        if isinstance(raw_value, str):
            setattr(user_input, fname, coerce_enum(enum_cls, raw_value))
        # If already an enum member, leave it as-is
    return user_input
