"""Video-generate runner adapter.

Phase B: the heavy simulation runs in a ``runners._worker`` SUBPROCESS. The
worker calls ``execute(params)`` below, which builds a ``VideoGenerateRunner``
and runs DiT inference. ``VideoGenerateRunner.run_inference`` wraps
``cli/inference/video_generate.py:run_inference`` and already prints the CLI's
progress + result logs, so those land in the job log via the streamed stdout.
The adapter's ``run()`` is a thin spawner.

From the returned ``Runtime`` it persists the ``execution_time_s`` /
``breakdowns`` / ``table_rows`` envelope into the result record.

NOTE: ``Runtime.table_averages()`` returns a STRING; reshaping it into flat rows
depends on a _private_ Runtime helper (``_aggregate_average_table_data``). We
best-effort reshape when that helper is available and otherwise persist the raw
string under ``tables.table_averages_text`` (request a
PUBLIC structured accessor as a follow-up).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from models.entities import ResultRecord
from runners._multicase import (
    aggregate_runtime_events,
    as_list as _as_list,
    parse_int_list as _parse_int_list,
    resolve_model_id_path,
    run_cases,
)

logger = logging.getLogger(__name__)

# The backend runs from web/backend; repo-relative paths (e.g. a model_id under
# tests/assets/) are resolved against the repo root, not the backend cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _get_multi_case_fields() -> tuple[tuple[str, Callable], ...]:
    """Collect multi-case fields from UIFieldProps.multi_values=True.

    Returns tuple of (field_id, parse_fn) pairs. Parse function is inferred
    from Param.data_type: string → _as_list, integer → _parse_int_list,
    number → parse_float_list.
    """
    from cli.registry.modules import get_spec
    from web_ui.backend.services.ui_props.video_generate import UI

    spec = get_spec("video_generate")
    fields = []
    for p in spec.fields:
        ui = UI.get(p.name)
        if ui and ui.multi_values:
            # Infer parse function from data_type
            if p.data_type in ("integer",):
                fields.append((p.name, _parse_int_list))
            elif p.data_type in ("number",):
                # Import parse_float_list if needed
                from runners._multicase import parse_float_list

                fields.append((p.name, parse_float_list))
            else:  # string, string[], etc.
                fields.append((p.name, _as_list))
    return tuple(fields)


_VIDEO_MULTI_FIELDS_WITH_PARSERS = _get_multi_case_fields()
_VIDEO_MULTI_FIELDS = {fid: parser for fid, parser in _VIDEO_MULTI_FIELDS_WITH_PARSERS}


def _run_one_video_case(params: dict[str, Any]) -> dict[str, Any]:
    """Run a single (already concrete) video-gen case; return its record dict."""
    from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
    from runners._video_generate_runner import VideoGenerateRunner
    from services.enum_utils import coerce_enum

    quant_action = params.get("quantize-linear-action", "W8A8_DYNAMIC")
    # Use coerce_enum for consistent kebab/UPPER_SNAKE handling (RFC §3.8.1)
    try:
        quant_enum = coerce_enum(QuantizeLinearAction, quant_action)
    except ValueError:
        quant_enum = QuantizeLinearAction.W8A8_DYNAMIC

    # Resolve a relative model_id: prefer tests/assets/model_config/<id>,
    # then repo root, then fall back to remote id.
    model_id = resolve_model_id_path(params.get("model-id"))

    runner = VideoGenerateRunner(
        device=params.get("device"),
        model_id=model_id,
        dtype=params.get("dtype", "float16"),
        quantize_linear_action=quant_enum,
        mxfp4_group_size=int(params.get("mxfp4-group-size", 32) or 32),
        world_size=int(params.get("num-devices", 1) or 1),
        ulysses_size=int(params.get("ulysses-size", 1) or 1),
    )

    # VideoGenerateRunner.run_inference wraps cli/inference/video_generate.py
    # :run_inference, which prints the CLI's progress + result to stdout.
    import time as _time

    print(f"[video_generate] starting run_inference at {_time.strftime('%H:%M:%S')}", flush=True)
    _t0 = _time.time()
    runtime = runner.run_inference(
        # Number fields arrive null/"" when cleared in the form; guard every int
        # (mirrors the height/width/... fields below) so int(None)/int("") can't crash.
        batch_size=int(params.get("batch-size", 1) or 1),
        seq_len=int(params.get("seq-len", 128) or 128),
        height=int(params.get("height", 832) or 832),
        width=int(params.get("width", 400) or 400),
        frame_num=int(params.get("frame-num", 81) or 81),
        sample_step=int(params.get("sample-step", 50) or 50),
        use_cfg=bool(params.get("use-cfg", False)),
        cfg_parallel=bool(params.get("cfg-parallel", False)),
        dit_cache=bool(params.get("dit-cache", False)),
        # Optional str range fields: the form sends "" when cleared, but
        # run_inference checks `cache_*_range is None` then splits on "," — a
        # stray "" bypasses the None check and crashes on int(""). Coerce "" -> None.
        cache_step_range=(params.get("cache-step-range") or None),
        cache_step_interval=int(params.get("cache-step-interval", 1) or 1),
        cache_block_range=(params.get("cache-block-range") or None),
    )
    print(f"[video_generate] run_inference completed in {_time.time() - _t0:.1f}s", flush=True)

    chrome_trace = params.get("chrome-trace-file")
    if chrome_trace:
        try:
            runtime.export_chrome_trace(str(chrome_trace))
        except Exception:
            logger.debug("Failed to export chrome trace", exc_info=True)

    # NOTE: run_inference (cli/inference/video_generate.py:run_inference) already
    # prints runtime.table_averages() to stdout, so we do NOT re-print it here —
    # doing so would duplicate the result table in the job log.

    envelope = _runtime_to_envelope(runtime)
    # Structured per-op breakdown (Name / total / avg / # of Calls) for the
    # comparison UI — aggregate the runtime's event_list (NOT the printed log).
    try:
        evs = runtime.event_list or []
        pm = next(iter((evs[0].perf_results or {})), None) if evs else None
        envelope["op_breakdown"] = aggregate_runtime_events(evs, pm)
    except Exception:
        envelope["op_breakdown"] = []
    return {
        "config": {
            "device": params.get("device"),
            "model_id": params.get("model-id"),
            "world_size": params.get("num-devices", 1),
            "quantize_linear_action": params.get("quantize-linear-action"),
            "ulysses_size": params.get("ulysses-size", 1),
        },
        "summary": {"execution_time_s": envelope.get("execution_time_s")},
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
    """Worker-side entry (runs in the ``runners._worker`` subprocess).

    Multi-case (Phase D2): device / quantize_linear_action / ulysses_size may
    carry multiple values; their cartesian product is run sequentially (see
    ``runners._multicase.run_cases``). Case-level dedup: cases whose hash is in
    ``cached_hashes`` are skipped (in the returned ``skipped`` list).
    """
    from cli.logo import print_logo

    print_logo()  # CLI banner
    print(f"[case-dedup] video_generate cached_hashes={len(cached_hashes or [])}", flush=True)
    return run_cases(
        params,
        _VIDEO_MULTI_FIELDS,
        _run_one_video_case,
        cached_hashes=cached_hashes,
        case_hash_ctx=("video_generate", form_schema_version),
        job_id=job_id,
        provided=provided,
    )


class VideoGenerateRunnerAdapter:
    """Adapter implementing ``application/ports/RunnerPort``.

    Thin spawner (Phase B): delegates to ``runners._subprocess.run_module_subprocess``.
    """

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
            "video_generate",
            params,
            job_id=job_id,
            on_progress=on_progress,
            cancel_flag=cancel_flag,
            cached_hashes=cached_hashes,
            form_schema_version=form_schema_version,
            provided=provided,
        )


def _runtime_to_envelope(runtime: Any) -> dict[str, Any]:
    """Reshape a ``Runtime`` into the persisted result envelope."""
    envelope: dict[str, Any] = {
        "execution_time_s": {},
        "breakdowns": {},
        "table_rows": [],
    }
    try:
        envelope["execution_time_s"] = dict(runtime.total_execution_time_s() or {})
    except Exception:  # core metric — degrade to {} but make it visible
        logger.warning("total_execution_time_s() failed; execution_time_s will be empty", exc_info=True)
    try:
        envelope["breakdowns"] = {name: dict(bd) for name, bd in (runtime.get_breakdowns() or {}).items()}
    except Exception:  # core metric — degrade to {} but make it visible
        logger.warning("get_breakdowns() failed; breakdowns will be empty", exc_info=True)

    # Best-effort structured table rows (private helper; may be absent).
    rows = _try_structured_table_rows(runtime)
    if rows:
        envelope["table_rows"] = rows
    else:
        try:
            envelope["table_averages_text"] = runtime.table_averages()
        except Exception:
            envelope["table_averages_text"] = ""
    return envelope


def _try_structured_table_rows(runtime: Any) -> list[dict[str, Any]]:
    """Use ``Runtime._aggregate_average_table_data`` if present to build rows.

    The dict maps ``OpAverageGroupKey`` (op_name/bound/input_shapes) ->
    ``OpAverageGroupData`` (total_runtimes: Dict[model,float], count,
    bound_components). We flatten to one row per (key, model).
    """
    helper = getattr(runtime, "_aggregate_average_table_data", None)
    if not callable(helper):
        return []
    try:
        aggregated = helper()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for key, data in (aggregated or {}).items():
        op_name = getattr(key, "op_name", None)
        bound = getattr(key, "bound", None)
        input_shapes = getattr(key, "input_shapes", None)
        totals = getattr(data, "total_runtimes", {}) or {}
        count = getattr(data, "count", 0) or 1
        for model, total in totals.items():
            rows.append(
                {
                    "op_name": str(op_name),
                    "bound": str(bound),
                    "input_shapes": str(input_shapes),
                    "model": str(model),
                    "total": float(total),
                    "avg": float(total) / float(count) if count else None,
                    "call_count": int(count),
                }
            )
    return rows
