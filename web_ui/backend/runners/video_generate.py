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
import os
from pathlib import Path
from typing import Any, Callable

from models.entities import ResultRecord
from runners._multicase import (
    aggregate_runtime_events,
    as_list as _as_list,
    parse_int_list as _parse_int_list,
    run_cases,
)

logger = logging.getLogger(__name__)

# The backend runs from web/backend; repo-relative paths (e.g. a model_id under
# tests/assets/) are resolved against the repo root, not the backend cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# Multi-case expansion fields (Phase D2): device + quantize are multi-select
# arrays, ulysses_size is a free-number comma-list.
_VIDEO_MULTI_FIELDS = {
    "device": _as_list,
    "quantize_linear_action": _as_list,
    "ulysses_size": _parse_int_list,
}


def _run_one_video_case(params: dict[str, Any]) -> dict[str, Any]:
    """Run a single (already concrete) video-gen case; return its record dict."""
    from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
    from runners._video_generate_runner import VideoGenerateRunner

    quant_action = params.get("quantize_linear_action", "W8A8_DYNAMIC")
    try:
        quant_enum = QuantizeLinearAction(quant_action)
    except (KeyError, ValueError):
        quant_enum = QuantizeLinearAction.W8A8_DYNAMIC

    # Resolve a relative model_id against the repo root (cwd is web/backend).
    model_id = params.get("model_id")
    if model_id and not os.path.isabs(model_id) and not os.path.isdir(model_id):
        candidate = _REPO_ROOT / model_id
        if candidate.is_dir():
            model_id = str(candidate)

    runner = VideoGenerateRunner(
        device=params.get("device"),
        model_id=model_id,
        dtype=params.get("dtype", "float16"),
        quantize_linear_action=quant_enum,
        mxfp4_group_size=int(params.get("mxfp4_group_size", 32) or 32),
        world_size=int(params.get("world_size", 1) or 1),
        ulysses_size=int(params.get("ulysses_size", 1) or 1),
    )

    # VideoGenerateRunner.run_inference wraps cli/inference/video_generate.py
    # :run_inference, which prints the CLI's progress + result to stdout.
    import time as _time

    print(f"[video_generate] starting run_inference at {_time.strftime('%H:%M:%S')}", flush=True)
    _t0 = _time.time()
    runtime = runner.run_inference(
        # Number fields arrive null/"" when cleared in the form; guard every int
        # (mirrors the height/width/... fields below) so int(None)/int("") can't crash.
        batch_size=int(params.get("batch_size", 1) or 1),
        seq_len=int(params.get("seq_len", 128) or 128),
        height=int(params.get("height", 832) or 832),
        width=int(params.get("width", 400) or 400),
        frame_num=int(params.get("frame_num", 81) or 81),
        sample_step=int(params.get("sample_step", 50) or 50),
        use_cfg=bool(params.get("use_cfg", False)),
        cfg_parallel=bool(params.get("cfg_parallel", False)),
        dit_cache=bool(params.get("dit_cache", False)),
        # Optional str range fields: the form sends "" when cleared, but
        # run_inference checks `cache_*_range is None` then splits on "," — a
        # stray "" bypasses the None check and crashes on int(""). Coerce "" -> None.
        cache_step_range=(params.get("cache_step_range") or None),
        cache_step_interval=int(params.get("cache_step_interval", 1) or 1),
        cache_block_range=(params.get("cache_block_range") or None),
    )
    print(f"[video_generate] run_inference completed in {_time.time() - _t0:.1f}s", flush=True)

    chrome_trace = params.get("chrome_trace")
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
            "model_id": params.get("model_id"),
            "world_size": params.get("world_size", 1),
            "quantize_linear_action": params.get("quantize_linear_action"),
            "ulysses_size": params.get("ulysses_size", 1),
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
