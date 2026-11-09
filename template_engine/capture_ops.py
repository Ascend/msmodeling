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

"""Capture one msmodeling runtime execution for the template graph engine.

The public :func:`capture` follows Card 1's contract.  It builds one
``tensor_cast.core.model_runner.ModelRunner``, applies compile switches before
model construction, executes one request through the existing observer hook,
and returns a pickle-safe operation list plus the analytic execution time.

Each process should capture at most one configuration at a time.  Do not use a
thread pool around this function: torch compiler state and model construction
are process-global.  If multiple configurations are needed, call this module's
CLI once per configuration or use separate ``spawn`` children.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from tensor_cast.core.compilation_config import apply_compilation_config
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction


MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro"
TEST_DEVICE = "TEST_DEVICE"
MIN_DYNAMO_CACHE_SIZE_LIMIT = 256


class CaptureSerializationError(ValueError):
    """Raised when a runtime value cannot be made pickle-safe."""


@dataclass(frozen=True)
class CapturedOp:
    """Serializable subset of ``OpInvokeInfo`` used by template generation.

    ``func`` is intentionally a string.  The template engine compares and
    emits operator names; it does not need to restore dispatcher callables.
    ``args``, ``kwargs`` and ``out`` contain Meta tensors created by one shared
    identity pool.
    """

    func: str
    args: Any
    kwargs: Any
    out: Any


class _MetaTensorPool:
    """Replace live tensors while preserving aliases across all op entries."""

    def __init__(self) -> None:
        self._by_source_id: dict[int, torch.Tensor] = {}
        self._source_tensors: list[torch.Tensor] = []

    def replace(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            source_id = id(value)
            if source_id not in self._by_source_id:
                self._source_tensors.append(value)
                try:
                    placeholder = torch.empty_strided(
                        tuple(int(dimension) for dimension in value.shape),
                        tuple(int(stride) for stride in value.stride()),
                        dtype=value.dtype,
                        device="meta",
                        requires_grad=bool(value.requires_grad),
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    raise CaptureSerializationError(f"cannot create Meta placeholder for tensor {value!r}") from error
                self._by_source_id[source_id] = placeholder
            return self._by_source_id[source_id]
        if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
            return value
        if isinstance(value, torch.Size):
            return torch.Size(int(dimension) for dimension in value)
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return value
        if isinstance(value, slice):
            return slice(self.replace(value.start), self.replace(value.stop), self.replace(value.step))
        if isinstance(value, Mapping):
            return {self.replace(key): self.replace(item) for key, item in value.items()}
        if isinstance(value, tuple):
            items = tuple(self.replace(item) for item in value)
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return items
        if isinstance(value, list):
            return [self.replace(item) for item in value]
        if isinstance(value, set):
            return {self.replace(item) for item in value}
        if isinstance(value, frozenset):
            return frozenset(self.replace(item) for item in value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            copied = copy.copy(value)
            for field in dataclasses.fields(value):
                try:
                    object.__setattr__(copied, field.name, self.replace(getattr(value, field.name)))
                except (AttributeError, TypeError, ValueError) as error:
                    raise CaptureSerializationError(
                        f"cannot recursively copy dataclass value {type(value).__name__}.{field.name}"
                    ) from error
            return copied
        # Scalar objects used as dispatcher arguments are normally covered by
        # the branches above.  Failing closed is safer than pickling a Runtime,
        # module, generator, or another object with hidden live state.
        raise CaptureSerializationError(
            f"unsupported runtime argument value: {type(value).__module__}.{type(value).__name__}"
        )


def _normalise_func(func: Any) -> str:
    if isinstance(func, str):
        return func
    result = str(func)
    if not result.strip():
        raise CaptureSerializationError(f"operator func string is empty: {func!r}")
    return result


def serialize_op_invoke_infos(op_invoke_infos: Iterable[Any]) -> list[CapturedOp]:
    """Convert raw runtime records into pickle-safe ``CapturedOp`` objects."""

    pool = _MetaTensorPool()
    serialized: list[CapturedOp] = []
    for index, op in enumerate(op_invoke_infos):
        for field_name in ("func", "args", "kwargs", "out"):
            if not hasattr(op, field_name):
                raise CaptureSerializationError(f"op record {index} does not expose {field_name}")
        serialized.append(
            CapturedOp(
                func=_normalise_func(op.func),
                args=pool.replace(op.args),
                kwargs=pool.replace(op.kwargs),
                out=pool.replace(op.out),
            )
        )
    return serialized


def serialize_runtime(runtime: Any) -> list[CapturedOp]:
    """Serialize ``runtime.op_invoke_infos`` with one cross-entry tensor pool."""

    records = getattr(runtime, "op_invoke_infos", None)
    if records is None:
        raise CaptureSerializationError("runtime does not expose op_invoke_infos")
    return serialize_op_invoke_infos(records)


def ensure_dynamo_cache_size_limit(minimum: int = MIN_DYNAMO_CACHE_SIZE_LIMIT) -> int:
    """Raise Dynamo's cache-size limit before the first compiled model build."""

    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError(f"minimum must be a positive integer, got {minimum!r}")
    import torch._dynamo.config as dynamo_config

    current = int(getattr(dynamo_config, "cache_size_limit", 0))
    if current < minimum:
        dynamo_config.cache_size_limit = minimum
        current = minimum
    return current


def _coerce_enum(enum_type: type[enum.Enum], value: Any) -> enum.Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unsupported {enum_type.__name__}: {value!r}") from error


def _resolve_compile_options(*, enable_dispatch_ffn_combine: bool, compile_options: Iterable[str] | None) -> list[str]:
    if compile_options is not None:
        return list(compile_options)
    return ["enable_dispatch_ffn_combine"] if enable_dispatch_ffn_combine else []


def _execution_time(metrics: Any) -> float:
    execution_times = getattr(metrics, "execution_time_s", None)
    if isinstance(execution_times, Mapping):
        if execution_times.get("analytic") is not None:
            return float(execution_times["analytic"])
        for value in execution_times.values():
            return float(value)
    run_time = getattr(metrics, "run_time_s", None)
    if run_time is None:
        raise RuntimeError("ModelRunner metrics expose neither execution_time_s nor run_time_s")
    return float(run_time)


def capture(
    B: int,
    S: int,
    is_decode: bool = False,
    tp: int = 1,
    ep: int = 1,
    moe_dp: int = 1,
    dcp: int = 1,
    do_compile: bool = False,
    num_mtp_tokens: int = 0,
    quantize_linear_action: str | QuantizeLinearAction = "DISABLED",
    quantize_attention_action: str | QuantizeAttentionAction = "DISABLED",
    enable_dispatch_ffn_combine: bool = False,
    model_id: str = MODEL_ID,
    num_hidden_layers_override: int = 0,
    compile_options: Iterable[str] | None = None,
    with_sampler: bool = False,
    meta_out: dict[str, Any] | None = None,
) -> tuple[list[CapturedOp], float]:
    """Capture one configuration and return ``(ops_list, exec_time_s)``.

    The model is selected by ``model_id`` and the device is ``TEST_DEVICE`` as
    required by Card 1.  ``exec_time_s`` prefers the analytic runtime estimate;
    if a custom runner has no execution-time map it falls back to wall-clock
    ``run_time_s``.  When ``meta_out`` is a dict it is filled with the
    model metadata (weight size, kv per-token cost, model_config) that
    lets the optimizer serve this weight key without modeling - the
    runner is built here anyway, so the metadata is free.
    """

    # Integer-axis positivity is enforced by build_cache_key, which every
    # entry point runs before capture(); only the divisibility rule needs a
    # local check because the key spelling does not encode it.
    if tp % dcp != 0:
        raise ValueError(f"tp ({tp}) must be divisible by dcp ({dcp})")

    selected_options = _resolve_compile_options(
        enable_dispatch_ffn_combine=enable_dispatch_ffn_combine,
        compile_options=compile_options,
    )
    if do_compile:
        ensure_dynamo_cache_size_limit()

    # This import must remain before ModelRunner construction.  Compilation
    # passes read the global config while build_model wraps the model.
    apply_compilation_config(selected_options)

    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner

    linear_action = _coerce_enum(QuantizeLinearAction, quantize_linear_action)
    attention_action = _coerce_enum(QuantizeAttentionAction, quantize_attention_action)
    from .config_adapter import user_input_for

    user_input = user_input_for(
        dict(
            B=B,
            S=S,
            TP=tp,
            EP=ep,
            MOE_DP=moe_dp,
            DCP=dcp,
            is_decode=is_decode,
            do_compile=do_compile,
            num_mtp_tokens=num_mtp_tokens,
            quantize_linear_action=linear_action,
            quantize_attention_action=attention_action,
            enable_dispatch_ffn_combine=enable_dispatch_ffn_combine,
        ),
        model_id=model_id,
        layers=num_hidden_layers_override,
        compilation_config=selected_options,
    )

    request = user_input.get_request_info()
    runner = ModelRunner(user_input)
    observed_runtimes: list[Any] = []
    metrics = runner.run_inference(
        requests=[request],
        generate_inputs_func=generate_inputs,
        with_sampler=with_sampler,
        runtime_observer=observed_runtimes.append,
    )
    if not observed_runtimes:
        raise RuntimeError("ModelRunner did not invoke runtime_observer")
    if meta_out is not None:
        _fill_model_meta(
            meta_out,
            user_input=user_input,
            metrics=metrics,
            runner=runner,
            request_tokens=int(request.concurrency * request.seq_len),
        )
    return serialize_runtime(observed_runtimes[-1]), _execution_time(metrics)


def _fill_model_meta(
    meta_out: dict[str, Any],
    *,
    user_input: Any,
    metrics: Any,
    runner: Any,
    request_tokens: int,
) -> None:
    """Record the weight-key model metadata from the live capture runner.

    ``kv_per_request_token_gb`` is derived from the measured kv bytes over
    the capture's billed token count, so serving-side scaling by the
    request's own (B, seq_len) is exact for ANY capture semantics.
    """

    from tensor_cast.core.input_generator import dcp_kv_token_capacity_factor  # noqa: PLC0415
    from template_engine.model_meta import user_weight_key  # noqa: PLC0415

    kv_cache_size_gb = float(getattr(metrics, "kv_cache_size_gb", 0.0) or 0.0)
    indexer_cache_size_gb = float(getattr(metrics, "indexer_cache_size_gb", 0.0) or 0.0)
    model_config = runner.model.model_config
    hf_config = getattr(model_config, "hf_config", None)
    raw_layers = getattr(hf_config, "num_hidden_layers", None)
    if raw_layers is None:
        # VL checkpoints keep the text depth one level down.
        text_config = getattr(hf_config, "text_config", None)
        raw_layers = getattr(text_config, "num_hidden_layers", None)
    if raw_layers is None and isinstance(model_config, dict):
        raw_layers = model_config.get("num_hidden_layers")
    try:
        depth = int(raw_layers or 0)
    except (TypeError, ValueError):
        depth = 0
    from .model_parameters import formula_parameters

    meta_out.update(
        {
            "model_parameters": formula_parameters(model_config),
            "num_hidden_layers": depth,
            "weight_key": user_weight_key(user_input),
            "do_compile": bool(user_input.do_compile),
            "decode_query_semantics": "mtp_plus_1",
            "model_weight_size_gb": float(metrics.model_weight_size_gb),
            "kv_cache_size_gb": kv_cache_size_gb,
            "kv_cache_per_token_gb": float(getattr(metrics, "kv_cache_per_token_gb", 0.0) or 0.0),
            "indexer_cache_size_gb": indexer_cache_size_gb,
            "indexer_cache_per_token_gb": float(getattr(metrics, "indexer_cache_per_token_gb", 0.0) or 0.0),
            "kv_per_request_token_gb": (kv_cache_size_gb / request_tokens if request_tokens > 0 else 0.0),
            "indexer_per_request_token_gb": (indexer_cache_size_gb / request_tokens if request_tokens > 0 else 0.0),
            "kv_cache_token_capacity_factor": int(dcp_kv_token_capacity_factor(runner.model)),
            "model_config": runner.model.model_config,
        }
    )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def write_capture(args) -> int:
    """Persist a CLI capture; parameter parsing belongs to cli/."""
    import json

    from template_engine.cache_io import build_cache_key, cache_is_valid, load_capture, save_capture, locate_capture

    from argparse import Namespace
    from .cache_io import cache_variant_dir

    args = Namespace(**vars(args))
    args.cache_dir = cache_variant_dir(args.cache_dir, args.layers or 0, args.sampler)
    key = build_cache_key(
        args.tp,
        args.ep,
        args.moe_dp,
        args.B,
        args.S,
        num_mtp_tokens=args.num_mtp_tokens,
        dcp=args.dcp,
        enable_dispatch_ffn_combine=args.dfc,
        quantize_linear_action=args.quant_linear,
        quantize_attention_action=args.quant_attention,
        is_decode=args.decode,
        do_compile=args.compile,
    )
    if cache_is_valid(key, args.cache_dir, model_id=args.model_id) and not args.force:
        cached = load_capture(key, args.cache_dir, model_id=args.model_id)
        assert cached is not None
        ops, exec_time = cached
        print(
            json.dumps(
                {
                    "key": key,
                    "op_count": len(ops),
                    "exec_time_s": exec_time,
                    "cached": True,
                    "path": str(locate_capture(key, args.cache_dir, model_id=args.model_id)),
                }
            )
        )
        return 0

    model_meta: dict[str, Any] = {}
    ops, exec_time = capture(
        args.B,
        args.S,
        is_decode=args.decode,
        tp=args.tp,
        ep=args.ep,
        moe_dp=args.moe_dp,
        dcp=args.dcp,
        do_compile=args.compile,
        num_mtp_tokens=args.num_mtp_tokens,
        quantize_linear_action=args.quant_linear,
        quantize_attention_action=args.quant_attention,
        enable_dispatch_ffn_combine=args.dfc,
        model_id=args.model_id,
        num_hidden_layers_override=args.layers,
        with_sampler=args.sampler,
        meta_out=model_meta,
    )
    path = save_capture(
        key,
        ops,
        exec_time,
        args.cache_dir,
        model_id=args.model_id,
        overwrite=args.force,
    )
    payload: dict[str, Any] = {
        "key": key,
        "op_count": len(ops),
        "exec_time_s": exec_time,
        "cached": False,
        "path": str(path),
    }
    model_meta["capture_options"] = {"layers": args.layers or 0, "sampler": bool(args.sampler)}
    from template_engine.graph_codec import atomic_write_bytes, encode_document

    model_meta.setdefault('model_id', args.model_id)
    atomic_write_bytes(Path(str(path) + '.meta'), encode_document(model_meta, 'model_meta/1'))
    payload['model_meta'] = model_meta.get('weight_key')
    print(json.dumps(payload))
    return 0


def main(argv=None):
    from cli.inference.template_graph import legacy_capture_main

    return legacy_capture_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
