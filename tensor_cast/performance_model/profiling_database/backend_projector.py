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

"""Versioned projection boundary between TensorCast and kernel queries."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Iterable

from .query_demand import (
    QUERY_TRACE_MODEL_ENV,
    QUERY_TRACE_WORKLOAD_ENV,
    KernelQueryDemand,
    QueryDemandTraceWriter,
    projector_version_from_mapping,
    trace_writer_from_environment,
)


V4_CACHE_CONTRACT = "ascend_v4_cache_v1"


def v4_full_prefill_metadata_matches(row: dict, tokens: int) -> bool:
    """Reject contradictory request evidence; preserve verified legacy seeds."""
    phase = row.get("Runtime phase", "")
    if phase is not None and str(phase).strip().lower() not in {"", "nan", "prefill", "full_prefill"}:
        return False
    for key in ("query_lengths", "kv_lengths", "actual_seq_lengths_values", "actual_seq_lengths_kv_values"):
        raw = row.get(f"Runtime {key}", "")
        if raw is None or str(raw).strip().lower() in {"", "nan"}:
            continue
        try:
            values = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(values, Real):
                values = [values]
            if list(values) != [tokens]:
                return False
        except (ValueError, TypeError):
            return False
    return True


def v4_profile_shapes(value: str) -> tuple[tuple[int, ...], ...]:
    """Preserve empty ABI slots, including QLI's optional value output."""
    try:
        return _v4_shapes(value)
    except (ValueError, AttributeError):
        return ()


def _v4_profile_dtypes(value: str) -> tuple[str, ...]:
    aliases = {"FLOAT16": "DT_FLOAT16", "BF16": "DT_BF16", "DT_INT8": "INT8", "DT_INT32": "INT32"}
    return tuple(aliases.get(slot, slot) for slot in _v4_slots(value))


def validate_v4_indexer_physical_shapes(
    inputs: tuple[tuple[int, ...], ...], outputs: tuple[tuple[int, ...], ...]
) -> bool:
    """Validate the shared INT8 ABI, retaining allocation capacity and padded top-k."""
    if (
        len(inputs) != 9
        or len(inputs[0]) != 3
        or len(inputs[1]) != 4
        or len(outputs) != 2
        or len(outputs[0]) != 3
        or outputs[1] != ()
        or any(dim <= 0 for shape in inputs for dim in shape)
    ):
        return False
    tokens, heads, dim = inputs[0]
    cache = inputs[1]
    if cache[2:] != (1, dim) or inputs[2:4] != ((tokens, heads), (tokens, heads)) or inputs[4] != cache[:3]:
        return False
    if inputs[5:7] != ((1,), (1,)) or len(inputs[7]) != 2 or inputs[7][0] != 1 or inputs[8] != (1024,):
        return False
    active = tokens // 4
    return (
        active > 0
        and outputs[0][:2] == (tokens, 1)
        and 0 < outputs[0][-1] <= 2048
        and cache[0] * cache[1] >= active
        and inputs[7][1] * cache[1] >= active
    )


def project_v4_cache_leaf(demand: KernelQueryDemand, template: dict) -> KernelQueryDemand | None:
    """Lower verified Ascend V4 leaves; shared by runtime lookup and collection.

    Pool allocation is not active sequence length. Preserve the captured pool
    and block-table allocations only when they can hold the requested work.
    A caller must opt into the versioned contract; ordinary/reference ops must
    never be reinterpreted as these physical kernels by shape alone.
    """
    if demand.attributes.get("cache_layout") != "PA_BSND":
        return None
    recorded = v4_profile_shapes(str(template.get("Input Shapes", "")))
    outputs = v4_profile_shapes(str(template.get("Output Shapes", "")))
    types = _v4_profile_dtypes(str(template.get("Input Data Types", "")))
    output_types = _v4_profile_dtypes(str(template.get("Output Data Types", "")))
    formats = str(template.get("Input Formats", "")).strip('"').split(";")
    output_formats = str(template.get("Output Formats", "")).strip('"').split(";")
    shapes, dtypes = demand.input_shapes, demand.input_dtypes
    if any(dim <= 0 for shape in recorded for dim in shape):
        return None
    if demand.kernel_type == "ScatterNdUpdateV2":
        if len(shapes) != 3 or len(recorded) != 3 or len(dtypes) != 3 or len(types) != 3:
            return None
        update, cache, indices = shapes
        if len(update) != 3 or len(cache) != 3 or len(recorded[0]) != 4:
            return None
        tokens, width = math.prod(update[:-1]), update[-1]
        if min(*update, *cache) <= 0 or cache[-1] != width or math.prod(indices) != tokens:
            return None
        if (
            dtypes[0] != dtypes[1]
            or dtypes[0] not in {"DT_BF16", "DT_FLOAT16", "INT8"}
            or dtypes[2] not in {"INT32", "INT64"}
        ):
            return None
        if types != (dtypes[0], "INT32", dtypes[0]) or recorded[0][1:] != (cache[1], 1, width):
            return None
        if outputs != (recorded[0],) or output_types != (types[0],):
            return None
        if len(recorded[2]) not in (2, 3) or recorded[2][-1] != width:
            return None
        update_format = "ND" if len(recorded[2]) == 2 else "NCL"
        if formats != ["NCHW", "ND", update_format] or output_formats != ["NCHW"]:
            return None
        boundary = demand.attributes.get("boundary_rows", 0)
        if type(boundary) is not int or not 0 <= boundary <= update[0] or boundary >= tokens:
            return None
        if cache[0] * cache[1] < tokens - boundary or recorded[0][0] * cache[1] < tokens:
            return None
        physical_update = (tokens, width) if len(recorded[2]) == 2 else (tokens, 1, width)
        inputs = (recorded[0], (tokens, 2), physical_update)
        return replace(
            demand, input_shapes=inputs, output_shapes=(recorded[0],), input_dtypes=types, output_dtypes=output_types
        )
    if demand.kernel_type != "VllmQuantLightningIndexer" or len(shapes) != 4 or len(recorded) != 9:
        return None
    query, weights, cache, query_scale = shapes
    query_lengths, kv_lengths = (demand.attributes.get("query_lengths"), demand.attributes.get("kv_lengths"))
    if not isinstance(query_lengths, (list, tuple)) or len(query_lengths) != 1 or list(query_lengths) != kv_lengths:
        return None  # The verified replay contract is single-request full prefill.
    if demand.attributes.get("tp_world_size") != 1 or len(query) != 4 or query[0] != 1 or len(cache) != 3:
        return None
    _, tokens, heads, dim = query
    topk = demand.attributes.get("topk")
    if min(*query, *cache) <= 0 or query_lengths != [tokens] or not isinstance(topk, int) or not 0 < topk <= 2048:
        return None
    if dtypes != ("INT8", "DT_FLOAT16", "INT8", "DT_FLOAT16") or weights != query[:3] or query_scale != weights:
        return None
    expected_types = ("INT8", "INT8", "DT_FLOAT16", "DT_FLOAT16", "DT_FLOAT16", "INT32", "INT32", "INT32", "INT32")
    if types != expected_types or output_types != ("INT32", "FLOAT") or len(outputs) != 2 or outputs[1] != ():
        return None
    if formats != ["ND"] * 9 or output_formats != ["ND", "ND"] or len(outputs[0]) != 3 or outputs[0][1] != 1:
        return None
    if len(recorded[0]) != 3 or recorded[0][1:] != (heads, dim) or cache[-1] != dim:
        return None
    if not v4_full_prefill_metadata_matches(template, recorded[0][0]) or tokens // 4 == 0:
        return None
    if len(recorded[1]) != 4 or recorded[1][1:] != (cache[1], 1, dim):
        return None
    if recorded[5:7] != ((1,), (1,)) or len(recorded[7]) != 2 or recorded[7][0] != 1 or recorded[8] != (1024,):
        return None
    active = tokens // 4
    if min(cache[0], recorded[1][0]) * cache[1] < active or recorded[7][1] * cache[1] < active:
        return None
    inputs = (
        (tokens, heads, dim),
        recorded[1],
        (tokens, heads),
        (tokens, heads),
        (recorded[1][0], cache[1], 1),
        (1,),
        (1,),
        recorded[7],
        (1024,),
    )
    outputs = ((tokens, 1, topk), ())
    if not validate_v4_indexer_physical_shapes(inputs, outputs):
        return None
    return replace(
        demand,
        input_shapes=inputs,
        output_shapes=outputs,
        input_dtypes=types,
        output_dtypes=output_types,
    )


class CANNBackendProjector:
    """Record the exact normalized demands attempted by one CANN mapping."""

    def __init__(
        self,
        op_mapping: dict[str, Any],
        *,
        tensor_parallel_size: int | None = None,
        expert_parallel_size: int | None = None,
        writer: QueryDemandTraceWriter | None = None,
    ) -> None:
        self.version = projector_version_from_mapping(op_mapping)
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.writer = writer if writer is not None else trace_writer_from_environment()

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def record(
        self,
        *,
        op_name: str,
        kernel_types: Iterable[str],
        query_mode: str,
        input_shapes: Iterable[Iterable[int]] = (),
        output_shapes: Iterable[Iterable[int]] = (),
        input_dtypes: Iterable[str] = (),
        output_dtypes: Iterable[str] = (),
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self.writer is None:
            return
        normalized_inputs = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
        normalized_outputs = tuple(tuple(int(dim) for dim in shape) for shape in output_shapes)
        normalized_input_dtypes = tuple(str(dtype) for dtype in input_dtypes)
        normalized_output_dtypes = tuple(str(dtype) for dtype in output_dtypes)
        for candidate_rank, kernel_type in enumerate(dict.fromkeys(kernel_types)):
            if not kernel_type:
                continue
            self.writer.record(
                KernelQueryDemand(
                    projector_version=self.version,
                    op_name=op_name,
                    kernel_type=kernel_type,
                    query_mode=query_mode,
                    input_shapes=normalized_inputs,
                    output_shapes=normalized_outputs,
                    input_dtypes=normalized_input_dtypes,
                    output_dtypes=normalized_output_dtypes,
                    attributes=dict(attributes or {}),
                    tensor_parallel_size=self.tensor_parallel_size,
                    expert_parallel_size=self.expert_parallel_size,
                    candidate_rank=candidate_rank,
                    model_id=os.environ.get(QUERY_TRACE_MODEL_ENV, ""),
                    workload_id=os.environ.get(QUERY_TRACE_WORKLOAD_ENV, ""),
                )
            )


# V4 semantic-to-physical projection shared by database lookup and shape generation.
_V4_CONTRACT = "v4_requests_v2"
_V4_LEGACY_CONTRACT = "v4_single_prefill_v1"
_V4_KERNELS = {"Compressor", "SparseAttnSharedkv"}
_V4_RUNTIME_WORKLOAD = "Runtime v4_workload"
_V4_DTYPES = {
    "BF16": "DT_BF16",
    "FLOAT": "DT_FLOAT",
    "INT32": "DT_INT32",
    "bfloat16": "DT_BF16",
    "float32": "DT_FLOAT",
    "int32": "DT_INT32",
    "int8": "INT8",
}


def _v4_dtype_name(value: Any) -> str:
    value = str(value).removeprefix("torch.")
    return _V4_DTYPES.get(value, value)


def _v4_slots(value: Any) -> list[str]:
    text = str(value).strip().strip('"')
    return [part.strip().strip('"') for part in text.split(";")] if text else []


def _v4_shapes(value: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(dim) for dim in part.split(",")) if part else () for part in _v4_slots(value))


@dataclass(frozen=True)
class _V4PhysicalCase:
    inputs: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]

    @property
    def input_formats(self) -> tuple[str, ...]:
        return tuple("ND" if shape else "NULL" for shape in self.inputs)

    @property
    def output_formats(self) -> tuple[str, ...]:
        return ("ND",) * len(self.outputs)

    def matches(self, row: Mapping[str, Any]) -> bool:
        return (
            self.inputs == _v4_shapes(row.get("Input Shapes", ""))
            and self.outputs == _v4_shapes(row.get("Output Shapes", ""))
            and self.input_dtypes == tuple(_v4_dtype_name(v) for v in _v4_slots(row.get("Input Data Types", "")))
            and self.output_dtypes == tuple(_v4_dtype_name(v) for v in _v4_slots(row.get("Output Data Types", "")))
            and self.input_formats == tuple(_v4_slots(row.get("Input Formats", "")))
            and self.output_formats == tuple(_v4_slots(row.get("Output Formats", "")))
        )


def _v4_lengths(value: Any) -> list[int] | None:
    # Attention metadata is CPU data. Never synchronize a device tensor here.
    if value is None or getattr(getattr(value, "device", None), "type", None) != "cpu":
        return None
    try:
        raw = value.reshape(-1).tolist()
        return [int(item) for item in raw] if all(int(item) == item for item in raw) else None
    except (RuntimeError, TypeError, ValueError, OverflowError):
        return None


def _v4_request_workload(query: list[int], kv: list[int]) -> dict[str, Any]:
    if (
        not query
        or len(query) != len(kv)
        or any(type(v) is not int for v in [*query, *kv])
        or any(q <= 0 or s < q or s >= 2**31 for q, s in zip(query, kv))
        or sum(query) >= 2**31
        or sum(kv) >= 2**31
    ):
        raise ValueError("v4_request_lengths_invalid")
    phase = "decode" if all(q == 1 for q in query) else "prefill" if query == kv else "chunked_prefill"
    return {"tokens": sum(query), "query_lengths": query, "kv_lengths": kv, "phase": phase}


def normalize_v4_workload(workload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(workload)
    if result.get("rotate", False):
        raise ValueError("v4_rotated_compressor_unverified")
    if result.get("kv_quant_dtype") is not None:
        raise ValueError("v4_quantized_kv_unverified")
    if result.get("contract") not in (_V4_CONTRACT, _V4_LEGACY_CONTRACT):
        raise ValueError("v4_projection_contract_missing")
    query = result.get("query_lengths", [result["tokens"]])
    kv = result.get("kv_lengths", [result["tokens"]])
    requests = _v4_request_workload(query, kv)
    if result["tokens"] != requests["tokens"] or result.get("phase", requests["phase"]) != requests["phase"]:
        raise ValueError("v4_request_metadata_conflict")
    result.update(requests, contract=_V4_CONTRACT)
    return result


def _v4_semantic_workload(kernel: str, args: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("contract") not in (_V4_CONTRACT, _V4_LEGACY_CONTRACT) or kernel not in _V4_KERNELS:
        raise ValueError("v4_projection_contract_missing")
    if args.get("cache_layout", "BND") != "PA_BSND":
        raise ValueError("v4_paged_runtime_required")
    x = args.get("hidden_states" if kernel == "Compressor" else "q")
    rank = 3 if kernel == "Compressor" else 4
    if x is None or x.ndim != rank:
        raise ValueError("v4_input_rank_invalid")
    seq = _v4_lengths(args.get("seq_lens"))
    query = _v4_lengths(args.get("query_lens"))
    if seq is None or query is None:
        raise ValueError("v4_sequence_metadata_missing")
    requests = _v4_request_workload(query, seq)
    if policy.get("contract") == _V4_LEGACY_CONTRACT and (len(query) != 1 or requests["phase"] != "prefill"):
        raise ValueError("v4_request_projection_requires_v2")
    tokens = requests["tokens"]
    if x.shape[0] == 1:
        valid_shape = x.shape[1] == tokens
    else:
        valid_shape = x.shape[0] == len(query) and x.shape[1] == max(query)
    if not valid_shape:
        raise ValueError("v4_request_tensor_shape_mismatch")
    if _v4_dtype_name(x.dtype) != "DT_BF16":
        raise ValueError("v4_bf16_required")
    ratio = int(args.get("compress_ratio", -1))
    result = {"contract": _V4_CONTRACT, "compress_ratio": ratio, **requests}
    if kernel == "Compressor":
        head = int(args["head_dim"])
        rope = int(args["rope_head_dim"])
        cache = args.get("kv_cache")
        cache_dtype = _v4_dtype_name(cache.dtype) if cache is not None else None
        if cache_dtype != "DT_BF16" and not (args.get("rotate") is True and cache_dtype == "INT8"):
            raise ValueError("v4_compressor_output_dtype_unverified")
        if ratio not in (4, 128) or not 0 < rope <= head:
            raise ValueError("v4_compressor_parameters_invalid")
        result.update(hidden_size=int(x.shape[-1]), head_dim=head, rope_head_dim=rope)
    else:
        head = int(args["head_dim"])
        kv = args.get("kv")
        sink = args.get("attn_sink")
        topk = int(args.get("compress_topk", 0))
        window = int(args.get("window_size", 0))
        if ratio not in (0, 4, 128) or window != 128 or head != x.shape[-1]:
            raise ValueError("v4_sparse_parameters_missing_or_unsupported")
        if args.get("kv_quant_dtype") is not None or kv is None or _v4_dtype_name(kv.dtype) != "DT_BF16":
            raise ValueError("v4_quantized_kv_unverified")
        expected_kv = int(x.shape[1]) + (int(x.shape[1]) // ratio if ratio else 0)
        if requests["phase"] == "prefill" and tuple(kv.shape) != (x.shape[0], expected_kv, head):
            raise ValueError("v4_merged_kv_mismatch")
        if kv.shape[-1] != head:
            raise ValueError("v4_kv_head_dim_mismatch")
        if sink is None or tuple(sink.shape) != (x.shape[2],) or _v4_dtype_name(sink.dtype) != "DT_FLOAT":
            raise ValueError("v4_sink_signature_mismatch")
        indices = args.get("topk_indices")
        expected_topk = 0 if ratio == 0 else topk
        if (ratio != 0 and not 0 < topk <= max(1, max(seq) // ratio)) or (ratio == 0 and topk != 0):
            raise ValueError("v4_sparse_topk_invalid")
        if (
            indices is None
            or tuple(indices.shape)
            != (*x.shape[:2], (window if requests["phase"] == "decode" else min(x.shape[1], window)) + expected_topk)
            or _v4_dtype_name(indices.dtype) != "DT_INT32"
        ):
            raise ValueError("v4_merged_indices_mismatch")
        if not math.isclose(float(args["softmax_scale"]), head**-0.5, rel_tol=1e-6):
            raise ValueError("v4_softmax_scale_unverified")
        result.update(head_dim=head, num_heads=int(x.shape[2]), window_size=window, compress_topk=topk)
    return result


def _v4_project_workload(kernel: str, workload: Mapping[str, Any], template: Mapping[str, Any]) -> _V4PhysicalCase:
    """Construct physical metadata; retain the template's allocation capacities."""
    if kernel not in _V4_KERNELS:
        raise ValueError("v4_projection_contract_missing")
    workload = normalize_v4_workload(workload)
    inputs = _v4_shapes(template.get("Input Shapes", ""))
    count = 12 if kernel == "Compressor" else 14
    if len(inputs) != count:
        raise ValueError("v4_physical_slot_count_invalid")
    n, ratio, d = (int(workload[key]) for key in ("tokens", "compress_ratio", "head_dim"))
    if n <= 0 or d <= 0:
        raise ValueError("v4_dimensions_invalid")
    query, kv = workload["query_lengths"], workload["kv_lengths"]
    batch = len(query)
    ordinary_prefill = workload["phase"] == "prefill" and batch == 1
    if kernel == "Compressor":
        if ratio not in (4, 128) or len(inputs[3]) != 3 or len(inputs[8]) != 2:
            raise ValueError("v4_compressor_template_invalid")
        h, rope = int(workload["hidden_size"]), int(workload["rope_head_dim"])
        coff = 2 if ratio == 4 else 1
        width, compressed = coff * d, sum(q // ratio + 1 for q in query)
        blocks, block, _ = inputs[3]
        if min(h, rope, blocks, block, inputs[8][-1]) <= 0 or rope > d:
            raise ValueError("v4_compressor_template_invalid")
        state = (max(blocks, 1 + batch * math.ceil(coff * ratio / block)), block, 2 * width)
        physical = (
            (n, h),
            (width, h),
            (width, h),
            state,
            (ratio, width),
            (d,),
            (compressed, rope),
            (compressed, rope),
            (batch, max(inputs[8][-1], math.ceil(max(kv) / block))),
            (batch + 1,),
            (),
            (batch,),
        )
        dtypes = ("DT_BF16",) * 3 + ("DT_FLOAT",) * 2 + ("DT_BF16",) + ("DT_FLOAT",) * 2
        dtypes += ("DT_INT32", "DT_INT32", "DT_UNDEFINED", "DT_INT32")
        return _V4PhysicalCase(physical, ((compressed, d), state), dtypes, ("DT_BF16", "DT_FLOAT"))
    if ratio not in (0, 4, 128) or len(inputs[1]) != 4 or inputs[1][1:3] != (128, 1):
        raise ValueError("v4_sparse_template_invalid")
    heads, topk = int(workload["num_heads"]), int(workload["compress_topk"])
    if workload.get("window_size") != 128 or heads <= 0:
        raise ValueError("v4_sparse_parameters_missing_or_unsupported")
    if (
        (ratio == 4 and not 0 < topk <= max(1, max(kv) // ratio))
        or (ratio == 128 and (topk != max(1, max(kv) // ratio) or topk <= 0))
        or (ratio == 0 and topk != 0)
    ):
        raise ValueError("v4_sparse_topk_invalid")
    cache = (max(inputs[1][0], sum(math.ceil(s / 128) for s in kv)), 128, 1, d)
    ori_capacity = inputs[5][-1] if len(inputs[5]) == 2 else 0
    cmp_capacity = inputs[6][-1] if len(inputs[6]) == 2 else 0
    physical = (
        (n, heads, d),
        cache,
        cache if ratio else (),
        (),
        (n, 1, topk) if ratio == 4 else (),
        (batch, max(ori_capacity, math.ceil(max(kv) / 128))),
        (batch, max(cmp_capacity, math.ceil((max(kv) // ratio + 1) / 128))) if ratio else (),
        (batch + 1,),
        (batch + 1,),
        (batch + 1,) if ratio and not ordinary_prefill else (),
        (batch,) if not ordinary_prefill else (),
        (batch,),
        (heads,),
        (1024,),
    )
    dtypes = tuple(
        "DT_UNDEFINED" if not shape else "DT_BF16" if i < 3 else "DT_FLOAT" if i == 12 else "DT_INT32"
        for i, shape in enumerate(physical)
    )
    return _V4PhysicalCase(physical, ((n, heads, d), ()), dtypes, ("DT_BF16", "DT_FLOAT"))


def _v4_row_workload(kernel: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode only the physical signatures actually built by existing replay."""
    inputs = _v4_shapes(row.get("Input Shapes", ""))
    raw = row.get(_V4_RUNTIME_WORKLOAD)
    if isinstance(raw, str) and raw.strip():
        result = normalize_v4_workload(json.loads(raw))
    elif kernel == "Compressor" and len(inputs) == 12 and len(inputs[0]) == 2 and len(inputs[4]) == 2:
        outputs = _v4_shapes(row.get("Output Shapes", ""))
        if len(outputs) != 2 or len(outputs[0]) != 2 or len(inputs[6]) != 2:
            raise ValueError("v4_compressor_template_invalid")
        result = {
            "contract": _V4_CONTRACT,
            "tokens": inputs[0][0],
            "hidden_size": inputs[0][1],
            "head_dim": outputs[0][-1],
            "rope_head_dim": inputs[6][-1],
            "compress_ratio": inputs[4][0],
        }
    elif kernel == "SparseAttnSharedkv" and len(inputs) == 14 and len(inputs[0]) == 3:
        n, heads, d = inputs[0]
        ratio = 0 if not inputs[2] else 4 if inputs[4] else 128
        result = {
            "contract": _V4_CONTRACT,
            "tokens": n,
            "head_dim": d,
            "num_heads": heads,
            "compress_ratio": ratio,
            "window_size": 128,
            "compress_topk": inputs[4][-1] if ratio == 4 else n // ratio if ratio else 0,
        }
    else:
        raise ValueError("v4_physical_template_invalid")
    result = normalize_v4_workload(result)
    # Metadata-free historical measurements were single-request Prefill.
    # A multi-request/Decode signature must never inherit this assumption.
    if (not isinstance(raw, str) or not raw.strip()) and not (
        result["phase"] == "prefill" and len(result["query_lengths"]) == 1
    ):
        raise ValueError("v4_runtime_metadata_required")
    if not _v4_project_workload(kernel, result, row).matches(row):
        raise ValueError("v4_physical_signature_invalid")
    return result


V4_GROUPED_CONTRACT = "ascend_v4_grouped_v1"


def v4_grouped_row_workload(kernel: str, row: Mapping[str, Any]) -> dict[str, int]:
    """Read the packed W8A8 grouped-matmul operator ABIs.

    GroupedMatmul's weight is reported as logical EKN despite NZ storage;
    GroupedMatmulSwigluQuant reports the physical EN1K1K0N0 allocation.
    Empty optional slots are significant and must not be stripped.
    """
    inputs, outputs = (_v4_shapes(row.get("Input Shapes", "")), _v4_shapes(row.get("Output Shapes", "")))
    dtypes = tuple(_v4_dtype_name(x) for x in _v4_slots(row.get("Input Data Types", "")))
    output_dtypes = tuple(_v4_dtype_name(x) for x in _v4_slots(row.get("Output Data Types", "")))
    formats = tuple(_v4_slots(row.get("Input Formats", "")))
    output_formats = tuple(_v4_slots(row.get("Output Formats", "")))
    if not inputs or len(inputs[0]) != 2 or any(d <= 0 for shape in inputs for d in shape):
        raise ValueError("v4_grouped_shape_invalid")
    m, k = inputs[0]
    if kernel == "GroupedMatmulSwigluQuant":
        if len(inputs) != 5 or len(inputs[1]) != 5 or inputs[1][-2:] != (16, 32):
            raise ValueError("v4_grouped_up_weight_invalid")
        e, n1, k1, _, _ = inputs[1]
        n = n1 * 32
        if k != k1 * 16 or n % 2 or inputs[2:] != ((e, n), (m,), (e,)):
            raise ValueError("v4_grouped_up_metadata_invalid")
        if outputs != ((m, n // 2), (m,)) or output_dtypes != ("INT8", "DT_FLOAT"):
            raise ValueError("v4_grouped_up_output_invalid")
        expected_types = ("INT8", "INT8", "DT_FLOAT", "DT_FLOAT", "INT64")
    elif kernel == "GroupedMatmul":
        if len(inputs) != 9 or len(inputs[1]) != 3:
            raise ValueError("v4_grouped_down_weight_invalid")
        e, wk, n = inputs[1]
        if wk != k or inputs[2:] != ((), (e, n), (), (), (), (e,), (m,)):
            raise ValueError("v4_grouped_down_metadata_invalid")
        if outputs != ((m, n),) or output_dtypes != ("DT_BF16",):
            raise ValueError("v4_grouped_down_output_invalid")
        expected_types = ("INT8", "INT8", "DT_INT32", "DT_BF16", "DT_FLOAT", "FLOAT16", "FLOAT16", "INT64", "DT_FLOAT")
    else:
        raise ValueError("v4_grouped_kernel_unsupported")
    expected_formats = ("ND", "FRACTAL_NZ", *("ND",) * (len(inputs) - 2))
    if dtypes != expected_types or formats != expected_formats or output_formats != ("ND",) * len(outputs):
        raise ValueError("v4_grouped_dtype_or_format_invalid")
    workload = {"tokens": m, "experts": e, "input_width": k, "output_width": n}
    if kernel == "GroupedMatmulSwigluQuant":
        limit = row.get("Runtime swiglu_limit")
        if limit is not None and str(limit).strip().lower() not in ("", "nan"):
            limit = float(limit)
            if not math.isfinite(limit) or limit < 0:
                raise ValueError("v4_grouped_swiglu_limit_invalid")
            workload["swiglu_limit"] = limit
    return workload
