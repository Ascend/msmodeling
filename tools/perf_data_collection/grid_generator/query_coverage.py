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

"""Project captured kernel queries into replayable CSV coverage candidates.

The module deliberately has no model-name or serving-scene tables.  Exact
queries are the anchors; interpolation points are inferred only from axes that
actually vary across anchors with the same kernel/runtime schema.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from itertools import combinations, product, zip_longest
import json
import math
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator, Sequence

from tensor_cast.performance_model.profiling_database.query_demand import KernelQueryDemand

from .constraint_fallback import build_constraint_fallback_rows
from .utils import (
    INPUT_SHAPES_COLUMN,
    OUTPUT_SHAPES_COLUMN,
    build_generated_row,
    build_shape_text,
    parse_shape_text,
    profile_dedupe_key,
)


_MATMUL_KERNELS = {
    "MatMulV2",
    "MatMulV3",
    "MatMulCommon",
    "BatchMatMulV2",
    "QuantBatchMatmulV3",
    "TransposeBatchMatMul",
    "GroupedMatmul",
    "GroupedMatmulSwigluQuant",
}
_ROPE_KERNELS = {"InterleaveRope", "split_qkv_rmsnorm_rope_kernel", "_triton_rope"}
_ELEMENTWISE_BINARY_KERNELS = {"Add", "AddRmsNormBias", "Mul"}
# BatchMatMulV2 dispatches torch.bmm for batched rows; contraction dims below
# this floor dispatch to trivial/non-GEMM kernels that msprof records under a
# different operator name, so they can never be matched back to the CSV row.
_MIN_BATCH_MATMUL_K = 16
# MatMul kernels whose replay adapter applies transpose=True to the second
# input, so the CSV must store the weight as (N, K) / (E, N, K).
_MATMUL_TRANSPOSE_WEIGHT_KERNELS = {"MatMulV2", "MatMulV3", "MatMulCommon", "QuantBatchMatmulV3"}
_FLATTEN_KERNELS = {
    "AscendQuantV2",
    "DynamicQuant",
    "RmsNorm",
    "AddRmsNormBias",
    "DispatchFFNCombine",
}
_RUNTIME_ATTENTION_KERNELS = {"FusedInferAttentionScore", "SparseFlashAttention", "LightningIndexer"}
_SCHEMA_RUNTIME_FIELDS = {
    "compute_subcategory",
    "phase",
    "sparse_mode",
    "num_heads",
    "num_kv_heads",
    "input_layout",
    "topk",
    "block_size",
    "cache_layout",
    "kv_cache_mode",
    "sparse_block_size",
    "sparse_indices_pattern",
    "weight_format",
    "quant_mode",
    "cache_mode",
    "enable_inner_out",
    "weight_quantized",
    "ep_size",
    "scale_mode",
    "auxiliary_modes",
}
_DTYPE_ALIASES = {
    "BF16": "DT_BF16",
    "FLOAT16": "DT_BF16",
    "DT_FLOAT16": "DT_BF16",
    "DT_INT8": "INT8",
    "DT_INT32": "INT32",
    "DT_INT64": "INT64",
    "DT_FLOAT": "FLOAT",
    "DT_BOOL": "BOOL",
}


def _split_slots(value: str) -> list[str]:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return []
    return [part.strip().strip('"') for part in raw.split(";")]


def _dtype_key(value: str, *, preserve_fp16: bool = False) -> str:
    normalized = str(value or "").strip().upper()
    if preserve_fp16 and normalized in {"FLOAT16", "FP16", "DT_FLOAT16"}:
        return "DT_FLOAT16"
    return _DTYPE_ALIASES.get(normalized, normalized)


def _dtype_compatible(
    left: str,
    right: str,
    kernel_type: str,
    *,
    preserve_fp16: bool = False,
) -> bool:
    left_key = _dtype_key(left, preserve_fp16=preserve_fp16)
    right_key = _dtype_key(right, preserve_fp16=preserve_fp16)
    if left_key == right_key:
        return True
    return kernel_type in _ROPE_KERNELS | _MATMUL_KERNELS and {left_key, right_key} <= {
        "DT_BF16",
        "FLOAT",
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_format_value(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _restore_fractal(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) < 4:
        return shape
    *prefix, height, width, block_h, block_w = shape
    return (*prefix, height * block_w, width * block_h)


def _to_fractal(shape: tuple[int, ...], template_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(shape) < 2 or len(template_shape) < 4:
        return None
    block_h, block_w = template_shape[-2:]
    rows, columns = shape[-2:]
    if rows % block_w or columns % block_h:
        return None
    return (*shape[:-2], rows // block_w, columns // block_h, block_h, block_w)


def _fit_rank(shape: tuple[int, ...], template_shape: tuple[int, ...], kernel_type: str) -> tuple[int, ...] | None:
    if len(shape) == len(template_shape):
        return shape
    if not shape and template_shape:
        # CANN profiling serializes scalar tensor outputs as a one-element
        # physical buffer even when TensorCast exposes rank zero.
        return (1,)
    if len(shape) == len(template_shape) + 1 and shape[0] == 1:
        return shape[1:]
    if kernel_type in _FLATTEN_KERNELS and len(shape) == 3 and len(template_shape) == 2:
        return (shape[0] * shape[1], shape[2])
    return None


def _normalize_query_inputs(
    kernel_type: str,
    shapes: Sequence[tuple[int, ...]],
    dtypes: Sequence[str],
) -> tuple[list[tuple[int, ...]], list[str]]:
    pairs = list(zip(shapes, [*dtypes, *([""] * max(0, len(shapes) - len(dtypes)))]))
    if kernel_type == "MoeTokenPermute" and len(pairs) >= 2:
        activations, indices = pairs[:2]
        activation_shape = activations[0]
        index_shape = indices[0]
        if len(activation_shape) >= 2 and len(index_shape) >= 2:
            return [
                (math.prod(activation_shape[:-1]), activation_shape[-1]),
                (math.prod(index_shape[:-1]), index_shape[-1]),
            ], [activations[1], "INT32"]
    if kernel_type == "MoeTokenUnpermute" and pairs:
        activations = pairs[0]
        if len(activations[0]) >= 2:
            routed = math.prod(activations[0][:-1])
            return [(routed, activations[0][-1]), (routed,)], [activations[1], "INT32"]
    if kernel_type == "SwiGlu" and len(pairs) == 2:
        first, second = pairs
        if first[0][:-1] == second[0][:-1] and first[1] == second[1]:
            return [first[0][:-1] + (first[0][-1] + second[0][-1],)], [first[1]]
    if kernel_type in _ROPE_KERNELS and len(pairs) >= 2:
        query, key = pairs[0], pairs[1]

        def transpose(value: tuple[int, ...]) -> tuple[int, ...]:
            return (value[0], value[2], value[1], value[3]) if len(value) == 4 else value

        normalized = [(transpose(key[0]), key[1]), (transpose(query[0]), query[1])]
        for shape, dtype in pairs[2:4]:
            if len(shape) == 3:
                shape = (shape[0], shape[1], 1, shape[2])
            normalized.append((shape, dtype))
        return [item[0] for item in normalized], [item[1] for item in normalized]
    if kernel_type == "ReshapeAndCacheNdKernel" and len(pairs) == 4:
        key, value, cache, slots = pairs
        if len(key[0]) == 2 and len(value[0]) == 2 and len(cache[0]) >= 2 and cache[0][0] == 2:
            cache_shape = cache[0][1:]
            num_heads = cache_shape[-2] if len(cache_shape) >= 4 else 1
            head_dim = cache_shape[-1]

            def restore_kv_heads(shape: tuple[int, ...]) -> tuple[int, ...]:
                if shape[-1] == num_heads * head_dim:
                    return (shape[0], num_heads, head_dim)
                return (shape[0], 1, shape[-1])

            normalized = [
                (restore_kv_heads(key[0]), key[1]),
                (restore_kv_heads(value[0]), value[1]),
                (cache_shape, cache[1]),
                (cache_shape, cache[1]),
                (slots[0], "INT32"),
            ]
            return [item[0] for item in normalized], [item[1] for item in normalized]
    return list(shapes), list(dtypes)


@dataclass(frozen=True)
class ProjectedCandidate:
    kernel_type: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    input_formats: tuple[str, ...]
    output_formats: tuple[str, ...]
    extra_values: tuple[tuple[str, str], ...]
    schema_key: str
    exact: bool = True

    def to_row(self, headers: list[str], template: dict[str, str]) -> dict[str, str]:
        extras = dict(self.extra_values)
        extras.update(
            {
                "Input Data Types": ";".join(self.input_dtypes),
                "Input Formats": ";".join(self.input_formats),
                "Output Data Types": ";".join(self.output_dtypes),
                "Output Formats": ";".join(self.output_formats),
            }
        )
        row = build_generated_row(
            headers,
            template,
            list(self.input_shapes),
            list(self.output_shapes),
            extra_values={key: value for key, value in extras.items() if key in headers},
        )
        if "Runtime source_profile" in headers:
            row["Runtime source_profile"] = ""
        return row


def _template_schema(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row.get("Input Data Types", ""),
        row.get("Input Formats", ""),
        row.get("Output Data Types", ""),
        row.get("Output Formats", ""),
        tuple(len(shape) for shape in parse_shape_text(row.get(INPUT_SHAPES_COLUMN, ""))),
        tuple(len(shape) for shape in parse_shape_text(row.get(OUTPUT_SHAPES_COLUMN, ""))),
    )


def _unique_templates(source_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    templates: list[dict[str, str]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in source_rows:
        key = _template_schema(row)
        if key in seen:
            continue
        seen.add(key)
        templates.append(row)
    return templates


def _template_score(demand: KernelQueryDemand, template: dict[str, str]) -> int | None:
    shapes = parse_shape_text(template.get(INPUT_SHAPES_COLUMN, ""))
    dtypes = _split_slots(template.get("Input Data Types", ""))
    present = [index for index, shape in enumerate(shapes) if shape]
    demand_shapes, demand_dtypes = _normalize_query_inputs(
        demand.kernel_type,
        [shape for shape in demand.input_shapes if shape],
        [dtype for shape, dtype in zip(demand.input_shapes, demand.input_dtypes) if shape],
    )
    if demand.query_mode == "attention_special":
        if demand.kernel_type != "FusedInferAttentionScore":
            return None
        completeness = template.get("Runtime metadata_completeness", "").strip().lower()
        return 100 if completeness in {"", "legacy"} else 0
    if demand.input_shapes and abs(len(present) - len(demand_shapes)) > 2 and demand.kernel_type != "Index":
        return None
    score = abs(len(present) - len(demand_shapes)) * 10
    preserve_fp16 = demand.query_mode == "compute_scale"
    for position, demand_dtype in enumerate(demand_dtypes):
        if position >= len(present):
            score += 10
            continue
        template_dtype = dtypes[present[position]] if present[position] < len(dtypes) else ""
        if not _dtype_compatible(
            demand_dtype,
            template_dtype,
            demand.kernel_type,
            preserve_fp16=preserve_fp16,
        ):
            if preserve_fp16 and {
                _dtype_key(demand_dtype, preserve_fp16=True),
                _dtype_key(template_dtype, preserve_fp16=True),
            } == {"DT_FLOAT16", "DT_BF16"}:
                # Dtype is not a performance match, but the row can still be
                # used as a structural schema template. The generated row is
                # overwritten with the captured physical FP16 dtype below.
                score += 100
                continue
            return None
        if _dtype_key(demand_dtype, preserve_fp16=preserve_fp16) != _dtype_key(
            template_dtype,
            preserve_fp16=preserve_fp16,
        ):
            score += 1
    for demand_shape, index in zip(demand_shapes, present):
        template_shape = shapes[index]
        logical_template = (
            _restore_fractal(template_shape)
            if index < len(_split_slots(template.get("Input Formats", "")))
            and _split_slots(template.get("Input Formats", ""))[index] == "FRACTAL_NZ"
            else template_shape
        )
        score += abs(len(demand_shape) - len(logical_template))
    completeness = template.get("Runtime metadata_completeness", "").strip().lower()
    if demand.kernel_type in _RUNTIME_ATTENTION_KERNELS and completeness in {"", "legacy"}:
        score += 100
    return score


def _select_template(
    demand: KernelQueryDemand,
    templates: Sequence[dict[str, str]],
) -> dict[str, str] | None:
    scored = [
        (score, index, template)
        for index, template in enumerate(templates)
        if (score := _template_score(demand, template)) is not None
    ]
    return min(scored, default=(0, 0, None))[2]


def _runtime_extras(
    demand: KernelQueryDemand,
    headers: Sequence[str],
    input_shapes: Sequence[tuple[int, ...]],
) -> dict[str, str]:
    extras: dict[str, str] = {}
    for key, value in demand.attributes.items():
        header = f"Runtime {key}"
        if header in headers:
            extras[header] = _format_value(value)
    # The demand attribute key is "num_kv_heads" but the CSV column is
    # "Runtime num_key_value_heads" (see FIA_RUNTIME_COLUMNS). The generic loop
    # above writes "Runtime num_kv_heads", which is not a CSV column and is
    # silently dropped, so the num_key_value_heads column would otherwise be
    # inherited from a potentially stale template row. Map the attribute to
    # the real column name explicitly so every attention kernel (SFA, LI, FIA)
    # carries the traced value.
    kv_heads_value = demand.attributes.get("num_kv_heads")
    if "Runtime num_key_value_heads" in headers and kv_heads_value is not None:
        extras["Runtime num_key_value_heads"] = _format_value(kv_heads_value)
    if "EP Size" in headers and demand.attributes.get("ep_size") is not None:
        extras["EP Size"] = _format_value(demand.attributes["ep_size"])
    for values_key, shape_header in (
        ("actual_seq_lengths_values", "Runtime actual_seq_lengths_shape"),
        ("actual_seq_lengths_kv_values", "Runtime actual_seq_lengths_kv_shape"),
    ):
        values = demand.attributes.get(values_key)
        if shape_header in headers and isinstance(values, (list, tuple)):
            extras[shape_header] = str(len(values))
    if "Runtime metadata_completeness" in headers:
        extras["Runtime metadata_completeness"] = "query_trace"
    if "Runtime operator_input_shapes_raw" in headers:
        extras["Runtime operator_input_shapes_raw"] = build_shape_text(list(input_shapes))
    return extras


def _stable_case_id(kernel_type: str, demand: KernelQueryDemand) -> str:
    prefix = "li" if kernel_type == "LightningIndexer" else "sfa" if kernel_type == "SparseFlashAttention" else "query"
    digest = hashlib.sha256(demand.signature.encode("ascii")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _attention_runtime_shapes(
    demand: KernelQueryDemand,
    template: dict[str, str],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]] | None:
    attrs = demand.attributes
    raw_q_shape = attrs.get("q_shape_3d")
    if not isinstance(raw_q_shape, (list, tuple)) or len(raw_q_shape) != 3:
        return None
    tokens, heads, head_dim = (int(value) for value in raw_q_shape)
    kv_values = attrs.get("actual_seq_lengths_kv_values")
    batch = int(attrs.get("batch_size") or (len(kv_values) if isinstance(kv_values, (list, tuple)) else 1))
    block_size = max(1, int(attrs.get("block_size") or 128))
    avg_seq_len = max(1, int(attrs.get("avg_seq_len") or 1))
    if not isinstance(kv_values, (list, tuple)):
        kv_values = [avg_seq_len] * batch
    valid_blocks = attrs.get("block_table_valid_blocks")
    if not isinstance(valid_blocks, (list, tuple)):
        valid_blocks = [math.ceil(max(0, int(value)) / block_size) for value in kv_values]
    total_blocks = max(1, sum(max(0, int(value)) for value in valid_blocks))
    block_capacity = max(1, max((int(value) for value in valid_blocks), default=1))
    topk = max(1, int(attrs.get("topk") or 1))

    if demand.kernel_type == "SparseFlashAttention":
        rope_dim = 64
        template_shapes = parse_shape_text(template.get(INPUT_SHAPES_COLUMN, ""))
        if len(template_shapes) > 7 and template_shapes[7]:
            rope_dim = template_shapes[7][-1]
        cache = (total_blocks, block_size, 1, head_dim)
        rope_cache = (total_blocks, block_size, 1, rope_dim)
        return (
            [
                (tokens, heads, head_dim),
                cache,
                cache,
                (tokens, 1, topk),
                (batch, block_capacity),
                (batch,),
                (batch,),
                (tokens, heads, rope_dim),
                rope_cache,
            ],
            [(tokens, heads, head_dim)],
        )
    if demand.kernel_type == "LightningIndexer":
        cache = (total_blocks, block_size, 1, head_dim)
        return (
            [
                (tokens, heads, head_dim),
                cache,
                (tokens, heads),
                (batch,),
                (batch,),
                (batch, block_capacity),
            ],
            [(tokens, 1, topk), (tokens, 1, topk)],
        )
    if demand.kernel_type == "FusedInferAttentionScore":
        sparse_mode = int(attrs.get("sparse_mode") or 0)
        num_kv_heads = max(1, int(attrs.get("num_kv_heads") or 1))
        raw_inputs = [()] * 31
        raw_inputs[0] = (tokens, heads, head_dim)
        if sparse_mode == 0:
            raw_inputs[1] = (total_blocks, num_kv_heads, block_size, head_dim)
            raw_inputs[2] = raw_inputs[1]
            raw_inputs[5] = (batch,)
            raw_inputs[6] = (batch,)
            raw_inputs[14] = (batch, block_capacity)
        else:
            raw_inputs[1] = (tokens, num_kv_heads, head_dim)
            raw_inputs[2] = raw_inputs[1]
            raw_inputs[4] = (2048, 2048)
            raw_inputs[5] = (batch,)
            raw_inputs[6] = (batch,)
        return raw_inputs, [(tokens, heads, head_dim)]
    return None


def _attention_special_shapes(demand: KernelQueryDemand) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]] | None:
    if len(demand.input_shapes) < 7:
        return None
    tc = list(demand.input_shapes) + [()] * (8 - len(demand.input_shapes))
    raw = [()] * 31
    raw[0], raw[1], raw[2] = tc[0], tc[1], tc[2]
    raw[4] = tc[3]
    raw[14] = tc[4]
    raw[5] = tc[7] or tc[6]
    raw[6] = tc[6]
    if raw[14] and len(raw[1]) == 4:
        raw[1] = (raw[1][0], raw[1][2], raw[1][1], raw[1][3])
        raw[2] = (raw[2][0], raw[2][2], raw[2][1], raw[2][3])
    outputs = list(demand.output_shapes) or ([raw[0]] if raw[0] else [])
    return raw, outputs


def _physical_input_shape(
    kernel_type: str,
    index: int,
    shape: tuple[int, ...],
    template_shapes: Sequence[tuple[int, ...]],
    template_formats: Sequence[str],
    output_shapes: Sequence[tuple[int, ...]],
) -> tuple[int, ...] | None:
    template_shape = template_shapes[index]
    fmt = template_formats[index] if index < len(template_formats) else "ND"
    logical_template = _restore_fractal(template_shape) if fmt == "FRACTAL_NZ" else template_shape
    physical_logical = shape
    if kernel_type in _MATMUL_KERNELS and index >= 1 and len(shape) in {2, 3}:
        # The performance model reports the weight / second matrix as (K, N),
        # but the CSV and replay adapters use (N, K) with transpose=True at
        # replay time.  Swap the last two dims when the demand shape's last
        # dim matches the output N (confirming it is (K, N), not already
        # (N, K)).  Square matrices (K == N) are a no-op so they are skipped.
        should_transpose = False
        if output_shapes and output_shapes[0] and len(shape) >= 2:
            out_n = output_shapes[0][-1]
            if shape[-1] == out_n and shape[-2] != out_n:
                should_transpose = True
        if should_transpose:
            physical_logical = (*shape[:-2], shape[-1], shape[-2])
    fitted = _fit_rank(physical_logical, logical_template, kernel_type)
    if fitted is None:
        return None
    if fmt == "FRACTAL_NZ":
        if len(template_shape) < 4:
            return fitted
        return _to_fractal(fitted, template_shape)
    if kernel_type == "Index" and len(fitted) == 2 and len(template_shape) == 2:
        if fitted[0] == 1 and template_shape[1] == 1:
            return (fitted[1], 1)
    return fitted


def _project_generic(
    demand: KernelQueryDemand,
    template: dict[str, str],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[str], list[str]] | None:
    template_inputs = parse_shape_text(template.get(INPUT_SHAPES_COLUMN, ""))
    template_outputs = parse_shape_text(template.get(OUTPUT_SHAPES_COLUMN, ""))
    input_formats = _split_slots(template.get("Input Formats", ""))
    demand_inputs, demand_dtypes = _normalize_query_inputs(
        demand.kernel_type,
        [shape for shape in demand.input_shapes if shape],
        [dtype for shape, dtype in zip(demand.input_shapes, demand.input_dtypes) if shape],
    )
    if demand.query_mode == "compute_scale" and demand.kernel_type != "AscendQuantV2":
        if len(demand_inputs) != 1 or len(demand.output_shapes) < 2:
            return None
        outputs = list(demand.output_shapes)
        if len(outputs) != len(template_outputs):
            return None
        physical_outputs = []
        for index, (shape, template_shape) in enumerate(zip(outputs, template_outputs)):
            if not shape:
                if index >= 1 and demand_inputs and demand_inputs[0] and len(demand_inputs[0]) >= 1:
                    # DynamicQuant exposes the per-token scale as a real second
                    # output whose physical shape is the input without its last
                    # dim (1,2,256 -> 1,2; 1,1,1254 -> 1,1).  The trace emits an
                    # empty scalar marker; replay/profiling records the actual
                    # leading input dims, so project them here or the CSV row
                    # can never be matched back to the profile signature.
                    physical_outputs.append(demand_inputs[0][:-1])
                elif template_shape:
                    # Some replay APIs expose a scalar while profiling serializes
                    # its physical storage as a one-element buffer.
                    physical_outputs.append((1,))
                else:
                    physical_outputs.append(())
            else:
                physical_outputs.append(shape)
        return demand_inputs, physical_outputs, demand_dtypes, list(demand.output_dtypes)
    if demand.query_mode == "elementwise":
        physical_inputs = list(template_inputs)
        while len(physical_inputs) < len(demand_inputs):
            physical_inputs.append(())
        for index, shape in enumerate(demand_inputs):
            physical_inputs[index] = shape
        outputs = list(demand.output_shapes) or ([demand_inputs[0]] if demand_inputs else [])
        return physical_inputs, outputs, demand_dtypes, list(demand.output_dtypes)
    present = [index for index, shape in enumerate(template_inputs) if shape]
    if len(demand_inputs) < len(present):
        # Internal trailing parameters not visible to TensorCast retain their
        # template shapes; visible leading inputs are still query-driven.
        missing = len(present) - len(demand_inputs)
        demand_inputs.extend(template_inputs[index] for index in present[-missing:])
        demand_dtypes.extend([""] * missing)
    if len(demand_inputs) != len(present):
        return None
    # Output tuples come from real tensors, so ``()`` means a scalar rather
    # than an absent optional slot. Input raw-slot traces use a separate path.
    outputs = list(demand.output_shapes)
    if demand.kernel_type == "RmsNorm" and demand_inputs:
        outputs = [demand_inputs[0], (*demand_inputs[0][:-1], 1)]
    elif demand.kernel_type == "AddRmsNormBias" and demand_inputs:
        normalized = demand_inputs[0]
        outputs = [normalized, (*normalized[:-1], 1), normalized]
    elif demand.kernel_type == "DispatchFFNCombine" and len(demand_inputs) >= 2:
        # demand_inputs[1] is the projected weight, so output[1] records
        # (num_experts,) exactly as the replay contract and existing rows do.
        outputs = [demand_inputs[0], (demand_inputs[1][0],)]
    elif demand.kernel_type == "MoeTokenPermute" and outputs:
        routed_shape = outputs[0]
        outputs = [routed_shape, (routed_shape[0],)]
    elif demand.kernel_type == "MoeTokenUnpermute" and outputs:
        raw_output = outputs[0]
        if len(raw_output) >= 3:
            outputs = [(math.prod(raw_output[:-2]), raw_output[-1])]
    elif demand.kernel_type == "ScatterNdUpdate" and len(demand_inputs) >= 3:
        demand_inputs[0] = template_inputs[present[0]]
        outputs = [demand_inputs[0]]
    elif demand.kernel_type == "ReshapeAndCacheNdKernel" and len(demand_inputs) >= 4:
        outputs = [demand_inputs[2], demand_inputs[3]]
    elif demand.kernel_type == "AscendQuantV2" and outputs:
        # AscendQuant consumes scale/offset as physical inputs and exposes only
        # the quantized tensor, unlike DynamicQuant's tuple result.
        outputs = outputs[:1]
    elif not outputs:
        outputs = list(template_outputs)
    elif len(outputs) != len([shape for shape in template_outputs if shape]):
        return None
    physical_inputs = list(template_inputs)
    for demand_shape, index in zip(demand_inputs, present):
        physical = _physical_input_shape(
            demand.kernel_type,
            index,
            demand_shape,
            template_inputs,
            input_formats,
            outputs,
        )
        if physical is None:
            return None
        physical_inputs[index] = physical
    physical_outputs: list[tuple[int, ...]] = []
    for shape, template_shape in zip(outputs, [item for item in template_outputs if item]):
        fitted = _fit_rank(shape, template_shape, demand.kernel_type)
        if fitted is None:
            return None
        physical_outputs.append(fitted)
    return physical_inputs, physical_outputs, demand_dtypes, list(demand.output_dtypes)


def project_query_demand(
    demand: KernelQueryDemand,
    headers: list[str],
    templates: Sequence[dict[str, str]],
) -> tuple[ProjectedCandidate, dict[str, str]] | None:
    """Project one backend demand into the physical schema used by replay."""
    template = _select_template(demand, templates)
    if template is None:
        return None
    attention_shapes = None
    if demand.query_mode == "attention_special" and demand.kernel_type == "FusedInferAttentionScore":
        attention_shapes = _attention_special_shapes(demand)
    elif demand.kernel_type in _RUNTIME_ATTENTION_KERNELS and not any(demand.input_shapes):
        attention_shapes = _attention_runtime_shapes(demand, template)

    template_input_dtypes = _split_slots(template.get("Input Data Types", ""))
    template_output_dtypes = _split_slots(template.get("Output Data Types", ""))
    template_input_formats = _split_slots(template.get("Input Formats", ""))
    template_output_formats = _split_slots(template.get("Output Formats", ""))
    if attention_shapes is not None:
        input_shapes, output_shapes = attention_shapes
        demand_input_dtypes: list[str] = []
        demand_output_dtypes: list[str] = []
    else:
        generic = _project_generic(demand, template)
        if generic is None:
            return None
        input_shapes, output_shapes, demand_input_dtypes, demand_output_dtypes = generic

    if demand.query_mode == "compute_scale":
        input_format = demand.attributes.get("input_format")
        if isinstance(input_format, str) and input_format:
            if template_input_formats:
                template_input_formats[0] = input_format
        output_formats = demand.attributes.get("output_formats")
        if isinstance(output_formats, (list, tuple)) and len(output_formats) == len(output_shapes):
            template_output_formats = [str(value) for value in output_formats]

    while len(template_input_dtypes) < len(input_shapes):
        template_input_dtypes.append("DT_UNDEFINED" if not input_shapes[len(template_input_dtypes)] else "DT_BF16")
    while len(template_input_formats) < len(input_shapes):
        template_input_formats.append("NULL" if not input_shapes[len(template_input_formats)] else "ND")
    while len(template_output_dtypes) < len(output_shapes):
        template_output_dtypes.append("DT_BF16")
    while len(template_output_formats) < len(output_shapes):
        template_output_formats.append("ND")

    present_inputs = [index for index, shape in enumerate(input_shapes) if shape]
    for index, dtype in zip(present_inputs, demand_input_dtypes):
        if dtype:
            template_input_dtypes[index] = dtype

    # AscendQuantV2: aclnnAscendQuantV3 rejects scale/offset dtypes that differ
    # from the input dtype ("dtype of input x is not compatible with scale").
    # Demands usually record only the input dtype, so inherit it for the
    # per-channel scale/offset slots instead of leaving the template default,
    # which would otherwise mix e.g. FP16 input with BF16 scale.
    if demand.kernel_type == "AscendQuantV2" and template_input_dtypes:
        input_dtype = template_input_dtypes[0] or "DT_BF16"
        for index in range(1, len(template_input_dtypes)):
            template_input_dtypes[index] = input_dtype
    for index, dtype in enumerate(demand_output_dtypes):
        if dtype and index < len(template_output_dtypes):
            template_output_dtypes[index] = dtype

    extras = _runtime_extras(demand, headers, input_shapes)
    if "Runtime case_id" in headers:
        extras["Runtime case_id"] = _stable_case_id(demand.kernel_type, demand)
    # The block_table slot in Input Shapes is the authoritative projected shape
    # (batch, block_capacity). A stale Runtime block_table_shape value carried
    # from a backend trace (e.g. (batch, valid_blocks)) can conflict with the
    # projected slot and make the replay adapter drop the row. Override
    # unconditionally from input_shapes so the runtime metadata always matches
    # the emitted shapes, whether the column was empty or held a stale value.
    if "Runtime block_table_shape" in headers:
        block_table_index = 14 if demand.kernel_type == "FusedInferAttentionScore" else 4 if demand.kernel_type == "SparseFlashAttention" else 5
        if block_table_index < len(input_shapes) and input_shapes[block_table_index]:
            extras["Runtime block_table_shape"] = ",".join(str(dim) for dim in input_shapes[block_table_index])
    # SparseFlashAttention and LightningIndexer paged KV caches are emitted by
    # _attention_runtime_shapes with the num_kv_heads dimension hard-coded to 1
    # (see cache = (total_blocks, block_size, 1, head_dim)). The traced
    # num_kv_heads attribute is now mapped to Runtime num_key_value_heads in
    # _runtime_extras, but older traces or missing attributes can still leave
    # the column empty or 0, which makes the replay adapter raise
    # "Runtime num_key_value_heads=0 conflicts with key shape" and drop the row.
    # Derive the value from the projected cache shape as a fallback so the
    # runtime metadata always matches the shapes.
    if demand.kernel_type in ("SparseFlashAttention", "LightningIndexer") and "Runtime num_key_value_heads" in headers:
        existing_kv = (extras.get("Runtime num_key_value_heads") or "").strip()
        if existing_kv in ("", "0"):
            cache_index = 1
            if cache_index < len(input_shapes) and len(input_shapes[cache_index]) >= 3:
                derived_kv = input_shapes[cache_index][2]
                if derived_kv and derived_kv > 0:
                    extras["Runtime num_key_value_heads"] = str(derived_kv)

    runtime_schema = {key: demand.attributes.get(key) for key in sorted(_SCHEMA_RUNTIME_FIELDS) if key in demand.attributes}
    schema_payload = {
        "kernel_type": demand.kernel_type,
        "query_mode": demand.query_mode,
        "input_ranks": [len(shape) for shape in input_shapes],
        "output_ranks": [len(shape) for shape in output_shapes],
        "input_dtypes": template_input_dtypes,
        "input_formats": template_input_formats,
        "output_dtypes": template_output_dtypes,
        "output_formats": template_output_formats,
        "runtime": runtime_schema,
    }
    schema_key = json.dumps(schema_payload, sort_keys=True, default=str, separators=(",", ":"))
    return (
        ProjectedCandidate(
            kernel_type=demand.kernel_type,
            input_shapes=tuple(input_shapes),
            output_shapes=tuple(output_shapes),
            input_dtypes=tuple(template_input_dtypes[: len(input_shapes)]),
            output_dtypes=tuple(template_output_dtypes[: len(output_shapes)]),
            input_formats=tuple(template_input_formats[: len(input_shapes)]),
            output_formats=tuple(template_output_formats[: len(output_shapes)]),
            extra_values=tuple(sorted(extras.items())),
            schema_key=schema_key,
        ),
        template,
    )


def _flatten_positions(candidate: ProjectedCandidate) -> tuple[list[int], list[tuple[str, int, int]]]:
    values: list[int] = []
    positions: list[tuple[str, int, int]] = []
    for kind, shapes in (("input", candidate.input_shapes), ("output", candidate.output_shapes)):
        for slot, shape in enumerate(shapes):
            for axis, value in enumerate(shape):
                values.append(value)
                positions.append((kind, slot, axis))
    return values, positions


def _replace_positions(
    candidate: ProjectedCandidate,
    updates: dict[tuple[str, int, int], int],
) -> ProjectedCandidate:
    inputs = [list(shape) for shape in candidate.input_shapes]
    outputs = [list(shape) for shape in candidate.output_shapes]
    for (kind, slot, axis), value in updates.items():
        target = inputs if kind == "input" else outputs
        target[slot][axis] = int(value)
    return replace(
        candidate,
        input_shapes=tuple(tuple(shape) for shape in inputs),
        output_shapes=tuple(tuple(shape) for shape in outputs),
        exact=False,
    )


def _proportional_groups(candidates: Sequence[ProjectedCandidate]) -> list[tuple[list[tuple[str, int, int]], list[int]]]:
    if len(candidates) < 2:
        return []
    flattened = [_flatten_positions(candidate) for candidate in candidates]
    if any(item[1] != flattened[0][1] for item in flattened[1:]):
        return []
    series = list(zip(*(item[0] for item in flattened)))
    positions = flattened[0][1]
    dynamic = [index for index, values in enumerate(series) if len(set(values)) > 1 and all(value > 0 for value in values)]
    groups: list[list[int]] = []
    for index in dynamic:
        for group in groups:
            base = group[0]
            if all(series[index][row] * series[base][0] == series[base][row] * series[index][0] for row in range(len(series[index]))):
                group.append(index)
                break
        else:
            groups.append([index])

    result = []
    for group in groups:
        base = group[0]
        observed = sorted(set(int(value) for value in series[base]))
        result.append(([positions[index] for index in group], observed))
    return result


def _axis_domain(observed: Sequence[int]) -> list[int]:
    values = set(int(value) for value in observed)
    ordered = sorted(values)
    for lower, upper in zip(ordered, ordered[1:]):
        if upper - lower <= 1:
            continue
        values.add((lower + upper) // 2)
        geometric = int(math.sqrt(lower * upper))
        if lower < geometric < upper:
            values.add(geometric)
    if len(ordered) >= 2:
        lower_step = max(1, ordered[1] - ordered[0])
        upper_step = max(1, ordered[-1] - ordered[-2])
        values.add(max(1, ordered[0] - lower_step))
        values.add(ordered[-1] + upper_step)
    return sorted(values)


def _group_updates(
    base: ProjectedCandidate,
    positions: Sequence[tuple[str, int, int]],
    value: int,
) -> dict[tuple[str, int, int], int] | None:
    flat_values, flat_positions = _flatten_positions(base)
    by_position = dict(zip(flat_positions, flat_values))
    base_value = by_position[positions[0]]
    updates: dict[tuple[str, int, int], int] = {}
    for position in positions:
        ratio = Fraction(by_position[position], base_value)
        projected = Fraction(value) * ratio
        if projected.denominator != 1 or projected.numerator <= 0:
            return None
        updates[position] = projected.numerator
    return updates


def _permuted_product(
    domains: Sequence[Sequence[int]],
    *,
    seed: int,
    schema_key: str,
) -> Iterator[tuple[int, ...]]:
    """Visit a Cartesian product deterministically without materializing it.

    A hash-derived affine permutation over the mixed-radix index space keeps
    ``--seed`` meaningful while bounding planner memory by the number of axes.
    """
    if not domains or any(not domain for domain in domains):
        return
    total = math.prod(len(domain) for domain in domains)
    digest = hashlib.sha256(f"{seed}:{schema_key}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % total
    step = int.from_bytes(digest[8:16], "big") % total or 1
    while math.gcd(step, total) != 1:
        step = (step + 1) % total or 1

    for offset in range(total):
        flat_index = (start + offset * step) % total
        values = []
        for domain in reversed(domains):
            flat_index, index = divmod(flat_index, len(domain))
            values.append(domain[index])
        yield tuple(reversed(values))


# Structural axis positions that coverage interpolation must never vary for a
# kernel. DispatchFFNCombine's expert dimension (weight1/weight2 dim0, the
# per-expert scale slots, and the expert_token_nums output) is tied to the EP
# configuration recorded in the row's EP Size; perturbing it independently
# produces rows the DFC kernel rejects at tiling time (EZ9999).
_COVERAGE_STRUCTURAL_POSITIONS: dict[str, frozenset[tuple[str, int, int]]] = {
    "DispatchFFNCombine": frozenset(
        {("input", 1, 0), ("input", 2, 0), ("input", 4, 0), ("input", 5, 0), ("output", 1, 0)}
    ),
}


def _coverage_candidates(exact: Sequence[ProjectedCandidate], seed: int) -> Iterator[ProjectedCandidate]:
    by_schema: dict[str, list[ProjectedCandidate]] = {}
    for candidate in exact:
        by_schema.setdefault(candidate.schema_key, []).append(candidate)
    for schema_key in sorted(by_schema):
        schema_candidates = by_schema[schema_key]
        # All operators now go through coverage interpolation. Attention
        # kernels previously skipped this (exact-only), but the unified
        # three-stage pipeline lets them interpolate from real anchors too.
        structural = _COVERAGE_STRUCTURAL_POSITIONS.get(schema_candidates[0].kernel_type, frozenset())
        groups = _proportional_groups(schema_candidates)
        if structural:
            groups = [group for group in groups if not any(position in structural for position in group[0])]
            if not groups:
                continue
        base = schema_candidates[0]
        domains = [_axis_domain(observed) for _positions, observed in groups]
        for group_index, (positions, _observed) in enumerate(groups):
            for value in domains[group_index]:
                updates = _group_updates(base, positions, value)
                if updates is not None:
                    yield _replace_positions(base, updates)
        for left, right in combinations(range(len(groups)), 2):
            for left_value, right_value in product(domains[left], domains[right]):
                updates = _group_updates(base, groups[left][0], left_value)
                right_updates = _group_updates(base, groups[right][0], right_value)
                if updates is not None and right_updates is not None:
                    yield _replace_positions(base, {**updates, **right_updates})
        if len(groups) > 2:
            for values in _permuted_product(domains, seed=seed, schema_key=schema_key):
                updates: dict[tuple[str, int, int], int] = {}
                for (positions, _observed), value in zip(groups, values):
                    group_update = _group_updates(base, positions, value)
                    if group_update is None:
                        updates = {}
                        break
                    updates.update(group_update)
                if updates:
                    yield _replace_positions(base, updates)


# Maximum HBM budget for a single replay case (32 GiB on a 64 GiB card).
_QUERY_MAX_HBM_BYTES = 32 * 1024 ** 3

_DTYPE_TO_BYTES: dict[str, int] = {
    "DT_BF16": 2, "DT_FLOAT16": 2, "FLOAT": 4, "DT_FLOAT": 4,
    "INT8": 1, "DT_INT8": 1, "INT32": 4, "DT_INT32": 4,
    "INT64": 8, "DT_INT64": 8, "BOOL": 1,
}


def _estimate_candidate_bytes(candidate: "ProjectedCandidate") -> int:
    total = 0
    for shape, dtype in zip(candidate.input_shapes, candidate.input_dtypes):
        if shape:
            bpe = _DTYPE_TO_BYTES.get(dtype.upper(), 2)
            total += math.prod(shape) * bpe
    for shape, dtype in zip(candidate.output_shapes, candidate.output_dtypes):
        if shape:
            bpe = _DTYPE_TO_BYTES.get(dtype.upper(), 2)
            total += math.prod(shape) * bpe
    return total


def _broadcast_result(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...] | None:
    """Return the NumPy-compatible broadcast of two shapes, or None if invalid."""
    result: list[int] = []
    for l_dim, r_dim in zip_longest(reversed(left), reversed(right), fillvalue=1):
        if l_dim == r_dim:
            result.append(l_dim)
        elif l_dim == 1:
            result.append(r_dim)
        elif r_dim == 1:
            result.append(l_dim)
        else:
            return None
    result.reverse()
    return tuple(result)


def _validates_elementwise_binary(candidate: "ProjectedCandidate") -> bool:
    """Reject elementwise binary candidates whose inputs are not broadcastable."""
    present = [shape for shape in candidate.input_shapes if shape]
    if len(present) < 2:
        return True
    merged = present[0]
    for shape in present[1:]:
        merged = _broadcast_result(merged, shape)
        if merged is None:
            return False
    if candidate.output_shapes and candidate.output_shapes[0]:
        return candidate.output_shapes[0] == merged
    return True


def _validate_candidate(candidate: "ProjectedCandidate") -> bool:
    """Return False if the candidate violates kernel-specific replay constraints."""
    kernel = candidate.kernel_type

    # TransposeBatchMatMul: K and N (last two dims of weight matrix) must be
    # divisible by 64 for the fused kernel.
    if kernel == "TransposeBatchMatMul":
        for shape in candidate.input_shapes[1:]:
            if len(shape) >= 2 and (shape[-1] % 64 != 0 or shape[-2] % 64 != 0):
                return False

    # Index: output dims must not exceed source dims.
    if kernel == "Index" and candidate.input_shapes and candidate.output_shapes:
        source = candidate.input_shapes[0]
        for out_shape in candidate.output_shapes:
            if not source or not out_shape:
                continue
            for s_dim, o_dim in zip(source, out_shape):
                if o_dim > s_dim:
                    return False

    # MatMul family: the replay adapter applies transpose=True to the second
    # input, so the CSV must store the weight matrix as (N, K).  When the
    # projection or coverage interpolation leaves it as (K, N), the replay
    # would fail with "mat1 and mat2 shapes cannot be multiplied".  Reject
    # candidates whose second input's last dim does not match the first
    # input's last dim (K), unless the matrix is square (K == N).
    # TransposeBatchMatMul stores (B, K, N) without transpose, so skip it.
    _TRANSPOSE_MATMUL_KERNELS = {"TransposeBatchMatMul"}
    if kernel in _MATMUL_KERNELS and kernel not in _TRANSPOSE_MATMUL_KERNELS and len(candidate.input_shapes) >= 2:
        in1 = candidate.input_shapes[0]
        in2 = candidate.input_shapes[1]
        out = candidate.output_shapes[0] if candidate.output_shapes else ()
        # GroupedMatmul/SwiGluQuant store the expert weight natively as
        # (E, K, N) (or FRACTAL_NZ); only the transpose-convention kernels
        # below need the (N, K) orientation validation.
        if kernel not in _MATMUL_TRANSPOSE_WEIGHT_KERNELS and kernel != "BatchMatMulV2":
            pass
        elif kernel == "BatchMatMulV2" and in1 and in2 and len(in1) == 3 and len(in2) == 3:
            batch, m, k = in1
            if len(out) != 3 or out[0] != batch or out[1] != m:
                return False
            # The adapter accepts (B, K, N) or (B, N, K) for the second input
            # and selects transpose_b accordingly.
            if in2 not in {(batch, k, out[2]), (batch, out[2], k)}:
                return False
        elif in1 and in2 and out and len(in1) >= 2 and len(in2) >= 2 and len(out) >= 1:
            k = in1[-1]
            n = out[-1]
            if in2[-1] != k or in2[-2] != n:
                return False

    # Elementwise binary (Add/Mul/AddRmsNormBias): inputs must be broadcastable
    # and the recorded output must equal the broadcast result.  Coverage
    # interpolation previously mixed independent axis domains across operands,
    # e.g. "1,1,256;1,32,6144", which fails NPU replay at runtime.
    if kernel in _ELEMENTWISE_BINARY_KERNELS and not _validates_elementwise_binary(candidate):
        return False

    # AscendQuantV2: npu_quantize(axis=-1) requires per-channel scale/offset to
    # match the input's last dim (or be scalar), and the quantized output to keep
    # the input shape.  Coverage interpolation produced e.g. "1,1,6144;384;384"
    # (scale 384 for last dim 6144) which fails with aclnnAscendQuantV3 161002.
    if kernel == "AscendQuantV2" and candidate.input_shapes and candidate.input_shapes[0]:
        last = candidate.input_shapes[0][-1]
        for scale in candidate.input_shapes[1:3]:
            if scale and scale != (1,) and (not scale or scale[-1] != last):
                return False
        # scale/offset dtypes must equal the input dtype; mixing e.g. FP16
        # input with BF16 scale fails aclnnAscendQuantV3 (error 161002).
        if len(candidate.input_dtypes) >= 2 and candidate.input_dtypes[0]:
            input_dtype = candidate.input_dtypes[0]
            for scale_dtype in candidate.input_dtypes[1:3]:
                if scale_dtype and scale_dtype != input_dtype:
                    return False
        if candidate.output_shapes and candidate.output_shapes[0]:
            if candidate.output_shapes[0] != candidate.input_shapes[0]:
                return False

    # DynamicQuant: the per-token scale is a real second output whose shape is
    # the input without its last dim.  Rows with a missing or wrong scale slot
    # can never be matched back from the profiler (profile records e.g.
    # 1,1,1254;1,1), so reject them during generation.
    if (
        kernel == "DynamicQuant"
        and candidate.input_shapes
        and candidate.input_shapes[0]
        and len(candidate.input_shapes[0]) >= 2
    ):
        in_shape = candidate.input_shapes[0]
        if len(candidate.output_shapes) < 2 or not candidate.output_shapes[1]:
            return False
        if candidate.output_shapes[1] != in_shape[:-1]:
            return False

    # TransposeBatchMatMul: the fused kernel returns (M, B, N).  The recorded
    # output must equal the batch-matmul result of the projected inputs or the
    # replay adapter drops the row (ValueError: fused kernel output layout).
    if kernel == "TransposeBatchMatMul" and len(candidate.input_shapes) >= 2:
        lhs, rhs = candidate.input_shapes[0], candidate.input_shapes[1]
        if lhs and rhs and len(lhs) == 3 and len(rhs) == 3:
            lhs_batch, lhs_m, lhs_k = lhs
            rhs_batch, rhs_k, rhs_n = rhs
            if lhs_batch != rhs_batch or lhs_k != rhs_k:
                return False
            expected = (lhs_m, lhs_batch, rhs_n)
            if candidate.output_shapes and candidate.output_shapes[0] != expected:
                return False

    # BatchMatMulV2 batched rows: reject degenerate contraction dims that never
    # dispatch a measurable GEMM kernel (K=1 rows replayed OK but msprof records
    # them under a different operator type, so the CSV can never be backfilled).
    if kernel == "BatchMatMulV2" and len(candidate.input_shapes) >= 2:
        lhs, rhs = candidate.input_shapes[0], candidate.input_shapes[1]
        if lhs and rhs and len(lhs) == 3 and len(rhs) == 3:
            lhs_k = lhs[2]
            if rhs[1] == lhs_k:
                rhs_k = rhs[1]  # (B, K, N)
            elif rhs[2] == lhs_k:
                rhs_k = rhs[2]  # (B, N, K)
            else:
                return False
            if lhs_k < _MIN_BATCH_MATMUL_K or rhs_k < _MIN_BATCH_MATMUL_K:
                return False

    # Memory budget: reject shapes exceeding single-card HBM.
    if _estimate_candidate_bytes(candidate) > _QUERY_MAX_HBM_BYTES:
        return False

    return True


_EXACT_STATUS_PENDING = "pending"


def build_query_generated_rows(
    *,
    csv_path: Path,
    headers: list[str],
    source_rows: list[dict[str, str]],
    demands: Iterable[KernelQueryDemand],
    row_limit: int,
    seed: int,
    exact_unlimited_workload_ids: Collection[str] = (),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return up to ``row_limit`` new rows; duplicates never consume budget.

    Exact demands whose ``workload_id`` is in ``exact_unlimited_workload_ids``
    are always appended and never consume the ``--rows`` budget. Other exact
    demands share that budget with coverage and constraint fallback candidates.
    Every exact demand receives a deterministic status so no demand is silently
    dropped.
    """
    demand_list = list(demands)
    templates = _unique_templates(source_rows)
    projected: list[tuple[ProjectedCandidate, dict[str, str]]] = []
    rejected = 0
    seen_candidate_keys: set[tuple[Any, ...]] = set()
    candidate_keys: dict[int, tuple[Any, ...]] = {}
    demand_keys: list[tuple[Any, ...] | None] = []
    key_owner: dict[tuple[Any, ...], int] = {}
    unlimited_keys: set[tuple[Any, ...]] = set()
    for demand in demand_list:
        result = project_query_demand(demand, headers, templates)
        if result is None:
            rejected += 1
            demand_keys.append(None)
            continue
        candidate, template = result
        key = (
            candidate.schema_key,
            candidate.input_shapes,
            candidate.output_shapes,
            candidate.extra_values,
        )
        candidate_keys[id(candidate)] = key
        demand_keys.append(key)
        is_unlimited = demand.workload_id in exact_unlimited_workload_ids
        if is_unlimited:
            unlimited_keys.add(key)
        if key in seen_candidate_keys:
            if is_unlimited and demand_list[key_owner[key]].workload_id not in exact_unlimited_workload_ids:
                key_owner[key] = len(demand_keys) - 1
            continue
        seen_candidate_keys.add(key)
        key_owner[key] = len(demand_keys) - 1
        projected.append((candidate, template))

    projected.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0].schema_key}:{item[0].input_shapes}:{item[0].extra_values}".encode()
        ).digest()
    )
    candidate_template: dict[int, dict[str, str]] = {
        id(candidate): template for candidate, template in projected
    }
    template_by_schema = {candidate.schema_key: template for candidate, template in projected}
    exact = [candidate for candidate, _template in projected]

    seen_rows = {profile_dedupe_key(csv_path, headers, row) for row in source_rows}
    generated: list[dict[str, str]] = []
    duplicate_count = 0
    rejected_count = 0
    key_status: dict[tuple[Any, ...], str] = {}
    exact_existing = 0
    exact_generated = 0
    exact_budget_truncated = 0
    generated_exact_demand_indexes: list[int] = []

    def candidate_key(candidate: ProjectedCandidate) -> tuple[Any, ...]:
        return candidate_keys[id(candidate)]

    budget_used = 0
    for candidate in exact:
        key = candidate_key(candidate)
        is_unlimited = key in unlimited_keys
        if not is_unlimited and budget_used >= row_limit:
            key_status[key] = "budget_truncated"
            exact_budget_truncated += 1
            continue
        if not _validate_candidate(candidate):
            rejected_count += 1
            key_status[key] = "validation_rejected"
            continue
        template = candidate_template.get(id(candidate)) or template_by_schema[candidate.schema_key]
        row = candidate.to_row(headers, template)
        dedupe_key = profile_dedupe_key(csv_path, headers, row)
        if dedupe_key in seen_rows:
            duplicate_count += 1
            key_status[key] = "existing"
            exact_existing += 1
            continue
        seen_rows.add(dedupe_key)
        generated.append(row)
        generated_exact_demand_indexes.append(key_owner[key])
        key_status[key] = "generated"
        exact_generated += 1
        if not is_unlimited:
            budget_used += 1

    coverage_appended = 0
    if budget_used < row_limit:
        for candidate in _coverage_candidates(exact, seed):
            if budget_used >= row_limit:
                break
            if not _validate_candidate(candidate):
                rejected_count += 1
                continue
            template = template_by_schema[candidate.schema_key]
            row = candidate.to_row(headers, template)
            dedupe_key = profile_dedupe_key(csv_path, headers, row)
            if dedupe_key in seen_rows:
                duplicate_count += 1
                continue
            seen_rows.add(dedupe_key)
            generated.append(row)
            coverage_appended += 1
            budget_used += 1

    # ── Constraint fallback (stage 3): when exact + coverage (stages 1-2)
    #    produce fewer rows than requested, fill the remainder with shapes
    #    derived from build_case constraints. Only operators registered in
    #    constraint_fallback._REGISTRY are handled; others stop here.
    kernel_type = csv_path.stem
    fallback_attempted = 0
    fallback_duplicates = 0
    fallback_appended = 0
    fallback_rejected_safety = 0
    if budget_used < row_limit:
        remaining = row_limit - budget_used
        fallback_rows, fallback_summary = build_constraint_fallback_rows(
            kernel_type=kernel_type,
            csv_path=csv_path,
            headers=headers,
            source_rows=source_rows,
            row_limit=remaining,
            seed=seed,
        )
        fallback_attempted = fallback_summary.get("attempted", 0)
        fallback_duplicates = fallback_summary.get("duplicates", 0)
        fallback_appended = len(fallback_rows)
        fallback_rejected_safety = fallback_summary.get("rejected_safety", 0)
        generated.extend(fallback_rows)

    # One deterministic status per demand occurrence, aligned with the input
    # demand order; identical signatures still get their own occurrence entry.
    demand_statuses: list[str] = []
    for index, key in enumerate(demand_keys):
        if key is None:
            status = "projection_rejected"
        elif key_owner.get(key) != index:
            status = "duplicate"
        else:
            status = key_status.get(key, "unresolved")
        demand_statuses.append(status)

    return generated, {
        "demands": len(demand_list),
        "projected_exact": len(exact),
        "rejected": rejected,
        "duplicates": duplicate_count,
        "appended": len(generated),
        "validation_rejected": rejected_count,
        "exact_existing": exact_existing,
        "exact_generated": exact_generated,
        "exact_budget_truncated": exact_budget_truncated,
        "coverage_appended": coverage_appended,
        "demand_statuses": demand_statuses,
        "generated_exact_demand_indexes": generated_exact_demand_indexes,
        "fallback_attempted": fallback_attempted,
        "fallback_duplicates": fallback_duplicates,
        "fallback_appended": fallback_appended,
        "fallback_rejected_safety": fallback_rejected_safety,
    }
