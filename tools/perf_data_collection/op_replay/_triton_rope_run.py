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

"""Replay vLLM-Ascend MHA Triton RoPE cases."""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def resolve_rope_api():
    from vllm_ascend.ops.triton.rope import rope_forward_triton
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    return rope_forward_triton


def _resolve_profiler_dtype(dtype_name: str, fallback: str) -> str:
    """Map opaque Triton profiler dtype IDs to the q tensor dtype."""
    cleaned = dtype_name.strip()
    if cleaned.lstrip("-").isdigit():
        return fallback
    return normalize_dtype_name(cleaned)


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    input_shape_cells = split_metadata_field(row["Input Shapes"])
    input_dtypes = split_metadata_field(row["Input Data Types"])
    input_formats = split_metadata_field(row["Input Formats"])
    if len(input_shape_cells) != 3:
        raise ValueError("_triton_rope expects q, k, and cos_sin_cache slots")

    input_shapes = [parse_shape(cell) for cell in input_shape_cells]
    q_shape, k_shape, cache_shape = input_shapes
    if len(q_shape) != 3 or len(k_shape) != 3 or len(cache_shape) != 2:
        raise ValueError(
            "_triton_rope requires q/k=(tokens,heads,head_dim) and a rank-2 cos_sin_cache"
        )
    if q_shape[0] != k_shape[0]:
        raise ValueError("_triton_rope q/k token count mismatch")
    rope_dim = cache_shape[-1]
    if rope_dim > q_shape[-1]:
        raise ValueError(f"_triton_rope rope_dim {rope_dim} exceeds head_dim {q_shape[-1]}")

    q_dtype = normalize_dtype_name(input_dtypes[0])
    resolved_dtypes = [
        q_dtype,
        _resolve_profiler_dtype(input_dtypes[1], q_dtype),
        _resolve_profiler_dtype(input_dtypes[2], q_dtype),
    ]
    tensors = [
        build_input_tensor(shape, tensor_format, dtype_name)
        for shape, tensor_format, dtype_name in zip(input_shapes, input_formats, resolved_dtypes)
    ]
    positions = runtime_torch.arange(q_shape[0], dtype=runtime_torch.int64).npu()
    return {
        "api": resolve_rope_api(),
        "inputs": tensors,
        "positions": positions,
        "rope_dim": rope_dim,
    }


def run_case(case):
    q, k, cos_sin_cache = case["inputs"]
    return case["api"](
        q,
        k,
        cos_sin_cache=cos_sin_cache,
        positions=case["positions"],
        rope_dim=case["rope_dim"],
    )


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    q_out, k_out = result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} "
        f"output={tuple(q_out.shape)};{tuple(k_out.shape)}"
    )


op = OpReplay(
    kernel_type="_triton_rope",
    description=("Replay _triton_rope.csv through vLLM-Ascend rope_forward_triton on Ascend NPU."),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/_triton_rope_run.py --database-path /path/to/database"
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
