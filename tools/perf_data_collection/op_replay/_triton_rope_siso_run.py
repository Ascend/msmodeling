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

"""Replay vLLM-Ascend single-input/single-output Triton RoPE cases."""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        normalize_dtype_name,
        parse_shape,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        normalize_dtype_name,
        parse_shape,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def resolve_rope_api():
    from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    return rope_forward_triton_siso


def _resolve_profiler_dtype(dtype_name: str, fallback: str) -> str:
    """Map opaque Triton profiler dtype IDs to the qk tensor dtype."""

    cleaned = dtype_name.strip()
    if cleaned.lstrip("-").isdigit():
        return fallback
    return normalize_dtype_name(cleaned)


def build_case(row: dict[str, str]):
    input_shape_cells = split_metadata_field(row["Input Shapes"])
    input_dtypes = split_metadata_field(row["Input Data Types"])
    input_formats = split_metadata_field(row["Input Formats"])
    output_shape_cells = split_metadata_field(row["Output Shapes"])

    if len(input_shape_cells) != 3 or len(input_dtypes) != 3 or len(input_formats) != 3:
        raise ValueError("_triton_rope_siso expects qk, cos, and sin input slots")
    if len(output_shape_cells) != 1:
        raise ValueError("_triton_rope_siso expects one output")

    input_shapes = [parse_shape(cell) for cell in input_shape_cells]
    output_shape = parse_shape(output_shape_cells[0])
    qk_shape, cos_shape, sin_shape = input_shapes
    if len(qk_shape) != 3 or len(cos_shape) != 2 or sin_shape != cos_shape:
        raise ValueError("_triton_rope_siso requires qk=(tokens,heads,head_dim) and matching rank-2 cos/sin")
    if qk_shape[0] != cos_shape[0] or output_shape != qk_shape:
        raise ValueError(
            f"_triton_rope_siso token/output mismatch: qk={qk_shape}, cos={cos_shape}, output={output_shape}"
        )
    rope_dim = cos_shape[-1] * 2
    if rope_dim > qk_shape[-1]:
        raise ValueError(f"_triton_rope_siso rope_dim {rope_dim} exceeds head_dim {qk_shape[-1]}")

    qk_dtype = normalize_dtype_name(input_dtypes[0])
    resolved_dtypes = [
        qk_dtype,
        _resolve_profiler_dtype(input_dtypes[1], qk_dtype),
        _resolve_profiler_dtype(input_dtypes[2], qk_dtype),
    ]
    tensors = [
        build_input_tensor(shape, tensor_format, dtype_name)
        for shape, tensor_format, dtype_name in zip(input_shapes, input_formats, resolved_dtypes)
    ]
    return {
        "api": resolve_rope_api(),
        "inputs": tensors,
        "rope_dim": rope_dim,
        "output_shape": output_shape,
    }


def run_case(case):
    qk, cos, sin = case["inputs"]
    return case["api"](qk, cos, sin, rope_dim=case["rope_dim"])


op = OpReplay(
    kernel_type="_triton_rope_siso",
    description=("Replay _triton_rope_siso.csv through vLLM-Ascend rope_forward_triton_siso on Ascend NPU."),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/_triton_rope_siso_run.py --database-path /path/to/database"
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
