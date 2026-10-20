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

"""
Replay ReduceSum cases from the performance database on Ascend NPU.

Purpose:
  Read ReduceSum rows from the profiling database, rebuild the
  recorded input tensor, infer the reduction dim from the recorded
  output shape, then execute torch.sum(input, dim).
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def infer_reduce_dim(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int | None:
    """Return the reduced dim, or None for a full reduction."""
    if len(input_shape) - len(output_shape) == 1:
        for dim in range(len(input_shape)):
            reduced = input_shape[:dim] + input_shape[dim + 1:]
            if reduced == output_shape:
                return dim
    if not output_shape:
        return None
    raise ValueError(
        f"Cannot infer ReduceSum dim from input={input_shape} output={output_shape}"
    )


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not input_shapes:
        raise ValueError("ReduceSum requires at least one recorded input tensor")
    if len(output_shapes) != 1:
        raise ValueError("ReduceSum requires exactly one recorded output shape")

    tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0] if input_formats else "ND",
        dtype_name=input_dtypes[0] if input_dtypes else "DT_FLOAT",
    )
    dim = infer_reduce_dim(tuple(tensor.shape), output_shapes[0])
    return {
        "inputs": [tensor],
        "kwargs": {"dim": dim},
        "api": op.resolve_api(),
    }


def run_case(case):
    dim = case["kwargs"]["dim"]
    if dim is None:
        return case["api"](case["inputs"][0])
    return case["api"](case["inputs"][0], dim)


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"dim={case['kwargs']['dim']}"
    )


op = OpReplay(
    kernel_type="ReduceSum",
    api_path="torch.sum",
    description=(
        "Run ReduceSum workload replay on Ascend NPU.\n"
        "The script reads ReduceSum.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the input tensor,\n"
        "infers the reduction dim from the recorded output shape, then\n"
        "runs torch.sum(input, dim)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/ReduceSum_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
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
