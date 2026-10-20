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
Replay ConcatD cases from the performance database on Ascend NPU.

Purpose:
  Read ConcatD rows from the profiling database, rebuild the
  recorded tensor inputs, infer the concat dimension from the
  recorded output shape, then execute torch.cat().
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def infer_concat_dim(input_shapes: list[tuple[int, ...]], output_shape: tuple[int, ...]) -> int:
    candidates = []
    for dim in range(len(output_shape)):
        if all(
            len(shape) == len(output_shape) and shape[:dim] == output_shape[:dim]
            and shape[dim + 1:] == output_shape[dim + 1:]
            for shape in input_shapes
        ) and output_shape[dim] == sum(shape[dim] for shape in input_shapes):
            candidates.append(dim)
    if not candidates:
        raise ValueError(
            f"Cannot infer ConcatD dim from inputs={input_shapes} output={output_shape}"
        )
    return candidates[0]


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if len(input_shapes) < 2:
        raise ValueError("ConcatD expects at least two input tensors")
    if len(output_shapes) != 1:
        raise ValueError("ConcatD expects exactly one output shape")
    if len(input_shapes) != len(input_dtypes) or len(input_shapes) != len(input_formats):
        raise ValueError(
            f"ConcatD metadata length mismatch: shapes={len(input_shapes)}, "
            f"dtypes={len(input_dtypes)}, formats={len(input_formats)}"
        )

    inputs = [
        build_input_tensor(
            shape=shape,
            input_format=input_formats[index],
            dtype_name=dtypes,
        )
        for index, (shape, dtypes) in enumerate(zip(input_shapes, input_dtypes))
    ]
    dim = infer_concat_dim(input_shapes, output_shapes[0])
    return {
        "inputs": inputs,
        "kwargs": {"dim": dim},
        "api": op.resolve_api(),
    }


def run_case(case):
    return case["api"](case["inputs"], case["kwargs"]["dim"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"dim={case['kwargs']['dim']}"
    )


op = OpReplay(
    kernel_type="ConcatD",
    api_path="torch.cat",
    description=(
        "Run ConcatD workload replay on Ascend NPU.\n"
        "The script reads ConcatD.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded tensor\n"
        "inputs, infers the concat dimension from the recorded output\n"
        "shape, then runs torch.cat(inputs, dim)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/ConcatD_run.py "
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
