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
Replay Tile cases from the performance database on Ascend NPU.

Purpose:
  Read Tile rows from the profiling database, rebuild the recorded
  source tensor, infer the multiples from input/output shapes
  (the multiples tensor values are not recorded), then execute
  torch.tile(input, multiples).
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def infer_multiples(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(input_shape) != len(output_shape):
        raise ValueError(
            f"Tile rank mismatch: input={input_shape} output={output_shape}"
        )
    multiples = []
    for in_dim, out_dim in zip(input_shape, output_shape):
        if in_dim == 0 or out_dim % in_dim != 0:
            raise ValueError(
                f"Cannot infer Tile multiples from input={input_shape} output={output_shape}"
            )
        multiples.append(out_dim // in_dim)
    return tuple(multiples)


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not input_shapes:
        raise ValueError("Tile requires at least one recorded input tensor")
    if len(output_shapes) != 1:
        raise ValueError("Tile requires exactly one recorded output shape")

    tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0] if input_formats else "ND",
        dtype_name=input_dtypes[0] if input_dtypes else "DT_FLOAT",
    )
    multiples = infer_multiples(tuple(tensor.shape), output_shapes[0])
    return {
        "inputs": [tensor],
        "kwargs": {"multiples": multiples},
        "api": op.resolve_api(),
    }


def run_case(case):
    return case["api"](case["inputs"][0], case["kwargs"]["multiples"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"multiples={case['kwargs']['multiples']}"
    )


op = OpReplay(
    kernel_type="Tile",
    api_path="torch.tile",
    description=(
        "Run Tile workload replay on Ascend NPU.\n"
        "The script reads Tile.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the input tensor,\n"
        "infers the multiples from the recorded output shape, then\n"
        "runs torch.tile(input, multiples)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/Tile_run.py "
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
