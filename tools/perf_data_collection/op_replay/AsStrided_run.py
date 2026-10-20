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
Replay AsStrided cases from the performance database on Ascend NPU.

Purpose:
  Read AsStrided rows from the profiling database, rebuild the
  recorded source tensor (size/stride/offset are scalar-slot
  tensors whose values are not recorded), then execute
  torch.as_strided(input, size, stride, offset) using the
  recorded output shape and a contiguous stride.
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not input_shapes:
        raise ValueError("AsStrided requires at least one recorded input tensor")
    if len(output_shapes) != 1:
        raise ValueError("AsStrided requires exactly one recorded output shape")
    # A recorded scalar output parses to () and is a valid
    # torch.as_strided size; only a missing output slot (already
    # filtered by parse_list_field) is rejected above.

    tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0] if input_formats else "ND",
        dtype_name=input_dtypes[0] if input_dtypes else "DT_FLOAT",
    )
    size = output_shapes[0]
    needed = 1
    for dim in size:
        needed *= dim
    if tensor.numel() < needed:
        raise ValueError(
            f"AsStrided source tensor too small: numel={tensor.numel()} needed={needed}"
        )
    return {
        "inputs": [tensor],
        "kwargs": {"size": size, "stride": contiguous_stride(size), "storage_offset": 0},
        "api": op.resolve_api(),
    }


def run_case(case):
    kwargs = case["kwargs"]
    return case["api"](
        case["inputs"][0],
        kwargs["size"],
        kwargs["stride"],
        kwargs["storage_offset"],
    )


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"stride={case['kwargs']['stride']}"
    )


op = OpReplay(
    kernel_type="AsStrided",
    api_path="torch.as_strided",
    description=(
        "Run AsStrided workload replay on Ascend NPU.\n"
        "The script reads AsStrided.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the input tensor,\n"
        "then runs torch.as_strided(input, size, stride, offset) with\n"
        "the recorded output shape and a contiguous stride."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/AsStrided_run.py "
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
