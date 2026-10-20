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
Replay DequantSwigluQuant cases from the performance database on Ascend NPU.

Purpose:
  Read DequantSwigluQuant rows from the profiling database,
  rebuild the recorded (x, weight_scale, activation_scale,
  group_index) tensors, then execute
  torch_npu.npu_dequant_swiglu_quant() exactly like the
  vLLM-Ascend MoE MLP path does (quant_mode=1).
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape,
    )
    from replay_framework import OpReplay


def split_slots(raw_value: str) -> list[str]:
    return raw_value.strip().strip('"').split(";")


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    # split_metadata_field keeps empty slots in place (unlike
    # parse_list_field) so the group_index slot position stays stable.
    raw_shapes = split_slots(row["Input Shapes"])
    input_shapes = [parse_shape(item) for item in raw_shapes]
    input_formats = split_slots(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in split_slots(row["Input Data Types"])]
    if len(input_shapes) < 3:
        raise ValueError(
            "DequantSwigluQuant expects x, weight_scale, and activation_scale slots"
        )

    x = build_input_tensor(input_shapes[0], input_formats[0], input_dtypes[0])
    weight_scale = build_input_tensor(input_shapes[1], input_formats[1], input_dtypes[1])
    activation_scale = build_input_tensor(input_shapes[2], input_formats[2], input_dtypes[2])

    # The last slot records the cumulative group_index tensor (one entry
    # per group); empty intermediate slots (bias/quant_scale/quant_offset)
    # keep its position stable.  quant_mode=1 requires a group_index.
    group_index = None
    if len(input_shapes) >= 7 and input_shapes[6]:
        group_count = input_shapes[6][0]
        base = x.shape[0] // group_count
        remainder = x.shape[0] % group_count
        cumulative = []
        running = 0
        for idx in range(group_count):
            running += base + (1 if idx < remainder else 0)
            cumulative.append(running)
        group_index = runtime_torch.tensor(cumulative, dtype=runtime_torch.int64).npu()
    else:
        raise ValueError(
            "DequantSwigluQuant requires a group_index slot for quant_mode=1"
        )

    return {
        "inputs": [],
        "kwargs": {
            "x": x,
            "weight_scale": weight_scale,
            "activation_scale": activation_scale,
            "bias": None,
            "quant_scale": None,
            "quant_offset": None,
            "group_index": group_index,
            "activate_left": True,
            "quant_mode": 1,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    # The npu_dequant_swiglu_quant op schema takes x positionally and
    # every other argument as keyword-only.
    result = case["api"](case["kwargs"]["x"], **{
        key: value for key, value in case["kwargs"].items() if key != "x"
    })
    if isinstance(result, (tuple, list)):
        return tuple(result[:2])
    return result


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    output = result[0] if isinstance(result, (tuple, list)) and result else result
    output_shape = tuple(output.shape) if hasattr(output, "shape") else str(output)
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={output_shape}"
    )


op = OpReplay(
    kernel_type="DequantSwigluQuant",
    api_path="torch.ops.npu.npu_dequant_swiglu_quant",
    description=(
        "Run DequantSwigluQuant workload replay on Ascend NPU.\n"
        "The script reads DequantSwigluQuant.csv under the selected\n"
        "device and vllm_ascend version directory, rebuilds the\n"
        "(x, weight_scale, activation_scale) tensors, then runs\n"
        "torch_npu.npu_dequant_swiglu_quant(quant_mode=1)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/DequantSwigluQuant_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.23.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
