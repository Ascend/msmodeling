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
Replay BroadcastTo cases from the performance database on Ascend NPU.

Purpose:
  Read BroadcastTo rows from the profiling database, rebuild the
  recorded source tensor (the shape/size tensor slot is ignored),
  then execute tensor.expand(output_shape).contiguous() using the
  recorded output shape.
"""

from __future__ import annotations

try:
    from .common import (
        build_host_tensor,
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_host_tensor,
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not input_shapes:
        raise ValueError("BroadcastTo requires at least one recorded input slot")
    if len(output_shapes) != 1 or not output_shapes[0]:
        raise ValueError("BroadcastTo requires exactly one recorded output shape")

    source_shape = input_shapes[0]
    source_dtype = input_dtypes[0] if input_dtypes else "DT_FLOAT"
    source_format = input_formats[0] if input_formats else "ND"
    if source_shape:
        tensor = build_input_tensor(
            shape=source_shape,
            input_format=source_format,
            dtype_name=source_dtype,
        )
    else:
        # Scalar source slot: BroadcastTo can also broadcast a scalar.
        dtype = runtime_torch.int32 if "INT32" in source_dtype else (
            runtime_torch.int64 if "INT64" in source_dtype else runtime_torch.float32
        )
        tensor = build_host_tensor((), dtype).npu()

    output_shape = output_shapes[0]
    return {
        "inputs": [tensor],
        "kwargs": {"output_shape": output_shape},
        "api": None,
    }


def run_case(case):
    tensor = case["inputs"][0]
    return tensor.expand(*case["kwargs"]["output_shape"]).contiguous()


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)}"
    )


op = OpReplay(
    kernel_type="BroadcastTo",
    description=(
        "Run BroadcastTo workload replay on Ascend NPU.\n"
        "The script reads BroadcastTo.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded source\n"
        "tensor, then runs tensor.expand(output_shape).contiguous()."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/BroadcastTo_run.py "
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
