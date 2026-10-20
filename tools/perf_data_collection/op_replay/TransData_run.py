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
Replay TransData cases from the performance database on Ascend NPU.

Purpose:
  Read TransData rows from the profiling database, rebuild the
  recorded FRACTAL_NZ input tensor, then execute
  torch_npu.npu_format_cast(input, ND) to reproduce the
  FRACTAL_NZ -> ND data-format conversion.
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from replay_framework import OpReplay

# ACL format id for ND.
ND_FORMAT_ID = 2


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    if not input_shapes:
        raise ValueError("TransData requires at least one recorded input tensor")
    if input_formats and input_formats[0] != "FRACTAL_NZ":
        raise ValueError(
            f"TransData expects a FRACTAL_NZ input, got {input_formats[0]!r}"
        )

    tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format="FRACTAL_NZ",
        dtype_name=input_dtypes[0] if input_dtypes else "DT_INT8",
    )
    return {
        "inputs": [tensor],
        "kwargs": {"nd_format_id": ND_FORMAT_ID},
        "api": None,
    }


def run_case(case):
    _, runtime_torch_npu = get_runtime_modules()
    return runtime_torch_npu.npu_format_cast(case["inputs"][0], case["kwargs"]["nd_format_id"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)}"
    )


op = OpReplay(
    kernel_type="TransData",
    description=(
        "Run TransData workload replay on Ascend NPU.\n"
        "The script reads TransData.csv under the selected device and\n"
        "vllm_ascend version directory, rebuilds the FRACTAL_NZ input\n"
        "tensor, then runs torch_npu.npu_format_cast(input, ND)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/TransData_run.py "
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
