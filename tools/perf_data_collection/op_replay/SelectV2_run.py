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
Replay SelectV2 cases from the performance database on Ascend NPU.

Purpose:
  Read SelectV2 rows from the profiling database, rebuild the
  (condition, x, y) inputs — empty slots are scalar scalar-slots —
  then execute torch.where(condition, x, y).
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
        resolve_runtime_dtype,
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
        resolve_runtime_dtype,
    )
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    raw_shapes = split_slots(row["Input Shapes"])
    raw_dtypes = split_slots(row["Input Data Types"])
    raw_formats = split_slots(row["Input Formats"])
    if len(raw_shapes) < 3:
        raise ValueError("SelectV2 expects condition, x, and y slots")

    tensors = []
    for index in range(3):
        dtype_name = normalize_dtype_name(
            raw_dtypes[index] if index < len(raw_dtypes) else "DT_FLOAT"
        )
        fmt = raw_formats[index] if index < len(raw_formats) else "ND"
        shape_text = raw_shapes[index].strip()
        if shape_text:
            tensors.append(
                build_input_tensor(
                    shape=parse_shape(shape_text),
                    input_format=fmt,
                    dtype_name=dtype_name,
                )
            )
        else:
            # Empty (scalar) slot: SelectV2 allows scalar x or y operands.
            tensors.append(build_host_tensor((), resolve_runtime_dtype(dtype_name)).npu())
    return {
        "inputs": tensors,
        "kwargs": {},
        "api": op.resolve_api(),
    }


def split_slots(raw_value: str) -> list[str]:
    return raw_value.strip().strip('"').split(";")


def run_case(case):
    condition, x_value, y_value = case["inputs"][:3]
    return case["api"](condition, x_value, y_value)


op = OpReplay(
    kernel_type="SelectV2",
    api_path="torch.where",
    description=(
        "Run SelectV2 workload replay on Ascend NPU.\n"
        "The script reads SelectV2.csv under the selected device and\n"
        "vllm_ascend version directory, rebuilds the (condition, x, y)\n"
        "inputs (empty slots become scalar tensors), then runs\n"
        "torch.where(condition, x, y)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/SelectV2_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
