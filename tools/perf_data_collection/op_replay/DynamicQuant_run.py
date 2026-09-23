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
Replay DynamicQuant cases from the performance database on Ascend NPU.

Purpose:
  Read DynamicQuant rows from
  profiling_database/data/{device}/vllm_ascend/{version}/DynamicQuant.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_dynamic_quant().
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def format_success(csv_path, row_index: int, row: dict[str, str], _case, result) -> str:
    output, scale = result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output0={tuple(output.shape)} "
        f"output1={tuple(scale.shape)}"
    )


op = OpReplay(
    kernel_type="DynamicQuant",
    api_path="torch_npu.npu_dynamic_quant",
    description=(
        "Run DynamicQuant workload replay on Ascend NPU.\n"
        "The script reads DynamicQuant.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch_npu.npu_dynamic_quant()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/DynamicQuant_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    input_count=1,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
