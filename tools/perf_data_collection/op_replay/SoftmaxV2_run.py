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
Replay SoftmaxV2 cases from the performance database on Ascend NPU.

Purpose:
  Read SoftmaxV2 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/SoftmaxV2.csv,
  rebuild the recorded tensor inputs, then execute
  torch.nn.functional.softmax() along the last dimension.
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    input_tensor = case["inputs"][0]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shape={tuple(input_tensor.shape)} dtype={row['Input Data Types']} "
        f"format={row['Input Formats']} output={tuple(result.shape)} dim=-1"
    )


op = OpReplay(
    kernel_type="SoftmaxV2",
    api_path="torch.nn.functional.softmax",
    description=(
        "Run SoftmaxV2 workload replay on Ascend NPU.\n"
        "The script reads SoftmaxV2.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch.nn.functional.softmax(input, dim=-1)."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/SoftmaxV2_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    input_count=1,
    fixed_kwargs={"dim": -1},
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
