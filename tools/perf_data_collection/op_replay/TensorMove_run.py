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
Replay TensorMove cases from the performance database on Ascend NPU.

Purpose:
  Read TensorMove rows from
  profiling_database/data/{device}/vllm_ascend/{version}/TensorMove.csv,
  rebuild the recorded source tensor, create a matching destination tensor,
  then execute torch.Tensor.copy_().
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) != 1:
        raise ValueError("TensorMove expects exactly one recorded input")
    src = inputs[0]
    dst = src.clone()
    return {
        "inputs": [src],
        "dst": dst,
        "kwargs": {},
        "api": None,
    }


def run_case(case):
    case["dst"].copy_(case["inputs"][0])
    return case["dst"]


op = OpReplay(
    kernel_type="TensorMove",
    description=(
        "Run TensorMove workload replay on Ascend NPU.\n"
        "The script reads TensorMove.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded source\n"
        "tensor, creates a matching destination tensor, then runs\n"
        "torch.Tensor.copy_()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/TensorMove_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    input_count=1,
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
