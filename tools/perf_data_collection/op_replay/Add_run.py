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
Replay Add cases from the performance database on Ascend NPU, CUDA, or CPU.

Purpose:
  Read Add rows from
  profiling_database/data/{device}/vllm_ascend/{version}/Add.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.add() workload.

Usage:
  python tools/perf_data_collection/op_replay/Add_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    case = {
        "inputs": op.build_inputs(row),
        "kwargs": {},
        "api": op.resolve_api(),
    }
    if not case["inputs"]:
        raise ValueError("Add expects at least one input")
    return case


def run_case(case):
    if len(case["inputs"]) >= 2:
        return case["api"](case["inputs"][0], case["inputs"][1])
    # Keep integral inputs integral when the database records an omitted scalar
    # slot (for example ``Input Shapes=8;``).  A float literal promotes an
    # INT32 tensor and makes the profiler signature impossible to match back
    # to the recorded INT32 output.
    return case["api"](case["inputs"][0], 1)


op = OpReplay(
    kernel_type="Add",
    api_path="torch.add",
    description=(
        "Run Add workload replay on Ascend NPU/CUDA/CPU.\n"
        "The script reads Add.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs torch.add()."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/Add_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.9.2.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":  # Keep imports side-effect free for tooling tests.
    main()
