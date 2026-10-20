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
Replay Cumsum cases from the performance database on Ascend NPU.

Purpose:
  Read Cumsum rows from the profiling database, rebuild the
  recorded input tensor (the dim slot tensor is recorded but its
  value is unknown; 1-D rows use dim=0), then execute
  torch.cumsum(input, dim).
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
        raise ValueError("Cumsum expects at least one input tensor")
    if case["inputs"][0].dim() != 1:
        raise ValueError("Cumsum replay only supports 1-D recorded rows")
    return case


def run_case(case):
    return case["api"](case["inputs"][0], 0)


op = OpReplay(
    kernel_type="Cumsum",
    api_path="torch.cumsum",
    description=(
        "Run Cumsum workload replay on Ascend NPU.\n"
        "The script reads Cumsum.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the input tensor,\n"
        "then runs torch.cumsum(input, 0) for the recorded 1-D rows."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/Cumsum_run.py "
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
