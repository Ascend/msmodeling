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
Replay GatherV3 cases from the performance database on Ascend NPU.

Purpose:
  Read GatherV3 rows from the profiling database, rebuild the
  recorded (input, indices) tensors (the scalar dim slot is
  recorded as a one-element tensor whose value is unknown; all
  recorded rows gather along dim 0), then execute
  torch.index_select(input, 0, indices).
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) < 2:
        raise ValueError("GatherV3 expects input and indices tensors")
    return {
        "inputs": [inputs[0], inputs[1]],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def run_case(case):
    tensor, indices = case["inputs"]
    return case["api"](tensor, 0, indices)


op = OpReplay(
    kernel_type="GatherV3",
    api_path="torch.index_select",
    description=(
        "Run GatherV3 workload replay on Ascend NPU.\n"
        "The script reads GatherV3.csv under the selected device and\n"
        "vllm_ascend version directory, rebuilds the (input, indices)\n"
        "tensors, then runs torch.index_select(input, 0, indices)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/GatherV3_run.py "
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
