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
Replay GatherElementsV2 cases from the performance database on Ascend NPU.

Purpose:
  Read GatherElementsV2 rows from the profiling database, rebuild
  the recorded (input, index) tensors, then execute
  torch.gather(input, dim=-1, index).
"""

from __future__ import annotations

try:
    from .common import get_runtime_modules
    from .replay_framework import OpReplay
except ImportError:
    from common import get_runtime_modules
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    inputs = op.build_inputs(row)
    if len(inputs) != 2:
        raise ValueError("GatherElementsV2 expects input and index tensors")
    source, index = inputs
    # torch.gather requires an int64 index; convert once at case build
    # time so the timed run only executes the op itself.
    if index.dtype != runtime_torch.int64:
        index = index.to(runtime_torch.int64)
    return {
        "inputs": [source, index],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def run_case(case):
    source, index = case["inputs"]
    return case["api"](source, -1, index)


op = OpReplay(
    kernel_type="GatherElementsV2",
    api_path="torch.gather",
    description=(
        "Run GatherElementsV2 workload replay on Ascend NPU.\n"
        "The script reads GatherElementsV2.csv under the selected\n"
        "device and vllm_ascend version directory, rebuilds the\n"
        "(input, index) tensors, then runs torch.gather(input, -1, index)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/GatherElementsV2_run.py "
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
