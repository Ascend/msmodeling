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
Replay ClipByValueV2 cases from the performance database on Ascend NPU.

Purpose:
  Read ClipByValueV2 rows from the profiling database, rebuild the
  recorded source tensor (clip min/max are scalar slots), then
  execute torch.clamp(input, min=1, max=1) with matching dtype.
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
        raise ValueError("ClipByValueV2 expects at least one input tensor")
    return case


def run_case(case):
    return case["api"](case["inputs"][0], min=1, max=1)


op = OpReplay(
    kernel_type="ClipByValueV2",
    api_path="torch.clamp",
    description=(
        "Run ClipByValueV2 workload replay on Ascend NPU.\n"
        "The script reads ClipByValueV2.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded input\n"
        "tensor (min/max scalar slots are replayed as constant bounds),\n"
        "then runs torch.clamp(input, min=1, max=1)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/ClipByValueV2_run.py "
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
