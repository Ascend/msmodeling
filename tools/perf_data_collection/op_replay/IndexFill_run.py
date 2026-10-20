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
Replay IndexFill cases from the performance database on Ascend NPU.

Purpose:
  Read IndexFill rows from the profiling database, rebuild the
  recorded source tensor (index/value are scalar slots whose values
  are not recorded), then execute torch.index_fill(input, 0, index).
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
    if not inputs:
        raise ValueError("IndexFill expects at least one input tensor")
    index = runtime_torch.tensor([0], dtype=runtime_torch.int64).npu()
    return {
        "inputs": [inputs[0], index],
        "kwargs": {"value": 1},
        "api": op.resolve_api(),
    }


def run_case(case):
    tensor, index = case["inputs"]
    return case["api"](tensor, 0, index, case["kwargs"]["value"])


op = OpReplay(
    kernel_type="IndexFill",
    api_path="torch.index_fill",
    description=(
        "Run IndexFill workload replay on Ascend NPU.\n"
        "The script reads IndexFill.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the input tensor,\n"
        "then runs torch.index_fill(input, 0, index, value) with a\n"
        "single recorded index/value pair."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/IndexFill_run.py "
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
