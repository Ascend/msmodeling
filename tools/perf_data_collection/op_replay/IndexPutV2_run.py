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
Replay IndexPutV2 cases from the performance database on Ascend NPU.

Purpose:
  Read IndexPutV2 rows from the profiling database, rebuild the
  recorded (input, indices, values) tensors (the scalar metadata
  slots are ignored), then execute torch.index_put(input, (indices,),
  values).
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
    if len(inputs) != 5:
        raise ValueError("IndexPutV2 expects five recorded input slots")
    tensor = inputs[0]
    indices = inputs[1].to(runtime_torch.int64)
    values = inputs[4].to(tensor.dtype)
    return {
        "inputs": [tensor, indices, values],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def run_case(case):
    tensor, indices, values = case["inputs"]
    return case["api"](tensor, (indices,), values)


op = OpReplay(
    kernel_type="IndexPutV2",
    api_path="torch.index_put",
    description=(
        "Run IndexPutV2 workload replay on Ascend NPU.\n"
        "The script reads IndexPutV2.csv under the selected device\n"
        "and vllm_ascend version directory, rebuilds the\n"
        "(input, indices, values) tensors, then runs\n"
        "torch.index_put(input, (indices,), values)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/IndexPutV2_run.py "
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
