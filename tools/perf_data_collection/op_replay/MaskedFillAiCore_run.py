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
Replay MaskedFillAiCore cases from the performance database on Ascend NPU.

Purpose:
  MaskedFillAiCore shares the MaskedFill row contract (AI_CORE
  recording variant; the value slot is a scalar).  This script
  reuses the MaskedFill masked_fill_ logic with the
  MaskedFillAiCore CSV name.
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def run_case(case):
    tensor, mask_tensor = case["inputs"][:2]
    tensor.masked_fill_(mask_tensor, 0)
    return tensor


op = OpReplay(
    kernel_type="MaskedFillAiCore",
    description=(
        "Run MaskedFillAiCore workload replay on Ascend NPU/CUDA/CPU.\n"
        "The script reads MaskedFillAiCore.csv under the selected\n"
        "device and vllm_ascend version directory, reconstructs the\n"
        "input and mask tensors, then runs tensor.masked_fill_(mask, 0)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/MaskedFillAiCore_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    input_count=2,
    input_dtype_overrides={1: "DT_BOOL"},
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
