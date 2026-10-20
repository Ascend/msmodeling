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
Replay MulAiCore cases from the performance database on Ascend NPU.

Purpose:
  MulAiCore shares the Mul row contract (AI_CORE recording
  variant).  This script reuses the Mul build/run logic with the
  MulAiCore CSV name.
"""

from __future__ import annotations

try:
    from .Mul_run import build_case, run_case
    from .replay_framework import OpReplay
except ImportError:
    from Mul_run import build_case, run_case
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="MulAiCore",
    api_path="torch.mul",
    description=(
        "Run MulAiCore workload replay on Ascend NPU/CUDA/CPU.\n"
        "The script reads MulAiCore.csv under the selected device and\n"
        "vllm_ascend version directory and reuses the Mul replay\n"
        "logic (torch.mul)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/MulAiCore_run.py "
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
