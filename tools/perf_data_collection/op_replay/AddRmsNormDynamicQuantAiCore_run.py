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
Replay AddRmsNormDynamicQuantAiCore cases from the performance database on Ascend NPU.

Purpose:
  AddRmsNormDynamicQuantAiCore shares the AddRmsNormDynamicQuant
  row contract (AI_CORE recording variant).  This script reuses
  the AddRmsNormDynamicQuant build/run logic with the
  AddRmsNormDynamicQuantAiCore CSV name.
"""

from __future__ import annotations

try:
    from .AddRmsNormDynamicQuant_run import build_case, run_case, format_success
    from .replay_framework import OpReplay
except ImportError:
    from AddRmsNormDynamicQuant_run import build_case, run_case, format_success
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="AddRmsNormDynamicQuantAiCore",
    api_path="torch.ops.npu.npu_add_rms_norm_dynamic_quant",
    description=(
        "Run AddRmsNormDynamicQuantAiCore workload replay on Ascend NPU.\n"
        "The script reads AddRmsNormDynamicQuantAiCore.csv under the\n"
        "selected device and vllm_ascend version directory and reuses\n"
        "the AddRmsNormDynamicQuant replay logic\n"
        "(torch.ops.npu.npu_add_rms_norm_dynamic_quant)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/AddRmsNormDynamicQuantAiCore_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
