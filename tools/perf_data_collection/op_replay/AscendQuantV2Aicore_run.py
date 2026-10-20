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
Replay AscendQuantV2Aicore cases from the performance database on Ascend NPU.

Purpose:
  AscendQuantV2Aicore shares the AscendQuantV2 row contract
  (AI_CORE recording variant).  This script reuses the
  AscendQuantV2 build/run logic with the AscendQuantV2Aicore
  CSV name.
"""

from __future__ import annotations

try:
    from .AscendQuantV2_run import build_case, run_case, format_success
    from .replay_framework import OpReplay
except ImportError:
    from AscendQuantV2_run import build_case, run_case, format_success
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="AscendQuantV2Aicore",
    api_path="torch_npu.npu_quantize",
    description=(
        "Run AscendQuantV2Aicore workload replay on Ascend NPU.\n"
        "The script reads AscendQuantV2Aicore.csv under the selected\n"
        "device and vllm_ascend version directory and reuses the\n"
        "AscendQuantV2 replay logic (torch_npu.npu_quantize)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/AscendQuantV2Aicore_run.py "
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
