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
Replay LayerNormV3WithImplMode cases from the performance database on Ascend NPU.

Purpose:
  LayerNormV3WithImplMode shares the LayerNormV3 row contract
  (impl-mode recording variant).  This script reuses the
  LayerNormV3 build/run logic with the LayerNormV3WithImplMode
  CSV name.
"""

from __future__ import annotations

try:
    from .LayerNormV3_run import build_case, run_case
    from .replay_framework import OpReplay
except ImportError:
    from LayerNormV3_run import build_case, run_case
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="LayerNormV3WithImplMode",
    api_path="torch.native_layer_norm",
    description=(
        "Run LayerNormV3WithImplMode workload replay on Ascend NPU.\n"
        "The script reads LayerNormV3WithImplMode.csv under the\n"
        "selected device and vllm_ascend version directory and\n"
        "reuses the LayerNormV3 replay logic\n"
        "(torch.native_layer_norm)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/LayerNormV3WithImplMode_run.py "
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
