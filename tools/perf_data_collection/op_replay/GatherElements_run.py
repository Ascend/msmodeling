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
Replay GatherElements cases from the performance database on Ascend NPU.

Purpose:
  GatherElements shares the GatherElementsV2 row contract.  This
  script reuses the GatherElementsV2 build/run logic with the
  GatherElements CSV name.
"""

from __future__ import annotations

try:
    from .GatherElementsV2_run import build_case, run_case
    from .replay_framework import OpReplay
except ImportError:
    from GatherElementsV2_run import build_case, run_case
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="GatherElements",
    api_path="torch.gather",
    description=(
        "Run GatherElements workload replay on Ascend NPU.\n"
        "The script reads GatherElements.csv under the selected\n"
        "device and vllm_ascend version directory, rebuilds the\n"
        "(input, index) tensors, then runs torch.gather(input, -1, index)."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/GatherElements_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.23.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
