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
Replay ArgMaxV2 cases from the performance database on Ascend NPU.

Purpose:
  Read ArgMaxV2 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/ArgMaxV2.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.argmax() along the last dimension.
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay

op = OpReplay(
    kernel_type="ArgMaxV2",
    api_path="torch.argmax",
    description=(
        "Run ArgMaxV2 workload replay on Ascend NPU.\n"
        "The script reads ArgMaxV2.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch.argmax(input, dim=-1)."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/ArgMaxV2_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.15.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.15.0.",
    input_count=1,
    fixed_kwargs={"dim": -1},
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
