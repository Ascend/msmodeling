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

"""Replay SliceAiCore cases through the same torch_npu Slice API."""

from __future__ import annotations

try:
    from .Slice_run import build_slice_case, format_success, run_case
    from .replay_framework import OpReplay
except ImportError:
    from Slice_run import build_slice_case, format_success, run_case
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    return build_slice_case(op, row)


op = OpReplay(
    kernel_type="SliceAiCore",
    api_path="torch_npu.npu_slice",
    description="Replay SliceAiCore.csv through torch_npu.npu_slice on Ascend NPU.",
    usage_examples=[
        "python tools/perf_data_collection/op_replay/SliceAiCore_run.py --database-path /path/to/database"
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
