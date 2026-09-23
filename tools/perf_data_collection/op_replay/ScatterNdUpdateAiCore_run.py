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

"""Replay the AiCore variant of ScatterNdUpdate on Ascend NPU."""

from __future__ import annotations

try:
    from .ScatterNdUpdate_run import build_scatter_case, format_success
    from .replay_framework import OpReplay
except ImportError:
    from ScatterNdUpdate_run import build_scatter_case, format_success
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    return build_scatter_case(op, row)


op = OpReplay(
    kernel_type="ScatterNdUpdateAiCore",
    api_path="torch_npu.npu_scatter_nd_update",
    description=("Replay ScatterNdUpdateAiCore.csv through torch_npu.npu_scatter_nd_update on Ascend NPU."),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/ScatterNdUpdateAiCore_run.py --database-path /path/to/database"
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
