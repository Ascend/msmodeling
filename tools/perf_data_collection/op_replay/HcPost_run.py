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

"""Replay DeepSeek-V4 HcPost rows with ``npu_hc_post``."""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="HcPost",
    api_path="torch.ops._C_ascend.npu_hc_post",
    description="Replay DeepSeek-V4 fused Hyper-Connection post kernels.",
    usage_examples=[
        ("python tools/perf_data_collection/op_replay/HcPost_run.py --database-path <vllm0.23.0 database directory>"),
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    input_count=4,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
