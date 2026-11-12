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

"""Replay the DeepSeek-V4 in-place partial RoPE custom kernel."""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    x, cos, sin = op.build_inputs(row)
    rotary_width = int(cos.shape[-1])
    head_dim = int(x.shape[-1])
    if sin.shape != cos.shape or rotary_width > head_dim:
        raise ValueError(
            f"invalid partial RoPE shapes: x={tuple(x.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )
    return {
        "inputs": [x, cos, sin],
        "kwargs": {
            "rotary_mode": "interleave",
            "partial_slice": [head_dim - rotary_width, head_dim],
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    case["api"](*case["inputs"], **case["kwargs"])
    return case["inputs"][0]


op = OpReplay(
    kernel_type="InplacePartialRotaryMul",
    api_path="torch.ops._C_ascend.inplace_partial_rotary_mul",
    description="Replay DeepSeek-V4 in-place partial rotary multiplication.",
    usage_examples=[
        (
            "python tools/perf_data_collection/op_replay/InplacePartialRotaryMul_run.py "
            "--database-path <vllm0.23.0 database directory>"
        ),
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    input_count=3,
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
