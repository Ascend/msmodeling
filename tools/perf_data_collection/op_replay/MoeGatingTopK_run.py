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

"""Replay fused MoeGatingTopK rows on Ascend NPU."""

from __future__ import annotations

try:
    from .common import parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import parse_list_field, parse_shape
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) != 2:
        raise ValueError("MoeGatingTopK expects logits and expert-bias tensors")
    logits, bias = inputs
    if len(logits.shape) != 2 or len(bias.shape) != 1:
        raise ValueError("MoeGatingTopK requires rank-2 logits and rank-1 bias")
    if logits.shape[-1] != bias.shape[0]:
        raise ValueError("MoeGatingTopK bias width must match the expert count")

    output_shapes = [parse_shape(item) for item in parse_list_field(row.get("Output Shapes", ""))]
    if len(output_shapes) != 3 or len(output_shapes[0]) != 2:
        raise ValueError("MoeGatingTopK requires top-k weights, indices, and norm outputs")
    top_k = output_shapes[0][-1]
    expected = [
        (logits.shape[0], top_k),
        (logits.shape[0], top_k),
        tuple(logits.shape),
    ]
    if output_shapes != expected:
        raise ValueError(
            f"MoeGatingTopK output shapes must be {expected}, got {output_shapes}"
        )
    if not 1 <= top_k <= logits.shape[-1]:
        raise ValueError(f"MoeGatingTopK top-k must be within the expert width, got {top_k}")
    return {
        "inputs": inputs,
        "kwargs": {
            "k": top_k,
            "bias": bias,
            "k_group": 1,
            "group_count": 1,
            "group_select_mode": 0,
            "renorm": 0,
            "norm_type": 1,
            "out_flag": True,
            "routed_scaling_factor": 2.5,
            "eps": 1e-20,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    logits, _bias = case["inputs"]
    return case["api"](logits, **case["kwargs"])


def format_success(csv_path, row_index: int, row: dict[str, str], _case, result) -> str:
    weights, indices, normalized = result
    return (
        f"[OK] {csv_path}:{row_index} shapes={row['Input Shapes']} "
        f"weights={tuple(weights.shape)} indices={tuple(indices.shape)} "
        f"normalized={tuple(normalized.shape)}"
    )


op = OpReplay(
    kernel_type="MoeGatingTopK",
    api_path="torch_npu.npu_moe_gating_top_k",
    description=(
        "Replay fused sigmoid gating and top-k routing on Ascend NPU. "
        "This adapter supports 256 experts, top-k 8, one expert group, "
        "renormalization, and routed scaling factor 2.5."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/MoeGatingTopK_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    input_count=2,
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
