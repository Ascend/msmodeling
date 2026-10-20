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
Replay PadV3AiCore cases from the performance database on Ascend NPU.

Purpose:
  PadV3AiCore shares the PadV3 row contract (AI_CORE recording
  variant; the constant-value slot is a scalar).  The paddings are
  inferred from the recorded input/output shape delta, then
  torch.nn.functional.pad() is executed.
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])
    if not input_shapes:
        raise ValueError("PadV3AiCore requires at least one recorded input tensor")

    input_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0] if input_formats else "ND",
        dtype_name=normalize_dtype_name(input_dtypes[0] if input_dtypes else "DT_FLOAT"),
    )

    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not output_shapes:
        raise ValueError("PadV3AiCore requires at least one recorded output shape")

    paddings: list[int] = []
    for input_dim, output_dim in zip(reversed(tuple(input_tensor.shape)), reversed(output_shapes[0])):
        paddings.extend([0, output_dim - input_dim])

    return {
        "inputs": [input_tensor],
        "kwargs": {"pad": paddings, "mode": "constant", "value": 0.0},
        "api": op.resolve_api(),
    }


def run_case(case):
    return case["api"](case["inputs"][0], **case["kwargs"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"paddings={case['kwargs']['pad']}"
    )


op = OpReplay(
    kernel_type="PadV3AiCore",
    api_path="torch.nn.functional.pad",
    description=(
        "Run PadV3AiCore workload replay on Ascend NPU.\n"
        "The script reads PadV3AiCore.csv under the selected device\n"
        "and vllm_ascend version directory, reconstructs the input\n"
        "tensor, infers paddings from the recorded input/output\n"
        "shapes, then runs torch.nn.functional.pad()."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/PadV3AiCore_run.py "
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
