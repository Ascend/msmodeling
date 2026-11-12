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

"""Replay both observed DeepSeek-V4 MoeGatingTopKHash input variants."""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape_or_none,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape_or_none,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    shapes = [parse_shape_or_none(x) for x in split_metadata_field(row["Input Shapes"])]
    dtypes = [normalize_dtype_name(x) for x in split_metadata_field(row["Input Data Types"])]
    formats = split_metadata_field(row["Input Formats"])
    output_shapes = [parse_shape_or_none(x) for x in split_metadata_field(row["Output Shapes"])]
    if not (len(shapes) == len(dtypes) == len(formats) == 4):
        raise ValueError("MoeGatingTopKHash must preserve x/bias/input_ids/tid2eid slots")

    def optional(index: int):
        if shapes[index] is None:
            return None
        return build_input_tensor(shapes[index], formats[index], dtypes[index])

    x = optional(0)
    bias = optional(1)
    input_ids = optional(2)
    tid2eid = optional(3)
    if input_ids is not None:
        values = runtime_torch.arange(input_ids.numel(), dtype=runtime_torch.int64, device="npu")
        if tid2eid is not None:
            values = values.remainder(int(tid2eid.shape[0]))
        input_ids.copy_(values.reshape(input_ids.shape))
    if tid2eid is not None:
        expert_count = int(x.shape[-1])
        values = runtime_torch.arange(tid2eid.numel(), dtype=runtime_torch.int32, device="npu")
        tid2eid.copy_(values.remainder(expert_count).reshape(tid2eid.shape))
    k = int(output_shapes[0][-1])
    return {
        "inputs": [x, k],
        "kwargs": {
            "bias": bias,
            "input_ids": input_ids,
            "tid2eid": tid2eid,
            "k_group": 1,
            "group_count": 1,
            "routed_scaling_factor": 1.0,
            "eps": 1e-20,
            "group_select_mode": 1,
            "renorm": 0,
            "norm_type": 2,
            "out_flag": False,
        },
        "api": op.resolve_api(),
    }


op = OpReplay(
    kernel_type="MoeGatingTopKHash",
    api_path="torch.ops._C_ascend.moe_gating_top_k_hash",
    description="Replay DeepSeek-V4 hash-routing top-k kernels.",
    usage_examples=[
        (
            "python tools/perf_data_collection/op_replay/MoeGatingTopKHash_run.py "
            "--database-path <vllm0.23.0 database directory>"
        ),
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    build_case=build_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
