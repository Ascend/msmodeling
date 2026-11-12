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

"""Replay DeepSeek-V4 Compressor rows using the physical custom-op ABI."""

from __future__ import annotations

import math

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape_or_none,
        parse_v4_runtime_workload,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_shape_or_none,
        parse_v4_runtime_workload,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def _slots(row: dict[str, str]):
    shapes = [parse_shape_or_none(x) for x in split_metadata_field(row["Input Shapes"])]
    dtypes = [normalize_dtype_name(x) for x in split_metadata_field(row["Input Data Types"])]
    formats = split_metadata_field(row["Input Formats"])
    if not (len(shapes) == len(dtypes) == len(formats) == 12):
        raise ValueError("Compressor CSV must preserve all 12 physical input slots")
    return shapes, dtypes, formats


def _tensor(shapes, dtypes, formats, index: int):
    shape = shapes[index]
    if shape is None:
        return None
    return build_input_tensor(shape, formats[index], dtypes[index])


def _page_table(shape: tuple[int, ...], block_count: int):
    runtime_torch, _ = get_runtime_modules()
    values = runtime_torch.arange(1, shape[-1] + 1, dtype=runtime_torch.int32)
    values = values.remainder(max(block_count - 1, 1)).add(1)
    return values.reshape((1,) * (len(shape) - 1) + (shape[-1],)).expand(shape).contiguous().npu()


def infer_runtime_workload(row, shapes, dtypes, formats):
    """Read request values and validate the Compressor ABI before allocating tensors."""
    workload = parse_v4_runtime_workload(row)
    if workload is None:
        return None
    query, kv = workload["query_lengths"], workload["kv_lengths"]

    ratio, head = int(workload["compress_ratio"]), int(workload["head_dim"])
    hidden, rope = int(workload["hidden_size"]), int(workload["rope_head_dim"])
    if ratio not in (4, 128) or min(head, hidden, rope) <= 0 or rope > head:
        raise ValueError("v4_compressor_parameters_invalid")
    if shapes[3] is None or len(shapes[3]) != 3 or shapes[8] is None or len(shapes[8]) != 2:
        raise ValueError("v4_compressor_template_invalid")
    blocks, block, _ = shapes[3]
    coff = 2 if ratio == 4 else 1
    batch, width = len(query), coff * head
    if (
        block <= 0 or blocks < 1 + batch * math.ceil(coff * ratio / block)
        or shapes[8][0] != batch or shapes[8][1] < math.ceil(max(kv) / block)
    ):
        raise ValueError("v4_physical_signature_invalid")
    compressed = sum(q // ratio + 1 for q in query)
    expected = [
        (sum(query), hidden), (width, hidden), (width, hidden), (blocks, block, 2 * width),
        (ratio, width), (head,), (compressed, rope), (compressed, rope), shapes[8],
        (batch + 1,), None, (batch,),
    ]
    expected_dtypes = ["DT_BF16"] * 3 + ["DT_FLOAT"] * 2 + ["DT_BF16"] + ["DT_FLOAT"] * 2
    expected_dtypes += ["DT_INT32", "DT_INT32", "DT_UNDEFINED", "DT_INT32"]
    outputs = [parse_shape_or_none(x) for x in split_metadata_field(row["Output Shapes"])]
    if (
        shapes != expected or dtypes != expected_dtypes
        or formats != ["ND" if shape else "NULL" for shape in expected]
        or outputs != [(compressed, head), expected[3]]
        or [normalize_dtype_name(x) for x in split_metadata_field(row["Output Data Types"])]
        != ["DT_BF16", "DT_FLOAT"]
        or split_metadata_field(row["Output Formats"]) != ["ND", "ND"]
    ):
        raise ValueError("v4_physical_signature_invalid")
    return workload


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    shapes, dtypes, formats = _slots(row)
    workload = infer_runtime_workload(row, shapes, dtypes, formats)
    output_shapes = [parse_shape_or_none(x) for x in split_metadata_field(row["Output Shapes"])]
    if not output_shapes or output_shapes[0] is None:
        raise ValueError("Compressor requires its compressed output shape")

    tensors = [_tensor(shapes, dtypes, formats, i) for i in range(8)]
    x, _wkv, _wgate, state_cache, _ape, _norm, _sin, _cos = tensors
    head_dim = int(output_shapes[0][-1])
    compressed_tokens = int(output_shapes[0][0])
    # APE carries the ratio explicitly. Inferring it from output capacity is
    # ambiguous for short inputs (e.g. two tokens give one row for both ratios).
    cmp_ratio = int(_ape.shape[0])
    query_lengths = workload["query_lengths"] if workload else [int(x.shape[0])]
    if cmp_ratio not in (4, 128) or compressed_tokens != sum(q // cmp_ratio + 1 for q in query_lengths):
        raise ValueError("Compressor APE ratio conflicts with compressed output capacity")
    coff = int(state_cache.shape[-1]) // (2 * head_dim)
    if coff not in (1, 2):
        raise ValueError(f"unsupported Compressor coff={coff}")

    state_cache.zero_()
    tensors[4].zero_()
    tensors[5].fill_(1)
    tensors[6].zero_()
    tensors[7].zero_()
    if workload:
        kv_lengths = workload["kv_lengths"]
        pages_per_request = (coff * cmp_ratio + shapes[3][1] - 1) // shapes[3][1]
        offsets = runtime_torch.arange(shapes[8][-1], dtype=runtime_torch.int32) % pages_per_request
        request_bases = runtime_torch.arange(len(query_lengths), dtype=runtime_torch.int32) * pages_per_request + 1
        state_block_table = (request_bases[:, None] + offsets[None, :]).contiguous().npu()
        cu_values = [0]
        for length in query_lengths:
            cu_values.append(cu_values[-1] + length)
        start_values = [s - q for s, q in zip(kv_lengths, query_lengths)]
    else:
        state_block_table = _page_table(shapes[8], int(state_cache.shape[0]))
        cu_values, start_values = [0, int(x.shape[0])], [0]
    cu_seqlens = runtime_torch.tensor(cu_values, dtype=runtime_torch.int32, device="npu")
    start_pos = runtime_torch.tensor(start_values, dtype=runtime_torch.int32, device="npu")

    return {
        "inputs": tensors,
        "kwargs": {
            "state_block_table": state_block_table,
            "cu_seqlens": cu_seqlens,
            "seqused": None,
            "start_pos": start_pos,
            "rope_head_dim": int(tensors[6].shape[-1]),
            "cmp_ratio": cmp_ratio,
            "coff": coff,
            "norm_eps": 1e-6,
            "rotary_mode": 2,
            "cache_mode": 1,
        },
        "api": op.resolve_api(),
    }


op = OpReplay(
    kernel_type="Compressor",
    api_path="torch.ops._C_ascend.compressor",
    exact_runtime_match=True,
    aggregation="mean",
    description="Replay DeepSeek-V4 fused KV compressor kernels.",
    usage_examples=[
        (
            "python tools/perf_data_collection/op_replay/Compressor_run.py "
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
