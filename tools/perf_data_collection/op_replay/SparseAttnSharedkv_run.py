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

"""Replay DeepSeek-V4 shared-KV sparse attention rows."""

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
    if not (len(shapes) == len(dtypes) == len(formats) == 14):
        raise ValueError("SparseAttnSharedkv CSV must preserve all 14 physical slots")
    return shapes, dtypes, formats


def _build_optional(shapes, dtypes, formats, index: int):
    if shapes[index] is None:
        return None
    return build_input_tensor(shapes[index], formats[index], dtypes[index])


def _block_table(shape: tuple[int, ...], page_count: int, lengths=None):
    runtime_torch, _ = get_runtime_modules()
    if lengths is not None:
        table = runtime_torch.zeros(shape, dtype=runtime_torch.int32)
        cursor = 0
        for request, length in enumerate(lengths):
            pages = (length + 127) // 128
            if pages > shape[-1] or cursor + pages > page_count:
                raise ValueError("SparseAttnSharedkv request pages exceed physical cache capacity")
            table[request, :pages] = runtime_torch.arange(cursor, cursor + pages, dtype=runtime_torch.int32)
            cursor += pages
        return table.contiguous().npu()
    values = runtime_torch.arange(shape[-1], dtype=runtime_torch.int32)
    values = values.remainder(max(page_count, 1))
    return values.reshape((1,) * (len(shape) - 1) + (shape[-1],)).expand(shape).contiguous().npu()


def _topk_indices(shape: tuple[int, ...], limit: int, query_lengths=None, kv_lengths=None):
    runtime_torch, _ = get_runtime_modules()
    if query_lengths is not None:
        chunks = []
        for query, length in zip(query_lengths, kv_lengths):
            values = runtime_torch.arange(shape[-1], dtype=runtime_torch.int32).remainder(max(length, 1))
            if length == 0:
                values.fill_(-1)
            chunks.append(values.reshape(1, 1, -1).expand(query, shape[1], shape[-1]))
        return runtime_torch.cat(chunks, dim=0).contiguous().npu()
    values = runtime_torch.arange(shape[-1], dtype=runtime_torch.int32).remainder(max(limit, 1))
    return values.reshape((1,) * (len(shape) - 1) + (shape[-1],)).expand(shape).contiguous().npu()


def infer_runtime_workload(row, shapes, dtypes, formats):
    """Read request values and validate the shared-KV ABI without query-side imports."""
    workload = parse_v4_runtime_workload(row)
    if workload is None:
        return None
    query, kv = workload["query_lengths"], workload["kv_lengths"]

    ratio, head = int(workload["compress_ratio"]), int(workload["head_dim"])
    heads, topk = int(workload["num_heads"]), int(workload["compress_topk"])
    if ratio not in (0, 4, 128) or workload.get("window_size") != 128 or min(head, heads) <= 0:
        raise ValueError("v4_sparse_parameters_missing_or_unsupported")
    if (
        (ratio == 4 and not 0 < topk <= max(1, max(kv) // ratio))
        or (ratio == 128 and topk != max(1, max(kv) // ratio))
        or (ratio == 0 and topk != 0)
    ):
        raise ValueError("v4_sparse_topk_invalid")
    if (
        shapes[1] is None
        or len(shapes[1]) != 4
        or shapes[1][1:] != (128, 1, head)
        or shapes[1][0] < sum(math.ceil(s / 128) for s in kv)
    ):
        raise ValueError("v4_physical_signature_invalid")
    batch, n = len(query), sum(query)
    for slot, capacity in ((5, math.ceil(max(kv) / 128)), (6, math.ceil((max(kv) // ratio + 1) / 128) if ratio else 0)):
        if slot == 6 and not ratio:
            continue
        if shapes[slot] is None or len(shapes[slot]) != 2 or shapes[slot][0] != batch or shapes[slot][1] < capacity:
            raise ValueError("v4_physical_signature_invalid")
    single_prefill = workload["phase"] == "prefill" and batch == 1
    expected = [
        (n, heads, head),
        shapes[1],
        shapes[1] if ratio else None,
        None,
        (n, 1, topk) if ratio == 4 else None,
        shapes[5],
        shapes[6] if ratio else None,
        (batch + 1,),
        (batch + 1,),
        (batch + 1,) if ratio and not single_prefill else None,
        (batch,) if not single_prefill else None,
        (batch,),
        (heads,),
        (1024,),
    ]
    expected_dtypes = [
        "DT_UNDEFINED" if not shape else "DT_BF16" if i < 3 else "DT_FLOAT" if i == 12 else "DT_INT32"
        for i, shape in enumerate(expected)
    ]
    outputs = [parse_shape_or_none(x) for x in split_metadata_field(row["Output Shapes"])]
    if (
        shapes != expected
        or dtypes != expected_dtypes
        or formats != ["ND" if shape else "NULL" for shape in expected]
        or outputs != [(n, heads, head), None]
        or [normalize_dtype_name(x) for x in split_metadata_field(row["Output Data Types"])] != ["DT_BF16", "DT_FLOAT"]
        or split_metadata_field(row["Output Formats"]) != ["ND", "ND"]
    ):
        raise ValueError("v4_physical_signature_invalid")
    return workload


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    shapes, dtypes, formats = _slots(row)
    workload = infer_runtime_workload(row, shapes, dtypes, formats)
    q = _build_optional(shapes, dtypes, formats, 0)
    ori_kv = _build_optional(shapes, dtypes, formats, 1)
    cmp_kv = _build_optional(shapes, dtypes, formats, 2)
    query_tokens = int(q.shape[0])
    cmp_ratio = 1
    if cmp_kv is not None:
        cmp_ratio = 4 if shapes[4] is not None else 128
    cmp_tokens = query_tokens // cmp_ratio + 1
    if workload:
        query_lengths, kv_lengths = workload["query_lengths"], workload["kv_lengths"]
        compressed_lengths = [length // cmp_ratio for length in kv_lengths]
        ori_sparse_indices = (
            None
            if shapes[3] is None
            else _topk_indices(
                shapes[3],
                query_tokens,
                query_lengths,
                kv_lengths,
            )
        )
        cmp_sparse_indices = (
            None
            if shapes[4] is None
            else _topk_indices(
                shapes[4],
                cmp_tokens,
                query_lengths,
                compressed_lengths,
            )
        )
        ori_block_table = _block_table(shapes[5], int(ori_kv.shape[0]), kv_lengths)
        cmp_block_table = (
            None
            if shapes[6] is None
            else _block_table(
                shapes[6],
                int(cmp_kv.shape[0]),
                compressed_lengths,
            )
        )

        def length_tensor(slot, values, cumulative=False):
            if shapes[slot] is None:
                return None
            if cumulative:
                offsets = [0]
                for length in values:
                    offsets.append(offsets[-1] + length)
                values = offsets
            return runtime_torch.tensor(values, dtype=runtime_torch.int32, device="npu")

        cu_q = length_tensor(7, query_lengths, True)
        cu_ori = length_tensor(8, kv_lengths, True)
        cu_cmp = length_tensor(9, compressed_lengths, True)
        seqused_q = length_tensor(10, query_lengths)
        seqused_kv = length_tensor(11, kv_lengths)
    else:
        query_lengths, kv_lengths = [query_tokens], [query_tokens]

        ori_sparse_indices = None if shapes[3] is None else _topk_indices(shapes[3], query_tokens)
        cmp_sparse_indices = None if shapes[4] is None else _topk_indices(shapes[4], cmp_tokens)
        ori_block_table = None if shapes[5] is None else _block_table(shapes[5], int(ori_kv.shape[0]))
        cmp_block_table = None if shapes[6] is None else _block_table(shapes[6], int(cmp_kv.shape[0]))
        cu_q = (
            None
            if shapes[7] is None
            else runtime_torch.tensor([0, query_tokens], dtype=runtime_torch.int32, device="npu")
        )
        cu_ori = (
            None
            if shapes[8] is None
            else runtime_torch.tensor([0, query_tokens], dtype=runtime_torch.int32, device="npu")
        )
        cu_cmp = (
            None
            if shapes[9] is None
            else runtime_torch.tensor([0, cmp_tokens], dtype=runtime_torch.int32, device="npu")
        )
        seqused_q = (
            None
            if shapes[10] is None
            else runtime_torch.tensor([query_tokens], dtype=runtime_torch.int32, device="npu")
        )
        seqused_kv = (
            None
            if shapes[11] is None
            else runtime_torch.tensor([query_tokens], dtype=runtime_torch.int32, device="npu")
        )
    sinks = _build_optional(shapes, dtypes, formats, 12)
    if sinks is None:
        raise ValueError("SparseAttnSharedkv requires the sinks slot for replay")
    sinks.zero_()

    metadata_api = runtime_torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
    metadata = metadata_api(
        num_heads_q=int(q.shape[1]),
        num_heads_kv=int(ori_kv.shape[-2]),
        head_dim=int(q.shape[-1]),
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=len(query_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_kv=max(kv_lengths),
        ori_topk=0 if ori_sparse_indices is None else int(ori_sparse_indices.shape[-1]),
        cmp_topk=0 if cmp_sparse_indices is None else int(cmp_sparse_indices.shape[-1]),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=ori_kv is not None,
        has_cmp_kv=cmp_kv is not None,
        device="npu",
    )
    return {
        "inputs": [q],
        "kwargs": {
            "ori_kv": ori_kv,
            "cmp_kv": cmp_kv,
            "ori_sparse_indices": ori_sparse_indices,
            "cmp_sparse_indices": cmp_sparse_indices,
            "ori_block_table": ori_block_table,
            "cmp_block_table": cmp_block_table,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_ori_kv": cu_ori,
            "cu_seqlens_cmp_kv": cu_cmp,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "sinks": sinks,
            "metadata": metadata,
            "softmax_scale": 1.0 / math.sqrt(int(q.shape[-1])),
            "cmp_ratio": cmp_ratio,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": 127,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
            "return_softmax_lse": False,
        },
        "api": op.resolve_api(),
    }


op = OpReplay(
    kernel_type="SparseAttnSharedkv",
    api_path="torch.ops._C_ascend.npu_sparse_attn_sharedkv",
    exact_runtime_match=True,
    aggregation="mean",
    description="Replay DeepSeek-V4 shared-KV sparse attention kernels.",
    usage_examples=[
        (
            "python tools/perf_data_collection/op_replay/SparseAttnSharedkv_run.py "
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
