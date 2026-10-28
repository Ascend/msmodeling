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


"""Analytic Step5 costs derived from StepFun's region selector and GQA math.

The sparse selector ranks regions of eight tokens, not individual tokens.
Costs count causal pairs for each packed request, including cached prefixes.
Index projections/norms/RoPE, gates and KV writes are separate runtime ops.
"""

import math

import torch

from ..op_invoke_info import OpInvokeInfo


def capped_causal_pairs(seq_len, query_len, cap):
    start = seq_len - query_len + 1
    uncapped_end = min(seq_len, cap)
    count = max(0, uncapped_end - start + 1)
    return count * (start + uncapped_end) // 2 + (query_len - count) * cap


def visible_region_sum(length, region_size):
    complete, tail = divmod(length, region_size)
    return region_size * complete * (complete + 1) // 2 + tail * (complete + 1)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.step5_attention.default)
def step5_attention_properties(info):
    query, cache, selected, seqs, queries, window, topk, region = info.args
    cap = window if window else (topk * region if selected is not None else max(seqs))
    pairs = sum(capped_causal_pairs(s, q, cap) for s, q in zip(seqs, queries))
    heads, dim = query.shape[-2:]
    kv_heads = cache.shape[-2]
    props = info.get_memory_access_properties(exclude_input_ids={1})
    props.compute_ops[query.dtype] = OpInvokeInfo.ComputeOps(mma_ops=4 * pairs * heads * dim)
    softmax = props.compute_ops.setdefault(torch.float32, OpInvokeInfo.ComputeOps())
    softmax.gp_ops += 5 * pairs * heads
    # Per-query KV traffic is a conservative gather bound, not a calibrated
    # accelerator tiling estimate; do not charge the whole allocated cache.
    props.memory_read_bytes += 2 * pairs * kv_heads * dim * cache.element_size()
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.step5_indexer.default)
def step5_indexer_properties(info):
    query, key, z, weights, cache, seqs, queries, topk, region, groups = info.args
    heads, dim = query.shape[-2:]
    # The selector scores complete regions and the current partial region.
    region_pairs = sum(visible_region_sum(s, region) - visible_region_sum(s - q, region) for s, q in zip(seqs, queries))
    tokens = sum(queries)
    props = info.get_memory_access_properties(exclude_input_ids={4})
    # The reference rounds activations to E4M3, then widens for BF16 score GEMM.
    props.compute_ops[query.dtype] = OpInvokeInfo.ComputeOps(mma_ops=2 * region_pairs * heads * dim)
    selector = props.compute_ops.setdefault(torch.float32, OpInvokeInfo.ComputeOps())
    # topk <= 1 selects a single region; floor the log2 operand to keep topk=0 valid.
    log2_topk = max(1, math.ceil(math.log2(max(topk, 2))))
    selector.gp_ops += (
        # CSA softmax-weighted sum over K/Z, weighted ReLU scores and selection.
        8 * tokens * dim
        + 3 * region_pairs * heads
        + sum(q * groups * math.ceil(s / region) * log2_topk for s, q in zip(seqs, queries))
    )
    props.memory_read_bytes += region_pairs * dim
    # Pending K/Z read+write and FP8 summary updates, not a full-cache rewrite.
    props.memory_readwrite_bytes += 8 * tokens * dim
    props.memory_write_bytes += sum((s // region - (s - q) // region) * dim for s, q in zip(seqs, queries))
    return props
