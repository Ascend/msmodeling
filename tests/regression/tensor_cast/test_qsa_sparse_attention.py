# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""Tests for QSA TensorCast ops (qsa_indexer, qsa_sparse_attention) and their
registered analytic performance formulas (Qwen3.8-Flash-Next / qwen4_exp).
"""

import math

import torch

from tensor_cast.performance_model import (
    _estimate_qsa_indexer_breakdown,
    _estimate_qsa_prefill_kv_read_pairs,
    _estimate_qsa_sparse_attention_breakdown,
    _qsa_is_prefill_request,
)
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


def test_qsa_meta_ops_preserve_shapes_and_dtypes():
    idx_q = torch.empty(3, 4, 16, dtype=torch.float16, device="meta")
    idx_k_cache = torch.empty(32, 1, 16, dtype=torch.float16, device="meta")
    seq_lens = torch.tensor([12], dtype=torch.long)
    query_lens = torch.tensor([12], dtype=torch.long)
    block_table = torch.zeros(1, 3, dtype=torch.long)

    topk_idx = torch.ops.tensor_cast.qsa_indexer.default(
        idx_q,
        idx_k_cache,
        seq_lens,
        query_lens,
        block_table,
        topk_blocks=2,
        block_size=1,  # paged cache page size (idx_k_cache.shape[1])
        compress_ratio=4,  # indexer micro-block size
        seq_lens_values=[12],
        query_lens_values=[12],
        is_decode_values=[False],
    )
    # QSA block indices are head-shared: no head axis in the output.
    assert topk_idx.shape == (3, 2)
    assert topk_idx.dtype == torch.int32

    query = torch.empty(3, 32, dtype=torch.float16, device="meta")
    out = torch.ops.tensor_cast.qsa_sparse_attention.default(
        query,
        idx_k_cache,
        idx_k_cache,
        topk_idx,
        seq_lens,
        query_lens,
        block_table,
        hidden_size=32,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=16,
        topk_blocks=2,
        block_size=1,
        compress_ratio=4,
        seq_lens_values=[12],
        query_lens_values=[12],
        is_decode_values=[False],
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype


def test_qsa_is_prefill_request():
    assert _qsa_is_prefill_request(1, 0, [True]) is False
    assert _qsa_is_prefill_request(1, 0, [False]) is True
    assert _qsa_is_prefill_request(4, 0, None) is True
    assert _qsa_is_prefill_request(1, 0, None) is False


def test_qsa_indexer_decode_reads_raw_keys_and_compresses_in_kernel():
    batch_size = 8
    seq_len = 4336  # context 4335 + 1 decode token
    num_index_heads = 4
    head_dim = 128
    topk_blocks = 512
    compress_ratio = 4
    idx_q = torch.empty(batch_size, num_index_heads, head_dim, dtype=torch.float16, device="meta")
    seq_lens = torch.empty(batch_size, dtype=torch.long, device="meta")
    query_lens = torch.empty(batch_size, dtype=torch.long, device="meta")

    breakdown = _estimate_qsa_indexer_breakdown(
        idx_q,
        seq_lens,
        query_lens,
        topk_blocks=topk_blocks,
        compress_ratio=compress_ratio,
        seq_lens_values=[seq_len] * batch_size,
        query_lens_values=[1] * batch_size,
        is_decode_values=[True] * batch_size,
    )

    num_blocks = math.ceil(seq_len / compress_ratio)  # 1084 micro-blocks
    s = idx_q.element_size()
    expected_mma = 2 * batch_size * 1 * num_index_heads * num_blocks * head_dim
    expected_head_reduce = batch_size * 1 * num_index_heads * num_blocks
    # FP32 mean-pool of compress_ratio raw key vectors per micro-block.
    expected_compress_gp = batch_size * num_blocks * compress_ratio * head_dim
    expected_topk_gp = 9 * batch_size * num_blocks  # c_topk = ceil(log2(512)) = 9
    # The cache stores per-token raw keys (official update_indexer
    # semantics): the kernel reads all B_n * c raw vectors and mean-pools.
    expected_k_read_bytes = batch_size * num_blocks * compress_ratio * head_dim * s
    expected_bytes = (
        idx_q.numel() * s
        + expected_k_read_bytes
        + 4 * batch_size * 1 * num_index_heads * num_blocks
        + 4 * batch_size * num_blocks
        + 4 * batch_size * topk_blocks
    )
    assert breakdown == {
        "mma_total": expected_mma,
        "gp_total": expected_head_reduce + expected_compress_gp + expected_topk_gp,
        "bytes_total": expected_bytes,
    }
    # QSA saving vs a token-granularity indexer: the raw K *read* covers every
    # token once, but scoring MMA scales with the compressed block count.
    assert expected_k_read_bytes == batch_size * seq_len * head_dim * s
    assert expected_mma < 2 * batch_size * 1 * num_index_heads * seq_len * head_dim


def test_qsa_indexer_prefill_scores_compressed_blocks():
    query_length = 16
    seq_len = 64
    num_index_heads = 2
    head_dim = 32
    topk_blocks = 8
    compress_ratio = 4
    idx_q = torch.empty(query_length, num_index_heads, head_dim, dtype=torch.float16, device="meta")
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")

    breakdown = _estimate_qsa_indexer_breakdown(
        idx_q,
        seq_lens,
        query_lens,
        topk_blocks=topk_blocks,
        compress_ratio=compress_ratio,
        seq_lens_values=[seq_len],
        query_lens_values=[query_length],
        is_decode_values=[False],
    )

    num_blocks = seq_len // compress_ratio  # 16
    expected_mma = 2 * query_length * num_index_heads * num_blocks * head_dim
    expected_gp = (
        query_length * num_index_heads * num_blocks  # head-mean reduce
        + num_blocks * compress_ratio * head_dim  # FP32 mean-pool compression
        + 3 * query_length * num_blocks  # c_topk = ceil(log2(8)) = 3
    )
    s = idx_q.element_size()
    expected_bytes = (
        idx_q.numel() * s
        + num_blocks * compress_ratio * head_dim * s  # raw per-token key read
        + 4 * query_length * num_index_heads * num_blocks
        + 4 * query_length * num_blocks
        + 4 * query_length * topk_blocks
    )
    assert breakdown == {
        "mma_total": expected_mma,
        "gp_total": expected_gp,
        "bytes_total": expected_bytes,
    }


def test_qsa_sparse_attention_decode_caps_kv_at_budget():
    batch_size = 8
    seq_len = 4336
    num_q_heads = 24
    num_kv_heads = 2
    head_dim = 256
    topk_blocks = 512
    compress_ratio = 4
    query = torch.empty(batch_size, num_q_heads * head_dim, dtype=torch.float16, device="meta")
    seq_lens = torch.empty(batch_size, dtype=torch.long, device="meta")
    query_lens = torch.empty(batch_size, dtype=torch.long, device="meta")

    breakdown = _estimate_qsa_sparse_attention_breakdown(
        query,
        seq_lens,
        query_lens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        topk_blocks=topk_blocks,
        compress_ratio=compress_ratio,
        seq_lens_values=[seq_len] * batch_size,
        query_lens_values=[1] * batch_size,
        is_decode_values=[True] * batch_size,
    )

    attended_tokens = min(seq_len, topk_blocks * compress_ratio)  # budget 2048 < seq
    expected_mma = 4 * batch_size * num_q_heads * attended_tokens * head_dim
    expected_gp = 6 * batch_size * num_q_heads * attended_tokens
    s = query.element_size()
    # Decode KV read is capped by the selected capacity, and topk indices are
    # head-shared (no head scaling on the 4 * T * K term).
    expected_bytes = (
        2 * s * batch_size * num_q_heads * head_dim
        + 2 * s * batch_size * attended_tokens * num_kv_heads * head_dim
        + 4 * batch_size * topk_blocks
    )
    assert breakdown == {
        "mma_total": expected_mma,
        "gp_total": expected_gp,
        "bytes_total": expected_bytes,
    }


def test_qsa_sparse_attention_prefill_kv_pairs_use_selected_capacity_geometric_mean():
    query_length = 16
    context_length = 0
    block_size = 4
    topk_blocks = 4

    attended_pairs = 0
    for query_idx in range(query_length):
        visible_tokens = context_length + query_idx + 1
        visible_blocks = math.ceil(visible_tokens / block_size)
        selected_blocks = min(visible_blocks, topk_blocks)
        attended_pairs += min(visible_tokens, selected_blocks * block_size)

    selected_kv_capacity = topk_blocks * block_size
    assert _estimate_qsa_prefill_kv_read_pairs(
        context_length,
        query_length,
        topk_blocks,
        block_size,
        attended_pairs,
    ) == max(selected_kv_capacity, math.ceil(math.sqrt(selected_kv_capacity * attended_pairs)))


def test_qsa_prefill_kv_read_pairs_capped_by_short_prompt_attended_pairs():
    # Short prompt that has not filled the sparse budget: only 2 tokens with
    # K=512 blocks of 8 tokens means 3 semantic attended pairs; the estimate
    # must never bill more KV than actually exists.
    context_length = 0
    query_length = 2
    topk_blocks = 512
    block_size = 8
    attended_pairs = 3  # 1 (first token) + 2 (second token)

    assert (
        _estimate_qsa_prefill_kv_read_pairs(
            context_length,
            query_length,
            topk_blocks,
            block_size,
            attended_pairs,
        )
        == attended_pairs
    )


def test_qsa_prefill_kv_read_pairs_capped_below_one_block():
    # Prompt shorter than a single compressed block: geometric mean and
    # selected capacity both exceed the attended pairs and must be clamped.
    context_length = 0
    query_length = 3
    topk_blocks = 64
    block_size = 8
    attended_pairs = 6  # 1 + 2 + 3

    assert (
        _estimate_qsa_prefill_kv_read_pairs(
            context_length,
            query_length,
            topk_blocks,
            block_size,
            attended_pairs,
        )
        == attended_pairs
    )


def test_qsa_indexer_perf_properties_use_input_dtype():
    idx_q = torch.empty(1, 4, 128, dtype=torch.bfloat16, device="meta")
    idx_k_cache = torch.empty(1084, 1, 128, dtype=torch.bfloat16, device="meta")
    seq_lens = torch.empty(1, dtype=torch.long, device="meta")
    query_lens = torch.empty(1, dtype=torch.long, device="meta")
    block_table = torch.empty(1, 1, dtype=torch.long, device="meta")
    output = torch.empty(1, 512, dtype=torch.int32, device="meta")

    info = OpInvokeInfo(
        torch.ops.tensor_cast.qsa_indexer.default,
        (idx_q, idx_k_cache, seq_lens, query_lens, block_table),
        {
            "topk_blocks": 512,
            "block_size": 1,
            "compress_ratio": 4,
            "seq_lens_values": [4336],
            "query_lens_values": [1],
            "is_decode_values": [True],
        },
        output,
    )

    properties = info.get_perf_properties()

    assert torch.bfloat16 in properties.compute_ops
    assert torch.float32 not in properties.compute_ops
    assert properties.compute_ops[torch.bfloat16].mma_ops > 0
    assert properties.compute_ops[torch.bfloat16].gp_ops > 0
    assert properties.memory_readwrite_bytes > 0


def test_qsa_sparse_attention_perf_properties_are_registered():
    query = torch.empty(8, 24 * 256, dtype=torch.float16, device="meta")
    key_cache = torch.empty(1084, 4, 2, 256, dtype=torch.float16, device="meta")
    value_cache = torch.empty_like(key_cache)
    topk_idx = torch.empty(8, 512, dtype=torch.int32, device="meta")
    seq_lens = torch.empty(8, dtype=torch.long, device="meta")
    query_lens = torch.empty(8, dtype=torch.long, device="meta")
    block_table = torch.empty(8, 512, dtype=torch.long, device="meta")
    output = torch.empty_like(query)

    info = OpInvokeInfo(
        torch.ops.tensor_cast.qsa_sparse_attention.default,
        (query, key_cache, value_cache, topk_idx, seq_lens, query_lens, block_table),
        {
            "hidden_size": 24 * 256,
            "num_q_heads": 24,
            "num_kv_heads": 2,
            "head_dim": 256,
            "topk_blocks": 512,
            "block_size": 4,
            "compress_ratio": 4,
            "seq_lens_values": [4336] * 8,
            "query_lens_values": [1] * 8,
            "is_decode_values": [True] * 8,
        },
        output,
    )

    properties = info.get_perf_properties()

    assert properties.compute_ops[torch.float16].mma_ops > 0
    assert properties.compute_ops[torch.float16].gp_ops > 0
    assert properties.memory_readwrite_bytes > 0
