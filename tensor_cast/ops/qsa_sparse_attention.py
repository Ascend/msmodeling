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

from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("qsa_indexer")
def _(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    compress_ratio: int,
    seq_lens_values: Optional[list[int]] = None,
    query_lens_values: Optional[list[int]] = None,
    is_decode_values: Optional[list[bool]] = None,
) -> torch.Tensor:
    """
    QSA (Qwen Sparse Attention) micro-block indexer fused op.

    Boundary:
      compressed-block score (FP32 mean-pool of ``compress_ratio`` tokens,
      indexer k-norm, block-start RoPE) -> head-shared top-k block indices.

    Indexer Q/K projections, indexer q RMSNorm and the per-query partial RoPE
    are explicit ops in the model wrapper. ``idx_k_cache`` follows the official
    ``past_key_values.update_indexer`` semantics: it stores *per-token raw*
    keys of the shared KV head (one vector per token, written via
    ``siso_reshape_and_cache``), and this op compresses visible tokens into
    micro-blocks of ``compress_ratio`` before scoring. ``block_size`` is the
    paged KV block size used to traverse the cache via ``block_table``.

    Unlike the MiniMax-M3 indexer, QSA reduces scores over the indexer heads
    before top-k, so the selected block indices are shared across heads and
    the output has no head axis. Only *complete* micro-blocks compete for the
    budget; the frontier partial block is handled by ``qsa_sparse_attention``.

    Performance formula: see _estimate_qsa_indexer_breakdown in
    tensor_cast/performance_model/__init__.py.
    """
    total_tokens = idx_q.shape[0]
    return torch.empty(
        (total_tokens, topk_blocks),
        dtype=torch.int32,
        device="meta",
    )


@register_tensor_cast_op("qsa_sparse_attention")
def _(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    topk_blocks: int,
    block_size: int,
    compress_ratio: int,
    seq_lens_values: Optional[list[int]] = None,
    query_lens_values: Optional[list[int]] = None,
    is_decode_values: Optional[list[bool]] = None,
) -> torch.Tensor:
    """
    QSA sparse attention fused op (Qwen3.8-Flash-Next QSA layers).

    Boundary:
      sparse QK/PV attention body over the top-k selected micro-blocks plus
      the always-visible frontier partial block.

    Main q/k/v projection, partial RoPE, cache write, output sigmoid gate and
    o_proj are explicit ops in the model wrapper. ``block_size`` is the paged
    KV block size used to traverse ``key_cache``/``value_cache`` via
    ``block_table``; each selected micro-block gathers ``compress_ratio``
    consecutive tokens (the indexer compress ratio).
    """
    return torch.empty_like(query).contiguous()
