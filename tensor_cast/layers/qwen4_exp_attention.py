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

"""Qwen3.8-Flash-Next (qwen4_exp) QSA attention wrapper.

The upstream ``Qwen4ExpTextAttention`` runs a python batch/query double loop
inside ``Qwen4ExpTextQSAIndexer`` and mutates an HF ``Cache`` via
``update_indexer`` -- both are compile/meta unfriendly. This wrapper mirrors
the MiniMax-M3 sparse-attention pattern:

  1. q_proj (doubled width: per-head query+gate chunks), k_proj/v_proj,
     official q/k RMSNorm, ``fused_rope`` (partial RoPE), cache write
  2. wrapper-local fused indexer projection (per-rank query heads + shared
     replicated KV head), indexer q RMSNorm, partial RoPE on indexer queries
  3. per-token raw indexer key write via ``siso_reshape_and_cache``
  4. ``qsa_indexer`` for compressed-block scoring and head-shared top-k
  5. ``qsa_sparse_attention`` over the selected micro-blocks
  6. sigmoid output gate and o_proj

The indexer cache follows the official ``update_indexer`` semantics (raw
per-token keys of the shared KV head); block compression happens inside the
``qsa_indexer`` kernel.
"""

import torch


class Qwen4ExpQSAAttentionWrapper(torch.nn.Module):
    """Wrapper for qwen4_exp QSA attention layers (all full_attention layers)."""

    def __init__(
        self,
        original_module: torch.nn.Module,
        hidden_size: int,
        head_dim: int,
        indexer_n_heads: int,
        indexer_kv_heads: int,
        indexer_head_dim: int,
        token_budget: int,
        compress_ratio: int,
        rotary_dim: int,
        indexer_head_start: int = 0,
    ):
        super().__init__()
        self._inner = original_module
        self.is_sparse_layer = True
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_indexer_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_head_start = indexer_head_start
        self.token_budget = token_budget
        self.compress_ratio = compress_ratio
        self.block_topk = token_budget // compress_ratio
        self.rotary_dim = rotary_dim
        # Detection hooks for input_generator's indexer-cache allocation.
        self.indexer_cache_width = indexer_head_dim
        self.index_qk_linear = None

        inner = original_module
        while hasattr(inner, "_inner"):
            inner = inner._inner
        indexer = getattr(inner, "indexer", None)
        index_qk_proj = getattr(indexer, "index_qk_proj", None) if indexer is not None else None
        if index_qk_proj is not None:
            # The fused indexer projection holds [q_heads * D | kv_heads * D]
            # rows. Generic TP sharding would treat it as a plain Linear and
            # slice the shared KV rows apart, so keep the per-rank slice under
            # a wrapper-local name, as the MiniMax-M3 wrapper does.
            q_rows = indexer_n_heads * indexer_head_dim
            kv_rows = indexer_kv_heads * indexer_head_dim
            q_start = indexer_head_start * indexer_head_dim
            total_q_rows = index_qk_proj.out_features - kv_rows
            self.index_qk_linear = torch.nn.Linear(
                index_qk_proj.in_features,
                q_rows + kv_rows,
                bias=index_qk_proj.bias is not None,
            )
            with torch.no_grad():
                self.index_qk_linear.weight[:q_rows].copy_(index_qk_proj.weight[q_start : q_start + q_rows])
                self.index_qk_linear.weight[q_rows:].copy_(index_qk_proj.weight[total_q_rows : total_q_rows + kv_rows])
                if index_qk_proj.bias is not None:
                    self.index_qk_linear.bias[:q_rows].copy_(index_qk_proj.bias[q_start : q_start + q_rows])
                    self.index_qk_linear.bias[q_rows:].copy_(index_qk_proj.bias[total_q_rows : total_q_rows + kv_rows])
            # Keep the official fallback path (attention_meta is None) intact:
            # instead of clearing the projection, rebind the indexer to the
            # wrapper-local copy. Under TP=1 the copy holds the full weight
            # set, so the official HF forward still runs unmodified; under
            # TP>1 the golden/reference path is not used anyway (HF has no
            # per-rank slicing for this fused projection).
            indexer.index_qk_proj = self.index_qk_linear

    def _unwrap(self):
        inner = self._inner
        while hasattr(inner, "_inner"):
            inner = inner._inner
        return inner

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        attention_meta = kwargs.get("attention_meta", None)
        if attention_meta is None:
            # Non-TensorCast execution (e.g. golden tests): keep the official
            # HF forward, which owns the reference QSA semantics.
            return self._inner(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        inner = self._unwrap()
        seq_lens = attention_meta.seq_lens
        query_lens = attention_meta.query_lens
        block_table = attention_meta.block_table_tensor
        seq_lens_values = getattr(attention_meta, "seq_lens_values", None)
        query_lens_values = getattr(attention_meta, "query_lens_values", None)
        is_decode_values = getattr(attention_meta, "is_decode_values", None)

        input_shape = hidden_states.shape[:-1]
        num_tokens = 1
        for dim in input_shape:
            num_tokens *= dim
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Main attention projections. q_proj rows are [num_heads, 2 * head_dim]
        # (interleaved per-head query+gate chunks); COLWISE TP with
        # head_num = num_attention_heads slices whole per-head units, so the
        # chunk below stays aligned on every rank.
        query_and_gate = inner.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(query_and_gate, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = inner.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = inner.k_norm(inner.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = inner.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if hidden_states.ndim == 3:
            actual_num_q_heads = query_states.shape[1]
            actual_num_kv_heads = key_states.shape[1]
        else:
            # Packed [tokens, hidden] input: after transpose(1, 2) the layout is
            # [tokens, head_dim, heads], so the head count is the LAST dim.
            actual_num_q_heads = query_states.shape[-1]
            actual_num_kv_heads = key_states.shape[-1]

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.ndim == 3:
                cos_flat = cos.reshape(num_tokens, cos.shape[-1])
                sin_flat = sin.reshape(num_tokens, sin.shape[-1])
            else:
                cos_flat = cos.reshape(num_tokens, -1)
                sin_flat = sin.reshape(num_tokens, -1)
            cos_sin = torch.cat([cos_flat, sin_flat], dim=-1)
            query_3d = query_states.transpose(1, 2).reshape(num_tokens, actual_num_q_heads, self.head_dim)
            key_3d = key_states.transpose(1, 2).reshape(num_tokens, actual_num_kv_heads, self.head_dim)
            query_3d, key_3d = torch.ops.tensor_cast.fused_rope(query_3d, key_3d, cos_sin, self.rotary_dim, True)
            query_states = query_3d.reshape(*input_shape, actual_num_q_heads, self.head_dim).transpose(1, 2)
            key_states = key_3d.reshape(*input_shape, actual_num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            query_3d = query_states.transpose(1, 2).reshape(num_tokens, actual_num_q_heads, self.head_dim)
            key_3d = key_states.transpose(1, 2).reshape(num_tokens, actual_num_kv_heads, self.head_dim)
        value_3d = value_states.transpose(1, 2).reshape(num_tokens, actual_num_kv_heads, self.head_dim)

        kv_cache_by_layers = kwargs.get("kv_cache_by_layers", None)
        kv_cache = kv_cache_by_layers.get(inner.layer_idx) if kv_cache_by_layers else None
        if kv_cache is not None:
            key_flat = key_3d.reshape(num_tokens, -1)
            value_flat = value_3d.reshape(num_tokens, -1)
            torch.ops.tensor_cast.reshape_and_cache(
                key_flat,
                value_flat,
                kv_cache,
                attention_meta.slot_mapping,
            )

        # Indexer path: fused qk projection -> q RMSNorm + partial RoPE for the
        # per-rank query heads; raw (un-roped, un-normed) keys go to the cache
        # -- compression, k-norm and block-start RoPE happen inside the kernel
        # exactly as in the official indexer.
        idx_hidden_shape = (*input_shape, -1, self.indexer_head_dim)
        idx_qk = self.index_qk_linear(hidden_states)
        idx_q_rows = self.num_indexer_heads * self.indexer_head_dim
        idx_q = idx_qk[..., :idx_q_rows].view(*idx_hidden_shape)
        idx_k = idx_qk[..., idx_q_rows:].view(*input_shape, -1, self.indexer_head_dim)
        idx_q = inner.indexer.q_layernorm(idx_q)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.ndim == 3:
                idx_cos = cos.reshape(num_tokens, cos.shape[-1])
                idx_sin = sin.reshape(num_tokens, sin.shape[-1])
            else:
                idx_cos = cos.reshape(num_tokens, -1)
                idx_sin = sin.reshape(num_tokens, -1)
            idx_cos_sin = torch.cat([idx_cos, idx_sin], dim=-1)
            idx_q_3d = idx_q.reshape(num_tokens, self.num_indexer_heads, self.indexer_head_dim)
            idx_k_3d = idx_k.reshape(num_tokens, self.indexer_kv_heads, self.indexer_head_dim)
            idx_q_3d, _ = torch.ops.tensor_cast.fused_rope(idx_q_3d, idx_k_3d, idx_cos_sin, self.rotary_dim, True)
            idx_q = idx_q_3d.reshape(*input_shape, self.num_indexer_heads, self.indexer_head_dim)
        idx_q_flat = idx_q.reshape(num_tokens, self.num_indexer_heads, self.indexer_head_dim)
        idx_k_flat = idx_k.reshape(num_tokens, -1)

        indexer_cache_by_layers = kwargs.get("indexer_cache_by_layers", None)
        indexer_cache = indexer_cache_by_layers.get(inner.layer_idx) if indexer_cache_by_layers else None
        if indexer_cache is not None:
            torch.ops.tensor_cast.siso_reshape_and_cache(idx_k_flat, indexer_cache, attention_meta.slot_mapping)
            index_keys = indexer_cache
            indexer_page_size = int(indexer_cache.shape[1])
        else:
            index_keys = idx_k_flat
            indexer_page_size = 1

        topk_idx = torch.ops.tensor_cast.qsa_indexer(
            idx_q_flat,
            index_keys,
            seq_lens,
            query_lens,
            block_table,
            topk_blocks=self.block_topk,
            block_size=indexer_page_size,
            compress_ratio=self.compress_ratio,
            seq_lens_values=seq_lens_values,
            query_lens_values=query_lens_values,
            is_decode_values=is_decode_values,
        )

        query_flat = query_3d.reshape(num_tokens, -1)
        if kv_cache is not None:
            key_cache_tensor = kv_cache[0]
            value_cache_tensor = kv_cache[1]
            kv_page_size = int(kv_cache.shape[1])
        else:
            key_cache_tensor = key_3d.reshape(num_tokens, -1)
            value_cache_tensor = value_3d.reshape(num_tokens, -1)
            kv_page_size = 1

        out = torch.ops.tensor_cast.qsa_sparse_attention(
            query_flat,
            key_cache_tensor,
            value_cache_tensor,
            topk_idx,
            seq_lens,
            query_lens,
            block_table,
            hidden_size=self.hidden_size,
            num_q_heads=actual_num_q_heads,
            num_kv_heads=actual_num_kv_heads,
            head_dim=self.head_dim,
            topk_blocks=self.block_topk,
            block_size=kv_page_size,
            compress_ratio=self.compress_ratio,
            seq_lens_values=seq_lens_values,
            query_lens_values=query_lens_values,
            is_decode_values=is_decode_values,
        )

        out = out.reshape(*input_shape, -1).contiguous()
        out = out * torch.sigmoid(gate)
        return inner.o_proj(out), None
