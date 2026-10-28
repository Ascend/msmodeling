# Copyright 2026 The StepFun and HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Transformers v5.16.1 models/step3p7/modeling_step3p7.py.
# Modified for Step5 TensorCast simulation: sparse attention, cache accounting,
# routing, residual precision and shape-only vision execution.
# A copy of the Apache license is in
# tensor_cast/transformers/builtin_model/licenses/LICENSE.


"""Step5 projection, cache, indexer and gated-attention simulation path."""

import copy
import math

import torch
from transformers.models.step3p7.modeling_step3p7 import (
    Step3p7Attention,
    Step3p7RMSNorm,
    Step3p7RotaryEmbedding,
    Step3p7VisionAttention,
    apply_rotary_pos_emb_vision,
)

from .parallel_linear import ColumnParallelLinear


class Step5Attention(Step3p7Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config.per_layer_config[layer_idx], layer_idx)
        self.config = config
        sparse = config.sparse_config or {}
        self.is_sparse_layer = bool(sparse.get("enabled", False)) and config.layer_types[layer_idx] in sparse.get(
            "apply_to_layer_types", ["full_attention"]
        )
        self.proxy_dim = sparse.get("proxy_dim", 256)
        self.region_size = sparse.get("region_block_size", 8)
        self.topk = sparse.get("topk", 512)
        self.index_heads = sparse.get("sparse_indexer_num_heads", 16)
        self.provider_groups = sparse.get("num_provider_groups", config.num_key_value_heads)
        self.index_rope_dim = sparse.get("sparse_indexer_rope_dim", 32)
        if self.is_sparse_layer:
            if self.index_heads % self.provider_groups or self.provider_groups != config.num_key_value_heads:
                raise ValueError("Step5 indexer heads must divide into the GQA provider groups")
            self.index_query = torch.nn.Linear(config.hidden_size, self.index_heads * self.proxy_dim, bias=False)
            self.index_key = torch.nn.Linear(config.hidden_size, self.proxy_dim, bias=False)
            self.index_z = torch.nn.Linear(config.hidden_size, self.proxy_dim, bias=False)
            self.index_weights = torch.nn.Linear(config.hidden_size, self.index_heads, bias=False)
            self.index_q_norm = Step3p7RMSNorm(self.proxy_dim, eps=config.rms_norm_eps)
            self.index_k_norm = torch.nn.LayerNorm(self.proxy_dim, eps=config.rms_norm_eps)
            rope_config = copy.deepcopy(config)
            for params in rope_config.rope_parameters.values():
                params["partial_rotary_factor"] = self.index_rope_dim / config.head_dim
            self.index_rotary_emb = Step3p7RotaryEmbedding(rope_config)

    def configure_indexer_tp(self, group):
        if not self.is_sparse_layer:
            return
        # Provider groups replicate when TP exceeds KV heads, as does the KV cache.
        for name in ("index_query", "index_weights"):
            setattr(
                self,
                name,
                ColumnParallelLinear(
                    getattr(self, name), group, group, head_num=self.provider_groups, is_replicable=True
                ),
            )

    def allocate_indexer_cache(self, num_blocks, block_size, batch_size, total_kv_tokens):
        capacity = num_blocks * block_size
        batch = batch_size or 1
        regions = math.ceil(capacity / self.region_size)
        # Persistent FP8 region summaries, BF16 pending K/Z, int32 lengths.
        size = regions * self.proxy_dim + batch * (4 * self.region_size * self.proxy_dim + 4)
        cache = torch.empty(size, device="meta", dtype=torch.uint8)
        # The normalized span keeps per-token accounting independent of the flat layout.
        return cache, num_blocks, block_size

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        metadata = kwargs.get("attention_meta")
        if metadata is None:
            raise ValueError("Step5 requires TensorCast attention metadata (simulation only)")
        shape = hidden_states.shape[:-1]
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        query = self.q_norm(self.q_proj(hidden_states).reshape(tokens, -1, self.head_dim))
        key = self.k_norm(self.k_proj(hidden_states).reshape(tokens, -1, self.head_dim))
        value = self.v_proj(hidden_states).reshape_as(key)
        cos, sin = position_embeddings
        rotary_dim = cos.shape[-1]
        cos_sin = torch.cat((cos.reshape(-1, rotary_dim), sin.reshape(-1, rotary_dim)), dim=-1)
        query, key = torch.ops.tensor_cast.fused_rope(query, key, cos_sin, rotary_dim, True)
        cache = (kwargs.get("kv_cache_by_layers") or {}).get(self.layer_idx)
        if cache is None:
            raise ValueError("Step5 requires a paged KV cache")
        torch.ops.tensor_cast.reshape_and_cache(key.flatten(1), value.flatten(1), cache, metadata.slot_mapping)
        seqs = metadata.seq_lens_values
        queries = metadata.query_lens_values
        if seqs is None or queries is None:
            raise ValueError("Step5 requires materialized sequence/query lengths")
        selected = None
        if self.is_sparse_layer:
            iq = self.index_q_norm(self.index_query(hidden_states).reshape(query.shape[0], -1, self.proxy_dim))
            ik = self.index_k_norm(self.index_key(hidden_states).reshape(query.shape[0], 1, self.proxy_dim))
            iz = self.index_z(hidden_states).reshape_as(ik)
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                raise ValueError("Step5 sparse attention requires position_ids for indexer RoPE")
            # A 32-dimensional frequency basis is not a slice of the 64-dimensional main RoPE.
            index_cos, index_sin = self.index_rotary_emb(
                hidden_states, position_ids, self.config.layer_types[self.layer_idx]
            )
            index_cos_sin = torch.cat((index_cos, index_sin), -1)
            iq, ik = torch.ops.tensor_cast.fused_rope(
                iq, ik, index_cos_sin.reshape(query.shape[0], -1), self.index_rope_dim, True
            )
            weights = self.index_weights(hidden_states).float().reshape(query.shape[0], -1)
            weights = weights * (self.index_heads // self.provider_groups) ** -0.5
            index_cache = (kwargs.get("indexer_cache_by_layers") or {}).get(self.layer_idx)
            if index_cache is None:
                raise ValueError("Step5 sparse attention requires its summary/pending cache")
            selected = torch.ops.tensor_cast.step5_indexer(
                iq,
                ik,
                iz,
                weights,
                index_cache,
                seqs,
                queries,
                self.topk,
                self.region_size,
                key.shape[1],
            )
        output = torch.ops.tensor_cast.step5_attention(
            query,
            cache,
            selected,
            seqs,
            queries,
            self.sliding_window or 0,
            self.topk if self.is_sparse_layer else 0,
            self.region_size,
        )
        if self.config.use_head_wise_attn_gate:
            gate = self.g_proj(hidden_states).reshape(query.shape[0], -1, 1).sigmoid()
            output = output * gate.to(output.dtype)
        return self.o_proj(output.reshape(*shape, -1)), None


class Step5VisionAttention(Step3p7VisionAttention):
    """Perception encoder attention with explicit non-causal TensorCast cost."""

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).reshape(*shape, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape_as(query)
        value = self.v_proj(hidden_states).reshape_as(query)
        query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
        output = torch.ops.tensor_cast.attention(query, key, value, attention_mask, None, None, None, None)
        return self.out_proj(output.reshape(*shape, -1)), None
