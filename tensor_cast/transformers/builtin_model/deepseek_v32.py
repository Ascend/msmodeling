# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, DeepseekV3Config

from transformers.cache_utils import Cache
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ...layers.mla import DeepseekSparseAttention
from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="deepseek_v32",
        moe_module_name="DeepseekV32MoE",
        moe_num_experts_key="n_routed_experts",
        moe_gate_returns_raw_logits=False,
        mla_module_name="DeepseekV32SparseAttention",
        mla_module_class_type=DeepseekSparseAttention,
    )
)


class DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"

    def __init__(
        self,
        topk_limit: Optional[int] = None,
        **kwargs,
    ):
        index_topk = kwargs.pop("index_topk", None)
        super().__init__(**kwargs)
        self.topk_limit = index_topk if index_topk is not None else topk_limit


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV32MoE(DeepseekV3MoE):
    pass


class DeepseekV32MLP(DeepseekV3MLP):
    pass


class DeepseekV32Indexer(nn.Module):
    def __init__(self, config: "DeepseekV32Config", index_layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = index_layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.num_local_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.topk_limit = config.topk_limit
        self.q_lora_rank = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(
            self.hidden_size,
            self.num_heads,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values_index: "Cache",
        cache_position: torch.LongTensor | None,
    ) -> torch.LongTensor:
        return None


class DeepseekV32SparseAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV32RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV32RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.scaling = 1.0 / math.sqrt(self.qk_head_dim)
        self.softmax_scale = self.scaling
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, Tuple[torch.Tensor] | None]:
        return None


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV32SparseAttention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV32MoE(config)
        else:
            self.mlp = DeepseekV32MLP(config)

        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepseekV32Model(DeepseekV3Model):
    config: DeepseekV32Config

    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            layers.append(DeepseekV32DecoderLayer(config, layer_idx))
        self.layers = nn.ModuleList(layers)

        self.norm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


AutoConfig.register("deepseek_v32", DeepseekV32Config)
AutoModel.register(DeepseekV32Config, DeepseekV32Model)
