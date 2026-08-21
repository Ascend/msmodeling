# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
"""Qwen3 DFlash decoder blocks with KV injection and TensorCast attention.

Context K/V are produced once at draft-model level
(``fc`` → ``context_kv_proj`` → per-layer ``k_norm`` → **one fused** context
``apply_rope`` → ``reshape_and_cache`` × N), matching NPU trace (§2.4 / 阶段 F).

Each decoder layer only handles the short noise block (NPU §2.6–2.7):
  1) projects noise QKV (length = ``dflash_block_size``)
  2) ``apply_rope`` on noise Q + noise K only (not full-length context K)
  3) ``reshape_and_cache`` appends noise K/V
  4) TC attention reads context K/V from cache (already RoPE'd)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3MLP, Qwen3RMSNorm

from .. import ops  # noqa: F401
from .attention import AttentionMetadataBase


def _flatten_tokens(x_bshd: torch.Tensor) -> torch.Tensor:
    """[B, S, H, D] → [B*S, H*D]."""
    bsz, seq, num_heads, head_dim = x_bshd.shape
    return x_bshd.reshape(bsz * seq, num_heads * head_dim)


class Qwen3DFlashAttention(nn.Module):
    """DFlash attention: short noise QKV + RoPE/cache/attn (context already in cache)."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        type_layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        # Global index into model.attention_by_layers (may include target offset).
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        type_idx = int(type_layer_idx) if type_layer_idx is not None else int(layer_idx)
        if type_idx < 0 or type_idx >= len(layer_types):
            raise ValueError(f"type_layer_idx={type_idx} out of range for layer_types length={len(layer_types)}")
        self.layer_type = layer_types[type_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        context_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        draft_context_attention_meta: Optional[AttentionMetadataBase] = None,
        draft_noise_attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del draft_context_attention_meta  # Context write happens once at draft-model level.
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        head_dim = self.head_dim

        # Noise Q / K / V (short block only). Layout after norm: BHSD for apply_rope.
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, -1, head_dim)).transpose(1, 2)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        k_noise = self.k_norm(k_noise.view(bsz, q_len, -1, head_dim)).transpose(1, 2)
        v_noise = v_noise.view(bsz, q_len, -1, head_dim).transpose(1, 2)

        cos_n, sin_n = position_embeddings
        # NPU Decoder: _triton_rope only on block_size noise tokens (Q + noise K).
        q_bshd, k_noise_bshd = torch.ops.tensor_cast.apply_rope(q, k_noise, cos_n, sin_n, True)
        v_noise_bshd = v_noise.transpose(1, 2)

        attention_by_layers = kwargs.get("attention_by_layers")
        if attention_by_layers is None:
            raise ValueError("Dflash attention requires attention_by_layers for TensorCast ops")
        self_attn = attention_by_layers[self.layer_idx]

        kv_cache_by_layers = kwargs.get("kv_cache_by_layers")
        kv_cache = None
        if kv_cache_by_layers is not None and self.layer_idx in kv_cache_by_layers:
            kv_cache = kv_cache_by_layers[self.layer_idx]

        query = _flatten_tokens(q_bshd)
        k_noise_flat = _flatten_tokens(k_noise_bshd)
        v_noise_flat = _flatten_tokens(v_noise_bshd)

        use_cache = (
            kv_cache is not None
            and draft_noise_attention_meta is not None
            and draft_noise_attention_meta.slot_mapping is not None
        )
        if use_cache:
            # Context already written at draft-model level; only append noise + attend.
            attn_output = self_attn.forward(
                query,
                k_noise_flat,
                v_noise_flat,
                attention_mask,
                kv_cache=kv_cache,
                attention_meta=draft_noise_attention_meta,
                sliding_window=self.sliding_window,
            )
        else:
            # No-cache fallback requires valid pre-RoPE'd context K/V (BSHD).
            if context_kv is None:
                raise ValueError(
                    "no-cache DFlash attention requires context_kv "
                    "(pre-RoPE'd K/V); uninitialized new_empty() is not allowed"
                )
            k_ctx_bshd, v_ctx_bshd = context_kv
            if k_ctx_bshd.ndim == 3:
                # Raw [B, S, kv_dim] (rare); view only — no per-layer context RoPE.
                k_ctx_bshd = k_ctx_bshd.view(bsz, ctx_len, -1, head_dim)
                v_ctx_bshd = v_ctx_bshd.view(bsz, ctx_len, -1, head_dim)
            k_ctx_flat = _flatten_tokens(k_ctx_bshd)
            v_ctx_flat = _flatten_tokens(v_ctx_bshd)
            key = torch.cat([k_ctx_flat, k_noise_flat], dim=0)
            value = torch.cat([v_ctx_flat, v_noise_flat], dim=0)
            num_kv_heads = k_noise_bshd.size(2)
            key = key.view(-1, num_kv_heads, head_dim)
            value = value.view(-1, num_kv_heads, head_dim)
            attn_output = self_attn.forward(
                query,
                key,
                value,
                attention_mask,
                kv_cache=None,
                attention_meta=None,
                sliding_window=self.sliding_window,
            )

        # Attention API is token-flat; restore draft contract [B, S, H] for o_proj/residual.
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3DFlashDecoderLayer(nn.Module):
    """Pre-LN DFlash decoder layer with context-injected attention.

    Layout contract: ``hidden_states`` / residuals are ``[B, S, H]`` for the whole
    layer. The wrapper normalizes packed target layouts once; this module must not
    re-pack or ``reshape_as``-defend residuals.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        attention_layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        attn_idx = attention_layer_idx if attention_layer_idx is not None else layer_idx
        self.self_attn = Qwen3DFlashAttention(
            config=config,
            layer_idx=attn_idx,
            type_layer_idx=layer_idx,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            context_kv=context_kv,
            **kwargs,
        )[0]
        # Draft contract: noise / attn / mlp stay [B, S, H] (normalized once at wrapper).
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
