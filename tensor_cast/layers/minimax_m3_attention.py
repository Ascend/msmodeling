import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _get_eps(norm: torch.nn.Module) -> float:
    if hasattr(norm, "variance_epsilon"):
        return norm.variance_epsilon
    if hasattr(norm, "eps"):
        return norm.eps
    raise AttributeError(f"{type(norm).__name__} has neither 'variance_epsilon' nor 'eps' attribute")


class GemmaRMSNormFusedWrapper(torch.nn.Module):
    """Wrapper that replaces GemmaRMSNorm with fused rms_norm and add_rms_norm2 ops."""

    def __init__(self, original_norm: torch.nn.Module):
        super().__init__()
        self._inner = original_norm
        self.weight = original_norm.weight
        self.eps = _get_eps(original_norm)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        effective_weight = 1.0 + self.weight
        if residual is not None:
            out, residual_out = torch.ops.tensor_cast.add_rms_norm2(
                x,
                residual,
                effective_weight,
                self.eps,
            )
            return out, residual_out
        return torch.ops.tensor_cast.rms_norm(x, effective_weight, self.eps)


class RMSNormFusedWrapper(torch.nn.Module):
    """Wrapper that replaces RMSNorm with fused rms_norm op."""

    def __init__(self, original_norm: torch.nn.Module, is_gemma: bool = False):
        super().__init__()
        self._inner = original_norm
        self.weight = original_norm.weight
        self.eps = _get_eps(original_norm)
        self.is_gemma = is_gemma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_gemma:
            effective_weight = 1.0 + self.weight
        else:
            effective_weight = self.weight
        return torch.ops.tensor_cast.rms_norm(x, effective_weight, self.eps)


def _fused_decoder_layer_forward(self, hidden_states, **kwargs):
    """Source-faithful M3 decoder forward with one safe AddRMSNorm fusion."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    attn_out = self.self_attn(hidden_states, **kwargs)
    if isinstance(attn_out, tuple):
        hidden_states = attn_out[0]
    else:
        hidden_states = attn_out

    # Fuse source semantics: x = residual + attn_out; normed = post_norm(x).
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    hidden_states = self.mlp(hidden_states)

    hidden_states = hidden_states + residual
    return hidden_states


class MiniMaxM3AttentionWrapper(torch.nn.Module):
    """Wrapper for MiniMax-M3 attention that routes dense/sparse layers.

    Dense layers use the standard tensor_cast attention path (HF attention forward
    which calls q_proj, q_norm, fused_rope, attention, o_proj).

    Sparse layers mirror the upstream MiniMaxM3VLAttention path:
      1. q_proj/k_proj/v_proj, q_norm/k_norm, fused_rope, cache write
      2. indexer.q_proj/indexer.k_proj, indexer q/k norm, indexer fused_rope
      3. minimax_indexer for block score/top-k selection
      4. minimax_sparse_attention for sparse attention body
      5. o_proj

    Projection, norm, RoPE, cache, and o_proj are explicit ops. The two M3
    virtual ops only model the sparse index selection and sparse attention body.
    """

    def __init__(
        self,
        original_module: torch.nn.Module,
        is_sparse_layer: bool,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_indexer_heads: int,
        indexer_head_dim: int,
        topk_blocks: int,
        block_size: int,
        local_blocks: int,
        rotary_dim: int = 64,
        indexer_head_start: int = 0,
    ):
        super().__init__()
        self._inner = original_module
        self.is_sparse_layer = is_sparse_layer
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_indexer_heads = num_indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.rotary_dim = rotary_dim
        self.indexer_head_start = indexer_head_start
        self.index_query_linear = None
        self.index_key_linear = None

        inner = original_module
        while hasattr(inner, "_inner"):
            inner = inner._inner
        indexer = getattr(inner, "indexer", None)
        if indexer is not None and getattr(indexer, "q_proj", None) is not None:
            # Generic TP sharding matches "*.q_proj"/"*.k_proj"; keep MiniMax-M3
            # indexer projections under wrapper-local names so they are not
            # incorrectly sharded as normal attention Q/K projections.
            q_out_features = num_indexer_heads * indexer_head_dim
            q_start = indexer_head_start * indexer_head_dim
            q_stop = q_start + q_out_features
            self.index_query_linear = torch.nn.Linear(
                indexer.q_proj.in_features,
                q_out_features,
                bias=indexer.q_proj.bias is not None,
            )
            self.index_key_linear = torch.nn.Linear(
                indexer.k_proj.in_features,
                indexer_head_dim,
                bias=indexer.k_proj.bias is not None,
            )
            with torch.no_grad():
                self.index_query_linear.weight.copy_(indexer.q_proj.weight[q_start:q_stop])
                self.index_key_linear.weight.copy_(indexer.k_proj.weight[:indexer_head_dim])
                if indexer.q_proj.bias is not None:
                    self.index_query_linear.bias.copy_(indexer.q_proj.bias[q_start:q_stop])
                if indexer.k_proj.bias is not None:
                    self.index_key_linear.bias.copy_(indexer.k_proj.bias[:indexer_head_dim])
            indexer.q_proj = None
            indexer.k_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attention_meta = kwargs.get("attention_meta", None)

        if attention_meta is None:
            return self._inner(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        seq_lens = attention_meta.seq_lens
        query_lens = attention_meta.query_lens
        block_table = attention_meta.block_table_tensor
        seq_lens_values = getattr(attention_meta, "seq_lens_values", None)
        query_lens_values = getattr(attention_meta, "query_lens_values", None)
        is_decode_values = getattr(attention_meta, "is_decode_values", None)

        inner = self._inner
        while hasattr(inner, "_inner"):
            inner = inner._inner

        position_embeddings = kwargs.get("position_embeddings", None)

        if hidden_states.ndim == 3:
            num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        else:
            num_tokens = hidden_states.shape[0]

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = inner.q_norm(inner.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = inner.k_norm(inner.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = inner.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if hidden_states.ndim == 3:
            actual_num_q_heads = query_states.shape[1]
            actual_num_kv_heads = key_states.shape[1]
        else:
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

        query_3d = query_states.transpose(1, 2).reshape(
            num_tokens,
            actual_num_q_heads,
            self.head_dim,
        )
        key_3d = key_states.transpose(1, 2).reshape(
            num_tokens,
            actual_num_kv_heads,
            self.head_dim,
        )
        value_3d = value_states.transpose(1, 2).reshape(
            num_tokens,
            actual_num_kv_heads,
            self.head_dim,
        )

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

        if not self.is_sparse_layer:
            key_for_attention = kv_cache[0] if kv_cache is not None else key_3d
            value_for_attention = kv_cache[1] if kv_cache is not None else value_3d
            out = torch.ops.tensor_cast.attention(
                query_3d,
                key_for_attention,
                value_for_attention,
                attention_mask,
                block_table,
                attention_meta.query_start_loc,
                seq_lens,
                query_lens,
            )
            out = out.reshape(*input_shape, -1).contiguous()
            return inner.o_proj(out), None

        indexer = getattr(inner, "indexer", None)
        if indexer is not None and self.index_query_linear is not None and self.index_key_linear is not None:
            idx_hidden_shape = (*input_shape, -1, self.indexer_head_dim)
            idx_q_states = indexer.q_norm(self.index_query_linear(hidden_states).view(idx_hidden_shape)).transpose(1, 2)
            idx_k_states = indexer.k_norm(self.index_key_linear(hidden_states).view(idx_hidden_shape)).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                if cos.ndim == 3:
                    idx_cos = cos.reshape(num_tokens, cos.shape[-1])
                    idx_sin = sin.reshape(num_tokens, sin.shape[-1])
                else:
                    idx_cos = cos.reshape(num_tokens, -1)
                    idx_sin = sin.reshape(num_tokens, -1)
                idx_cos_sin = torch.cat([idx_cos, idx_sin], dim=-1)
                idx_q_3d = idx_q_states.transpose(1, 2).reshape(
                    num_tokens, self.num_indexer_heads, self.indexer_head_dim
                )
                idx_k_3d = idx_k_states.transpose(1, 2).reshape(num_tokens, 1, self.indexer_head_dim)
                idx_q_3d, idx_k_3d = torch.ops.tensor_cast.fused_rope(
                    idx_q_3d, idx_k_3d, idx_cos_sin, self.rotary_dim, True
                )
                idx_q_states = idx_q_3d.reshape(*input_shape, self.num_indexer_heads, self.indexer_head_dim).transpose(
                    1, 2
                )
                idx_k_states = idx_k_3d.reshape(*input_shape, 1, self.indexer_head_dim).transpose(1, 2)
        else:
            idx_q_states = hidden_states.new_empty(*input_shape, self.num_indexer_heads, self.indexer_head_dim)
            idx_k_states = hidden_states.new_empty(*input_shape, 1, self.indexer_head_dim)

        idx_q_flat = idx_q_states.transpose(1, 2).reshape(num_tokens, self.num_indexer_heads, self.indexer_head_dim)
        idx_k_flat = idx_k_states.transpose(1, 2).reshape(num_tokens, 1, self.indexer_head_dim)

        indexer_cache_by_layers = kwargs.get("indexer_cache_by_layers", None)
        indexer_cache = indexer_cache_by_layers.get(inner.layer_idx) if indexer_cache_by_layers else None
        if indexer_cache is not None and attention_meta is not None:
            torch.ops.tensor_cast.siso_reshape_and_cache(idx_k_flat, indexer_cache, attention_meta.slot_mapping)
        index_keys = indexer_cache if indexer_cache is not None else idx_k_flat

        topk_idx = torch.ops.tensor_cast.minimax_indexer(
            idx_q_flat,
            index_keys,
            seq_lens,
            query_lens,
            block_table,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            seq_lens_values=seq_lens_values,
            query_lens_values=query_lens_values,
            is_decode_values=is_decode_values,
        )

        query = query_3d.reshape(num_tokens, -1)
        if kv_cache is not None:
            key_cache_tensor = kv_cache[0]
            value_cache_tensor = kv_cache[1]
        else:
            key_cache_tensor = key_3d.reshape(num_tokens, -1)
            value_cache_tensor = value_3d.reshape(num_tokens, -1)

        out = torch.ops.tensor_cast.minimax_sparse_attention(
            query,
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
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            seq_lens_values=seq_lens_values,
            query_lens_values=query_lens_values,
            is_decode_values=is_decode_values,
        )

        out = out.reshape(*input_shape, -1).contiguous()
        return inner.o_proj(out), None
