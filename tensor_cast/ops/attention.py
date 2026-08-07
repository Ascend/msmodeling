from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("reshape_and_cache", mutates_args=("kv_cache",))
def _(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    pass


@register_tensor_cast_op("siso_reshape_and_cache", mutates_args=("kv_cache",))
def _(
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Single-input (K-only) counterpart of ``reshape_and_cache``.

    Writes ``key`` into ``kv_cache`` according to ``slot_mapping`` without a paired
    value tensor. Used for single-stream caches such as the MiniMax-M3 index K
    cache.
    """


@register_tensor_cast_op("attention")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    query_lens: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Normal attention: MHA/GQA/MQA

    Args:
        query: (num_tokens, hidden_size)
        key:
            (total_num_blocks, block_size, kv_head_num, head_size) if block_table exists,
            otherwise (*, kv_head_num, head_size)
        value:
            (total_num_blocks, block_size, kv_head_num, head_size) if block_table exists,
            otherwise (*, kv_head_num, head_size)
        attention_mask: (batch_size, num_heads, max_q_len, max_seq_len)
        block_table: (batch_size, max_blocks_per_seq)
        query_start_loc: (batch_size + 1,), the start location of each request in query Tensor
        seq_len: (batch_size,), the length of each request including both computed tokens and newly scheduled tokens
    """
    return torch.empty_like(query).contiguous()


@register_tensor_cast_op("attention_quant")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    query_lens: Optional[torch.Tensor],
    query_scale: torch.Tensor,
    query_offset: Optional[torch.Tensor],
    kv_scale: torch.Tensor,
    kv_offset: Optional[torch.Tensor],
    attention_prob_scale: torch.Tensor,
    attention_prob_offset: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    Quantized version of normal attention: MHA/GQA/MQA

    Args:
        query: (num_tokens, hidden_size)
        key:
            (total_num_blocks, block_size, kv_head_num, head_size) if block_table exists,
            otherwise (*, kv_head_num, head_size)
        value:
            (total_num_blocks, block_size, kv_head_num, head_size) if block_table exists,
            otherwise (*, kv_head_num, head_size)
        attention_mask: (batch_size, num_heads, max_q_len, max_seq_len)
        block_table: (batch_size, max_blocks_per_seq)
        query_start_loc: (batch_size + 1,), the start location of each request in query Tensor
        seq_len: (batch_size,), the length of each request including both computed tokens and newly scheduled tokens
        query_scale/query_offset: quant param for query, per-tensor or per-token
        kv_scale/kv_offset: quant param for KV cache, per-tensor or per channel (along head_size)
        attention_prob_scale/attention_prob_offset: quant param for input of the second BMM
    """
    if out_dtype is None:
        out_dtype = query.dtype
    return torch.empty_like(query, dtype=out_dtype).contiguous()


@register_tensor_cast_op("attention_route_generate")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    block_size: int,
    sparsity: float,
) -> torch.Tensor:
    """
    TensorCast semantic op for block-level attention route generation.

    The returned int32 tensor carries shape metadata for tracing and performance
    modeling. Its values are unspecified and are not consumed by TensorCast.
    Sparsity affects analytic BSA cost, not route metadata shape.

    Args:
        query: (batch_size, query_seq_len, num_heads, head_size)
        key: (batch_size, key_seq_len, num_kv_heads, head_size)
        block_size: query/key-value block size used by the route plan
        sparsity: skipped KV-block ratio in [0.0, 1.0)
    """
    if query.ndim == 4:
        batch_size, query_seq_len, num_heads, _ = query.shape
        key_seq_len = key.shape[1]
    else:
        batch_size = 1
        query_seq_len = query.shape[0]
        num_heads = 1
        key_seq_len = key.shape[0]
    q_blocks = (query_seq_len + block_size - 1) // block_size
    kv_blocks = (key_seq_len + block_size - 1) // block_size
    return torch.empty((batch_size, num_heads, q_blocks, kv_blocks), device=query.device, dtype=torch.int32)


@register_tensor_cast_op("block_sparse_attention")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    route_metadata: torch.Tensor,
    block_size: int,
    sparsity: float,
) -> torch.Tensor:
    """
    TensorCast semantic op for block sparse attention.

    This op models output shape and performance properties. It does not execute
    numerical sparse attention or read route metadata values.

    Args:
        query: (batch_size, query_seq_len, num_heads, head_size)
        key: (batch_size, key_seq_len, num_kv_heads, head_size)
        value: (batch_size, key_seq_len, num_kv_heads, head_size)
        attention_mask: optional dense attention mask
        route_metadata: shape-only metadata from attention_route_generate
        block_size: query/key-value block size used by the route plan
        sparsity: skipped KV-block ratio in [0.0, 1.0)
    """
    return torch.empty_like(query).contiguous()


@register_tensor_cast_op("linear_attention")
def _(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
) -> torch.Tensor:
    """
    Fused linear attention op for Qwen3.5 GatedDeltaNet blocks.

    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: optional mask tensor for left-padding cases
        cache_position: optional cache positions, used to determine if there's previous state
        num_k_heads: number of key heads
        num_v_heads: number of value heads
        head_k_dim: per-head key dimension
        head_v_dim: per-head value dimension
        conv_kernel_size: kernel size used by the causal depthwise conv1d
    """
    return torch.empty_like(hidden_states).contiguous()
