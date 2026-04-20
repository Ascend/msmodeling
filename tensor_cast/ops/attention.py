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
        cache_position: optional cache positions
        num_k_heads: number of key heads
        num_v_heads: number of value heads
        head_k_dim: per-head key dimension
        head_v_dim: per-head value dimension
        conv_kernel_size: kernel size used by the causal depthwise conv1d
    """
    return torch.empty_like(hidden_states).contiguous()
