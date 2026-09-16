from typing import Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("linear_attn_apply_padding_mask")
def _(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states).contiguous()


@register_tensor_cast_op("linear_attn_causal_conv")
def _(
    mixed_qkv: torch.Tensor,
    conv_kernel_size: int,
) -> torch.Tensor:
    return torch.empty_like(mixed_qkv).contiguous()


@register_tensor_cast_op("linear_attn_causal_conv_update")
def _(
    mixed_qkv: torch.Tensor,
    conv_kernel_size: int,
) -> torch.Tensor:
    return torch.empty_like(mixed_qkv).contiguous()


@register_tensor_cast_op("linear_attn_fused_gdn_gating")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_v_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _, head_k_dim = query.shape
    out_shape = (batch_size, seq_len, num_v_heads, head_k_dim)
    query_out = torch.empty(out_shape, dtype=query.dtype, device=query.device)
    key_out = torch.empty(out_shape, dtype=key.dtype, device=key.device)
    beta = torch.empty((batch_size, seq_len, num_v_heads), dtype=b.dtype, device=b.device)
    g = torch.empty((batch_size, seq_len, num_v_heads), dtype=torch.float32, device=a.device)
    return query_out, key_out, beta, g


@register_tensor_cast_op("linear_attn_chunk_gated_delta_rule")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    state_read_passes: int,
    state_write_passes: int,
) -> torch.Tensor:
    del key, beta, g, chunk_size, state_read_passes, state_write_passes
    batch_size, seq_len, num_v_heads, _ = query.shape
    head_v_dim = value.shape[-1]
    return torch.empty(
        (batch_size, seq_len, num_v_heads, head_v_dim),
        dtype=query.dtype,
        device=query.device,
    )


@register_tensor_cast_op("linear_attn_recurrent_gated_delta_rule")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state_read_passes: int,
    state_write_passes: int,
) -> torch.Tensor:
    del key, beta, g, state_read_passes, state_write_passes
    batch_size, seq_len, num_v_heads, _ = query.shape
    head_v_dim = value.shape[-1]
    return torch.empty(
        (batch_size, seq_len, num_v_heads, head_v_dim),
        dtype=query.dtype,
        device=query.device,
    )


@register_tensor_cast_op("linear_attn_gated_rmsnorm")
def _(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    del z, weight, eps
    return torch.empty_like(core_attn_out).contiguous()
