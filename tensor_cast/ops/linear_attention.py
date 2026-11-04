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


@register_tensor_cast_op("linear_attn_state_gather")
def _(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Select recurrent-state rows without materializing cache contents."""
    if state.ndim != 4 or state_indices.ndim != 1 or has_initial_state.ndim != 1:
        raise ValueError("Linear-attention state gather expects a 4D cache and 1D metadata")
    if state_indices.numel() != has_initial_state.numel():
        raise ValueError("State indices and initial-state flags must have the same length")
    return torch.empty(
        (state_indices.numel(), *state.shape[1:]),
        dtype=state.dtype,
        device=state.device,
    )


@register_tensor_cast_op("linear_attn_state_scatter", mutates_args=("state",))
def _(
    state: torch.Tensor,
    final_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Record indexed recurrent-state cache writes on meta tensors."""
    if state.ndim != 4 or final_state.ndim != 4 or state_indices.ndim != 1:
        raise ValueError("Linear-attention state scatter expects 4D states and 1D indices")
    if final_state.shape != (state_indices.numel(), *state.shape[1:]):
        raise ValueError("Final-state rows must match the selected cache rows")


@register_tensor_cast_op("linear_attn_recurrent_state_io")
def _(
    output: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Keep fused recurrent-state traffic dependent on the recurrent output."""
    if state.ndim < 1 or state_indices.ndim != 1:
        raise ValueError("Linear-attention recurrent state I/O expects a row-based cache and 1D indices")
    return torch.empty_like(output)


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
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del key, beta, g, chunk_size, state_read_passes, state_write_passes
    batch_size, seq_len, num_v_heads, _ = query.shape
    head_v_dim = value.shape[-1]
    output = torch.empty(
        (batch_size, seq_len, num_v_heads, head_v_dim),
        dtype=query.dtype,
        device=query.device,
    )
    final_state = (
        torch.empty_like(initial_state)
        if initial_state is not None
        else torch.empty((0,), dtype=torch.float32, device=query.device)
    )
    return output, final_state


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
    activation: str = "silu",
) -> torch.Tensor:
    del z, weight, eps, activation
    return torch.empty_like(core_attn_out).contiguous()
