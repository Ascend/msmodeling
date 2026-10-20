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

"""Shared semantic operators for manifold-constrained Hyper-Connections."""

from typing import Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("hc_pre_inv_rms")
def _(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Return the inverse-RMS factor for an ``[B, S, Hc, D]`` stream tensor."""
    batch_shape = x.shape[:-2]
    return torch.empty(*batch_shape, 1, dtype=torch.float32, device=x.device)


@register_tensor_cast_op("hc_pre_sinkhorn")
def _(
    x: torch.Tensor,
    hidden_states: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int = 1,
    hc_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split HC mixes, project ``comb`` with Sinkhorn and collapse streams.

    ``x`` is the FP32 mix projection with width ``(2 + hc_mult) * hc_mult``.
    The outputs are reduced hidden states, post weights and the stream-combine
    matrix, in that order.
    """
    batch_shape = x.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    return (
        torch.empty(*batch_shape, hidden_size, dtype=hidden_states.dtype, device=x.device),
        torch.empty(*batch_shape, hc_mult, dtype=x.dtype, device=x.device),
        torch.empty(*batch_shape, hc_mult, hc_mult, dtype=x.dtype, device=x.device),
    )


@register_tensor_cast_op("hc_split_sinkhorn")
def _(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split HC coefficients without reducing the residual stream."""
    del hc_scale, hc_base, sinkhorn_iters, hc_eps
    batch_shape = mixes.shape[:-1]
    return (
        torch.empty(*batch_shape, hc_mult, dtype=mixes.dtype, device=mixes.device),
        torch.empty(*batch_shape, hc_mult, dtype=mixes.dtype, device=mixes.device),
        torch.empty(*batch_shape, hc_mult, hc_mult, dtype=mixes.dtype, device=mixes.device),
    )


@register_tensor_cast_op("hc_pre_reduce")
def _(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse HC streams using a pre-mix produced by an earlier sublayer."""
    del pre_mix
    return torch.empty(
        *hidden_states.shape[:-2],
        hidden_states.shape[-1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


@register_tensor_cast_op(
    "hc_pre_v3",
    schema="(Tensor hidden_states, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, Tensor? pre_mix, "
    "int hc_mult, int sinkhorn_iters, float norm_eps, float hc_eps, str pre_mix_mode='incoming') "
    "-> (Tensor, Tensor, Tensor, Tensor?)",
)
def _(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: Optional[torch.Tensor],
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    pre_mix_mode: str = "incoming",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fuse HC projection/Sinkhorn and stream collapse.

    ``current`` (V4) collapses with the newly generated pre coefficients,
    which stay inside the op. ``incoming`` (V4.1) collapses with ``pre_mix``
    and returns the new coefficients for the next sublayer.
    """
    _validate_hc_pre_v3(hidden_states, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, sinkhorn_iters, pre_mix_mode)
    del norm_eps, hc_eps
    leading = hidden_states.shape[:-2]
    return (
        torch.empty(*leading, hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device),
        torch.empty(*leading, hc_mult, dtype=torch.float32, device=hidden_states.device),
        torch.empty(*leading, hc_mult, hc_mult, dtype=torch.float32, device=hidden_states.device),
        torch.empty(*leading, hc_mult, dtype=torch.float32, device=hidden_states.device)
        if pre_mix_mode == "incoming"
        else None,
    )


def _validate_hc_pre_v3(hidden_states, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, sinkhorn_iters, pre_mix_mode):
    if pre_mix_mode not in ("current", "incoming"):
        raise ValueError(f"Unknown HC pre mix mode: {pre_mix_mode}")
    if hc_mult <= 0 or sinkhorn_iters < 1:
        raise ValueError("HC pre requires positive hc_mult and sinkhorn_iters")
    if hidden_states.ndim < 2 or hidden_states.shape[-2] != hc_mult:
        raise ValueError("HC pre expects streams [..., hc_mult, hidden]")
    mix_width = (2 + hc_mult) * hc_mult
    if hc_fn.shape != (mix_width, hc_mult * hidden_states.shape[-1]):
        raise ValueError("HC pre projection must have shape [(2 + hc_mult) * hc_mult, hc_mult * hidden]")
    if hc_scale.shape != (3,) or hc_base.shape != (mix_width,):
        raise ValueError("HC pre scale/base must have shapes [3] and [(2 + hc_mult) * hc_mult]")
    if pre_mix_mode == "current" and pre_mix is not None:
        raise ValueError("HC pre current mode requires pre_mix=None")
    if pre_mix_mode == "incoming" and (pre_mix is None or pre_mix.shape != hidden_states.shape[:-1]):
        raise ValueError("HC pre incoming mode requires pre_mix [..., hc_mult]")


@register_tensor_cast_op("hc_post")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    hc_weight: Optional[torch.Tensor],
    hc_combine: Optional[torch.Tensor],
    hc_mult: int,
) -> torch.Tensor:
    """Expand a sublayer update and restore the residual HC stream layout.

    Compiled attention or MLP paths may expose token-flattened tensors, or
    temporarily retain an extra token axis.  The residual is the authoritative
    logical ``[..., hc_mult, hidden]`` stream shape and prevents those physical
    layouts from leaking into the next decoder layer.
    """
    if residual.shape[-2] != int(hc_mult) or residual.shape[-1] != x.shape[-1]:
        raise ValueError(
            "HC post expects residual [..., hc_mult, hidden] matching the sublayer hidden size, "
            f"got x={tuple(x.shape)}, residual={tuple(residual.shape)}, hc_mult={hc_mult}"
        )
    return torch.empty_like(residual, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("hc_head")
def _(
    x: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    hc_mult: int,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a learned HC head and reduce ``[B, S, Hc, D]`` to ``[B, S, D]``."""
    batch_shape = x.shape[:-2]
    hidden = x.shape[-1]
    return torch.empty(*batch_shape, hidden, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("mhc_head")
def _(streams: torch.Tensor) -> torch.Tensor:
    """Collapse GLM5Next's manifold HC streams with an unweighted mean."""
    batch, seq, _mult, hidden = streams.shape
    return streams.new_empty((batch, seq, hidden))
