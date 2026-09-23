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


@register_tensor_cast_op("hc_post")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    hc_weight: Optional[torch.Tensor],
    hc_combine: Optional[torch.Tensor],
    hc_mult: int,
) -> torch.Tensor:
    """Expand a sublayer update and mix it with the residual HC streams."""
    batch_shape = x.shape[:-1]
    hidden = x.shape[-1]
    return torch.empty(*batch_shape, hc_mult, hidden, dtype=x.dtype, device=x.device)


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
