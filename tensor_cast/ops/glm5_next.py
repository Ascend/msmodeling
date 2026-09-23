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

"""Shape-only GLM5Next semantic blocks; analytic costs are not measured kernels."""

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("glm5_next_kda", mutates_args=("state",))
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forget: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    state: torch.Tensor,
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    is_decode: list[bool],
    lower_bound: float | None,
    eps: float,
) -> torch.Tensor:
    """Convolution, channel-wise KDA gates/rule and sigmoid-gated RMSNorm.

    Projections remain outside this composite. State is packed as one uint8 row
    per request (FP32 recurrent matrix plus activation-dtype convolution state).
    The estimator accounts for the constituent kernels and scratch traffic.
    """
    return torch.empty_like(v).contiguous()


@register_tensor_cast_op("glm5_next_kpool_indexer", mutates_args=("cache",))
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    gates: torch.Tensor,
    weights: torch.Tensor,
    ape: torch.Tensor,
    cache: torch.Tensor,
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    topk: int,
    always_select_tail: bool,
) -> torch.Tensor:
    if pool_size <= 0 or topk <= 0 or topk % pool_size:
        raise ValueError("k-pool requires a positive pool size dividing topk")
    # Keep HF's padded upper bound stable under FakeTensor tracing. Invalid
    # candidates remain semantic padding; the estimator uses active lengths.
    width = topk + (pool_size - 1 if always_select_tail else 0)
    return torch.empty((*q.shape[:2], width), dtype=torch.int32, device=q.device)
