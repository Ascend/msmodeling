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


"""Shape-only Step5 sparse index and attention operators."""

from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("step5_indexer", mutates_args=("cache",))
def step5_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    z: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    topk: int,
    region_size: int,
    provider_groups: int,
) -> torch.Tensor:
    if topk <= 0 or region_size <= 0 or provider_groups <= 0:
        raise ValueError("Step5 sparse dimensions must be positive")
    return torch.empty((query.shape[0], provider_groups, topk), device=query.device, dtype=torch.int32)


@register_tensor_cast_op("step5_attention")
def step5_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: Optional[torch.Tensor],
    seq_lens: list[int],
    query_lens: list[int],
    window: int,
    topk: int,
    region_size: int,
) -> torch.Tensor:
    if len(seq_lens) != len(query_lens) or sum(query_lens) != query.shape[0]:
        raise ValueError("Step5 packed lengths must match query tokens")
    if any(q <= 0 or s < q for s, q in zip(seq_lens, query_lens)):
        raise ValueError("Step5 query lengths must be positive and no greater than sequence lengths")
    return torch.empty_like(query)
