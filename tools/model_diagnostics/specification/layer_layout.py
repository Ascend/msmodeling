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
"""Model-specific physical-layer layout derivation used by Spec materialization."""

from __future__ import annotations

from collections.abc import Mapping

from tools.model_diagnostics.errors import SourceLoadError

_QWEN3_VL_MOE_DENSE_LAYER = "qwen3_vl_moe_dense_text_decoder"
_QWEN3_VL_MOE_LAYER = "qwen3_vl_moe_text_decoder"


def qwen3_vl_moe_layer_kinds(
    config: Mapping[str, object],
    *,
    start: int,
    count: int,
) -> tuple[str, ...]:
    """Derive Qwen3-VL MoE's physical dense/MoE layer kinds."""

    sparse_step = config.get("decoder_sparse_step")
    if isinstance(sparse_step, bool) or not isinstance(sparse_step, int) or sparse_step <= 0:
        raise SourceLoadError("Qwen3-VL MoE decoder_sparse_step must be a positive integer")
    mlp_only_layers = config.get("mlp_only_layers")
    if not isinstance(mlp_only_layers, (list, tuple)) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in mlp_only_layers
    ):
        raise SourceLoadError("Qwen3-VL MoE mlp_only_layers must contain non-negative integers")
    if len(mlp_only_layers) != len(set(mlp_only_layers)):
        raise SourceLoadError("Qwen3-VL MoE mlp_only_layers must not contain duplicates")
    num_experts = config.get("num_experts")
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0:
        raise SourceLoadError("Qwen3-VL MoE num_experts must be a positive integer")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise SourceLoadError("Qwen3-VL MoE layer start must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SourceLoadError("Qwen3-VL MoE layer count must be a non-negative integer")

    dense_layers = set(mlp_only_layers)
    return tuple(
        _QWEN3_VL_MOE_LAYER
        if layer_index not in dense_layers and (layer_index + 1) % sparse_step == 0
        else _QWEN3_VL_MOE_DENSE_LAYER
        for layer_index in range(start, start + count)
    )
