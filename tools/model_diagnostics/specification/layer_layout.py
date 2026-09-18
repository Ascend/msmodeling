# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
