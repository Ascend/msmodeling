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

"""Generic HuggingFace architecture facts needed by query workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tensor_cast.transformers.utils import AutoModelConfigLoader


@dataclass(frozen=True)
class QueryModelArchitecture:
    """Small, model-name-agnostic view of an inference architecture."""

    max_context_length: int
    num_experts: int
    num_mtp_layers: int
    tp_sizes: tuple[int, ...]
    ep_sizes: tuple[int, ...]


def _positive_int(config: Any, names: Iterable[str], default: int = 0) -> int:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def _power_of_two_divisors(*values: int) -> tuple[int, ...]:
    positive = [value for value in values if value > 0]
    limit = min(positive, default=1)
    result = []
    size = 1
    while size <= limit:
        if all(value % size == 0 for value in positive):
            result.append(size)
        size *= 2
    return tuple(result or [1])


def resolve_query_model_architecture(model_id: str) -> QueryModelArchitecture:
    """Load one HF config and derive legal parallel axes without name tables."""
    config = AutoModelConfigLoader().load_config(model_id)
    if config is None:
        raise ValueError(f"Unable to load HuggingFace model architecture: {model_id}")

    hidden_size = _positive_int(config, ("hidden_size", "d_model"))
    num_attention_heads = _positive_int(config, ("num_attention_heads", "n_head"))
    if hidden_size <= 0 or num_attention_heads <= 0:
        raise ValueError(
            f"Model {model_id!r} does not expose positive hidden_size and num_attention_heads"
        )
    max_context_length = _positive_int(
        config,
        ("max_position_embeddings", "max_sequence_length", "seq_length"),
        default=131072,
    )
    num_experts = _positive_int(config, ("n_routed_experts", "num_experts", "num_local_experts"))
    num_mtp_layers = _positive_int(
        config,
        ("num_nextn_predict_layers", "num_mtp_layers", "num_nextn_predict_layers_per_block"),
    )
    return QueryModelArchitecture(
        max_context_length=max_context_length,
        num_experts=num_experts,
        num_mtp_layers=num_mtp_layers,
        tp_sizes=_power_of_two_divisors(hidden_size, num_attention_heads),
        ep_sizes=_power_of_two_divisors(num_experts) if num_experts else (1,),
    )
