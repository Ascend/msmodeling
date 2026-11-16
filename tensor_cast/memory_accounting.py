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

"""Independent memory totals and lightweight resident tensor identities."""

from dataclasses import dataclass


_CACHE_MEMORY_COLLECTORS = {}


@dataclass(frozen=True)
class ForwardMemorySummary:
    request_input_bytes: int = 0
    activation_peak_bytes: int = 0

    def peak_memory_bytes(self, weight_bytes: int, cache_bytes: int) -> int:
        return weight_bytes + cache_bytes + self.request_input_bytes + self.activation_peak_bytes


@dataclass(frozen=True)
class CacheMemoryEntry:
    owner_key: str
    tensors: tuple
    size_bytes: int
    token_pool_bytes: int
    layer_index: int | None = None


@dataclass(frozen=True)
class CacheMemoryInventory:
    entries: tuple[CacheMemoryEntry, ...]

    @property
    def fixed_bytes(self) -> int:
        return sum(entry.size_bytes - entry.token_pool_bytes for entry in self.entries)


def register_cache_memory_collector(model_type):
    """Register a model adapter that describes additional persistent cache state."""

    def decorator(collector):
        _CACHE_MEMORY_COLLECTORS[model_type] = collector
        return collector

    return decorator


def collect_model_cache_memory_entries(model, input_kwargs, excluded) -> tuple[CacheMemoryEntry, ...]:
    """Collect model-specific cache entries without teaching the generic input path model names."""
    model_config = getattr(model, "model_config", None)
    model_type = getattr(getattr(model_config, "hf_config", None), "model_type", None)
    collector = _CACHE_MEMORY_COLLECTORS.get(model_type)
    return () if collector is None else tuple(collector(model, input_kwargs, excluded))
