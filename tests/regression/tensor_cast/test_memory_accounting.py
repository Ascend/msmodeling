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

from types import SimpleNamespace

import torch

from tensor_cast import memory_accounting
from tensor_cast.core.input_generator import collect_cache_memory
from tensor_cast.memory_accounting import (
    CacheMemoryEntry,
    collect_model_cache_memory_entries,
    register_cache_memory_collector,
)


def test_register_cache_memory_collector_dispatches_model_adapter():
    model_type = "test_memory_accounting_collector"
    entry = CacheMemoryEntry("fixed:test", (), 17, 0)

    @register_cache_memory_collector(model_type)
    def collector(model, input_kwargs, excluded):
        assert model.marker == "model"
        assert input_kwargs == {"marker": "inputs"}
        assert excluded == frozenset({3})
        return (entry,)

    try:
        model = SimpleNamespace(
            marker="model",
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        )
        assert collector is memory_accounting._CACHE_MEMORY_COLLECTORS[model_type]
        assert collect_model_cache_memory_entries(model, {"marker": "inputs"}, frozenset({3})) == (entry,)
    finally:
        memory_accounting._CACHE_MEMORY_COLLECTORS.pop(model_type, None)


def test_cache_inventory_includes_scales_and_counts_shared_objects_once():
    cache = torch.empty(10, device="meta", dtype=torch.uint8)
    scale = torch.empty(2, device="meta", dtype=torch.float32)
    state = torch.empty(20, device="meta", dtype=torch.uint8)
    state.tensor_cast_fixed_cache = True
    model = SimpleNamespace(text_config=SimpleNamespace(model_type="test"))
    inventory = collect_cache_memory(
        model,
        {
            "kv_cache_by_layers": {0: cache, 1: state},
            "indexer_scale_cache_by_layers": {0: scale},
            "compressed_kv_scale_cache_by_source": {0: scale},
        },
    )
    assert sum(entry.size_bytes for entry in inventory.entries) == 38
    assert inventory.fixed_bytes == 20
    assert len(inventory.entries) == 3


def test_kimi_kda_fixed_cache_capacity_is_stored_in_inventory(monkeypatch):
    model = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
        text_config=SimpleNamespace(
            num_hidden_layers=6,
            linear_attn_config={"kda_layers": [2, 4, 6], "full_attn_layers": [1, 3, 5]},
        ),
    )
    caches = {layer_idx: torch.empty(1, device="meta") for layer_idx in (1, 3, 5)}
    calls = []

    def fake_state_size(_model, batch_size):
        calls.append(batch_size)
        return 10

    monkeypatch.setattr(
        "tensor_cast.transformers.builtin_model.kimi_k3._k3_kda_state_bytes_per_layer",
        fake_state_size,
    )

    inventory = collect_cache_memory(
        model,
        {
            "attention_meta": SimpleNamespace(seq_lens=torch.empty(4, device="meta", dtype=torch.long)),
            "kv_cache_by_layers": caches,
        },
    )

    assert calls == [4]
    assert [entry.owner_key for entry in inventory.entries] == [
        "kimi_kda_state:1",
        "kimi_kda_state:3",
        "kimi_kda_state:5",
    ]
    assert [entry.size_bytes for entry in inventory.entries] == [10, 10, 10]
    assert inventory.fixed_bytes == 30
    assert all(entry.token_pool_bytes == 0 for entry in inventory.entries)
    assert all(not hasattr(cache, "tensor_cast_cache_capacity_bytes") for cache in caches.values())
