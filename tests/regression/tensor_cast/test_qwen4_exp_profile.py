# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under the Mulan PSL v2.
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
"""Registration and patch-structure tests for the qwen4_exp ModelProfile
(Qwen3.8-Flash-Next: 48-layer hybrid [GDN x3, QSA x1] x 12 stack).
"""

from types import SimpleNamespace

import torch

from tensor_cast.layers.qwen4_exp_attention import Qwen4ExpQSAAttentionWrapper
from tensor_cast.transformers.builtin_model.qwen4_exp import (
    _patch_qwen4_exp_qsa_attention,
    _set_qwen4_exp_linear_attn_tp_size,
)
from tensor_cast.transformers.custom_model_registry import get_model_profile


def _fake_text_config(**overrides):
    config = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=24,
        head_dim=256,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        partial_rotary_factor=0.25,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _FakeIndexer(torch.nn.Module):
    def __init__(self, in_features, q_heads, kv_heads, head_dim):
        super().__init__()
        self.index_qk_proj = torch.nn.Linear(in_features, q_heads * head_dim + kv_heads * head_dim, bias=False)


class _FakeQsaAttention(torch.nn.Module):
    def __init__(self, in_features, q_heads, kv_heads, head_dim):
        super().__init__()
        self.indexer = _FakeIndexer(in_features, q_heads, kv_heads, head_dim)

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class Qwen4ExpTextDecoderLayer(torch.nn.Module):
    """Name-matched stand-in: patch discovery matches on type name."""

    def __init__(self, in_features, q_heads, kv_heads, head_dim):
        super().__init__()
        self.self_attn = _FakeQsaAttention(in_features, q_heads, kv_heads, head_dim)


def _fake_model(num_layers, in_features=2560, q_heads=4, kv_heads=1, head_dim=128):
    inner = torch.nn.Module()
    for layer_idx in range(num_layers):
        inner.add_module(
            f"layer_{layer_idx}",
            Qwen4ExpTextDecoderLayer(in_features, q_heads, kv_heads, head_dim),
        )
    model = SimpleNamespace(
        _inner=inner,
        parallel_group_manager=None,
        text_config=_fake_text_config(),
    )
    return model


def test_qwen4_exp_profile_is_registered():
    profile = get_model_profile("qwen4_exp")
    assert profile is not None
    # The GDN linear-attention TP plan in transformations.py is gated on
    # model_family == "qwen3_5" (four independent in_proj_qkv/z/b/a
    # projections); qwen4_exp reuses that exact layout.
    assert profile.model_family == "qwen3_5"
    assert profile.moe_module_name == "Qwen4ExpTextSparseMoeBlock"
    assert profile.moe_gate_returns_raw_logits is False
    # qwen4_exp ships as a VL wrapper: num_experts lives under text_config.
    assert profile.moe_num_experts_key == ["text_config", "num_experts"]
    assert profile.moe_field_names_override is not None
    assert profile.patch_method is not None


def test_patch_qsa_attention_installs_wrappers():
    model = _fake_model(num_layers=2)
    patched_count = _patch_qwen4_exp_qsa_attention(model, model.text_config)

    assert patched_count == 2
    for layer in model._inner.modules():
        if not isinstance(layer, Qwen4ExpTextDecoderLayer):
            continue
        wrapper = layer.self_attn
        assert isinstance(wrapper, Qwen4ExpQSAAttentionWrapper)
        # input_generator detection hooks for the indexer cache allocation.
        assert wrapper.is_sparse_layer is True
        assert wrapper.indexer_cache_width == 128
        assert wrapper.block_topk == 2048 // 4
        # The fused indexer projection must be rebound to the wrapper-local
        # copy so the generic TP plan cannot slice the shared KV rows apart,
        # while the official fallback path still finds a usable projection.
        assert wrapper._inner.indexer.index_qk_proj is wrapper.index_qk_linear
        assert wrapper.index_qk_linear is not None
        assert wrapper.index_qk_linear.out_features == (4 + 1) * 128
    # Idempotent: re-running must not double-wrap.
    assert _patch_qwen4_exp_qsa_attention(model, model.text_config) == 0


def test_qsa_wrapper_slices_indexer_q_rows_under_tp():
    tp_size = 2
    tp_rank = 1
    model = _fake_model(num_layers=1)
    model.parallel_group_manager = SimpleNamespace(tp_group=SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank))
    layer = next(m for m in model._inner.modules() if isinstance(m, Qwen4ExpTextDecoderLayer))
    fused = layer.self_attn.indexer.index_qk_proj
    full_q_rows = 4 * 128

    _patch_qwen4_exp_qsa_attention(model, model.text_config)

    wrapper = layer.self_attn
    assert isinstance(wrapper, Qwen4ExpQSAAttentionWrapper)
    assert wrapper.num_indexer_heads == 4 // tp_size
    assert wrapper.indexer_head_start == tp_rank * (4 // tp_size)
    q_rows = (4 // tp_size) * 128
    kv_rows = 1 * 128
    # Rank slice: [rank q rows | full shared kv rows] against the fused weight.
    expected_q = fused.weight[tp_rank * q_rows : (tp_rank + 1) * q_rows]
    expected_kv = fused.weight[full_q_rows : full_q_rows + kv_rows]
    assert torch.equal(wrapper.index_qk_linear.weight[:q_rows], expected_q)
    assert torch.equal(wrapper.index_qk_linear.weight[q_rows:], expected_kv)


def test_qsa_wrapper_without_attention_meta_falls_back_to_inner():
    model = _fake_model(num_layers=1)
    _patch_qwen4_exp_qsa_attention(model, model.text_config)
    layer = next(m for m in model._inner.modules() if isinstance(m, Qwen4ExpTextDecoderLayer))

    # The official fallback path must stay callable after wrapper
    # construction: the indexer projection is rebound to the wrapper-local
    # copy (TP-shard safe) instead of being cleared to None.
    official_proj = layer.self_attn._inner.indexer.index_qk_proj
    assert isinstance(official_proj, torch.nn.Linear)
    assert official_proj.out_features == (4 + 1) * 128

    hidden_states = torch.randn(2, 3, 2560)
    out = layer.self_attn(hidden_states)
    # The fake inner forward is an identity, so the golden-test passthrough
    # must return the input untouched.
    assert out is hidden_states


def test_qsa_wrapper_unwraps_nested_inner_and_copies_bias():
    # The wrapper constructor walks nested _inner chains to find the HF
    # module holding the indexer, and copies the fused index_qk_proj rows
    # (per-rank q rows + full shared kv rows) including the bias slot.

    class _OuterWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

    config = _fake_text_config()
    layer = Qwen4ExpTextDecoderLayer(2560, config.indexer_n_heads, config.indexer_kv_heads, config.indexer_head_dim)
    # Give the fused indexer projection a bias so the bias-copy branch runs,
    # and wrap the attention in an extra _inner level so the discovery loop
    # must see through it.
    layer.self_attn.indexer.index_qk_proj = torch.nn.Linear(
        2560, (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim, bias=True
    )
    layer.self_attn = _OuterWrapper(layer.self_attn)
    holder = torch.nn.Module()
    holder.add_module("layer_0", layer)
    model = SimpleNamespace(
        _inner=holder,
        parallel_group_manager=None,
        text_config=config,
    )

    assert _patch_qwen4_exp_qsa_attention(model, config) == 1
    wrapper = layer.self_attn
    assert isinstance(wrapper, Qwen4ExpQSAAttentionWrapper)
    # The discovery loop saw through the _OuterWrapper level.
    assert isinstance(wrapper._inner, _OuterWrapper)
    assert isinstance(wrapper._unwrap(), type(layer.self_attn._inner._inner))
    assert wrapper.index_qk_linear is not None
    assert wrapper.index_qk_linear.bias is not None
    expected_rows = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim
    assert wrapper.index_qk_linear.bias.shape == (expected_rows,)


def test_set_linear_attn_tp_size_validates_head_divisibility():
    gdn = SimpleNamespace(num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128)
    model = SimpleNamespace(
        _inner=SimpleNamespace(modules=lambda: iter([gdn])),
        parallel_group_manager=SimpleNamespace(tp_group=SimpleNamespace(world_size=4, rank_in_group=0)),
    )
    _set_qwen4_exp_linear_attn_tp_size(model)
    assert gdn.tensor_cast_tp_size == 4

    bad = SimpleNamespace(num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128)
    model_bad = SimpleNamespace(
        _inner=SimpleNamespace(modules=lambda: iter([bad])),
        parallel_group_manager=SimpleNamespace(tp_group=SimpleNamespace(world_size=3, rank_in_group=0)),
    )
    try:
        _set_qwen4_exp_linear_attn_tp_size(model_bad)
    except ValueError as exc:
        assert "num_k_heads" in str(exc)
    else:
        raise AssertionError("tp_size=3 must not divide num_k_heads=16")
