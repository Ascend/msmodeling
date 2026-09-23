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

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import transformers.utils.import_utils as import_utils  # pylint: disable=no-name-in-module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # pylint: disable=no-name-in-module

from tensor_cast import ops  # noqa: F401
from tensor_cast.performance_model import OpInvokeInfo
from tensor_cast.transformers.custom_model_registry import get_model_profile


def test_bailing_v3_profile_registers_model_structure():
    profile = get_model_profile("bailing_hybrid")

    assert profile is not None
    assert profile.moe_module_name == "BailingMoeV3SparseMoeBlock"
    assert profile.mla_module_name == "BailingMoeV3MultiLatentAttention"
    assert profile.mtp_block_module_name == "BailingMoeV3MTPLayer"
    assert profile.mla_module_class_type.__name__ == "BailingV3MultiheadLatentAttentionTensorCast"
    assert profile.moe_gate_router is not None
    assert profile.patch_method is not None


def test_bailing_v3_patch_restores_transformers_5_torch_fx_helper(monkeypatch):
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_hf_config_for_bailing_v3

    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    config = SimpleNamespace(model_type="bailing_hybrid")

    assert _patch_hf_config_for_bailing_v3(config) is True
    assert import_utils.is_torch_fx_available() is True


def test_bailing_v3_torch_fx_helper_caches_availability(monkeypatch):
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_hf_config_for_bailing_v3

    calls = []

    def find_spec(name):
        calls.append(name)
        return object()

    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    config = SimpleNamespace(model_type="bailing_hybrid")

    assert _patch_hf_config_for_bailing_v3(config) is True
    assert import_utils.is_torch_fx_available() is True
    assert import_utils.is_torch_fx_available() is True
    assert calls == ["torch.fx"]


def test_bailing_v3_patch_restores_none_rope_scaling_from_transformers_5_default():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_hf_config_for_bailing_v3

    config = SimpleNamespace(
        model_type="bailing_hybrid",
        rope_scaling={
            "rope_theta": 6_000_000,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
    )

    assert _patch_hf_config_for_bailing_v3(config) is True
    assert config.rope_scaling is None


def test_bailing_v3_patch_restores_transformers_5_default_rope(monkeypatch):
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_hf_config_for_bailing_v3

    monkeypatch.delitem(ROPE_INIT_FUNCTIONS, "default", raising=False)
    config = SimpleNamespace(model_type="bailing_hybrid", rope_scaling=None)

    assert _patch_hf_config_for_bailing_v3(config) is True

    rope_config = SimpleNamespace(head_dim=64, partial_rotary_factor=0.5, rope_theta=10_000)
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["default"](rope_config, torch.device("cpu"))
    assert inv_freq.shape == (16,)
    assert attention_scaling == 1.0


def test_mlapo_supports_direct_q_projection():
    hidden_states = torch.empty((4, 16), device="meta")
    cos = torch.empty((1, 4, 4), device="meta")
    sin = torch.empty((1, 4, 4), device="meta")
    q_proj_weight = torch.empty((24, 16), device="meta")
    kv_a_proj_weight = torch.empty((12, 16), device="meta")
    kv_a_layernorm_weight = torch.empty((8,), device="meta")

    q_states, kv_c_normed, k_rot, qa_normed = torch.ops.tensor_cast.mlapo(
        hidden_states,
        cos,
        sin,
        q_proj_weight,
        None,
        None,
        kv_a_proj_weight,
        kv_a_layernorm_weight,
        3,
        8,
        4,
        4,
        8,
        None,
    )

    assert q_states.shape == (4, 3, 8)
    assert kv_c_normed.shape == (4, 8)
    assert k_rot.shape == (4, 4)
    assert qa_normed.shape == (4, 0)


def test_bailing_v3_router_uses_indices_weights_logits_order():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _route_bailing_v3_gate

    expected_indices = torch.tensor([[7, 2]])
    expected_weights = torch.tensor([[0.6, 0.4]])

    class Gate:
        def __call__(self, hidden_states):
            logits = torch.empty((hidden_states.shape[0], 512))
            return expected_indices, expected_weights, logits

    indices, weights = _route_bailing_v3_gate(Gate(), torch.empty((1, 16)), 2, None, None)

    assert indices is expected_indices
    assert weights is expected_weights


def test_bailing_v3_router_accepts_and_applies_tp_slice():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _route_bailing_v3_gate

    hidden_states = torch.empty((5, 4))
    all_indices = torch.arange(10).view(5, 2)
    all_weights = torch.arange(10, dtype=torch.float32).view(5, 2)

    class Gate:
        def __call__(self, states):
            assert states is hidden_states
            return all_indices, all_weights, torch.empty((5, 8))

    topk_indices, topk_weights = _route_bailing_v3_gate(
        Gate(),
        hidden_states,
        top_k=2,
        input_ids=None,
        moe_layer_idx=0,
        tp_size=2,
        tp_rank=1,
    )

    assert topk_indices.shape == (3, 2)
    assert topk_weights.shape == (3, 2)
    assert torch.equal(topk_indices[:2], all_indices[3:])
    assert torch.equal(topk_weights[:2], all_weights[3:])
    assert torch.equal(topk_indices[-1], torch.zeros(2, dtype=all_indices.dtype))
    assert torch.equal(topk_weights[-1], torch.zeros(2, dtype=all_weights.dtype))


def test_bailing_v3_router_rejects_invalid_tp_rank():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _route_bailing_v3_gate

    class Gate:
        def __call__(self, hidden_states):
            del hidden_states
            return torch.zeros((2, 1)), torch.zeros((2, 1)), torch.zeros((2, 1))

    with pytest.raises(ValueError, match=r"tp_rank must be in \[0, 2\)"):
        _route_bailing_v3_gate(
            Gate(),
            torch.empty((2, 4)),
            top_k=1,
            input_ids=None,
            moe_layer_idx=0,
            tp_size=2,
            tp_rank=2,
        )


def test_bailing_v3_mla_recovers_tensor_cast_side_channel():
    from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast
    from tensor_cast.transformers.builtin_model.bailing_v3 import BailingV3MultiheadLatentAttentionTensorCast

    wrapper = BailingV3MultiheadLatentAttentionTensorCast.__new__(BailingV3MultiheadLatentAttentionTensorCast)
    torch.nn.Module.__init__(wrapper)
    attention_meta = object()
    wrapper._extra_forward_kwargs = {"attention_meta": attention_meta}

    with patch.object(
        MultiheadLatentAttentionTensorCast,
        "forward",
        return_value=(torch.empty((1, 1, 1)), None),
    ) as base_forward:
        result = wrapper.forward(torch.empty((1, 1, 1)))

    assert len(result) == 3
    assert base_forward.call_args.kwargs["attention_meta"] is attention_meta


def test_bailing_v3_model_patch_propagates_metadata_to_self_attn():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_bailing_v3_model

    class DecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Identity()

    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DecoderLayer()])
            self.num_nextn_predict_layers = 0

    class Wrapper(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    base_model = BaseModel()
    wrapper = Wrapper()
    model = SimpleNamespace(_inner=wrapper, unwrap=lambda: base_model)
    attention_meta = object()

    _patch_bailing_v3_model(model)
    result = wrapper(attention_meta=attention_meta)

    assert result["attention_meta"] is attention_meta
    assert base_model.layers[0].self_attn._extra_forward_kwargs["attention_meta"] is attention_meta


def test_bailing_v3_model_patch_caches_attention_modules_after_first_forward():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_bailing_v3_model

    class DecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Identity()

    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DecoderLayer()])
            self.num_nextn_predict_layers = 0

    class Wrapper(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    base_model = BaseModel()
    wrapper = Wrapper()
    unwrap_calls = 0

    def unwrap():
        nonlocal unwrap_calls
        unwrap_calls += 1
        return base_model

    model = SimpleNamespace(_inner=wrapper, unwrap=unwrap)
    _patch_bailing_v3_model(model)

    wrapper(attention_meta=object())
    calls_after_first_forward = unwrap_calls
    wrapper(attention_meta=object())

    assert unwrap_calls == calls_after_first_forward


def test_bailing_v3_mla_shards_dense_output_projection_rowwise():
    from tensor_cast.layers import ROWWISE_LINEAR
    from tensor_cast.transformers.builtin_model.bailing_v3 import BailingV3MultiheadLatentAttentionTensorCast

    params = {"tp_group": object(), "global_tp_group": object(), "head_num": 32}

    plan = BailingV3MultiheadLatentAttentionTensorCast.build_o_proj_tp_plan_extras(
        "model.layers",
        params,
        SimpleNamespace(),
    )

    assert plan["model.layers.*.dense"] == (ROWWISE_LINEAR, params)


def test_kimi_delta_attention_core_preserves_value_shape():
    q = torch.empty((1, 8, 4, 16), device="meta")
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    f = torch.empty_like(q)
    beta = torch.empty((1, 8, 4), device="meta")
    gate = torch.empty_like(q)
    state = torch.empty((1, 4096), device="meta")
    query_lens = torch.tensor([8])
    seq_lens = torch.tensor([8])

    output = torch.ops.tensor_cast.kimi_delta_attention_core(
        q,
        k,
        v,
        f,
        beta,
        gate,
        state,
        query_lens,
        seq_lens,
        4,
    )

    assert output.shape == v.shape
    state_argument = torch.ops.tensor_cast.kimi_delta_attention_core.default._schema.arguments[6]
    assert state_argument.name == "state"
    assert state_argument.alias_info is not None
    assert state_argument.alias_info.is_write


def test_bailing_v3_kda_requires_runtime_metadata():
    from tensor_cast.transformers.builtin_model.bailing_v3 import BailingV3KimiDeltaAttentionTensorCast

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 2
            self.conv_size = 3
            self.no_kda_lora = True
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 4, bias=False)
            self.v_proj = torch.nn.Linear(4, 4, bias=False)
            self.f_proj = torch.nn.Linear(4, 4, bias=False)
            self.g_proj = torch.nn.Linear(4, 4, bias=False)
            self.b_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)

    wrapper = BailingV3KimiDeltaAttentionTensorCast(Attention())

    with pytest.raises(ValueError, match="requires attention_meta and kv_cache_by_layers"):
        wrapper(torch.empty((1, 2, 4)))


@pytest.mark.parametrize("no_kda_lora", [True, False])
def test_bailing_v3_kda_forwards_both_projection_layouts(no_kda_lora):
    from tensor_cast.transformers.builtin_model.bailing_v3 import BailingV3KimiDeltaAttentionTensorCast

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 2
            self.conv_size = 3
            self.no_kda_lora = no_kda_lora
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 4, bias=False)
            self.v_proj = torch.nn.Linear(4, 4, bias=False)
            if no_kda_lora:
                self.f_proj = torch.nn.Linear(4, 4, bias=False)
                self.g_proj = torch.nn.Linear(4, 4, bias=False)
            else:
                self.f_a_proj = torch.nn.Linear(4, 2, bias=False)
                self.f_b_proj = torch.nn.Linear(2, 4, bias=False)
                self.g_a_proj = torch.nn.Linear(4, 2, bias=False)
                self.g_b_proj = torch.nn.Linear(2, 4, bias=False)
            self.b_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Identity()

    wrapper = BailingV3KimiDeltaAttentionTensorCast(Attention())
    hidden_states = torch.empty((1, 3, 4))
    state = torch.empty((1, 64))
    attention_meta = SimpleNamespace(query_lens=torch.tensor([3]), seq_lens=torch.tensor([3]))
    past_key_value = object()

    with patch(
        "torch.ops.tensor_cast.kimi_delta_attention_core",
        side_effect=lambda q, _k, v, _f, _beta, _gate, *_args: v,
    ) as core:
        output, attention_weights, returned_cache = wrapper(
            hidden_states,
            past_key_value=past_key_value,
            attention_meta=attention_meta,
            kv_cache_by_layers={0: state},
        )

    q, k, v, f, beta, gate = core.call_args.args[:6]
    assert all(tensor.shape == (1, 3, 2, 2) for tensor in (q, k, v, f, gate))
    assert beta.shape == (1, 3, 2)
    assert core.call_args.args[6] is state
    assert core.call_args.args[7] is attention_meta.query_lens
    assert core.call_args.args[8] is attention_meta.seq_lens
    assert output.shape == hidden_states.shape
    assert attention_weights is None
    assert returned_cache is past_key_value


def test_bailing_v3_kda_reports_missing_cache_layer():
    from tensor_cast.transformers.builtin_model.bailing_v3 import BailingV3KimiDeltaAttentionTensorCast

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 3
            self.head_dim = 2
            self.conv_size = 3
            self.no_kda_lora = True
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 4, bias=False)
            self.v_proj = torch.nn.Linear(4, 4, bias=False)
            self.f_proj = torch.nn.Linear(4, 4, bias=False)
            self.g_proj = torch.nn.Linear(4, 4, bias=False)
            self.b_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Identity()

    wrapper = BailingV3KimiDeltaAttentionTensorCast(Attention())
    attention_meta = SimpleNamespace(query_lens=torch.tensor([1]), seq_lens=torch.tensor([1]))

    with pytest.raises(KeyError, match=r"layer_idx=3.*available layer indices: \[0, 2\]"):
        wrapper(
            torch.empty((1, 1, 4)),
            attention_meta=attention_meta,
            kv_cache_by_layers={0: torch.empty((1, 1)), 2: torch.empty((1, 1))},
        )


def _get_performance_properties(op, args):
    output = op(*args)
    return OpInvokeInfo(op, args, {}, output).get_perf_properties()


def _kimi_delta_attention_properties(query_len):
    q = torch.empty((1, query_len, 2, 4), device="meta", dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    f = torch.empty_like(q)
    beta = torch.empty((1, query_len, 2), device="meta", dtype=torch.float16)
    gate = torch.empty_like(q)
    state = torch.empty((1, 32), device="meta", dtype=torch.float32)
    query_lens = torch.tensor([query_len])
    seq_lens = torch.tensor([query_len])
    args = (q, k, v, f, beta, gate, state, query_lens, seq_lens, 3)
    return _get_performance_properties(torch.ops.tensor_cast.kimi_delta_attention_core.default, args)


def test_kimi_delta_attention_core_models_recurrent_boundary_ops():
    properties = _kimi_delta_attention_properties(64)

    assert properties.compute_ops[torch.float16] == OpInvokeInfo.ComputeOps(mma_ops=8192, gp_ops=31746)
    assert properties.compute_ops[torch.float32] == OpInvokeInfo.ComputeOps(gp_ops=5888)


def test_kimi_delta_attention_core_models_chunk_path_ops():
    properties = _kimi_delta_attention_properties(65)

    assert properties.compute_ops[torch.float16] == OpInvokeInfo.ComputeOps(mma_ops=679936, gp_ops=32242)
    assert properties.compute_ops[torch.float32] == OpInvokeInfo.ComputeOps(gp_ops=788100)


def test_kimi_delta_attention_core_warns_when_query_lens_are_unavailable(caplog):
    class UnavailableQueryLens:
        @staticmethod
        def tolist():
            raise RuntimeError("query lengths are unavailable")

    query_len = 2
    q = torch.empty((1, query_len, 2, 4), device="meta", dtype=torch.float16)
    args = (
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty((1, query_len, 2), device="meta", dtype=torch.float16),
        torch.empty_like(q),
        torch.empty((1, 32), device="meta", dtype=torch.float32),
        UnavailableQueryLens(),
        torch.tensor([query_len]),
        3,
    )
    invoke_info = OpInvokeInfo(
        torch.ops.tensor_cast.kimi_delta_attention_core.default,
        args,
        {},
        torch.empty_like(q),
    )

    with caplog.at_level("WARNING", logger="tensor_cast.performance_model"):
        invoke_info.get_perf_properties()

    assert "falling back to a single-request assumption with num_tokens=2" in caplog.text


def _direct_q_mlapo_args(quantized=False):
    weight_dtype = torch.int8 if quantized else torch.float16
    hidden_states = torch.empty((4, 16), device="meta", dtype=torch.float16)
    cos = torch.empty((1, 4, 4), device="meta", dtype=torch.float16)
    sin = torch.empty_like(cos)
    q_proj_weight = torch.empty((24, 16), device="meta", dtype=weight_dtype)
    kv_a_proj_weight = torch.empty((12, 16), device="meta", dtype=weight_dtype)
    kv_a_layernorm_weight = torch.empty((8,), device="meta", dtype=torch.float16)
    args = (
        hidden_states,
        cos,
        sin,
        q_proj_weight,
        None,
        None,
        kv_a_proj_weight,
        kv_a_layernorm_weight,
        3,
        8,
        4,
        4,
        8,
        None,
    )
    if quantized:
        scale = torch.ones(1, device="meta")
        args += (scale, None, None, None, scale, None)
    return args


def test_mlapo_direct_q_models_float_ops():
    args = _direct_q_mlapo_args()
    properties = _get_performance_properties(torch.ops.tensor_cast.mlapo.default, args)

    assert properties.compute_ops[torch.float16] == OpInvokeInfo.ComputeOps(mma_ops=4608, gp_ops=352)
    assert properties.memory_read_bytes == 1360
    assert properties.memory_readwrite_bytes == 576


def test_mlapo_direct_q_models_quantized_ops():
    args = _direct_q_mlapo_args(quantized=True)
    properties = _get_performance_properties(torch.ops.tensor_cast.mlapo_quant.default, args)

    assert properties.compute_ops[torch.int8] == OpInvokeInfo.ComputeOps(mma_ops=4608)
    assert properties.compute_ops[torch.float16] == OpInvokeInfo.ComputeOps(gp_ops=1056)


def test_bailing_v3_mtp_adapter_uses_native_two_input_block():
    from tensor_cast.layers.mtp import BailingV3MultiTokenPredictorLayer

    class NativeMtpBlock(torch.nn.Module):
        def forward(self, input_embeds, hidden_states, **kwargs):
            self.received = (input_embeds, hidden_states, kwargs)
            return (hidden_states + 1,)

    block = NativeMtpBlock()
    layer = BailingV3MultiTokenPredictorLayer(SimpleNamespace(), block)
    input_embeds = torch.zeros((1, 2, 4))
    hidden_states = torch.zeros((1, 2, 4))

    output = layer(input_embeds, torch.arange(2), hidden_states, position_embeddings=(None, None))

    assert torch.equal(output, torch.ones_like(hidden_states))
    assert block.received[0] is input_embeds
    assert block.received[1] is hidden_states


def test_bailing_v3_model_patch_strips_native_mtp_tail():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_bailing_v3_model

    native_mtp_type = type("BailingMoeV3MTPLayer", (torch.nn.Module,), {})
    main_layer = torch.nn.Identity()
    base_model = torch.nn.Module()
    base_model.layers = torch.nn.ModuleList([main_layer, native_mtp_type()])
    base_model.num_nextn_predict_layers = 1

    class Wrapper(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    wrapper = Wrapper()
    model = SimpleNamespace(_inner=wrapper, unwrap=lambda: base_model)

    _patch_bailing_v3_model(model)

    assert list(base_model.layers) == [main_layer]
    assert base_model.num_nextn_predict_layers == 0


def test_bailing_v3_model_patch_skips_completed_patch_work():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_bailing_v3_model

    base_model = torch.nn.Module()
    base_model.layers = torch.nn.ModuleList([torch.nn.Identity()])
    base_model.num_nextn_predict_layers = 0

    class Wrapper(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    wrapper = Wrapper()
    model = SimpleNamespace(_inner=wrapper, unwrap=lambda: base_model)

    with patch("tensor_cast.transformers.builtin_model.bailing_v3._replace_bailing_v3_kda_modules") as replace:
        _patch_bailing_v3_model(model)
        calls_after_first_patch = replace.call_count
        _patch_bailing_v3_model(model)

    assert replace.call_count == calls_after_first_patch


def test_bailing_v3_model_patch_rejects_unexpected_native_mtp_tail():
    from tensor_cast.transformers.builtin_model.bailing_v3 import _patch_bailing_v3_model

    base_model = torch.nn.Module()
    base_model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Linear(1, 1)])
    base_model.num_nextn_predict_layers = 1
    model = SimpleNamespace(_inner=torch.nn.Module(), unwrap=lambda: base_model)

    with pytest.raises(ValueError, match="Bailing V3 native MTP layers"):
        _patch_bailing_v3_model(model)


def test_bailing_v3_mtp_attention_resolves_copy_layer_representative():
    from tensor_cast.layers.internal import CopyLayerWrapper, RegionMarkerWrapper
    from tensor_cast.transformers.builtin_model.bailing_v3 import (
        _collect_bailing_v3_attention_modules,
        _get_mtp_attention_modules,
    )

    attention = torch.nn.Identity()
    mtp_block = torch.nn.Module()
    mtp_block.attention = attention
    layer = torch.nn.Module()
    layer.mtp_block = mtp_block
    representative = RegionMarkerWrapper(region_id=1, layer=layer, repeat_count=2)
    copy_layer = CopyLayerWrapper(region_id=1, layer=layer, representative=representative)
    mtp = SimpleNamespace(layers=[representative, copy_layer])

    assert _get_mtp_attention_modules(mtp) == [attention, attention]
    model = SimpleNamespace(unwrap=lambda: SimpleNamespace(layers=[]))
    assert _collect_bailing_v3_attention_modules(model, SimpleNamespace(mtp=mtp)) == (attention,)


def _make_fake_bailing_v3_kda(layer_idx):
    attention_type = type("BailingMoeV3KimiDeltaAttention", (torch.nn.Module,), {})
    attention = attention_type()
    attention.layer_idx = layer_idx
    attention.head_dim = 2
    attention.conv_size = 3
    attention.no_kda_lora = True
    return attention


def test_patch_bailing_v3_model_replaces_kda_in_base_and_generated_mtp():
    from tensor_cast.transformers.builtin_model.bailing_v3 import (
        BailingV3KimiDeltaAttentionTensorCast,
        _patch_bailing_v3_model,
    )

    class DecoderLayer(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.self_attn = attention

    class MtpLayer(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.mtp_block = torch.nn.Module()
            self.mtp_block.attention = attention

    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DecoderLayer(_make_fake_bailing_v3_kda(0))])

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = BaseModel()
            self.mtp = torch.nn.Module()
            self.mtp.layers = torch.nn.ModuleList([MtpLayer(_make_fake_bailing_v3_kda(1))])

        def forward(self, **kwargs):
            return kwargs

    inner = Wrapper()
    model = SimpleNamespace(
        _inner=inner,
        unwrap=lambda: inner.model,
        num_nextn_predict_layers=1,
    )

    _patch_bailing_v3_model(model)

    assert isinstance(inner.model.layers[0].self_attn, BailingV3KimiDeltaAttentionTensorCast)
    assert isinstance(inner.mtp.layers[0].mtp_block.attention, BailingV3KimiDeltaAttentionTensorCast)
