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

"""Official MiniMax DSpark architecture and compiled shape regressions."""

import json
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.compilation import get_backend
from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.dflash import build_draft_hf_config, load_dflash_draft_config_dict
from tensor_cast.layers.dspark import apply_cli_overrides_to_dspark_config, build_dspark_draft_and_wrapper
from tensor_cast.layers.minimax_m31_dspark import (
    MINIMAX_M31_DRAFT_CONFIG,
    MiniMaxM31DsparkDraft,
    MiniMaxM31DsparkWrapper,
    normalize_minimax_draft_config,
)
from tensor_cast.layers.minimax_m3_attention import GemmaRMSNormFusedWrapper
from tensor_cast.model_config import DsparkConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


def test_official_draft_config():
    source = load_dflash_draft_config_dict(str(MINIMAX_M31_DRAFT_CONFIG))
    assert source["num_hidden_layers"] == 5
    assert source["hidden_size"] == 6144
    assert (source["num_attention_heads"], source["num_key_value_heads"]) == (64, 4)
    assert source["intermediate_size"] == 12288
    assert source["rotary_dim"] == 64
    assert source["rope_parameters"]["partial_rotary_factor"] == 0.5
    assert source["sliding_window"] == 4096
    assert source["block_size"] == 8
    assert source["dflash_config"]["target_layer_ids"] == [3, 17, 31, 45, 59]


def test_minimax_dspark_builder_selects_native_draft_and_shares_target_weights(tmp_path):
    source = json.loads(MINIMAX_M31_DRAFT_CONFIG.read_text())
    source["text_config"].update(
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rotary_dim=8,
        dense_intermediate_size=128,
    )
    source["dspark_ffn_hidden_size"] = 128
    config_path = tmp_path / "minimax_draft.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    scfg = DsparkConfig(dspark_block_size=8, draft_model_config_path=str(config_path), context_length=16, markov_rank=8)
    target = torch.nn.Identity()
    embed = torch.nn.Embedding(128, 64)
    lm_head = torch.nn.Linear(64, 128, bias=False)
    model = SimpleNamespace(_inner=target, embed_tokens=embed, lm_head=lm_head)
    target_layers = torch.nn.ModuleList([torch.nn.Identity()])
    wrapper = build_dspark_draft_and_wrapper(
        model,
        scfg,
        SimpleNamespace(),
        num_target_hidden_layers=60,
        target_hidden_size=64,
        target_vocab_size=128,
        target_max_position_embeddings=256,
        dtype=torch.bfloat16,
        target_layers=target_layers,
    )
    assert isinstance(wrapper, MiniMaxM31DsparkWrapper)
    assert isinstance(wrapper.draft, MiniMaxM31DsparkDraft)
    assert wrapper._inner is target
    assert wrapper._target_layers is target_layers
    assert wrapper.draft.embed_tokens is embed
    assert wrapper.draft.lm_head is lm_head
    assert embed.weight.dtype == lm_head.weight.dtype == torch.float32
    assert wrapper.draft.hidden_norm.weight.dtype == torch.bfloat16
    assert scfg.aux_hidden_state_layer_ids == [3, 17, 31, 45, 59]
    assert wrapper._draft_query_length() == 7


@pytest.mark.parametrize("sparse", [True, False])
def test_default_profile_only_selected_for_all_sparse_minimax(sparse):
    resolver = object.__new__(ConfigResolver)
    layers = ["minimax_m3_sparse"] * 60
    if not sparse:
        layers[:3] = ["full_attention"] * 3
    resolver.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(model_type="minimax_m3_vl_text", layer_types=layers)
    )
    resolver.model_config = SimpleNamespace(mtp_config=None, dflash_config=None)
    resolver.update_dspark_config(dspark=True, context_length=131072)
    scfg = resolver.model_config.dspark_config
    if sparse:
        assert scfg.draft_model_config_path == str(MINIMAX_M31_DRAFT_CONFIG)
        assert scfg.num_draft_layers == 5
        assert scfg.aux_hidden_state_layer_ids == [3, 17, 31, 45, 59]
        assert scfg.sliding_window == 4096
    else:
        assert scfg.draft_model_config_path is None
        assert scfg.num_draft_layers == 6


def _tiny_draft():
    scfg = DsparkConfig(
        dspark_block_size=8, draft_model_config_path=str(MINIMAX_M31_DRAFT_CONFIG), context_length=16, markov_rank=8
    )
    apply_cli_overrides_to_dspark_config(scfg)
    dcfg = scfg.to_dflash_config()
    config = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.rotary_dim = 8
    config.intermediate_size = config.dense_intermediate_size = 128
    draft = MiniMaxM31DsparkDraft(config, dcfg, scfg, layer_idx_offset=60)
    draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
    return scfg, dcfg, config, draft


@pytest.mark.parametrize("packed_lengths", [None, (3, 1), (3, 1) * 5])
def test_native_draft_emits_five_attention_and_clipped_swiglu_blocks(packed_lengths):
    _, _, config, draft = _tiny_draft()
    assert isinstance(draft.hidden_norm, GemmaRMSNormFusedWrapper)
    assert all(isinstance(layer.dflash_block.self_attn.q_norm, GemmaRMSNormFusedWrapper) for layer in draft.layers)
    assert draft.layers[0].dflash_block.mlp.gate_proj.out_features == 128
    batch = 1 if packed_lengths is None else len(packed_lengths)
    rows = 8 if packed_lengths is None else sum(packed_lengths)
    noise = torch.randn(batch, 7, 64)
    aux = [torch.randn(1, rows, 64) for _ in draft.target_layer_ids]
    context = draft.build_context_features(aux, rows, align_seq=False)
    positions = torch.arange(7).expand(batch, -1)
    attention_layers = {i: AttentionTensorCast() for i in range(60, 65)}
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime, torch.no_grad():
        if packed_lengths is None:
            out = draft(noise, context, positions, attention_by_layers=attention_layers)
        else:
            prepared = draft.prepare_context_kv(
                context, batch=batch, block=7, attn_use_configured_context=True, packed_lengths=packed_lengths
            )
            out = draft.run_noise_decoder(noise, positions, prepared, attention_by_layers=attention_layers)
    assert out.shape == (batch, 7, 64)
    events = [event.op_invoke_info for event in runtime.event_list]
    attention = [event for event in events if str(event.func) == "tensor_cast.attention.default"]
    assert len(attention) == 5
    assert all(event.args[0].shape[0] == batch * 7 for event in attention)
    activations = [event for event in events if str(event.func) == "tensor_cast.m3_swiglu.default"]
    assert len(activations) == 5
    assert all(event.args[2:] == (config.swiglu_alpha, config.swiglu_limit) for event in activations)
    rope = [event for event in events if str(event.func) == "tensor_cast.apply_rope_single.default"]
    assert len(rope) == 1
    assert rope[0].args[0].shape[-2:] == (rows * 5, config.rotary_dim)
    if packed_lengths is not None:
        # Packed context plus decoder metadata emit the same ranges for both batch sizes.
        assert sum(str(event.func) == "aten.arange.default" for event in events) == 26
        writes = [e for e in events if str(e.func) == "tensor_cast.reshape_and_cache.default"][:5]
        assert len(writes) == 5
        for event in writes:
            assert event.args[0].shape[0] == rows
            assert event.args[2].shape[1] == batch
            assert event.args[3].shape == (rows,)


def test_packed_heterogeneous_minimax_draft_keeps_constant_op_counts_across_batch_sizes():
    from collections import Counter

    _, _, _, draft = _tiny_draft()
    histograms = []
    for packed_lengths in ((3, 1), (3, 1) * 5):
        batch = len(packed_lengths)
        rows = sum(packed_lengths)
        noise = torch.randn(batch, 7, 64)
        aux = [torch.randn(1, rows, 64) for _ in draft.target_layer_ids]
        context = draft.build_context_features(aux, rows, align_seq=False)
        positions = torch.arange(7).expand(batch, -1)
        attention_layers = {i: AttentionTensorCast() for i in range(60, 65)}
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime, torch.no_grad():
            prepared = draft.prepare_context_kv(
                context, batch=batch, block=7, attn_use_configured_context=True, packed_lengths=packed_lengths
            )
            draft.run_noise_decoder(noise, positions, prepared, attention_by_layers=attention_layers)
        histograms.append(Counter(str(event.op_invoke_info.func) for event in runtime.event_list))
    assert histograms[0] == histograms[1]
    assert sum(histograms[0].values()) == sum(histograms[1].values())
    for op_name in ("aten.slice.Tensor", "aten.cat.default", "aten.arange.default"):
        assert histograms[0][op_name] == histograms[1][op_name]


def test_native_target_block_eight_draft_and_sampling_seven():
    scfg, dcfg, config, draft = _tiny_draft()
    wrapper = MiniMaxM31DsparkWrapper(scfg, dcfg, config, torch.nn.Identity(), draft, config)
    assert wrapper.dflash_block_size == 8
    assert wrapper._draft_query_length() == 7
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE), torch.no_grad():
        tokens, confidence = wrapper._sample_sequential(
            torch.randn(1, 7, 64), torch.randn(1, 7, 128), 1, 7, torch.zeros(1, 8, dtype=torch.long)
        )
    assert tokens.shape == confidence.shape == (1, 7)


@pytest.mark.parametrize("second_output", [False, True])
def test_compiled_gemma_residual_preserves_batch_axis(second_output):
    class ResidualNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(64))

        def forward(self, x, residual):
            summed = x + residual
            norm = torch.ops.tensor_cast.rms_norm(summed, self.weight + 1.0, 1e-6)
            return (norm, summed) if second_output else norm

    model = torch.compile(ResidualNorm(), backend=get_backend(), fullgraph=True)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE), torch.no_grad():
        out = model(torch.randn(1, 7, 64), torch.randn(1, 7, 64))
    if second_output:
        assert all(item.shape == (1, 7, 64) for item in out)
    else:
        assert out.shape == (1, 7, 64)


@pytest.mark.parametrize("batch", [1, 2])
def test_compiled_tp_draft_residual_restores_batch_sequence_shape(batch):
    class TPResidualNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(64))

        def forward(self, x, residual):
            flat = x.reshape(-1, 64)
            restored = flat.view(batch, 7, 64)
            reduced = torch.ops.tensor_cast.all_reduce(restored, 0, [0, 1, 2, 3])
            summed = reduced + residual
            return torch.ops.tensor_cast.rms_norm(summed, self.weight + 1.0, 1e-6), summed

    model = torch.compile(TPResidualNorm(), backend=get_backend(), fullgraph=True)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE), torch.no_grad():
        out = model(torch.randn(batch * 7, 64), torch.randn(batch, 7, 64))
    assert all(item.shape == (batch, 7, 64) for item in out)


def test_draft_rotary_dim_derived_when_not_explicit():
    source = json.loads(MINIMAX_M31_DRAFT_CONFIG.read_text())
    del source["text_config"]["rotary_dim"]
    assert normalize_minimax_draft_config(source)["rotary_dim"] == 64
    source["text_config"]["rotary_dim"] = 32
    assert normalize_minimax_draft_config(source)["rotary_dim"] == 32


@pytest.mark.parametrize("shape", [(1, 8, 4), (8, 1, 4), (8, 2, 2), (1, 0, 4)])
def test_peep_hole_only_sinks_nonempty_leading_singleton(shape):
    from torch.fx.experimental.proxy_tensor import make_fx
    from tensor_cast.compilation.passes.peep_hole_pass import PeepHolePass

    def forward(x):
        return torch.ops.tensor_cast.all_reduce(x.view(shape), 0, [0, 1])

    x = torch.empty(shape[0] * shape[1], shape[2], device="meta")
    graph = make_fx(forward)(x)
    graph = PeepHolePass()(graph)
    collective = next(n for n in graph.graph.nodes if n.target == torch.ops.tensor_cast.all_reduce.default)
    should_sink = shape == (1, 8, 4)
    assert (collective.args[0].op == "placeholder") == should_sink
    assert graph(x).shape == shape
