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

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from parameterized import parameterized
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.deepseek_v4 import HyperConnectedMultiTokenPredictorLayer
from tensor_cast.layers.mtp import (
    HcStreamMultiTokenPredictorLayer,
    MultiTokenPredictorLayer,
    Qwen4ExpMultiTokenPredictorLayer,
    _build_position_embeddings,
    _find_text_rotary_emb,
    _resolve_mtp_layer_cls,
    _rotary_emb_accepts_dtype,
)
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.patch_torch import patch_torch
from tensor_cast.transformers.custom_model_registry import get_mtp_block_module_name

from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
    has_submodule_with_cls_name,
)

# Core MTP configuration assertions were moved to the unified entry in test_layers.py.


class TestResolveMtpLayerCls(unittest.TestCase):
    def test_v3_style_returns_base_layer(self):
        # Real namespaces, not MagicMock: auto-generated mock
        # attributes would poison the hc_mult fallback chain
        # (int(mock) == 0).
        hf_config = SimpleNamespace(hc_mult=1)
        mtp_block = SimpleNamespace(hc_mult=None)
        assert _resolve_mtp_layer_cls(hf_config, mtp_block) is MultiTokenPredictorLayer

    def test_v4_hc_mult_returns_hyper_connected_layer(self):
        mtp_block = SimpleNamespace(hc_mult=4)
        assert _resolve_mtp_layer_cls(SimpleNamespace(), mtp_block) is HyperConnectedMultiTokenPredictorLayer

    def test_block_with_own_hc_sublayers_returns_hc_stream_layer(self):
        class _HcSublayer(torch.nn.Module):
            def __init__(self, mult):
                super().__init__()
                self.hc_mult = mult

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_hc = _HcSublayer(4)
                self.ffn_hc = _HcSublayer(4)

        mtp_block = _Block()
        assert _resolve_mtp_layer_cls(MagicMock(hc_mult=4), mtp_block) is HcStreamMultiTokenPredictorLayer

    def test_hc_stream_layer_expands_entry_and_reduces_exit(self):
        class _HcSublayer(torch.nn.Module):
            def __init__(self, mult):
                super().__init__()
                self.hc_mult = mult

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_hc = _HcSublayer(4)
                self.ffn_hc = _HcSublayer(4)
                self.seen_shape = None

            def forward(self, hidden_states, **kwargs):
                self.seen_shape = tuple(hidden_states.shape)
                return hidden_states

        block = _Block()
        hf_config = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-5)
        layer = HcStreamMultiTokenPredictorLayer(hf_config, block)
        assert layer.hc_mult == 4

        hidden_states = torch.randn(2, 3, 8)
        output = layer(hidden_states, torch.zeros(2, 3, dtype=torch.long), hidden_states)

        # Block receives HC-expanded streams; the layer returns the
        # parameter-free mhc_head reduction back to [B, S, D].
        assert block.seen_shape == (2, 3, 4, 8)
        assert tuple(output.shape) == (2, 3, 8)

    def test_hc_stream_layer_falls_back_to_block_and_config_mult(self):
        # HC submodules without an exposed hc_mult must not silently
        # degrade the layer's mult: the fallback chain mirrors
        # _resolve_mtp_layer_cls (block attrs, then HF config).
        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_hc = torch.nn.Module()
                self.ffn_hc = torch.nn.Module()
                self.hc_mult = None

        hf_config = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-5, hc_mult=4)

        assert HcStreamMultiTokenPredictorLayer(hf_config, _Block()).hc_mult == 4

    def test_resolve_routes_on_submodule_mult_when_block_and_config_lack_it(self):
        # A model may declare the multiplier only on its HC
        # submodules: the routing decision must read the same
        # source chain as HcStreamMultiTokenPredictorLayer.__init__
        # (PR !1000 review), otherwise the block is routed to the
        # base layer and receives unexpanded [B,S,D] streams.
        class _HcSublayer(torch.nn.Module):
            def __init__(self, mult):
                super().__init__()
                self.hc_mult = mult

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_hc = _HcSublayer(4)
                self.ffn_hc = _HcSublayer(1)

        mtp_block = _Block()
        assert _resolve_mtp_layer_cls(SimpleNamespace(), mtp_block) is HcStreamMultiTokenPredictorLayer
        # The routed layer reports the same mult it routed on.
        layer = HcStreamMultiTokenPredictorLayer(SimpleNamespace(hidden_size=8, rms_norm_eps=1e-5), mtp_block)
        assert layer.hc_mult == 4

    def test_qwen4_hc_count_routes_to_qwen4_exp_layer(self):
        # qwen4_exp MTP blocks expose the flattened HC stream count on their
        # attn_hyper_connection mixer; hc_count > 1 requires the dedicated
        # layer that expands/reduces [B,S,hc*D] bundles.
        pytest.importorskip("transformers", minversion="5.16")
        pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
        mtp_block = SimpleNamespace(attn_hyper_connection=SimpleNamespace(hc_count=4))
        assert _resolve_mtp_layer_cls(SimpleNamespace(), mtp_block) is Qwen4ExpMultiTokenPredictorLayer

        plain_block = SimpleNamespace(attn_hyper_connection=SimpleNamespace(hc_count=1))
        assert _resolve_mtp_layer_cls(SimpleNamespace(), plain_block) is MultiTokenPredictorLayer


class TestQwen4ExpMultiTokenPredictorLayer(unittest.TestCase):
    def _make_layer(self, block):
        pytest.importorskip("transformers", minversion="5.16")
        pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
        hf_config = SimpleNamespace(
            hidden_size=8,
            rms_norm_eps=1e-5,
            hc_count=4,
            hc_lowrank=2,
        )
        return Qwen4ExpMultiTokenPredictorLayer(hf_config, block)

    def test_expands_streams_and_reduces_with_hc_mixer(self):
        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_shape = None

            def forward(self, hidden_states, **kwargs):
                self.seen_shape = tuple(hidden_states.shape)
                return hidden_states

        block = _Block()
        layer = self._make_layer(block)
        assert layer.hc_count == 4

        hidden_states = torch.randn(2, 3, 8)
        output = layer(hidden_states, torch.zeros(2, 3, dtype=torch.long), hidden_states)

        # The block consumes the flattened HC stream bundle
        # [B, S, hc_count * D]; the layer reduces it back to [B, S, D]
        # with its own hyper_connection_mixer.
        assert block.seen_shape == (2, 3, 32)
        assert tuple(output.shape) == (2, 3, 8)

    def test_unwraps_tuple_block_output(self):
        class _Block(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return (hidden_states, None)

        layer = self._make_layer(_Block())
        hidden_states = torch.randn(1, 2, 8)
        output = layer(hidden_states, torch.zeros(1, 2, dtype=torch.long), hidden_states)

        assert tuple(output.shape) == (1, 2, 8)


class TestRotaryEmbeddingSignature(unittest.TestCase):
    def test_two_argument_rotary_embedding_does_not_accept_dtype(self):
        class RotaryEmbedding(torch.nn.Module):
            def forward(self, hidden_states, position_ids):
                return hidden_states, position_ids

        assert not _rotary_emb_accepts_dtype(RotaryEmbedding())

    def test_minimax_m3_style_rotary_embedding_accepts_dtype(self):
        class RotaryEmbedding(torch.nn.Module):
            def forward(self, hidden_states, position_ids, dtype):
                return hidden_states.to(dtype), position_ids

        assert _rotary_emb_accepts_dtype(RotaryEmbedding())

    def test_two_argument_rotary_embedding_call_is_preserved(self):
        class RotaryEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.call_arg_count = 0

            def forward(self, hidden_states, position_ids):
                self.call_arg_count = 2
                return hidden_states, position_ids

        rotary_emb = RotaryEmbedding()
        hidden_states = torch.empty(1, 2, dtype=torch.float16)
        position_ids = torch.empty(1, 2, dtype=torch.long)

        _build_position_embeddings(rotary_emb, False, hidden_states, position_ids)

        assert rotary_emb.call_arg_count == 2

    def test_dtype_rotary_embedding_receives_hidden_state_dtype(self):
        class RotaryEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received_dtype = None

            def forward(self, hidden_states, position_ids, dtype):
                self.received_dtype = dtype
                return hidden_states.to(dtype), position_ids

        rotary_emb = RotaryEmbedding()
        hidden_states = torch.empty(1, 2, dtype=torch.bfloat16)
        position_ids = torch.empty(1, 2, dtype=torch.long)

        _build_position_embeddings(rotary_emb, True, hidden_states, position_ids)

        assert rotary_emb.received_dtype is torch.bfloat16

    def test_vl_mtp_prefers_language_rotary_embedding(self):
        class RotaryEmbedding(torch.nn.Module):
            pass

        class VlModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = torch.nn.Module()
                self.vision_tower.rotary_emb = RotaryEmbedding()
                self.language_model = torch.nn.Module()
                self.language_model.rotary_emb = RotaryEmbedding()

        model = VlModel()

        assert _find_text_rotary_emb(model) is model.language_model.rotary_emb


class MtpTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    @classmethod
    def _get_model(cls, user_config: UserInputConfig):
        return get_cached_build_model(cls._model_cache, user_config)

    def setUp(self):
        torch.compiler.reset()


class MtpTestCase(MtpTestMixin, unittest.TestCase):
    def _run_test_deepseek_prefill_without_kvcache(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        num_tokens = 100
        model = self._get_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)

        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids, sampling_metadata=SamplingMetadata())
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    def _run_test_deepseek_prefill_with_kvcache(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        model = self._get_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(query_start_loc=attn_meta.query_start_loc),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    def _run_test_deepseek_decode_with_kvcache(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        model = self._get_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model,
            model.model_config,
            query_len_1=num_mtp_layers + 1,
            query_len_2=num_mtp_layers + 1,
        )
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc,
                    selected_token_indices=None,
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    def _run_test_automatic_mtp_mode(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        model = self._get_model(user_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc,
                    selected_token_indices=None,
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek_prefill_without_kvcache(self, model_id, do_compile):
        self._run_test_deepseek_prefill_without_kvcache(model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek_prefill_with_kvcache(self, model_id, do_compile):
        self._run_test_deepseek_prefill_with_kvcache(model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek_decode_with_kvcache(self, model_id, do_compile):
        self._run_test_deepseek_decode_with_kvcache(model_id, do_compile)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3.5-27B", False],
            ["zai-org/GLM-4.5", False],
        ]
    )
    def test_automatic_mtp_mode(self, model_id, do_compile):
        self._run_test_automatic_mtp_mode(model_id, do_compile)


@pytest.mark.nightly
class MtpNightlyTestCase(MtpTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek_prefill_without_kvcache(self, model_id, do_compile):
        MtpTestCase._run_test_deepseek_prefill_without_kvcache(self, model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek_prefill_with_kvcache(self, model_id, do_compile):
        MtpTestCase._run_test_deepseek_prefill_with_kvcache(self, model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek_decode_with_kvcache(self, model_id, do_compile):
        MtpTestCase._run_test_deepseek_decode_with_kvcache(self, model_id, do_compile)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3.5-27B", True],
        ]
    )
    def test_automatic_mtp_mode(self, model_id, do_compile):
        MtpTestCase._run_test_automatic_mtp_mode(self, model_id, do_compile)
