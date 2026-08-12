import unittest
from unittest.mock import MagicMock

import pytest
import torch
from parameterized import parameterized
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.deepseek_v4 import HyperConnectedMultiTokenPredictorLayer
from tensor_cast.layers.mtp import (
    MultiTokenPredictorLayer,
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
        hf_config = MagicMock(hc_mult=1)
        mtp_block = MagicMock(hc_mult=None)
        assert _resolve_mtp_layer_cls(hf_config, mtp_block) is MultiTokenPredictorLayer

    def test_v4_hc_mult_returns_hyper_connected_layer(self):
        mtp_block = MagicMock(hc_mult=4)
        assert _resolve_mtp_layer_cls(MagicMock(), mtp_block) is HyperConnectedMultiTokenPredictorLayer


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
