import unittest

import pytest
import torch
from parameterized import parameterized
from tensor_cast.compilation import get_backend
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.patch_torch import patch_torch

from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
    has_submodule_with_cls_name,
)

# TODO: we comment all the compilation cases for large MoE models due to slow compilation time
#       need to find out solution to speed things up...


class ModelLoadTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    @classmethod
    def _get_model(cls, user_config: UserInputConfig):
        return get_cached_build_model(cls._model_cache, user_config)

    def setUp(self):
        torch.compiler.reset()


class ModelLoadTestCase(ModelLoadTestMixin, unittest.TestCase):
    def _run_test_vanilla_transformer_model(self, model_id, do_compile):
        num_tokens = 100
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile)
        model = self._get_model(user_config)
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model.vocab_size))

    def _run_test_deepseek_without_kvcache(self, model_id, do_compile):
        num_tokens = 100
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile)
        model = self._get_model(user_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model.vocab_size))

    def _run_test_deepseek_with_kvcache(self, model_id, do_compile):
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile)
        model = self._get_model(user_config)
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
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

    def _run_test_prefill_without_kvcache(self, model_id, do_compile):
        num_tokens = 100
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile, num_hidden_layers_override=2)
        model = self._get_model(user_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

    def _run_test_prefill_with_kvcache(self, model_id, do_compile):
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile, num_hidden_layers_override=2)
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
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-235B-A22B", False],
            ["zai-org/GLM-4.5", False],
            ["baidu/ERNIE-4.5-300B-A47B-PT", False],
        ]
    )
    def test_vanilla_transformer_model(self, model_id, do_compile):
        self._run_test_vanilla_transformer_model(model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek_without_kvcache(self, model_id, do_compile):
        self._run_test_deepseek_without_kvcache(model_id, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek_with_kvcache(self, model_id, do_compile):
        self._run_test_deepseek_with_kvcache(model_id, do_compile)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-235B-A22B", False],
            ["zai-org/GLM-4.5", False],
        ]
    )
    def test_prefill_without_kvcache(self, model_id, do_compile):
        self._run_test_prefill_without_kvcache(model_id, do_compile)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-235B-A22B", False],
            ["zai-org/GLM-4.5", False],
        ]
    )
    def test_prefill_with_kvcache(self, model_id, do_compile):
        self._run_test_prefill_with_kvcache(model_id, do_compile)

    def _run_test_qwen3_next_with_kvcache(self, model_id, do_compile):
        user_config = UserInputConfig(model_id=model_id, do_compile=do_compile, num_hidden_layers_override=2)
        model = self._get_model(user_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
        if do_compile:
            model = torch.compile(
                model,
                backend=get_backend(),
                dynamic=True,
                fullgraph=False,  # data dependency code in QwenNext
            )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                cache_position=torch.arange(0, num_tokens, dtype=torch.long, device="cpu"),
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

    def _run_test_qwen3_5(self, model_id, remote_source):
        user_config = UserInputConfig(model_id=model_id, do_compile=False, remote_source=remote_source)
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
                cache_position=torch.arange(0, num_tokens, dtype=torch.long, device="cpu"),
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))


@pytest.mark.nightly
class ModelLoadQwen35NightlyTestCase(ModelLoadTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["Qwen/Qwen3.5-397B-A17B", "modelscope"],
        ]
    )
    def test_qwen3_5(self, model_id, remote_source):
        ModelLoadTestCase._run_test_qwen3_5(self, model_id, remote_source)


@pytest.mark.nightly
class ModelLoadNightlyTestCase(ModelLoadTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
            ["baidu/ERNIE-4.5-300B-A47B-PT"],
        ]
    )
    def test_vanilla_transformer_model(self, model_id):
        ModelLoadTestCase._run_test_vanilla_transformer_model(self, model_id, True)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek_without_kvcache(self, model_id):
        ModelLoadTestCase._run_test_deepseek_without_kvcache(self, model_id, True)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek_with_kvcache(self, model_id):
        ModelLoadTestCase._run_test_deepseek_with_kvcache(self, model_id, True)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_prefill_without_kvcache(self, model_id):
        ModelLoadTestCase._run_test_prefill_without_kvcache(self, model_id, True)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_prefill_with_kvcache(self, model_id):
        ModelLoadTestCase._run_test_prefill_with_kvcache(self, model_id, True)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
        ]
    )
    def _test_qwen3_next_with_kvcache(self, model_id):
        ModelLoadTestCase._run_test_qwen3_next_with_kvcache(self, model_id, True)
