import unittest

import pytest
import torch
from parameterized import parameterized
from tensor_cast import ops  # noqa: F401
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.parallel_linear import ColumnParallelLinear
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.patch_torch import patch_torch
from tensor_cast.transformers.custom_model_registry import get_mtp_block_module_name
from tensor_cast.transformers.utils import strip_module_name

from .test_common import create_mla_metadata_and_kv_cache, get_cached_build_model, has_submodule_with_cls_name

# Core MTP+EP configuration assertions were moved to the unified entry in test_layers.py.


class MtpEpTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    @classmethod
    def _get_model(cls, user_config: UserInputConfig):
        return get_cached_build_model(cls._model_cache, user_config)

    def setUp(self):
        torch.compiler.reset()


class MtpTestCase(MtpEpTestMixin, unittest.TestCase):
    def _run_test_deepseek_prefill_without_kvcache(self, model_id, parallel_configuration, do_compile):
        num_tokens = 100

        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=do_compile,
            world_size=parallel_configuration[0],
            ep_size=parallel_configuration[0] if parallel_configuration[1] else 1,
            moe_dp_size=1 if parallel_configuration[1] else parallel_configuration[0],
            moe_tp_size=1,
        )
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

    def _run_test_deepseek_prefill_with_kvcache(self, model_id, parallel_configuration, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=do_compile,
            world_size=parallel_configuration[0],
            ep_size=parallel_configuration[0] if parallel_configuration[1] else 1,
            moe_dp_size=1 if parallel_configuration[1] else parallel_configuration[0],
            moe_tp_size=1,
        )
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

    def _run_test_deepseek_decode_with_kvcache(self, model_id, parallel_configuration, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=do_compile,
            world_size=parallel_configuration[0],
            ep_size=parallel_configuration[0] if parallel_configuration[1] else 1,
            moe_dp_size=1 if parallel_configuration[1] else parallel_configuration[0],
            moe_tp_size=1,
        )
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

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True), False],
        ]
    )
    def test_deepseek_prefill_without_kvcache(self, model_id, parallel_configuration, do_compile):
        self._run_test_deepseek_prefill_without_kvcache(model_id, parallel_configuration, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True), False],
        ]
    )
    def test_deepseek_prefill_with_kvcache(self, model_id, parallel_configuration, do_compile):
        self._run_test_deepseek_prefill_with_kvcache(model_id, parallel_configuration, do_compile)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.2", 128, 128],
        ]
    )
    def test_mtp_self_attn_q_b_proj_sharded_by_tp(self, model_id, tp_size, ep_size):
        """MTP block's self_attn.q_b_proj should be TP-sharded when tp>1."""
        num_mtp_layers = 1
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=False,
            world_size=tp_size,
            tp_size=tp_size,
            ep_size=ep_size,
            moe_dp_size=1,
            moe_tp_size=1,
        )
        model = self._get_model(user_config)

        text_config = model.text_config
        assert text_config.num_attention_heads % tp_size == 0
        qk_head_dim = text_config.qk_nope_head_dim + text_config.qk_rope_head_dim
        expected_out_features = text_config.num_attention_heads * qk_head_dim // tp_size
        expected_shape = torch.Size([expected_out_features, text_config.q_lora_rank])

        sharded_q_b_projs = []
        for name, module in model.named_modules():
            if not isinstance(module, ColumnParallelLinear):
                continue
            stripped = strip_module_name(name)
            if stripped.startswith("mtp.layers.") and stripped.endswith(".self_attn.q_b_proj"):
                sharded_q_b_projs.append((stripped, module))

        self.assertEqual(
            len(sharded_q_b_projs),
            num_mtp_layers,
            f"expected {num_mtp_layers} sharded MTP q_b_proj, "
            f"found {len(sharded_q_b_projs)}: {[n for n, _ in sharded_q_b_projs]}",
        )
        for name, module in sharded_q_b_projs:
            self.assertEqual(
                module.out_features_per_partition,
                expected_out_features,
                f"{name} out_features_per_partition mismatch",
            )
            inner_weight = getattr(module._inner, module.inner_weight_name)
            self.assertEqual(
                inner_weight.shape,
                expected_shape,
                f"{name} {module.inner_weight_name} shape "
                f"{tuple(inner_weight.shape)} != expected {tuple(expected_shape)}",
            )

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True), False],
        ]
    )
    def test_deepseek_decode_with_kvcache(self, model_id, parallel_configuration, do_compile):
        self._run_test_deepseek_decode_with_kvcache(model_id, parallel_configuration, do_compile)


@pytest.mark.nightly
class MtpEpNightlyTestCase(MtpEpTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True)],
            ["deepseek-ai/DeepSeek-V3.1", (16, False)],
            ["moonshotai/Kimi-K2-Base", (16, True)],
        ]
    )
    def test_deepseek_prefill_without_kvcache(self, model_id, parallel_configuration):
        MtpTestCase._run_test_deepseek_prefill_without_kvcache(self, model_id, parallel_configuration, True)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True)],
            ["deepseek-ai/DeepSeek-V3.1", (16, False)],
            ["moonshotai/Kimi-K2-Base", (16, True)],
        ]
    )
    def test_deepseek_prefill_with_kvcache(self, model_id, parallel_configuration):
        MtpTestCase._run_test_deepseek_prefill_with_kvcache(self, model_id, parallel_configuration, True)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True)],
            ["deepseek-ai/DeepSeek-V3.1", (16, False)],
            ["moonshotai/Kimi-K2-Base", (16, True)],
        ]
    )
    def test_deepseek_decode_with_kvcache(self, model_id, parallel_configuration):
        MtpTestCase._run_test_deepseek_decode_with_kvcache(self, model_id, parallel_configuration, True)
