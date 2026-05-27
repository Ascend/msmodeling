import unittest

import torch
from parameterized import parameterized

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.patch_torch import patch_torch
from tensor_cast.transformers.custom_model_registry import get_mtp_block_module_name
from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


class MtpTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_prefill_without_kvcache(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        num_tokens = 100
        model = build_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)

        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids, sampling_metadata=SamplingMetadata())
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_prefill_with_kvcache(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        model = build_model(user_config)
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

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_decode_with_kvcache(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        model = build_model(user_config)
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
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3.5-27B", False],
            ["Qwen/Qwen3.5-27B", True],
            ["zai-org/GLM-4.5", False],
        ]
    )
    def test_automatic_mtp_mode(self, model_id, do_compile):
        num_mtp_layers = 3
        user_config = UserInputConfig(model_id=model_id, num_mtp_tokens=num_mtp_layers, do_compile=do_compile)
        model = build_model(user_config)

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
