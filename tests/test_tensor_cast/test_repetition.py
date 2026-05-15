import unittest

import torch
from parameterized import parameterized

from tensor_cast.compilation import get_backend
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.internal import CopyLayerWrapper
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_mtp_block_module_name
from tensor_cast.transformers.model import TransformerModel
from .test_common import (
    assert_close,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


class RepetitionTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def check_num_effective_layers(self, layers, expected_num):
        count = sum(1 for layer in layers if not isinstance(layer, CopyLayerWrapper))
        self.assertEqual(count, expected_num, f"{layers}")

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
        ]
    )
    def test_vanilla_transformer_model(self, model_id, do_compile):
        num_tokens = 100
        # Note that specifying `AttentionTensorCast` as the `attention_cls`
        # is needed otherwise CSE would optimize out the attention mask
        # computation from the original attention implementation across layers,
        # resulting in larger op count gap between original trace and repetitive trace.
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=3,
        )
        model_config_with_repeats = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=3,
            enable_repetition=True,
        )
        model = TransformerModel(model_id, model_config)
        model_with_repeats = TransformerModel(model_id, model_config_with_repeats)
        self.check_num_effective_layers(model_with_repeats.unwrap().layers, 1)
        if do_compile:
            model = torch.compile(model, backend=get_backend(), dynamic=True, fullgraph=True)
            model_with_repeats = torch.compile(model_with_repeats, backend=get_backend(), dynamic=True, fullgraph=True)
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model.vocab_size))

        with (
            Runtime(perf_model, device_profile, MemoryTracker(device_profile)) as runtime_with_repeats,
            torch.no_grad(),
        ):
            outputs = model_with_repeats.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model_with_repeats.vocab_size))

        # NOTE: we might miss some cross-layer fusion patterns with repetitions
        #       so we allow some errors here.
        assert_close(
            self,
            len(runtime.event_list),
            len(runtime_with_repeats.event_list),
            rtol=0.027 if do_compile else 0,
        )
        runtime_cost_s = runtime.total_execution_time_s()[perf_model.name]
        runtime_cost_with_repeats_s = runtime_with_repeats.total_execution_time_s()[perf_model.name]
        assert_close(
            self,
            runtime_cost_s,
            runtime_cost_with_repeats_s,
            rtol=0.01 if do_compile else 0,
        )
        peak_mem_usage = runtime.memory_tracker.peak_mem_usage()
        peak_mem_usage_with_repeats = runtime_with_repeats.memory_tracker.peak_mem_usage()
        assert_close(
            self,
            peak_mem_usage,
            peak_mem_usage_with_repeats,
        )

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek_with_kvcache(self, model_id):
        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
        )

        model = build_model(user_config)

        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)

        self.check_num_effective_layers(model.unwrap().layers, 2)
        self.check_num_effective_layers(model._inner.mtp.layers, 1)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile) as runtime,
            torch.no_grad(),
        ):
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
        result = runtime.table_averages()
        start_str = "tensor_cast.multihead_latent_attention.default"
        end_str = "64"
        found = any(
            line.strip().startswith(start_str) and line.strip().endswith(end_str) for line in result.splitlines()
        )
        self.assertTrue(found, result)
