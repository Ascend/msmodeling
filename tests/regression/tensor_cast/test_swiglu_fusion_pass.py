import unittest

import pytest
import torch
from parameterized import parameterized
from tensor_cast.compilation import get_backend
from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.model_config import ModelConfig
from tensor_cast.patch_torch import patch_torch
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel
from tests.helpers.model_cache import user_config_build_cache_key

from .test_common import (
    count_events,
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
)

# Core SwiGLU fusion-entry assertions were moved to the unified entry in test_ops.py.


class SwiGLUFusionPassTestMixin:
    """Unified, parameterized test verifying SwiGLU fusion presence.

    Simulates tensor_cast.scripts.text_generate via ModelRunner, compiles models,
    and asserts the fused op `tensor_cast.swiglu` appears in
    runtime table results.
    """

    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    def setUp(self):
        pass

    @classmethod
    def _get_compiled_qwen_swiglu_model(
        cls,
        model_id: str,
        linear_act: QuantizeLinearAction,
    ) -> TransformerModel:
        key = ("compiled_qwen_swiglu", model_id, linear_act)
        if key not in cls._model_cache:
            user_input = UserInputConfig(
                model_id=model_id,
                do_compile=True,
                quantize_linear_action=linear_act,
            )
            config_resolver = ConfigResolver(user_input=user_input)
            model_config = config_resolver.resolve()
            model = TransformerModel(model_id, model_config)
            cls._model_cache[key] = torch.compile(model, backend=get_backend(), fullgraph=True)
        return cls._model_cache[key]

    @classmethod
    def _get_compiled_gmm_model(cls, user_input: UserInputConfig) -> TransformerModel:
        key = ("compiled_gmm", user_config_build_cache_key(user_input))
        if key not in cls._model_cache:
            config_resolver = ConfigResolver(user_input=user_input)
            model_config = config_resolver.resolve()
            model = TransformerModel(user_input.model_id, model_config)
            cls._model_cache[key] = torch.compile(model, backend=get_backend(), fullgraph=True)
        return cls._model_cache[key]

    @classmethod
    def _get_model(cls, user_config: UserInputConfig) -> TransformerModel:
        return get_cached_build_model(cls._model_cache, user_config)


class SwiGLUFusionPassTestCase(SwiGLUFusionPassTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ("Qwen/Qwen3-32B", QuantizeLinearAction.DISABLED),
            ("Qwen/Qwen3-32B", QuantizeLinearAction.W8A8_STATIC),
            ("Qwen/Qwen3-32B", QuantizeLinearAction.W4A8_DYNAMIC),
        ]
    )
    def test_swiglu_fused_op_present(self, model_id: str, linear_act: QuantizeLinearAction):
        num_tokens = 100
        model = self._get_compiled_qwen_swiglu_model(model_id, linear_act)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

        self.assertEqual(
            count_events(runtime, torch.ops.tensor_cast.swiglu.default),
            64,
        )


@pytest.mark.nightly
class SwiGLUFusionPassNightlyTestCase(SwiGLUFusionPassTestMixin, unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    @parameterized.expand(
        [
            ("Qwen/Qwen3-235B-A22B", QuantizeLinearAction.DISABLED),
            ("Qwen/Qwen3-235B-A22B", QuantizeLinearAction.W8A8_STATIC),
            ("Qwen/Qwen3-235B-A22B", QuantizeLinearAction.W4A8_DYNAMIC),
        ]
    )
    def test_gmm_swiglu_fused_op_present(self, model_id: str, linear_act: QuantizeLinearAction):
        user_input = UserInputConfig(
            model_id=model_id,
            do_compile=True,
            num_hidden_layers_override=1,
            quantize_linear_action=linear_act,
        )
        model = self._get_compiled_gmm_model(user_input)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        self.assertEqual(
            count_events(runtime, torch.ops.tensor_cast.grouped_matmul_swiglu.default)
            + count_events(runtime, torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default)
            + count_events(runtime, torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default)
            + count_events(runtime, torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default)
            + count_events(runtime, torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default),
            1,
        )

    @parameterized.expand(
        [
            ("deepseek-ai/DeepSeek-V3.1", QuantizeLinearAction.W4A8_DYNAMIC),
            ("deepseek-ai/DeepSeek-V3.1", QuantizeLinearAction.DISABLED),
        ]
    )
    def test_swiglu_fused_op_present_deepseek(self, model_id: str, linear_act: QuantizeLinearAction):
        num_tokens = 100

        user_config = UserInputConfig(
            model_id=model_id,
            num_queries=1,
            query_len=num_tokens,
            context_length=1000,
            do_compile=True,
            quantize_linear_action=linear_act,
            num_mtp_tokens=0,
        )

        model = self._get_model(user_config)
        model_config = ModelConfig(
            user_config.get_parallel_config(),
            user_config.get_quant_config(),
            attention_cls=AttentionTensorCast,
        )

        attn_meta, kv_cache_by_layers, actual_num_tokens = create_mla_metadata_and_kv_cache(model, model_config)
        inputs = torch.empty([1, actual_num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, actual_num_tokens], dtype=torch.long, device="meta")

        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
            patch_torch(),
        ):
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(query_start_loc=attn_meta.query_start_loc),
            )
            self.assertEqual(outputs.shape, (1, 1, model.model_config.hf_config.vocab_size))

        self.assertGreaterEqual(
            count_events(runtime, torch.ops.tensor_cast.swiglu.default),
            1,
        )


if __name__ == "__main__":
    unittest.main()
