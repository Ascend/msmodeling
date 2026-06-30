import unittest

import pytest
import torch
from parameterized import parameterized
import tensor_cast.config as tc_config
from tensor_cast.compilation import get_backend
from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import ModelConfig, ParallelConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_moe_config
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.utils import AutoModelConfigLoader

from .test_common import count_events, get_quant_config


@pytest.mark.nightly
class GmmPassTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self._orig_dfc = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = False
        self.addCleanup(self._restore_dfc)

    def _restore_dfc(self):
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = self._orig_dfc

    @parameterized.expand(
        [
            "Qwen/Qwen3-235B-A22B",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ]
    )
    def test_qwen3_fp(self, model_id):
        user_input = UserInputConfig(
            model_id=model_id,
            do_compile=True,
            num_hidden_layers_override=1,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        config_resolver = ConfigResolver(user_input=user_input)
        model_config = config_resolver.resolve()
        model = TransformerModel(model_id, model_config)
        model = torch.compile(model, backend=get_backend(), fullgraph=True)

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
        self.assertEqual(count_events(runtime, torch.ops.tensor_cast.grouped_matmul.default), 1)

    def test_qwen3_static_int8(self):
        model_id = "Qwen/Qwen3-235B-A22B"
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                quant_type=LinearQuantType.W8A8,
                activation_scale=torch.tensor(1.0),
            ),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(model, backend=get_backend(), fullgraph=True)
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
        self.assertEqual(count_events(runtime, torch.ops.tensor_cast.grouped_matmul_quant.default), 1)
        self.assertEqual(
            count_events(runtime, torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default),
            1,
        )

    @parameterized.expand(
        [
            [LinearQuantType.W8A8],
            [LinearQuantType.W4A8],
            [LinearQuantType.FP8],
            [LinearQuantType.MXFP4],
        ]
    )
    def test_qwen3_dynamic_quant(self, quant_type):
        model_id = "Qwen/Qwen3-235B-A22B"
        num_tokens = 100
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                quant_type=quant_type,
                weight_quant_granularity=QuantGranularity.PER_GROUP
                if quant_type == LinearQuantType.MXFP4
                else QuantGranularity.PER_TENSOR,
                weight_group_size=32 if quant_type == LinearQuantType.MXFP4 else None,
            ),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(model, backend=get_backend(), fullgraph=True)
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
        expected_op = None
        expected_swiglu_op = None
        if quant_type == LinearQuantType.W8A8:
            expected_op = torch.ops.tensor_cast.grouped_matmul_quant.default
            expected_swiglu_op = torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default
        elif quant_type == LinearQuantType.W4A8:
            expected_op = torch.ops.tensor_cast.grouped_matmul_quant_int4.default
            expected_swiglu_op = torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default
        elif quant_type == LinearQuantType.FP8:
            expected_op = torch.ops.tensor_cast.grouped_matmul_fp8.default
            expected_swiglu_op = torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default
        elif quant_type == LinearQuantType.MXFP4:
            expected_op = torch.ops.tensor_cast.grouped_matmul_mxfp4.default
            expected_swiglu_op = torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default
        self.assertEqual(count_events(runtime, expected_op), 1)
        self.assertEqual(count_events(runtime, expected_swiglu_op), 1)
