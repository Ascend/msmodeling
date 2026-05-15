import unittest

import torch
from parameterized import parameterized

from tensor_cast.compilation import get_backend
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel
from .test_common import count_events, get_quant_config


class MergeLinearPassTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def test_qwen3_32b_fp(self):
        model_id = "Qwen/Qwen3-32B"
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
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
        self.assertEqual(count_events(runtime, torch.ops.aten.mm.default), 5)
        self.assertEqual(count_events(runtime, torch.ops.aten.split_with_sizes.default), 2)

    def test_qwen3_32b_static_int8(self):
        model_id = "Qwen/Qwen3-32B"
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                quant_type=LinearQuantType.W8A8,
                activation_scale=torch.empty([num_tokens], dtype=torch.float, device="meta"),
            ),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
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
        self.assertEqual(count_events(runtime, torch.ops.tensor_cast.static_quant_linear.default), 4)
        self.assertEqual(count_events(runtime, torch.ops.aten.split_with_sizes.default), 2)

    @parameterized.expand(
        [
            [LinearQuantType.W8A8],
            [LinearQuantType.W4A8],
            [LinearQuantType.FP8],
            [LinearQuantType.MXFP4],
        ]
    )
    def test_qwen3_32b_dynamic_quant(self, quant_type):
        model_id = "Qwen/Qwen3-32B"
        num_tokens = 100
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
        if quant_type == LinearQuantType.W8A8:
            expected_op = torch.ops.tensor_cast.static_quant_linear.default
        elif quant_type == LinearQuantType.W4A8:
            expected_op = torch.ops.tensor_cast.static_quant_linear_int4.default
        elif quant_type == LinearQuantType.FP8:
            expected_op = torch.ops.tensor_cast.fp8_linear.default
        elif quant_type == LinearQuantType.MXFP4:
            expected_op = torch.ops.tensor_cast.mxfp4_linear.default
        self.assertEqual(count_events(runtime, expected_op), 4)
        self.assertEqual(count_events(runtime, torch.ops.aten.split_with_sizes.default), 2)
