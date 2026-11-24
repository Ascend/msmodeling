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

import pytest
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

# Core merge-linear mapping assertions were moved to the unified entry in test_ops.py.


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
        # MXFP4 microscale is shaped as (N, K_group), so Q/K projections have
        # incompatible per-output-channel scale shapes and remain separate.
        expected_linear_count = 5 if quant_type == LinearQuantType.MXFP4 else 4
        self.assertEqual(count_events(runtime, expected_op), expected_linear_count)
        self.assertEqual(count_events(runtime, torch.ops.aten.split_with_sizes.default), 2)


@pytest.mark.parametrize("quant_format,dtype", [("mxfp4", torch.int4), ("mxfp8", torch.float8_e4m3fn)])
def test_merge_mx_linear_concatenates_each_projection_weight_scale(quant_format, dtype):
    from torch.fx.experimental.proxy_tensor import make_fx
    from tensor_cast.compilation.passes.merge_linear_pass import MergeLinearPass

    op = getattr(torch.ops.tensor_cast, f"{quant_format}_linear").default

    def projections(x, w1, w2, x_scale, w_scale1, w_scale2):
        return (
            op(x, w1, x_scale, w_scale1, None, torch.bfloat16),
            op(x, w2, x_scale, w_scale2, None, torch.bfloat16),
        )

    args = (
        torch.empty((3, 64), device="meta", dtype=dtype),
        torch.empty((64, 32), device="meta", dtype=dtype),
        torch.empty((64, 32), device="meta", dtype=dtype),
        torch.empty((3, 2), device="meta", dtype=torch.float8_e8m0fnu),
        torch.empty((32, 2), device="meta", dtype=torch.float8_e8m0fnu),
        torch.empty((32, 2), device="meta", dtype=torch.float8_e8m0fnu),
    )
    gm = make_fx(projections)(*args)
    original_scales = [node.args[3].name for node in gm.graph.nodes if node.target == op]

    MergeLinearPass()(gm)

    merged = [node for node in gm.graph.nodes if node.target == op]
    assert len(merged) == 1
    scale_cat = merged[0].args[3]
    assert scale_cat.target == torch.ops.tensor_cast.cat.default
    assert [node.name for node in scale_cat.args[0]] == original_scales
    assert scale_cat.args[1] == 0
    interpreter = torch.fx.Interpreter(gm, garbage_collect_values=False)
    outputs = interpreter.run(*args)
    assert interpreter.env[scale_cat].shape == (64, 2)
    assert interpreter.env[merged[0].args[1]].shape == (64, 64)
    assert [output.shape for output in outputs] == [(3, 32), (3, 32)]
