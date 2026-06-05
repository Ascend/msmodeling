import unittest

import pytest
import torch
import torch.fx as fx
from parameterized import parameterized
from tensor_cast import ops  # noqa: F401
from tensor_cast.compilation import get_backend
from tensor_cast.compilation.freezing_passes.freezing_pattern_pass import (
    FreezingPatternPass,
)
from tensor_cast.compilation.pass_base import TensorCastGraphModulePass
from tensor_cast.compilation.passes.pattern_match_pass import PatternMatchPass
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel

from .conftest import get_session_hf_config
from .test_common import get_quant_config

# Core RMS pattern-consistency assertions were moved to test_ops.py::test_rms_norm_non_default_eps_path_consistency.


def test_pass_uuid_and_pattern_pass_loop():
    class IdentityPass(TensorCastGraphModulePass):
        def __call__(self, graph):
            return graph

    first_uuid = IdentityPass().uuid()
    assert first_uuid == IdentityPass().uuid()
    assert len(first_uuid) == 64

    class FakePatternPass:
        patterns = {}

        def __init__(self):
            self.calls = 0

        def apply(self, _gm):
            self.calls += 1
            return 2 if self.calls == 1 else 0

    gm = fx.symbolic_trace(torch.nn.Identity())
    pattern_pass = PatternMatchPass()
    pattern_pass.pattern_pass = FakePatternPass()
    assert pattern_pass(gm) is gm
    assert pattern_pass.pattern_pass.calls == 2
    pattern_pass.pattern_replacements["existing"] = (lambda x: x, lambda x: x)
    assert pattern_pass.has_pattern("existing")
    with pytest.raises(ValueError, match="already registered"):
        pattern_pass.register_pattern("existing", lambda x: x, lambda x: x, [torch.empty(1)])

    freezing_pass = FreezingPatternPass()
    freezing_pass.pattern_pass = FakePatternPass()
    assert freezing_pass(gm) is gm
    freezing_pass.pattern_handlers["existing"] = (object(), lambda *_: None)
    assert freezing_pass.has_pattern("existing")
    with pytest.raises(ValueError, match="already registered"):
        freezing_pass.register_pattern("existing", object(), lambda *_: None)


class NonDefaultEpsRMSNormModule(torch.nn.Module):
    def __init__(self, dtype=torch.float16, eps: float = 1e-5):
        super().__init__()
        self.dtype = dtype
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=dtype, device="meta"))

    def _rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states, residual):
        rms = self._rms_norm(hidden_states)
        add_rms = self._rms_norm(hidden_states + residual)
        added = hidden_states + residual
        add_rms2 = self._rms_norm(added)
        return rms, add_rms, add_rms2, added


class PatternReplaceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._transformer_cache = {}

    @classmethod
    def _get_transformer_model(cls, model_id: str, model_config: ModelConfig) -> TransformerModel:
        key = (model_id, repr(model_config))
        if key not in cls._transformer_cache:
            cls._transformer_cache[key] = TransformerModel(model_id, model_config)
        return cls._transformer_cache[key]

    def setUp(self):
        torch.compiler.reset()
        num_tokens = 100
        self.compile_backend = get_backend()
        with torch.device("meta"):
            self.inputs = torch.empty([1, num_tokens], dtype=torch.long)
            self.position_ids = torch.empty([1, num_tokens], dtype=torch.long)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    def test_rms_norm_pattern(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), num_hidden_layers_override=2)
        model_config.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm2.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    def test_rms_norm_static_quant_pattern(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(activation_scale=torch.max(torch.abs(torch.randn(1))) / 127.0),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=1,
        )
        model_config.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.rms_norm_quant.default", result)
        self.assertIn("tensor_cast.add_rms_norm_quant2.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-32B", False],
        ]
    )
    def test_rms_norm_dynamic_quant_pattern(self, model_id, per_sample):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                dynamic_quant_granularity=QuantGranularity.PER_SAMPLE if per_sample else QuantGranularity.PER_TENSOR
            ),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
        )
        model_config.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.rms_norm_dynamic_quant_symmetric.default", result)
        self.assertIn("tensor_cast.add_rms_norm_dynamic_quant2_symmetric.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-32B", False],
        ]
    )
    def test_rms_norm_dynamic_quant_pattern_fp8(self, model_id, per_sample):
        num_tokens = 100
        fp8_quant_config = get_quant_config(
            quant_type=LinearQuantType.FP8,
        )
        model_config = ModelConfig(
            ParallelConfig(),
            fp8_quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
        )
        model_config.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.rms_norm_dynamic_quant_symmetric.default", result)
        self.assertIn("tensor_cast.add_rms_norm_dynamic_quant2_symmetric.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", 64],
            ["Qwen/Qwen3-32B", 32],
        ]
    )
    def test_rms_norm_dynamic_quant_pattern_mxfp4(self, model_id, group_size):
        num_tokens = 100
        mxfp4_quant_config = get_quant_config(
            quant_type=LinearQuantType.MXFP4,
            weight_group_size=group_size,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            weight_quant_scheme=QuantScheme.SYMMETRIC,
        )
        model_config = ModelConfig(
            ParallelConfig(),
            mxfp4_quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
        )
        model_config.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.rms_norm_dynamic_quant_mxfp4.default", result)
        self.assertIn("tensor_cast.add_rms_norm_dynamic_quant2_mxfp4.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    def test_rope_pattern(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(activation_scale=torch.max(torch.abs(torch.randn(1))) / 127.0),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
        )
        model_config.hf_config = get_session_hf_config(model_id)
        model = self._get_transformer_model(model_id, model_config)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.apply_rope.default", result)

    # deprecated: migrated to test_ops.py::test_rms_norm_non_default_eps_path_consistency
    def test_rms_norm_pattern_non_default_eps(self):
        model = NonDefaultEpsRMSNormModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 4, device="meta", dtype=torch.float16)
        residual = torch.empty(2, 4, device="meta", dtype=torch.float16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states, residual)
            self.assertEqual(len(outputs), 4)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm2.default", result)


if __name__ == "__main__":
    unittest.main()
