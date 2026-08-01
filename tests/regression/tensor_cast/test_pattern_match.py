import math
import unittest

import pytest
import torch
import torch.fx as fx
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from parameterized import parameterized
from tensor_cast import config, ops  # noqa: F401
from tensor_cast.compilation import get_backend
from tensor_cast.compilation import patterns as compilation_patterns
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


class FP32LayerNormModule(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16, affine: bool = False, hidden_size: int = 4):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.weight = (
            torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32, device="meta")) if affine else None
        )
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32, device="meta")) if affine else None

    def forward(self, hidden_states):
        out = torch.nn.functional.layer_norm(
            hidden_states.float(),
            (self.hidden_size,),
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            1e-6,
        )
        return out.to(self.dtype)


class WanStyleFP32LayerNormModule(torch.nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states):
        out = torch.nn.functional.layer_norm(
            hidden_states.float().float(),
            (self.hidden_size,),
            None,
            None,
            1e-6,
        )
        return out.type_as(hidden_states)


class TorchRMSNormModule(torch.nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.norm = torch.nn.RMSNorm(hidden_size, eps=1e-6, device="meta")

    def forward(self, hidden_states):
        return self.norm(hidden_states)


class DiffusersRMSNormModule(torch.nn.Module):
    def __init__(self, weight_dtype: torch.dtype = torch.bfloat16, hidden_size: int = 4):
        super().__init__()
        self.norm = DiffusersRMSNorm(hidden_size, eps=1e-6).to(device="meta", dtype=weight_dtype)

    def forward(self, hidden_states):
        return self.norm(hidden_states)


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


class GELUModule(torch.nn.Module):
    def __init__(self, approximate: str):
        super().__init__()
        self.gelu = torch.nn.GELU(approximate=approximate)

    def forward(self, hidden_states):
        return self.gelu(hidden_states)


class DecomposedTanhGELUModule(torch.nn.Module):
    def forward(self, hidden_states):
        hidden_states_fp32 = hidden_states.to(torch.float32)
        x_cubed = torch.pow(hidden_states_fp32, 3)
        tanh_arg = math.sqrt(2 / math.pi) * (hidden_states_fp32 + 0.044715 * x_cubed)
        out = 0.5 * hidden_states_fp32 * (1 + torch.tanh(tanh_arg))
        return out.to(hidden_states.dtype)


class DecomposedErfGELUModule(torch.nn.Module):
    def forward(self, hidden_states):
        hidden_states_fp32 = hidden_states.to(torch.float32)
        out = 0.5 * hidden_states_fp32 * (1 + torch.erf(hidden_states_fp32 / math.sqrt(2)))
        return out.to(hidden_states.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SingleRopeModule(torch.nn.Module):
    def forward(self, hidden_states, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        hidden_states = (hidden_states * cos) + (rotate_half(hidden_states) * sin)
        return hidden_states.transpose(1, 2)


class SingleRopeSameLayoutModule(torch.nn.Module):
    def forward(self, hidden_states, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class DiffusersHunyuanSingleRopeModule(torch.nn.Module):
    def forward(self, hidden_states, cos, sin):
        return apply_rotary_emb(hidden_states, (cos, sin), sequence_dim=1)


class SiLUModule(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.nn.functional.silu(hidden_states)


class SwiGLUModule(torch.nn.Module):
    def forward(self, gate, up):
        gate_fp32 = gate.to(torch.float32)
        silu_gate = (gate_fp32 * torch.sigmoid(gate_fp32)).to(gate.dtype)
        return silu_gate * up


class DecomposedSiLUModule(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.sigmoid(hidden_states)


class ResidualAddModule(torch.nn.Module):
    def forward(self, residual, update):
        return residual + update


class SameShapeMulAddModule(torch.nn.Module):
    def forward(self, x, y, z):
        return x + y * z


class GatedResidualAddModule(torch.nn.Module):
    def __init__(self, reverse_mul: bool = False, reverse_add: bool = False):
        super().__init__()
        self.reverse_mul = reverse_mul
        self.reverse_add = reverse_add

    def forward(self, residual, update, gate):
        gated = gate * update if self.reverse_mul else update * gate
        return gated + residual if self.reverse_add else residual + gated


class ModulatedLayerNormModule(torch.nn.Module):
    def __init__(self, affine: bool = False, hidden_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = (
            torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32, device="meta")) if affine else None
        )
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32, device="meta")) if affine else None

    def forward(self, hidden_states, scale, shift):
        normed = torch.nn.functional.layer_norm(
            hidden_states.float(),
            (self.hidden_size,),
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            1e-6,
        ).to(hidden_states.dtype)
        return normed * (1 + scale) + shift


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

    @staticmethod
    def _gelu_approximate(event):
        args = event.op_invoke_info.args
        if len(args) > 1:
            return args[1]
        return event.op_invoke_info.kwargs.get("approximate", "none")

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

    @parameterized.expand(
        [
            ["none"],
            ["tanh"],
        ]
    )
    def test_gelu_pattern(self, approximate):
        model = GELUModule(approximate)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.gelu.default", result)
        gelu_event = next(
            event for event in runtime.event_list if event.op_invoke_info.func == torch.ops.tensor_cast.gelu.default
        )
        self.assertEqual(self._gelu_approximate(gelu_event), approximate)

    def test_decomposed_tanh_gelu_pattern(self):
        model = DecomposedTanhGELUModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.gelu.default", result)
        gelu_event = next(
            event for event in runtime.event_list if event.op_invoke_info.func == torch.ops.tensor_cast.gelu.default
        )
        self.assertEqual(self._gelu_approximate(gelu_event), "tanh")

    def test_decomposed_erf_gelu_pattern(self):
        model = DecomposedErfGELUModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.gelu.default", result)
        gelu_event = next(
            event for event in runtime.event_list if event.op_invoke_info.func == torch.ops.tensor_cast.gelu.default
        )
        self.assertEqual(self._gelu_approximate(gelu_event), "none")

    def test_gelu_pattern_config_gate(self):
        old_enable_gelu = config.compilation.fusion_patterns.enable_gelu
        try:
            config.compilation.fusion_patterns.enable_gelu = False
            compilation_patterns.all_passes = [PatternMatchPass(), PatternMatchPass(), PatternMatchPass()]
            compilation_patterns.lazy_init.cache_clear()
            compilation_patterns.lazy_init()
            for pattern_pass in compilation_patterns.all_passes:
                self.assertFalse(any("gelu" in name for name in pattern_pass.pattern_replacements))
        finally:
            config.compilation.fusion_patterns.enable_gelu = old_enable_gelu
            compilation_patterns.all_passes = [PatternMatchPass(), PatternMatchPass(), PatternMatchPass()]
            compilation_patterns.lazy_init.cache_clear()

    @parameterized.expand(
        [
            [SingleRopeModule, (2, 5, 3, 4)],
            [SingleRopeSameLayoutModule, (2, 3, 5, 4)],
        ]
    )
    def test_single_rope_pattern(self, module_cls, expected_shape):
        model = module_cls()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=False)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 5, 4, device="meta", dtype=torch.bfloat16)
        cos = torch.empty(2, 5, 4, device="meta", dtype=torch.bfloat16)
        sin = torch.empty(2, 5, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states, cos, sin)
            self.assertEqual(outputs.shape, expected_shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.apply_rope_single.default", result)

    def test_diffusers_hunyuan_pairwise_single_rope_pattern(self):
        model = DiffusersHunyuanSingleRopeModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=False)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 5, 4, device="meta", dtype=torch.bfloat16)
        cos = torch.empty(3, 4, device="meta", dtype=torch.float32)
        sin = torch.empty(3, 4, device="meta", dtype=torch.float32)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states, cos, sin)
            self.assertEqual(outputs.shape, hidden_states.shape)
            self.assertEqual(outputs.dtype, hidden_states.dtype)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.apply_rope_single.default", result)
        rope_event = next(
            event
            for event in runtime.event_list
            if event.op_invoke_info.func == torch.ops.tensor_cast.apply_rope_single.default
        )
        self.assertEqual(rope_event.op_invoke_info.args[3:], (False, False))

    @parameterized.expand(
        [
            [SiLUModule],
            [DecomposedSiLUModule],
        ]
    )
    def test_silu_pattern(self, module_cls):
        model = module_cls()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.silu.default", result)

    def test_swiglu_pattern_precedence(self):
        model = SwiGLUModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        gate = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        up = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(gate, up)
            self.assertEqual(outputs.shape, gate.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.swiglu.default", result)
        self.assertNotIn("tensor_cast.silu.default", result)

    def test_gated_residual_add_negative_pattern(self):
        model = ResidualAddModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        residual = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        update = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(residual, update)
            self.assertEqual(outputs.shape, residual.shape)
        result = runtime.table_averages()
        self.assertNotIn("tensor_cast.gated_residual_add.default", result)

    def test_same_shape_mul_add_negative_pattern(self):
        model = SameShapeMulAddModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        x = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        y = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        z = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(x, y, z)
            self.assertEqual(outputs.shape, x.shape)
        result = runtime.table_averages()
        self.assertNotIn("tensor_cast.gated_residual_add.default", result)

    @parameterized.expand(
        [
            [False, False],
            [False, True],
            [True, False],
            [True, True],
        ]
    )
    def test_gated_residual_add_pattern_when_enabled(self, reverse_mul, reverse_add):
        old_enable_gated_residual_add = config.compilation.fusion_patterns.enable_gated_residual_add
        try:
            config.compilation.fusion_patterns.enable_gated_residual_add = True
            compilation_patterns.all_passes = [PatternMatchPass(), PatternMatchPass(), PatternMatchPass()]
            compilation_patterns.lazy_init.cache_clear()
            compilation_patterns.lazy_init()
            torch.compiler.reset()

            model = GatedResidualAddModule(reverse_mul, reverse_add)
            model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
            machine_config = TEST_DEVICE
            perf_model = AnalyticPerformanceModel(machine_config)
            residual = torch.empty(1, 3, 1, device="meta", dtype=torch.bfloat16)
            update = torch.empty(2, 1, 4, device="meta", dtype=torch.bfloat16)
            gate = torch.empty(1, 3, 4, device="meta", dtype=torch.bfloat16)
            with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
                outputs = model(residual, update, gate)
                self.assertEqual(outputs.shape, (2, 3, 4))
            result = runtime.table_averages()
            self.assertIn("tensor_cast.gated_residual_add.default", result)
        finally:
            config.compilation.fusion_patterns.enable_gated_residual_add = old_enable_gated_residual_add
            compilation_patterns.all_passes = [PatternMatchPass(), PatternMatchPass(), PatternMatchPass()]
            compilation_patterns.lazy_init.cache_clear()

    @parameterized.expand(
        [
            [False],
            [True],
        ]
    )
    def test_modulated_layer_norm_pattern(self, affine):
        model = ModulatedLayerNormModule(affine=affine)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        scale = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        shift = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states, scale, shift)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.modulated_layer_norm.default", result)

    @parameterized.expand(
        [
            [False],
            [True],
        ]
    )
    def test_layer_norm_pattern(self, affine):
        model = FP32LayerNormModule(affine=affine)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.layer_norm.default", result)

    def test_wan_style_layer_norm_pattern(self):
        model = WanStyleFP32LayerNormModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.layer_norm.default", result)

    def test_torch_rms_norm_pattern(self):
        model = TorchRMSNormModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)

    def test_rank4_torch_rms_norm_pattern(self):
        model = TorchRMSNormModule()
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 5, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)

    @parameterized.expand(
        [
            [torch.bfloat16, (2, 3, 4)],
            [torch.float16, (2, 3, 4)],
            [torch.bfloat16, (2, 3, 5, 4)],
            [torch.float16, (2, 3, 5, 4)],
        ]
    )
    def test_diffusers_rms_norm_pattern(self, weight_dtype, shape):
        model = DiffusersRMSNormModule(weight_dtype=weight_dtype)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=False)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(*shape, device="meta", dtype=weight_dtype)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
            self.assertEqual(outputs.dtype, hidden_states.dtype)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)

    def test_diffusers_rms_norm_mixed_dtype_pattern_is_not_fused(self):
        model = DiffusersRMSNormModule(weight_dtype=torch.float32)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=False)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
            self.assertEqual(outputs.dtype, torch.float32)
        result = runtime.table_averages()
        self.assertNotIn("tensor_cast.rms_norm.default", result)

    @parameterized.expand(
        [
            [False],
            [True],
        ]
    )
    def test_rank4_layer_norm_pattern(self, affine):
        model = FP32LayerNormModule(affine=affine)
        model = torch.compile(model, backend=self.compile_backend, fullgraph=True, dynamic=True)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        hidden_states = torch.empty(2, 3, 5, 4, device="meta", dtype=torch.bfloat16)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model(hidden_states)
            self.assertEqual(outputs.shape, hidden_states.shape)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.layer_norm.default", result)

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
