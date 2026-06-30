import operator
import unittest

import pytest
import torch
import torch.fx as fx
from parameterized import parameterized
import tensor_cast.config as tc_config
from tensor_cast.compilation import get_backend
from tensor_cast.compilation.freezing_passes.grouped_matmul_swiglu_pass import (
    GroupedMatmulSwigluPass,
)
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


def _build_grouped_matmul_swiglu_graph(split_target, split_args):
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    bias = graph.placeholder("bias")
    gmm = graph.call_function(torch.ops.tensor_cast.grouped_matmul.default, args=([x], [w], [bias]))
    split = graph.call_function(split_target, args=(gmm, *split_args))
    gate = graph.call_function(operator.getitem, args=(split, 0))
    up = graph.call_function(operator.getitem, args=(split, 1))
    swiglu = graph.call_function(torch.ops.tensor_cast.swiglu.default, args=(gate, up))
    graph.output(swiglu)
    return fx.GraphModule({}, graph)


def test_grouped_matmul_swiglu_pass_fuses_valid_graphs():
    graph_module = _build_grouped_matmul_swiglu_graph(torch.ops.aten.split.Tensor, (2, -1))

    result = GroupedMatmulSwigluPass()(graph_module)

    targets = [node.target for node in result.graph.nodes if node.op == "call_function"]
    assert torch.ops.tensor_cast.grouped_matmul_swiglu.default in targets
    assert torch.ops.tensor_cast.swiglu.default not in targets


def test_grouped_matmul_swiglu_pass_rejects_unsafe_shapes_and_users():
    pass_ = GroupedMatmulSwigluPass()
    invalid = _build_grouped_matmul_swiglu_graph(torch.ops.aten.split.Tensor, (2, 0))
    assert pass_(invalid) is invalid
    assert any(node.target == torch.ops.tensor_cast.swiglu.default for node in invalid.graph.nodes)

    split_sizes = _build_grouped_matmul_swiglu_graph(torch.ops.aten.split_with_sizes.default, ([2, 2], -1))
    assert pass_(split_sizes) is split_sizes
    assert not any(node.target == torch.ops.tensor_cast.swiglu.default for node in split_sizes.graph.nodes)


class SwiGLUFusionPassTestMixin:
    """Unified, parameterized test verifying SwiGLU fusion presence.

    Simulates cli.inference.text_generate via ModelRunner, compiles models,
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
        self._orig_dfc = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = False
        self.addCleanup(self._restore_dfc)

    def _restore_dfc(self):
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = self._orig_dfc

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
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc,
                    selected_token_indices=attn_meta.query_start_loc[1:] - 1,
                ),
            )
            num_sequences = attn_meta.query_start_loc.shape[0] - 1
            self.assertEqual(outputs.shape, (1, num_sequences, model.model_config.hf_config.vocab_size))

        self.assertGreaterEqual(
            count_events(runtime, torch.ops.tensor_cast.swiglu.default),
            1,
        )


if __name__ == "__main__":
    unittest.main()
