# tests/test_tensor_cast/test_dfc_pass.py
import operator
import unittest
from dataclasses import asdict

import torch
import torch.fx as fx
from parameterized import parameterized

import tensor_cast.ops  # noqa: F401 — must register ops before compilation imports them
from tensor_cast import config
from tensor_cast.compilation import get_backend
from tensor_cast.compilation.freezing_passes.dispatch_ffn_combine_pass import (
    DispatchFFNCombinePass,
)
from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel


class DfcPassTestCase(unittest.TestCase):
    """DispatchFFNCombine fusion pass tests.

    DSv3 configs from Phase 1 E2E (5525b21):
      Prefill: W8A8, TP=8/DP=2/EP=16, nq=1, ql=256
      Decode:  W8A8, TP=8/DP=2/EP=16, nq=16, ql=1, cl=4096
    DSv3 first_k_dense_replace=3, so num_hidden_layers_override≥4 to include MoE.
    """

    def setUp(self):
        torch.compiler.reset()
        self._orig_enable_dispatch_ffn_combine = config.compilation.fusion_patterns.enable_dispatch_ffn_combine

    def tearDown(self):
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = self._orig_enable_dispatch_ffn_combine

    def _assert_no_dfc_residual_ops(self, table_result: str):
        residual_ops = [
            "tensor_cast.init_routing_v2.default",
            "tensor_cast.all_to_all.default",
            "tensor_cast.unpermute_tokens.default",
            "tensor_cast.grouped_matmul_quant_swiglu.default",
            "tensor_cast.grouped_matmul_quant.default",
            "tensor_cast.grouped_matmul_quant_int4_swiglu.default",
            "tensor_cast.grouped_matmul_quant_int4.default",
        ]
        for op_name in residual_ops:
            self.assertNotIn(op_name, table_result, f"Residual DFC op found: {op_name}")

    @parameterized.expand(
        [
            # (scenario, num_queries, query_len, context_length, quantize_linear_action)
            ("prefill_w8a8_static", 1, 256, 0, QuantizeLinearAction.W8A8_STATIC),
            ("decode_w8a8_static", 16, 1, 4096, QuantizeLinearAction.W8A8_STATIC),
            ("prefill_w8a8_dynamic", 1, 256, 0, QuantizeLinearAction.W8A8_DYNAMIC),
            ("decode_w8a8_dynamic", 16, 1, 4096, QuantizeLinearAction.W8A8_DYNAMIC),
            ("prefill_w4a8_dynamic", 1, 256, 0, QuantizeLinearAction.W4A8_DYNAMIC),
            ("decode_w4a8_dynamic", 16, 1, 4096, QuantizeLinearAction.W4A8_DYNAMIC),
        ]
    )
    def test_dfc_dsv3_ep(
        self,
        scenario,
        num_queries,
        query_len,
        context_length,
        quantize_linear_action,
    ):
        """Verify that DFC is effective for DSv3 large EP configuration (Phase 1)"""
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
        model_id = "deepseek-ai/DeepSeek-V3"
        user_input = UserInputConfig(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id=model_id,
            num_queries=num_queries,
            query_len=query_len,
            context_length=context_length,
            do_compile=True,
            allow_graph_break=True,
            world_size=16,
            tp_size=8,
            dp_size=2,
            ep_size=16,
            quantize_linear_action=quantize_linear_action,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        # Any DFC variant should be present (quant variant depends on model config)
        self.assertTrue(
            "tensor_cast.dispatch_ffn_combine" in result["table_result"],
            f"No DFC op found in table_result:\n{result['table_result']}",
        )
        self._assert_no_dfc_residual_ops(result["table_result"])

    def test_dfc_output_shape_matches_baseline(self):
        """Verify that the DFC output shape is consistent with the unpermute_tokens baseline (Qwen3 non-EP)"""
        model_id = "Qwen/Qwen3-235B-A22B"

        def run_model(enable_dfc):
            config.compilation.fusion_patterns.enable_dispatch_ffn_combine = enable_dfc
            torch.compiler.reset()
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
            inputs = torch.empty([1, 100], dtype=torch.long, device="meta")
            pos = torch.empty([1, 100], dtype=torch.long, device="meta")
            perf_model = AnalyticPerformanceModel(TEST_DEVICE)
            with (
                Runtime(
                    perf_model,
                    TEST_DEVICE,
                    memory_tracker=MemoryTracker(TEST_DEVICE),
                ) as rt,  # noqa: F841
                torch.no_grad(),
            ):
                out = model.forward(inputs, pos)
            return out.shape

        baseline_shape = run_model(enable_dfc=False)
        dfc_shape = run_model(enable_dfc=True)
        self.assertEqual(baseline_shape, dfc_shape)

    def test_dfc_dsv3_w8a8_dynamic_profiling(self):
        """Verify DFC profiling performance model with DeepSeek-V3 config."""
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
        user_input = UserInputConfig(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id="deepseek-ai/DeepSeek-V3",
            num_queries=2,
            query_len=4096,
            do_compile=True,
            allow_graph_break=True,
            world_size=16,
            tp_size=8,
            dp_size=2,
            ep_size=16,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            performance_model="profiling",
            profiling_database=(
                "tensor_cast/performance_model/profiling_database/data/"
                "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/"
                "vllm0.15.0_torch2.9.0_cann8.5"
            ),
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)

        # Any DFC variant should be present (quant variant depends on model config)
        self.assertTrue(
            "tensor_cast.dispatch_ffn_combine" in result["table_result"],
            f"No DFC op found in table_result:\n{result['table_result']}",
        )
        self._assert_no_dfc_residual_ops(result["table_result"])

    def test_dfc_estimator_produces_nonzero_time(self):
        """Verify DFC estimator computes meaningful execution time (not memory-only)."""
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
        model_id = "deepseek-ai/DeepSeek-V3"
        user_input = UserInputConfig(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id=model_id,
            num_queries=8,
            query_len=1,
            context_length=4096,
            do_compile=True,
            allow_graph_break=True,
            world_size=16,
            tp_size=8,
            dp_size=2,
            ep_size=16,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)

        # DFC quant variant should appear in table output
        self.assertTrue(
            "tensor_cast.dispatch_ffn_combine" in result["table_result"],
            f"No DFC op found in table_result:\n{result['table_result']}",
        )
        self._assert_no_dfc_residual_ops(result["table_result"])

        # Analytic model should produce non-trivial execution time
        analytic_time = result["execution_time_s"].get("analytic", 0)
        self.assertGreater(
            analytic_time,
            0,
            "Analytic execution time should be > 0",
        )

    def test_resolve_dfc_variant_collects_all_unfused_gate_up_linears(self):
        """Case 2 should collect one gate_up and one down_proj linear per expert."""
        graph = fx.Graph()

        x = graph.placeholder("x")
        expert_indices = graph.placeholder("expert_indices")
        output_splits = graph.placeholder("output_splits")
        input_splits = graph.placeholder("input_splits")
        rank = graph.placeholder("rank")
        rank_group = graph.placeholder("rank_group")

        routed = graph.call_function(torch.ops.tensor_cast.init_routing_v2.default, args=(x, expert_indices))
        dispatched = graph.call_function(
            torch.ops.tensor_cast.all_to_all.default,
            args=(routed, output_splits, input_splits, rank, rank_group),
        )

        def add_quant_linear(name, activation):
            w = graph.placeholder(f"{name}_w")
            w_scale = graph.placeholder(f"{name}_w_scale")
            w_offset = graph.placeholder(f"{name}_w_offset")
            x_scale = graph.placeholder(f"{name}_x_scale")
            x_offset = graph.placeholder(f"{name}_x_offset")
            bias = graph.placeholder(f"{name}_bias")
            return graph.call_function(
                torch.ops.tensor_cast.static_quant_linear.default,
                args=(
                    activation,
                    w,
                    w_scale,
                    w_offset,
                    x_scale,
                    x_offset,
                    bias,
                    torch.bfloat16,
                ),
            )

        gate_up_0 = add_quant_linear("gate_up_0", dispatched)
        split_0 = graph.call_function(torch.ops.aten.split_with_sizes.default, args=(gate_up_0, [4, 4], 1))
        gate_0 = graph.call_function(operator.getitem, args=(split_0, 0))
        up_0 = graph.call_function(operator.getitem, args=(split_0, 1))
        swiglu_0 = graph.call_function(torch.ops.tensor_cast.swiglu.default, args=(gate_0, up_0))
        down_0 = add_quant_linear("down_0", swiglu_0)

        gate_up_1 = add_quant_linear("gate_up_1", dispatched)
        split_1 = graph.call_function(torch.ops.aten.split_with_sizes.default, args=(gate_up_1, [4, 4], 1))
        gate_1 = graph.call_function(operator.getitem, args=(split_1, 0))
        up_1 = graph.call_function(operator.getitem, args=(split_1, 1))
        swiglu_1 = graph.call_function(torch.ops.tensor_cast.swiglu.default, args=(gate_1, up_1))
        down_1 = add_quant_linear("down_1", swiglu_1)

        combined = graph.call_function(torch.ops.aten.cat.default, args=([down_0, down_1], 0))
        reduced = graph.call_function(
            torch.ops.tensor_cast.all_to_all.default,
            args=(combined, input_splits, output_splits, rank, rank_group),
        )
        unpermuted = graph.call_function(
            torch.ops.tensor_cast.unpermute_tokens.default,
            args=(reduced, expert_indices),
        )
        graph.output(unpermuted)

        gm = fx.GraphModule(torch.nn.Module(), graph)
        region_nodes = {node for node in gm.graph.nodes if node.op != "output"}

        result = DispatchFFNCombinePass()._resolve_dfc_variant(region_nodes)

        self.assertIsNotNone(result)
        _, gmm1_w_args, gmm2_w_args, resolved_rank, resolved_rank_group = result

        self.assertEqual(gmm1_w_args[0], [gate_up_0.args[1], gate_up_1.args[1]])
        self.assertEqual(gmm2_w_args[0], [down_0.args[1], down_1.args[1]])
        self.assertEqual(len(gmm1_w_args), 5)
        self.assertEqual(len(gmm2_w_args), 5)
        self.assertNotIn(gate_up_0.args[1], gmm2_w_args[0])
        self.assertNotIn(gate_up_1.args[1], gmm2_w_args[0])
        self.assertNotIn(gate_up_0.args[4], gmm1_w_args)
        self.assertNotIn(gate_up_0.args[5], gmm1_w_args)
        self.assertNotIn(down_0.args[4], gmm2_w_args)
        self.assertNotIn(down_0.args[5], gmm2_w_args)
        self.assertIs(resolved_rank, rank)
        self.assertIs(resolved_rank_group, rank_group)

    def test_quant_arg_index_mapping_is_shared_between_gmm_and_linear_paths(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        w_scale = graph.placeholder("w_scale")
        w_offset = graph.placeholder("w_offset")
        x_scale = graph.placeholder("x_scale")
        x_offset = graph.placeholder("x_offset")
        bias = graph.placeholder("bias")

        grouped_gmm = graph.call_function(
            torch.ops.tensor_cast.grouped_matmul_quant.default,
            args=(x, w, w_scale, w_offset, x_scale, x_offset, bias, torch.bfloat16),
        )

        expected_indices = DispatchFFNCombinePass._QUANT_WEIGHT_ARG_INDICES
        extracted_args = DispatchFFNCombinePass()._extract_grouped_gmm_args(grouped_gmm)

        self.assertEqual(
            DispatchFFNCombinePass._get_linear_arg_indices_for_dfc(torch.ops.tensor_cast.static_quant_linear.default),
            expected_indices,
        )
        self.assertEqual(
            extracted_args,
            tuple(grouped_gmm.args[i] for i in expected_indices),
        )

    def test_linear_arg_index_helper_rejects_target_without_schema(self):
        def plain_callable():
            return None

        with self.assertRaisesRegex(TypeError, "expects a torch op overload"):
            DispatchFFNCombinePass._get_linear_arg_indices_for_dfc(plain_callable)

    def test_quant_arg_extraction_rejects_missing_args(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        w_scale = graph.placeholder("w_scale")

        grouped_gmm = graph.call_function(
            torch.ops.tensor_cast.grouped_matmul_quant.default,
            args=(x, w, w_scale),
        )

        with self.assertRaisesRegex(ValueError, "Unexpected argument count"):
            DispatchFFNCombinePass()._extract_grouped_gmm_args(grouped_gmm)
