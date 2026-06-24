import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from unittest.mock import patch

import pytest
import torch
from cli.inference.text_generate import main
from parameterized import parameterized
from tensor_cast.core.input_generator import generate_image_inputs, generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.parallel_embedding import ParallelEmbedding
from tensor_cast.model_config import WordEmbeddingTPMode


class TextGenerateTestMixin:
    def setUp(self):
        """Set up test fixtures."""
        self.device = "TEST_DEVICE"
        self.model_id = "Qwen/Qwen3-32B"
        self.num_queries = 2
        self.query_len = 10
        self.context_length = 0
        torch.compiler.reset()

    def _validate_inference_result(self, result: Union[dict, ModelRunnerMetrics], test_name: str = ""):
        """Validate the result from run_inference.

        Args:
            result: Dictionary containing inference metrics
            test_name: Name of the test for better error messages
        """
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        # Check that result is a dictionary
        self.assertIsInstance(result, dict, f"{test_name}: Result should be a dict")

        # Check required keys exist
        required_keys = [
            "total_device_memory_gb",
            "model_weight_size_gb",
            "peak_memory_usage_gb",
            "kv_cache_size_gb",
            "model_activation_size_gb",
            "device_memory_available_gb",
            "execution_time_s",
            "table_result",
            "breakdowns",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"{test_name}: Missing key '{key}' in result")

        # Validate memory metrics are non-negative
        self.assertGreaterEqual(
            result["total_device_memory_gb"],
            0,
            f"{test_name}: Total device memory should be non-negative",
        )
        self.assertGreaterEqual(
            result["model_weight_size_gb"],
            0,
            f"{test_name}: Model weight size should be non-negative",
        )
        self.assertGreaterEqual(
            result["peak_memory_usage_gb"],
            0,
            f"{test_name}: Peak memory usage should be non-negative",
        )
        self.assertGreaterEqual(
            result["kv_cache_size_gb"],
            0,
            f"{test_name}: KV cache size should be non-negative",
        )
        self.assertGreaterEqual(
            result["model_activation_size_gb"],
            0,
            f"{test_name}: Model activation size should be non-negative",
        )

        # Validate memory consistency: peak = weight + kv_cache + activation
        expected_peak = result["model_weight_size_gb"] + result["kv_cache_size_gb"] + result["model_activation_size_gb"]
        self.assertAlmostEqual(
            result["peak_memory_usage_gb"],
            expected_peak,
            places=2,
            msg=f"{test_name}: Peak memory should equal weight + kv_cache + activation",
        )

        # Validate execution time is positive
        exec_time = result["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        self.assertGreater(
            exec_time,
            0,
            f"{test_name}: Execution time should be positive",
        )

        # Validate table result is a string
        self.assertIsInstance(result["table_result"], str, f"{test_name}: Table result should be a string")
        self.assertGreater(
            len(result["table_result"]),
            0,
            f"{test_name}: Table result should not be empty",
        )

        # Validate breakdowns is a dictionary
        self.assertIsInstance(result["breakdowns"], dict, f"{test_name}: Breakdowns should be a dict")


class TestTextGenerate(TextGenerateTestMixin, unittest.TestCase):
    """Unit tests for text_generate.py script."""

    def test_main_given_invalid_log_level_argument_when_invoked_then_system_exits_with_code_2(
        self,
    ):
        '''Test the "main" function in "text_generate"'''
        original_argv = sys.argv

        try:
            sys.argv = [
                self.model_id,
                "--num-queries",
                str(self.num_queries),
                "--query-length",
                str(self.query_len),
                "--log-level",
                "2",
            ]
            with self.assertRaises(SystemExit) as cm:
                main()

            self.assertEqual(cm.exception.code, 2)
        finally:
            sys.argv = original_argv

    def test_basic_prefill(self):
        """Test basic prefill operation without quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=self.num_queries,
            query_len=self.query_len,
            context_length=self.context_length,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_basic_prefill")

    def test_prefix_cache_rewrites_request_info(self):
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=200,
            context_length=1000,
            prefix_cache_hit_rate=0.5,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        request_info = user_input.get_request_info()

        self.assertEqual(request_info.query_len, 100)
        self.assertEqual(request_info.seq_len, 1200)

    def test_prefix_cache_is_ignored_in_decode_mode(self):
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=1,
            context_length=100,
            prefix_cache_hit_rate=0.5,
            decode=True,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        with self.assertLogs("tensor_cast.core.user_config", "WARNING") as captured:
            request_info = user_input.get_request_info()

        self.assertEqual(request_info.query_len, 1)
        self.assertEqual(request_info.seq_len, 101)
        self.assertIn("Ignoring prefix_cache_hit_rate", captured.output[0])

    def test_main_invalid_prefix_cache_hit_rate_exits_with_code_2(self):
        original_argv = sys.argv

        try:
            sys.argv = [
                self.model_id,
                "--num-queries",
                str(self.num_queries),
                "--query-length",
                str(self.query_len),
                "--prefix-cache-hit-rate",
                "1.0",
            ]
            with self.assertRaises(SystemExit) as cm:
                main()

            self.assertEqual(cm.exception.code, 2)
        finally:
            sys.argv = original_argv

    def test_hit_rate_zero_keeps_original_request_info(self):
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=200,
            context_length=1000,
            prefix_cache_hit_rate=0.0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        request_info = user_input.get_request_info()

        self.assertEqual(request_info.query_len, 200)
        self.assertEqual(request_info.seq_len, 1200)

    def test_prefill_with_context(self):
        """Test prefill with context length (similar to README example)."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=100,
            context_length=200,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_prefill_with_context")

    def test_prefill_with_w8a8_dynamic_quant(self):
        """Test prefill with W8A8 dynamic quantization (README example)."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_prefill_with_w8a8_dynamic_quant")

    def test_decode_with_w8a8_static_quant(self):
        """Test decode with W8A8 static quantization (README example)."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=10,
            query_len=1,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_decode_with_w8a8_static_quant")

    def test_decode_mode(self):
        """Test decode mode with single token input."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_decode_mode")

    def test_w4a8_dynamic_quantization(self):
        """Test with W4A8 dynamic quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_w4a8_dynamic_quantization")

    def test_w4a8_static_quantization(self):
        """Test with W4A8 static quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_STATIC,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_w4a8_static_quantization")

    def test_fp8_quantization(self):
        """Test with FP8 quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.FP8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_fp8_quantization")

    def test_fp8_with_context(self):
        """Test FP8 quantization with context length."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.FP8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_fp8_with_context")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertGreater(result["kv_cache_size_gb"], 0)

    def test_fp8_decode_mode(self):
        """Test FP8 quantization in decode mode."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.FP8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_fp8_decode_mode")

    def test_mxfp4_quantization(self):
        """Test with MXFP4 quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_mxfp4_quantization")

    def test_mxfp4_with_context(self):
        """Test MXFP4 quantization with context length."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_mxfp4_with_context")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertGreater(result["kv_cache_size_gb"], 0)

    def test_mxfp4_decode_mode(self):
        """Test MXFP4 quantization in decode mode."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_mxfp4_decode_mode")

    def test_kvcache_int8_quantization(self):
        """Test with INT8 KV cache quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_kvcache_int8_quantization")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertGreater(result["kv_cache_size_gb"], 0)
        self.assertIn("tensor_cast.attention_quant", result["table_result"])

    def test_kvcache_int8_with_linear_quant(self):
        """Test INT8 KV cache quantization combined with linear quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_kvcache_int8_with_linear_quant")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertGreater(result["kv_cache_size_gb"], 0)
        self.assertIn("tensor_cast.attention_quant", result["table_result"])

    def test_kvcache_int8_decode_mode(self):
        """Test INT8 KV cache quantization in decode mode."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_kvcache_int8_decode_mode")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.attention_quant", result["table_result"])

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_mla_int8_with_linear_quant(self, model_id):
        """Test INT8 KV cache quantization combined with linear quantization."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_kvcache_int8_with_linear_quant")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertGreater(result["kv_cache_size_gb"], 0)
        self.assertIn("tensor_cast.multihead_latent_attention_quant", result["table_result"])

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_mla_int8_decode_mode(self, model_id):
        """Test INT8 KV cache quantization in decode mode."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_kvcache_int8_decode_mode")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.multihead_latent_attention_quant", result["table_result"])

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_mlapo_quant_disabled(self, model_id):
        """Ensure MLAPO fusion stays enabled when linear quantization is disabled."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=2,
            query_len=32,
            context_length=64,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            quantize_attention_action=QuantizeAttentionAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.mlapo.default", result["table_result"])

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_mlapo_linear_quant(self, model_id):
        """Ensure MLAPO fusion stays enabled when linear quantization is applied."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=2,
            query_len=32,
            context_length=64,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=QuantizeAttentionAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.mlapo_quant.default", result["table_result"])

    @parameterized.expand(
        [
            ["zai-org/GLM-4.5V"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_moe_gating_top_k_softmax(self, model_id):
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=1,
            query_len=1,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_moe_gating_top_k_softmax")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.moe_gating_top_k_softmax.default", result["table_result"])

    def _run_test_gate_returns_precomputed_topk(self, model_id):
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=1,
            query_len=1,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_gate_returns_precomputed_topk")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertNotIn("tensor_cast.moe_gating_top_k_softmax.default", result["table_result"])

    @parameterized.expand(
        [
            ["baidu/ERNIE-4.5-300B-A47B-PT"],
            ["XiaomiMiMo/MiMo-V2-Flash"],
            ["MiniMaxAI/MiniMax-M2"],
            ["Qwen/Qwen3.5-397B-A17B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
            ["Qwen/Qwen3-VL-30B-A3B-Instruct"],
        ]
    )
    def test_gate_returns_precomputed_topk(self, model_id):
        self._run_test_gate_returns_precomputed_topk(model_id)

    def _run_test_single_token_prefill_vs_decode(self, model_id):
        """Test that single-token prefill is slower than single-token decode."""
        prefill_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=1,
            query_len=1,
            context_length=0,
        )
        prefill_runner = ModelRunner(prefill_input)
        prefill_result = prefill_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(prefill_result, "test_single_token_prefill")

        decode_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=1,
            query_len=1,
            context_length=10,
        )
        decode_runner = ModelRunner(decode_input)
        decode_result = decode_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(decode_result, "test_single_token_decode")

        if isinstance(prefill_result, ModelRunnerMetrics) and isinstance(decode_result, ModelRunnerMetrics):
            prefill_time = prefill_result.execution_time_s.get("analytic", 0) * 1e6
            decode_time = decode_result.execution_time_s.get("analytic", 0) * 1e6
            min_prefill = decode_time * (1 - 1e-3)
            self.assertGreater(
                prefill_time,
                min_prefill,
                f"Single-token prefill should be slower than decode, but got prefill={prefill_time}"
                f" vs decode={decode_time} (min prefill={min_prefill})",
            )

    def test_with_quantized_lmhead(self):
        """Test with LM head quantization enabled."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            quantize_lmhead=True,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_with_quantized_lmhead")

    def test_tensor_parallel(self):
        """Test with tensor parallelism."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_tensor_parallel")

    def test_data_parallel(self):
        """Test with data parallelism."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=4,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=1,
            dp_size=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_data_parallel")

    def test_mixed_parallelism(self):
        """Test with mixed TP and DP."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=4,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            dp_size=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_mixed_parallelism")

    def _run_test_with_different_parallel_mtp_tokens(self, tp_size, ep_size, moe_tp_size, do_compile):
        """Test with MTP (Multi-Token Prediction) tokens."""
        user_input = UserInputConfig(
            device=self.device,
            model_id="deepseek-ai/DeepSeek-V3.1",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=do_compile,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            num_mtp_tokens=2,
            world_size=16,
            tp_size=tp_size,
            dp_size=16 // tp_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=16 // moe_tp_size // ep_size,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_with_different_parallel_mtp_tokens")

    @parameterized.expand(
        [
            [1, 16, 1, False],
            [4, 4, 1, False],
        ]
    )
    def test_with_different_parallel_mtp_tokens(self, tp_size, ep_size, moe_tp_size, do_compile):
        self._run_test_with_different_parallel_mtp_tokens(tp_size, ep_size, moe_tp_size, do_compile)

    def test_with_auto_mtp(self):
        """Test with MTP (Multi-Token Prediction) tokens with auto mode."""
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-32B",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            num_mtp_tokens=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_with_auto_mtp")

    def test_disable_repetition(self):
        """Test with repetition disabled."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            disable_repetition=True,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_disable_repetition")

    def test_with_reserved_memory(self):
        """Test with reserved memory configuration."""
        reserved_gb = 5
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            reserved_memory_gb=reserved_gb,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_with_reserved_memory")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        expected_available = result["total_device_memory_gb"] - result["peak_memory_usage_gb"] - reserved_gb
        self.assertAlmostEqual(result["device_memory_available_gb"], expected_available, places=2)

    def test_num_hidden_layers_override(self):
        """Test with overridden number of hidden layers."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            num_hidden_layers_override=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_num_hidden_layers_override")

    def test_mlp_specific_parallelism(self):
        """Test with MLP-specific tensor/data parallelism."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            mlp_tp_size=2,
            mlp_dp_size=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_mlp_specific_parallelism")

    def test_lmhead_specific_parallelism(self):
        """Test with LM head-specific tensor/data parallelism."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            lmhead_tp_size=2,
            lmhead_dp_size=2,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_lmhead_specific_parallelism")

    def test_expert_parallel(self):
        """Test with expert parallelism enabled."""
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-235B-A22B",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=1,
            ep_size=2,
            moe_tp_size=1,
            moe_dp_size=1,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_expert_parallel")

    def test_invalid_device(self):
        """Test with invalid device name."""
        with self.assertRaises(ValueError):
            user_input = UserInputConfig(
                device="INVALID_DEVICE",
                model_id=self.model_id,
                num_queries=self.num_queries,
                query_len=self.query_len,
                context_length=self.context_length,
                do_compile=False,
                allow_graph_break=False,
                quantize_linear_action=QuantizeLinearAction.DISABLED,
            )
            model_runner = ModelRunner(user_input)
            model_runner.run_inference(generate_inputs_func=generate_inputs)

    def test_large_batch_size(self):
        """Test with large batch size."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=32,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_large_batch_size")

    def test_long_context(self):
        """Test with long context length."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=500,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_long_context")

    def test_padding(self):
        """Test with padding tokens."""
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-235B-A22B",
            num_queries=1,
            query_len=1,
            context_length=500,
            world_size=16,
            ep_size=16,
            moe_tp_size=1,
            moe_dp_size=1,
            tp_size=2,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_padding")

    def test_o_proj_specific_parallelism(self):
        """Test with o_proj-specific tensor/data parallelism."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            o_proj_tp_size=4,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_o_proj_specific_parallelism")

    @parameterized.expand([["col"], ["row"]])
    def test_word_embedding_parallel(self, embedding_tp_mode):
        """Test with word embedding parallel."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            word_embedding_tp=embedding_tp_mode,
        )
        model_runner = ModelRunner(user_input)
        embedding_layers = [module for module in model_runner.model.modules() if isinstance(module, ParallelEmbedding)]
        self.assertGreaterEqual(
            len(embedding_layers),
            1,
            "Expected at least one ParallelEmbedding when word_embedding_tp is enabled.",
        )
        embedding_layer = max(embedding_layers, key=lambda module: module.num_embeddings)
        self.assertEqual(embedding_layer.shard_mode, WordEmbeddingTPMode(embedding_tp_mode))
        sharded_vocab, sharded_hidden = embedding_layer._inner.weight.shape
        if embedding_tp_mode == WordEmbeddingTPMode.col.value:
            self.assertEqual(sharded_vocab, embedding_layer.num_embeddings)
            self.assertLess(sharded_hidden, embedding_layer.embedding_dim)
            self.assertGreaterEqual(sharded_hidden * embedding_layer.tp_size, embedding_layer.embedding_dim)
        else:
            self.assertEqual(sharded_hidden, embedding_layer.embedding_dim)
            self.assertLess(sharded_vocab, embedding_layer.num_embeddings)
            self.assertGreaterEqual(sharded_vocab * embedding_layer.tp_size, embedding_layer.num_embeddings)
            self.assertLess(embedding_layer._row_start, embedding_layer._row_end)
            self.assertLessEqual(embedding_layer._row_end, embedding_layer._vocab_size)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, f"test_word_embedding_parallel_{embedding_tp_mode}")

    def test_qwen3_32b_tp16(self):
        """Make sure tp_size can be greater than num_key_value_heads."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=16,
            tp_size=16,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "qwen3_32b_tp16")

    def test_redundant_experts(self):
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-235B-A22B",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=16,
            tp_size=16,
            ep_size=16,
            moe_tp_size=1,
            moe_dp_size=1,
            enable_redundant_experts=True,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_redundant_experts")

    @parameterized.expand(
        [
            [True],
            [False],
        ]
    )
    def test_external_shared_experts(self, host_external_shared_experts):
        user_input = UserInputConfig(
            device=self.device,
            model_id="deepseek-ai/DeepSeek-V3.1",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=16,
            tp_size=16,
            ep_size=16,
            moe_dp_size=1,
            moe_tp_size=1,
            enable_external_shared_experts=True,
            host_external_shared_experts=host_external_shared_experts,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_external_shared_experts")

    @parameterized.expand(
        [
            ["Qwen/Qwen3-VL-32B-Instruct"],
            ["Qwen/Qwen3-VL-30B-A3B-Instruct"],
            ["zai-org/GLM-4.5V"],
        ]
    )
    def test_vl_with_basic_prefill(self, model_id):
        """Test vl prefill operation."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=self.num_queries,
            query_len=self.query_len,
            context_length=self.context_length,
            image_batch_size=1,
            image_width=1920,
            image_height=1080,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        self.assertTrue(model_runner.model.is_vl_model, msg="Model should be vl model")
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )
        image_kwargs = generate_image_inputs(
            model_runner.model,
            user_input.image_batch_size,
            user_input.image_height,
            user_input.image_width,
            user_input.num_queries,
        )
        num_image_tokens = image_kwargs.get("num_image_tokens")
        seq_len = input_kwargs.get("attention_meta").seq_lens[0].item()
        self.assertEqual(seq_len, num_image_tokens + user_input.context_length + user_input.query_len)
        query_len = input_kwargs.get("attention_meta").query_lens[0].item()
        self.assertEqual(query_len, num_image_tokens + user_input.query_len)
        self.assertIn("pixel_values", input_kwargs)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_qwen3_vl_with_basic_prefill")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("aten.addmm.default", result["table_result"])

    def test_qwen3_vl_without_img_prefill(self):
        """Test qwen3_vl without image input prefill operation."""
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            num_queries=self.num_queries,
            query_len=self.query_len,
            context_length=self.context_length,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        self.assertTrue(model_runner.model.is_vl_model, msg="Model should be vl model")
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )
        self.assertNotIn("pixel_values", input_kwargs)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_qwen3_vl_without_img_prefill")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertNotIn("aten.addmm.default", result["table_result"])

    def test_qwen3_vl_decode_mode(self):
        """Test qwen3_vl decode mode"""
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            num_queries=self.num_queries,
            query_len=self.query_len,
            context_length=self.context_length,
            image_batch_size=1,
            image_width=1920,
            image_height=1080,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            decode=True,
        )
        model_runner = ModelRunner(user_input)
        self.assertTrue(model_runner.model.is_vl_model, msg="Model should be vl model")
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )
        image_kwargs = generate_image_inputs(
            model_runner.model,
            user_input.image_batch_size,
            user_input.image_height,
            user_input.image_width,
            user_input.num_queries,
        )
        num_image_tokens = image_kwargs.get("num_image_tokens")
        seq_len = input_kwargs.get("attention_meta").seq_lens[0].item()
        self.assertEqual(seq_len, num_image_tokens + user_input.context_length + user_input.query_len)
        query_len = input_kwargs.get("attention_meta").query_lens[0].item()
        self.assertEqual(query_len, user_input.query_len)
        self.assertNotIn("pixel_values", input_kwargs)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_qwen3_vl_decode_mode")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertNotIn("aten.addmm.default", result["table_result"])

    @parameterized.expand(
        [
            ["Qwen/Qwen3-VL-32B-Instruct", False],
            ["Qwen/Qwen3-VL-30B-A3B-Instruct", True],
            ["zai-org/GLM-4.5V", True],
        ]
    )
    def test_vl_parallel(self, model_id, ep):
        """Test vl parallel operation."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=self.num_queries,
            query_len=self.query_len,
            context_length=self.context_length,
            image_batch_size=1,
            image_width=1920,
            image_height=1080,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=2,
            ep_size=2 if ep else 1,
            moe_dp_size=1 if ep else 2,
            moe_tp_size=1,
        )
        model_runner = ModelRunner(user_input)
        self.assertTrue(model_runner.model.is_vl_model, msg="Model should be vl model")
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )
        self.assertIn("pixel_values", input_kwargs)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_qwen3_vl_with_basic_prefill")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("aten.addmm.default", result["table_result"])
        self.assertIn("tensor_cast.all_reduce.default", result["table_result"])
        self.assertIn("tensor_cast.all_gather.default", result["table_result"])
        if ep:
            self.assertIn("tensor_cast.all_to_all.default", result["table_result"])
        else:
            self.assertNotIn("tensor_cast.all_to_all.default", result["table_result"])

    @parameterized.expand(
        [
            ["inclusionAI/Ling-1T"],
            ["inclusionAI/Ling-flash-2.0"],
        ]
    )
    def test_ling_basic(self, model_id):
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=1,
            query_len=1,
            context_length=7,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=64,
        )
        model_runner = ModelRunner(user_input)
        _ = model_runner.run_inference(generate_inputs_func=generate_inputs)

    def test_ling_tp_size_greater_than_num_kv_heads(self):
        user_input = UserInputConfig(
            device=self.device,
            model_id="inclusionAI/Ling-1T",
            num_queries=1,
            query_len=1,
            context_length=7,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=64,
            tp_size=16,
        )
        model_runner = ModelRunner(user_input)
        _ = model_runner.run_inference(generate_inputs_func=generate_inputs)

    def test_tps_per_model_basic(self):
        # test config
        num_queries = 3
        query_len = 2500
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=num_queries,
            query_len=query_len,
            context_length=7,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=64,
            tp_size=16,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        exec_time = result["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        expected_tps = (num_queries * query_len) / (exec_time * user_input.world_size)
        actual_tps = next(iter(result["tps_per_model"].values()))
        tolerance = expected_tps * 0.05
        if tolerance < 1e-10:  # avoid too small tolerance
            tolerance = max(abs(expected_tps * 0.01), 1e-6)
        self.assertAlmostEqual(
            actual_tps,
            expected_tps,
            delta=tolerance,
            msg=(
                f"TPS calculation is wrong: expected={expected_tps:.4g}, "
                f"actual={actual_tps:.4g}, tolerance={tolerance:.2g}"
            ),
        )

    @parameterized.expand(
        [
            ["inclusionAI/Ling-1T", 8, 8],
            ["Qwen/Qwen3-235B-A22B", 16, 4],
            ["deepseek-ai/DeepSeek-V3.1", 4, 16],
            ["Qwen/Qwen3-32B", 8, 8],  # non moe model, should ignore ep-size
        ]
    )
    def test_ep_moe_tp_hybrid(self, model_id, ep_size, moe_tp_size):
        user_input = UserInputConfig(
            device=self.device,
            model_id=model_id,
            num_queries=1,
            query_len=1,
            context_length=7,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=64,
            tp_size=8,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
        )
        model_runner = ModelRunner(user_input)
        _ = model_runner.run_inference(generate_inputs_func=generate_inputs)


@pytest.mark.nightly
class TestTextGenerateNightly(TextGenerateTestMixin, unittest.TestCase):
    def test_with_compilation(self):
        """Test with torch.compile enabled."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_with_compilation")

    def test_with_compilation_and_graph_break(self):
        """Test with torch.compile and allow graph break."""
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=True,
            allow_graph_break=True,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_with_compilation_and_graph_break")

    @parameterized.expand(
        [
            [2, 16, 1],
            [8, 4, 2],
        ]
    )
    def test_with_different_parallel_mtp_tokens(self, tp_size, ep_size, moe_tp_size):
        TestTextGenerate._run_test_with_different_parallel_mtp_tokens(self, tp_size, ep_size, moe_tp_size, True)

    def test_qwen3_32b_4_a3die_decode_result(self):
        """Make sure the result of qwen3-32b model on 4 A3 dies is as expected in some range"""
        user_input = UserInputConfig(
            device="ATLAS_800_A3_560T_128G_DIE",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=4,
            tp_size=4,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "qwen3_32b_4_a3die_decode")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        exec_time = result["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        self.assertLess(exec_time, 0.0328)

    def test_deepseek_v3_1_a3_ep64_decode_result(self):
        """Make sure the result of deepseek v3.1 model on 64 A3 dies with EP 64 is as expected in some range"""
        user_input = UserInputConfig(
            device="ATLAS_800_A3_560T_128G_DIE",
            model_id="deepseek-ai/DeepSeek-V3.1",
            num_queries=256,
            query_len=4,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            world_size=64,
            num_mtp_tokens=3,
            ep_size=64,
            moe_tp_size=1,
            moe_dp_size=1,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_deepseek_v3_1_a3_ep64_decode")
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        exec_time = result["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        self.assertLess(exec_time, 0.063)

    def test_fullmesh_subgroup_bandwidth_result(self):
        """Full Mesh with subgroup bandwidth is smaller than CLOS"""
        user_input_a3 = UserInputConfig(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=4,
            tp_size=4,
        )
        model_runner_a3 = ModelRunner(user_input_a3)
        result_a3 = model_runner_a3.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result_a3)
        user_input_a2 = UserInputConfig(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=4,
            tp_size=4,
        )
        model_runner_a2 = ModelRunner(user_input_a2)
        result_a2 = model_runner_a2.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result_a2)
        if isinstance(result_a3, ModelRunnerMetrics):
            result_a3 = asdict(result_a3)
        if isinstance(result_a2, ModelRunnerMetrics):
            result_a2 = asdict(result_a2)
        exec_time_a3 = result_a3["execution_time_s"]
        exec_time_a2 = result_a2["execution_time_s"]
        if isinstance(exec_time_a3, dict):
            exec_time_a3 = next(iter(exec_time_a3.values()))
        if isinstance(exec_time_a2, dict):
            exec_time_a2 = next(iter(exec_time_a2.values()))
        self.assertLess(exec_time_a3, exec_time_a2)

    def test_fullmesh_fullgroup_bandwidth_result(self):
        """Full Mesh with full group bandwidth is smaller than CLOS"""
        user_input_a3 = UserInputConfig(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=8,
            tp_size=8,
        )
        model_runner_a3 = ModelRunner(user_input_a3)
        result_a3 = model_runner_a3.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result_a3)
        user_input_a2 = UserInputConfig(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=8,
            tp_size=8,
        )
        model_runner_a2 = ModelRunner(user_input_a2)
        result_a2 = model_runner_a2.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result_a2)
        if isinstance(result_a3, ModelRunnerMetrics):
            result_a3 = asdict(result_a3)
        if isinstance(result_a2, ModelRunnerMetrics):
            result_a2 = asdict(result_a2)
        exec_time_a3 = result_a3["execution_time_s"]
        exec_time_a2 = result_a2["execution_time_s"]
        if isinstance(exec_time_a3, dict):
            exec_time_a3 = next(iter(exec_time_a3.values()))
        if isinstance(exec_time_a2, dict):
            exec_time_a2 = next(iter(exec_time_a2.values()))
        self.assertEqual(exec_time_a3, exec_time_a2)

    @parameterized.expand(
        [
            [QuantizeLinearAction.W8A8_DYNAMIC],
            [QuantizeLinearAction.W8A8_STATIC],
            [QuantizeLinearAction.DISABLED],
        ]
    )
    def test_qwen2_5_with_compile(self, quant_linear_action):
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen2.5-7B",
            num_queries=2,
            query_len=1,
            context_length=500,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=quant_linear_action,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_qwen2_5_with_compile")

    @parameterized.expand(
        [
            [QuantizeLinearAction.W8A8_DYNAMIC, False, False],
            [QuantizeLinearAction.W8A8_STATIC, False, False],
            [QuantizeLinearAction.DISABLED, False, False],
            [QuantizeLinearAction.W8A8_DYNAMIC, True, False],
            [QuantizeLinearAction.W8A8_STATIC, True, False],
            [QuantizeLinearAction.DISABLED, True, False],
            [QuantizeLinearAction.W8A8_DYNAMIC, False, True],
            [QuantizeLinearAction.DISABLED, False, True],
        ]
    )
    def test_gmm_fusion(self, quant_linear_action, enable_ep, enable_tp):
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-235B-A22B",
            num_queries=2,
            query_len=1,
            context_length=500,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=quant_linear_action,
            world_size=8,
            ep_size=8 if enable_ep else 1,
            moe_dp_size=1 if enable_ep else 8,
            moe_tp_size=1,
            tp_size=8 if enable_tp else 1,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.grouped_matmul", result["table_result"])

    def test_vl_moe_tp_ep_different_parallel(self):
        """Test vl moe tp ep different parallel"""
        user_input = UserInputConfig(
            device=self.device,
            model_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
            num_queries=4,
            query_len=20,
            image_batch_size=4,
            image_width=1920,
            image_height=1080,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            world_size=8,
            tp_size=2,
            ep_size=8,
            moe_dp_size=1,
            moe_tp_size=1,
        )
        model_runner = ModelRunner(user_input)
        self.assertTrue(model_runner.model.is_vl_model, msg="Model should be vl model")
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )
        self.assertIn("pixel_values", input_kwargs)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        self._validate_inference_result(result, "test_vl_moe_tp_ep_different_parallel")

    @parameterized.expand(
        [
            ["baidu/ERNIE-4.5-300B-A47B-PT"],
            ["Qwen/Qwen3.5-397B-A17B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
        ]
    )
    def test_gate_returns_precomputed_topk(self, model_id):
        TestTextGenerate._run_test_gate_returns_precomputed_topk(self, model_id)

    @parameterized.expand(
        [
            ["Qwen/Qwen3.5-397B-A17B"],
            ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
        ]
    )
    def test_single_token_prefill_vs_decode(self, model_id):
        """Develop guard: huge MoE models; too slow for non-nightly regression."""
        TestTextGenerate._run_test_single_token_prefill_vs_decode(self, model_id)


class TestModelRunnerMetricsPrintInfo(unittest.TestCase):
    """Unit tests for ModelRunner.print_info static method."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.metrics = ModelRunnerMetrics(
            total_device_memory_gb=24.0,
            model_weight_size_gb=5.0,
            peak_memory_usage_gb=12.0,
            kv_cache_size_gb=3.0,
            kv_cache_per_token_gb=0.001,
            model_activation_size_gb=4.0,
            reserved_memory_gb=1.0,
            device_memory_available_gb=6.0,
            tps_per_model={"analytic": 200.0},
            execution_time_s={"analytic": 0.05},
            run_time_s=0.06,
            batch_size=4,
            table_result="performance_data",
            breakdowns={
                "memory": {"activation": 2.0, "weights": 3.0},
                "compute": {"matmul": 1.5, "attention": 0.8},
            },
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_info_basic(self, mock_stdout):
        """Test that print_info prints the expected information."""
        # Call the static method
        self.metrics.print_info()

        # Get the printed output
        output = mock_stdout.getvalue()

        # Check that the output contains expected elements
        self.assertIn("Total device memory: 24.000 GB", output)
        self.assertIn("Model weight size: 5.000 GB", output)
        self.assertIn("KV cache: 3.000 GB", output)
        self.assertIn("Model activation size: 4.000 GB", output)
        self.assertIn("Reserved memory: 1.000 GB", output)
        self.assertIn("Memory available: 6.000 GB", output)

        # Check that breakdowns are printed
        self.assertIn("Stats breakdowns:", output)
        self.assertIn("memory", output)
        self.assertIn("compute", output)
        self.assertIn("matmul", output)
        self.assertIn("attention", output)

    def test_dump_json_writes_expected_payload(self):
        """ModelRunnerMetrics.dump_json should write the full metrics payload."""
        self.metrics.perf_model_name = "analytic"
        self.metrics.runtime_event_list = [
            {
                "name": "aten.matmul",
                "perf_model": "analytic",
                "perf_total": 0.4,
                "perf_avg": 0.2,
                "call_times": 2,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            self.metrics.dump_json(str(output_path))

            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertEqual(payload["batch_size"], 4)
        self.assertAlmostEqual(payload["run_time_s"], 0.06)
        self.assertEqual(payload["execution_time_s"], {"analytic": 0.05})
        self.assertEqual(payload["tps_per_model"], {"analytic": 200.0})

        memory = payload["memory_gb"]
        self.assertAlmostEqual(memory["total_device"], 24.0)
        self.assertAlmostEqual(memory["model_weight"], 5.0)
        self.assertAlmostEqual(memory["peak_usage"], 12.0)
        self.assertAlmostEqual(memory["kv_cache"], 3.0)
        self.assertAlmostEqual(memory["kv_cache_per_token"], 0.001)
        self.assertAlmostEqual(memory["model_activation"], 4.0)
        self.assertAlmostEqual(memory["reserved"], 1.0)
        self.assertAlmostEqual(memory["available"], 6.0)

        # breakdowns_raw is the unmodified mapping
        self.assertEqual(payload["breakdowns_raw"]["memory"], {"activation": 2.0, "weights": 3.0})
        # breakdowns_percent sums to 100 per category
        for category, percent in payload["breakdowns_percent"].items():
            self.assertAlmostEqual(sum(percent.values()), 100.0, places=2, msg=category)

        self.assertEqual(payload["perf_model_name"], "analytic")
        self.assertEqual(payload["runtime_event_list"], self.metrics.runtime_event_list)

    def test_dump_json_skips_zero_total_breakdowns_in_percent(self):
        """Breakdowns whose values sum to zero should not appear in breakdowns_percent."""
        self.metrics.breakdowns = {
            "empty": {"a": 0.0, "b": 0.0},
            "non_empty": {"a": 1.0, "b": 3.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            self.metrics.dump_json(str(output_path))
            with output_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertNotIn("empty", payload["breakdowns_percent"])
        self.assertIn("non_empty", payload["breakdowns_percent"])
        self.assertAlmostEqual(payload["breakdowns_percent"]["non_empty"]["a"], 25.0)
        self.assertAlmostEqual(payload["breakdowns_percent"]["non_empty"]["b"], 75.0)


class TestAggregateRuntimeEvents(unittest.TestCase):
    """Unit tests for ModelRunner._aggregate_runtime_events."""

    @staticmethod
    def _event(func_name: str, perf_results):
        return SimpleNamespace(
            op_invoke_info=SimpleNamespace(func=func_name),
            perf_results=perf_results,
        )

    @staticmethod
    def _result(t: float):
        return SimpleNamespace(execution_time_s=t)

    def test_aggregates_by_func_and_sorts_by_total_descending(self):
        events = [
            self._event("aten.matmul", {"empirical": self._result(0.1)}),
            self._event("aten.softmax", {"empirical": self._result(0.05)}),
            self._event("aten.matmul", {"empirical": self._result(0.3)}),
        ]

        # _aggregate_runtime_events does not use self; call as unbound.
        result = ModelRunner._aggregate_runtime_events(None, events, perf_model_name="empirical")

        self.assertEqual([entry["name"] for entry in result], ["aten.matmul", "aten.softmax"])
        matmul = next(entry for entry in result if entry["name"] == "aten.matmul")
        self.assertEqual(matmul["perf_model"], "empirical")
        self.assertAlmostEqual(matmul["perf_total"], 0.4)
        self.assertAlmostEqual(matmul["perf_avg"], 0.2)
        self.assertEqual(matmul["call_times"], 2)

        softmax = next(entry for entry in result if entry["name"] == "aten.softmax")
        self.assertAlmostEqual(softmax["perf_total"], 0.05)
        self.assertAlmostEqual(softmax["perf_avg"], 0.05)
        self.assertEqual(softmax["call_times"], 1)

    def test_counts_event_when_perf_model_missing(self):
        """Events without the requested perf model still increment call count."""
        events = [
            self._event("aten.add", {"analytic": self._result(0.2)}),
            self._event("aten.add", {"empirical": self._result(0.1)}),
        ]

        result = ModelRunner._aggregate_runtime_events(None, events, perf_model_name="empirical")

        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry["name"], "aten.add")
        self.assertEqual(entry["call_times"], 2)
        self.assertAlmostEqual(entry["perf_total"], 0.1)
        # avg is total / count regardless of which events had the perf result
        self.assertAlmostEqual(entry["perf_avg"], 0.05)

    def test_respects_custom_perf_model_name(self):
        events = [
            self._event("aten.matmul", {"analytic": self._result(0.4)}),
            self._event("aten.matmul", {"empirical": self._result(0.9)}),
        ]

        result = ModelRunner._aggregate_runtime_events(None, events, perf_model_name="analytic")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["perf_model"], "analytic")
        self.assertAlmostEqual(result[0]["perf_total"], 0.4)
        self.assertEqual(result[0]["call_times"], 2)

    def test_no_perf_model_name_records_counts_only(self):
        """When perf_model_name is None, durations are zero but counts still aggregate."""
        events = [
            self._event("aten.add", {"analytic": self._result(0.2)}),
            self._event("aten.add", {"empirical": self._result(0.1)}),
        ]

        result = ModelRunner._aggregate_runtime_events(None, events)

        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertIsNone(entry["perf_model"])
        self.assertEqual(entry["call_times"], 2)
        self.assertEqual(entry["perf_total"], 0.0)
        self.assertEqual(entry["perf_avg"], 0.0)

    def test_empty_event_list_returns_empty(self):
        self.assertEqual(ModelRunner._aggregate_runtime_events(None, []), [])


class TestUserInputConfigPrintInfo(unittest.TestCase):
    """Unit tests for UserInputConfig._print_info."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_info_includes_multimodal_and_bound_options(self, mock_stdout):
        user_config = UserInputConfig(
            device="TEST_DEVICE",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            num_queries=2,
            query_len=128,
            context_length=256,
            decode=True,
            dump_input_shapes=True,
            dump_op_bound_results=True,
            image_batch_size=1,
            image_height=720,
            image_width=1080,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )

        user_config._print_info()

        output = mock_stdout.getvalue()
        self.assertIn("Device: TEST_DEVICE", output)
        self.assertIn("Model ID: Qwen/Qwen3-VL-8B-Instruct", output)
        self.assertIn("Number of Queries: 2", output)
        self.assertIn("Input Length (per query): 128", output)
        self.assertIn("Context Length (per query): 256", output)
        self.assertIn("Is Decode: True", output)
        self.assertIn("Quantization Linear: MXFP4", output)
        self.assertIn("MXFP4 group size: 32", output)
        self.assertIn("Quantization Attention: INT8", output)
        self.assertIn("Group table averages by input shapes: True", output)
        self.assertIn("Dump operator bound ratios: True", output)
        self.assertIn("image_batch_size: 1", output)
        self.assertIn("image_height: 720", output)
        self.assertIn("image_width: 1080", output)


if __name__ == "__main__":
    unittest.main()
