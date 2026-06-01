import unittest
from dataclasses import asdict
from typing import Union

import pytest
import torch
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


@pytest.mark.nightly
class TestKimiK25(unittest.TestCase):
    """Unit tests for Kimi K2.5 model simulation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "ATLAS_800_A3_560T_128G_DIE"
        self.model_id = "moonshotai/Kimi-K2.5"
        torch.compiler.reset()

    def _validate_inference_result(self, result: Union[dict, ModelRunnerMetrics], test_name: str = ""):
        """
        Validate the result from run_inference.

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

    def test_kimi_k25_text_only_generation(self):
        """
        Test Case 1: Text-only Generation Simulation

        Validates Kimi K2.5 text inference performance under complex parallel
        strategies (TP/EP/DP) and W4A8 dynamic quantization.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=24,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            world_size=16,
            dp_size=2,
            tp_size=8,
            ep_size=16,
            moe_tp_size=1,
            moe_dp_size=1,
            enable_shared_expert_tp=True,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k25_text_only_generation")

        # Additional checks specific to Kimi K2.5 with MoE and MLA
        if isinstance(result, ModelRunnerMetrics):
            result_dict = asdict(result)

        # Verify that quantization-related ops are present in the trace
        self.assertIn(
            "tensor_cast.mlapo_quant",
            result_dict["table_result"],
            "MLA quantization should be present in the operation trace",
        )

        # Verify EP communication ops are present due to ep_size=16
        self.assertIn(
            "tensor_cast.all_to_all", result_dict["table_result"], "EP communication (all_to_all) should be present"
        )

    def test_kimi_k25_vision_language_generation(self):
        """
        Test Case 2: Vision-Language Generation Simulation

        Validates Kimi K2.5 multi-modal pipeline simulation with image input,
        ensuring proper coordination between vision encoder and language model.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=24,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            world_size=16,
            dp_size=2,
            tp_size=8,
            ep_size=16,
            moe_tp_size=1,
            moe_dp_size=1,
            enable_shared_expert_tp=True,
            # Vision-language specific parameters
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        )

        model_runner = ModelRunner(user_input)

        # Verify the model is correctly identified as a VLM
        self.assertTrue(model_runner.model.is_vl_model, msg="Kimi K2.5 should be identified as a vision-language model")

        # Generate inputs to verify visual features are produced
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )

        # Verify that pixel_values or visual features are present
        self.assertIn("pixel_values", input_kwargs, "pixel_values should be present for vision-language input")

        # Run inference
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k25_vision_language_generation")

        # Additional checks for VLM-specific operations
        if isinstance(result, ModelRunnerMetrics):
            result_dict = asdict(result)

        # Verify vision encoder operations are present
        # The visual tower should contribute to the computation trace
        self.assertGreater(
            result_dict["model_weight_size_gb"], 0, "Model weight size should include vision tower weights"
        )

        # Verify that the execution completed successfully with reasonable TPS
        exec_time = result_dict["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))

        # For a model of this size with compilation, execution should complete in reasonable time
        # (This is a sanity check, not a strict performance benchmark)
        self.assertLess(exec_time, 10.0, "Execution time should be reasonable (< 10s for meta device simulation)")

    def test_kimi_k25_long_context_text_generation(self):
        """
        Test Case 3: Long Context Text-only Generation Simulation

        Validates Kimi K2.5 inference performance under long sequence (4500 tokens)
        with complex parallel strategies and W4A8 dynamic quantization.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=24,
            query_len=4500,
            context_length=0,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            world_size=16,
            dp_size=2,
            tp_size=8,
            ep_size=16,
            moe_tp_size=1,
            moe_dp_size=1,
            enable_shared_expert_tp=True,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k25_long_context_text_generation")

        # Additional checks specific to long context scenarios
        if isinstance(result, ModelRunnerMetrics):
            result_dict = asdict(result)

        # Verify that KV cache occupies significant memory due to long context
        self.assertGreater(
            result_dict["kv_cache_size_gb"], 0, "KV cache size should be non-zero for long context generation"
        )

        # Verify MLA quantization ops are present
        self.assertIn(
            "tensor_cast.mlapo_quant",
            result_dict["table_result"],
            "MLA quantization should be present in the operation trace",
        )

        # Verify EP communication ops are present
        self.assertIn(
            "tensor_cast.all_to_all", result_dict["table_result"], "EP communication (all_to_all) should be present"
        )

    def test_kimi_k25_long_context_vision_language_generation(self):
        """
        Test Case 4: Long Context Vision-Language Generation Simulation

        Validates Kimi K2.5 multi-modal pipeline simulation with image input
        and long text context, ensuring proper coordination between vision
        encoder and language model.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=24,
            query_len=4500,
            context_length=0,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            world_size=16,
            dp_size=2,
            tp_size=8,
            ep_size=16,
            moe_tp_size=1,
            moe_dp_size=1,
            enable_shared_expert_tp=True,
            # Vision-language specific parameters
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        )

        model_runner = ModelRunner(user_input)

        # Verify the model is correctly identified as a VLM
        self.assertTrue(model_runner.model.is_vl_model, msg="Kimi K2.5 should be identified as a vision-language model")

        # Generate inputs to verify visual features are produced
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )

        # Verify that pixel_values or visual features are present
        self.assertIn("pixel_values", input_kwargs, "pixel_values should be present for vision-language input")

        # Run inference
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k25_long_context_vision_language_generation")

        # Additional checks for VLM-specific operations
        if isinstance(result, ModelRunnerMetrics):
            result_dict = asdict(result)

        # Verify vision encoder operations contribute to weight size
        self.assertGreater(
            result_dict["model_weight_size_gb"], 0, "Model weight size should include vision tower weights"
        )

        # Verify KV cache is allocated for long context
        self.assertGreater(
            result_dict["kv_cache_size_gb"], 0, "KV cache size should be non-zero for long context generation"
        )

        # Verify that the execution completed successfully with reasonable time
        exec_time = result_dict["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))

        # For a model of this size with compilation, execution should complete in reasonable time
        self.assertLess(
            exec_time, 15.0, "Execution time should be reasonable (< 15s for meta device simulation with long context)"
        )


if __name__ == "__main__":
    unittest.main()
