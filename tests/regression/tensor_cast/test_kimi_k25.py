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
            decode=True,
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

        # Verify that pixel_values behavior is correct based on mode
        # In decode mode, pixel_values are intentionally removed (see input_generator.py line 53-55)
        if user_input.decode:
            self.assertNotIn(
                "pixel_values",
                input_kwargs,
                "pixel_values should NOT be present in decode mode (image input is removed after prefill)",
            )
        else:
            self.assertIn(
                "pixel_values", input_kwargs, "pixel_values should be present for vision-language input in prefill mode"
            )

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
            exec_time, 60.0, "Execution time should be reasonable (< 60s for meta device simulation with long context)"
        )

    def test_kimi_k25_text_only_decode_with_mtp(self):
        """
        Test Case 5: Text-only Decode Simulation with MTP

        Validates Kimi K2.5 text decode inference performance with multi-token
        prediction (3 MTP tokens) under complex parallel strategies (TP/EP/DP)
        and W4A8 dynamic quantization.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=24,
            query_len=4,
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
            num_mtp_tokens=3,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k25_text_only_decode_with_mtp")

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

    def test_kimi_k25_vision_language_decode_with_mtp(self):
        """
        Test Case 6: Vision-Language Decode Simulation with MTP

        Validates Kimi K2.5 multi-modal decode pipeline simulation with image input
        and multi-token prediction (3 MTP tokens), ensuring proper coordination
        between vision encoder and language model.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=24,
            query_len=4,
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
            decode=True,
            num_mtp_tokens=3,
        )

        model_runner = ModelRunner(user_input)

        # Verify the model is correctly identified as a VLM
        self.assertTrue(model_runner.model.is_vl_model, msg="Kimi K2.5 should be identified as a vision-language model")

        # Run inference
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k25_vision_language_decode_with_mtp")

        # Additional checks for VLM-specific operations
        if isinstance(result, ModelRunnerMetrics):
            result_dict = asdict(result)

        # Verify vision encoder operations contribute to weight size
        self.assertGreater(
            result_dict["model_weight_size_gb"], 0, "Model weight size should include vision tower weights"
        )

        # Verify that the execution completed successfully with reasonable time
        exec_time = result_dict["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))

        # For a model of this size with compilation, execution should complete in reasonable time
        # (This is a sanity check, not a strict performance benchmark)
        self.assertLess(exec_time, 10.0, "Execution time should be reasonable (< 10s for meta device simulation)")


class TestKimiK25Patches(unittest.TestCase):
    """Guard-condition / boundary tests for Kimi-K2.5 monkey-patch functions.

    These tests exercise the four functions identified by CI gate
    coverage analysis WITHOUT requiring network access or model
    instantiation, so they can run in the standard ``-m 'not nightly'``
    job.
    """

    def setUp(self):
        """Reset global patch state before each test."""
        import tensor_cast.transformers.builtin_model.kimi_k25 as _km

        _km._patched_kimi_k25 = False
        _km._shard_model_patched = False

    # ------------------------------------------------------------------
    # _hf_config_patch_for_kimi_k25
    # ------------------------------------------------------------------

    def test_hf_config_patch_guard_wrong_model_type(self):
        """Guard: returns early for non-``kimi_k25`` configs."""
        from tensor_cast.transformers.builtin_model.kimi_k25 import _hf_config_patch_for_kimi_k25

        class _Fake:
            model_type = "llama"

        self.assertIsNone(_hf_config_patch_for_kimi_k25(_Fake()))

    def test_hf_config_patch_guard_already_patched(self):
        """Guard: second call is a no-op."""
        import tensor_cast.transformers.builtin_model.kimi_k25 as _km
        from tensor_cast.transformers.builtin_model.kimi_k25 import _hf_config_patch_for_kimi_k25

        _saved = _km._patched_kimi_k25
        try:
            _km._patched_kimi_k25 = True

            class _Kimi:
                model_type = "kimi_k25"
                hidden_size = 7168
                intermediate_size = 18432
                num_attention_heads = 64
                num_key_value_heads = 64
                num_hidden_layers = 61
                vocab_size = 163840

            self.assertIsNone(_hf_config_patch_for_kimi_k25(_Kimi(), model_id=None))
        finally:
            _km._patched_kimi_k25 = _saved

    # ------------------------------------------------------------------
    # _patch_model_classes_for_kimi_k25
    # ------------------------------------------------------------------

    def test_patch_model_classes_guard_wrong_model_type(self):
        """Guard: returns False when model_type is not ``kimi_k25``."""
        from tensor_cast.transformers.builtin_model.kimi_k25 import _patch_model_classes_for_kimi_k25

        class _Fake:
            model_type = "llama"

        self.assertFalse(_patch_model_classes_for_kimi_k25(_Fake(), "moonshotai/Kimi-K2.5"))

    def test_patch_model_classes_guard_none_model_id(self):
        """Guard: returns False when model_id is None."""
        from tensor_cast.transformers.builtin_model.kimi_k25 import _patch_model_classes_for_kimi_k25

        class _Kimi:
            model_type = "kimi_k25"

        self.assertFalse(_patch_model_classes_for_kimi_k25(_Kimi(), None))

    # ------------------------------------------------------------------
    # _patch_shard_model_for_kimi_vl
    # ------------------------------------------------------------------

    def test_patch_shard_model_idempotent(self):
        """Second call is a no-op; restores original afterwards."""
        import tensor_cast.transformers.builtin_model.kimi_k25 as _km
        from tensor_cast.transformers import transformations as _t
        from tensor_cast.transformers.builtin_model.kimi_k25 import _patch_shard_model_for_kimi_vl

        _orig = _t.shard_model
        try:
            self.assertFalse(_km._shard_model_patched)
            _patch_shard_model_for_kimi_vl()
            self.assertTrue(_km._shard_model_patched)

            # Second call: no-op
            _patch_shard_model_for_kimi_vl()
            self.assertTrue(_km._shard_model_patched)
        finally:
            _t.shard_model = _orig
            _km._shard_model_patched = False

    # ------------------------------------------------------------------
    # _shard_lm_head_for_kimi_vl
    # ------------------------------------------------------------------

    def test_shard_lm_head_guard_no_tp(self):
        """Guard: returns early when lmhead_tp_group.world_size <= 1."""
        from unittest.mock import MagicMock
        from tensor_cast.transformers.builtin_model.kimi_k25 import _shard_lm_head_for_kimi_vl

        mock_model = MagicMock()
        mock_model.parallel_group_manager.lmhead_tp_group.world_size = 1

        self.assertIsNone(_shard_lm_head_for_kimi_vl(mock_model))

    # ------------------------------------------------------------------
    # _patch_resize_image_for_kimi_k25
    # ------------------------------------------------------------------

    def test_patch_resize_image_idempotent(self):
        """Second call is a no-op; global flag prevents re-patching."""
        import tensor_cast.core.input_generator as _ig
        import tensor_cast.transformers.builtin_model.kimi_k25 as _km
        from tensor_cast.transformers.builtin_model.kimi_k25 import _patch_resize_image_for_kimi_k25

        _orig = _ig.resize_image
        try:
            self.assertFalse(_km._resize_image_patched)
            _patch_resize_image_for_kimi_k25("moonshotai/Kimi-K2.5")
            self.assertTrue(_km._resize_image_patched)

            # Second call: no-op
            _patch_resize_image_for_kimi_k25("moonshotai/Kimi-K2.5")
            self.assertTrue(_km._resize_image_patched)
        finally:
            _ig.resize_image = _orig
            _km._resize_image_patched = False

    def test_patch_resize_image_non_kimi_fallback(self):
        """Non-Kimi model_id falls back to original resize_image."""
        import tensor_cast.core.input_generator as _ig
        import tensor_cast.transformers.builtin_model.kimi_k25 as _km
        from tensor_cast.transformers.builtin_model.kimi_k25 import _patch_resize_image_for_kimi_k25

        _orig = _ig.resize_image
        try:
            _patch_resize_image_for_kimi_k25("moonshotai/Kimi-K2.5")

            # Call with non-Kimi model_id should delegate to original
            result = _ig.resize_image(
                mid="Qwen/Qwen2-VL-7B",
                mtype="qwen2_vl",
                image_height=1080,
                image_width=1920,
                patch_size=14,
                merge_size=2,
                temporal_patch_size=1,
            )
            # Original resize logic returns a tuple of (height, width)
            self.assertIsInstance(result, tuple, "Original resize should return tuple")
            self.assertEqual(len(result), 2, "Result should have 2 elements")
            # The key point is that it uses the original logic, not Kimi's rounding
            # We just verify it returns valid dimensions
            self.assertGreater(result[0], 0, "Height should be positive")
            self.assertGreater(result[1], 0, "Width should be positive")
        finally:
            _ig.resize_image = _orig
            _km._resize_image_patched = False

    def test_patch_resize_image_kimi_rounding(self):
        """Kimi K2.5 rounds dimensions to factor multiples."""
        import tensor_cast.core.input_generator as _ig
        import tensor_cast.transformers.builtin_model.kimi_k25 as _km
        from tensor_cast.transformers.builtin_model.kimi_k25 import _patch_resize_image_for_kimi_k25

        _orig = _ig.resize_image
        try:
            _patch_resize_image_for_kimi_k25("moonshotai/Kimi-K2.5")

            # Kimi K2.5: factor = patch_size * merge_size = 14 * 2 = 28
            # 1080 -> ceil(1080/28)*28 = 39*28 = 1092
            # 1920 -> ceil(1920/28)*28 = 69*28 = 1932
            result = _ig.resize_image(
                mid="moonshotai/Kimi-K2.5",
                mtype="kimi_k25",
                image_height=1080,
                image_width=1920,
                patch_size=14,
                merge_size=2,
                temporal_patch_size=1,
            )
            self.assertEqual(result[0], 1092, "Height should round up to multiple of 28")
            self.assertEqual(result[1], 1932, "Width should round up to multiple of 28")
        finally:
            _ig.resize_image = _orig
            _km._resize_image_patched = False

    # ------------------------------------------------------------------
    # _shard_lm_head_for_kimi_vl — functional test with TP>1
    # ------------------------------------------------------------------

    def test_shard_lm_head_with_tp(self):
        """Replaces nested lm_head with ColumnParallelLinear when TP>1."""
        from unittest.mock import MagicMock, patch
        import torch.nn as nn
        from tensor_cast.transformers.builtin_model.kimi_k25 import _shard_lm_head_for_kimi_vl

        # Create a mock model with nested language_model.lm_head
        mock_model = MagicMock()
        mock_lm_head = nn.Linear(7168, 163840)
        mock_model._inner.named_modules.return_value = [
            ("language_model.lm_head", mock_lm_head),
        ]
        mock_model.parallel_group_manager.lmhead_tp_group.world_size = 8
        mock_model.parallel_group_manager.tp_group.world_size = 8

        # Mock ColumnParallelLinear to avoid actual sharding logic
        with patch("tensor_cast.layers.parallel_linear.ColumnParallelLinear") as mock_cpl:
            mock_cpl_instance = MagicMock()
            mock_cpl.return_value = mock_cpl_instance

            _shard_lm_head_for_kimi_vl(mock_model)

            # Verify ColumnParallelLinear was called with correct params
            mock_cpl.assert_called_once()
            args, kwargs = mock_cpl.call_args
            self.assertEqual(args[0], mock_lm_head)
            self.assertTrue(kwargs["gather_output"])

            # Verify _replace_module was called
            mock_model._replace_module.assert_called_once_with("language_model.lm_head", mock_cpl_instance)

    def test_shard_lm_head_mtp_suffix(self):
        """Also shards mtp.lm_head when present."""
        from unittest.mock import MagicMock, patch
        import torch.nn as nn
        from tensor_cast.transformers.builtin_model.kimi_k25 import _shard_lm_head_for_kimi_vl

        mock_model = MagicMock()
        mock_lm_head = nn.Linear(7168, 163840)
        mock_model._inner.named_modules.return_value = [
            ("mtp.lm_head", mock_lm_head),
        ]
        mock_model.parallel_group_manager.lmhead_tp_group.world_size = 8
        mock_model.parallel_group_manager.tp_group.world_size = 8

        with patch("tensor_cast.layers.parallel_linear.ColumnParallelLinear") as mock_cpl:
            mock_cpl_instance = MagicMock()
            mock_cpl.return_value = mock_cpl_instance

            _shard_lm_head_for_kimi_vl(mock_model)

            mock_cpl.assert_called_once()
            mock_model._replace_module.assert_called_once_with("mtp.lm_head", mock_cpl_instance)


if __name__ == "__main__":
    unittest.main()
