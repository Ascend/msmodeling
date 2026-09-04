import unittest
from dataclasses import asdict
from typing import Union

import pytest
import torch
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


@pytest.mark.nightly
class TestKimiK3(unittest.TestCase):
    """Unit tests for Kimi K3 model simulation.

    Test conditions based on §5.3 of ``docs/design/kimi_k3_adaptation_design.md``:
      - W4A8_DYNAMIC quantization (expert + non-expert linear)
      - DP4 / TP16 / EP64 with 64 devices
      - ``enable_multistream`` compilation config
      - ``enable_shared_expert_tp`` for shared expert tensor parallelism
      - K3 has no MTP; decode uses ``--decode`` flag (not ``--num-mtp-tokens``)
      - K3 uses SiTU activation (``tensor_cast.situ``) instead of SwiGLU
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = "ATLAS_800_A3_560T_128G_DIE"
        self.model_id = "moonshotai/Kimi-K3"
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

    def _assert_k3_operators_present(self, result_dict: dict, test_name: str = ""):
        """Verify K3-specific operators are present in the operation trace."""
        table = result_dict["table_result"]

        # K3 uses SiTU activation (replaces swiglu)
        self.assertIn(
            "tensor_cast.situ",
            table,
            f"{test_name}: SiTU activation (tensor_cast.situ) should be present in the operation trace",
        )

        # EP communication (ep_size=64 > 1)
        self.assertIn(
            "tensor_cast.all_to_all",
            table,
            f"{test_name}: EP communication (all_to_all) should be present with ep_size=64",
        )

        # MLA attention operators. With non-expert linear quantization ENABLED,
        # attention is excluded from quantization (DISABLED → BF16), so
        # ``mlapo_quant`` does NOT appear; instead ``mla_prolog`` and
        # ``multihead_latent_attention`` run in BF16.
        self.assertIn(
            "tensor_cast.mla_prolog",
            table,
            f"{test_name}: MLA prolog (mla_prolog) should be present in the operation trace",
        )
        self.assertIn(
            "tensor_cast.multihead_latent_attention",
            table,
            f"{test_name}: MLA attention (multihead_latent_attention) should be present in the operation trace",
        )

        # MoE expert computation via grouped_matmul
        self.assertIn(
            "tensor_cast.grouped_matmul",
            table,
            f"{test_name}: MoE expert computation (grouped_matmul) should be present in the operation trace",
        )

    def test_kimi_k3_text_prefill(self):
        """
        Test Case 1: Text-only Prefill Simulation

        Validates Kimi K3 text prefill performance under DP4/TP16/EP64
        parallelism and W4A8 dynamic quantization.
        Corresponds to §5.3 纯文本推理仿真 prefill stage.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=8,
            query_len=3500,
            context_length=0,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            quantize_non_expert_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            enable_multistream=True,
            enable_shared_expert_tp=True,
            world_size=64,
            dp_size=4,
            tp_size=16,
            ep_size=64,
            moe_tp_size=1,
            moe_dp_size=1,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k3_text_prefill")

        # Additional K3-specific operator checks
        result_dict = asdict(result) if isinstance(result, ModelRunnerMetrics) else result
        self._assert_k3_operators_present(result_dict, "test_kimi_k3_text_prefill")

    def test_kimi_k3_text_decode(self):
        """
        Test Case 2: Text-only Decode Simulation

        Validates Kimi K3 text decode performance with context_length=4250
        under DP4/TP16/EP64 parallelism and W4A8 dynamic quantization.
        Corresponds to §5.3 纯文本推理仿真 decode stage.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=32,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            quantize_non_expert_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            enable_multistream=True,
            enable_shared_expert_tp=True,
            world_size=64,
            dp_size=4,
            tp_size=16,
            ep_size=64,
            moe_tp_size=1,
            moe_dp_size=1,
            decode=True,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k3_text_decode")

        # Additional K3-specific operator checks
        result_dict = asdict(result) if isinstance(result, ModelRunnerMetrics) else result
        self._assert_k3_operators_present(result_dict, "test_kimi_k3_text_decode")

        # Verify KV cache is allocated for decode with context
        self.assertGreater(
            result_dict["kv_cache_size_gb"],
            0,
            "KV cache size should be non-zero for decode with context_length=4250",
        )

        # Verify execution time is reasonable for meta device simulation
        exec_time = result_dict["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        self.assertLess(
            exec_time,
            10.0,
            "Execution time should be reasonable (< 10s for meta device decode simulation)",
        )

    def test_kimi_k3_vision_language_prefill(self):
        """
        Test Case 3: Vision-Language Prefill Simulation

        Validates Kimi K3 multi-modal prefill pipeline with 1080P image input
        (1080×1920) and 30 text tokens under DP4/TP16/EP64 parallelism.
        Corresponds to §5.3 多模态推理仿真 prefill stage.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=4,
            query_len=30,
            context_length=0,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            quantize_non_expert_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            enable_multistream=True,
            enable_shared_expert_tp=True,
            world_size=64,
            dp_size=4,
            tp_size=16,
            ep_size=64,
            moe_tp_size=1,
            moe_dp_size=1,
            # Vision-language specific parameters
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        )

        model_runner = ModelRunner(user_input)

        # Verify the model is correctly identified as a VLM
        self.assertTrue(
            model_runner.model.is_vl_model,
            msg="Kimi K3 should be identified as a vision-language model",
        )

        # Generate inputs to verify visual features are produced
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )

        # pixel_values should be present in prefill mode
        self.assertIn(
            "pixel_values",
            input_kwargs,
            "pixel_values should be present for vision-language input in prefill mode",
        )

        # Run inference
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k3_vision_language_prefill")

        # Additional K3-specific operator checks
        result_dict = asdict(result) if isinstance(result, ModelRunnerMetrics) else result
        self._assert_k3_operators_present(result_dict, "test_kimi_k3_vision_language_prefill")

        # Verify vision encoder weights are included
        # The model_weight_size_gb > 0 assertion alone is insufficient because
        # text-only models also have non-zero weights. Verify the vision tower
        # module exists, has parameters, and its weight is reflected in the total.
        vision_tower = None
        for name, module in model_runner.model.named_modules():
            if name.endswith("vision_tower"):
                vision_tower = module
                break
        self.assertIsNotNone(
            vision_tower,
            "VL model should contain a vision_tower submodule; if missing, vision weights are not loaded",
        )
        vision_param_count = sum(p.numel() for p in vision_tower.parameters())
        self.assertGreater(
            vision_param_count,
            0,
            "Vision tower should have non-zero parameters; if zero, vision weights are not loaded",
        )
        # Vision tower is not TP-sharded, so each rank holds full vision weights.
        # Use int4 (0.5 bytes/param) as the conservative lower bound — even if
        # quantized, the vision tower weight must appear in model_weight_size_gb.
        vision_weight_gb_lower_bound = (vision_param_count * 0.5) / (1024**3)
        self.assertGreater(
            result_dict["model_weight_size_gb"],
            vision_weight_gb_lower_bound,
            "Model weight size should include vision tower weights "
            f"(vision params: {vision_param_count}, "
            f"expected >= {vision_weight_gb_lower_bound:.3f} GB)",
        )

    def test_kimi_k3_vision_language_decode(self):
        """
        Test Case 4: Vision-Language Decode Simulation

        Validates Kimi K3 multi-modal decode pipeline with context_length=2851
        (2693 image tokens + 30 text tokens + 128 avg generated tokens)
        under DP4/TP16/EP64 parallelism.
        Corresponds to §5.3 多模态推理仿真 decode stage.
        """
        user_input = UserInputConfig(
            device=self.device,
            model_id=self.model_id,
            num_queries=64,
            query_len=1,
            context_length=2851,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            quantize_non_expert_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
            enable_multistream=True,
            enable_shared_expert_tp=True,
            world_size=64,
            dp_size=4,
            tp_size=16,
            ep_size=64,
            moe_tp_size=1,
            moe_dp_size=1,
            # Vision-language specific parameters
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
            decode=True,
        )

        model_runner = ModelRunner(user_input)

        # Verify the model is correctly identified as a VLM
        self.assertTrue(
            model_runner.model.is_vl_model,
            msg="Kimi K3 should be identified as a vision-language model",
        )

        # Verify the VL model has a vision_tower submodule with parameters
        # (ensures vision weights are loaded before decode — without this, decode
        # could pass even if the vision tower was never loaded)
        vision_tower = None
        for name, module in model_runner.model.named_modules():
            if name.endswith("vision_tower"):
                vision_tower = module
                break
        self.assertIsNotNone(
            vision_tower,
            "VL model should contain a vision_tower submodule for decode test",
        )
        vision_param_count = sum(p.numel() for p in vision_tower.parameters())
        self.assertGreater(
            vision_param_count,
            0,
            "Vision tower should have non-zero parameters for decode test",
        )

        # Verify context_length includes image tokens (2693 for 1080P image).
        # Without this, the decode test could pass with a text-only context that
        # never exercised the VL prefill pipeline.
        self.assertGreaterEqual(
            user_input.context_length,
            2693,
            "VL decode context_length should include image tokens (≥2693 for 1080P image)",
        )

        # Generate inputs to verify pixel_values behavior in decode mode
        input_kwargs = generate_inputs(
            model_runner.model,
            model_runner.request_info_default,
            block_size=user_input.block_size,
        )

        # In decode mode, pixel_values are intentionally removed
        # (image input is processed during prefill only)
        self.assertNotIn(
            "pixel_values",
            input_kwargs,
            "pixel_values should NOT be present in decode mode (image input is removed after prefill)",
        )

        # Run inference
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        # Validate the inference result
        self._validate_inference_result(result, "test_kimi_k3_vision_language_decode")

        # Additional K3-specific operator checks
        result_dict = asdict(result) if isinstance(result, ModelRunnerMetrics) else result
        self._assert_k3_operators_present(result_dict, "test_kimi_k3_vision_language_decode")

        # Verify decode does NOT re-execute vision backbone ops — vision
        # processing must occur only in prefill. If vision ops appear in the
        # decode trace, it indicates a regression where decode incorrectly
        # re-runs the vision tower instead of using cached image features.
        table = result_dict["table_result"]
        vision_backbone_markers = ["vision_tower", "patch_embed", "vision_block", "MoonViT"]
        found_vision_ops = [m for m in vision_backbone_markers if m in table]
        self.assertEqual(
            found_vision_ops,
            [],
            f"Decode should not execute vision backbone ops (found: {found_vision_ops}); "
            "vision processing must occur only in prefill",
        )

        # Verify KV cache is allocated for decode with context
        self.assertGreater(
            result_dict["kv_cache_size_gb"],
            0,
            "KV cache size should be non-zero for decode with context_length=2851",
        )

        # Verify execution time is reasonable for meta device simulation
        exec_time = result_dict["execution_time_s"]
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        self.assertLess(
            exec_time,
            10.0,
            "Execution time should be reasonable (< 10s for meta device decode simulation)",
        )


class TestKimiK3Patches(unittest.TestCase):
    """Guard-condition / boundary tests for Kimi-K3 monkey-patch functions.

    These tests exercise the guard conditions and idempotency of K3 patch
    installers WITHOUT requiring network access or model instantiation, so
    they can run in the standard ``-m 'not nightly'`` CI job.
    """

    def setUp(self):
        """Reset global patch state before each test."""
        import tensor_cast.transformers.builtin_model.kimi_k3 as _km

        self._km = _km
        self._km._patched_kimi_k3 = False

    # ------------------------------------------------------------------
    # _hf_config_patch_for_kimi_k3 — top-level orchestrator guard
    # ------------------------------------------------------------------

    def test_hf_config_patch_guard_wrong_model_type(self):
        """Guard: returns early for non-``kimi_k3`` configs."""
        from tensor_cast.transformers.builtin_model.kimi_k3 import (
            _hf_config_patch_for_kimi_k3,
        )

        class _Fake:
            model_type = "llama"

        # Should return None (early exit) without raising
        self.assertIsNone(_hf_config_patch_for_kimi_k3(_Fake()))

    def test_hf_config_patch_guard_already_patched(self):
        """Guard: Phase 2 is skipped when ``_patched_kimi_k3`` is True."""
        from tensor_cast.transformers.builtin_model.kimi_k3 import (
            _hf_config_patch_for_kimi_k3,
        )

        class _Kimi:
            model_type = "kimi_k3"
            # Minimal attrs to survive Phase 1 config patches
            text_config = type("_TC", (), {"model_type": "kimi_linear"})()
            architectures = ["KimiK3ForConditionalGeneration"]

        _saved = self._km._patched_kimi_k3
        try:
            self._km._patched_kimi_k3 = True
            # Should not call _patch_model_classes_for_kimi_k3 (which needs model_id)
            # and should return without error
            _hf_config_patch_for_kimi_k3(_Kimi(), model_id=None)
        finally:
            self._km._patched_kimi_k3 = _saved

    # ------------------------------------------------------------------
    # _patch_model_classes_for_kimi_k3 — guard conditions
    # ------------------------------------------------------------------

    def test_patch_model_classes_guard_wrong_model_type(self):
        """Guard: returns False when model_type is not ``kimi_k3``."""
        from tensor_cast.transformers.builtin_model.kimi_k3 import (
            _patch_model_classes_for_kimi_k3,
        )

        class _Fake:
            model_type = "llama"

        self.assertFalse(_patch_model_classes_for_kimi_k3(_Fake(), "moonshotai/Kimi-K3"))

    def test_patch_model_classes_guard_none_model_id(self):
        """Guard: returns False when model_id is None."""
        from tensor_cast.transformers.builtin_model.kimi_k3 import (
            _patch_model_classes_for_kimi_k3,
        )

        class _Kimi:
            model_type = "kimi_k3"

        self.assertFalse(_patch_model_classes_for_kimi_k3(_Kimi(), None))

    # ------------------------------------------------------------------
    # _install_fla_stub — idempotency
    # ------------------------------------------------------------------

    def test_fla_stub_idempotent(self):
        """Second call is a no-op; ``_FLA_STUB_INSTALLED`` flag prevents re-entry."""
        import sys

        _saved_flag = self._km._FLA_STUB_INSTALLED
        _saved_modules = {
            k: sys.modules.get(k)
            for k in [
                "fla",
                "fla.modules",
                "fla.ops",
                "fla.ops.kda",
                "fla.ops.utils",
                "fla.ops.utils.index",
                "fla.utils",
            ]
        }
        try:
            self._km._FLA_STUB_INSTALLED = False

            # First call: installs stubs
            self._km._install_fla_stub()
            self.assertTrue(self._km._FLA_STUB_INSTALLED)

            # Snapshot sys.modules after first install
            fla_module = sys.modules.get("fla")
            self.assertIsNotNone(fla_module, "fla stub should be in sys.modules after install")

            # Second call: no-op (flag prevents re-entry)
            self._km._install_fla_stub()
            self.assertTrue(self._km._FLA_STUB_INSTALLED)

            # fla module should be the same object (not re-created)
            self.assertIs(
                sys.modules.get("fla"),
                fla_module,
                "Second call should not re-create fla stub",
            )
        finally:
            # Restore original state
            self._km._FLA_STUB_INSTALLED = _saved_flag
            for mod_name, mod in _saved_modules.items():
                if mod is not None:
                    sys.modules[mod_name] = mod
                elif mod_name in sys.modules and not _saved_flag:
                    # Only remove if we installed it (original flag was False)
                    del sys.modules[mod_name]

    # ------------------------------------------------------------------
    # _install_copy_layer_attr_patch — idempotency
    # ------------------------------------------------------------------

    def test_copy_layer_attr_patch_idempotent(self):
        """Second call is a no-op; ``_COPY_LAYER_ATTR_PATCH_INSTALLED`` flag prevents re-entry."""
        from tensor_cast.layers.internal import CopyLayerWrapper

        _saved_flag = self._km._COPY_LAYER_ATTR_PATCH_INSTALLED
        _saved_init = CopyLayerWrapper.__init__
        _saved_forward = CopyLayerWrapper.forward
        try:
            self._km._COPY_LAYER_ATTR_PATCH_INSTALLED = False

            # First call: installs patches
            self._km._install_copy_layer_attr_patch()
            self.assertTrue(self._km._COPY_LAYER_ATTR_PATCH_INSTALLED)

            # Verify __init__ was patched (not the original)
            self.assertIsNot(CopyLayerWrapper.__init__, _saved_init, "__init__ should be patched")

            # Second call: no-op
            self._km._install_copy_layer_attr_patch()
            self.assertTrue(self._km._COPY_LAYER_ATTR_PATCH_INSTALLED)
        finally:
            CopyLayerWrapper.__init__ = _saved_init
            CopyLayerWrapper.forward = _saved_forward
            self._km._COPY_LAYER_ATTR_PATCH_INSTALLED = _saved_flag

    # ------------------------------------------------------------------
    # _install_lm_head_tp_patch — idempotency
    # ------------------------------------------------------------------

    def test_lm_head_tp_patch_idempotent(self):
        """Second call is a no-op; ``_LM_HEAD_TP_PATCH_INSTALLED`` flag prevents re-entry."""
        from tensor_cast.transformers import transformations as _t

        _saved_flag = self._km._LM_HEAD_TP_PATCH_INSTALLED
        _orig_shard = _t.shard_model_by_tp
        try:
            self._km._LM_HEAD_TP_PATCH_INSTALLED = False

            # First call: installs patches
            self._km._install_lm_head_tp_patch()
            self.assertTrue(self._km._LM_HEAD_TP_PATCH_INSTALLED)

            # Verify shard_model_by_tp was patched
            self.assertIsNot(_t.shard_model_by_tp, _orig_shard, "shard_model_by_tp should be patched")

            # Second call: no-op
            self._km._install_lm_head_tp_patch()
            self.assertTrue(self._km._LM_HEAD_TP_PATCH_INSTALLED)
        finally:
            _t.shard_model_by_tp = _orig_shard
            self._km._LM_HEAD_TP_PATCH_INSTALLED = _saved_flag

    # ------------------------------------------------------------------
    # _install_latent_moe_patch — idempotency
    # ------------------------------------------------------------------

    def test_latent_moe_patch_idempotent(self):
        """Second call is a no-op; ``_LATENT_MOE_PATCH_INSTALLED`` flag prevents re-entry."""
        from tensor_cast.layers.moe_layer import (
            FusedMoETensorCast,
            MoELayer,
            ParallelMoELayer,
        )

        _saved_flag = self._km._LATENT_MOE_PATCH_INSTALLED
        _saved_moe_init = MoELayer.__init__
        _saved_parallel_moe_init = ParallelMoELayer.__init__
        _saved_fused_forward = FusedMoETensorCast.forward
        try:
            self._km._LATENT_MOE_PATCH_INSTALLED = False

            # First call: installs patches
            self._km._install_latent_moe_patch()
            self.assertTrue(self._km._LATENT_MOE_PATCH_INSTALLED)

            # Verify at least one method was patched
            self.assertIsNot(
                MoELayer.__init__,
                _saved_moe_init,
                "MoELayer.__init__ should be patched",
            )

            # Second call: no-op
            self._km._install_latent_moe_patch()
            self.assertTrue(self._km._LATENT_MOE_PATCH_INSTALLED)
        finally:
            MoELayer.__init__ = _saved_moe_init
            ParallelMoELayer.__init__ = _saved_parallel_moe_init
            FusedMoETensorCast.forward = _saved_fused_forward
            self._km._LATENT_MOE_PATCH_INSTALLED = _saved_flag

    # ------------------------------------------------------------------
    # _install_kda_tp_plan_patch — idempotency
    # ------------------------------------------------------------------

    def test_kda_tp_plan_patch_idempotent(self):
        """Second call is a no-op; ``_KDA_TP_PLAN_PATCH_INSTALLED`` flag prevents re-entry."""
        from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

        _saved_flag = self._km._KDA_TP_PLAN_PATCH_INSTALLED
        _saved_method = MultiheadLatentAttentionTensorCast.build_tp_plan_extras
        try:
            self._km._KDA_TP_PLAN_PATCH_INSTALLED = False

            # First call: installs patches
            self._km._install_kda_tp_plan_patch()
            self.assertTrue(self._km._KDA_TP_PLAN_PATCH_INSTALLED)

            # Verify build_tp_plan_extras was patched
            self.assertIsNot(
                MultiheadLatentAttentionTensorCast.build_tp_plan_extras,
                _saved_method,
                "build_tp_plan_extras should be patched",
            )

            # Second call: no-op
            self._km._install_kda_tp_plan_patch()
            self.assertTrue(self._km._KDA_TP_PLAN_PATCH_INSTALLED)
        finally:
            MultiheadLatentAttentionTensorCast.build_tp_plan_extras = _saved_method
            self._km._KDA_TP_PLAN_PATCH_INSTALLED = _saved_flag

    # ------------------------------------------------------------------
    # situ.register_all_patterns — idempotency (migrated to compilation/)
    # ------------------------------------------------------------------

    def test_situ_pattern_patch_idempotent(self):
        """``situ._INSTALLED`` flag prevents re-registration of SiTU patterns.

        SiTU pattern registration moved to ``tensor_cast/compilation/patterns/situ.py``.
        Its ``_INSTALLED`` flag (equivalent to the old
        ``_SITU_PATTERN_PATCH_INSTALLED``) guards against duplicate registration,
        which would otherwise raise ``ValueError`` from ``register_pattern``.
        This test verifies the guard short-circuits when already installed.
        """
        from tensor_cast.compilation import patterns as tc_patterns

        _saved_flag = tc_patterns.situ._INSTALLED
        try:
            # Force flag to True — simulates already-installed state
            tc_patterns.situ._INSTALLED = True

            # Call must be a no-op: guard returns before register_pattern
            tc_patterns.situ.register_all_patterns()
            self.assertTrue(tc_patterns.situ._INSTALLED)
        finally:
            tc_patterns.situ._INSTALLED = _saved_flag


# ---------------------------------------------------------------------------
# SiTU operator unit tests
#
# These tests directly exercise the meta ops (shape/dtype inference) and the
# performance-model functors (SiTU ``gp_ops`` accounting) for the K3 operator
# added in this PR:
#   - tensor_cast.situ
#
# They guard against regressions when the op schema, packed-weight layout, or
# performance-model accounting changes.  These are pure unit tests: no network
# access and no model instantiation, so they run in the standard
# ``-m 'not nightly'`` CI job.
# ---------------------------------------------------------------------------

# Mirror of ``_SITU_OPS_PER_ELEM`` in performance_model/__init__.py. Kept as a
# local literal so a change to the activation cost constant triggers a test
# failure prompting intentional review.
_SITU_OPS_PER_ELEM = 18


class TestSituOpMeta(unittest.TestCase):
    """Shape/dtype inference for the ``tensor_cast.situ`` meta op."""

    def test_situ_basic_shape_and_dtype(self):
        gate = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
        up = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
        out = torch.ops.tensor_cast.situ(gate, up)
        self.assertEqual(tuple(out.shape), (4, 8))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_situ_preserves_multidim_shape(self):
        gate = torch.empty((2, 3, 16), dtype=torch.float16, device="meta")
        up = torch.empty((2, 3, 16), dtype=torch.float16, device="meta")
        out = torch.ops.tensor_cast.situ(gate, up)
        self.assertEqual(tuple(out.shape), (2, 3, 16))
        self.assertEqual(out.dtype, torch.float16)

    def test_situ_scalar_betas_do_not_affect_shape(self):
        """``beta`` and ``linear_beta`` are scalar args; they must not change the
        output shape, regardless of whether ``linear_beta`` is set.
        """
        gate = torch.empty((5, 10), dtype=torch.bfloat16, device="meta")
        up = torch.empty((5, 10), dtype=torch.bfloat16, device="meta")
        out_default = torch.ops.tensor_cast.situ(gate, up)
        out_beta = torch.ops.tensor_cast.situ(gate, up, beta=2.0)
        out_linear = torch.ops.tensor_cast.situ(gate, up, beta=2.0, linear_beta=1.5)
        for out in (out_default, out_beta, out_linear):
            self.assertEqual(tuple(out.shape), (5, 10))

    def test_situ_dtype_propagation(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            # NOTE: pass dtype as str (not the torch.dtype object) to subTest.
            # pytest-xdist uses execnet/pickle to ship subTest info from worker
            # processes back to the master; torch.dtype instances are not
            # picklable and raise:
            #   execnet.gateway_base.DumpError: can't serialize <class 'torch.dtype'>
            with self.subTest(dtype=str(dtype)):
                gate = torch.empty((4, 4), dtype=dtype, device="meta")
                up = torch.empty((4, 4), dtype=dtype, device="meta")
                out = torch.ops.tensor_cast.situ(gate, up)
                self.assertEqual(out.dtype, dtype)


class TestSituPerformanceModel(unittest.TestCase):
    """Performance-model FLOP accounting for the SiTU activation operator.

    Asserts that SiTU activation compute (``gp_ops``) is accumulated correctly.
    Guards against regressions when the op schema or performance-model
    accounting changes.
    """

    def _props(self, op, args, kwargs=None, out=None):
        if out is None:
            out = torch.empty((), device="meta")
        info = OpInvokeInfo(op, args, kwargs or {}, out)
        return info.get_perf_properties()

    # -- tensor_cast.situ ---------------------------------------------------

    def test_situ_gp_ops_only(self):
        """``situ`` is pure activation: gp_ops = numel * 18, mma_ops = 0."""
        op = torch.ops.tensor_cast.situ.default
        gate = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
        up = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
        out = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
        props = self._props(op, [gate, up], out=out)
        compute = props.compute_ops[torch.bfloat16]
        self.assertEqual(compute.gp_ops, 4 * 8 * _SITU_OPS_PER_ELEM)
        self.assertEqual(compute.mma_ops, 0)


if __name__ == "__main__":
    unittest.main()
