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
from dataclasses import asdict
from types import SimpleNamespace
from typing import Union
from unittest import mock

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

    Test conditions based on §5.3 of Wiki design ``kimi_k3_adaptation_design.md``:
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
        self.assertGreater(
            result["model_activation_size_gb"],
            0,
            f"{test_name}: Model activation size should be positive (Issue #463)",
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

    def _assert_k3_mla_boundary(self, result_dict: dict, test_name: str = ""):
        """Verify K3 retains its prolog instead of the generic projection ops."""
        table = result_dict["table_result"]
        self.assertIn(
            "tensor_cast.mla_merge_phase_outputs",
            table,
            f"{test_name}: MLA phase outputs should be merged",
        )
        self.assertIn("tensor_cast.mla_prolog", table, f"{test_name}: K3 prolog should remain fused")
        for operator_name in (
            "tensor_cast.mla_kv_projection.default",
            "tensor_cast.mla_q_absorb_projection.default",
        ):
            self.assertNotIn(operator_name, table, f"{test_name}: K3 should not use the generic MLA projection op")

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
        self._assert_k3_mla_boundary(result_dict, "test_kimi_k3_text_prefill")

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
        self._assert_k3_mla_boundary(result_dict, "test_kimi_k3_text_decode")

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
        import sys

        import tensor_cast.transformers.builtin_model.kimi_k3 as _km

        self._km = _km
        self._saved_fla_stub_installed = self._km._FLA_STUB_INSTALLED
        self._fla_module_names = (
            "fla",
            "fla.modules",
            "fla.ops",
            "fla.ops.kda",
            "fla.ops.utils",
            "fla.ops.utils.index",
            "fla.utils",
        )
        self._missing_fla_module = object()
        self._saved_fla_modules = {
            name: sys.modules.get(name, self._missing_fla_module) for name in self._fla_module_names
        }
        self._km._FLA_STUB_INSTALLED = False
        for name in self._fla_module_names:
            sys.modules.pop(name, None)

    def tearDown(self):
        """Restore global state after each test to prevent test infection.

        K3 patches install global monkey-patches (fla stubs in sys.modules,
        quantize_linear replacement, create_causal_mask bridge, etc.) that
        persist across tests and can corrupt non-K3 model tests (Issue 402:
        Qwen3-MoE UT failures from modules_to_not_convert pollution).
        """
        self._km._restore_k3_global_state()

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

    def test_hf_config_patch_without_model_id_applies_config_patches_only(self):
        """The orchestrator always delegates to the class patcher, which
        returns early without ``model_id`` (Issue #596 removed the
        process-level boolean short-circuit).
        """
        from tensor_cast.transformers.builtin_model.kimi_k3 import (
            _hf_config_patch_for_kimi_k3,
        )

        class _Kimi:
            model_type = "kimi_k3"
            # Minimal attrs to survive Phase 1 config patches
            text_config = type("_TC", (), {"model_type": "kimi_linear"})()
            architectures = ["KimiK3ForConditionalGeneration"]

        # Must complete without error: the class patcher returns False on
        # model_id=None and only config-level patches run.
        self.assertIsNone(_hf_config_patch_for_kimi_k3(_Kimi(), model_id=None))

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

    def test_patch_model_classes_reapplied_after_remote_module_reexecution(self):
        """Issue #596: K3 class patches are re-applied when transformers
        re-executes the remote modeling files (fresh module objects under the
        same sys.modules keys); already-patched module objects are skipped.
        """
        import sys
        import types

        from tensor_cast.core.model_runner import ModelRunner
        from tensor_cast.transformers.builtin_model.kimi_k3 import _patch_model_classes_for_kimi_k3
        from tensor_cast.transformers.model import ModelWrapper

        package = "transformers_modules.test_kimi_k3_hash"

        def _build_fake_package():
            def _noop(self, *args, **kwargs):
                return None

            class _FakeVL:
                def forward(self, *args, **kwargs):
                    return None

                def _merge_input_ids_with_image_features(self, *args, **kwargs):
                    return None

            _FakeVL.__module__ = f"{package}.modeling_kimi_k3"

            k3_module = types.ModuleType(f"{package}.modeling_kimi_k3")
            linear_module = types.ModuleType(f"{package}.modeling_kimi_linear")

            def _stub(name, module):
                # Minimal stand-in exposing every method the K3 patch blocks
                # may wrap, so a complete patch pass finishes without any
                # block swallowing an exception and the module marker (which
                # requires a clean pass) gets stamped.
                cls = type(
                    name,
                    (),
                    {
                        "forward": _noop,
                        "moe_infer": _noop,
                        "_update_linear_attn_mask": _noop,
                        "__module__": module.__name__,
                    },
                )
                setattr(module, name, cls)
                return cls

            k3_module.KimiK3ForConditionalGeneration = _FakeVL
            k3_module.VL_VISION_ATTENTION_FUNCTIONS = {}
            _stub("MoonViT3dEncoder", k3_module)
            _stub("MoonVision3dPatchEmbed", k3_module)
            for name in (
                "KimiDeltaAttention",
                "KimiLinearModel",
                "KimiSparseMoeBlock",
                "KimiMoEGate",
                "KimiDecoderLayer",
                "SituAndMul",
                "KimiMLP",
                "KimiBlockSparseMLP",
                "KimiLinearForCausalLM",
            ):
                _stub(name, linear_module)
            linear_module._apply_attn_res = lambda *args, **kwargs: None
            return k3_module, linear_module, _FakeVL

        def _install(k3_module, linear_module):
            names = (k3_module.__name__, linear_module.__name__)
            saved = {name: sys.modules.get(name) for name in names}
            sys.modules[names[0]] = k3_module
            sys.modules[names[1]] = linear_module

            def _fake_get_class(class_ref, *args, **kwargs):
                module_name, _, class_name = class_ref.partition(".")
                module = sys.modules[f"{package}.{module_name}"]
                try:
                    return getattr(module, class_name)
                except AttributeError as error:
                    raise RuntimeError(f"fake class {class_ref} unavailable") from error

            def _restore():
                for name, original in saved.items():
                    if original is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = original

            return _fake_get_class, _restore

        def _restore_flag(cls, name, saved_value):
            if saved_value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, saved_value)

        class _KimiConfig:
            model_type = "kimi_k3"

        # Neutralize the process-level patches (ModelWrapper / ModelRunner
        # accounting) so this test only exercises the remote-module guard.
        saved_mw_flag = getattr(ModelWrapper, "_patched_for_k3_mtp", None)
        saved_runner_flag = getattr(ModelRunner, "_tensor_cast_k3_constant_folder_guard", None)
        ModelWrapper._patched_for_k3_mtp = True
        ModelRunner._tensor_cast_k3_constant_folder_guard = True

        try:
            k3_module, linear_module, vl_cls = _build_fake_package()
            fake_get_class, restore = _install(k3_module, linear_module)
            try:
                with mock.patch.object(self._km, "_get_k3_class_from_source", side_effect=fake_get_class):
                    # First pass patches and stamps both remote module objects.
                    self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                    self.assertTrue(getattr(k3_module, self._km._K3_MODULE_PATCH_MARKER, False))
                    self.assertTrue(getattr(linear_module, self._km._K3_MODULE_PATCH_MARKER, False))
                    self.assertTrue(hasattr(vl_cls, "_original_vl_forward"))

                    # Second pass over the SAME module objects is a no-op.
                    self.assertFalse(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
            finally:
                restore()

            # Simulate transformers re-executing the remote files: brand-new
            # module objects (pristine classes) under the same sys.modules keys.
            k3_module2, linear_module2, vl_cls2 = _build_fake_package()
            fake_get_class2, restore2 = _install(k3_module2, linear_module2)
            try:
                with mock.patch.object(self._km, "_get_k3_class_from_source", side_effect=fake_get_class2):
                    self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                    self.assertTrue(getattr(k3_module2, self._km._K3_MODULE_PATCH_MARKER, False))
                    self.assertTrue(getattr(linear_module2, self._km._K3_MODULE_PATCH_MARKER, False))
                    self.assertTrue(hasattr(vl_cls2, "_original_vl_forward"))
            finally:
                restore2()
        finally:
            _restore_flag(ModelWrapper, "_patched_for_k3_mtp", saved_mw_flag)
            _restore_flag(ModelRunner, "_tensor_cast_k3_constant_folder_guard", saved_runner_flag)

    def test_patch_model_classes_reapplied_after_in_place_remote_reexecution(self):
        """transformers re-executes a remote file *in place* when its content
        hash changes: the module object (and its module-level marker) survives,
        but every class object is replaced by an unpatched copy.

        The guard must detect the replaced class and re-apply the patches, or
        the model is built from a class whose ``forward`` rejects TensorCast
        kwargs (``unexpected keyword argument 'attention_meta'``).
        """
        import sys
        import types

        from tensor_cast.core.model_runner import ModelRunner
        from tensor_cast.transformers.builtin_model.kimi_k3 import _patch_model_classes_for_kimi_k3
        from tensor_cast.transformers.model import ModelWrapper

        package = "transformers_modules.test_kimi_k3_inplace"

        def _noop(self, *args, **kwargs):
            return None

        def _define(name, module_name, base=None):
            cls = type(
                name,
                (base or object,),
                {
                    "forward": _noop,
                    "moe_infer": _noop,
                    "_update_linear_attn_mask": _noop,
                    "__module__": module_name,
                },
            )
            return cls

        k3_name = f"{package}.modeling_kimi_k3"
        linear_name = f"{package}.modeling_kimi_linear"
        k3_module = types.ModuleType(k3_name)
        linear_module = types.ModuleType(linear_name)

        def _populate(k3_module, linear_module):
            vl_cls = _define("KimiK3ForConditionalGeneration", k3_module.__name__)
            vl_cls._merge_input_ids_with_image_features = _noop
            k3_module.KimiK3ForConditionalGeneration = vl_cls
            k3_module.VL_VISION_ATTENTION_FUNCTIONS = {}
            k3_module.MoonViT3dEncoder = _define("MoonViT3dEncoder", k3_module.__name__)
            k3_module.MoonVision3dPatchEmbed = _define("MoonVision3dPatchEmbed", k3_module.__name__)
            for name in (
                "KimiDeltaAttention",
                "KimiLinearModel",
                "KimiSparseMoeBlock",
                "KimiMoEGate",
                "KimiDecoderLayer",
                "SituAndMul",
                "KimiMLP",
                "KimiBlockSparseMLP",
                "KimiLinearForCausalLM",
            ):
                setattr(linear_module, name, _define(name, linear_module.__name__))
            linear_module._apply_attn_res = lambda *args, **kwargs: None
            return vl_cls

        def _fake_get_class(class_ref, *args, **kwargs):
            module_name, _, class_name = class_ref.partition(".")
            return getattr(sys.modules[f"{package}.{module_name}"], class_name)

        class _KimiConfig:
            model_type = "kimi_k3"

        saved_mw_flag = getattr(ModelWrapper, "_patched_for_k3_mtp", None)
        saved_runner_flag = getattr(ModelRunner, "_tensor_cast_k3_kda_accounting", None)
        ModelWrapper._patched_for_k3_mtp = True
        ModelRunner._tensor_cast_k3_kda_accounting = True
        saved_modules = {name: sys.modules.get(name) for name in (k3_name, linear_name)}
        sys.modules[k3_name] = k3_module
        sys.modules[linear_name] = linear_module
        try:
            first_vl = _populate(k3_module, linear_module)
            with mock.patch.object(self._km, "_get_k3_class_from_source", side_effect=_fake_get_class):
                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                self.assertTrue(hasattr(first_vl, "_original_vl_forward"))

                # Same module objects, fresh classes: transformers re-executed
                # the file in place and kept the module-level marker.
                second_vl = _populate(k3_module, linear_module)
                self.assertTrue(getattr(k3_module, self._km._K3_MODULE_PATCH_MARKER, False))
                self.assertFalse(hasattr(second_vl, "_original_vl_forward"))

                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                self.assertTrue(hasattr(second_vl, "_original_vl_forward"))

                # A third pass over the now-patched generation is a no-op.
                self.assertFalse(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
        finally:
            for name, original in saved_modules.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            ModelWrapper._patched_for_k3_mtp = saved_mw_flag
            ModelRunner._tensor_cast_k3_kda_accounting = saved_runner_flag

    def test_patch_model_classes_reapplied_when_only_linear_module_reexecuted(self):
        """transformers re-executes each remote file on its own content hash.

        Replacing only ``modeling_kimi_linear`` (attention / MoE / MLP) keeps
        ``modeling_kimi_k3`` and every marker on the wrapper class intact, so a
        wrapper-only guard skips the repair and the live modules keep running
        unpatched code. The guard must inspect the class generation of *both*
        remote modules.
        """
        import sys
        import types

        from tensor_cast.core.model_runner import ModelRunner
        from tensor_cast.transformers.builtin_model.kimi_k3 import _patch_model_classes_for_kimi_k3
        from tensor_cast.transformers.model import ModelWrapper

        package = "transformers_modules.test_kimi_k3_partial"

        def _noop(self, *args, **kwargs):
            return None

        def _define(name, module_name):
            return type(
                name,
                (object,),
                {
                    "forward": _noop,
                    "moe_infer": _noop,
                    "_update_linear_attn_mask": _noop,
                    "__module__": module_name,
                },
            )

        k3_name = f"{package}.modeling_kimi_k3"
        linear_name = f"{package}.modeling_kimi_linear"
        k3_module = types.ModuleType(k3_name)
        linear_module = types.ModuleType(linear_name)

        def _populate_k3():
            vl_cls = _define("KimiK3ForConditionalGeneration", k3_name)
            vl_cls._merge_input_ids_with_image_features = _noop
            k3_module.KimiK3ForConditionalGeneration = vl_cls
            k3_module.VL_VISION_ATTENTION_FUNCTIONS = {}
            k3_module.MoonViT3dEncoder = _define("MoonViT3dEncoder", k3_name)
            k3_module.MoonVision3dPatchEmbed = _define("MoonVision3dPatchEmbed", k3_name)
            return vl_cls

        def _populate_linear():
            for name in (
                "KimiDeltaAttention",
                "KimiLinearModel",
                "KimiSparseMoeBlock",
                "KimiMoEGate",
                "KimiDecoderLayer",
                "SituAndMul",
                "KimiMLP",
                "KimiBlockSparseMLP",
                "KimiLinearForCausalLM",
            ):
                setattr(linear_module, name, _define(name, linear_name))
            linear_module._apply_attn_res = lambda *args, **kwargs: None

        def _fake_get_class(class_ref, *args, **kwargs):
            module_name, _, class_name = class_ref.partition(".")
            return getattr(sys.modules[f"{package}.{module_name}"], class_name)

        class _KimiConfig:
            model_type = "kimi_k3"

        saved_mw_flag = getattr(ModelWrapper, "_patched_for_k3_mtp", None)
        saved_runner_flag = getattr(ModelRunner, "_tensor_cast_k3_kda_accounting", None)
        ModelWrapper._patched_for_k3_mtp = True
        ModelRunner._tensor_cast_k3_kda_accounting = True
        saved_modules = {name: sys.modules.get(name) for name in (k3_name, linear_name)}
        sys.modules[k3_name] = k3_module
        sys.modules[linear_name] = linear_module
        try:
            _populate_k3()
            _populate_linear()
            with mock.patch.object(self._km, "_get_k3_class_from_source", side_effect=_fake_get_class):
                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                wrapper = getattr(k3_module, "KimiK3ForConditionalGeneration")
                self.assertTrue(self._km._k3_class_patches_intact(wrapper))

                # Only the linear file is re-executed in place: the k3 module
                # object and its wrapper class keep their markers while every
                # attention / MoE / MLP class is replaced by a pristine copy.
                _populate_linear()
                fresh_attention = getattr(linear_module, "KimiDeltaAttention")
                self.assertFalse(self._km._k3_class_patches_intact(fresh_attention))
                self.assertTrue(self._km._k3_class_patches_intact(wrapper))

                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                self.assertTrue(self._km._k3_class_patches_intact(fresh_attention))
                self.assertTrue(self._km._k3_class_patches_intact(wrapper))
                # Nothing left to repair once both generations are marked.
                self.assertFalse(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
        finally:
            for name, original in saved_modules.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            ModelWrapper._patched_for_k3_mtp = saved_mw_flag
            ModelRunner._tensor_cast_k3_kda_accounting = saved_runner_flag

    def test_ensure_k3_class_patches_targets_instantiated_classes(self):
        """Post-load ensure re-patches the exact classes the model uses.

        When the built model holds a stale class generation, resolving classes
        by name would patch the newer (unused) generation and leave the live
        model unpatched. The ensure pass must pin the instantiated classes.
        """
        import sys
        import types

        from tensor_cast.transformers.builtin_model import kimi_k3 as _km

        package = "transformers_modules.test_kimi_k3_ensure"
        module_name = f"{package}.modeling_kimi_k3"
        module = types.ModuleType(module_name)

        class KimiK3ForConditionalGeneration:
            pass

        KimiK3ForConditionalGeneration.__module__ = module_name
        module.KimiK3ForConditionalGeneration = KimiK3ForConditionalGeneration
        sys.modules[module_name] = module
        try:

            class _Config:
                model_type = "kimi_k3"
                name_or_path = "moonshotai/Kimi-K3"

            model = SimpleNamespace(
                _inner=object.__new__(KimiK3ForConditionalGeneration),
                hf_config=_Config(),
            )
            captured = {}

            def _fake_patch(config, model_id):
                captured["overrides"] = dict(_km._k3_class_overrides())
                captured["model_id"] = model_id
                return True

            with mock.patch.object(_km, "_patch_model_classes_for_kimi_k3", side_effect=_fake_patch):
                self.assertTrue(_km._ensure_k3_class_patches(model))

            self.assertIs(
                captured["overrides"][f"modeling_kimi_k3.{KimiK3ForConditionalGeneration.__name__}"],
                KimiK3ForConditionalGeneration,
            )
            self.assertEqual(captured["model_id"], "moonshotai/Kimi-K3")
            self.assertEqual(_km._k3_class_overrides(), {})

            # Already-marked classes are left alone.
            setattr(KimiK3ForConditionalGeneration, _km._K3_CLASS_PATCH_MARKER, True)
            with mock.patch.object(_km, "_patch_model_classes_for_kimi_k3", side_effect=_fake_patch):
                self.assertFalse(_km._ensure_k3_class_patches(model))
        finally:
            sys.modules.pop(module_name, None)
            if hasattr(KimiK3ForConditionalGeneration, _km._K3_CLASS_PATCH_MARKER):
                delattr(KimiK3ForConditionalGeneration, _km._K3_CLASS_PATCH_MARKER)

    def test_ensure_k3_class_patches_probes_every_live_class(self):
        """A marked wrapper must not short-circuit the repair.

        transformers re-executes each remote file independently, so the wrapper
        can still carry the marker while the attention classes the instance
        holds were replaced. Probing only the wrapper skips the repair exactly
        then and the model keeps running unpatched code.
        """
        import sys
        import types

        import torch.nn as nn

        from tensor_cast.transformers.builtin_model import kimi_k3 as _km

        package = "transformers_modules.test_kimi_k3_partial_ensure"
        k3_name = f"{package}.modeling_kimi_k3"
        linear_name = f"{package}.modeling_kimi_linear"

        class KimiDeltaAttention(nn.Module):
            def __init__(self):
                nn.Module.__init__(self)

        KimiDeltaAttention.__module__ = linear_name

        class KimiK3ForConditionalGeneration(nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                self.self_attn = KimiDeltaAttention()

        KimiK3ForConditionalGeneration.__module__ = k3_name

        class _Config:
            model_type = "kimi_k3"
            name_or_path = "moonshotai/Kimi-K3"

        saved = sys.modules.get(k3_name)
        sys.modules[k3_name] = types.ModuleType(k3_name)
        try:
            model = SimpleNamespace(_inner=KimiK3ForConditionalGeneration(), hf_config=_Config())
            captured = {}

            def _fake_patch(config, model_id):
                captured["overrides"] = dict(_km._k3_class_overrides())
                return True

            setattr(KimiK3ForConditionalGeneration, _km._K3_CLASS_PATCH_MARKER, True)
            with mock.patch.object(_km, "_patch_model_classes_for_kimi_k3", side_effect=_fake_patch):
                self.assertTrue(_km._ensure_k3_class_patches(model))

            overrides = captured["overrides"]
            self.assertIs(overrides[f"modeling_kimi_linear.{KimiDeltaAttention.__name__}"], KimiDeltaAttention)
            self.assertIs(
                overrides[f"modeling_kimi_k3.{KimiK3ForConditionalGeneration.__name__}"], KimiK3ForConditionalGeneration
            )
            self.assertEqual(_km._k3_class_overrides(), {})
        finally:
            if saved is None:
                sys.modules.pop(k3_name, None)
            else:
                sys.modules[k3_name] = saved
            if hasattr(KimiK3ForConditionalGeneration, _km._K3_CLASS_PATCH_MARKER):
                delattr(KimiK3ForConditionalGeneration, _km._K3_CLASS_PATCH_MARKER)

    def test_class_overrides_do_not_leak_across_threads(self):
        """Parallel captures must not share the in-flight class pins.

        A module-level dict made ``update``/``clear`` visible to every capture
        thread in the process, so one thread's ``clear()`` could yank the pins
        out from under another's patch pass. The context-local storage keeps
        each pass isolated.
        """
        import threading

        from tensor_cast.transformers.builtin_model import kimi_k3 as _km

        pinned = type("Pinned", (), {})
        token = _km._K3_CLASS_OVERRIDES.set({"modeling_kimi_k3.Pinned": pinned})
        try:
            self.assertIs(_km._k3_class_overrides()["modeling_kimi_k3.Pinned"], pinned)
            seen = {}

            def _worker():
                seen["overrides"] = dict(_km._k3_class_overrides())

            thread = threading.Thread(target=_worker)
            thread.start()
            thread.join()
            self.assertEqual(seen["overrides"], {})
        finally:
            _km._K3_CLASS_OVERRIDES.reset(token)

    def test_ensure_k3_class_patches_skips_the_model_walk_until_the_generation_changes(self):
        """The guard runs before every inference; only a new generation re-walks.

        ``named_modules()`` over a 61-layer K3 tree costs ~1.6 ms and the guard
        runs once per ``run_inference``.  Fingerprinting the two remote modules
        replaces that walk with two ``vars()`` scans, and still detaches as soon
        as either file is re-executed in place.
        """
        import sys
        import types

        import torch.nn as nn

        from tensor_cast.transformers.builtin_model import kimi_k3 as _km

        package = "transformers_modules.test_kimi_k3_generation"
        k3_name = f"{package}.modeling_kimi_k3"
        linear_name = f"{package}.modeling_kimi_linear"

        class KimiDeltaAttention(nn.Module):
            def __init__(self):
                nn.Module.__init__(self)

        class KimiK3ForConditionalGeneration(nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                self.self_attn = KimiDeltaAttention()

        KimiDeltaAttention.__module__ = linear_name
        KimiK3ForConditionalGeneration.__module__ = k3_name

        k3_module = types.ModuleType(k3_name)
        k3_module.KimiK3ForConditionalGeneration = KimiK3ForConditionalGeneration
        linear_module = types.ModuleType(linear_name)
        linear_module.KimiDeltaAttention = KimiDeltaAttention

        class _Config:
            model_type = "kimi_k3"

        saved = {name: sys.modules.get(name) for name in (k3_name, linear_name)}
        sys.modules[k3_name] = k3_module
        sys.modules[linear_name] = linear_module
        try:
            model = SimpleNamespace(_inner=KimiK3ForConditionalGeneration(), hf_config=_Config())
            patch_calls = []

            def _fake_patch(config, model_id):
                patch_calls.append(model_id)
                for cls in (KimiK3ForConditionalGeneration, KimiDeltaAttention):
                    setattr(cls, _km._K3_CLASS_PATCH_MARKER, True)
                return True

            with mock.patch.object(_km, "_patch_model_classes_for_kimi_k3", side_effect=_fake_patch):
                self.assertTrue(_km._ensure_k3_class_patches(model))
                self.assertEqual(len(patch_calls), 1)

                # Same generation: the verified fingerprint short-circuits the
                # walk, so the guard costs two module scans per inference.
                with mock.patch.object(_km, "_k3_live_classes", side_effect=AssertionError("walked the model")):
                    self.assertFalse(_km._ensure_k3_class_patches(model))

                # Only the linear file is re-executed in place.  The instance's
                # classes are still the marked ones, but the generation moved,
                # so the guard must walk again instead of trusting the memo.
                linear_module.KimiDeltaAttention = type(
                    "KimiDeltaAttention",
                    (nn.Module,),
                    {"__init__": lambda self: nn.Module.__init__(self), "__module__": linear_name},
                )
                with mock.patch.object(_km, "_k3_live_classes", wraps=_km._k3_live_classes) as walk:
                    self.assertFalse(_km._ensure_k3_class_patches(model))
                walk.assert_called_once()
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            for cls in (KimiK3ForConditionalGeneration, KimiDeltaAttention):
                if hasattr(cls, _km._K3_CLASS_PATCH_MARKER):
                    delattr(cls, _km._K3_CLASS_PATCH_MARKER)

    def test_ensure_k3_class_patches_pins_every_live_remote_class(self):
        """Re-patching must span the whole instance, not just the wrapper.

        The patch pass resolves each target through ``_get_k3_class_from_source``
        by name. After an in-place re-execution those names resolve to the new
        generation, so pinning only the wrapper would patch attention/MoE/MLP
        classes nothing calls and leave the live model unpatched.
        """
        import sys
        import types

        import torch.nn as nn

        from tensor_cast.transformers.builtin_model import kimi_k3 as _km

        package = "transformers_modules.test_kimi_k3_deep"

        class _Config:
            model_type = "kimi_k3"
            name_or_path = "moonshotai/Kimi-K3"

        def _module_cls(name, module_name, children=(), base=nn.Module):
            cls = type(name, (base,), {"__module__": module_name})

            def __init__(self, _children=children):
                base.__init__(self)
                for attr, child in _children:
                    if isinstance(child, tuple):
                        setattr(self, attr, nn.ModuleList([child[0]() for _ in range(child[1])]))
                    else:
                        setattr(self, attr, child())

            cls.__init__ = __init__
            return cls

        linear = f"{package}.modeling_kimi_linear"
        KimiMoEGate = _module_cls("KimiMoEGate", linear)
        KimiSparseMoeBlock = _module_cls("KimiSparseMoeBlock", linear, (("gate", KimiMoEGate),))
        KimiDeltaAttention = _module_cls("KimiDeltaAttention", linear)
        KimiMLP = _module_cls("KimiMLP", linear)
        KimiDecoderLayer = _module_cls(
            "KimiDecoderLayer",
            linear,
            (("self_attn", KimiDeltaAttention), ("mlp", KimiMLP), ("block_sparse_moe", KimiSparseMoeBlock)),
        )
        KimiLinearModel = _module_cls("KimiLinearModel", linear, (("layers", (KimiDecoderLayer, 1)),))
        KimiLinearForCausalLM = _module_cls("KimiLinearForCausalLM", linear, (("model", KimiLinearModel),))
        KimiK3ForConditionalGeneration = _module_cls(
            "KimiK3ForConditionalGeneration",
            f"{package}.modeling_kimi_k3",
            (("language_model", KimiLinearForCausalLM),),
        )

        k3_name = f"{package}.modeling_kimi_k3"
        saved = sys.modules.get(k3_name)
        sys.modules[k3_name] = types.ModuleType(k3_name)
        try:
            inner = KimiK3ForConditionalGeneration()
            model = SimpleNamespace(_inner=inner, hf_config=_Config())
            captured = {}

            def _fake_patch(config, model_id):
                captured["overrides"] = dict(_km._k3_class_overrides())
                return True

            with mock.patch.object(_km, "_patch_model_classes_for_kimi_k3", side_effect=_fake_patch):
                self.assertTrue(_km._ensure_k3_class_patches(model))

            overrides = captured["overrides"]
            for cls in (
                KimiK3ForConditionalGeneration,
                KimiLinearForCausalLM,
                KimiLinearModel,
                KimiDecoderLayer,
                KimiDeltaAttention,
                KimiMLP,
                KimiSparseMoeBlock,
                KimiMoEGate,
            ):
                # Both ref prefixes name the live class, so whichever prefix a
                # patch site asks for, it lands on the object the model calls.
                self.assertIs(overrides[f"modeling_kimi_linear.{cls.__name__}"], cls)
                self.assertIs(overrides[f"modeling_kimi_k3.{cls.__name__}"], cls)
            self.assertEqual(_km._k3_class_overrides(), {})
        finally:
            if saved is None:
                sys.modules.pop(k3_name, None)
            else:
                sys.modules[k3_name] = saved

    def test_patch_model_classes_not_marked_after_partial_failure(self):
        """Issue #596 review: when a patch block swallows an exception, the
        remote modules must stay unmarked so the next entry point retries the
        full (idempotent) pass instead of permanently caching a half-patched
        module.
        """
        import sys
        import types

        from tensor_cast.core.model_runner import ModelRunner
        from tensor_cast.transformers.builtin_model.kimi_k3 import _patch_model_classes_for_kimi_k3
        from tensor_cast.transformers.model import ModelWrapper

        package = "transformers_modules.test_kimi_k3_partial"

        def _noop(self, *args, **kwargs):
            return None

        class _FakeVL:
            def forward(self, *args, **kwargs):
                return None

            def _merge_input_ids_with_image_features(self, *args, **kwargs):
                return None

        _FakeVL.__module__ = f"{package}.modeling_kimi_k3"

        k3_module = types.ModuleType(f"{package}.modeling_kimi_k3")
        linear_module = types.ModuleType(f"{package}.modeling_kimi_linear")

        def _stub(name, module):
            cls = type(
                name,
                (),
                {
                    "forward": _noop,
                    "moe_infer": _noop,
                    "_update_linear_attn_mask": _noop,
                    "__module__": module.__name__,
                },
            )
            setattr(module, name, cls)
            return cls

        k3_module.KimiK3ForConditionalGeneration = _FakeVL
        k3_module.VL_VISION_ATTENTION_FUNCTIONS = {}
        _stub("MoonViT3dEncoder", k3_module)
        _stub("MoonVision3dPatchEmbed", k3_module)
        for name in (
            "KimiDeltaAttention",
            "KimiLinearModel",
            "KimiSparseMoeBlock",
            "KimiMoEGate",
            "KimiDecoderLayer",
            "SituAndMul",
            "KimiMLP",
            "KimiBlockSparseMLP",
            "KimiLinearForCausalLM",
        ):
            _stub(name, linear_module)
        linear_module._apply_attn_res = lambda *args, **kwargs: None

        names = (f"{package}.modeling_kimi_k3", f"{package}.modeling_kimi_linear")
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules[names[0]] = k3_module
        sys.modules[names[1]] = linear_module

        def _fake_get_class(class_ref, *args, **kwargs):
            module_name, _, class_name = class_ref.partition(".")
            module = sys.modules[f"{package}.{module_name}"]
            try:
                return getattr(module, class_name)
            except AttributeError as error:
                raise RuntimeError(f"fake class {class_ref} unavailable") from error

        def _restore():
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        def _restore_flag(cls, name, saved_value):
            if saved_value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, saved_value)

        class _KimiConfig:
            model_type = "kimi_k3"

        saved_mw_flag = getattr(ModelWrapper, "_patched_for_k3_mtp", None)
        saved_runner_flag = getattr(ModelRunner, "_tensor_cast_k3_constant_folder_guard", None)
        ModelWrapper._patched_for_k3_mtp = True
        ModelRunner._tensor_cast_k3_constant_folder_guard = True

        marker = self._km._K3_MODULE_PATCH_MARKER
        saved_gate = getattr(linear_module, "KimiMoEGate")
        try:
            with mock.patch.object(self._km, "_get_k3_class_from_source", side_effect=_fake_get_class):
                # Force a swallowed patch failure: KimiMoEGate unresolvable.
                delattr(linear_module, "KimiMoEGate")
                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                self.assertFalse(getattr(k3_module, marker, False))
                self.assertFalse(getattr(linear_module, marker, False))
                # The next entry retries the full pass instead of no-op.
                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                self.assertFalse(getattr(linear_module, marker, False))
                # Heal -> the retry completes and stamps both modules.
                setattr(linear_module, "KimiMoEGate", saved_gate)
                self.assertTrue(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
                self.assertTrue(getattr(k3_module, marker, False))
                self.assertTrue(getattr(linear_module, marker, False))
                # Fully patched modules short-circuit again.
                self.assertFalse(_patch_model_classes_for_kimi_k3(_KimiConfig(), "moonshotai/Kimi-K3"))
        finally:
            setattr(linear_module, "KimiMoEGate", saved_gate)
            _restore()
            _restore_flag(ModelWrapper, "_patched_for_k3_mtp", saved_mw_flag)
            _restore_flag(ModelRunner, "_tensor_cast_k3_constant_folder_guard", saved_runner_flag)

    def test_non_quantized_mla_prolog_forwards_phase_metadata(self):
        """The K3 custom BF16 prolog must preserve Chunked-Prefill phase metadata."""
        phase_values = [False, False]
        captured_kwargs = {}
        expected_result = object()

        def original_compute_mla_prolog(*_args, **_kwargs):
            self.fail("The K3 non-quantized path must call tensor_cast.mla_prolog directly")

        def fake_mla_prolog(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return expected_result

        hook = self._km._make_k3_mla_prolog_hook(original_compute_mla_prolog)
        attention = SimpleNamespace(
            _inner=SimpleNamespace(use_output_gate=True),
            q_a_proj=SimpleNamespace(weight=torch.ones(2, 3)),
            q_b_proj=SimpleNamespace(weight=torch.ones(2, 2)),
            kv_a_proj_with_mqa=SimpleNamespace(weight=torch.ones(2, 3)),
            q_a_layernorm=SimpleNamespace(weight=torch.ones(2)),
            kv_a_layernorm=SimpleNamespace(weight=torch.ones(2)),
            extract_qparams=lambda module: (module.weight, None, None),
            _num_heads_per_rank=1,
            qk_head_dim=2,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            kv_lora_rank=2,
            q_lora_rank=2,
        )

        with mock.patch.object(torch.ops.tensor_cast, "mla_prolog", new=fake_mla_prolog, create=True):
            result = hook(
                attention,
                torch.ones(2, 3),
                torch.ones(2, 1),
                torch.zeros(2, 1),
                is_decode_values=phase_values,
            )

        self.assertIs(result, expected_result)
        self.assertIs(captured_kwargs["is_decode_values"], phase_values)

    def _make_k3_quantized_attention(self):
        """Build a quantized K3 MLA stand-in: all three projection scales set."""
        return SimpleNamespace(
            _inner=SimpleNamespace(use_output_gate=True),
            q_a_proj=SimpleNamespace(weight=torch.ones(2, 3)),
            q_b_proj=SimpleNamespace(weight=torch.ones(2, 2)),
            kv_a_proj_with_mqa=SimpleNamespace(weight=torch.ones(2, 3)),
            q_a_layernorm=SimpleNamespace(weight=torch.ones(2)),
            kv_a_layernorm=SimpleNamespace(weight=torch.ones(2)),
            extract_qparams=lambda module: (module.weight, torch.ones(1), torch.zeros(1)),
            _num_heads_per_rank=1,
            qk_head_dim=2,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            kv_lora_rank=2,
            q_lora_rank=2,
        )

    def _call_k3_prolog_hook(self, hook, attention, *, enable_mlapo):
        from tensor_cast import config

        previous = config.compilation.fusion_patterns.enable_mlapo
        config.compilation.fusion_patterns.enable_mlapo = enable_mlapo
        try:
            return hook(attention, torch.ones(2, 3), torch.ones(2, 1), torch.zeros(2, 1))
        finally:
            config.compilation.fusion_patterns.enable_mlapo = previous

    def test_quantized_mla_prolog_preserves_k3_contract_for_both_switch_values(self):
        attention = self._make_k3_quantized_attention()
        prolog_outputs = tuple(object() for _ in range(6))
        attention._compute_fused_mla_prolog = lambda *args, **kwargs: prolog_outputs

        def original_compute_mla_prolog(*_args, **_kwargs):
            self.fail("K3 retains its original quantized prolog independently of the generic switch")

        hook = self._km._make_k3_mla_prolog_hook(original_compute_mla_prolog)
        for enabled in (False, True):
            with self.subTest(enable_mlapo=enabled):
                result = self._call_k3_prolog_hook(hook, attention, enable_mlapo=enabled)
                self.assertEqual(result, prolog_outputs[:4])

    def test_registered_mla_prolog_accepts_phase_metadata(self):
        """The real custom-op schema must accept the forwarded phase metadata."""
        self._km._install_mla_prolog_op()
        phase_values = [False, True]

        result = torch.ops.tensor_cast.mla_prolog(
            torch.ones(2, 3),
            torch.ones(2, 2),
            torch.ones(2, 1),
            torch.zeros(2, 1),
            torch.ones(2),
            torch.ones(2, 2),
            torch.ones(3, 3),
            torch.ones(2),
            1,
            2,
            1,
            1,
            2,
            2,
            is_decode_values=phase_values,
        )

        self.assertEqual(
            [argument.name for argument in torch.ops.tensor_cast.mla_prolog.default._schema.arguments][-1],
            "is_decode_values",
        )
        self.assertEqual([tuple(tensor.shape) for tensor in result], [(2, 1, 2), (2, 2), (2, 1), (2, 2)])

    # ------------------------------------------------------------------
    # _install_fla_stub — idempotency
    # ------------------------------------------------------------------

    def test_fla_stub_idempotent(self):
        """The import-compatible stub is installed once and supports discovery."""
        import importlib.util
        import sys

        from transformers.utils.import_utils import is_flash_linear_attention_available

        self._km._install_fla_stub()
        self.assertTrue(self._km._FLA_STUB_INSTALLED)

        fla_module = sys.modules.get("fla")
        self.assertIsNotNone(fla_module, "fla stub should be in sys.modules after install")
        self.assertIsNotNone(fla_module.__spec__)
        self.assertEqual(importlib.util.find_spec("fla").name, "fla")
        for name in self._fla_module_names:
            self.assertIsNotNone(sys.modules[name].__spec__, f"{name} should have an import spec")
        self.assertIsInstance(is_flash_linear_attention_available(), bool)

        self._km._install_fla_stub()
        self.assertTrue(self._km._FLA_STUB_INSTALLED)
        self.assertIs(
            sys.modules.get("fla"),
            fla_module,
            "Second call should not re-create fla stub",
        )

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

    # ------------------------------------------------------------------
    # install → restore → install: no wrapper stacking (Issue 402 review)
    # ------------------------------------------------------------------

    def test_install_restore_install_no_wrapper_stacking(self):
        """Verify ``_restore_k3_global_state`` fully restores class-level methods.

        If the restore function resets install flags but does NOT restore the
        class-level callable, the next install wraps the already-wrapped
        method, forming a callable chain.  For ``FusedMoETensorCast.forward``
        this could double-execute ``routed_expert_down_proj`` and cause
        dimension errors (PR review feedback on Issue 402 fix).

        The test verifies that after restore, each class-level method is back
        to the **original** (pre-K3) callable.  This ensures the next install
        starts from the original, not from an already-wrapped method.
        """
        from tensor_cast.layers.internal import CopyLayerWrapper
        from tensor_cast.layers.moe_layer import (
            FusedMoETensorCast,
            MoELayer,
            ParallelMoELayer,
        )
        from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

        # --- Phase 0: Save originals (before any K3 install) ---
        self._km._restore_k3_global_state()  # start clean
        orig_moe_init = MoELayer.__init__
        orig_parallel_moe_init = ParallelMoELayer.__init__
        orig_fused_moe_fwd = FusedMoETensorCast.forward
        orig_copy_init = CopyLayerWrapper.__init__
        orig_copy_fwd = CopyLayerWrapper.forward
        orig_tp_plan = MultiheadLatentAttentionTensorCast.build_tp_plan_extras

        # --- Phase 1: Install all patches ---
        self._km._install_latent_moe_patch()
        self._km._install_copy_layer_attr_patch()
        self._km._install_kda_tp_plan_patch()

        # After install, methods must be different from originals (wrapped)
        self.assertIsNot(MoELayer.__init__, orig_moe_init)
        self.assertIsNot(ParallelMoELayer.__init__, orig_parallel_moe_init)
        self.assertIsNot(FusedMoETensorCast.forward, orig_fused_moe_fwd)
        self.assertIsNot(CopyLayerWrapper.__init__, orig_copy_init)
        self.assertIsNot(CopyLayerWrapper.forward, orig_copy_fwd)
        self.assertIsNot(MultiheadLatentAttentionTensorCast.build_tp_plan_extras, orig_tp_plan)

        # --- Phase 2: Restore global state ---
        self._km._restore_k3_global_state()

        # After restore, methods MUST be back to the originals.
        # If they are NOT, the next install will stack a new wrapper on top
        # of the Phase-1 wrapper, forming a callable chain.
        self.assertIs(MoELayer.__init__, orig_moe_init)
        self.assertIs(ParallelMoELayer.__init__, orig_parallel_moe_init)
        self.assertIs(FusedMoETensorCast.forward, orig_fused_moe_fwd)
        self.assertIs(CopyLayerWrapper.__init__, orig_copy_init)
        self.assertIs(CopyLayerWrapper.forward, orig_copy_fwd)
        # classmethod access creates a new bound method each time, so compare
        # the underlying function (__func__) instead of the bound method object.
        self.assertIs(
            MultiheadLatentAttentionTensorCast.build_tp_plan_extras.__func__,
            orig_tp_plan.__func__,
        )

        # --- Phase 3: Re-install all patches ---
        self._km._install_latent_moe_patch()
        self._km._install_copy_layer_attr_patch()
        self._km._install_kda_tp_plan_patch()

        # The re-installed methods should be different from originals (wrapped)
        self.assertIsNot(MoELayer.__init__, orig_moe_init)
        self.assertIsNot(FusedMoETensorCast.forward, orig_fused_moe_fwd)

        # --- Phase 4: Restore again ---
        self._km._restore_k3_global_state()
        self.assertIs(MoELayer.__init__, orig_moe_init)
        self.assertIs(FusedMoETensorCast.forward, orig_fused_moe_fwd)
        self.assertIs(
            MultiheadLatentAttentionTensorCast.build_tp_plan_extras.__func__,
            orig_tp_plan.__func__,
        )


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


class TestKdaMemoryAccounting(unittest.TestCase):
    def test_persistent_state_scales_with_rank_local_sequences(self):
        from tensor_cast.transformers.builtin_model.kimi_k3 import _k3_kda_state_bytes_per_layer

        hf_config = SimpleNamespace(
            model_type="kimi_k3",
            text_config=SimpleNamespace(
                linear_attn_config={
                    "kda_layers": [1, 2, 3],
                    "head_dim": 16,
                    "num_heads": 8,
                    "short_conv_kernel_size": 4,
                }
            ),
        )
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=hf_config,
                parallel_config=SimpleNamespace(tensor_parallel_size=2),
            ),
            text_config=hf_config.text_config,
        )

        one_sequence = _k3_kda_state_bytes_per_layer(model, batch_size=1)
        self.assertEqual(_k3_kda_state_bytes_per_layer(model, batch_size=8), 8 * one_sequence)


if __name__ == "__main__":
    unittest.main()
