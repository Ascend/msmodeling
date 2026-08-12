import argparse
import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from tests.helpers.cli_runner import run_module_main

import pytest
import torch
from cli.inference.video_generate import (
    check_attention_sparsity,
    main,
    process_input,
    run_inference,
)
from parameterized import parameterized
from tensor_cast.core.quantization.config import create_attention_quant_config
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.diffusers.cache_agent.cache import CacheState
from tensor_cast.diffusers.cache_agent.dit_block_cache import DiTBlockCache
from tensor_cast.diffusers.diffusers_attention import (
    _attention,
    _get_block_sparse_attention_fallback_reason,
    get_sp_group,
    set_sp_group,
    use_custom_sdpa,
)
from tensor_cast.diffusers.diffusers_model import DiffusersTransformerModel
from tensor_cast.model_config import AttentionBackend, AttentionRoutePlan
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.transformations import wrap_model
from tensor_cast.diffusers.diffusers_utils import SafeMetaTensor
from tensor_cast.diffusers.dit_cache_registry import (
    DiTBlockCacheSpec,
    _get_hunyuanvideo15_blocks_with_setters,
    _get_hunyuanvideo_blocks_with_setters,
    _get_wan_blocks_with_setters,
    _module_list_blocks_with_setters,
    get_dit_block_cache_spec,
    register_dit_block_cache_spec,
    replace_blocks_in_range,
)
from tensor_cast.diffusers.model_resolver import DiffusersModelSelection


def test_safe_meta_tensor_boolean_index_returns_plain_meta_tensor():
    image_embeds = SafeMetaTensor((2, 3), dtype=torch.float16)
    image_mask = torch.ones(2, dtype=torch.bool, device="meta")

    result = image_embeds[image_mask]

    assert type(result) is torch.Tensor
    assert result.device.type == "meta"
    assert result.shape == image_embeds.shape


def test_bsa_fallback_reason_uses_documented_priority():
    route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention)
    query = torch.zeros((1, 2, 4, 3), device="meta")
    key = torch.zeros((1, 2, 5, 3), device="meta")

    assert (
        _get_block_sparse_attention_fallback_reason(query, key, key, route_plan, is_causal=True, enable_gqa=True)
        == "qkv_shape_mismatch"
    )
    assert (
        _get_block_sparse_attention_fallback_reason(torch.zeros((1, 4, 3), device="meta"), key, key, route_plan)
        == "non_4d_qkv"
    )
    assert (
        _get_block_sparse_attention_fallback_reason(
            query,
            query,
            query,
            route_plan,
            attention_mask=torch.zeros((1, 2, 4, 4), device="meta"),
        )
        == "unsupported_attention_mask"
    )


def test_bsa_context_aggregates_fallbacks_and_warns_once(caplog):
    route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention)
    query = torch.zeros((1, 2, 4, 3), device="meta")

    with patch.object(torch.ops.tensor_cast, "attention", return_value=query):
        with use_custom_sdpa(route_plan=route_plan) as stats:
            torch.nn.functional.scaled_dot_product_attention(query, query, query, is_causal=True)
            torch.nn.functional.scaled_dot_product_attention(query, query, query, is_causal=True)
            torch.nn.functional.scaled_dot_product_attention(query, query, query, enable_gqa=True)
            assert stats["dense_fallback_calls"] == 3
            assert stats["dense_fallback_reasons"] == {"causal": 2, "gqa": 1}

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "block_sparse_attention_calls=0" in warnings[0].message
    assert "causal=2, gqa=1" in warnings[0].message


class TestVideoGeneration(unittest.TestCase):
    """Unit tests for video_generate.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock transformer config
        transformer_config = {
            "_class_name": "HunyuanVideoTransformer3DModel",
            "_diffusers_version": "0.32.0.dev0",
            "attention_head_dim": 128,
            "guidance_embeds": "true",
            "in_channels": 16,
            "mlp_ratio": 4.0,
            "num_attention_heads": 24,
            "num_layers": 20,
            "num_refiner_layers": 2,
            "num_single_layers": 40,
            "out_channels": 16,
            "patch_size": 2,
            "patch_size_t": 1,
            "pooled_projection_dim": 768,
            "qk_norm": "rms_norm",
            "rope_axes_dim": [16, 56, 56],
            "rope_theta": 256.0,
            "text_embed_dim": 4096,
        }

        # Create a mock vae config
        vae_config = {
            "_class_name": "AutoencoderKLCogVideoX",
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": [
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ],
            "up_block_types": [
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
            ],
            "block_out_channels": [128, 256, 512],
            "layers_per_block": 4,
            "act_fn": "silu",
            "sample_size": [16, 128, 128],
            "mid_block_type": "CogVideoXMidBlock3D",
            "norm_num_groups": 32,
            "temporal_compression_ratio": 4,
            "z_dim": 16,
        }

        self.temp_dir, self.model_id = self._create_mock_model_dir(transformer_config, vae_config)
        self.device = "TEST_DEVICE"
        self.batch_size = 2
        self.seq_len = 10
        self.height = 400
        self.width = 832
        self.frame_num = 81
        self.sample_step = 1
        torch.compiler.reset()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_model_dir(self, transformer_config, vae_config):
        from tests.helpers.local_model_path_permissions import harden_local_model_tree

        temp_dir = tempfile.mkdtemp(dir=os.path.realpath(os.getcwd()))
        model_dir = os.path.join(temp_dir, "mock_model")
        os.makedirs(model_dir, exist_ok=True)

        transformer_dir = os.path.join(model_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(transformer_config, f)
        if transformer_config.get("_class_name") == "HunyuanVideo15Transformer3DModel":
            with open(os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "_class_name": "HunyuanVideo15Pipeline",
                        "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
                    },
                    f,
                )

        # Write VAE config
        vae_dir = os.path.join(model_dir, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        with open(os.path.join(vae_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vae_config, f)
        # tempfile/makedirs honor process umask; keep mock trees security-check safe.
        harden_local_model_tree(model_dir)
        return temp_dir, model_dir

    def _validate_inference_result(self, test_name: str = ""):
        """Validate the result from run_inference doesn't raise exceptions.
        Since run_inference returns None, we check for successful execution.

        Args:
            test_name: Name of the test for better error messages
        """
        # If we reach this point, the function executed successfully
        self.assertTrue(True, f"{test_name}: Inference ran without exceptions")

    def test_main_given_invalid_log_level_argument_when_invoked_then_system_exits_with_code_2(
        self,
    ):
        '''Test the "main" function in "text_generate"'''
        original_argv = sys.argv

        try:
            sys.argv = [
                self.model_id,
                "--batch-size",
                str(self.batch_size),
                "--seq-len",
                str(self.seq_len),
                "--log-level",
                "2",
            ]
            with self.assertRaises(SystemExit) as cm:
                main()

            self.assertEqual(cm.exception.code, 2)
        finally:
            sys.argv = original_argv

    def test_basic_video_inference(self):
        """Test basic video inference without Ulysses parallelism."""
        run_inference(
            device=self.device,
            model_id=self.model_id,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            height=self.height,
            width=self.width,
            frame_num=self.frame_num,
            sample_step=self.sample_step,
            dtype="float16",
            world_size=1,
            ulysses_size=1,
        )
        self._validate_inference_result("test_basic_video_inference")

    def test_run_inference_restores_outer_sequence_parallel_group(self):
        outer_sp_group = MagicMock(world_size=1)
        set_sp_group(outer_sp_group)
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                dtype="float16",
                world_size=1,
                ulysses_size=1,
            )

            self.assertIs(get_sp_group(), outer_sp_group)
        finally:
            set_sp_group(None)

    def test_main_given_fp8_attention_quantization_when_invoked_then_passes_action_to_inference(self):
        original_argv = sys.argv
        try:
            sys.argv = [
                "video_generate.py",
                self.model_id,
                "--batch-size",
                str(self.batch_size),
                "--seq-len",
                str(self.seq_len),
                "--quantize-attention-action",
                "FP8",
            ]
            with patch("cli.inference.video_generate.run_inference") as mock_run_inference:
                main()

            self.assertEqual(
                mock_run_inference.call_args.kwargs["quantize_attention_action"], QuantizeAttentionAction.FP8
            )
        finally:
            sys.argv = original_argv

    def test_cli_run_inference_is_covered_in_process(self):
        from types import SimpleNamespace
        from cli.inference import video_generate as video_generate_mod

        fake_device = SimpleNamespace(name="TEST_DEVICE")
        fake_model_config = SimpleNamespace(
            transformer_config=SimpleNamespace(
                parallel_config=SimpleNamespace(ulysses_size=1, world_size=1),
                dtype=torch.float16,
                model_config={
                    "_class_name": "HunyuanVideoTransformer3DModel",
                    "in_channels": 16,
                    "text_embed_dim": 4096,
                    "guidance_embeds": "true",
                    "pooled_projection_dim": 768,
                },
            )
        )
        fake_model = MagicMock()
        fake_model.forward.return_value = torch.zeros(1, device="meta")
        fake_model_selection = object()
        fake_runtime = MagicMock()
        fake_runtime.__enter__.return_value = fake_runtime
        fake_runtime.__exit__.return_value = False
        fake_runtime.table_averages.return_value = {"ok": True}
        fake_sdpa = MagicMock()
        fake_sdpa.__enter__.return_value = fake_sdpa
        fake_sdpa.__exit__.return_value = False

        with (
            patch.object(video_generate_mod.DeviceProfile, "all_device_profiles", {"TEST_DEVICE": fake_device}),
            patch.object(video_generate_mod, "AnalyticPerformanceModel") as mock_perf_model,
            patch.object(video_generate_mod, "ParallelConfig") as mock_parallel_config,
            patch.object(video_generate_mod, "create_quant_config") as mock_create_quant_config,
            patch.object(video_generate_mod, "str_to_dtype", return_value=torch.float16),
            patch(
                "tensor_cast.diffusers.model_resolver.resolve_diffusers_model_selection",
                return_value=fake_model_selection,
            ),
            patch(
                "tensor_cast.diffusers.diffusers_model.build_diffusers_transformer_model",
                return_value=(fake_model, fake_model_config),
            ) as mock_build_model,
            patch("tensor_cast.diffusers.diffusers_attention.use_custom_sdpa", return_value=fake_sdpa),
            patch("tensor_cast.diffusers.diffusers_attention.set_sp_group"),
            patch.object(video_generate_mod, "Runtime", return_value=fake_runtime),
            patch.object(video_generate_mod, "MemoryTracker", return_value=MagicMock()),
            patch.object(video_generate_mod, "time") as mock_time,
            patch.object(video_generate_mod, "print"),
        ):
            mock_parallel_config.return_value = SimpleNamespace(ulysses_size=1, world_size=1)
            mock_create_quant_config.return_value = SimpleNamespace(attention_configs={-1: None})
            mock_perf_model.return_value = MagicMock()
            mock_time.perf_counter.side_effect = [1.0, 2.0]

            video_generate_mod.run_inference(
                device="TEST_DEVICE",
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=1,
                dtype="float16",
                dit_cache=True,
                cache_step_range="0,0",
                cache_step_interval=2,
            )

        mock_perf_model.assert_called_once()
        mock_create_quant_config.assert_called_once()
        assert mock_build_model.call_count == 2
        assert all(call.kwargs["model_selection"] is fake_model_selection for call in mock_build_model.call_args_list)

    def test_cli_main_is_covered_in_process(self):
        from cli.inference import video_generate as video_generate_mod

        with patch.object(video_generate_mod, "run_inference") as mock_run_inference:
            result = run_module_main(
                "cli.inference.video_generate",
                [
                    "--device",
                    "TEST_DEVICE",
                    self.model_id,
                    "--batch-size",
                    str(self.batch_size),
                    "--seq-len",
                    str(self.seq_len),
                    "--quantize-attention-action",
                    "DISABLED",
                ],
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        mock_run_inference.assert_called_once()

    def test_cli_main_given_compile_arguments_when_invoked_then_passes_compile_options_to_inference(self):
        from cli.inference import video_generate as video_generate_mod

        with patch.object(video_generate_mod, "run_inference") as mock_run_inference:
            result = run_module_main(
                "cli.inference.video_generate",
                [
                    "--device",
                    "TEST_DEVICE",
                    self.model_id,
                    "--batch-size",
                    str(self.batch_size),
                    "--seq-len",
                    str(self.seq_len),
                    "--compile",
                    "--compile-allow-graph-break",
                ],
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(mock_run_inference.call_args.kwargs["compile"])
        self.assertTrue(mock_run_inference.call_args.kwargs["compile_allow_graph_break"])

    def test_cli_given_block_sparse_attention_options_then_forwards_route_plan(self):
        from cli.inference import video_generate as video_generate_mod

        with patch.object(video_generate_mod, "run_inference") as mock_run_inference:
            result = run_module_main(
                "cli.inference.video_generate",
                [
                    "--device",
                    "TEST_DEVICE",
                    self.model_id,
                    "--batch-size",
                    str(self.batch_size),
                    "--seq-len",
                    str(self.seq_len),
                    "--attention-backend",
                    "block_sparse_attention",
                    "--attention-block-size",
                    "64",
                    "--attention-sparsity",
                    "0.5",
                ],
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        kwargs = mock_run_inference.call_args.kwargs
        self.assertEqual(kwargs["attention_backend"], AttentionBackend.block_sparse_attention)
        self.assertEqual(kwargs["attention_block_size"], 64)
        self.assertEqual(kwargs["attention_sparsity"], 0.5)

    def test_block_sparse_attention_rejects_attention_quantization(self):
        with self.assertRaises(ValueError, msg="attention quantization must be rejected"):
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                attention_backend=AttentionBackend.block_sparse_attention,
                quantize_attention_action=QuantizeAttentionAction.FP8,
            )
        with self.assertRaises(argparse.ArgumentTypeError):
            check_attention_sparsity("1.0")

    def test_direct_sdpa_rejects_bsa_attention_quantization_before_installation(self):
        quant_config = create_attention_quant_config(QuantizeAttentionAction.FP8)
        route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention)
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        with self.assertRaisesRegex(ValueError, "does not support attention quantization"):
            with use_custom_sdpa(quant_config, route_plan):
                self.fail("use_custom_sdpa should reject the incompatible configuration")

        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)

    def test_block_sparse_attention_routes_equal_qkv_before_sparse_attention(self):
        route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention, block_size=2, sparsity=0.5)
        query = torch.zeros((1, 4, 1, 2), device="meta")
        route_metadata = torch.empty((1, 1, 2, 2), dtype=torch.int32, device="meta")
        events = []

        def generate_route(*args):
            events.append("attention_route_generate")
            return route_metadata

        def run_sparse_attention(*args):
            events.append("block_sparse_attention")
            return query

        def run_dense_attention(*args):
            events.append("attention")
            return query

        with (
            patch.object(
                torch.ops.tensor_cast, "attention_route_generate", side_effect=generate_route
            ) as route_generate,
            patch.object(
                torch.ops.tensor_cast, "block_sparse_attention", side_effect=run_sparse_attention
            ) as sparse_attention,
            patch.object(torch.ops.tensor_cast, "attention", side_effect=run_dense_attention) as dense_attention,
            use_custom_sdpa(route_plan=route_plan) as stats,
        ):
            _attention(query, query, query)
            self.assertEqual(events, ["attention_route_generate", "block_sparse_attention"])
            _attention(query, torch.zeros((1, 2, 1, 2), device="meta"), torch.zeros((1, 2, 1, 2), device="meta"))
            _attention(query, query, query, attn_mask=torch.zeros((1, 1, 4, 4), device="meta"))

        route_generate.assert_called_once_with(query, query, 2, 0.5)
        sparse_attention.assert_called_once()
        self.assertEqual(dense_attention.call_count, 2)
        self.assertEqual(events, ["attention_route_generate", "block_sparse_attention", "attention", "attention"])
        self.assertEqual(stats["block_sparse_attention_calls"], 1)
        self.assertEqual(stats["dense_fallback_calls"], 2)
        self.assertEqual(
            stats["dense_fallback_reasons"],
            {"qkv_shape_mismatch": 1, "unsupported_attention_mask": 1},
        )

    def test_direct_sdpa_block_sparse_attention_normalizes_tensor_layout(self):
        route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention, block_size=2, sparsity=0.5)
        sdpa_query = torch.zeros((1, 2, 4, 3), device="meta")
        route_metadata = torch.empty((1, 3, 2, 2), dtype=torch.int32, device="meta")
        events = []

        def generate_route(query, key, *args):
            events.append("attention_route_generate")
            self.assertEqual(query.shape, (1, 4, 2, 3))
            self.assertEqual(key.shape, (1, 4, 2, 3))
            return route_metadata

        def run_sparse_attention(query, key, value, *args):
            events.append("block_sparse_attention")
            self.assertEqual(query.shape, (1, 4, 2, 3))
            return query

        with (
            patch.object(
                torch.ops.tensor_cast, "attention_route_generate", side_effect=generate_route
            ) as route_generate,
            patch.object(
                torch.ops.tensor_cast, "block_sparse_attention", side_effect=run_sparse_attention
            ) as sparse_attention,
            use_custom_sdpa(route_plan=route_plan),
        ):
            output = torch.nn.functional.scaled_dot_product_attention(sdpa_query, sdpa_query, sdpa_query)

        self.assertEqual(output.shape, sdpa_query.shape)
        self.assertEqual(events, ["attention_route_generate", "block_sparse_attention"])
        route_generate.assert_called_once()
        sparse_attention.assert_called_once()

    def test_direct_sdpa_unsupported_semantics_fall_back_to_dense_attention(self):
        route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention, block_size=2, sparsity=0.5)
        query = torch.zeros((1, 2, 4, 3), device="meta")

        def run_dense_attention(query, key, value, *args):
            self.assertEqual(query.shape, (1, 4, 2, 3))
            self.assertEqual(key.shape, (1, 4, 2, 3))
            self.assertEqual(value.shape, (1, 4, 2, 3))
            return query

        with (
            patch.object(torch.ops.tensor_cast, "attention_route_generate") as route_generate,
            patch.object(torch.ops.tensor_cast, "block_sparse_attention") as sparse_attention,
            patch.object(torch.ops.tensor_cast, "attention", side_effect=run_dense_attention) as dense_attention,
            use_custom_sdpa(route_plan=route_plan),
        ):
            output = torch.nn.functional.scaled_dot_product_attention(query, query, query, is_causal=True)

        self.assertEqual(output.shape, query.shape)
        route_generate.assert_not_called()
        sparse_attention.assert_not_called()
        dense_attention.assert_called_once()

    def test_sequence_parallel_non_4d_bsa_call_falls_back_before_all_to_all(self):
        route_plan = AttentionRoutePlan(backend=AttentionBackend.block_sparse_attention, block_size=2, sparsity=0.5)
        query = torch.zeros((4, 2, 3), device="meta")
        sp_group = MagicMock(world_size=2)

        set_sp_group(sp_group)
        try:
            with (
                patch.object(torch.ops.tensor_cast, "attention", return_value=query) as dense_attention,
                use_custom_sdpa(route_plan=route_plan) as stats,
            ):
                output = _attention(query, query, query)
        finally:
            set_sp_group(None)

        self.assertEqual(output.shape, query.shape)
        sp_group.all_to_all.assert_not_called()
        dense_attention.assert_called_once()
        self.assertEqual(stats["dense_fallback_calls"], 1)
        self.assertEqual(stats["dense_fallback_reasons"], {"non_4d_qkv": 1})

    def test_run_inference_given_compile_when_invoked_then_compiles_model_before_forward(self):
        from types import SimpleNamespace
        from cli.inference import video_generate as video_generate_mod

        fake_device = SimpleNamespace(name="TEST_DEVICE")
        fake_model_config = SimpleNamespace(
            transformer_config=SimpleNamespace(
                parallel_config=SimpleNamespace(ulysses_size=1, world_size=1),
                dtype=torch.float16,
                model_config={
                    "_class_name": "HunyuanVideoTransformer3DModel",
                    "in_channels": 16,
                    "text_embed_dim": 4096,
                    "guidance_embeds": "true",
                    "pooled_projection_dim": 768,
                },
            )
        )
        fake_model = MagicMock()
        fake_model.forward.return_value = torch.zeros(1, device="meta")
        fake_runtime = MagicMock()
        fake_runtime.__enter__.return_value = fake_runtime
        fake_runtime.__exit__.return_value = False
        fake_runtime.table_averages.return_value = {"ok": True}
        fake_sdpa = MagicMock()
        fake_sdpa.__enter__.return_value = fake_sdpa
        fake_sdpa.__exit__.return_value = False
        fake_backend = MagicMock()

        with (
            patch.object(video_generate_mod.DeviceProfile, "all_device_profiles", {"TEST_DEVICE": fake_device}),
            patch.object(video_generate_mod, "AnalyticPerformanceModel"),
            patch.object(
                video_generate_mod, "ParallelConfig", return_value=SimpleNamespace(ulysses_size=1, world_size=1)
            ),
            patch.object(
                video_generate_mod, "create_quant_config", return_value=SimpleNamespace(attention_configs={-1: None})
            ),
            patch.object(video_generate_mod, "str_to_dtype", return_value=torch.float16),
            patch.object(video_generate_mod, "get_backend", return_value=fake_backend) as mock_get_backend,
            patch.object(
                video_generate_mod.torch, "compile", side_effect=lambda model, **kwargs: model
            ) as mock_compile,
            patch(
                "tensor_cast.diffusers.diffusers_model.build_diffusers_transformer_model",
                return_value=(fake_model, fake_model_config),
            ),
            patch("tensor_cast.diffusers.diffusers_attention.use_custom_sdpa", return_value=fake_sdpa),
            patch.object(video_generate_mod, "Runtime", return_value=fake_runtime),
            patch.object(video_generate_mod, "MemoryTracker", return_value=MagicMock()),
            patch.object(video_generate_mod, "time") as mock_time,
            patch.object(video_generate_mod, "print"),
        ):
            mock_time.perf_counter.side_effect = [1.0, 2.0]

            video_generate_mod.run_inference(
                device="TEST_DEVICE",
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=1,
                dtype="float16",
                compile=True,
                compile_allow_graph_break=True,
            )

        mock_get_backend.assert_called_once_with(device_name="TEST_DEVICE")
        mock_compile.assert_called_once_with(fake_model, backend=fake_backend, dynamic=False, fullgraph=False)
        fake_model.forward.assert_called_once()

    def test_run_inference_given_compile_with_dit_cache_then_compiles_both_models_with_fullgraph(self):
        from types import SimpleNamespace
        from cli.inference import video_generate as video_generate_mod

        fake_device = SimpleNamespace(name="TEST_DEVICE")

        def make_model_config():
            return SimpleNamespace(
                transformer_config=SimpleNamespace(
                    parallel_config=SimpleNamespace(ulysses_size=1, world_size=1),
                    dtype=torch.float16,
                    model_config={
                        "_class_name": "HunyuanVideoTransformer3DModel",
                        "in_channels": 16,
                        "text_embed_dim": 4096,
                        "guidance_embeds": "true",
                        "pooled_projection_dim": 768,
                    },
                )
            )

        primary_model_config = make_model_config()
        cache_model_config = make_model_config()
        primary_model = MagicMock()
        primary_model.forward.return_value = torch.zeros(1, device="meta")
        cache_model = MagicMock()
        cache_model.forward.return_value = torch.zeros(1, device="meta")
        cache_model.enable_dit_block_cache.return_value = CacheState()
        fake_runtime = MagicMock()
        fake_runtime.__enter__.return_value = fake_runtime
        fake_runtime.__exit__.return_value = False
        fake_runtime.table_averages.return_value = {"ok": True}
        fake_sdpa = MagicMock()
        fake_sdpa.__enter__.return_value = fake_sdpa
        fake_sdpa.__exit__.return_value = False
        fake_backend = MagicMock()

        with (
            patch.object(video_generate_mod.DeviceProfile, "all_device_profiles", {"TEST_DEVICE": fake_device}),
            patch.object(video_generate_mod, "AnalyticPerformanceModel"),
            patch.object(
                video_generate_mod, "ParallelConfig", return_value=SimpleNamespace(ulysses_size=1, world_size=1)
            ),
            patch.object(
                video_generate_mod, "create_quant_config", return_value=SimpleNamespace(attention_configs={-1: None})
            ),
            patch.object(video_generate_mod, "str_to_dtype", return_value=torch.float16),
            patch.object(video_generate_mod, "get_backend", return_value=fake_backend),
            patch.object(
                video_generate_mod.torch, "compile", side_effect=lambda model, **kwargs: model
            ) as mock_compile,
            patch(
                "tensor_cast.diffusers.diffusers_model.build_diffusers_transformer_model",
                side_effect=[(primary_model, primary_model_config), (cache_model, cache_model_config)],
            ),
            patch(
                "tensor_cast.diffusers.diffusers_attention.use_custom_sdpa", return_value=fake_sdpa
            ) as mock_use_custom_sdpa,
            patch.object(video_generate_mod, "Runtime", return_value=fake_runtime),
            patch.object(video_generate_mod, "MemoryTracker", return_value=MagicMock()),
            patch.object(video_generate_mod, "time") as mock_time,
            patch.object(video_generate_mod, "print"),
        ):
            mock_time.perf_counter.side_effect = [1.0, 2.0]
            video_generate_mod.run_inference(
                device="TEST_DEVICE",
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=3,
                dtype="float16",
                compile=True,
                dit_cache=True,
                cache_step_range="0,1",
                cache_step_interval=2,
            )

        assert mock_compile.call_args_list == [
            unittest.mock.call(primary_model, backend=fake_backend, dynamic=False, fullgraph=True),
            unittest.mock.call(cache_model, backend=fake_backend, dynamic=False, fullgraph=True),
        ]
        primary_model.forward.assert_called_once()
        self.assertEqual(cache_model.forward.call_count, 2)
        mock_use_custom_sdpa.assert_called_once()
        self.assertEqual(
            mock_use_custom_sdpa.call_args.kwargs["route_plan"].backend,
            AttentionBackend.dense,
        )
        self.assertFalse(hasattr(primary_model_config.transformer_config, "attention_route_plan"))
        self.assertFalse(hasattr(cache_model_config.transformer_config, "attention_route_plan"))

    def test_diffusers_transformer_config_does_not_own_attention_route_plan(self):
        from tensor_cast.model_config import DiffusersTransformerConfig, ParallelConfig

        config = DiffusersTransformerConfig(
            parallel_config=ParallelConfig(),
            quant_config=MagicMock(),
        )

        self.assertFalse(hasattr(config, "attention_route_plan"))

    def test_unrelated_diffusers_import_does_not_register_attention_backend(self):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import tensor_cast.diffusers.dit_cache_registry; "
                    "print('tensor_cast.diffusers.diffusers_attention' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_wan_attention_uses_upstream_processor_with_tensor_cast_backend(self):
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor, WanTransformer3DModel

        transformer = WanTransformer3DModel(
            patch_size=(1, 1, 1),
            num_attention_heads=2,
            attention_head_dim=2,
            in_channels=4,
            out_channels=4,
            text_dim=4,
            freq_dim=4,
            ffn_dim=8,
            num_layers=1,
        )
        model = object.__new__(DiffusersTransformerModel)
        torch.nn.Module.__init__(model)
        model._inner = transformer

        wrap_model(model)

        wan_attention = next(module for module in transformer.modules() if isinstance(module, WanAttention))
        assert isinstance(wan_attention.processor, WanAttnProcessor)
        assert wan_attention.processor._attention_backend == "tensor_cast"

    def test_wan_upstream_processor_given_text_context_then_dispatches_once(self):
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

        attn = WanAttention(
            dim=4,
            heads=2,
            dim_head=2,
            cross_attention_dim_head=2,
            processor=WanAttnProcessor(),
        )
        attn.processor._attention_backend = "tensor_cast"
        hidden_states = torch.zeros((1, 3, 4))
        encoder_hidden_states = torch.zeros((1, 5, 4))

        with patch(
            "diffusers.models.transformers.transformer_wan.dispatch_attention_fn",
            side_effect=lambda query, *args, **kwargs: query,
        ) as dispatch:
            output = attn(hidden_states, encoder_hidden_states)

        assert output.shape == hidden_states.shape
        dispatch.assert_called_once()
        assert dispatch.call_args.kwargs["backend"] == "tensor_cast"

    def test_wan_upstream_processor_given_image_context_then_dispatches_twice(self):
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

        attn = WanAttention(
            dim=4,
            heads=2,
            dim_head=2,
            cross_attention_dim_head=2,
            added_kv_proj_dim=4,
            processor=WanAttnProcessor(),
        )
        attn.processor._attention_backend = "tensor_cast"
        hidden_states = torch.zeros((1, 3, 4))
        encoder_hidden_states = torch.zeros((1, 514, 4))

        with patch(
            "diffusers.models.transformers.transformer_wan.dispatch_attention_fn",
            side_effect=lambda query, *args, **kwargs: query,
        ) as dispatch:
            output = attn(hidden_states, encoder_hidden_states)

        assert output.shape == hidden_states.shape
        assert dispatch.call_count == 2
        assert all(call.kwargs["backend"] == "tensor_cast" for call in dispatch.call_args_list)

    def test_custom_sdpa_given_fp8_attention_quantization_when_called_then_uses_quantized_attention_op(self):
        quant_config = create_attention_quant_config(QuantizeAttentionAction.FP8)
        query = torch.zeros((1, 2, 1, 4), device="meta", dtype=torch.float16)
        key = torch.zeros((1, 2, 1, 4), device="meta", dtype=torch.float16)
        value = torch.zeros((1, 2, 1, 4), device="meta", dtype=torch.float16)

        with (
            patch.object(torch.ops.tensor_cast, "attention_quant", return_value=query) as mock_attention_quant,
            patch.object(torch.ops.tensor_cast, "attention") as mock_attention,
            use_custom_sdpa(quant_config),
        ):
            torch.nn.functional.scaled_dot_product_attention(query, key, value)

        mock_attention_quant.assert_called_once()
        mock_attention.assert_not_called()

    def test_diffusers_attention_backend_given_fp8_attention_quantization_when_called_then_uses_quantized_attention_op(
        self,
    ):
        quant_config = create_attention_quant_config(QuantizeAttentionAction.FP8)
        query = torch.zeros((1, 2, 1, 4), device="meta", dtype=torch.float16)
        key = torch.zeros((1, 2, 1, 4), device="meta", dtype=torch.float16)
        value = torch.zeros((1, 2, 1, 4), device="meta", dtype=torch.float16)

        with (
            patch.object(torch.ops.tensor_cast, "attention_quant", return_value=query) as mock_attention_quant,
            patch.object(torch.ops.tensor_cast, "attention") as mock_attention,
            use_custom_sdpa(quant_config),
        ):
            _attention(query, key, value)

        mock_attention_quant.assert_called_once()
        mock_attention.assert_not_called()

    def test_dit_cache_requires_step_range(self):
        """Test dit_cache requires cache_step_range."""
        with self.assertRaises(ValueError):
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=2,
                dtype="float16",
                world_size=1,
                ulysses_size=1,
                dit_cache=True,
                cache_step_range=None,
            )

    def test_dit_cache_runs_with_ranges(self):
        """Test dit_cache runs with valid step/block ranges."""
        run_inference(
            device=self.device,
            model_id=self.model_id,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            height=self.height,
            width=self.width,
            frame_num=self.frame_num,
            sample_step=2,
            dtype="float16",
            world_size=1,
            ulysses_size=1,
            dit_cache=True,
            cache_step_range="0,1",
            cache_step_interval=1,
            cache_block_range="0,1",
        )
        self._validate_inference_result("test_dit_cache_runs_with_ranges")

    @parameterized.expand(
        [
            ("",),
            ("1",),
            ("1,",),
            (",2",),
            ("a,b",),
            ("-1,2",),
            ("3,-1",),
            ("5,3",),
        ]
    )
    def test_dit_cache_invalid_step_range(self, value):
        """Range parse errors should surface through run_inference."""
        with self.assertRaises(ValueError):
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=2,
                dtype="float16",
                world_size=1,
                ulysses_size=1,
                dit_cache=True,
                cache_step_range=value,
                cache_step_interval=1,
            )

    @parameterized.expand(
        [
            ("",),
            ("1",),
            ("1,",),
            (",2",),
            ("x,y",),
            ("-1,2",),
            ("5,3",),
        ]
    )
    def test_dit_cache_invalid_block_range(self, value):
        """Invalid block range should be rejected when cache is enabled."""
        with self.assertRaises(ValueError):
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=2,
                dtype="float16",
                world_size=1,
                ulysses_size=1,
                dit_cache=True,
                cache_step_range="0,1",
                cache_step_interval=1,
                cache_block_range=value,
            )

    @parameterized.expand(
        [
            # Test combinations: (use_cfg, cfg_parallel, world_size, test description)
            (False, False, 1, "CFG disabled + parallel disabled → no extra operations"),
            (True, False, 1, "CFG enabled + parallel disabled → execute extra forward"),
            (True, True, 2, "CFG enabled + parallel enabled → execute cfg all_gather"),
            (False, True, 2, "CFG disabled + parallel enabled → no extra operations"),
        ]
    )
    def test_classifier_free_guidance_parallel(self, use_cfg, cfg_parallel, world_size, test_desc):
        """Test basic video inference without Ulysses parallelism."""
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                dtype="float16",
                world_size=world_size,
                ulysses_size=1,
                use_cfg=use_cfg,
                cfg_parallel=cfg_parallel,
            )
            self._validate_inference_result(f"test_classifier_free_guidance_parallel {test_desc}")
        except Exception as e:
            self.fail(f"test_classifier_free_guidance_parallel {test_desc} failed with exception: {e!s}")

    def test_process_input_with_ulysses_size_1(self):
        """Test process_input function when ulysses_size is 1."""
        # Mock model_config
        mock_parallel_config = MagicMock()
        mock_parallel_config.ulysses_size = 1

        mock_transformer_config = MagicMock()
        mock_transformer_config.parallel_config = mock_parallel_config

        mock_model_config = MagicMock()
        mock_model_config.transformer_config = mock_transformer_config

        input_kwargs = {
            "hidden_states": torch.randn(2, 10, 16, 10, 25)  # Example tensor
        }

        result_kwargs, split_dim = process_input(input_kwargs, mock_model_config)

        # When ulysses_size is 1, input should remain unchanged
        self.assertEqual(result_kwargs, input_kwargs)
        self.assertIsNone(split_dim)

    @parameterized.expand(
        [
            ["float16"],
            ["float32"],
            ["bfloat16"],
        ]
    )
    def test_video_inference_with_different_dtypes_param(self, dtype):
        """Parameterized test for different data types."""
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                dtype=dtype,
                world_size=1,
                ulysses_size=1,
            )
            self._validate_inference_result(f"test_video_inference_with_{dtype}_param")
        except Exception as e:
            self.fail(f"test_video_inference_with_{dtype}_param failed with exception: {e!s}")

    @parameterized.expand(
        [
            [1, 1],
            [2, 2],
            [4, 4],
            [8, 2],
        ]
    )
    def test_video_inference_with_different_parallel_sizes(self, world_size, ulysses_size):
        """Parameterized test for different parallel configurations."""
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                dtype="float16",
                world_size=world_size,
                ulysses_size=ulysses_size,
            )
            self._validate_inference_result(f"test_video_inference_with_world_{world_size}_ulysses_{ulysses_size}")
        except Exception as e:
            self.fail(
                f"test_video_inference_with_world_{world_size}_ulysses_{ulysses_size} failed with exception: {e!s}"
            )

    @parameterized.expand(
        [
            (
                "Hunyuanvideo",
                {
                    "_class_name": "HunyuanVideoTransformer3DModel",
                    "_diffusers_version": "0.32.0.dev0",
                    "attention_head_dim": 128,
                    "guidance_embeds": "true",
                    "in_channels": 16,
                    "mlp_ratio": 4.0,
                    "num_attention_heads": 24,
                    "num_layers": 20,
                    "num_refiner_layers": 2,
                    "num_single_layers": 40,
                    "out_channels": 16,
                    "patch_size": 2,
                    "patch_size_t": 1,
                    "pooled_projection_dim": 768,
                    "qk_norm": "rms_norm",
                    "rope_axes_dim": [16, 56, 56],
                    "rope_theta": 256.0,
                    "text_embed_dim": 4096,
                },
                {
                    "_class_name": "AutoencoderKLCogVideoX",
                    "in_channels": 3,
                    "out_channels": 3,
                    "down_block_types": [
                        "CogVideoXDownBlock3D",
                        "CogVideoXDownBlock3D",
                        "CogVideoXDownBlock3D",
                    ],
                    "up_block_types": [
                        "CogVideoXUpBlock3D",
                        "CogVideoXUpBlock3D",
                        "CogVideoXUpBlock3D",
                    ],
                    "block_out_channels": [128, 256, 512],
                    "layers_per_block": 4,
                    "act_fn": "silu",
                    "sample_size": [16, 128, 128],
                    "mid_block_type": "CogVideoXMidBlock3D",
                    "norm_num_groups": 32,
                    "temporal_compression_ratio": 4,
                    "z_dim": 16,
                },
            ),
            (
                "WAN",
                {
                    "_class_name": "WanTransformer3DModel",
                    "_diffusers_version": "0.35.0.dev0",
                    "added_kv_proj_dim": None,
                    "attention_head_dim": 128,
                    "cross_attn_norm": True,
                    "eps": 1e-06,
                    "ffn_dim": 13824,
                    "freq_dim": 256,
                    "image_dim": None,
                    "in_channels": 16,
                    "num_attention_heads": 40,
                    "num_layers": 40,
                    "out_channels": 16,
                    "patch_size": [1, 2, 2],
                    "pos_embed_seq_len": None,
                    "qk_norm": "rms_norm_across_heads",
                    "rope_max_seq_len": 1024,
                    "text_dim": 4096,
                },
                {
                    "_class_name": "AutoencoderKLWan",
                    "_diffusers_version": "0.35.0.dev0",
                    "attn_scales": [],
                    "base_dim": 96,
                    "dim_mult": [1, 2, 4, 4],
                    "dropout": 0.0,
                    "latents_mean": [
                        -0.7571,
                        -0.7089,
                        -0.9113,
                        0.1075,
                        -0.1745,
                        0.9653,
                        -0.1517,
                        1.5508,
                        0.4134,
                        -0.0715,
                        0.5517,
                        -0.3632,
                        -0.1922,
                        -0.9497,
                        0.2503,
                        -0.2921,
                    ],
                    "latents_std": [
                        2.8184,
                        1.4541,
                        2.3275,
                        2.6558,
                        1.2196,
                        1.7708,
                        2.6052,
                        2.0743,
                        3.2687,
                        2.1526,
                        2.8652,
                        1.5579,
                        1.6382,
                        1.1253,
                        2.8251,
                        1.916,
                    ],
                    "num_res_blocks": 2,
                    "temperal_downsample": [False, True, True],
                    "z_dim": 16,
                },
            ),
            (
                "hunyuan_video15",
                {
                    "_class_name": "HunyuanVideo15Transformer3DModel",
                    "_diffusers_version": "0.36.0.dev0",
                    "attention_head_dim": 128,
                    "image_embed_dim": 1152,
                    "in_channels": 65,
                    "mlp_ratio": 4.0,
                    "num_attention_heads": 16,
                    "num_layers": 54,
                    "num_refiner_layers": 2,
                    "out_channels": 32,
                    "patch_size": 1,
                    "patch_size_t": 1,
                    "qk_norm": "rms_norm",
                    "rope_axes_dim": [16, 56, 56],
                    "rope_theta": 256.0,
                    "target_size": 640,
                    "task_type": "t2v",
                    "text_embed_2_dim": 1472,
                    "text_embed_dim": 3584,
                    "use_meanflow": False,
                },
                {
                    "_class_name": "AutoencoderKLHunyuanVideo15",
                    "_diffusers_version": "0.36.0.dev0",
                    "block_out_channels": [128, 256, 512, 1024, 1024],
                    "downsample_match_channel": True,
                    "in_channels": 3,
                    "latent_channels": 32,
                    "layers_per_block": 2,
                    "out_channels": 3,
                    "scaling_factor": 1.03682,
                    "spatial_compression_ratio": 16,
                    "temporal_compression_ratio": 4,
                    "upsample_match_channel": True,
                },
            ),
        ]
    )
    def test_video_inference_with_model_configs(self, config_name, transformer_config, vae_config):
        temp_dir, model_dir = self._create_mock_model_dir(transformer_config, vae_config)
        try:
            run_inference(
                device="TEST_DEVICE",
                model_id=model_dir,
                batch_size=2,
                seq_len=10,
                height=800,
                width=600,
                frame_num=121,
                sample_step=1,
                dtype="float16",
                world_size=1,
                ulysses_size=1,
                quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            )
            self._validate_inference_result(f"test_video_inference_with_model_configs[{config_name}]")
        except Exception as e:
            self.fail(f"test_video_inference_with_model_configs[{config_name}] failed with exception: {e!s}")
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_video_inference_with_raw_tencent_hunyuanvideo15_t2v_selector(self):
        temp_dir = tempfile.mkdtemp(dir=os.path.realpath(os.getcwd()))
        snapshot_root = os.path.join(temp_dir, "snapshot")
        variant_dir = os.path.join(snapshot_root, "transformer", "480p_t2v_distilled")
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(snapshot_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "_class_name": "HunyuanVideo_1_5_Pipeline",
                    "transformer": [
                        "hyvideo.models.transformers.hunyuanvideo_1_5_transformer",
                        "HunyuanVideo_1_5_DiffusionTransformer",
                    ],
                    "vision_num_semantic_tokens": 729,
                    "vision_states_dim": 1152,
                },
                f,
            )
        with open(os.path.join(variant_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "_class_name": "HunyuanVideo_1_5_DiffusionTransformer",
                    "concat_condition": True,
                    "glyph_byT5_v2": True,
                    "guidance_embed": False,
                    "heads_num": 16,
                    "hidden_size": 2048,
                    "ideal_resolution": "480p",
                    "ideal_task": "t2v",
                    "in_channels": 32,
                    "is_reshape_temporal_channels": False,
                    "mlp_act_type": "gelu_tanh",
                    "mlp_width_ratio": 4,
                    "mm_double_blocks_depth": 54,
                    "mm_single_blocks_depth": 0,
                    "out_channels": 32,
                    "patch_size": [1, 1, 1],
                    "qk_norm": True,
                    "qk_norm_type": "rms",
                    "qkv_bias": True,
                    "rope_dim_list": [16, 56, 56],
                    "rope_theta": 256,
                    "text_pool_type": None,
                    "text_projection": "single_refiner",
                    "text_states_dim": 3584,
                    "text_states_dim_2": None,
                    "use_attention_mask": True,
                    "use_cond_type_embedding": True,
                    "use_meanflow": False,
                    "vision_projection": "linear",
                    "vision_states_dim": 1152,
                },
                f,
            )
        captured_runtimes = []

        class CapturingRuntime(Runtime):
            def __enter__(self):
                captured_runtimes.append(self)
                return super().__enter__()

        try:
            with (
                patch(
                    "tensor_cast.diffusers.model_resolver.snapshot_huggingface_config_only",
                    return_value=snapshot_root,
                ),
                patch("cli.inference.video_generate.Runtime", CapturingRuntime),
            ):
                for compile_model in (False, True):
                    run_inference(
                        device="TEST_DEVICE",
                        model_id="tencent/HunyuanVideo-1.5/transformer/480p_t2v_distilled",
                        batch_size=1,
                        seq_len=128,
                        height=480,
                        width=832,
                        frame_num=121,
                        sample_step=1,
                        dtype="float16",
                        world_size=1,
                        ulysses_size=1,
                        attention_backend=AttentionBackend.block_sparse_attention,
                        compile=compile_model,
                    )
                    event_funcs = {str(event.op_invoke_info.func) for event in captured_runtimes[-1].event_list}
                    self.assertIn(str(torch.ops.tensor_cast.attention_route_generate.default), event_funcs)
                    self.assertIn(str(torch.ops.tensor_cast.block_sparse_attention.default), event_funcs)
            self._validate_inference_result("test_video_inference_with_raw_tencent_hunyuanvideo15_t2v_selector")
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def _make_cache_wrapped_forward(agent):
    def factory(orig_forward):
        def wrapped(_self, hidden_states, encoder_hidden_states=None, scale=1):
            return agent.apply(
                orig_forward,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                scale=scale,
            )

        return wrapped

    return factory


class _CacheBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = "inner"

    def forward(self, hidden_states, encoder_hidden_states=None, scale=1):
        hidden = hidden_states + scale
        if encoder_hidden_states is None:
            return hidden
        return hidden, encoder_hidden_states + scale


def test_dit_block_cache_update_and_reuse_paths():
    state = CacheState()
    block = DiTBlockCache(
        _CacheBlock(),
        state,
        block_index=0,
        block_start=0,
        block_end=1,
        make_wrapped_forward=_make_cache_wrapped_forward,
    )
    hidden = torch.ones(2, 2)
    encoder = torch.full((2, 2), 2.0)

    assert block.marker == "inner"
    first_hidden, first_encoder = block(hidden, encoder, scale=3)
    assert torch.equal(first_hidden, hidden + 3)
    assert torch.equal(first_encoder, encoder + 3)
    assert torch.equal(state.delta_hidden, torch.full((2, 2), 3.0))
    assert torch.equal(state.delta_encoder, torch.full((2, 2), 3.0))

    state.reuse = True
    reused_hidden, reused_encoder = block(hidden, encoder)
    assert torch.equal(reused_hidden, hidden + 3)
    assert torch.equal(reused_encoder, encoder + 3)

    later = DiTBlockCache(_CacheBlock(), state, 1, 0, 2, _make_cache_wrapped_forward)
    assert torch.equal(later(hidden, encoder)[0], hidden)


def test_dit_block_cache_validates_error_paths():
    def make_wrapped_forward(agent):
        def factory(orig_forward):
            def wrapped(_self, **kwargs):
                return agent.apply(orig_forward, **kwargs)

            return wrapped

        return factory

    state = CacheState()
    block = DiTBlockCache(torch.nn.Identity(), state, 0, 0, 1, make_wrapped_forward)
    with pytest.raises(ValueError, match="hidden_states"):
        block()

    state.reuse = True
    with pytest.raises(RuntimeError, match="Cache delta is empty"):
        block(hidden_states=torch.ones(1))

    state.delta_hidden = torch.ones(1)
    state.delta_encoder = torch.ones(1)
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        block(hidden_states=torch.ones(1))


def test_dit_cache_registry_helpers_replace_and_select_blocks():
    blocks = [torch.nn.Identity(), torch.nn.ReLU(), torch.nn.Sigmoid()]
    pairs = _module_list_blocks_with_setters(blocks)

    replaced = replace_blocks_in_range(pairs, 1, 10, lambda block, idx: (idx, block))

    assert replaced == 2
    assert isinstance(blocks[0], torch.nn.Identity)
    assert blocks[1][0] == 1
    assert blocks[2][0] == 2

    spec = DiTBlockCacheSpec("Example", lambda inner: [], lambda agent: lambda forward: forward)
    register_dit_block_cache_spec("ExampleTransformer", spec)
    assert get_dit_block_cache_spec("ExampleTransformer") is spec
    assert get_dit_block_cache_spec("") is None

    wan = types.SimpleNamespace(blocks=[torch.nn.Identity()])
    assert len(_get_wan_blocks_with_setters(wan)) == 1
    assert _get_wan_blocks_with_setters(types.SimpleNamespace()) == []

    hunyuan = types.SimpleNamespace(
        transformer_blocks=[torch.nn.Identity()],
        single_transformer_blocks=[torch.nn.ReLU()],
    )
    assert len(_get_hunyuanvideo_blocks_with_setters(hunyuan)) == 2

    hunyuan15 = types.SimpleNamespace(transformer_blocks=[torch.nn.Identity()])
    assert len(_get_hunyuanvideo15_blocks_with_setters(hunyuan15)) == 1
    assert _get_hunyuanvideo15_blocks_with_setters(types.SimpleNamespace()) == []


class TestCliVideoGenerateMain(unittest.TestCase):
    """Coverage anchor for cli.inference.video_generate.main."""

    def test_main_forwards_arguments_into_run_inference(self):
        captured: dict[str, object] = {}

        def fake_run_inference(**kwargs: object) -> None:
            captured.update(kwargs)

        with patch("cli.inference.video_generate.run_inference", fake_run_inference):
            result = run_module_main(
                "cli.inference.video_generate",
                [
                    "--device",
                    "TEST_DEVICE",
                    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "128",
                    "--quantize-linear-action",
                    "DISABLED",
                    "--sample-step",
                    "2",
                    "--attention-backend",
                    "block_sparse_attention",
                ],
            )

        assert result.returncode == 0, result.stderr
        assert captured["model_id"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        assert captured["device"] == "TEST_DEVICE"
        assert captured["batch_size"] == 1
        assert captured["seq_len"] == 128
        assert captured["sample_step"] == 2
        assert captured["remote_source"] == "huggingface"
        assert captured["attention_backend"] is AttentionBackend.block_sparse_attention
        assert "trace" not in captured

    @parameterized.expand(["--trace", "--trace-ops"])
    def test_main_rejects_removed_trace_flags(self, removed_flag):
        with patch("cli.inference.video_generate.run_inference") as mock_run_inference:
            result = run_module_main(
                "cli.inference.video_generate",
                [
                    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "128",
                    removed_flag,
                ],
            )

        assert result.returncode == 2
        assert f"unrecognized arguments: {removed_flag}" in result.stderr
        mock_run_inference.assert_not_called()


class TestCliVideoGenerateRunInference(unittest.TestCase):
    """Coverage anchor for cli.inference.video_generate.run_inference."""

    def test_cfg_batch_concat_path_doubles_batch_dimension(self):
        from cli.inference import video_generate as video_generate_mod

        captured: dict[str, object] = {}

        class DummyRuntime:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __enter__(self) -> "DummyRuntime":
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def table_averages(self, *args: object, **kwargs: object) -> str:
                return "runtime table"

        class DummyModel:
            sp_group = None

            def forward(self, **kwargs: object) -> torch.Tensor:
                captured.setdefault("forward_batch_shapes", []).append(kwargs["hidden_states"].shape[0])
                return torch.zeros([1], device="meta")

        model_config = types.SimpleNamespace(
            transformer_config=types.SimpleNamespace(
                parallel_config=types.SimpleNamespace(ulysses_size=1),
                model_config={"_class_name": "WanTransformer3DModel"},
                dtype=torch.float16,
            )
        )

        def fake_build_diffusers_transformer_model(*args: object, **kwargs: object) -> tuple[DummyModel, object]:
            return DummyModel(), model_config

        model_selection = DiffusersModelSelection(
            repository_root="/cache/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            variant_path="/cache/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            variant_id=None,
            source=None,
            is_remote=True,
        )

        with (
            patch.object(video_generate_mod, "AnalyticPerformanceModel", lambda device_profile: object()),
            patch.object(video_generate_mod, "MemoryTracker", lambda device_profile: object()),
            patch.object(video_generate_mod, "Runtime", DummyRuntime),
            patch.object(
                video_generate_mod,
                "generate_diffusers_inputs",
                lambda *args, **kwargs: {"hidden_states": torch.zeros([1, 3], device="meta")},
            ),
            patch.object(
                video_generate_mod,
                "process_input",
                lambda input_kwargs, model_config: (input_kwargs, None),
            ),
            patch.dict(
                sys.modules,
                {
                    "tensor_cast.diffusers.diffusers_attention": types.SimpleNamespace(
                        get_sp_group=lambda: None,
                        set_sp_group=lambda group: None,
                        use_custom_sdpa=lambda *args, **kwargs: contextlib.nullcontext(),
                    ),
                    "tensor_cast.diffusers.diffusers_model": types.SimpleNamespace(
                        build_diffusers_transformer_model=fake_build_diffusers_transformer_model
                    ),
                    "tensor_cast.diffusers.model_resolver": types.SimpleNamespace(
                        resolve_diffusers_model_selection=lambda model_id, remote_source: model_selection
                    ),
                },
            ),
        ):
            video_generate_mod.run_inference(
                device="TEST_DEVICE",
                model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                batch_size=1,
                seq_len=128,
                sample_step=1,
                use_cfg=True,
                cfg_parallel=False,
                quantize_linear_action=QuantizeLinearAction.DISABLED,
            )

        assert captured["forward_batch_shapes"] == [2]

    def test_hunyuanvideo15_registry_helpers_select_blocks(self):
        hunyuan15 = types.SimpleNamespace(transformer_blocks=[torch.nn.Identity()])
        assert len(_get_hunyuanvideo15_blocks_with_setters(hunyuan15)) == 1
        assert _get_hunyuanvideo15_blocks_with_setters(types.SimpleNamespace()) == []


if __name__ == "__main__":
    unittest.main()
