import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from cli.inference.video_generate import main, process_input, run_inference
from parameterized import parameterized
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction


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
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, "mock_model")
        os.makedirs(model_dir, exist_ok=True)

        transformer_dir = os.path.join(model_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(transformer_config, f)

        # Write VAE config
        vae_dir = os.path.join(model_dir, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        with open(os.path.join(vae_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vae_config, f)
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


if __name__ == "__main__":
    unittest.main()
