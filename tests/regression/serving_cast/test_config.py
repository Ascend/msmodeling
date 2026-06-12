# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import tempfile
import unittest

import yaml
from serving_cast.config import (
    CommonConfig,
    CommunicationConfig,
    Config,
    InstanceConfig,
    LoadGenConfig,
    ModelConfig,
    ParallelConfig,
    ServingConfig,
)


class TestParallelConfig(unittest.TestCase):
    def test_custom_values(self):
        """Test ParallelConfig with custom values."""
        config = ParallelConfig(
            world_size=8,
            tp_size=4,
            dp_size=2,
            ep_size=8,
            mlp_tp_size=2,
            mlp_dp_size=2,
        )
        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.tp_size, 4)
        self.assertEqual(config.dp_size, 2)
        self.assertEqual(config.ep_size, 8)
        self.assertEqual(config.mlp_tp_size, 2)
        self.assertEqual(config.mlp_dp_size, 2)


class TestCommunicationConfig(unittest.TestCase):
    def test_default_values(self):
        """Test CommunicationConfig default values."""
        config = CommunicationConfig()
        self.assertEqual(config.host2device_bandwidth, 1e10)
        self.assertEqual(config.host2device_rate, 0.5)
        self.assertEqual(config.device2device_bandwidth, 4e9)
        self.assertEqual(config.device2device_rate, 0.5)

    def test_custom_values(self):
        """Test CommunicationConfig with custom values."""
        config = CommunicationConfig(
            host2device_bandwidth=1e9,
            host2device_rate=0.8,
            device2device_bandwidth=1e10,
            device2device_rate=0.9,
        )
        self.assertEqual(config.host2device_bandwidth, 1e9)
        self.assertEqual(config.host2device_rate, 0.8)
        self.assertEqual(config.device2device_bandwidth, 1e10)
        self.assertEqual(config.device2device_rate, 0.9)


class TestInstanceConfig(unittest.TestCase):
    def test_instance_config_creation(self):
        """Test InstanceConfig creation."""
        config = InstanceConfig(
            num_instances=4,
            num_devices_per_instance=8,
            pd_role="prefill",
            parallel_config=ParallelConfig(),
            communication_config=CommunicationConfig(),
        )
        self.assertEqual(config.num_instances, 4)
        self.assertEqual(config.num_devices_per_instance, 8)
        self.assertEqual(config.pd_role, "prefill")
        self.assertEqual(config.device_type, "TEST_DEVICE")


class TestLoadGenConfig(unittest.TestCase):
    def test_load_gen_config_creation(self):
        """Test LoadGenConfig creation."""
        config = LoadGenConfig(
            load_gen_type="fixed_length",
            num_requests=100,
            num_input_tokens=1000,
            num_output_tokens=100,
            request_rate=1.0,
        )
        self.assertEqual(config.load_gen_type, "fixed_length")
        self.assertEqual(config.num_requests, 100)
        self.assertEqual(config.num_input_tokens, 1000)
        self.assertEqual(config.num_output_tokens, 100)
        self.assertEqual(config.request_rate, 1.0)


class TestServingConfig(unittest.TestCase):
    def test_default_values(self):
        """Test ServingConfig default values."""
        config = ServingConfig()
        self.assertEqual(config.max_concurrency, 100)
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.max_tokens_budget, 8192)

    def test_custom_values(self):
        """Test ServingConfig with custom values."""
        config = ServingConfig(
            max_concurrency=200,
            block_size=256,
            max_tokens_budget=16384,
        )
        self.assertEqual(config.max_concurrency, 200)
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.max_tokens_budget, 16384)


class TestModelConfig(unittest.TestCase):
    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(name="test-model")
        self.assertEqual(config.name, "test-model")
        self.assertEqual(config.num_mtp_tokens, 0)
        self.assertFalse(config.do_compile)
        self.assertFalse(config.allow_graph_break)
        self.assertFalse(config.dump_input_shapes)
        self.assertEqual(config.quantize_linear_action, "W8A8_DYNAMIC")
        self.assertFalse(config.quantize_lmhead)
        self.assertEqual(config.mxfp4_group_size, 32)
        self.assertEqual(config.quantize_attention_action, "DISABLED")

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            name="custom-model",
            num_mtp_tokens=4,
            do_compile=True,
            allow_graph_break=True,
            quantize_linear_action="FP8",
            quantize_lmhead=True,
            enable_multi_process=True,
            num_processes=8,
            predict_steps=10,
            enable_interpolate=False,
            interpolation_seed=42,
        )
        self.assertEqual(config.name, "custom-model")
        self.assertEqual(config.num_mtp_tokens, 4)
        self.assertTrue(config.do_compile)
        self.assertTrue(config.allow_graph_break)
        self.assertEqual(config.quantize_linear_action, "FP8")
        self.assertTrue(config.quantize_lmhead)
        self.assertTrue(config.enable_multi_process)
        self.assertEqual(config.num_processes, 8)
        self.assertEqual(config.predict_steps, 10)
        self.assertFalse(config.enable_interpolate)
        self.assertEqual(config.interpolation_seed, 42)


class TestCommonConfig(unittest.TestCase):
    def test_common_config_creation(self):
        """Test CommonConfig creation."""
        model_config = ModelConfig(name="test-model")
        load_gen_config = LoadGenConfig(
            load_gen_type="fixed_length",
            num_requests=100,
            num_input_tokens=1000,
            num_output_tokens=100,
            request_rate=1.0,
        )
        serving_config = ServingConfig()

        config = CommonConfig(
            model_config=model_config,
            load_gen=load_gen_config,
            serving_config=serving_config,
        )
        self.assertEqual(config.model_config, model_config)
        self.assertEqual(config.load_gen, load_gen_config)
        self.assertEqual(config.serving_config, serving_config)


class TestConfig(unittest.TestCase):
    def setUp(self):
        """Reset Config singleton before each test."""
        Config._instance = None
        Config._initialized = False

    def test_config_get_instance_not_initialized(self):
        """Test that get_instance raises error when not initialized."""
        with self.assertRaises(ValueError):
            Config.get_instance()

    def test_config_singleton(self):
        """Test that Config is a singleton."""
        # Create temporary YAML files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create instance config
            instance_config = {
                "instance_groups": [
                    {
                        "num_instances": 1,
                        "num_devices_per_instance": 4,
                        "pd_role": "both",
                    }
                ]
            }
            instance_path = os.path.join(tmpdir, "instance.yaml")
            with open(instance_path, "w", encoding="utf-8") as f:
                yaml.dump(instance_config, f)

            # Create common config
            common_config = {
                "model_config": {"name": "test-model"},
                "load_gen": {
                    "load_gen_type": "fixed_length",
                    "num_requests": 10,
                    "num_input_tokens": 100,
                    "num_output_tokens": 10,
                    "request_rate": 1.0,
                },
            }
            common_path = os.path.join(tmpdir, "common.yaml")
            with open(common_path, "w", encoding="utf-8") as f:
                yaml.dump(common_config, f)

            # Create parsed_args
            class ParsedArgs:
                instance_config_path = instance_path
                common_config_path = common_path
                enable_profiling = False

            config1 = Config(ParsedArgs())
            config2 = Config(ParsedArgs())
            self.assertIs(config1, config2)

    def test_config_get_instance_after_init(self):
        """Test get_instance returns config after initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_config = {
                "instance_groups": [
                    {
                        "num_instances": 1,
                        "num_devices_per_instance": 4,
                        "pd_role": "both",
                    }
                ]
            }
            instance_path = os.path.join(tmpdir, "instance.yaml")
            with open(instance_path, "w", encoding="utf-8") as f:
                yaml.dump(instance_config, f)

            common_config = {
                "model_config": {"name": "test-model"},
                "load_gen": {
                    "load_gen_type": "fixed_length",
                    "num_requests": 10,
                    "num_input_tokens": 100,
                    "num_output_tokens": 10,
                    "request_rate": 1.0,
                },
            }
            common_path = os.path.join(tmpdir, "common.yaml")
            with open(common_path, "w", encoding="utf-8") as f:
                yaml.dump(common_config, f)

            class ParsedArgs:
                instance_config_path = instance_path
                common_config_path = common_path
                enable_profiling = False

            config = Config(ParsedArgs())
            self.assertEqual(Config.get_instance(), config)


if __name__ == "__main__":
    unittest.main()
