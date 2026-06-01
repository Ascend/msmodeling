# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import patch

import pandas as pd
from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
from serving_cast.service.utils import OptimizerData
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile

from .test_common import SimpleArgs


class TestAggThroughputOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.strategy = AggThroughputOptimizer()
        self.args = SimpleArgs()
        self.args.model_id = "Qwen/Qwen3-32B"

        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]

        self.user_input = UserInputConfig.from_args(self.args)
        self.model_runner = ModelRunner(self.user_input)
        # Initialize strategy
        self.strategy.initialize(self.model_runner)

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.strategy.name, "aggregation")

    def test_get_or_compute_prefill_latency_cached(self):
        """Test _get_or_compute_prefill_latency with cached value"""
        # Set up cache with a pre-computed value
        self.strategy._prefill_cache[4] = (50.0, 2.0, "")

        optimizer_data = OptimizerData()
        latency, memory_left, _ = self.strategy._get_or_compute_latency(4, optimizer_data, is_decode=False)

        # Should return cached value
        self.assertEqual(latency, 50.0)
        self.assertEqual(memory_left, 2.0)

    def test_get_or_compute_prefill_latency_new(self):
        """Test _get_or_compute_prefill_latency with new value"""
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=10,
        )
        latency, memory_left, breakdown = self.strategy._get_or_compute_latency(4, optimizer_data, is_decode=False)

        # Should cache the result
        self.assertEqual(
            self.strategy._prefill_cache[4],
            (latency, memory_left, breakdown),
        )

    def test_get_or_compute_decode_latency_cached(self):
        """Test _get_or_compute_decode_latency with cached value"""
        # Set up cache with a pre-computed value
        self.strategy._decode_cache[4] = (10.0, 2.0, "")

        optimizer_data = OptimizerData()
        latency, memory_left, _ = self.strategy._get_or_compute_latency(4, optimizer_data, is_decode=True)

        self.assertEqual(latency, 10.0)
        self.assertEqual(memory_left, 2.0)

    def test_get_inference_info_prefill_batch_size_uses_effective_input_length(self):
        optimizer_data = OptimizerData(
            input_length=200,
            output_length=10,
            batch_size=2,
            max_prefill_tokens=200,
            prefix_cache_hit_rate=0.5,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        captured_calls = []

        def fake_latency(batch_size, optimizer_data, is_decode=False):
            captured_calls.append((batch_size, is_decode))
            return (1.0, 1.0, "")

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            self.strategy.get_inference_info(optimizer_data)

        self.assertEqual(captured_calls[0], (2, False))

    def test_get_inference_info_uses_effective_prefill_memory_for_early_stop(self):
        optimizer_data = OptimizerData(
            input_length=32,
            output_length=256,
            batch_size=1,
            max_prefill_tokens=8192,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_latency(batch_size, optimizer_data, is_decode=False):
            if not is_decode and batch_size == 256:
                return (1000.0, -37.15, "wave")
            if not is_decode and batch_size == 1:
                return (342.0, 12.5, "effective")
            if is_decode and batch_size == 1:
                return (15.0, 9.0, "decode")
            raise AssertionError(f"unexpected call: batch_size={batch_size}, is_decode={is_decode}")

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            summary = self.strategy.get_inference_info(optimizer_data)

        self.assertFalse(summary.check_early_stop_flag())
        result_df = summary.get_summary_df()
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.iloc[0]["batch_size"], 1)

    def test_get_inference_info_checks_prefill_wave_memory_when_remainder_exists(self):
        optimizer_data = OptimizerData(
            input_length=32,
            output_length=256,
            batch_size=9,
            max_prefill_tokens=256,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_latency(batch_size, optimizer_data, is_decode=False):
            if not is_decode and batch_size == 8:
                return (1000.0, -37.15, "wave")
            if not is_decode and batch_size == 1:
                return (342.0, 12.5, "remainder")
            if is_decode and batch_size == 9:
                return (15.0, 9.0, "decode")
            raise AssertionError(f"unexpected call: batch_size={batch_size}, is_decode={is_decode}")

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            summary = self.strategy.get_inference_info(optimizer_data)

        self.assertTrue(summary.check_early_stop_flag())


if __name__ == "__main__":
    unittest.main()
