# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import patch

import pandas as pd
from serving_cast.service.disagg_throughput_optimizer import DisaggThroughputOptimizer
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import OptimizerData

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from .test_common import SimpleArgs


class TestDisaggStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.strategy = DisaggThroughputOptimizer()
        self.args = SimpleArgs()
        self.args.model_id = "Qwen/Qwen3-32B"
        self.args.num_devices = 4

        self.device_profiler = DeviceProfile.all_device_profiles[self.args.device]

        self.user_input = UserInputConfig.from_args(self.args)
        self.model_runner = ModelRunner(self.user_input)
        # Initialize strategy
        self.strategy.initialize(self.model_runner)

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.strategy.name, "disaggregation")

    def test_initialize_method(self):
        """Test initialize method sets up backend correctly"""
        self.assertEqual(self.strategy.model_runner, self.model_runner)
        self.assertEqual(self.strategy.dp, 4)
        self.assertEqual(self.strategy.tp, 1)
        self.assertEqual(self.strategy.pp, 1)

    def test_get_inference_info_decode_mode(self):
        """Test get_inference_info method in decode mode"""
        # data config for decode mode
        optimizer_data = OptimizerData(
            ttft_limits=None,  # Decode mode
            tpot_limits=50,
            batch_size=2,
            input_length=512,
            output_length=128,
            serving_cost=0,
            num_mtp_tokens=1,
            mtp_acceptance_rate=[0.9],
        )

        result = self.strategy.get_inference_info(optimizer_data)

        # Verify result is a Summary object
        self.assertIsInstance(result, OptimizerSummary)

        # Verify the summary data frame
        summary_df = result.get_summary_df()
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), 1)

        # Check key columns
        row = summary_df.iloc[0]
        self.assertEqual(row["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(row["input_length"], 512)
        self.assertEqual(row["output_length"], 128)
        self.assertIsNone(row["ttft"])
        self.assertEqual(row["concurrency"], 8)  # batch_size * dp * pp = 2 * 4 * 1 = 8
        self.assertEqual(row["device_name"], "TEST_DEVICE")
        self.assertEqual(row["parallel"], "TP=1 | PP=1 | DP=4")

    def test_get_inference_info_prefill_mode(self):
        """Test get_inference_info method in prefill mode"""
        # Mock data config for prefill mode
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=5,
            input_length=1024,
            output_length=50,
            serving_cost=0,
        )

        result = self.strategy.get_inference_info(optimizer_data)
        # Verify result is a Summary object
        self.assertIsInstance(result, OptimizerSummary)

        # Check key columns
        summary_df = result.get_summary_df()
        row = summary_df.iloc[0]
        self.assertEqual(row["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(row["input_length"], 1024)
        self.assertEqual(row["output_length"], 50)
        self.assertIsNone(row["tpot"])

    def test_prefix_cache_changes_prefill_shape_but_not_decode_shape(self):
        optimizer_data = OptimizerData(
            batch_size=2,
            input_length=200,
            output_length=32,
            prefix_cache_hit_rate=0.5,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        captured = []

        def fake_forward(concurrency, optimizer_data, is_decode):
            captured.append(
                (is_decode, optimizer_data.get_effective_input_length(is_decode))
            )

            class DummyMetrics:
                execution_time_s = {"analytic": 0.001}
                device_memory_available_gb = 1.0
                breakdowns = {}

            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            optimizer_data.ttft_limits = 1000
            optimizer_data.tpot_limits = None
            self.strategy.get_inference_info(optimizer_data)
            optimizer_data.ttft_limits = None
            optimizer_data.tpot_limits = 1000
            self.strategy.get_inference_info(optimizer_data)

        self.assertEqual(captured[0], (False, 100))
        self.assertEqual(captured[1], (True, 200))


if __name__ == "__main__":
    unittest.main()
