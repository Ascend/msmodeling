# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import AGG_COLUMNS, OptimizerData


class ConcreteThroughputOptimizer(BaseThroughputOptimizer):
    """Concrete implementation of BaseThroughputOptimizer for testing purposes"""

    def initialize(self, model):
        self.model = model

    def get_inference_info(self, optimizer_data):
        # Return a mock Summary object
        summary = Mock(spec=OptimizerSummary)
        summary.check_early_stop_flag.return_value = False
        summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)
        return summary


class TestBaseBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.backend = ConcreteThroughputOptimizer()
        self.mock_args = Mock()
        self.mock_args.batch_range = None
        self.mock_model = Mock()
        self.backend.initialize(self.mock_model)

        self.mock_data_config = Mock()
        self.mock_data_config.batch_size = 1
        self.mock_data_config.input_length = 128
        self.mock_data_config.output_length = 64
        self.mock_data_config.ttft_limits = 1000
        self.mock_data_config.tpot_limits = 200

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.backend.name, "base")

    @patch.object(ConcreteThroughputOptimizer, "get_inference_info")
    def test_optimizer_basic(self, mock_get_inference_info):
        """Test optimizer with basic scenario"""

        # Mock the run_inference method to return different stop flags
        def side_effect(data_config):
            summary = Mock(spec=OptimizerSummary)
            # Simulate the behavior: lower batch sizes don't stop, higher ones do
            if data_config.batch_size < 10:
                summary.check_early_stop_flag.return_value = False
            else:
                summary.check_early_stop_flag.return_value = True

            summary.get_summary_df.return_value = (
                pd.DataFrame(columns=AGG_COLUMNS, data=[[None] * len(AGG_COLUMNS)])
                if not summary.check_early_stop_flag.return_value
                else pd.DataFrame(columns=AGG_COLUMNS)
            )

            # Mock the get_summary_df to return proper data
            if not summary.check_early_stop_flag.return_value:
                mock_df = pd.DataFrame(
                    columns=AGG_COLUMNS,
                    data=[
                        [
                            "TEST_DEVICE",
                            1,
                            f"model_{data_config.batch_size}",
                            "DISABLED",
                            "DISABLED",
                            128,
                            64,
                            data_config.batch_size * 2,
                            100.0,
                            50.0,
                            1000.0,
                            500.0,
                            "tp1pp1dp1",
                            data_config.batch_size,
                            "prefill_breakdonws",
                            "decode_breakdowns",
                        ]
                    ],
                )
                summary.get_summary_df.return_value = mock_df
            else:
                summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)

            return summary

        mock_get_inference_info.side_effect = side_effect

        result = self.backend.run(self.mock_data_config, [5, 20])

        # Verify that run_inference was called
        self.assertGreater(mock_get_inference_info.call_count, 0)
        self.assertIsNotNone(result)

    @patch.object(ConcreteThroughputOptimizer, "get_inference_info")
    def test_optimizer_early_stop(self, mock_get_inference_info):
        """Test optimizer with early stop condition"""
        # Mock to always return stop flag
        mock_summary = Mock(spec=OptimizerSummary)
        mock_summary.check_early_stop_flag.return_value = True
        mock_get_inference_info.return_value = mock_summary

        result = self.backend.run(self.mock_data_config, None)

        # Should return None if early stop occurs
        self.assertIsNone(result)

    @patch.object(ConcreteThroughputOptimizer, "get_inference_info")
    def test_optimizer_no_results(self, mock_get_inference_info):
        """Test optimizer when no valid results found"""

        # Mock to return stop flag for all calls
        def side_effect(data_config):
            summary = Mock(spec=OptimizerSummary)
            summary.check_early_stop_flag.return_value = True
            summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)
            return summary

        mock_get_inference_info.side_effect = side_effect

        _ = self.backend.run(self.mock_data_config, None)

        mock_get_inference_info.assert_called()

    def test_abstract_methods_exist(self):
        """Test that abstract methods exist"""
        self.assertTrue(hasattr(BaseThroughputOptimizer, "initialize"))
        self.assertTrue(hasattr(BaseThroughputOptimizer, "get_inference_info"))

    def test_get_forward_info_uses_effective_input_length_for_prefill(self):
        self.backend.model_runner = Mock()
        self.backend.num_mtp_tokens = 0
        optimizer_data = OptimizerData(
            input_length=200,
            output_length=64,
            prefix_cache_hit_rate=0.5,
            batch_size=1,
        )

        self.backend._get_forward_info(4, optimizer_data, is_decode=False)

        requests = self.backend.model_runner.run_inference.call_args.args[0]
        self.assertEqual(requests[0].query_len, 100)
        self.assertEqual(requests[0].seq_len, 100)

    def test_get_forward_info_keeps_original_input_length_for_decode(self):
        self.backend.model_runner = Mock()
        self.backend.num_mtp_tokens = 0
        optimizer_data = OptimizerData(
            input_length=200,
            output_length=64,
            prefix_cache_hit_rate=0.5,
            batch_size=1,
        )

        self.backend._get_forward_info(4, optimizer_data, is_decode=True)

        requests = self.backend.model_runner.run_inference.call_args.args[0]
        self.assertEqual(requests[0].query_len, 1)
        self.assertEqual(requests[0].seq_len, 233)

    def test_get_forward_info_uses_explicit_image_batch_size_when_provided(self):
        self.backend.model_runner = Mock()
        optimizer_data = OptimizerData(
            input_length=32,
            output_length=64,
            batch_size=8,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        )

        self.backend._get_forward_info(8, optimizer_data, is_decode=False)

        requests = self.backend.model_runner.run_inference.call_args.args[0]
        self.assertEqual(requests[0].image_batch_size, 1)

    def test_get_forward_info_falls_back_to_batch_size_for_image_batch_size(self):
        self.backend.model_runner = Mock()
        optimizer_data = OptimizerData(
            input_length=32,
            output_length=64,
            batch_size=8,
            image_batch_size=None,
            image_height=1080,
            image_width=1920,
        )

        self.backend._get_forward_info(8, optimizer_data, is_decode=False)

        requests = self.backend.model_runner.run_inference.call_args.args[0]
        self.assertEqual(requests[0].image_batch_size, 8)

    def test_exponential_search_acc_search_clamps_right_boundary_on_early_stop(self):
        optimizer_data = OptimizerData(
            concurrency_search_strategy="linear_exponential", tpot_limits=50, ttft_limits=500
        )
        summary_left = Mock(spec=OptimizerSummary)
        summary_left.get_search_info.return_value = {"tpot": 10.0, "ttft": 100.0}

        summary_right = Mock(spec=OptimizerSummary)
        summary_right.check_early_stop_flag.return_value = True
        summary_right.get_search_info.return_value = {
            "per_request_memory_gb": 2.0,
            "device_memory_available_gb": -10.0,
            "tpot": 60.0,
            "ttft": 600.0,
        }

        with (
            patch.object(self.backend, "get_inference_info", return_value=summary_right),
            patch.object(self.backend, "_estimate_right_boundary", return_value=8) as mock_estimate,
        ):
            left, right = self.backend._exponential_search(optimizer_data, 1, 512, summary_left, True)

        self.assertEqual((left, right), (1, 8))
        mock_estimate.assert_called_once()

    def test_exponential_search_acc_search_uses_estimated_boundary_before_stop(self):
        optimizer_data = OptimizerData(
            concurrency_search_strategy="linear_exponential", tpot_limits=50, ttft_limits=500
        )
        summary_left = Mock(spec=OptimizerSummary)
        summary_left.get_search_info.return_value = {"tpot": 10.0, "ttft": 100.0}

        summary_right = Mock(spec=OptimizerSummary)
        summary_right.check_early_stop_flag.return_value = False
        summary_right.get_search_info.return_value = {
            "per_request_memory_gb": 0.5,
            "device_memory_available_gb": 10.0,
            "tpot": 40.0,
            "ttft": 400.0,
        }

        with (
            patch.object(self.backend, "get_inference_info", return_value=summary_right),
            patch.object(self.backend, "_estimate_right_boundary", return_value=530),
        ):
            left, right = self.backend._exponential_search(optimizer_data, 1, 512, summary_left, True)

        self.assertEqual((left, right), (1, 530))

    def test_compute_per_request_memory_gb_handles_zero_and_positive_batch_size(self):
        self.assertEqual(
            self.backend._compute_per_request_memory_gb(
                total_device_memory_gb=64,
                model_weight_size_gb=20,
                reserved_memory_gb=10,
                memory_left_gb=12,
                batch_size=0,
            ),
            0,
        )
        self.assertEqual(
            self.backend._compute_per_request_memory_gb(
                total_device_memory_gb=64,
                model_weight_size_gb=20,
                reserved_memory_gb=10,
                memory_left_gb=12,
                batch_size=4,
            ),
            5.5,
        )

    def test_estimate_right_boundary_falls_back_to_max_search_size(self):
        optimizer_data = OptimizerData(tpot_limits=None, ttft_limits=None)

        estimated = self.backend._estimate_right_boundary(
            {"batch_size": 1},
            {"batch_size": 8, "per_request_memory_gb": 0, "device_memory_available_gb": 0},
            optimizer_data,
        )
        self.assertEqual(estimated, 2**19 - 1)

    def test_estimate_right_boundary_uses_memory_limit_and_skips_fallback(self):
        optimizer_data = OptimizerData(tpot_limits=None, ttft_limits=None)

        estimated = self.backend._estimate_right_boundary(
            {"batch_size": 1},
            {
                "batch_size": 8,
                "per_request_memory_gb": 2.0,
                "device_memory_available_gb": 10.0,
            },
            optimizer_data,
        )

        self.assertEqual(estimated, 14)

    def test_exponential_search_without_acc_search_doubles_until_max_iterations(self):
        optimizer_data = OptimizerData(concurrency_search_strategy="exponential")
        summary_left = Mock(spec=OptimizerSummary)
        summary_right = Mock(spec=OptimizerSummary)
        summary_right.check_early_stop_flag.return_value = False

        with patch.object(self.backend, "get_inference_info", return_value=summary_right) as mock_get_inference_info:
            left, right = self.backend._exponential_search(optimizer_data, 1, 2, summary_left)

        self.assertEqual((left, right), (1024, 2048))
        self.assertEqual(mock_get_inference_info.call_count, 10)

    def test_estimate_by_latency_returns_relaxed_boundary_when_latency_grows(self):
        estimated = self.backend._estimate_by_latency(
            bs_left=2,
            bs_right=6,
            lat_left=10.0,
            lat_right=30.0,
            lat_limit=20.0,
            relax_factor=1.5,
            estimated_right=999,
        )

        self.assertEqual(estimated, 7)


if __name__ == "__main__":
    unittest.main()
