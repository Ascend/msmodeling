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


if __name__ == "__main__":
    unittest.main()
