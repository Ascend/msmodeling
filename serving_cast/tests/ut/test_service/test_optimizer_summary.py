# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

import pandas as pd
from serving_cast.service.optimizer_summary import _get_agg_table_buf, OptimizerSummary
from serving_cast.service.utils import OptimizerData


class TestSummary(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.data_config = OptimizerData()
        self.data_config.ttft_limits = 1000.0
        self.data_config.tpot_limits = 50.0
        self.summary = OptimizerSummary(self.data_config)

    def test_initialization(self):
        """Test Summary initialization"""
        self.assertIsNone(self.summary._early_stop_flag)
        self.assertIsNone(self.summary._summary_df)
        self.assertEqual(self.summary.data_config, self.data_config)

    def test_set_and_get_summary_df(self):
        """Test setting and getting summary DataFrame"""
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        self.summary.set_summary_df(test_df)

        retrieved_df = self.summary.get_summary_df()
        pd.testing.assert_frame_equal(retrieved_df, test_df)

    def test_set_stop_flag_memory_negative(self):
        """Test set_stop_flag when memory_left is negative"""
        self.summary.set_early_stop_flag(memory_left=-1, tpot=10.0, ttft=100.0)
        self.assertTrue(self.summary.check_early_stop_flag())

    def test_set_stop_flag_tpot_exceeds_limit(self):
        """Test set_stop_flag when tpot exceeds limit"""
        self.summary.set_early_stop_flag(memory_left=10, tpot=60.0, ttft=100.0)  # 60 > 50 (limit)
        self.assertTrue(self.summary.check_early_stop_flag())

    def test_set_stop_flag_ttft_exceeds_limit(self):
        """Test set_stop_flag when ttft exceeds limit"""
        self.summary.set_early_stop_flag(memory_left=10, tpot=10.0, ttft=1500.0)  # 1500 > 1000 (limit)
        self.assertTrue(self.summary.check_early_stop_flag())

    def test_set_stop_flag_all_within_limits(self):
        """Test set_stop_flag when all values are within limits"""
        self.summary.set_early_stop_flag(memory_left=10, tpot=10.0, ttft=100.0)  # All within limits
        self.assertFalse(self.summary.check_early_stop_flag())

    def test_check_early_stop_flag_initial_state(self):
        """Test check_early_stop_flag initial state (should be None which evaluates to False)"""
        # Initially _stop_flag is None, which should evaluate to False
        flag = self.summary.check_early_stop_flag()
        self.assertIsNone(flag)

    def test_report_final_result_successful(self):
        """Test report_final_result with valid DataFrame"""
        # Create a test DataFrame with values within limits
        test_df = pd.DataFrame(
            {
                "token/s": [100.0, 80.0, 90.0],
                "ttft": [100.0, 200.0, 150.0],
                "tpot": [20.0, 30.0, 25.0],
                "concurrency": [1, 2, 1],
                "num_devices": [1, 1, 1],
                "parallel": [1, 1, 1],
                "batch_size": [1, 2, 1],
            }
        )
        self.summary.set_summary_df(test_df)

        class Args:
            model_id = "Qwen/Qwen3-8B"
            num_devices = 1
            device = "test_device"
            dump_original_results = False
            quantize_linear_action = "DISABLED"
            quantize_attention_action = "DISABLED"
            disagg = False

        args = Args()

        # Should not raise exception
        self.summary.report_final_result(args)


class TestGetAggTableBuf(unittest.TestCase):
    def test_get_agg_table_buf_with_different_parallel_values(self):
        """Test _get_agg_table_buf with different parallel values"""
        df = pd.DataFrame(
            {
                "token/s": [100.0, 80.0, 90.0, 110.0],
                "ttft": [100.0, 200.0, 150.0, 90.0],
                "tpot": [20.0, 30.0, 25.0, 18.0],
                "concurrency": [1, 2, 1, 1],
                "num_devices": [1, 1, 1, 1],
                "parallel": [1, 2, 1, 2],  # Different parallel values
                "batch_size": [1, 2, 1, 1],
            }
        )

        result = _get_agg_table_buf(df)

        # Should group by parallel and take first of each group, then sort by token/s
        self.assertIn("Top 4 Aggregation Configurations:", result)
        self.assertIn("Throughput", result)

    def test_get_agg_table_buf_single_row(self):
        """Test _get_agg_table_buf with single row DataFrame"""
        df = pd.DataFrame(
            {
                "token/s": [100.0],
                "ttft": [100.0],
                "tpot": [20.0],
                "concurrency": [1],
                "num_devices": [1],
                "parallel": [1],
                "batch_size": [1],
            }
        )

        result = _get_agg_table_buf(df)
        self.assertIn("Top 1 Aggregation Configurations:", result)
        self.assertIn("1", result)  # Top rank
        self.assertIn("100.00", result)  # Throughput value


class SimpleArgs:
    """Simple args class for testing without mock."""

    def __init__(self):
        self.model_id = "test_model"
        self.device = "TEST_DEVICE"
        self.quantize_linear_action = "W8A8_DYNAMIC"
        self.quantize_attention_action = "DISABLED"
        self.dump_original_results = False


class TestSummaryPDMode(unittest.TestCase):
    """Test cases for OptimizerSummary PD ratio mode."""

    def setUp(self):
        """Set up test fixtures for PD mode."""
        self.pd_data_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=2,
            num_devices=8,
        )
        self.summary = OptimizerSummary(self.pd_data_config)

    def test_is_pd_ratio_mode_true(self):
        """Test _is_pd_ratio_mode returns True for PD config."""
        self.assertTrue(self.summary._is_pd_ratio_mode())

    def test_is_pd_ratio_mode_false(self):
        """Test _is_pd_ratio_mode returns False for non-PD config."""
        non_pd_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )
        summary = OptimizerSummary(non_pd_config)
        self.assertFalse(summary._is_pd_ratio_mode())

    def test_prepare_pd_ratio_results_deduplication(self):
        """Test _prepare_pd_ratio_results deduplicates by parallel combination."""
        df = pd.DataFrame(
            {
                "ttft_p": [100.0, 100.0],
                "tpot_d": [10.0, 10.0],
                "concurrency_p": [10, 10],
                "concurrency_d": [8, 8],
                "parallel_p": ["tp4pp1dp1", "tp4pp1dp1"],
                "parallel_d": ["tp2pp1dp1", "tp2pp1dp1"],
                "batch_size_p": [4, 4],
                "batch_size_d": [8, 8],
                "num_devices_p": [4, 4],
                "num_devices_d": [2, 2],
                "p_qps": [100.0, 100.0],
                "d_qps": [0.78125, 0.78125],
                "pd_ratio": [0.0078125, 0.0078125],
                "balanced_qps": [0.78125, 0.78125],
            }
        )
        self.summary.set_summary_df(df)
        result = self.summary._prepare_pd_ratio_results()
        self.assertEqual(len(result), 1)

    def test_calculate_instance_distribution(self):
        """Test _calculate_instance_distribution calculation."""
        p_inst, d_inst = self.summary._calculate_instance_distribution(
            pd_ratio=1.0,
            total_devices=8,
            p_devices_per_inst=4,
            d_devices_per_inst=2,
        )
        self.assertGreater(p_inst, 0)
        self.assertGreater(d_inst, 0)
        self.assertLessEqual(p_inst * 4 + d_inst * 2, 8)

    def test_get_pd_ratio_final_out_structure(self):
        """Test _get_pd_ratio_final_out output structure."""
        df = pd.DataFrame(
            {
                "ttft_p": [100.0],
                "tpot_d": [10.0],
                "concurrency_p": [10],
                "concurrency_d": [8],
                "parallel_p": ["tp4pp1dp1"],
                "parallel_d": ["tp2pp1dp1"],
                "batch_size_p": [4],
                "batch_size_d": [8],
                "num_devices_p": [4],
                "num_devices_d": [2],
                "p_qps": [100.0],
                "d_qps": [0.78125],
                "pd_ratio": [0.0078125],
                "balanced_qps": [0.78125],
            }
        )
        self.summary.set_summary_df(df)

        args = SimpleArgs()
        result = self.summary._get_pd_ratio_final_out(args, df)
        result_str = "\n".join(result)
        self.assertIn("Overall Best Configuration:", result_str)
        self.assertIn("PD Ratio:", result_str)
        self.assertIn("Prefill QPS:", result_str)
        self.assertIn("Decode QPS:", result_str)

    def test_report_final_result_pd_mode_empty(self):
        """Test report_final_result in PD mode with empty DataFrame does not raise."""
        self.summary.set_summary_df(pd.DataFrame())
        args = SimpleArgs()
        # Should not raise exception
        self.summary.report_final_result(args)


if __name__ == "__main__":
    unittest.main()
