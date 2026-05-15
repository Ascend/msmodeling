# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""Unit tests for PD Ratio Throughput Optimizer."""

import unittest

import pandas as pd

from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.pd_ratio_throughput_optimizer import (
    PDRatioThroughputOptimizer,
)
from serving_cast.service.utils import OptimizerData


class TestArgs:
    """Simple args class for testing."""

    def __init__(self):
        self.model_id = "test-model"
        self.quantize_linear_action = "W8A8_DYNAMIC"
        self.quantize_attention_action = "DISABLED"
        self.device = "TEST_DEVICE"
        self.input_length = 1024
        self.output_length = 1024
        self.ttft_limits = 100
        self.tpot_limits = 10
        self.prefill_devices_per_instance = 4
        self.decode_devices_per_instance = 2
        self.num_devices = 8
        self.dump_original_results = False
        self.max_prefill_tokens = None
        self.batch_range = None
        self.tp_sizes = None
        self.image_height = None
        self.image_width = None
        self.compile = False


class TestPDRatioThroughputOptimizer(unittest.TestCase):
    """Test cases for PDRatioThroughputOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_length = 1024
        self.optimizer = PDRatioThroughputOptimizer(
            output_length=self.output_length,
        )

    def test_set_p_results(self):
        """Test setting prefill results DataFrame."""
        p_df = pd.DataFrame(
            {
                "ttft": [100.0, 80.0],
                "tpot": [0.0, 0.0],  # Prefill phase has tpot=0
                "concurrency": [10, 8],
                "parallel": ["tp4pp1dp1", "tp2pp1dp1"],
                "batch_size": [4, 4],
                "num_devices": [4, 2],
            }
        )
        self.optimizer.set_p_results(p_df)
        self.assertEqual(self.optimizer._p_df.shape[0], 2)

    def test_set_d_results(self):
        """Test setting decode results DataFrame."""
        d_df = pd.DataFrame(
            {
                "ttft": [0.0, 0.0],  # Decode phase has ttft=0
                "tpot": [10.0, 8.0],
                "concurrency": [8, 6],
                "parallel": ["tp2pp1dp1", "tp1pp1dp1"],
                "batch_size": [8, 6],
                "num_devices": [2, 1],
            }
        )
        self.optimizer.set_d_results(d_df)
        self.assertEqual(self.optimizer._d_df.shape[0], 2)

    def test_optimize(self):
        """Test the optimization process."""
        # Set prefill results (with tpot=0 since it's prefill phase)
        p_df = pd.DataFrame(
            {
                "ttft": [100.0],
                "tpot": [0.0],
                "concurrency": [10],
                "parallel": ["tp4pp1dp1"],
                "batch_size": [4],
                "num_devices": [4],
            }
        )
        self.optimizer.set_p_results(p_df)

        # Set decode results (with ttft=0 since it's decode phase)
        d_df = pd.DataFrame(
            {
                "ttft": [0.0],
                "tpot": [10.0],
                "concurrency": [8],
                "parallel": ["tp2pp1dp1"],
                "batch_size": [8],
                "num_devices": [2],
            }
        )
        self.optimizer.set_d_results(d_df)

        # Run optimization
        result_df = self.optimizer.optimize()

        # Verify results
        self.assertEqual(len(result_df), 1)

        # Verify QPS calculations
        # P QPS = 10 / 100 * 1000 = 100
        self.assertEqual(result_df.iloc[0]["p_qps"], 100.0)
        # D QPS = 8 / (10 * 1024) * 1000
        expected_d_qps = 8 / (10 * self.output_length) * 1000
        self.assertAlmostEqual(result_df.iloc[0]["d_qps"], expected_d_qps, places=5)

        # Verify column names have suffixes
        self.assertIn("ttft_p", result_df.columns)
        self.assertIn("tpot_d", result_df.columns)

    def test_optimize_multiple_results(self):
        """Test optimization with multiple P and D results."""
        # Set prefill results (with tpot=0)
        p_df = pd.DataFrame(
            {
                "ttft": [100.0, 80.0, 120.0],
                "tpot": [0.0, 0.0, 0.0],
                "concurrency": [10, 8, 12],
                "parallel": ["tp1pp1dp1", "tp2pp1dp1", "tp3pp1dp1"],
                "batch_size": [4, 5, 6],
                "num_devices": [1, 2, 3],
            }
        )
        self.optimizer.set_p_results(p_df)

        # Set decode results (with ttft=0)
        d_df = pd.DataFrame(
            {
                "ttft": [0.0, 0.0, 0.0],
                "tpot": [10.0, 8.0, 12.0],
                "concurrency": [8, 6, 10],
                "parallel": ["tp1pp1dp1", "tp2pp1dp1", "tp3pp1dp1"],
                "batch_size": [8, 9, 10],
                "num_devices": [1, 2, 3],
            }
        )
        self.optimizer.set_d_results(d_df)

        # Run optimization
        result_df = self.optimizer.optimize()

        # Should have 3*3 = 9 combinations
        self.assertEqual(len(result_df), 9)

        # Results should be sorted by balanced QPS in descending order
        for i in range(len(result_df) - 1):
            self.assertGreaterEqual(result_df.iloc[i]["balanced_qps"], result_df.iloc[i + 1]["balanced_qps"])

    def test_optimize_empty_p_results(self):
        """Test optimization with empty prefill results."""
        self.optimizer.set_p_results(pd.DataFrame())
        self.optimizer.set_d_results(pd.DataFrame({"ttft": [0.0], "tpot": [10.0]}))

        result_df = self.optimizer.optimize()
        self.assertTrue(result_df.empty)

    def test_optimize_empty_d_results(self):
        """Test optimization with empty decode results."""
        self.optimizer.set_p_results(pd.DataFrame({"ttft": [100.0], "tpot": [0.0]}))
        self.optimizer.set_d_results(pd.DataFrame())

        result_df = self.optimizer.optimize()
        self.assertTrue(result_df.empty)

    def test_format_output_empty(self):
        """Test format output with no results using OptimizerSummary in PD ratio mode."""
        pd_data_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=2,
            num_devices=8,
        )
        summary = OptimizerSummary(pd_data_config)
        summary.set_summary_df(self.optimizer.optimize())

        args = TestArgs()

        # Should not raise exception
        summary.report_final_result(args)

    def test_format_output_with_results(self):
        """Test format output with results using OptimizerSummary in PD ratio mode."""
        # Set prefill results (with tpot=0)
        p_df = pd.DataFrame(
            {
                "ttft": [100.0],
                "tpot": [0.0],
                "concurrency": [10],
                "parallel": ["tp4pp1dp1"],
                "batch_size": [4],
                "num_devices": [4],
            }
        )
        self.optimizer.set_p_results(p_df)

        # Set decode results (with ttft=0)
        d_df = pd.DataFrame(
            {
                "ttft": [0.0],
                "tpot": [10.0],
                "concurrency": [8],
                "parallel": ["tp2pp1dp1"],
                "batch_size": [8],
                "num_devices": [2],
            }
        )
        self.optimizer.set_d_results(d_df)

        self.optimizer.optimize()

        pd_data_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=2,
            num_devices=8,
        )
        summary = OptimizerSummary(pd_data_config)
        summary.set_summary_df(self.optimizer.optimize())

        args = TestArgs()

        # Should not raise exception
        summary.report_final_result(args)


class TestOptimizerSummaryPDMode(unittest.TestCase):
    """Test cases for OptimizerSummary PD ratio mode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_length = 1024
        self.pd_data_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=2,
            num_devices=8,
        )
        self.args = TestArgs()

    def _create_pd_result_df(self) -> pd.DataFrame:
        """Helper to create a sample PD ratio result DataFrame."""
        return pd.DataFrame(
            {
                "ttft_p": [100.0, 80.0],
                "tpot_d": [10.0, 8.0],
                "concurrency_p": [10, 8],
                "concurrency_d": [8, 6],
                "parallel_p": ["tp4pp1dp1", "tp2pp1dp1"],
                "parallel_d": ["tp2pp1dp1", "tp1pp1dp1"],
                "batch_size_p": [4, 5],
                "batch_size_d": [8, 6],
                "num_devices_p": [4, 2],
                "num_devices_d": [2, 1],
                "p_qps": [100.0, 100.0],
                "d_qps": [0.78125, 0.732421875],
                "pd_ratio": [0.0078125, 0.00732421875],
                "balanced_qps": [0.78125, 0.732421875],
            }
        )

    def test_is_pd_ratio_mode(self):
        """Test _is_pd_ratio_mode method detection."""
        summary = OptimizerSummary(self.pd_data_config)
        self.assertTrue(summary._is_pd_ratio_mode())

        # Test non-PD mode
        non_pd_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )
        summary_non_pd = OptimizerSummary(non_pd_config)
        self.assertFalse(summary_non_pd._is_pd_ratio_mode())

    def test_prepare_pd_ratio_results_deduplication(self):
        """Test _prepare_pd_ratio_results deduplication logic."""
        summary = OptimizerSummary(self.pd_data_config)

        # Create results with same parallel combination
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
        summary.set_summary_df(df)

        result = summary._prepare_pd_ratio_results()

        # Should deduplicate to 1 result
        self.assertEqual(len(result), 1)

    def test_calculate_instance_distribution(self):
        """Test _calculate_instance_distribution method."""
        summary = OptimizerSummary(self.pd_data_config)

        p_inst, d_inst = summary._calculate_instance_distribution(
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
        summary = OptimizerSummary(self.pd_data_config)
        df = self._create_pd_result_df()
        summary.set_summary_df(df)

        result = summary._get_pd_ratio_final_out(self.args, df)

        # Check for required sections
        result_str = "\n".join(result)
        self.assertIn("Overall Best Configuration:", result_str)
        self.assertIn("PD Ratio:", result_str)
        self.assertIn("Prefill QPS:", result_str)
        self.assertIn("Decode QPS:", result_str)
        self.assertIn("P Instances:", result_str)
        self.assertIn("D Instances:", result_str)

    def test_report_final_result_with_dump(self):
        """Test report_final_result with dump_original_results=True."""
        summary = OptimizerSummary(self.pd_data_config)
        df = self._create_pd_result_df()
        summary.set_summary_df(df)

        self.args.dump_original_results = True

        # Should not raise exception
        summary.report_final_result(self.args)

    def test_report_final_result_empty_df(self):
        """Test report_final_result with empty DataFrame."""
        summary = OptimizerSummary(self.pd_data_config)
        summary.set_summary_df(pd.DataFrame())

        # Should not raise exception, just log warning
        summary.report_final_result(self.args)


if __name__ == "__main__":
    unittest.main()
