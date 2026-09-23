# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

import pandas as pd
from serving_cast.service.optimizer_summary import (
    EARLY_STOP_MEMORY_OOM,
    EARLY_STOP_PREFILL_OOM,
    EARLY_STOP_TPOT_LIMIT,
    EARLY_STOP_TTFT_LIMIT,
    OptimizerSummary,
    PP_RESULT_COLUMNS,
    _fmt_memory,
    _fmt_memory_info,
    _get_agg_disagg_table_buf_batched,
    _compute_disagg_request_qps,
    _get_agg_table_buf,
)
from serving_cast.service.pipeline_schedule import PipelineScheduleEstimate
from serving_cast.service.utils import OptimizerData, ParallelSearchCandidate


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

    def test_chunked_prefill_qps_uses_phase_makespan_not_average_ttft(self):
        row = pd.Series(
            {
                "concurrency": 5,
                "ttft": 25.0,
                "tpot": None,
                "prefill_phase_makespan_ms": 37.0,
            }
        )

        self.assertAlmostEqual(_compute_disagg_request_qps(row, output_length=None), 5 / 37 * 1000)

    def test_prefill_qps_does_not_fallback_when_phase_makespan_is_invalid(self):
        row = pd.Series(
            {
                "concurrency": 5,
                "ttft": 25.0,
                "tpot": None,
                "prefill_phase_makespan_ms": None,
            }
        )

        self.assertIsNone(_compute_disagg_request_qps(row, output_length=None))

    def test_prefill_qps_uses_ttft_for_legacy_rows_without_phase_makespan(self):
        row = pd.Series({"concurrency": 5, "ttft": 25.0, "tpot": None})

        self.assertAlmostEqual(_compute_disagg_request_qps(row, output_length=None), 5 / 25 * 1000)

    def test_set_and_get_summary_df(self):
        """Test setting and getting summary DataFrame"""
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        self.summary.set_summary_df(test_df)

        retrieved_df = self.summary.get_summary_df()
        pd.testing.assert_frame_equal(retrieved_df, test_df)

    def test_pp_result_columns_use_typed_context_not_parallel_display(self):
        candidate = ParallelSearchCandidate(
            tp_size=2,
            pp_size=2,
            ep_size=1,
            moe_dp_size=1,
            moe_tp_size=2,
            dp_size=2,
            num_mtp_tokens=0,
            layer_partition=(3, 5),
        )
        schedule = PipelineScheduleEstimate(
            makespan_s=0.012,
            first_completion_s=0.006,
            warmup_s=0.002,
            steady_s=0.008,
            cooldown_s=0.002,
            aggregate_bubble_s=0.003,
            bubble_ratio=0.125,
            bottleneck_stage_id=1,
            stage_busy_s=(0.011, 0.012),
            stage_idle_s=(0.001, 0.0),
            microbatch_completion_s=(0.006, 0.012),
            stage_compute_intervals_s=((), ()),
            stage_outgoing_transfer_intervals_s=((), ()),
            stage_incoming_transfer_intervals_s=((), ()),
            stage_incoming_payload_bytes_s=(0, 0),
            stage_outgoing_payload_bytes_s=(0, 0),
            stage_communication_buffer_bytes_s=(0, 0),
        )
        self.data_config.parallel_search_candidate = candidate
        self.data_config.pipeline_schedule_estimate = schedule
        self.summary.set_pp_mixed_pd_overlap_approx(True)
        self.summary.set_summary_df(
            pd.DataFrame(
                [
                    {
                        "parallel": "legacy display text must not be parsed",
                        "batch_size": 8,
                    }
                ]
            )
        )

        row = self.summary.get_summary_df().iloc[0]
        self.assertTrue(set(PP_RESULT_COLUMNS).issubset(row.index))
        self.assertEqual(row["parallel"], "legacy display text must not be parsed")
        self.assertEqual(row["tp_size"], 2)
        self.assertEqual(row["pp_size"], 2)
        self.assertEqual(row["dp_size"], 2)
        self.assertEqual(row["ep_size"], 1)
        self.assertEqual(row["moe_dp_size"], 1)
        self.assertEqual(row["pp_layer_partition"], (3, 5))
        self.assertEqual(row["pp_schedule"], "forward")
        self.assertEqual(row["pp_makespan_ms"], 12.0)
        self.assertEqual(row["pp_warmup_ms"], 2.0)
        self.assertEqual(row["pp_steady_ms"], 8.0)
        self.assertEqual(row["pp_cooldown_ms"], 2.0)
        self.assertEqual(row["pp_bubble_ratio"], 0.125)
        self.assertEqual(row["pp_bottleneck_stage"], 1)
        self.assertTrue(row["pp_mixed_pd_overlap_approx"])

    def test_pp1_without_schedule_has_zero_bubble_ratio(self):
        self.data_config.parallel_search_candidate = ParallelSearchCandidate(
            tp_size=1,
            pp_size=1,
            ep_size=1,
            moe_dp_size=1,
            moe_tp_size=1,
            dp_size=1,
            num_mtp_tokens=0,
            layer_partition=None,
        )
        self.summary.set_summary_df(pd.DataFrame([{"parallel": "legacy", "batch_size": 1}]))

        self.assertEqual(self.summary.get_summary_df().iloc[0]["pp_bubble_ratio"], 0.0)

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

    def test_get_early_stop_reason(self):
        """Test get_early_stop_reason returns the current early-stop reason."""
        self.assertIsNone(self.summary.get_early_stop_reason())

        self.summary.set_early_stop_flag(memory_left=-1, tpot=10.0, ttft=100.0)
        self.assertEqual(self.summary.get_early_stop_reason(), EARLY_STOP_MEMORY_OOM)

        self.summary.set_early_stop_flag(memory_left=-1, tpot=10.0, ttft=100.0, reason=EARLY_STOP_PREFILL_OOM)
        self.assertEqual(self.summary.get_early_stop_reason(), EARLY_STOP_PREFILL_OOM)

        self.summary.set_early_stop_flag(memory_left=10, tpot=60.0, ttft=100.0)
        self.assertEqual(self.summary.get_early_stop_reason(), EARLY_STOP_TPOT_LIMIT)

        self.summary.set_early_stop_flag(memory_left=10, tpot=10.0, ttft=1500.0)
        self.assertEqual(self.summary.get_early_stop_reason(), EARLY_STOP_TTFT_LIMIT)

        self.summary.set_early_stop_flag(memory_left=10, tpot=10.0, ttft=100.0)
        self.assertIsNone(self.summary.get_early_stop_reason())

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
            input_length = 1024

        args = Args()

        # Should not raise exception
        self.summary.report_final_result(args)

    def test_get_agg_disagg_final_out_missing_memory_keys(self):
        test_df = pd.DataFrame(
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
        self.summary.set_summary_df(test_df)
        self.summary.set_memory_info({"total_device_memory_gb": 64.0})

        class Args:
            model_id = "Qwen/Qwen3-8B"
            num_devices = 1
            device = "test_device"
            quantize_linear_action = "DISABLED"
            quantize_attention_action = "DISABLED"
            disagg = False
            input_length = 1024

        result = self.summary._get_agg_disagg_final_out(Args())
        result_str = "\n".join(result)
        self.assertIn("Total device memory:  64.000 GB", result_str)
        self.assertIn("Reserved memory:      -", result_str)


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
        self.assertRegex(result, r"Top 4 (?:PD Aggregated|Aggregation) Configurations:")
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
        self.assertRegex(result, r"Top 1 (?:PD Aggregated|Aggregation) Configurations:")
        self.assertIn("1", result)  # Top rank
        self.assertIn("100.00", result)  # Throughput value


class TestFormatMemory(unittest.TestCase):
    def test_fmt_memory_formats_values_and_missing_entries(self):
        row = pd.Series(
            {
                "weight_GB": 12.345,
                "kv_cache_GB": None,
                "activation_GB": float("nan"),
            }
        )

        self.assertEqual(_fmt_memory(row, "weight_GB"), "12.35")
        self.assertEqual(_fmt_memory(row, "kv_cache_GB"), "-")
        self.assertEqual(_fmt_memory(row, "activation_GB"), "-")
        self.assertEqual(_fmt_memory(row, "missing_GB"), "-")

    def test_fmt_memory_info_formats_values_and_missing_entries(self):
        memory_info = {
            "total_device_memory_gb": "64",
            "reserved_memory_gb": None,
            "bad": "not-a-number",
        }

        self.assertEqual(_fmt_memory_info(memory_info, "total_device_memory_gb"), "64.000 GB")
        self.assertEqual(_fmt_memory_info(memory_info, "reserved_memory_gb"), "-")
        self.assertEqual(_fmt_memory_info(memory_info, "bad"), "-")
        self.assertEqual(_fmt_memory_info(memory_info, "missing"), "-")


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

    def test_is_pd_ratio_mode(self):
        """Test PD mode detection for PD and regular configs."""
        non_pd_config = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )
        cases = (
            (self.summary, True),
            (OptimizerSummary(non_pd_config), False),
        )
        for summary, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(summary._is_pd_ratio_mode(), expected)

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

    def test_prepare_pd_ratio_results_stable_under_shuffle(self):
        """Tied balanced_qps rows must resolve deterministically across row orders."""
        df = pd.DataFrame(
            {
                "ttft_p": [100.0, 100.0, 80.0, 80.0],
                "tpot_d": [10.0, 10.0, 10.0, 10.0],
                "concurrency_p": [10, 10, 8, 8],
                "concurrency_d": [8, 32, 8, 32],
                "parallel_p": ["tp4pp1dp1", "tp4pp1dp1", "tp2pp1dp1", "tp2pp1dp1"],
                "parallel_d": ["tp2pp1dp1", "tp2pp1dp1", "tp2pp1dp1", "tp2pp1dp1"],
                "batch_size_p": [4, 4, 5, 5],
                "batch_size_d": [8, 32, 8, 32],
                "num_devices_p": [4, 4, 2, 2],
                "num_devices_d": [2, 2, 2, 2],
                "p_qps": [10.0, 10.0, 10.0, 10.0],
                "d_qps": [0.5, 2.0, 0.5, 2.0],
                "pd_ratio": [0.05, 0.2, 0.05, 0.2],
                "balanced_qps": [0.5, 0.5, 0.5, 0.5],
            }
        )

        # with tied balanced_qps, tie-break prefers higher d_qps -> batch_size_d=32, d_qps=2.0
        expected = (32, 2.0)

        for seed in range(20):
            self.summary.set_summary_df(df.sample(frac=1, random_state=seed))
            result = self.summary._prepare_pd_ratio_results()
            best = (int(result.iloc[0]["batch_size_d"]), float(result.iloc[0]["d_qps"]))
            self.assertEqual(best, expected)

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

    def test_calculate_instance_distribution_uses_budget_for_equal_ratio(self):
        """Equal-ratio candidates should use the available device budget."""
        p_inst, d_inst = self.summary._calculate_instance_distribution(
            pd_ratio=1.0,
            total_devices=16,
            p_devices_per_inst=4,
            d_devices_per_inst=4,
        )
        self.assertEqual((p_inst, d_inst), (2, 2))

    def test_calculate_instance_distribution_maximizes_allocated_balanced_qps(self):
        """Issue 406: allocation should maximize cluster balanced QPS."""
        p_inst, d_inst = self.summary._calculate_instance_distribution(
            pd_ratio=12.96 / 9.44,
            total_devices=32,
            p_devices_per_inst=8,
            d_devices_per_inst=16,
        )
        self.assertEqual((p_inst, d_inst), (2, 1))

    def test_calculate_instance_distribution_uses_more_devices_on_qps_tie(self):
        """A balanced-QPS tie should prefer the allocation using more devices."""
        p_inst, d_inst = self.summary._calculate_instance_distribution(
            pd_ratio=1.0,
            total_devices=6,
            p_devices_per_inst=2,
            d_devices_per_inst=2,
        )
        self.assertEqual(p_inst * 2 + d_inst * 2, 6)

    def test_prepare_pd_ratio_results_ranks_allocated_balanced_qps(self):
        """A scalable config should win after applying the total device budget."""
        self.pd_data_config.num_devices = 8
        self.summary.set_summary_df(
            pd.DataFrame(
                {
                    "ttft_p": [100.0, 100.0],
                    "tpot_d": [10.0, 10.0],
                    "concurrency_p": [10, 6],
                    "concurrency_d": [10, 6],
                    "parallel_p": ["p-large", "p-small"],
                    "parallel_d": ["d-large", "d-small"],
                    "batch_size_p": [1, 1],
                    "batch_size_d": [1, 1],
                    "num_devices_p": [4, 2],
                    "num_devices_d": [4, 2],
                    "p_qps": [10.0, 6.0],
                    "d_qps": [10.0, 6.0],
                    "pd_ratio": [1.0, 1.0],
                    "balanced_qps": [10.0, 6.0],
                }
            )
        )

        result = self.summary._prepare_pd_ratio_results()

        best = result.iloc[0]
        self.assertEqual(best["parallel_p"], "p-small")
        self.assertEqual((best["p_instances"], best["d_instances"]), (2, 2))
        self.assertEqual(best["allocated_p_qps"], 12.0)
        self.assertEqual(best["allocated_d_qps"], 12.0)
        self.assertEqual(best["balanced_qps"], 12.0)
        self.assertEqual(best["allocated_devices"], 8)

        output = "\n".join(self.summary._get_pd_ratio_final_out(SimpleArgs(), result))
        self.assertIn("Allocated Prefill QPS: 12.00 req/s", output)
        self.assertIn("Allocated Decode QPS:  12.00 req/s", output)
        self.assertIn("Balanced QPS: 12.00 req/s", output)
        self.assertIn("Devices Used: 8/8", output)

    def test_prepare_pd_ratio_results_without_budget_keeps_single_instance_qps(self):
        """Without a deployable total-device budget, keep the existing ratio result."""
        self.pd_data_config.num_devices = 1
        self.summary.set_summary_df(
            pd.DataFrame(
                {
                    "ttft_p": [100.0, 100.0],
                    "tpot_d": [10.0, 10.0],
                    "concurrency_p": [10, 6],
                    "concurrency_d": [10, 6],
                    "parallel_p": ["p-large", "p-small"],
                    "parallel_d": ["d-large", "d-small"],
                    "batch_size_p": [1, 1],
                    "batch_size_d": [1, 1],
                    "num_devices_p": [4, 2],
                    "num_devices_d": [4, 2],
                    "p_qps": [10.0, 6.0],
                    "d_qps": [10.0, 6.0],
                    "pd_ratio": [1.0, 1.0],
                    "balanced_qps": [10.0, 6.0],
                }
            )
        )

        result = self.summary._prepare_pd_ratio_results()

        best = result.iloc[0]
        self.assertEqual(best["parallel_p"], "p-large")
        self.assertEqual(best["balanced_qps"], 10.0)
        self.assertNotIn("p_instances", result.columns)

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

    def test_get_pd_ratio_final_out_missing_memory_keys(self):
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
        self.summary.set_memory_info({"total_device_memory_gb": 64.0})

        args = SimpleArgs()
        result = self.summary._get_pd_ratio_final_out(args, df)
        result_str = "\n".join(result)
        self.assertIn("Total device memory:  64.000 GB", result_str)
        self.assertIn("Reserved memory:      -", result_str)

    def test_report_final_result_pd_mode_empty(self):
        """Test report_final_result in PD mode with empty DataFrame does not raise."""
        self.summary.set_summary_df(pd.DataFrame())
        args = SimpleArgs()
        # Should not raise exception
        self.summary.report_final_result(args)


class TestMixedBatchOptimizerSummary(unittest.TestCase):
    def setUp(self):
        self.data_config = OptimizerData(
            ttft_limits=1000.0,
            tpot_limits=None,
        )
        self.summary = OptimizerSummary(self.data_config)

    def test_report_final_result_expands_composition_rows(self):
        test_df = pd.DataFrame(
            [
                {
                    "device_name": "TEST_DEVICE",
                    "num_devices": 4,
                    "model_id": "test-model",
                    "quantize_linear_action": "DISABLED",
                    "quantize_attention_action": "DISABLED",
                    "input_length": None,
                    "num_input_tokens": "all",
                    "output_length": 50,
                    "request_ratio": 1.0,
                    "samples": 4,
                    "concurrency": 4,
                    "ttft": 100.0,
                    "tpot": None,
                    "token/s": 2000.0,
                    "token/s/device": 500.0,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "percentage_breakdowns": "prefill:100%",
                },
                {
                    "device_name": "TEST_DEVICE",
                    "num_devices": 4,
                    "model_id": "test-model",
                    "quantize_linear_action": "DISABLED",
                    "quantize_attention_action": "DISABLED",
                    "input_length": None,
                    "num_input_tokens": 250,
                    "output_length": 50,
                    "request_ratio": 0.25,
                    "samples": 1,
                    "concurrency": 4,
                    "ttft": None,
                    "tpot": None,
                    "token/s": None,
                    "token/s/device": None,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "percentage_breakdowns": None,
                },
                {
                    "device_name": "TEST_DEVICE",
                    "num_devices": 4,
                    "model_id": "test-model",
                    "quantize_linear_action": "DISABLED",
                    "quantize_attention_action": "DISABLED",
                    "input_length": None,
                    "num_input_tokens": 1000,
                    "output_length": 50,
                    "request_ratio": 0.75,
                    "samples": 3,
                    "concurrency": 4,
                    "ttft": None,
                    "tpot": None,
                    "token/s": None,
                    "token/s/device": None,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "percentage_breakdowns": None,
                },
            ]
        )
        self.summary.set_summary_df(test_df)

        class Args:
            model_id = "test-model"
            num_devices = 4
            device = "TEST_DEVICE"
            dump_original_results = False
            quantize_linear_action = "DISABLED"
            quantize_attention_action = "DISABLED"
            disagg = True
            input_length = "serving_cast/example/length_distribution.yaml"

        result = self.summary._get_agg_disagg_final_out(Args())
        result_str = "\n".join(result)
        self.assertIn("Top 1 Disaggregation (Prefill) Configurations:", result_str)
        self.assertIn("num_input_tokens", result_str)
        self.assertIn("request_ratio", result_str)
        self.assertIn("samples", result_str)
        self.assertIn("all", result_str)
        self.assertIn("250", result_str)
        self.assertIn("1000", result_str)
        self.assertIn("-", result_str)

    def test_get_agg_disagg_final_out_batched_returns_message_when_all_rows_filtered(
        self,
    ):
        test_df = pd.DataFrame(
            [
                {
                    "device_name": "TEST_DEVICE",
                    "num_devices": 4,
                    "model_id": "test-model",
                    "quantize_linear_action": "DISABLED",
                    "quantize_attention_action": "DISABLED",
                    "input_length": None,
                    "num_input_tokens": "all",
                    "output_length": 50,
                    "request_ratio": 1.0,
                    "samples": 4,
                    "concurrency": 4,
                    "ttft": 2000.0,
                    "tpot": None,
                    "token/s": 2000.0,
                    "token/s/device": 500.0,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "percentage_breakdowns": "prefill:100%",
                }
            ]
        )
        self.summary.set_summary_df(test_df)

        class Args:
            model_id = "test-model"
            num_devices = 4
            device = "TEST_DEVICE"
            dump_original_results = False
            quantize_linear_action = "DISABLED"
            quantize_attention_action = "DISABLED"
            disagg = True
            input_length = "serving_cast/example/length_distribution.yaml"

        result = self.summary._get_agg_disagg_final_out(Args())
        self.assertEqual(
            result,
            [
                "*" * 80,
                "No configurations satisfy the current TTFT/TPOT filters.",
                "*" * 80,
            ],
        )

    def test_report_final_result_renders_batched_aggregation_rows(self):
        self.data_config.tpot_limits = 10.0
        test_df = pd.DataFrame(
            [
                {
                    "device_name": "TEST_DEVICE",
                    "num_devices": 4,
                    "model_id": "test-model",
                    "quantize_linear_action": "DISABLED",
                    "quantize_attention_action": "DISABLED",
                    "input_length": None,
                    "num_input_tokens": "all",
                    "output_length": 50,
                    "request_ratio": 1.0,
                    "samples": 4,
                    "concurrency": 4,
                    "ttft": 100.0,
                    "tpot": 5.0,
                    "token/s": 2000.0,
                    "token/s/device": 500.0,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "percentage_breakdowns(p)": "prefill:100%",
                    "percentage_breakdowns(d)": "decode:100%",
                },
                {
                    "device_name": "TEST_DEVICE",
                    "num_devices": 4,
                    "model_id": "test-model",
                    "quantize_linear_action": "DISABLED",
                    "quantize_attention_action": "DISABLED",
                    "input_length": None,
                    "num_input_tokens": 250,
                    "output_length": 50,
                    "request_ratio": 0.25,
                    "samples": 1,
                    "concurrency": 4,
                    "ttft": None,
                    "tpot": None,
                    "token/s": None,
                    "token/s/device": None,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "percentage_breakdowns(p)": None,
                    "percentage_breakdowns(d)": None,
                },
            ]
        )
        self.summary.set_summary_df(test_df)

        class Args:
            model_id = "test-model"
            num_devices = 4
            device = "TEST_DEVICE"
            dump_original_results = False
            quantize_linear_action = "DISABLED"
            quantize_attention_action = "DISABLED"
            disagg = False
            input_length = "serving_cast/example/length_distribution.yaml"

        result = self.summary._get_agg_disagg_final_out(Args())
        result_str = "\n".join(result)
        self.assertIn("Top 1 Aggregation Configurations:", result_str)
        self.assertIn("TPOT (ms)", result_str)
        self.assertIn("num_input_tokens", result_str)
        self.assertNotIn("Disaggregation (Prefill)", result_str)

    def test_expand_composition_rows_keeps_aggregate_first_then_detail_sorted_by_tokens(self):
        test_df = pd.DataFrame(
            [
                {
                    "num_input_tokens": 1000,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "concurrency": 4,
                    "num_devices": 4,
                },
                {
                    "num_input_tokens": "all",
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "concurrency": 4,
                    "num_devices": 4,
                },
                {
                    "num_input_tokens": 250,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "concurrency": 4,
                    "num_devices": 4,
                },
            ]
        )
        best_df = pd.DataFrame(
            [
                {
                    "num_input_tokens": "all",
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                    "concurrency": 4,
                    "num_devices": 4,
                }
            ]
        )
        self.summary.set_summary_df(test_df)

        expanded_df = self.summary._expand_composition_rows(best_df)

        self.assertEqual(list(expanded_df["num_input_tokens"]), ["all", 250, 1000])

    def test_get_agg_disagg_table_buf_batched_numbers_only_aggregate_rows(self):
        df = pd.DataFrame(
            [
                {
                    "num_devices": 4,
                    "num_input_tokens": "all",
                    "request_ratio": 1.0,
                    "samples": 4,
                    "concurrency": 4,
                    "ttft": 100.0,
                    "token/s": 2000.0,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                },
                {
                    "num_devices": 4,
                    "num_input_tokens": 250,
                    "request_ratio": 0.25,
                    "samples": 1,
                    "concurrency": 4,
                    "ttft": None,
                    "token/s": None,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                },
                {
                    "num_devices": 4,
                    "num_input_tokens": "all",
                    "request_ratio": 1.0,
                    "samples": 8,
                    "concurrency": 8,
                    "ttft": 200.0,
                    "token/s": 1500.0,
                    "parallel": "TP=2 | PP=1 | DP=2",
                    "batch_size": 4,
                },
                {
                    "num_devices": 4,
                    "num_input_tokens": 500,
                    "request_ratio": 0.5,
                    "samples": 4,
                    "concurrency": 8,
                    "ttft": None,
                    "token/s": None,
                    "parallel": "TP=2 | PP=1 | DP=2",
                    "batch_size": 4,
                },
            ]
        )

        result = _get_agg_disagg_table_buf_batched(df, is_disagg=True)

        self.assertIn("Top 2 Disaggregation (Prefill) Configurations:", result)
        self.assertIn("|  1  |", result)
        self.assertIn("|  2  |", result)
        self.assertIn("|  -  |", result)

    def test_get_agg_disagg_table_buf_batched_renders_aggregation_rows(self):
        df = pd.DataFrame(
            [
                {
                    "num_devices": 4,
                    "num_input_tokens": "all",
                    "request_ratio": 1.0,
                    "samples": 4,
                    "concurrency": 4,
                    "ttft": 100.0,
                    "tpot": 5.0,
                    "token/s": 2000.0,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                },
                {
                    "num_devices": 4,
                    "num_input_tokens": 250,
                    "request_ratio": 0.25,
                    "samples": 1,
                    "concurrency": 4,
                    "ttft": None,
                    "tpot": None,
                    "token/s": None,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                },
            ]
        )

        result = _get_agg_disagg_table_buf_batched(df, is_disagg=False)

        self.assertIn("Top 1 Aggregation Configurations:", result)
        self.assertIn("TPOT (ms)", result)
        self.assertIn("5.00", result)

    def test_get_agg_disagg_table_buf_batched_controls_tpot_column(self):
        df = pd.DataFrame(
            [
                {
                    "num_devices": 4,
                    "num_input_tokens": "all",
                    "request_ratio": 1.0,
                    "samples": 4,
                    "concurrency": 4,
                    "ttft": 100.0,
                    "tpot": 5.0,
                    "token/s": 2000.0,
                    "parallel": "TP=1 | PP=1 | DP=4",
                    "batch_size": 1,
                },
            ]
        )

        agg_result = _get_agg_disagg_table_buf_batched(df, is_disagg=False)
        disagg_result = _get_agg_disagg_table_buf_batched(df, is_disagg=True)

        self.assertIn("Top 1 Aggregation Configurations:", agg_result)
        self.assertIn("TPOT (ms)", agg_result)
        self.assertIn("Top 1 Disaggregation (Prefill) Configurations:", disagg_result)
        self.assertNotIn("TPOT (ms)", disagg_result)


if __name__ == "__main__":
    unittest.main()
