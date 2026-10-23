# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Unit tests for CapacityPlanner and target_qps validator."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from serving_cast.service.capacity_planner import (
    CapacityPlanConfig,
    CapacityPlanResult,
    CapacityPlanner,
    render_capacity_planning_result,
)

from cli.inference.throughput_optimizer import run_capacity_planning


def _agg_row(
    token_s=500.0,
    ttft=300.0,
    tpot=40.0,
    concurrency=64,
    num_devices=8,
    dp_size=1,
    tp_size=8,
    pp_size=1,
    batch_size=64,
    parallel="TP=8 | PP=1 | DP=1",
):
    row = {
        "device_name": "TEST_DEVICE",
        "num_devices": num_devices,
        "model_id": "test-model",
        "token/s": token_s,
        "ttft": ttft,
        "tpot": tpot,
        "concurrency": concurrency,
        "parallel": parallel,
        "batch_size": batch_size,
        "effective_input_length": 1024,
        "output_length": 512,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "ep_size": 1,
        "moe_dp_size": 1,
        "moe_tp_size": 1,
        "pp_layer_partition": None,
        "pp_schedule": "forward",
        "pp_makespan_ms": None,
        "pp_warmup_ms": None,
        "pp_steady_ms": None,
        "pp_cooldown_ms": None,
        "pp_bubble_ratio": 0.0,
        "pp_bottleneck_stage": None,
        "pp_mixed_pd_overlap_approx": False,
    }
    return row


def _disagg_prefill_row(
    token_s=50000.0,
    ttft=500.0,
    concurrency=32,
    num_devices=8,
    dp_size=1,
    tp_size=8,
    pp_size=1,
    batch_size=32,
    parallel="TP=8 | PP=1 | DP=1",
):
    row = _agg_row(
        token_s=token_s,
        ttft=ttft,
        tpot=None,
        concurrency=concurrency,
        num_devices=num_devices,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        batch_size=batch_size,
        parallel=parallel,
    )
    return row


def _disagg_decode_row(
    token_s=5000.0,
    tpot=20.0,
    concurrency=64,
    num_devices=8,
    dp_size=1,
    tp_size=8,
    pp_size=1,
    batch_size=64,
    parallel="TP=8 | PP=1 | DP=1",
):
    row = _agg_row(
        token_s=token_s,
        ttft=None,
        tpot=tpot,
        concurrency=concurrency,
        num_devices=num_devices,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        batch_size=batch_size,
        parallel=parallel,
    )
    return row


def _pd_ratio_row(
    p_qps=10.0,
    d_qps=20.0,
    num_devices_p=4,
    num_devices_d=2,
    ttft_p=100.0,
    tpot_d=15.0,
    parallel_p="TP=4 | PP=1 | DP=1",
    parallel_d="TP=2 | PP=1 | DP=1",
):
    return {
        "pd_ratio": d_qps / p_qps if p_qps > 0 else 0,
        "p_qps": p_qps,
        "d_qps": d_qps,
        "balanced_qps": min(p_qps, d_qps),
        "ttft_p": ttft_p,
        "tpot_d": tpot_d,
        "parallel_p": parallel_p,
        "parallel_d": parallel_d,
        "num_devices_p": num_devices_p,
        "num_devices_d": num_devices_d,
        "batch_size_p": 32,
        "batch_size_d": 64,
        "concurrency_p": 32,
        "concurrency_d": 64,
    }


# ==================================================================
# Extrapolation tests
# ==================================================================


class TestExtrapolateAggDisagg(unittest.TestCase):
    """Test Phase 2 extrapolation for aggregation/disagg modes."""

    def setUp(self):
        self.config = CapacityPlanConfig(
            target_qps=100.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
            mode="aggregation",
        )
        self.planner = CapacityPlanner(self.config)

    def test_agg_single_replica(self):
        df = pd.DataFrame([_agg_row(token_s=51200, dp_size=1, num_devices=8)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["per_replica_qps"], 100.0, places=1)
        self.assertEqual(row["per_replica_gpus"], 8)
        self.assertEqual(row["replicas_needed"], 1)
        self.assertEqual(row["total_gpus"], 8)

    def test_agg_dp2(self):
        df = pd.DataFrame([_agg_row(token_s=51200, dp_size=2, num_devices=16)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["per_replica_qps"], 50.0, places=1)
        self.assertEqual(row["per_replica_gpus"], 8)
        self.assertEqual(row["replicas_needed"], 2)
        self.assertEqual(row["total_gpus"], 16)

    def test_agg_multiple_configs(self):
        df = pd.DataFrame(
            [
                _agg_row(token_s=51200, dp_size=1, num_devices=8, parallel="TP=8 | DP=1"),
                _agg_row(token_s=25600, dp_size=2, num_devices=8, parallel="TP=4 | DP=2"),
            ]
        )
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(len(rows), 2)

    def test_agg_per_die_qps_calculation(self):
        df = pd.DataFrame([_agg_row(token_s=51200, dp_size=1, num_devices=8)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        row = rows[0]
        expected_per_die = row["per_replica_qps"] / row["per_replica_gpus"]
        self.assertAlmostEqual(row["per_die_qps"], expected_per_die, places=2)

    def test_agg_high_target_needs_many_replicas(self):
        config = CapacityPlanConfig(target_qps=10000.0, mode="aggregation")
        planner = CapacityPlanner(config)
        df = pd.DataFrame([_agg_row(token_s=51200, dp_size=1, num_devices=8)])
        rows = planner._extrapolate_agg_disagg(df, output_length=512)
        row = rows[0]
        self.assertEqual(row["replicas_needed"], 100)
        self.assertEqual(row["total_gpus"], 800)

    def test_empty_df(self):
        df = pd.DataFrame()
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(rows, [])

    def test_per_instance_qps_is_not_rounded_before_scaling(self):
        """Rounding per-instance QPS to 2 decimals scaled the error by the replica count.

        Reviewer case: a 0.014 QPS instance against a target of 1 needs 72
        replicas. Rounding to 0.01 first reports 0.72 achieved QPS instead of
        1.008, and anything below 0.005 would report 0.
        """
        config = CapacityPlanConfig(target_qps=1.0, mode="aggregation")
        planner = CapacityPlanner(config)
        # per_replica_qps = token/s / output_length = 14 / 1000 = 0.014
        df = pd.DataFrame([_agg_row(token_s=14.0, dp_size=1, num_devices=8, tp_size=8)])
        rows = planner._extrapolate_agg_disagg(df, output_length=1000)
        row = rows[0]
        self.assertAlmostEqual(row["per_replica_qps"], 0.014, places=9)
        self.assertEqual(row["replicas_needed"], 72)
        self.assertAlmostEqual(row["per_replica_qps"] * row["replicas_needed"], 1.008, places=9)

    def test_rank_tiebreak_uses_unrounded_per_die_qps(self):
        """Two equal-card-count rows decide on per-die QPS in the 3rd decimal."""
        config = CapacityPlanConfig(target_qps=1.0, mode="aggregation")
        planner = CapacityPlanner(config)
        # per_die_qps = (token/s / output_length) / 8 -> 1.004 vs 1.006.
        lower = _agg_row(token_s=8032.0, dp_size=1, num_devices=8, tp_size=8, parallel="TP=8 | PP=1 | DP=1")
        higher = _agg_row(token_s=8048.0, dp_size=1, num_devices=8, tp_size=8, parallel="TP=4 | PP=2 | DP=1")
        rows = planner._extrapolate_agg_disagg(pd.DataFrame([lower, higher]), output_length=1000)
        self.assertEqual({row["total_gpus"] for row in rows}, {8})
        best, ranked = planner._rank_and_select(rows)
        self.assertAlmostEqual(best["per_die_qps"], 1.006, places=9)
        self.assertEqual([row["per_die_qps"] for row in ranked], [1.006, 1.004])

    def test_disagg_prefill_qps(self):
        df = pd.DataFrame([_disagg_prefill_row(token_s=50000, ttft=500, concurrency=32, dp_size=1)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        expected_qps = 32 / 500 * 1000
        self.assertAlmostEqual(row["per_replica_qps"], expected_qps, places=1)

    def test_disagg_prefill_qps_pp_gt1_uses_steady_throughput_time_base(self):
        # Issue #543: PP>1 prefill rows carry steady-state token/s; the
        # per-instance QPS must share that denominator instead of the wave-1
        # warm-up makespan. The divisor is the full input_length: the
        # throughput numerator counts the complete prompt even under prefix
        # cache (review P1), so QPS here is NOT token/s / effective_input_length.
        row = _disagg_prefill_row(
            token_s=40000.0,
            ttft=7822.5,
            concurrency=6,
            dp_size=1,
            pp_size=4,
            batch_size=6,
            parallel="TP=8 | PP=4 | DP=1",
        )
        row["prefill_phase_makespan_ms"] = 9789.9
        row["input_length"] = 1024
        row["effective_input_length"] = 512
        df = pd.DataFrame([row])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(len(rows), 1)
        # Steady-state QPS = token/s / input_length = 40000 / 1024 (full
        # prompt), NOT token/s / effective (40000 / 512, the P1 bug) and NOT
        # concurrency / wave-1 makespan (6 / 9789.9 * 1000).
        self.assertAlmostEqual(rows[0]["per_replica_qps"], 40000.0 / 1024.0, places=6)

    def test_disagg_decode_qps(self):
        df = pd.DataFrame([_disagg_decode_row(token_s=5000, tpot=20, concurrency=64, dp_size=1)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        expected_qps = 64 / (20 * 512) * 1000
        self.assertAlmostEqual(row["per_replica_qps"], expected_qps, places=2)

    def test_disagg_prefill_dp2(self):
        df = pd.DataFrame([_disagg_prefill_row(token_s=50000, ttft=500, concurrency=32, dp_size=2, num_devices=16)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        row = rows[0]
        self.assertEqual(row["per_replica_gpus"], 8)
        expected_qps = (32 / 500 * 1000) / 2
        self.assertAlmostEqual(row["per_replica_qps"], expected_qps, places=1)

    def test_disagg_decode_dp2(self):
        df = pd.DataFrame([_disagg_decode_row(token_s=5000, tpot=20, concurrency=64, dp_size=2, num_devices=16)])
        rows = self.planner._extrapolate_agg_disagg(df, output_length=512)
        row = rows[0]
        self.assertEqual(row["per_replica_gpus"], 8)


class TestExtrapolatePDRatio(unittest.TestCase):
    """Test Phase 2 extrapolation for PD ratio mode."""

    def setUp(self):
        self.config = CapacityPlanConfig(
            target_qps=100.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
            mode="pd_ratio",
        )
        self.planner = CapacityPlanner(self.config)

    def test_pd_independent_scaling(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=10, d_qps=20, num_devices_p=4, num_devices_d=2)])
        rows = self.planner._extrapolate_pd_ratio(df)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["p_inst"], 10)
        self.assertEqual(row["d_inst"], 5)
        self.assertEqual(row["total_gpus"], 10 * 4 + 5 * 2)
        self.assertEqual(row["total_gpus"], 50)

    def test_pd_balanced(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=10, d_qps=10, num_devices_p=4, num_devices_d=2)])
        rows = self.planner._extrapolate_pd_ratio(df)
        row = rows[0]
        self.assertEqual(row["p_inst"], 10)
        self.assertEqual(row["d_inst"], 10)
        self.assertEqual(row["total_gpus"], 10 * 4 + 10 * 2)

    def test_pd_p_bottleneck(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=5, d_qps=20, num_devices_p=4, num_devices_d=2)])
        rows = self.planner._extrapolate_pd_ratio(df)
        row = rows[0]
        self.assertEqual(row["p_inst"], 20)
        self.assertEqual(row["d_inst"], 5)
        self.assertEqual(row["total_gpus"], 20 * 4 + 5 * 2)

    def test_pd_d_bottleneck(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=20, d_qps=5, num_devices_p=4, num_devices_d=2)])
        rows = self.planner._extrapolate_pd_ratio(df)
        row = rows[0]
        self.assertEqual(row["p_inst"], 5)
        self.assertEqual(row["d_inst"], 20)
        self.assertEqual(row["total_gpus"], 5 * 4 + 20 * 2)

    def test_pd_achieved_qps(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=10, d_qps=20, num_devices_p=4, num_devices_d=2)])
        rows = self.planner._extrapolate_pd_ratio(df)
        row = rows[0]
        expected_achieved = min(10 * 10, 5 * 20)
        self.assertAlmostEqual(row["achieved_qps"], expected_achieved, places=1)

    def test_pd_per_die_qps(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=10, d_qps=20, num_devices_p=4, num_devices_d=2)])
        rows = self.planner._extrapolate_pd_ratio(df)
        row = rows[0]
        total = 10 * 4 + 5 * 2
        expected = min(10 * 10, 5 * 20) / total
        self.assertAlmostEqual(row["per_die_qps"], expected, places=2)

    def test_pd_empty(self):
        df = pd.DataFrame()
        rows = self.planner._extrapolate_pd_ratio(df)
        self.assertEqual(rows, [])

    def test_pd_zero_qps_skipped(self):
        df = pd.DataFrame([_pd_ratio_row(p_qps=0, d_qps=20)])
        rows = self.planner._extrapolate_pd_ratio(df)
        self.assertEqual(len(rows), 0)


# ==================================================================
# Ranking tests
# ==================================================================


class TestRankAndSelect(unittest.TestCase):
    """Test ranking: sort by total_gpus, tiebreak per_die_qps."""

    def setUp(self):
        self.config = CapacityPlanConfig(target_qps=100.0)
        self.planner = CapacityPlanner(self.config)

    def test_sort_by_total_gpus(self):
        rows = [
            {"total_gpus": 48, "per_die_qps": 20},
            {"total_gpus": 32, "per_die_qps": 30},
            {"total_gpus": 64, "per_die_qps": 15},
        ]
        best, ranked = self.planner._rank_and_select(rows)
        self.assertEqual(best["total_gpus"], 32)
        self.assertEqual(ranked[0]["total_gpus"], 32)
        self.assertEqual(ranked[-1]["total_gpus"], 64)

    def test_tiebreak_per_die_qps(self):
        rows = [
            {"total_gpus": 32, "per_die_qps": 30},
            {"total_gpus": 32, "per_die_qps": 35},
        ]
        best, _ = self.planner._rank_and_select(rows)
        self.assertEqual(best["per_die_qps"], 35)

    def test_empty(self):
        best, ranked = self.planner._rank_and_select([])
        self.assertIsNone(best)
        self.assertEqual(ranked, [])

    def test_single_row(self):
        rows = [{"total_gpus": 16, "per_die_qps": 10}]
        best, ranked = self.planner._rank_and_select(rows)
        self.assertEqual(best["total_gpus"], 16)
        self.assertEqual(len(ranked), 1)


# ==================================================================
# SLO filter tests
# ==================================================================


class TestFilterSloCompliant(unittest.TestCase):
    """Test SLO filtering."""

    def setUp(self):
        self.config = CapacityPlanConfig(
            target_qps=100.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
        )
        self.planner = CapacityPlanner(self.config)

    def test_filters_over_limit(self):
        df = pd.DataFrame(
            [
                _agg_row(ttft=300, tpot=40),
                _agg_row(ttft=600, tpot=40),
                _agg_row(ttft=300, tpot=60),
            ]
        )
        filtered = self.planner._filter_slo_compliant(df)
        self.assertEqual(len(filtered), 1)

    def test_no_slo_limit(self):
        config = CapacityPlanConfig(target_qps=100.0)
        planner = CapacityPlanner(config)
        df = pd.DataFrame([_agg_row(ttft=999, tpot=999)])
        filtered = planner._filter_slo_compliant(df)
        self.assertEqual(len(filtered), 1)

    def test_ttft_only_limit(self):
        config = CapacityPlanConfig(target_qps=100.0, ttft_limit=500.0)
        planner = CapacityPlanner(config)
        df = pd.DataFrame([_agg_row(ttft=300, tpot=999), _agg_row(ttft=600, tpot=1)])
        filtered = planner._filter_slo_compliant(df)
        self.assertEqual(len(filtered), 1)

    def test_tpot_only_limit(self):
        config = CapacityPlanConfig(target_qps=100.0, tpot_limit=50.0)
        planner = CapacityPlanner(config)
        df = pd.DataFrame([_agg_row(ttft=999, tpot=40), _agg_row(ttft=1, tpot=60)])
        filtered = planner._filter_slo_compliant(df)
        self.assertEqual(len(filtered), 1)

    def test_pd_slo_filter(self):
        config = CapacityPlanConfig(target_qps=100.0, ttft_limit=500.0, tpot_limit=50.0, mode="pd_ratio")
        planner = CapacityPlanner(config)
        df = pd.DataFrame([_pd_ratio_row(ttft_p=100, tpot_d=15)])
        filtered = planner._filter_slo_compliant_pd(df)
        self.assertEqual(len(filtered), 1)


# ==================================================================
# Phase 0: determine search range
# ==================================================================


class TestDetermineSearchRange(unittest.TestCase):
    """Test Phase 0 search range determination."""

    def setUp(self):
        self.config = CapacityPlanConfig(target_qps=100.0)
        self.planner = CapacityPlanner(self.config)

    def test_user_specified(self):
        config = CapacityPlanConfig(target_qps=100.0, num_devices=16, device="TEST_DEVICE")
        planner = CapacityPlanner(config)
        budgets = planner._determine_search_range()
        self.assertEqual(budgets, [16])

    def test_user_specified_beyond_comm_grid_returns_no_budget(self):
        """A device count the profile cannot wire up yields no runnable budget."""
        config = CapacityPlanConfig(target_qps=100.0, num_devices=4096, device="TEST_DEVICE")
        planner = CapacityPlanner(config)
        with self.assertLogs("serving_cast.service.capacity_planner", "ERROR") as logs:
            budgets = planner._determine_search_range()
        self.assertEqual(budgets, [])
        self.assertIn("exceeds comm_grid capacity", "\n".join(logs.output))

    def test_auto_mode_enumerates_all_powers_of_two_up_to_capacity(self):
        config = CapacityPlanConfig(target_qps=100.0, num_devices=None, device="TEST_DEVICE")
        planner = CapacityPlanner(config)
        # TEST_DEVICE grid is (256, 8) -> seed 256, capacity 2048.
        self.assertEqual(planner._determine_search_range(), [256, 512, 1024, 2048])


# ==================================================================
# Run single mode (end-to-end with mocked optimizer)
# ==================================================================


class TestRunSingleMode(unittest.TestCase):
    """Test full single-mode run with mocked optimizer."""

    def setUp(self):
        self.config = CapacityPlanConfig(
            target_qps=100.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
            mode="aggregation",
        )
        self.planner = CapacityPlanner(self.config)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_user_specified_mode(self, mock_run):
        mock_run.return_value = pd.DataFrame([_agg_row(token_s=51200, dp_size=1, num_devices=8)])
        args = MagicMock()
        args.num_devices = 8
        args.device = "TEST_DEVICE"
        args.output_length = 512
        result = self.planner._run_single_mode(args)
        self.assertIsNotNone(result.minimum_devices)
        self.assertEqual(result.minimum_devices, 8)
        self.assertEqual(result.replicas_needed, 1)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_no_slo_compliant(self, mock_run):
        mock_run.return_value = pd.DataFrame([_agg_row(ttft=999, tpot=999)])
        args = MagicMock()
        args.num_devices = 8
        args.device = "TEST_DEVICE"
        args.output_length = 512
        result = self.planner._run_single_mode(args)
        self.assertIsNone(result.minimum_devices)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_oom_then_success(self, mock_run):
        mock_run.side_effect = [
            None,
            pd.DataFrame([_agg_row(token_s=51200, dp_size=1, num_devices=16)]),
        ]
        args = MagicMock()
        args.num_devices = None
        args.device = "TEST_DEVICE"
        args.output_length = 512
        with patch.object(self.planner, "_determine_search_range", return_value=[8, 16]):
            result = self.planner._run_single_mode(args)
        self.assertIsNotNone(result.minimum_devices)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    @patch.object(CapacityPlanner, "_determine_search_range")
    def test_all_budgets_searched_after_first_feasible(self, mock_range, mock_run):
        """Every budget is searched, not just the first feasible one plus one step.

        Per-die QPS is not monotonic in device count, so a later budget can still
        hold the global minimum. A budgets=[8, 16] fixture cannot tell the two
        behaviours apart (both run twice), so this uses three budgets with the
        smallest one already feasible.
        """
        mock_range.return_value = [8, 16, 32]
        mock_run.side_effect = [
            pd.DataFrame([_agg_row(token_s=51200, dp_size=1, num_devices=8, tp_size=8)]),
            pd.DataFrame([_agg_row(token_s=102400, dp_size=1, num_devices=16, tp_size=16)]),
            pd.DataFrame([_agg_row(token_s=204800, dp_size=1, num_devices=32, tp_size=32)]),
        ]
        args = MagicMock()
        args.num_devices = None
        args.device = "TEST_DEVICE"
        args.output_length = 512
        result = self.planner._run_single_mode(args)
        self.assertIsNotNone(result.minimum_devices)
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual([c.args[1] for c in mock_run.call_args_list], [8, 16, 32])

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    @patch.object(CapacityPlanner, "_determine_search_range")
    def test_reported_achieved_qps_is_not_scaled_by_rounding(self, mock_range, mock_run):
        """The reported achieved QPS must not be rebuilt from a rounded value.

        0.014 QPS per instance against a target of 1 gives 72 replicas: the
        reported value is 1.008, not 72 x 0.01 = 0.72.
        """
        config = CapacityPlanConfig(target_qps=1.0, mode="aggregation", output_length=1000)
        planner = CapacityPlanner(config)
        mock_range.return_value = [8]
        mock_run.return_value = pd.DataFrame([_agg_row(token_s=14.0, dp_size=1, num_devices=8, tp_size=8)])
        args = MagicMock()
        args.device = "TEST_DEVICE"
        args.output_length = 1000

        result = planner._run_single_mode(args)

        self.assertEqual(result.replicas_needed, 72)
        self.assertAlmostEqual(result.achieved_qps, 1.008, places=9)
        self.assertGreaterEqual(result.achieved_qps, config.target_qps)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    @patch.object(CapacityPlanner, "_determine_search_range")
    def test_no_runnable_budget_fails_without_running_optimizer(self, mock_range, mock_run):
        """An out-of-range --num-devices reports failure instead of running anyway."""
        mock_range.return_value = []
        args = MagicMock()
        args.num_devices = 4096
        args.device = "TEST_DEVICE"
        args.output_length = 512
        with self.assertLogs("serving_cast.service.capacity_planner", "ERROR") as logs:
            result = self.planner._run_single_mode(args)
        self.assertIsNone(result.minimum_devices)
        mock_run.assert_not_called()
        self.assertIn("No device-count budget to search", "\n".join(logs.output))

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    @patch.object(CapacityPlanner, "_determine_search_range")
    def test_all_oom(self, mock_range, mock_run):
        mock_range.return_value = [8, 16]
        mock_run.return_value = None
        args = MagicMock()
        args.num_devices = None
        args.device = "TEST_DEVICE"
        args.output_length = 512
        result = self.planner._run_single_mode(args)
        self.assertIsNone(result.minimum_devices)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    @patch.object(CapacityPlanner, "_determine_search_range")
    def test_disagg_prefill_mode(self, mock_range, mock_run):
        config = CapacityPlanConfig(target_qps=50.0, ttft_limit=1000.0, mode="aggregation")
        planner = CapacityPlanner(config)
        mock_range.return_value = [8]
        mock_run.return_value = pd.DataFrame([_disagg_prefill_row(token_s=50000, ttft=500, concurrency=32, dp_size=1)])
        args = MagicMock()
        args.num_devices = 8
        args.device = "TEST_DEVICE"
        args.output_length = 512
        result = planner._run_single_mode(args)
        self.assertIsNotNone(result.minimum_devices)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    @patch.object(CapacityPlanner, "_determine_search_range")
    def test_disagg_decode_mode(self, mock_range, mock_run):
        config = CapacityPlanConfig(target_qps=50.0, tpot_limit=50.0, mode="aggregation")
        planner = CapacityPlanner(config)
        mock_range.return_value = [8]
        mock_run.return_value = pd.DataFrame([_disagg_decode_row(token_s=5000, tpot=20, concurrency=64, dp_size=1)])
        args = MagicMock()
        args.num_devices = 8
        args.device = "TEST_DEVICE"
        args.output_length = 512
        result = planner._run_single_mode(args)
        self.assertIsNotNone(result.minimum_devices)


# ==================================================================
# Run PD ratio mode
# ==================================================================


class TestRunPDRatio(unittest.TestCase):
    """Test PD ratio mode."""

    def setUp(self):
        self.config = CapacityPlanConfig(
            target_qps=100.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
            mode="pd_ratio",
        )
        self.planner = CapacityPlanner(self.config)

    def test_fixed_per_instance(self):
        args = MagicMock()
        args.prefill_devices_per_instance = 4
        args.decode_devices_per_instance = 2
        args.device = "TEST_DEVICE"
        args.enable_optimize_prefill_decode_ratio = True

        with patch.object(CapacityPlanner, "_run_optimizer_at_n") as mock_run:
            mock_run.return_value = pd.DataFrame([_pd_ratio_row()])
            result = self.planner._run_pd_ratio_fixed(args, 4, 2)
        self.assertIsNotNone(result.minimum_devices)
        self.assertEqual(result.pd_ratio_info["p_inst"], 10)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_auto_search_enumerates_every_combination(self, mock_run):
        """Exact (P, D) coverage and the minimum-total-cards pick.

        Recording only the total budget would pass for a single combination and
        could not tell (2, 4) from (4, 2), so this records both per-instance
        counts on every call and returns a distinct capacity per combination.
        """

        # Same per-phase QPS for every pair, so the total-card count is decided
        # purely by the per-instance sizes: (2,2) -> 2*2 + 2*2 = 8 cards,
        # (2,4) and (4,2) -> 12, (4,4) -> 16. The winner must be (2, 2).
        def fake_run(run_args, n):
            combo = (run_args.prefill_devices_per_instance, run_args.decode_devices_per_instance)
            return pd.DataFrame([_pd_ratio_row(p_qps=5.0, d_qps=5.0, num_devices_p=combo[0], num_devices_d=combo[1])])

        config = CapacityPlanConfig(
            target_qps=10.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
            mode="pd_ratio",
            pd_instance_budgets=[2, 4],
        )
        planner = CapacityPlanner(config)
        mock_run.side_effect = fake_run
        args = MagicMock()
        args.device = "TEST_DEVICE"
        args.enable_optimize_prefill_decode_ratio = True
        args.prefill_devices_per_instance = None
        args.decode_devices_per_instance = None

        result = planner._run_pd_ratio_auto(args)

        combos = [
            (call.args[0].prefill_devices_per_instance, call.args[0].decode_devices_per_instance)
            for call in mock_run.call_args_list
        ]
        self.assertEqual(combos, [(2, 2), (2, 4), (4, 2), (4, 4)])
        # Every call carries the matching total device count for that pair.
        self.assertEqual([call.args[1] for call in mock_run.call_args_list], [4, 6, 6, 8])
        self.assertIsNotNone(result.minimum_devices)
        self.assertEqual(result.minimum_devices, 8)
        self.assertEqual(result.pd_ratio_info["p_inst"], 2)
        self.assertEqual(result.pd_ratio_info["d_inst"], 2)

    def test_pd_instance_budgets_are_capped_by_num_devices(self):
        """A user-stated device budget also bounds the per-instance size."""
        config = CapacityPlanConfig(target_qps=10.0, mode="pd_ratio", device="TEST_DEVICE", num_devices=1024)
        planner = CapacityPlanner(config)
        # TEST_DEVICE seed is 256, so the sequence is capped at 1024.
        self.assertEqual(planner._pd_instance_budgets(), [256, 512, 1024])

    def test_pd_instance_budgets_fall_back_to_the_cap_below_the_seed(self):
        config = CapacityPlanConfig(target_qps=10.0, mode="pd_ratio", device="TEST_DEVICE", num_devices=64)
        planner = CapacityPlanner(config)
        # The seed (256) exceeds the cap, so the cap itself is the only budget.
        self.assertEqual(planner._pd_instance_budgets(), [64])

    def test_auto_search_budgets_come_from_comm_grid_not_tp_sizes(self):
        """A TP candidate is not an instance size: budgets must exceed 8 devices.

        Reusing --tp-sizes as the per-instance device count capped every instance
        at the largest TP (8 by default), so a model needing 16/32-device
        instances reported 'no feasible configuration' even on a grid that
        supports them, and PP>1 topologies (TP x PP) were unreachable.
        """
        config = CapacityPlanConfig(target_qps=10.0, mode="pd_ratio", device="ATLAS_800_A3_752T_128G_DIE")
        planner = CapacityPlanner(config)
        budgets = planner._pd_instance_budgets()
        self.assertGreater(max(budgets), 8)
        self.assertIn(32, budgets)
        # Bounded by the grid capacity (A3 = 768).
        self.assertEqual(max(budgets), 768)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_auto_search_reaches_instance_requiring_tp_times_pp_devices(self, mock_run):
        """TP=8 with PP=4 needs 32-device instances; the search must try them."""
        mock_run.return_value = pd.DataFrame([_pd_ratio_row()])
        config = CapacityPlanConfig(
            target_qps=10.0,
            mode="pd_ratio",
            device="ATLAS_800_A3_752T_128G_DIE",
            pd_instance_budgets=[8, 32],
        )
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "ATLAS_800_A3_752T_128G_DIE"
        args.tp_sizes = [8]
        args.pp_sizes = [4]
        args.enable_optimize_prefill_decode_ratio = True
        args.prefill_devices_per_instance = None
        args.decode_devices_per_instance = None

        planner._run_pd_ratio_auto(args)

        budgets_used = {call.args[1] for call in mock_run.call_args_list}
        self.assertIn(64, budgets_used)  # (32, 32)
        # The user's TP/PP search dims are still handed to the optimizer inside
        # each budget, so TP=8 x PP=4 is actually searched at the 32-device budget.
        for call in mock_run.call_args_list:
            self.assertEqual(call.args[0].tp_sizes, [8])
            self.assertEqual(call.args[0].pp_sizes, [4])

    def test_pd_instance_budgets_all_over_capacity_log_the_cause(self):
        """The caller's "see the preceding error" hint must have an actual error."""
        config = CapacityPlanConfig(
            target_qps=10.0,
            mode="pd_ratio",
            device="TEST_DEVICE",
            pd_instance_budgets=[4096, 8192],
        )
        planner = CapacityPlanner(config)
        with self.assertLogs("serving_cast.service.capacity_planner", "ERROR") as logs:
            budgets = planner._pd_instance_budgets()
        self.assertEqual(budgets, [])
        self.assertIn("all exceed comm_grid capacity", "\n".join(logs.output))

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_auto_search_without_runnable_budget(self, mock_run):
        config = CapacityPlanConfig(
            target_qps=10.0,
            mode="pd_ratio",
            device="TEST_DEVICE",
            pd_instance_budgets=[4096],
        )
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        with self.assertLogs("serving_cast.service.capacity_planner", "ERROR") as logs:
            result = planner._run_pd_ratio_auto(args)
        self.assertIsNone(result.minimum_devices)
        mock_run.assert_not_called()
        self.assertIn("no per-instance device budget", "\n".join(logs.output))

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_auto_search_all_oom(self, mock_run):
        mock_run.return_value = None
        args = MagicMock()
        args.device = "TEST_DEVICE"
        args.enable_optimize_prefill_decode_ratio = True
        args.prefill_devices_per_instance = None
        args.decode_devices_per_instance = None

        result = self.planner._run_pd_ratio_auto(args)
        self.assertIsNone(result.minimum_devices)

    def test_fixed_per_instance_no_results(self):
        args = MagicMock()
        args.prefill_devices_per_instance = 4
        args.decode_devices_per_instance = 2
        args.device = "TEST_DEVICE"
        args.enable_optimize_prefill_decode_ratio = True

        with patch.object(CapacityPlanner, "_run_optimizer_at_n") as mock_run:
            mock_run.return_value = None
            result = self.planner._run_pd_ratio_fixed(args, 4, 2)
        self.assertIsNone(result.minimum_devices)

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_auto_search_all_rows_but_no_slo(self, mock_run):
        """Rows exist everywhere but none pass the SLO -> suggest relaxing SLO."""
        mock_run.return_value = pd.DataFrame([_pd_ratio_row(ttft_p=9999, tpot_d=9999)])
        args = MagicMock()
        args.device = "TEST_DEVICE"
        args.enable_optimize_prefill_decode_ratio = True
        args.prefill_devices_per_instance = None
        args.decode_devices_per_instance = None

        with self.assertLogs("serving_cast.service.capacity_planner", "ERROR") as logs:
            result = self.planner._run_pd_ratio_auto(args)
        self.assertIsNone(result.minimum_devices)
        self.assertIn("rows found but none SLO-compliant", "\n".join(logs.output))

    @patch.object(CapacityPlanner, "_run_optimizer_at_n")
    def test_auto_search_mixed_infeasible_and_no_slo(self, mock_run):
        """One combo yields no results, another yields non-compliant rows."""
        config = CapacityPlanConfig(
            target_qps=100.0,
            ttft_limit=500.0,
            tpot_limit=50.0,
            mode="pd_ratio",
            pd_instance_budgets=[2, 4],
        )
        planner = CapacityPlanner(config)
        mock_run.side_effect = [
            None,
            pd.DataFrame([_pd_ratio_row(ttft_p=9999, tpot_d=9999)]),
            None,
            None,
        ]
        args = MagicMock()
        args.device = "TEST_DEVICE"
        args.enable_optimize_prefill_decode_ratio = True
        args.prefill_devices_per_instance = None
        args.decode_devices_per_instance = None

        with self.assertLogs("serving_cast.service.capacity_planner", "ERROR") as logs:
            result = planner._run_pd_ratio_auto(args)
        self.assertIsNone(result.minimum_devices)
        self.assertIn("produced no results, others had", "\n".join(logs.output))
        self.assertEqual(mock_run.call_count, 4)


# ==================================================================
# Result rendering
# ==================================================================


class TestPlannerRunDispatch(unittest.TestCase):
    """Test CapacityPlanner.run() dispatch logic."""

    def setUp(self):
        self.args = MagicMock()
        self.args.device = "TEST_DEVICE"

    def test_run_pd_ratio_mode(self):
        config = CapacityPlanConfig(target_qps=100.0, mode="pd_ratio")
        planner = CapacityPlanner(config)
        with patch.object(planner, "_run_pd_ratio") as mock_pd:
            mock_pd.return_value = CapacityPlanResult(minimum_devices=16)
            result = planner.run(self.args)
        mock_pd.assert_called_once_with(self.args)
        self.assertEqual(result.minimum_devices, 16)

    def test_run_single_mode(self):
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation")
        planner = CapacityPlanner(config)
        with patch.object(planner, "_run_single_mode") as mock_single:
            mock_single.return_value = CapacityPlanResult(minimum_devices=8)
            result = planner.run(self.args)
        mock_single.assert_called_once_with(self.args)
        self.assertEqual(result.minimum_devices, 8)


class TestRunOptimizerAtN(unittest.TestCase):
    """Test CapacityPlanner._run_optimizer_at_n()."""

    @patch("serving_cast.service.capacity_planner.ParallelRunner", create=True)
    def test_agg_mode_returns_concatenated_df(self, _mock_runner_cls):
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        mock_runner = MagicMock()
        summary1 = MagicMock()
        summary1.get_summary_df.return_value = pd.DataFrame([_agg_row()])
        summary2 = MagicMock()
        summary2.get_summary_df.return_value = pd.DataFrame([_agg_row(token_s=600)])
        mock_runner.run_agg.return_value = [summary1, summary2]

        with patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_runner):
            df = planner._run_optimizer_at_n(args, 8)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    def test_empty_summaries_returns_none(self):
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        mock_runner = MagicMock()
        mock_runner.run_agg.return_value = []
        with patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_runner):
            df = planner._run_optimizer_at_n(args, 8)
        self.assertIsNone(df)

    def test_memory_error_is_treated_as_infeasible_budget(self):
        """Memory failures mean the budget cannot run the model -> escalate."""
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        mock_runner = MagicMock()
        mock_runner.run_agg.side_effect = MemoryError("host memory exhausted")
        with (
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_runner),
            self.assertLogs("serving_cast.service.capacity_planner", "WARNING") as logs,
        ):
            df = planner._run_optimizer_at_n(args, 8)
        self.assertIsNone(df)
        self.assertIn("escalating", "\n".join(logs.output))

    def test_broken_process_pool_is_treated_as_infeasible_budget(self):
        from concurrent.futures.process import BrokenProcessPool

        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        mock_runner = MagicMock()
        mock_runner.run_agg.side_effect = BrokenProcessPool("worker died")
        with patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_runner):
            df = planner._run_optimizer_at_n(args, 8)
        self.assertIsNone(df)

    def test_programming_error_propagates_instead_of_masquerading_as_oom(self):
        """Real defects must not be downgraded to 'likely OOM' and skipped."""
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        mock_runner = MagicMock()
        mock_runner.run_agg.side_effect = KeyError("qps_req_s")
        with (
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_runner),
            self.assertRaises(KeyError),
        ):
            planner._run_optimizer_at_n(args, 8)

    def test_unwireable_device_count_is_skipped_without_running(self):
        """A device count above comm_grid capacity is not worth a doomed run."""
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation", device="TEST_DEVICE")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        with (
            patch("serving_cast.parallel_runner.ParallelRunner") as runner_cls,
            self.assertLogs("serving_cast.service.capacity_planner", "WARNING") as logs,
        ):
            df = planner._run_optimizer_at_n(args, 4096)
        self.assertIsNone(df)
        runner_cls.assert_not_called()
        self.assertIn("exceeds TEST_DEVICE comm_grid capacity 2048", "\n".join(logs.output))

    def test_validation_error_below_capacity_propagates(self):
        """A ValueError from the run itself is a defect, not an infeasible budget."""
        config = CapacityPlanConfig(target_qps=100.0, mode="aggregation", device="TEST_DEVICE")
        planner = CapacityPlanner(config)
        args = MagicMock()
        args.device = "TEST_DEVICE"

        mock_runner = MagicMock()
        mock_runner.run_agg.side_effect = ValueError("bad parallel configuration")
        with (
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_runner),
            self.assertRaises(ValueError),
        ):
            planner._run_optimizer_at_n(args, 8)


class TestRunPDRatioDispatch(unittest.TestCase):
    """Test CapacityPlanner._run_pd_ratio() dispatch."""

    def setUp(self):
        self.args = MagicMock()
        self.args.device = "TEST_DEVICE"

    def test_both_specified_calls_fixed(self):
        config = CapacityPlanConfig(
            target_qps=100.0,
            mode="pd_ratio",
            prefill_devices_per_instance=4,
            decode_devices_per_instance=2,
        )
        planner = CapacityPlanner(config)
        with patch.object(planner, "_run_pd_ratio_fixed") as mock_fixed:
            mock_fixed.return_value = CapacityPlanResult(minimum_devices=6)
            planner._run_pd_ratio(self.args)
        mock_fixed.assert_called_once_with(self.args, 4, 2)

    def test_both_none_calls_auto(self):
        config = CapacityPlanConfig(
            target_qps=100.0,
            mode="pd_ratio",
        )
        planner = CapacityPlanner(config)
        with patch.object(planner, "_run_pd_ratio_auto") as mock_auto:
            mock_auto.return_value = CapacityPlanResult(minimum_devices=6)
            planner._run_pd_ratio(self.args)
        mock_auto.assert_called_once_with(self.args)


class TestRenderResult(unittest.TestCase):
    """Test output rendering."""

    def test_render_with_result(self):
        config = CapacityPlanConfig(target_qps=100.0, ttft_limit=500.0, tpot_limit=50.0)
        result = CapacityPlanResult(
            minimum_devices=32,
            achieved_qps=105.0,
            per_die_qps=3.28,
            replicas_needed=4,
            best_config={
                "parallel": "TP=8 | PP=1 | DP=1",
                "per_replica_gpus": 8,
                "batch_size": 64,
                "ttft": 300.0,
                "tpot": 40.0,
            },
            scaling_table=[
                {
                    "parallel": "TP=8 | PP=1 | DP=1",
                    "per_replica_qps": 26.25,
                    "per_replica_gpus": 8,
                    "replicas_needed": 4,
                    "total_gpus": 32,
                    "per_die_qps": 3.28,
                    "ttft": 300.0,
                    "tpot": 40.0,
                    "batch_size": 64,
                    "concurrency": 64,
                }
            ],
        )
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            render_capacity_planning_result(result, config, logger=MagicMock())
        output = f.getvalue()
        self.assertIn("Capacity Planning Result", output)
        self.assertIn("32", output)
        self.assertIn("TP=8", output)

    def test_render_no_result(self):
        config = CapacityPlanConfig(target_qps=100.0)
        result = CapacityPlanResult(minimum_devices=None)
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            render_capacity_planning_result(result, config, logger=MagicMock())
        output = f.getvalue()
        self.assertIn("No feasible", output)


class TestRunCapacityPlanning(unittest.TestCase):
    """Test cli/inference/throughput_optimizer.py::run_capacity_planning."""

    def _make_args(self, **overrides):
        defaults = dict(
            target_qps=100.0,
            ttft_limits=500.0,
            tpot_limits=None,
            disagg=False,
            enable_optimize_prefill_decode_ratio=False,
            output_length=512,
            num_devices=8,
            device="TEST_DEVICE",
            tp_sizes=[8],
            prefill_devices_per_instance=None,
            decode_devices_per_instance=None,
        )
        defaults.update(overrides)
        return MagicMock(**defaults)

    @patch("cli.inference.throughput_optimizer.sys.argv", ["prog", "--num-devices", "8"])
    @patch("serving_cast.service.capacity_planner.CapacityPlanner")
    @patch("serving_cast.service.capacity_planner.render_capacity_planning_result")
    def test_aggregation_mode(self, mock_render, mock_planner_cls):
        args = self._make_args()
        mock_planner = MagicMock()
        mock_planner.run.return_value = CapacityPlanResult(minimum_devices=8)
        mock_planner_cls.return_value = mock_planner

        ret = run_capacity_planning(args, 0.0, MagicMock())
        self.assertEqual(ret, 0)
        config = mock_planner_cls.call_args[0][0]
        self.assertEqual(config.mode, "aggregation")
        mock_render.assert_called_once()

    @patch("cli.inference.throughput_optimizer.sys.argv", ["prog", "--num-devices", "8"])
    @patch("serving_cast.service.capacity_planner.CapacityPlanner")
    @patch("serving_cast.service.capacity_planner.render_capacity_planning_result")
    def test_disagg_prefill_mode(self, mock_render, mock_planner_cls):
        args = self._make_args(disagg=True, ttft_limits=500.0, tpot_limits=None)
        mock_planner = MagicMock()
        mock_planner.run.return_value = CapacityPlanResult(minimum_devices=8)
        mock_planner_cls.return_value = mock_planner

        ret = run_capacity_planning(args, 0.0, MagicMock())
        self.assertEqual(ret, 0)
        config = mock_planner_cls.call_args[0][0]
        self.assertEqual(config.mode, "disagg_prefill")

    @patch("cli.inference.throughput_optimizer.sys.argv", ["prog", "--num-devices", "8"])
    @patch("serving_cast.service.capacity_planner.CapacityPlanner")
    @patch("serving_cast.service.capacity_planner.render_capacity_planning_result")
    def test_pd_ratio_mode(self, mock_render, mock_planner_cls):
        args = self._make_args(enable_optimize_prefill_decode_ratio=True)
        mock_planner = MagicMock()
        mock_planner.run.return_value = CapacityPlanResult(minimum_devices=8)
        mock_planner_cls.return_value = mock_planner

        ret = run_capacity_planning(args, 0.0, MagicMock())
        self.assertEqual(ret, 0)
        config = mock_planner_cls.call_args[0][0]
        self.assertEqual(config.mode, "pd_ratio")


if __name__ == "__main__":
    unittest.main()
