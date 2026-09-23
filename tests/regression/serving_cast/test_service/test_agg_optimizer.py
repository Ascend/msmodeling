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
from collections import deque
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from serving_cast.service.agg_throughput_optimizer import (
    AggThroughputOptimizer,
    _DecodeGroup,
    _PrefillGroup,
    _ScheduleStep,
)
from serving_cast.service.latency_table import ForwardLatencyRecord, ForwardShapeKey
from serving_cast.service.scheduler import DecodeFirstWithSlack
from serving_cast.service.utils import BYTES_TO_GB, LengthBin, LengthDistribution, OptimizerData, PrefillChunk
from tensor_cast.core.input_generator import generate_inputs_varlen
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile

from .test_common import SimpleArgs
from .test_disagg_optimizer import (
    _make_pp_strategy,
    _pp_prefill_validation_cases,
    _pp_profile,
    _run_pp_scaling_case,
    _pp_schedule_estimates,
    _PPMetrics,
)

# Agg optimizer uses concurrency = batch_size * dp (without pp),
# so it needs its own scaling matrix with different expected values.
_AGG_PP_SCALING_MATRIX = (
    ("single-replica", 1, 2, 1, 2, 2),
    ("dp-replicated", 2, 2, 2, 2, 4),
)


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

    def test_initialize_sets_aggregation_scheduler(self):
        self.assertIsInstance(self.strategy.scheduler, DecodeFirstWithSlack)

    def test_count_front_prefill_group_counts_only_front_chunk_shape(self):
        pending_prefill = deque(
            [
                _PrefillGroup(count=2, chunk_index=0),
                _PrefillGroup(count=3, chunk_index=0),
                _PrefillGroup(count=4, chunk_index=1),
            ]
        )

        self.assertEqual(self.strategy._count_front_prefill_group(pending_prefill), 5)
        self.assertEqual(self.strategy._count_front_prefill_group(deque()), 0)

    def test_advance_prefill_groups_requeues_non_final_and_moves_final_to_decode(self):
        pending_prefill = deque(
            [
                _PrefillGroup(count=3, chunk_index=0),
                _PrefillGroup(count=2, chunk_index=1),
            ]
        )
        ready_decode = deque()
        chunk_plan = [
            PrefillChunk(index=0, query_len=3, seq_len=3),
            PrefillChunk(index=1, query_len=2, seq_len=5),
        ]

        first_token_time_sum, finished, max_finish_time = self.strategy._advance_prefill_groups(
            pending_prefill,
            ready_decode,
            chunk_plan,
            p_step=4,
            current_time=10.0,
            remaining_decode_tokens=2,
            first_token_time_sum=5.0,
            finished=1,
            max_finish_time=7.0,
        )

        self.assertEqual(first_token_time_sum, 15.0)
        self.assertEqual(finished, 1)
        self.assertEqual(max_finish_time, 7.0)
        self.assertEqual(
            list(pending_prefill),
            [
                _PrefillGroup(count=1, chunk_index=1),
                _PrefillGroup(count=3, chunk_index=1),
            ],
        )
        self.assertEqual(
            list(ready_decode),
            [_DecodeGroup(count=1, remaining_decode_tokens=2, first_token_time=10.0)],
        )

    def test_advance_prefill_groups_finishes_when_first_token_is_final_output(self):
        pending_prefill = deque([_PrefillGroup(count=2, chunk_index=0)])
        ready_decode = deque()
        chunk_plan = [PrefillChunk(index=0, query_len=5, seq_len=5)]

        first_token_time_sum, finished, max_finish_time = self.strategy._advance_prefill_groups(
            pending_prefill,
            ready_decode,
            chunk_plan,
            p_step=2,
            current_time=3.5,
            remaining_decode_tokens=0,
            first_token_time_sum=0.0,
            finished=0,
            max_finish_time=0.0,
        )

        self.assertEqual(first_token_time_sum, 7.0)
        self.assertEqual(finished, 2)
        self.assertEqual(max_finish_time, 3.5)
        self.assertEqual(list(pending_prefill), [])
        self.assertEqual(list(ready_decode), [])

    def test_advance_decode_groups_finishes_and_requeues_partial_groups(self):
        ready_decode = deque(
            [
                _DecodeGroup(count=3, remaining_decode_tokens=1, first_token_time=4.0),
                _DecodeGroup(count=2, remaining_decode_tokens=3, first_token_time=5.0),
            ]
        )

        tpot_sum, finished, max_finish_time = self.strategy._advance_decode_groups(
            ready_decode,
            d_step=4,
            current_time=10.0,
            initial_decode_tokens=3,
            tpot_sum=2.0,
            finished=1,
            max_finish_time=7.0,
        )

        self.assertEqual(tpot_sum, 8.0)
        self.assertEqual(finished, 4)
        self.assertEqual(max_finish_time, 10.0)
        self.assertEqual(
            list(ready_decode),
            [
                _DecodeGroup(count=1, remaining_decode_tokens=3, first_token_time=5.0),
                _DecodeGroup(count=1, remaining_decode_tokens=2, first_token_time=5.0),
            ],
        )

    def test_get_full_prefill_metrics_accounts_for_remainder_wave_and_memory(self):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=5,
            batch_size=3,
            max_batched_tokens=20,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        calls = []

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            calls.append((batch_size, is_decode))
            if not is_decode and batch_size == 2:
                return (10.0, 8.0, "prefill", None)
            if not is_decode and batch_size == 1:
                return (4.0, 6.0, "remainder", None)
            if is_decode and batch_size == 3:
                return (2.0, 7.0, "decode", None)
            raise AssertionError(f"unexpected call: batch_size={batch_size}, is_decode={is_decode}")

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            metrics = self.strategy._get_full_prefill_metrics(optimizer_data, concurrency=5)

        self.assertEqual(calls, [(2, False), (1, False), (3, True)])
        self.assertAlmostEqual(metrics.ttft, 16.8)
        self.assertAlmostEqual(metrics.tpot, 5.36)
        self.assertAlmostEqual(metrics.output_throughput, 932.8358209)
        self.assertEqual(metrics.memory_left_gb, 6.0)
        self.assertEqual(metrics.prefill_latency, 10.0)
        self.assertEqual(metrics.prefill_last_latency, 4.0)
        self.assertEqual(metrics.prefill_memory_left_gb, 6.0)
        self.assertEqual(metrics.decode_latency, 2.0)
        self.assertEqual(metrics.prefill_breakdowns, "prefill")
        self.assertEqual(metrics.decode_breakdowns, "decode")

    def test_get_full_prefill_metrics_uses_the_combined_dp_token_budget(self):
        """Each DP replica may independently consume max_batched_tokens."""
        self.strategy.dp = 4
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=5,
            batch_size=2,
            max_batched_tokens=20,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        calls = []

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            calls.append((batch_size, is_decode))
            if not is_decode and batch_size == 8:
                return (10.0, 8.0, "prefill", None)
            if is_decode and batch_size == 2:
                return (2.0, 7.0, "decode", None)
            raise AssertionError(f"unexpected call: batch_size={batch_size}, is_decode={is_decode}")

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            metrics = self.strategy._get_full_prefill_metrics(optimizer_data, concurrency=8)

        self.assertEqual(calls, [(8, False), (2, True)])
        self.assertEqual(metrics.ttft, 10.0)
        self.assertEqual(metrics.tpot, 4.0)
        self.assertEqual(metrics.output_throughput, 2000.0)

    def test_get_full_prefill_metrics_stops_before_decode_when_prefill_memory_is_negative(
        self,
    ):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=5,
            batch_size=3,
            max_batched_tokens=20,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        calls = []

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            calls.append((batch_size, is_decode))
            if is_decode:
                raise AssertionError("decode should not be computed after negative prefill memory")
            return (10.0, -1.0, "prefill-oom", None)

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            metrics = self.strategy._get_full_prefill_metrics(optimizer_data, concurrency=5)

        self.assertEqual(calls, [(2, False)])
        self.assertEqual(metrics.ttft, float("inf"))
        self.assertEqual(metrics.tpot, float("inf"))
        self.assertEqual(metrics.output_throughput, 0)
        self.assertLess(metrics.memory_left_gb, 0)
        self.assertEqual(metrics.decode_latency, 0)
        self.assertEqual(metrics.decode_breakdowns, "")

    def test_get_full_prefill_metrics_stops_before_decode_when_remainder_memory_is_negative(
        self,
    ):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=5,
            batch_size=3,
            max_batched_tokens=20,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        calls = []

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            calls.append((batch_size, is_decode))
            if is_decode:
                raise AssertionError("decode should not be computed after negative prefill memory")
            if batch_size == 1:
                return (4.0, -1.0, "remainder-oom", None)
            return (10.0, 8.0, "prefill", None)

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            metrics = self.strategy._get_full_prefill_metrics(optimizer_data, concurrency=5)

        self.assertEqual(calls, [(2, False), (1, False)])
        self.assertEqual(metrics.ttft, float("inf"))
        self.assertEqual(metrics.tpot, float("inf"))
        self.assertEqual(metrics.output_throughput, 0)
        self.assertLess(metrics.memory_left_gb, 0)
        self.assertEqual(metrics.prefill_memory_left_gb, -1.0)
        self.assertEqual(metrics.decode_latency, 0)

    def test_get_inference_info_variable_mode_splits_prefill_by_token_budget(self):
        optimizer_data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=1.0),
                    LengthBin(min_tokens=100, max_tokens=200, weight=1.0),
                ]
            ),
            output_length=5,
            batch_size=4,
            max_batched_tokens=200,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        batch_result = SimpleNamespace(
            execution_time_s={"analytic": 0.01},
            total_device_memory_gb=64.0,
            model_weight_size_gb=20.0,
            kv_cache_size_gb=4.0,
            model_activation_size_gb=1.0,
            reserved_memory_gb=0.0,
            device_memory_available_gb=8.0,
            breakdowns={},
        )
        with (
            patch.object(self.strategy.model_runner, "run_inference", return_value=batch_result) as mock_run_inference,
            patch.object(self.strategy, "_get_or_compute_latency", return_value=(2.0, 7.0, "decode", None)),
        ):
            summary = self.strategy.get_inference_info(optimizer_data)

        self.assertEqual(mock_run_inference.call_count, 2)
        self.assertEqual(
            [
                [(request.query_len, request.seq_len) for request in call.args[0]]
                for call in mock_run_inference.call_args_list
            ],
            [
                [(50, 50), (50, 50), (100, 100)],
                [(50, 150), (150, 150)],
            ],
        )
        self.assertTrue(
            all(
                call.kwargs["generate_inputs_func"] is generate_inputs_varlen
                for call in mock_run_inference.call_args_list
            )
        )
        df = summary.get_summary_df()
        self.assertEqual(list(df["num_input_tokens"]), ["all", 50, 150])
        self.assertEqual(list(df["samples"]), [4, 2, 2])
        self.assertEqual(df.iloc[0]["prefill_num_chunks"], 2)
        self.assertEqual(df.iloc[0]["ttft"], 15.0)
        self.assertEqual(df.iloc[0]["tpot"], 5.0)
        self.assertTrue(pd.isna(df.iloc[1]["ttft"]))

    def test_variable_length_decode_uses_speculative_shape_resolution(self):
        batch_result = SimpleNamespace(
            execution_time_s={"analytic": 0.01},
            total_device_memory_gb=64.0,
            model_weight_size_gb=20.0,
            kv_cache_size_gb=4.0,
            model_activation_size_gb=1.0,
            reserved_memory_gb=0.0,
            device_memory_available_gb=8.0,
            breakdowns={},
        )

        for block_field, acceptance_field in (
            ("dspark", "dspark_acceptance_length"),
            ("dflash", "dflash_acceptance_length"),
        ):
            with self.subTest(block_field=block_field):
                optimizer_data = OptimizerData(
                    length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=1000, weight=1.0)]),
                    output_length=256,
                    batch_size=4,
                    **{f"{block_field}_block_size": 8, acceptance_field: 5.0},
                )
                resolved_shapes = []

                def fake_latency(batch_size, data, is_decode=False, **kwargs):
                    resolved_shapes.append(self.strategy._resolve_forward_shape(data, is_decode, **kwargs))
                    return 2.0, 7.0, "decode", None

                with (
                    patch.object(
                        self.strategy,
                        "_get_batched_forward_info",
                        return_value=([(batch_result, 4)], [{"samples": 4}]),
                    ),
                    patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency) as latency,
                ):
                    self.strategy._get_batched_full_prefill_metrics(optimizer_data, concurrency=4, chunk_plan=[])

                self.assertTrue(latency.call_args.kwargs["is_decode"])
                self.assertNotIn("query_len", latency.call_args.kwargs)
                self.assertNotIn("seq_len", latency.call_args.kwargs)
                self.assertEqual(resolved_shapes, [(8, 636)])

    def test_get_inference_info_variable_mode_aggregates_prefill_breakdowns(self):
        optimizer_data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=1.0),
                    LengthBin(min_tokens=100, max_tokens=200, weight=1.0),
                ]
            ),
            output_length=5,
            batch_size=4,
            max_batched_tokens=200,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def make_batch_result(latency_s, breakdowns):
            return SimpleNamespace(
                execution_time_s={"analytic": latency_s},
                total_device_memory_gb=64.0,
                model_weight_size_gb=20.0,
                kv_cache_size_gb=4.0,
                model_activation_size_gb=1.0,
                reserved_memory_gb=0.0,
                device_memory_available_gb=8.0,
                breakdowns=breakdowns,
            )

        # Two chunks with different latencies: the aggregated breakdown must be
        # normalized per chunk and weighted by each chunk's latency.
        batch_results = [
            make_batch_result(0.01, {"prefill": {"Mem": 1.0, "Comm": 1.0, "Cube": 1.0, "Vec": 1.0}}),
            make_batch_result(0.03, {"prefill": {"Mem": 4.0, "Comm": 0.0, "Cube": 0.0, "Vec": 0.0}}),
        ]
        with (
            patch.object(self.strategy.model_runner, "run_inference", side_effect=batch_results),
            patch.object(self.strategy, "_get_or_compute_latency", return_value=(2.0, 7.0, "decode", None)),
        ):
            summary = self.strategy.get_inference_info(optimizer_data)

        df = summary.get_summary_df()
        # chunk1: 10ms at 25% each; chunk2: 30ms at 100% Mem ->
        # Mem (25*10+100*30)/40=81.25, others (25*10+0)/40=6.25.
        self.assertEqual(
            df.iloc[0]["percentage_breakdowns(p)"],
            "Mem 81.25 | Comm 6.25 | Cube 6.25 | Vec 6.25",
        )
        self.assertEqual(df.iloc[0]["percentage_breakdowns(d)"], "decode")

    def test_simulate_chunked_prefill_accumulates_scheduler_metrics(self):
        optimizer_data = OptimizerData(
            input_length=5,
            output_length=3,
            batch_size=1,
            max_batched_tokens=3,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        chunk_plan = optimizer_data.get_prefill_chunk_plan()
        calls = []

        class ScriptedScheduler:
            def __init__(self):
                self.decisions = deque([(2, 0), (2, 0), (0, 2), (0, 2)])
                self.states = []

            def decide(self, state):
                self.states.append(state)
                p_step, d_step = self.decisions.popleft()
                return SimpleNamespace(p_step=p_step, d_step=d_step)

            def step_latency(self, prefill_step_latency, decode_step_latency):
                return max(prefill_step_latency, decode_step_latency)

        scheduler = ScriptedScheduler()

        def fake_record(key, optimizer_data):
            calls.append((key.model_concurrency, key.is_decode, key.query_len, key.seq_len))
            if key.is_decode:
                return ForwardLatencyRecord(5.0, 7.0, "decode")
            if key.query_len == 3:
                return ForwardLatencyRecord(10.0, 9.0, "prefill-0")
            if key.query_len == 2:
                return ForwardLatencyRecord(20.0, 8.0, "prefill-1")
            raise AssertionError(f"unexpected key: {key}")

        with patch.object(self.strategy, "_compute_forward_latency_record", side_effect=fake_record):
            metrics = self.strategy._simulate_chunked_prefill(
                optimizer_data,
                chunk_plan,
                concurrency=2,
                scheduler=scheduler,
            )

        self.assertEqual(
            calls,
            [
                (2, False, 3, 3),
                (2, False, 2, 5),
                (2, True, 1, 7),
            ],
        )
        self.assertEqual(
            [(state.ready_decode, state.pending_prefill, state.chunk_query_len) for state in scheduler.states],
            [(0, 2, 3), (0, 2, 2), (2, 0, 3), (2, 0, 3)],
        )
        self.assertEqual(metrics.ttft, 30.0)
        self.assertEqual(metrics.tpot, 5.0)
        self.assertEqual(metrics.output_throughput, 150.0)
        self.assertEqual(metrics.memory_left_gb, 7.0)
        self.assertEqual(metrics.prefill_memory_left_gb, 8.0)
        self.assertEqual(metrics.prefill_latency, 20.0)
        self.assertEqual(metrics.prefill_last_latency, 20.0)
        self.assertEqual(metrics.decode_latency, 5.0)
        self.assertEqual(metrics.prefill_breakdowns, "prefill-0")
        self.assertEqual(metrics.decode_breakdowns, "decode")

    def test_simulate_chunked_prefill_rejects_scheduler_without_progress(self):
        optimizer_data = OptimizerData(input_length=5, output_length=3, batch_size=1, max_batched_tokens=3)

        class StalledScheduler:
            def decide(self, state):
                return SimpleNamespace(p_step=0, d_step=0)

            def step_latency(self, prefill_step_latency, decode_step_latency):
                return 0

        with self.assertRaises(RuntimeError):
            self.strategy._simulate_chunked_prefill(
                optimizer_data,
                optimizer_data.get_prefill_chunk_plan(),
                concurrency=1,
                scheduler=StalledScheduler(),
            )

    def test_collect_schedule_keys_preserves_step_order_and_skips_empty_slots(self):
        prefill_key = ForwardShapeKey(False, 2, 3, 3)
        decode_key = ForwardShapeKey(True, 2, 1, 7)
        later_prefill_key = ForwardShapeKey(False, 1, 2, 5)
        schedule = [
            _ScheduleStep(prefill_key=prefill_key, decode_key=None, p_step=2, d_step=0),
            _ScheduleStep(prefill_key=None, decode_key=decode_key, p_step=0, d_step=2),
            _ScheduleStep(prefill_key=later_prefill_key, decode_key=decode_key, p_step=1, d_step=1),
        ]

        keys = self.strategy._collect_schedule_keys(schedule)

        self.assertEqual(keys, [prefill_key, decode_key, later_prefill_key, decode_key])

    def test_simulate_chunked_prefill_stops_when_any_record_memory_is_negative(self):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=1,
            batch_size=1,
            max_batched_tokens=4,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        chunk_plan = optimizer_data.get_prefill_chunk_plan()
        calls = []

        def fake_record(key, optimizer_data):
            calls.append((key.query_len, key.seq_len))
            if key.seq_len == 8:
                return ForwardLatencyRecord(2.0, -1.0, "oom")
            if key.seq_len == 10:
                raise AssertionError("chunk after negative memory should not be computed")
            return ForwardLatencyRecord(1.0, 1.0, "ok")

        with patch.object(self.strategy, "_compute_forward_latency_record", side_effect=fake_record):
            metrics = self.strategy._simulate_chunked_prefill(
                optimizer_data,
                chunk_plan,
                concurrency=1,
                scheduler=self.strategy.scheduler,
            )

        self.assertEqual(calls, [(4, 4), (4, 8)])
        self.assertLess(metrics.memory_left_gb, 0)

    def test_get_or_compute_prefill_latency_cached(self):
        """Test _get_or_compute_prefill_latency with cached value"""
        optimizer_data = OptimizerData(input_length=10, output_length=10)
        key = self.strategy._make_forward_shape_key(4, optimizer_data, is_decode=False)
        self.strategy._forward_record_cache[key] = ForwardLatencyRecord(50.0, 2.0, "")

        latency, memory_left, _, _ = self.strategy._get_or_compute_latency(4, optimizer_data, is_decode=False)

        self.assertEqual(latency, 50.0)
        self.assertEqual(memory_left, 2.0)

    def test_get_or_compute_prefill_latency_new(self):
        """Test _get_or_compute_prefill_latency with new value"""
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=10,
        )
        latency, memory_left, breakdown, _ = self.strategy._get_or_compute_latency(4, optimizer_data, is_decode=False)

        # Should cache the result
        key = self.strategy._make_forward_shape_key(4, optimizer_data, is_decode=False)
        record = self.strategy._forward_record_cache[key]
        self.assertEqual(record.latency_ms, latency)
        self.assertEqual(record.memory_left_gb, memory_left)
        self.assertEqual(record.breakdowns, breakdown)

    def test_get_or_compute_decode_latency_cached(self):
        """Test _get_or_compute_decode_latency with cached value"""
        optimizer_data = OptimizerData(input_length=10, output_length=10)
        key = self.strategy._make_forward_shape_key(4, optimizer_data, is_decode=True)
        self.strategy._forward_record_cache[key] = ForwardLatencyRecord(10.0, 2.0, "")

        latency, memory_left, _, _ = self.strategy._get_or_compute_latency(4, optimizer_data, is_decode=True)

        self.assertEqual(latency, 10.0)
        self.assertEqual(memory_left, 2.0)

    def test_get_or_compute_decode_latency_applies_current_mtp_rate_to_cached_raw_record(
        self,
    ):
        optimizer_data_a = OptimizerData(
            input_length=10,
            output_length=10,
            batch_size=4,
            num_mtp_tokens=2,
            mtp_acceptance_rate=[0.5, 0.5],
        )
        optimizer_data_b = OptimizerData(
            input_length=10,
            output_length=10,
            batch_size=4,
            num_mtp_tokens=2,
            mtp_acceptance_rate=[1.0, 1.0],
        )
        calls = []

        class DummyMetrics:
            execution_time_s = {"analytic": 0.1}
            total_device_memory_gb = 64.0
            model_weight_size_gb = 20.0
            kv_cache_size_gb = 4.0
            model_activation_size_gb = 1.0
            reserved_memory_gb = 10.0
            device_memory_available_gb = 2.0
            breakdowns = {}

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            calls.append((concurrency, is_decode, query_len, seq_len))
            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            latency_a, _, _, _ = self.strategy._get_or_compute_latency(
                4,
                optimizer_data_a,
                is_decode=True,
                query_len=3,
                seq_len=20,
            )
            latency_b, _, _, _ = self.strategy._get_or_compute_latency(
                4,
                optimizer_data_b,
                is_decode=True,
                query_len=3,
                seq_len=20,
            )

        self.assertEqual(calls, [(4, True, 3, 20)])
        self.assertAlmostEqual(latency_a, 50.0)
        self.assertAlmostEqual(latency_b, 100.0 / 3.0)

    def test_get_or_compute_latency_separates_image_shape_cache_entries(self):
        optimizer_data_a = OptimizerData(
            input_length=10,
            output_length=10,
            batch_size=4,
            image_height=224,
            image_width=224,
        )
        optimizer_data_b = OptimizerData(
            input_length=10,
            output_length=10,
            batch_size=4,
            image_height=448,
            image_width=224,
        )
        calls = []

        class DummyMetrics:
            execution_time_s = {"analytic": 0.01}
            total_device_memory_gb = 64.0
            model_weight_size_gb = 20.0
            kv_cache_size_gb = 4.0
            model_activation_size_gb = 1.0
            reserved_memory_gb = 10.0
            device_memory_available_gb = 2.0
            breakdowns = {}

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            calls.append((optimizer_data.image_height, optimizer_data.image_width))
            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            self.strategy._get_or_compute_latency(4, optimizer_data_a, is_decode=False)
            self.strategy._get_or_compute_latency(4, optimizer_data_b, is_decode=False)

        self.assertEqual(calls, [(224, 224), (448, 224)])

    def test_get_inference_info_prefill_batch_size_uses_effective_input_length(self):
        optimizer_data = OptimizerData(
            input_length=200,
            output_length=10,
            batch_size=2,
            max_batched_tokens=200,
            prefix_cache_hit_rate=0.5,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        captured_calls = []

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            captured_calls.append((batch_size, is_decode))
            return (1.0, 1.0, "", None)

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            self.strategy.get_inference_info(optimizer_data)

        self.assertEqual(captured_calls[0], (2, False))

    def test_get_inference_info_uses_chunked_prefill_for_long_prompt(self):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=3,
            batch_size=2,
            max_batched_tokens=4,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_record(key, optimizer_data):
            return ForwardLatencyRecord(1.0, 1.0, "")

        with patch.object(self.strategy, "_compute_forward_latency_record", side_effect=fake_record):
            summary = self.strategy.get_inference_info(optimizer_data)

        row = summary.get_summary_df().iloc[0]
        self.assertEqual(row["effective_input_length"], 10)
        self.assertEqual(row["max_batched_tokens"], 4)
        self.assertEqual(row["prefill_num_chunks"], 3)

    def test_get_inference_info_passes_configured_scheduler_to_chunked_prefill(self):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=3,
            batch_size=2,
            max_batched_tokens=4,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        custom_scheduler = object()
        self.strategy.scheduler = custom_scheduler
        metrics = SimpleNamespace(
            ttft=1.0,
            tpot=1.0,
            output_throughput=1.0,
            memory_left_gb=1.0,
            prefill_latency=1.0,
            prefill_last_latency=1.0,
            prefill_memory_left_gb=1.0,
            decode_latency=1.0,
            prefill_breakdowns="",
            decode_breakdowns="",
        )

        with patch.object(self.strategy, "_simulate_chunked_prefill", return_value=metrics) as mock_simulate:
            self.strategy.get_inference_info(optimizer_data)

        self.assertIs(mock_simulate.call_args.args[3], custom_scheduler)

    def test_get_inference_info_acc_search_records_metrics_search_info(self):
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=10,
            batch_size=2,
            max_batched_tokens=20,
            num_devices=1,
            serving_cost=0,
            concurrency_search_strategy="linear_exponential",
        )
        metrics = SimpleNamespace(
            ttft=7.0,
            tpot=3.0,
            output_throughput=100.0,
            memory_left_gb=8.0,
            prefill_latency=1.0,
            prefill_last_latency=1.0,
            prefill_memory_left_gb=8.0,
            decode_latency=1.0,
            prefill_breakdowns="",
            decode_breakdowns="",
        )

        self.strategy.model_runner.total_device_memory_gb = 20.0
        self.strategy.model_runner.model_weight_size_gb = 5.0
        self.strategy.model_runner.user_input.reserved_memory_gb = 1.0

        with patch.object(self.strategy, "_get_full_prefill_metrics", return_value=metrics):
            summary = self.strategy.get_inference_info(optimizer_data)

        search_info = summary.get_search_info()
        self.assertEqual(search_info["per_request_memory_gb"], 3.0)
        self.assertEqual(search_info["device_memory_available_gb"], 8.0)
        self.assertEqual(search_info["ttft"], 7.0)
        self.assertEqual(search_info["tpot"], 3.0)

    def test_get_inference_info_uses_effective_prefill_memory_for_early_stop(self):
        optimizer_data = OptimizerData(
            input_length=32,
            output_length=256,
            batch_size=1,
            max_batched_tokens=8192,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            if not is_decode and batch_size == 256:
                return (1000.0, -37.15, "wave", None)
            if not is_decode and batch_size == 1:
                return (342.0, 12.5, "effective", None)
            if is_decode and batch_size == 1:
                return (15.0, 9.0, "decode", None)
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
            max_batched_tokens=256,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_latency(batch_size, optimizer_data, is_decode=False, **kwargs):
            if not is_decode and batch_size == 8:
                return (1000.0, -37.15, "wave", None)
            if not is_decode and batch_size == 1:
                return (342.0, 12.5, "remainder", None)
            if is_decode and batch_size == 9:
                return (15.0, 9.0, "decode", None)
            raise AssertionError(f"unexpected call: batch_size={batch_size}, is_decode={is_decode}")

        with patch.object(self.strategy, "_get_or_compute_latency", side_effect=fake_latency):
            summary = self.strategy.get_inference_info(optimizer_data)

        self.assertTrue(summary.check_early_stop_flag())

    def test_chunked_prefill_decode_can_overlap_before_all_prefill_finishes(self):
        optimizer_data = OptimizerData(
            input_length=5,
            output_length=2,
            batch_size=2,
            max_batched_tokens=3,
            num_devices=1,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_record(key, optimizer_data):
            return ForwardLatencyRecord(1.0, 1.0, "")

        with patch.object(self.strategy, "_compute_forward_latency_record", side_effect=fake_record):
            summary = self.strategy.get_inference_info(optimizer_data)

        row = summary.get_summary_df().iloc[0]
        self.assertEqual(row["ttft"], 3.5)
        self.assertEqual(row["tpot"], 1.0)
        self.assertEqual(row["token/s"], 800.0)


_agg_pp_profile = partial(_pp_profile, include_transfers=False)
_AggPPMetrics = _PPMetrics
_make_pp_agg_strategy = partial(_make_pp_strategy, AggThroughputOptimizer)


class TestAggPipelineParallel(unittest.TestCase):
    def test_scaling_and_overlap_flag_use_shared_matrix(self):
        for name, dp, pp, tp, batch_size, expected_concurrency in _AGG_PP_SCALING_MATRIX:
            with self.subTest(name=name):
                summary, row = _run_pp_scaling_case(
                    AggThroughputOptimizer,
                    dp=dp,
                    pp=pp,
                    tp=tp,
                    batch_size=batch_size,
                )

                self.assertEqual(row["concurrency"], expected_concurrency)
                self.assertLessEqual(
                    abs(row["token/s/device"] - row["token/s"] / (dp * pp * tp)),
                    0.001 + 1e-12,
                )
                self.assertTrue(summary.get_pp_mixed_pd_overlap_approx())
                self.assertFalse(summary.check_early_stop_flag())

    def test_overlap_flag_survives_run_summary_rebuild(self):
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        optimizer_data = OptimizerData(
            input_length=4,
            output_length=8,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            summary = strategy.run(optimizer_data, batch_range=[1, 2])

        self.assertIsNotNone(summary)
        self.assertTrue(summary.get_pp_mixed_pd_overlap_approx())

    def test_formula_keeps_prefill_steady_request_ttft_and_decode_interval_separate(self):
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        _, repeated = _pp_schedule_estimates(
            makespan_s=0.06,
            interval_s=0.05,
            worst_tpot_s=0.08,
        )
        optimizer_data = OptimizerData(
            input_length=4,
            output_length=4,
            batch_size=2,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with (
            patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)),
            patch(
                "serving_cast.service.base_throughput_optimizer.estimate_repeated_pipeline",
                return_value=repeated,
            ),
        ):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

        # Prefill ttft comes from the steady wave's request-level mean
        # (steady completions (0.03, 0.06) - start 0.0 -> 45.0ms), not the
        # wave-1 makespan (60.0ms); decode tpot/interval stay the steady
        # worst_tpot / measured_interval.
        expected_e2el = 45.0 + 50.0 * (4 - 1)
        self.assertEqual(row["ttft"], 45.0)
        self.assertEqual(row["tpot"], 80.0)
        self.assertAlmostEqual(row["token/s"], 1000.0 * 4 * 2 / expected_e2el, places=2)

    def test_pp_decode_applies_mtp_fold(self):
        """PP>1 decode TPOT and throughput are folded by (accept+1) when MTP is enabled."""
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        first_wave, repeated = _pp_schedule_estimates(
            makespan_s=0.06,
            interval_s=0.05,
            worst_tpot_s=0.08,
        )
        optimizer_data = OptimizerData(
            input_length=4,
            output_length=4,
            batch_size=2,
            max_batched_tokens=2048,
            num_mtp_tokens=2,
            speculative_method="mtp",
            acceptance_length=1.5,
            mtp_acceptance_rate=[],
        )
        with (
            patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)),
            patch(
                "serving_cast.service.base_throughput_optimizer.estimate_forward_pipeline",
                return_value=first_wave,
            ),
            patch(
                "serving_cast.service.base_throughput_optimizer.estimate_repeated_pipeline",
                return_value=repeated,
            ),
        ):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

        # fold = clamp(1.5, 0, 2) + 1 = 2.5
        fold = 2.5
        expected_tpot = 80.0 / fold  # 32.0
        expected_interval_ms = 50.0 / fold  # 20.0
        # Prefill ttft is the steady-wave request-level mean (0.045s), and the
        # decode interval is folded by (accept+1).
        expected_e2el = 45.0 + expected_interval_ms * 3  # 105.0
        self.assertAlmostEqual(row["ttft"], 45.0, places=6)
        self.assertAlmostEqual(row["tpot"], expected_tpot, places=6)
        self.assertAlmostEqual(row["token/s"], 1000.0 * 4 * 2 / expected_e2el, places=2)

    def test_pp_decode_applies_block_fold_dflash_and_dspark(self):
        """PP>1 decode TPOT is folded by (clamp(accept, 0, block-1) + 1) for DFlash/DSpark."""
        for method, block_field, acceptance_field in (
            ("dflash", "dflash_block_size", "dflash_acceptance_length"),
            ("dspark", "dspark_block_size", "dspark_acceptance_length"),
        ):
            with self.subTest(method=method):
                strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
                profile = _agg_pp_profile((2.0, 2.0))
                first_wave, repeated = _pp_schedule_estimates(
                    makespan_s=0.06,
                    interval_s=0.05,
                    worst_tpot_s=0.08,
                )
                optimizer_data = OptimizerData(
                    input_length=4,
                    output_length=4,
                    batch_size=2,
                    max_batched_tokens=2048,
                    num_mtp_tokens=0,
                    speculative_method=method,
                    acceptance_length=None,
                    mtp_acceptance_rate=[],
                )
                setattr(optimizer_data, block_field, 6)  # n = block - 1 = 5
                setattr(optimizer_data, acceptance_field, 3.5)

                with (
                    patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)),
                    patch(
                        "serving_cast.service.base_throughput_optimizer.estimate_forward_pipeline",
                        return_value=first_wave,
                    ),
                    patch(
                        "serving_cast.service.base_throughput_optimizer.estimate_repeated_pipeline",
                        return_value=repeated,
                    ),
                ):
                    row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

                # fold = clamp(3.5, 0, 5) + 1 = 4.5
                fold = 4.5
                expected_tpot = 80.0 / fold
                expected_interval_ms = 50.0 / fold
                # Prefill ttft is the steady-wave request-level mean (0.045s).
                expected_e2el = 45.0 + expected_interval_ms * 3
                # summary rows round to 3 decimals; 80/4.5 is not exact
                self.assertAlmostEqual(row["ttft"], 45.0, places=6)
                self.assertAlmostEqual(row["tpot"], expected_tpot, delta=1e-2)
                self.assertAlmostEqual(row["token/s"], 1000.0 * 4 * 2 / expected_e2el, delta=1e-1)

    def test_single_token_skips_decode_profile_schedule_and_oom(self):
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        prefill_profile = _agg_pp_profile((2.0, 2.0))
        decode_oom_profile = _pp_profile(
            (2.0, 2.0),
            runtime_peak_bytes=10_000_000_000_000,
            include_transfers=False,
        )
        decode_calls = []

        def fake_forward(concurrency, optimizer_data, is_decode, **kwargs):
            if is_decode:
                decode_calls.append(concurrency)
                return _AggPPMetrics(decode_oom_profile)
            return _AggPPMetrics(prefill_profile)

        optimizer_data = OptimizerData(
            input_length=4,
            output_length=1,
            batch_size=2,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", side_effect=fake_forward):
            summary = strategy.get_inference_info(optimizer_data)

        row = summary.get_summary_df().iloc[0]
        self.assertEqual(decode_calls, [])
        self.assertEqual(row["tpot"], 0.0)
        # Real scheduler on 2 microbatches x 2 stages (2.0s, no transfers):
        # steady wave completions (8.0, 10.0) - start 4.0 -> request-level
        # mean 5.0s (NOT the wave-1 makespan 6.0s).
        self.assertAlmostEqual(row["token/s"], 1000.0 * 2 / 5000.0, places=3)
        self.assertFalse(summary.check_early_stop_flag())
        self.assertFalse(summary.get_pp_mixed_pd_overlap_approx())

    def test_mixed_lifecycle_evicts_only_cross_phase_peak(self):
        from serving_cast.service.optimizer_summary import EARLY_STOP_DECODE_OOM
        from serving_cast.service.pipeline_schedule import (
            estimate_pipeline_stage_memory_across_profiles,
        )

        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        strategy.model_runner.total_device_memory_gb = 3500.0 / BYTES_TO_GB
        prefill_profile = _pp_profile((2.0, 2.0), runtime_peak_bytes=2000, include_transfers=False)
        decode_profile = _pp_profile(
            (2.0, 2.0),
            runtime_peak_bytes=2000,
            kv_cache_bytes=1000,
            include_transfers=False,
        )
        decode_only = estimate_pipeline_stage_memory_across_profiles(
            (decode_profile, decode_profile),
            num_microbatches=2,
            resident_microbatches=2,
            device_memory_bytes=3500,
            reserved_memory_bytes=0,
        )
        self.assertFalse(decode_only.exceeds_budget)

        def fake_forward(concurrency, optimizer_data, is_decode, **kwargs):
            return _AggPPMetrics(decode_profile if is_decode else prefill_profile)

        optimizer_data = OptimizerData(
            input_length=4,
            output_length=2,
            batch_size=2,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", side_effect=fake_forward):
            summary = strategy.get_inference_info(optimizer_data)

        self.assertTrue(summary.check_early_stop_flag())
        self.assertEqual(summary.get_early_stop_reason(), EARLY_STOP_DECODE_OOM)

    def test_prefill_validation_matrix(self):
        from serving_cast.service.utils import UnsupportedPPConfigurationError

        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        for name, optimizer_data, expected in _pp_prefill_validation_cases(disaggregated=False):
            with self.subTest(name=name):
                with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
                    with self.assertRaises(UnsupportedPPConfigurationError) as ctx:
                        strategy.get_inference_info(optimizer_data)
                self.assertIn(expected, str(ctx.exception))

    def test_chunked_prefill_is_supported_for_pp(self):
        """PP>1 + chunked prefill is supported: each chunk is a pipeline microbatch."""
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        optimizer_data = OptimizerData(
            input_length=10,
            output_length=8,
            batch_size=1,
            max_batched_tokens=4,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            summary = strategy.get_inference_info(optimizer_data)
        self.assertFalse(summary.check_early_stop_flag())

    def test_batch_budget_returns_early_stop_not_exception(self):
        """Batch exceeding token budget returns early-stop, not UnsupportedPPConfigurationError."""
        from serving_cast.service.optimizer_summary import EARLY_STOP_PREFILL_OOM

        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        optimizer_data = OptimizerData(
            input_length=1024,
            output_length=8,
            batch_size=4,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            summary = strategy.get_inference_info(optimizer_data)
        self.assertTrue(summary.check_early_stop_flag())
        self.assertEqual(summary.get_early_stop_reason(), EARLY_STOP_PREFILL_OOM)


if __name__ == "__main__":
    unittest.main()
