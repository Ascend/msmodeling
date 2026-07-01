# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from collections import deque
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
from serving_cast.service.utils import OptimizerData, PrefillChunk
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
            [_PrefillGroup(count=1, chunk_index=1), _PrefillGroup(count=3, chunk_index=1)],
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
        self.assertAlmostEqual(metrics.output_throughput, 573.3944954)
        self.assertEqual(metrics.memory_left_gb, 6.0)
        self.assertEqual(metrics.prefill_latency, 10.0)
        self.assertEqual(metrics.prefill_last_latency, 4.0)
        self.assertEqual(metrics.prefill_memory_left_gb, 6.0)
        self.assertEqual(metrics.decode_latency, 2.0)
        self.assertEqual(metrics.prefill_breakdowns, "prefill")
        self.assertEqual(metrics.decode_breakdowns, "decode")

    def test_get_full_prefill_metrics_stops_before_decode_when_prefill_memory_is_negative(self):
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

    def test_get_full_prefill_metrics_stops_before_decode_when_remainder_memory_is_negative(self):
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

    def test_get_or_compute_decode_latency_applies_current_mtp_rate_to_cached_raw_record(self):
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


if __name__ == "__main__":
    unittest.main()
