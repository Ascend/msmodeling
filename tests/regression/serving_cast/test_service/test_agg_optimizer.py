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
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from serving_cast.service.agg_throughput_optimizer import (
    AggThroughputOptimizer,
    _overlap_pipeline_profiles,
    _reserved_kv_cache_size_gb,
    _reserved_pp_kv_bytes,
    _resident_pp_kv_bytes,
)
from serving_cast.service.aggregation_timeline import RequestShape, TimelineResult
from serving_cast.service.latency_table import ForwardLatencyRecord
from serving_cast.service.optimizer_summary import EARLY_STOP_TPOT_LIMIT
from serving_cast.service.scheduler import DecodeFirstWithSlack, StepDecision
from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

from .test_disagg_optimizer import (
    _make_pp_strategy,
    _pp_prefill_validation_cases,
    _pp_profile,
    _run_pp_scaling_case,
    _PPMetrics,
)

# Agg optimizer uses concurrency = batch_size * dp (without pp),
# so it needs its own scaling matrix with different expected values.
_AGG_PP_SCALING_MATRIX = (
    ("single-replica", 1, 2, 1, 2, 2),
    ("dp-replicated", 2, 2, 2, 2, 4),
)


class TestAggThroughputOptimizer(unittest.TestCase):
    """Finite-batch metrics use timestamps rather than inclusive latency formulas."""

    def test_empty_timeline_metrics_are_unavailable(self):
        timeline = TimelineResult(requests=[])
        self.assertEqual(timeline.ttft, float("inf"))
        self.assertEqual(timeline.e2e, float("inf"))

    def make_strategy(self, dp=1, latency=None):
        strategy = AggThroughputOptimizer()
        strategy.dp, strategy.tp, strategy.ep, strategy.pp = dp, 1, 1, 1
        strategy.num_mtp_tokens = 0
        strategy.scheduler = DecodeFirstWithSlack()
        strategy._mixed_record_cache = {}
        strategy._resident_kv_cache = {}
        strategy.is_moe_model = False
        strategy.model_runner = SimpleNamespace(
            user_input=SimpleNamespace(
                block_size=128,
                model_id="test",
                device="test",
                reserved_memory_gb=0,
                quantize_linear_action="none",
                quantize_attention_action="none",
            ),
            model=SimpleNamespace(is_vl_model=False, model_config=SimpleNamespace(parallel_config=SimpleNamespace())),
        )
        strategy._compute_forward_latency_record = lambda key, data: ForwardLatencyRecord(
            latency(key.is_decode) if latency else (2.0 if key.is_decode else 10.0), 20.0, "pure"
        )
        strategy.model_runner.run_inference = lambda requests, **kw: requests
        strategy._build_forward_latency_record = lambda requests: ForwardLatencyRecord(17.0, 20.0, "mixed")
        return strategy

    def data(self, **kw):
        values = dict(
            input_length=10,
            output_length=3,
            batch_size=2,
            max_batched_tokens=10,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        values.update(kw)
        return OptimizerData(**values)

    def test_one_request_tpot_excludes_prefill(self):
        strategy = self.make_strategy()
        metrics = strategy._get_timeline_metrics(self.data(batch_size=1), 1)
        self.assertEqual((metrics.ttft, metrics.tpot, metrics.e2e, metrics.makespan), (10, 2, 14, 14))
        self.assertAlmostEqual(metrics.output_throughput, 3000 / 14)

    def test_one_output_has_no_decode_and_zero_tpot(self):
        strategy = self.make_strategy()
        metrics = strategy._get_timeline_metrics(self.data(output_length=1), 2)
        self.assertEqual((metrics.ttft, metrics.tpot, metrics.e2e, metrics.makespan), (15, 0, 15, 20))
        self.assertEqual(metrics.output_throughput, 100)
        self.assertEqual(metrics.decode_latency, 0)

    def test_mixed_forward_uses_max_of_pure_costs(self):
        strategy = self.make_strategy()
        metrics = strategy._get_timeline_metrics(self.data(), 2)
        self.assertEqual(metrics.makespan, 24)
        self.assertEqual(metrics.ttft, 15)
        self.assertEqual(metrics.e2e, 23)
        self.assertEqual(metrics.tpot, 4)
        self.assertEqual(metrics.mixed_steps, 1)
        self.assertEqual(metrics.mixed_latency_source, "phase_max_approx")
        self.assertAlmostEqual(metrics.output_throughput, 6000 / 24)

    def test_vl_mixed_step_uses_marked_phase_fallback(self):
        strategy = self.make_strategy()
        strategy.model_runner.model.is_vl_model = True
        strategy.model_runner.run_inference = lambda *args, **kwargs: self.fail(
            "VL heterogeneous forward is unsupported"
        )

        def phase_record(key, data):
            activation = 2.0 if key.is_decode else 3.0
            info = {
                "total_device_memory_gb": 40.0,
                "reserved_memory_gb": 0.0,
                "model_weight_size_gb": 10.0,
                "model_activation_size_gb": activation,
                "kv_cache_size_gb": 1.0,
                "device_memory_available_gb": 40.0 - 10.0 - activation - 1.0,
            }
            return ForwardLatencyRecord(
                2.0 if key.is_decode else 10.0, info["device_memory_available_gb"], "pure", info
            )

        strategy._compute_forward_latency_record = phase_record
        metrics = strategy._get_timeline_metrics(self.data(), 2)
        self.assertEqual(metrics.mixed_latency_source, "phase_max_approx")
        self.assertEqual(metrics.mixed_steps, 1)
        self.assertEqual(strategy._last_aggregation_timeline.steps[1][1], 10.0)
        self.assertEqual(metrics.makespan, 24.0)

    def test_mtp_verification_pays_raw_cost_and_rounds_tail_iterations(self):
        strategy = self.make_strategy()
        strategy.num_mtp_tokens = 3
        data = self.data(batch_size=1, output_length=5, num_mtp_tokens=3, mtp_acceptance_rate=[0.9, 0.6, 0.4])
        metrics = strategy._get_timeline_metrics(data, 1)
        # 4 remaining / 2.9 expected outputs => two physical iterations, never 4 iterations / 2.9.
        self.assertEqual(metrics.makespan, 14)
        self.assertEqual(metrics.tpot, 1)
        self.assertEqual(len(strategy._last_aggregation_timeline.steps), 3)

    def test_mtp_does_not_discount_prefill_in_mixed_step(self):
        strategy = self.make_strategy()
        strategy.num_mtp_tokens = 3
        metrics = strategy._get_timeline_metrics(
            self.data(
                output_length=5,
                max_batched_tokens=100,
                input_length=100,
                num_mtp_tokens=3,
                mtp_acceptance_rate=[0.9, 0.6, 0.4],
            ),
            2,
        )
        self.assertEqual(strategy._last_aggregation_timeline.steps[1][1], 10)
        self.assertGreaterEqual(metrics.makespan, 20)

    def test_multichunk_and_dp_share_same_metric_identity(self):
        for dp in (1, 2):
            strategy = self.make_strategy(dp)
            data = self.data(input_length=25)
            metrics = strategy._get_timeline_metrics(data, 2 * dp)
            self.assertAlmostEqual(metrics.tpot, (metrics.e2e - metrics.ttft) / 2)
            self.assertAlmostEqual(metrics.output_throughput, 1000 * 2 * dp * 3 / metrics.makespan)
            for request in strategy._last_aggregation_timeline.requests:
                self.assertLessEqual(request.first_token_ms, request.finish_ms)
                self.assertLessEqual(request.finish_ms, metrics.makespan)

    def test_variable_lengths_keep_each_requests_context(self):
        from serving_cast.service.aggregation_timeline import build_requests

        data = self.data(
            input_length=None, length_distribution=LengthDistribution(bins=[LengthBin(10, 10, 1), LengthBin(30, 30, 1)])
        )
        strategy = self.make_strategy()
        metrics = strategy._get_timeline_metrics(data, 2)
        self.assertEqual(sorted(r.prompt_length for r in strategy._last_aggregation_timeline.requests), [10, 30])
        self.assertAlmostEqual(metrics.tpot, (metrics.e2e - metrics.ttft) / 2)
        self.assertTrue(all(c.query_len <= 10 for r in build_requests(data, 2) for c in r.chunks))

    def test_variable_length_pure_decode_keeps_diagnostic_latency(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), 0)
            if state.pending_prefill
            else StepDecision(0, state.ready_decode)
        )
        data = self.data(
            input_length=None,
            length_distribution=LengthDistribution(bins=[LengthBin(10, 10, 1), LengthBin(30, 30, 1)]),
        )
        metrics = strategy._get_timeline_metrics(data, 2)
        self.assertTrue(
            any(
                all(shape.is_decode for shape in shapes) and len({shape.seq_len for shape in shapes}) > 1
                for shapes, _ in strategy._last_aggregation_timeline.steps
            )
        )
        self.assertEqual((metrics.decode_latency, metrics.decode_breakdowns), (17, "mixed"))

    def test_heterogeneous_pure_phases_keep_diagnostic_latencies(self):
        strategy = self.make_strategy()
        strategy._build_forward_latency_record = lambda requests: ForwardLatencyRecord(
            13.0 if requests[0].is_decode else 11.0,
            20.0,
            "decode varlen" if requests[0].is_decode else "prefill varlen",
        )

        def simulate_heterogeneous_phases(requests, data, scheduler, query_len, accepted, evaluate):
            prefill_shapes = (RequestShape(False, 10, 10), RequestShape(False, 20, 20))
            decode_shapes = (RequestShape(True, 1, 15), RequestShape(True, 1, 25))
            prefill_record = evaluate(prefill_shapes, prefill_shapes)
            decode_record = evaluate(decode_shapes, decode_shapes)
            for request in requests:
                request.first_token_ms = prefill_record.latency_ms
                request.finish_ms = prefill_record.latency_ms + decode_record.latency_ms
            return TimelineResult(requests, elapsed_ms=requests[0].finish_ms)

        with patch("serving_cast.service.aggregation_timeline.simulate", side_effect=simulate_heterogeneous_phases):
            metrics = strategy._get_timeline_metrics(self.data(), 2)
        self.assertEqual((metrics.prefill_latency, metrics.prefill_last_latency, metrics.decode_latency), (11, 11, 13))
        self.assertEqual((metrics.prefill_breakdowns, metrics.decode_breakdowns), ("prefill varlen", "decode varlen"))

    def test_prefix_cache_reduces_queries_but_preserves_full_context(self):
        strategy = self.make_strategy()
        metrics = strategy._get_timeline_metrics(
            self.data(input_length=100, max_batched_tokens=100, prefix_cache_hit_rate=0.5), 2
        )
        shapes = strategy._last_aggregation_timeline.steps[0][0]
        self.assertEqual([(s.query_len, s.seq_len) for s in shapes], [(50, 100), (50, 100)])
        self.assertEqual(metrics.ttft, 10)
        self.assertTrue(all(r.prompt_length == 100 for r in strategy._last_aggregation_timeline.requests))

    def test_dp_replicates_completions_without_sharing_token_budgets(self):
        single = self.make_strategy()._get_timeline_metrics(self.data(), 2)
        replicated = self.make_strategy(dp=2)._get_timeline_metrics(self.data(), 4)
        self.assertEqual(replicated.ttft, single.ttft)
        self.assertEqual(replicated.makespan, single.makespan)
        self.assertEqual(replicated.output_throughput, single.output_throughput * 2)

    def test_homogeneous_forward_key_uses_global_concurrency_for_dp_input_generator(self):
        strategy = self.make_strategy(dp=2)
        keys = []
        original = strategy._compute_forward_latency_record

        def record(key, data):
            keys.append(key)
            return original(key, data)

        strategy._compute_forward_latency_record = record
        strategy._get_timeline_metrics(self.data(max_batched_tokens=20), 4)
        # generate_inputs divides RequestInfo.concurrency by DP; the local P2
        # forward therefore uses a global-concurrency key of four.
        self.assertTrue(any(not key.is_decode and key.model_concurrency == 4 for key in keys))

    def test_mixed_graph_is_not_run_across_candidates(self):
        strategy = self.make_strategy()
        with patch.object(strategy.model_runner, "run_inference", wraps=strategy.model_runner.run_inference) as run:
            strategy._get_timeline_metrics(self.data(), 2)
            strategy._get_timeline_metrics(self.data(), 2)
        self.assertEqual(run.call_count, 0)

    def test_mixed_steps_use_prefill_dominated_phase_max(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), state.ready_decode)
        )
        with patch.object(strategy.model_runner, "run_inference", wraps=strategy.model_runner.run_inference) as run:
            metrics = strategy._get_timeline_metrics(self.data(batch_size=4, output_length=10), 4)
        self.assertGreater(metrics.mixed_steps, 1)
        self.assertEqual(run.call_count, 0)
        self.assertEqual(metrics.mixed_latency_source, "phase_max_approx")
        self.assertTrue(
            all(
                latency == 10
                for shapes, latency in strategy._last_aggregation_timeline.steps
                if any(s.is_decode for s in shapes) and any(not s.is_decode for s in shapes)
            )
        )

    def test_mixed_phase_max_reused_across_prefill_batch_counts(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(
                min(state.pending_prefill, 2 if state.ready_decode else 1), state.ready_decode
            )
        )
        with patch.object(strategy.model_runner, "run_inference", wraps=strategy.model_runner.run_inference) as run:
            metrics = strategy._get_timeline_metrics(self.data(batch_size=4, output_length=5, max_batched_tokens=40), 4)
        self.assertGreater(metrics.mixed_steps, 1)
        self.assertEqual(run.call_count, 0)

    def test_decode_dominated_mixed_steps_use_decode_phase(self):
        strategy = self.make_strategy(latency=lambda is_decode: 12 if is_decode else 10)
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), state.ready_decode)
        )
        with patch.object(strategy.model_runner, "run_inference", wraps=strategy.model_runner.run_inference) as run:
            metrics = strategy._get_timeline_metrics(self.data(batch_size=4, output_length=10), 4)
        self.assertEqual(run.call_count, 0)
        self.assertEqual(metrics.mixed_latency_source, "phase_max_approx")
        self.assertTrue(
            any(
                latency == 12
                for shapes, latency in strategy._last_aggregation_timeline.steps
                if any(s.is_decode for s in shapes) and any(not s.is_decode for s in shapes)
            )
        )

    def test_mixed_memory_reserves_both_activations_and_full_kv(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), state.ready_decode)
        )

        def record(latency, activation):
            info = {
                "total_device_memory_gb": 100.0,
                "reserved_memory_gb": 0.0,
                "model_weight_size_gb": 10.0,
                "model_activation_size_gb": activation,
                "kv_cache_size_gb": 1.0,
                "device_memory_available_gb": 89.0 - activation,
            }
            return ForwardLatencyRecord(latency, info["device_memory_available_gb"], "", memory_info=info)

        strategy._compute_forward_latency_record = lambda key, data: record(
            2 if key.is_decode else 10, 3 if key.is_decode else 2
        )
        strategy._build_forward_latency_record = lambda requests: record(17, 4)
        strategy.model_runner.get_resident_kv_cache_size_gb = lambda requests: float(len(requests))
        metrics = strategy._get_timeline_metrics(self.data(batch_size=4, output_length=10), 4)
        self.assertEqual(metrics.mixed_latency_source, "phase_max_approx")
        self.assertEqual(metrics.memory_info["model_activation_size_gb"], 5)
        self.assertGreaterEqual(metrics.memory_info["kv_cache_size_gb"], 4)

    def test_mixed_memory_near_limit_does_not_run_actual_graph(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), state.ready_decode)
        )

        def record(latency, activation):
            info = {
                "total_device_memory_gb": 20.0,
                "reserved_memory_gb": 0.0,
                "model_weight_size_gb": 10.0,
                "model_activation_size_gb": activation,
                "kv_cache_size_gb": 1.0,
                "device_memory_available_gb": 9.0 - activation,
            }
            return ForwardLatencyRecord(latency, info["device_memory_available_gb"], "", memory_info=info)

        strategy._compute_forward_latency_record = lambda key, data: record(
            2 if key.is_decode else 10, 3 if key.is_decode else 2
        )
        strategy._build_forward_latency_record = lambda requests: record(17, 4)
        strategy.model_runner.get_resident_kv_cache_size_gb = lambda requests: float(len(requests))
        with patch.object(strategy.model_runner, "run_inference", wraps=strategy.model_runner.run_inference) as run:
            metrics = strategy._get_timeline_metrics(self.data(batch_size=4, output_length=10), 4)
        self.assertEqual(run.call_count, 0)
        self.assertEqual(metrics.mixed_latency_source, "phase_max_approx")

    def test_pure_decode_interpolates_between_power_of_two_anchors(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), state.ready_decode)
        )
        decode_sizes = []

        def counted(key, data):
            if key.is_decode:
                decode_sizes.append(key.model_concurrency)
                return ForwardLatencyRecord(float(key.model_concurrency), 20.0, "decode")
            return ForwardLatencyRecord(10.0, 20.0, "prefill")

        strategy._compute_forward_latency_record = counted
        metrics = strategy._get_timeline_metrics(
            self.data(batch_size=20, output_length=100, max_batched_tokens=100, tpot_limits=50), 20
        )
        self.assertEqual(metrics.decode_latency_source, "interpolated")
        self.assertEqual(metrics.decode_latency, 20)
        self.assertNotIn(19, decode_sizes)
        self.assertIn(16, decode_sizes)
        self.assertIn(32, decode_sizes)
        self.assertTrue(
            any(
                len(shapes) == 20 and all(s.is_decode for s in shapes) and latency == 20
                for shapes, latency in strategy._last_aggregation_timeline.steps
            )
        )

    def test_decode_anchor_does_not_exceed_budget_when_schedule_uses_slack(self):
        strategy = self.make_strategy()
        strategy.scheduler = SimpleNamespace(
            decide=lambda state: StepDecision(min(state.pending_prefill, 1), state.ready_decode)
        )
        decode_sizes = []
        original = strategy._compute_forward_latency_record

        def counted(key, data):
            if key.is_decode:
                decode_sizes.append(key.model_concurrency)
            return original(key, data)

        strategy._compute_forward_latency_record = counted
        strategy._get_timeline_metrics(self.data(batch_size=20, output_length=100), 20)
        self.assertIn(20, decode_sizes)
        self.assertLessEqual(max(decode_sizes), 20)

    def test_summary_exports_only_qps(self):
        strategy = self.make_strategy()
        data = self.data(output_length=1, num_devices=1)
        with patch("serving_cast.service.agg_throughput_optimizer.format_parallel_label", return_value="TP=1"):
            row = strategy.get_inference_info(data).get_summary_df().iloc[0]
        self.assertEqual(row["qps"], 100)
        self.assertNotIn("e2e_mean_ms", row.index)
        self.assertNotIn("makespan_ms", row.index)
        self.assertNotIn("metric_version", row.index)

    def test_resident_kv_includes_ready_requests_not_in_the_forward(self):
        strategy = self.make_strategy()
        resident_shapes = []

        def resident_kv(requests):
            resident_shapes.append(tuple((r.is_decode, r.query_len, r.seq_len) for r in requests))
            return 3.0

        strategy.model_runner.get_resident_kv_cache_size_gb = resident_kv
        strategy._compute_forward_latency_record = lambda key, data: ForwardLatencyRecord(
            2 if key.is_decode else 10, 20, "", memory_info={"kv_cache_size_gb": 1, "device_memory_available_gb": 20}
        )
        strategy._build_forward_latency_record = lambda requests: ForwardLatencyRecord(
            17, 20, "", memory_info={"kv_cache_size_gb": 2, "device_memory_available_gb": 20}
        )
        metrics = strategy._get_timeline_metrics(self.data(output_length=200), 2)
        self.assertLess(metrics.memory_left_gb, 20)
        self.assertGreater(metrics.memory_info["kv_cache_size_gb"], 1)
        self.assertTrue(any(is_decode for shapes in resident_shapes for is_decode, _, _ in shapes))

    def test_waiting_requests_reserve_kv_before_prefill(self):
        strategy = self.make_strategy()
        strategy.model_runner.get_resident_kv_cache_size_gb = lambda requests: float(len(requests))
        info = {"kv_cache_size_gb": 1.0, "device_memory_available_gb": 20.0}
        strategy._compute_forward_latency_record = lambda key, data: ForwardLatencyRecord(2.0, 20.0, "", info)
        strategy._build_forward_latency_record = lambda requests: ForwardLatencyRecord(2.0, 20.0, "", info)

        small = strategy._get_timeline_metrics(self.data(batch_size=2), 2)
        large = strategy._get_timeline_metrics(self.data(batch_size=4), 4)

        self.assertEqual(small.memory_left_gb, 19.0)
        self.assertEqual(large.memory_left_gb, 17.0)
        self.assertEqual(large.memory_info["kv_cache_size_gb"], 4.0)

    def test_waiting_kv_reservation_rounds_pages_per_request(self):
        shape = RequestShape(True, 1, 8)
        caches = {"kv_cache_by_layers": {0: torch.empty(128, dtype=torch.uint8)}}
        with (
            patch("serving_cast.service.agg_throughput_optimizer.generate_inputs_varlen", return_value=caches) as gen,
            patch("serving_cast.service.agg_throughput_optimizer.kv_cache_excluded_layer_indices", return_value=set()),
        ):
            reserved = _reserved_kv_cache_size_gb(
                SimpleNamespace(model=object(), user_input=SimpleNamespace(block_size=128)), (shape,) * 4
            )
        self.assertEqual(reserved * 1024**3, 4 * 128)
        gen.assert_called_once()

    def test_oom_stops_and_returns_no_throughput(self):
        strategy = self.make_strategy()
        strategy._compute_forward_latency_record = lambda key, data: ForwardLatencyRecord(10, -1, "")
        metrics = strategy._get_timeline_metrics(self.data(), 2)
        self.assertEqual(metrics.output_throughput, 0)
        self.assertEqual(metrics.ttft, float("inf"))
        self.assertEqual(metrics.makespan, float("inf"))
        self.assertLess(metrics.prefill_memory_left_gb, 0)

    def test_decode_oom_does_not_get_misclassified_as_prefill_oom(self):
        strategy = self.make_strategy()
        strategy._compute_forward_latency_record = lambda key, data: ForwardLatencyRecord(
            2 if key.is_decode else 10, -1 if key.is_decode else 20, ""
        )
        metrics = strategy._get_timeline_metrics(self.data(batch_size=1), 1)
        self.assertLess(metrics.memory_left_gb, 0)
        self.assertGreater(metrics.prefill_memory_left_gb, 0)
        self.assertEqual(metrics.makespan, float("inf"))

    def test_zero_query_length_raises_clear_error(self):
        from serving_cast.service.aggregation_timeline import simulate, TimelineRequest

        strategy = self.make_strategy()
        with self.assertRaisesRegex(ValueError, "at least one prefill chunk"):
            simulate([TimelineRequest(0, [])], self.data(), strategy.scheduler, 1, 1, lambda shapes, resident: None)


_agg_pp_profile = partial(_pp_profile, include_transfers=False)
_AggPPMetrics = _PPMetrics
_make_pp_agg_strategy = partial(_make_pp_strategy, AggThroughputOptimizer)


class TestAggPipelineParallel(unittest.TestCase):
    def test_mixed_pp_profile_uses_stagewise_phase_max(self):
        prefill = _pp_profile((0.010, 0.001, 0.004), comm_times=(0.003, 0.002))
        decode = _pp_profile((0.002, 0.006, 0.001), comm_times=(0.001, 0.004))
        mixed = _overlap_pipeline_profiles(prefill, decode)
        self.assertEqual(
            [stage.compute_time_s_by_model["analytic"] for stage in mixed.stages],
            [0.010, 0.006, 0.004],
        )
        self.assertEqual([transfer.time_s_by_model["analytic"] for transfer in mixed.transfers], [0.003, 0.004])

    def test_mixed_pp_profile_reserves_both_activations_once(self):
        prefill = _pp_profile((0.010, 0.001), include_transfers=False)
        decode = _pp_profile((0.002, 0.006), include_transfers=False)
        prefill = replace(
            prefill,
            stages=tuple(replace(stage, activation_bytes=300, runtime_peak_bytes=2_000) for stage in prefill.stages),
        )
        decode = replace(
            decode,
            stages=tuple(replace(stage, activation_bytes=400, runtime_peak_bytes=2_500) for stage in decode.stages),
        )
        mixed = _overlap_pipeline_profiles(prefill, decode)
        self.assertTrue(all(stage.activation_bytes == 700 for stage in mixed.stages))
        self.assertTrue(all(stage.runtime_peak_bytes == 2_800 for stage in mixed.stages))
        self.assertTrue(all(stage.weight_bytes == 1_000 for stage in mixed.stages))

    def test_variable_length_requests_share_a_pp_mixed_step(self):
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1, microbatch_size=2)
        strategy.model_runner.user_input.block_size = 128
        profile = _agg_pp_profile((0.001, 0.001))
        strategy.model_runner.run_inference.return_value = _AggPPMetrics(profile)
        data = OptimizerData(
            length_distribution=LengthDistribution(bins=[LengthBin(1, 2, 1), LengthBin(10, 11, 5)]),
            output_length=3,
            batch_size=6,
            max_batched_tokens=20,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            summary = strategy.get_inference_info(data)
        row = summary.get_summary_df().iloc[0]
        timeline = strategy._last_aggregation_timeline
        self.assertFalse(summary.check_early_stop_flag())
        self.assertTrue(
            any(
                any(s.is_decode for s in shapes) and any(not s.is_decode for s in shapes)
                for shapes, _ in timeline.steps
            )
        )
        self.assertGreater(strategy.model_runner.run_inference.call_count, 0)
        self.assertTrue(
            all(
                len({request.is_decode for request in call.args[0]}) == 1
                for call in strategy.model_runner.run_inference.call_args_list
            )
        )
        self.assertLess(len(strategy._pp_step_cache), len(timeline.steps))
        self.assertEqual(row["ttft"], round(timeline.ttft, 3))
        self.assertEqual(row["tpot"], round((timeline.e2e - timeline.ttft) / 2, 3))
        self.assertAlmostEqual(row["qps"], 1000 * 6 / timeline.elapsed_ms, places=3)
        self.assertEqual([r.prompt_length for r in timeline.requests], [1, 10, 10, 10, 10, 10])

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
                self.assertFalse(summary.get_pp_mixed_pd_overlap_approx())
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
        self.assertFalse(summary.get_pp_mixed_pd_overlap_approx())

    def test_pp_uses_request_completion_timestamps(self):
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        optimizer_data = OptimizerData(
            input_length=4,
            output_length=4,
            batch_size=2,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]
        timeline = strategy._last_aggregation_timeline
        self.assertEqual(row["ttft"], 5000.0)  # P microbatches finish at 4s and 6s.
        self.assertEqual(row["tpot"], round((timeline.e2e - timeline.ttft) / 3, 3))
        self.assertAlmostEqual(row["token/s"], 1000.0 * 4 * 2 / timeline.elapsed_ms, places=3)

    def test_pp_decode_tpot_and_qps_are_independent_slo_metrics(self):
        for pp in (2, 3):
            with self.subTest(pp=pp):
                strategy = _make_pp_agg_strategy(dp=1, pp=pp, tp=1)
                profile = _agg_pp_profile((2.0,) * pp)
                data = OptimizerData(
                    input_length=4,
                    output_length=4,
                    batch_size=2,
                    max_batched_tokens=2048,
                    num_mtp_tokens=0,
                    mtp_acceptance_rate=[],
                )
                with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
                    baseline = strategy.get_inference_info(data)
                    row = baseline.get_summary_df().iloc[0]
                    timeline = strategy._last_aggregation_timeline
                    self.assertAlmostEqual(row["qps"], 1000.0 * 2 / timeline.elapsed_ms, delta=0.001)
                    self.assertEqual(row["tpot"], round((timeline.e2e - timeline.ttft) / 3, 3))
                    data.tpot_limits = row["tpot"] - 1
                    limited = strategy.get_inference_info(data)
                self.assertTrue(limited.check_early_stop_flag())
                self.assertEqual(limited.get_early_stop_reason(), EARLY_STOP_TPOT_LIMIT)

    def test_pp_decode_applies_mtp_fold(self):
        """MTP advances output by the acceptance expectation per physical PP pass."""
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
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
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]
        timeline = strategy._last_aggregation_timeline
        decode_steps = [shapes for shapes, _ in timeline.steps if all(s.is_decode for s in shapes)]
        self.assertEqual(len(decode_steps), 4)  # two requests, two full MTP passes each
        self.assertTrue(all(s.query_len == 3 for shapes in decode_steps for s in shapes))
        self.assertEqual(row["tpot"], round((timeline.e2e - timeline.ttft) / 3, 3))
        self.assertAlmostEqual(row["token/s"], 8000 / timeline.elapsed_ms, places=3)

    def test_pp_decode_applies_block_fold_dflash_and_dspark(self):
        """Draft blocks retain their physical query width and acceptance progress."""
        for method, block_field, acceptance_field in (
            ("dflash", "dflash_block_size", "dflash_acceptance_length"),
            ("dspark", "dspark_block_size", "dspark_acceptance_length"),
        ):
            with self.subTest(method=method):
                strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
                profile = _agg_pp_profile((2.0, 2.0))
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

                with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
                    row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]
                timeline = strategy._last_aggregation_timeline
                decode_steps = [shapes for shapes, _ in timeline.steps if all(s.is_decode for s in shapes)]
                self.assertEqual(len(decode_steps), 2)  # one full pass per request
                self.assertTrue(all(s.query_len == 6 for shapes in decode_steps for s in shapes))
                self.assertEqual(row["tpot"], round((timeline.e2e - timeline.ttft) / 3, 3))

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
        self.assertAlmostEqual(row["token/s"], 1000.0 * 2 / 6000.0, places=3)
        self.assertFalse(summary.check_early_stop_flag())
        self.assertFalse(summary.get_pp_mixed_pd_overlap_approx())

    def test_resident_pp_kv_uses_stage_local_physical_layout(self):
        profile = _pp_profile((0.001, 0.001), include_transfers=False)
        runner = SimpleNamespace(model=object())
        caches = {
            "kv_cache_by_layers": {
                0: torch.zeros(16, dtype=torch.bfloat16),
                1: torch.zeros(32, dtype=torch.bfloat16),
            },
            "indexer_cache_by_layers": {1: torch.zeros(8, dtype=torch.int8)},
        }
        with (
            patch("serving_cast.service.agg_throughput_optimizer.generate_inputs_varlen", return_value=caches),
            patch("serving_cast.service.agg_throughput_optimizer.kv_cache_excluded_layer_indices", return_value={1}),
        ):
            resident = _resident_pp_kv_bytes(runner, (RequestShape(True, 1, 100),), profile, 128)
        self.assertEqual(resident, (32, 8))

    def test_waiting_pp_requests_reserve_stage_local_pages(self):
        profile = _pp_profile((0.001, 0.001), include_transfers=False)
        runner = SimpleNamespace(model=object())
        shape = RequestShape(True, 1, 8)
        caches = {
            "kv_cache_by_layers": {0: torch.zeros(16, dtype=torch.bfloat16), 1: torch.zeros(32, dtype=torch.bfloat16)}
        }
        with (
            patch("serving_cast.service.agg_throughput_optimizer.generate_inputs_varlen", return_value=caches) as gen,
            patch("serving_cast.service.agg_throughput_optimizer.kv_cache_excluded_layer_indices", return_value=set()),
        ):
            reserved = _reserved_pp_kv_bytes(runner, (shape,) * 4, profile, 128)
        self.assertEqual(reserved, (4 * 32, 4 * 64))
        gen.assert_called_once()

    def test_pp_communication_buffer_can_exceed_stage_memory(self):
        from serving_cast.service.optimizer_summary import EARLY_STOP_PREFILL_OOM

        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        strategy.model_runner.total_device_memory_gb = 3500 / 1024**3
        profile = _pp_profile((0.001, 0.001), runtime_peak_bytes=2000, payload_bytes=4096)
        data = OptimizerData(
            input_length=4,
            output_length=1,
            batch_size=1,
            max_batched_tokens=2048,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
            summary = strategy.get_inference_info(data)

        self.assertEqual(summary.get_early_stop_reason(), EARLY_STOP_PREFILL_OOM)

    def test_variable_length_prefill_is_supported_for_pp(self):
        strategy = _make_pp_agg_strategy(dp=1, pp=2, tp=1)
        profile = _agg_pp_profile((2.0, 2.0))
        strategy.model_runner.run_inference.return_value = _AggPPMetrics(profile)
        for name, optimizer_data, _ in _pp_prefill_validation_cases(disaggregated=False):
            with self.subTest(name=name):
                with patch.object(strategy, "_get_forward_info", return_value=_AggPPMetrics(profile)):
                    summary = strategy.get_inference_info(optimizer_data)
                self.assertFalse(summary.check_early_stop_flag())
                self.assertEqual(len(strategy._last_aggregation_timeline.requests), optimizer_data.batch_size)

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

    def test_batch_budget_is_applied_per_admitted_microbatch(self):
        """A finite batch larger than B proceeds as multiple admitted microbatches."""
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
        self.assertFalse(summary.check_early_stop_flag())
        self.assertEqual(len(strategy._last_aggregation_timeline.requests), 4)
        self.assertGreater(len(strategy._last_aggregation_timeline.steps), 1)


if __name__ == "__main__":
    unittest.main()
