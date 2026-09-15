# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from functools import partial
from unittest.mock import Mock, patch

import pandas as pd
from serving_cast.service.disagg_throughput_optimizer import (
    _format_profiling_sources,
    _profiling_result_kind,
    DisaggThroughputOptimizer,
)
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import (
    BYTES_TO_GB,
    LengthBin,
    LengthDistribution,
    OptimizerData,
)

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from tensor_cast.pipeline_parallel import (
    PipelineProfile,
    PipelineStageProfile,
    PipelineTransferProfile,
)

from .test_common import SimpleArgs


def _simple_length_distribution():
    return LengthDistribution(
        bins=[
            LengthBin(min_tokens=0, max_tokens=500, weight=0.6),
            LengthBin(min_tokens=500, max_tokens=1500, weight=0.4),
        ]
    )


class TestDisaggStrategy(unittest.TestCase):
    def test_profiling_result_kind_distinguishes_interpolation(self):
        self.assertEqual(_profiling_result_kind({}), "analytic")
        self.assertEqual(_profiling_result_kind({"analytic": 1.0}), "analytic")
        self.assertEqual(_profiling_result_kind({"measured": 1.0}), "empirical")
        self.assertEqual(
            _profiling_result_kind({"measured": 0.8, "interpolated": 0.2, "analytic": 0.0}),
            "interpolated",
        )
        self.assertEqual(
            _profiling_result_kind({"measured": 0.8, "interpolated": 0.0, "analytic": 0.2}),
            "hybrid",
        )
        self.assertEqual(
            _profiling_result_kind({"measured": 0.8, "analytic": 0.0, "hybrid": 0.2}),
            "hybrid",
        )

    def test_formats_partial_lookup_latency_as_hybrid(self):
        self.assertEqual(
            _format_profiling_sources({"measured": 0.7, "interpolated": 0.1, "analytic": 0.0, "hybrid": 0.2}),
            "Measured 70.00 | Interpolated 10.00 | Analytic 0.00 | Hybrid 20.00",
        )

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
            max_batched_tokens=2048,
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
        self.assertEqual(row["parallel"], "TP=1 | PP=1 | DP=4 | MTP=1")

    def test_get_inference_info_decode_uses_empirical_latency_when_present(self):
        # Regression guard for profiling mode: the decode/single-chunk path must
        # read the "empirical" key and not crash when "analytic" is absent.
        optimizer_data = OptimizerData(
            ttft_limits=None,
            tpot_limits=50,
            batch_size=2,
            input_length=512,
            output_length=128,
            max_batched_tokens=2048,
            serving_cost=3,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            class DummyMetrics:
                execution_time_s = {"empirical": 0.002}
                device_memory_available_gb = 1.0
                breakdowns = {}
                profiling_source_times_s = {
                    "measured": 0.0014,
                    "interpolated": 0.0002,
                    "analytic": 0.0004,
                }
                profiling_miss_reasons = {
                    "outside_axis_boundary": 75,
                    "generic_compute_no_compatible_regime": 3,
                }

            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        # latency_ms = empirical (0.002 s -> 2 ms) + serving_cost (3) = 5 ms
        self.assertEqual(row["tpot"], 5.0)
        self.assertEqual(row["profiling_sources"], "Measured 70.00 | Interpolated 10.00 | Analytic 20.00")
        self.assertEqual(row["profiling_source_scope"], "modeled_forward_lookup_latency")
        self.assertEqual(row["profiling_result"], "hybrid")
        self.assertEqual(
            row["profiling_misses"],
            "outside_axis_boundary x75 | generic_compute_no_compatible_regime x3",
        )

    def test_get_inference_info_prefill_mode(self):
        """Test get_inference_info method in prefill mode"""
        # Mock data config for prefill mode
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=5,
            input_length=1024,
            output_length=50,
            max_batched_tokens=2048,
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

    def test_unbounded_prefill_models_full_concurrency_without_waves(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=8,
            input_length=1024,
            output_length=16,
            max_batched_tokens=None,
            serving_cost=0,
        )
        captured_calls = []

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            captured_calls.append((concurrency, is_decode, query_len, seq_len))
            return Mock(
                execution_time_s={"analytic": 0.001},
                device_memory_available_gb=1.0,
                breakdowns={},
                total_device_memory_gb=64.0,
                model_weight_size_gb=20.0,
                kv_cache_size_gb=4.0,
                model_activation_size_gb=1.0,
                reserved_memory_gb=10.0,
            )

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        self.assertEqual(captured_calls, [(32, False, None, None)])
        self.assertEqual(result.get_summary_df().iloc[0]["prefill_num_chunks"], 1)

    def test_run_skips_automatic_budget_when_disagg_budget_is_omitted(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            input_length=1024,
            output_length=16,
            max_batched_tokens=None,
        )

        with (
            patch.object(self.strategy, "_run_once", return_value="full-forward") as run_once,
            patch.object(
                self.strategy,
                "_run_with_auto_max_batched_tokens",
            ) as auto_budget,
        ):
            result = self.strategy.run(optimizer_data, [1, 8])

        self.assertEqual(result, "full-forward")
        run_once.assert_called_once_with(optimizer_data, [1, 8])
        auto_budget.assert_not_called()

    def test_run_keeps_automatic_budget_for_decode_without_budget(self):
        optimizer_data = OptimizerData(
            ttft_limits=None,
            tpot_limits=50,
            input_length=1024,
            output_length=16,
            max_batched_tokens=None,
        )

        with (
            patch.object(self.strategy, "_run_once") as run_once,
            patch.object(
                self.strategy,
                "_run_with_auto_max_batched_tokens",
                return_value="auto-budget",
            ) as auto_budget,
        ):
            result = self.strategy.run(optimizer_data, [1, 8])

        self.assertEqual(result, "auto-budget")
        run_once.assert_not_called()
        auto_budget.assert_called_once_with(optimizer_data, [1, 8])

    def test_chunked_prefill_splits_each_chunk_into_per_dp_token_budget_waves(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=1,
            input_length=10,
            output_length=16,
            max_batched_tokens=4,
            serving_cost=2,
        )
        captured_calls = []

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            captured_calls.append((concurrency, query_len, seq_len))

            class DummyMetrics:
                execution_time_s = {"analytic": 0.001}
                total_device_memory_gb = 64.0
                model_weight_size_gb = 20.0
                kv_cache_size_gb = 4.0
                model_activation_size_gb = 1.0
                reserved_memory_gb = 10.0
                device_memory_available_gb = 1.0
                breakdowns = {
                    "stage": {
                        "first": float(len(captured_calls)),
                        "second": float(10 - len(captured_calls)),
                    }
                }
                profiling_source_times_s = {
                    "measured": 0.0008,
                    "interpolated": 0.0,
                    "analytic": 0.0002,
                }
                profiling_miss_reasons = {"outside_axis_boundary": 1}

            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertEqual(
            captured_calls,
            [
                (4, 4, 4),
                (4, 4, 8),
                (4, 2, 10),
            ],
        )
        self.assertTrue(
            all(
                concurrency * query_len <= optimizer_data.max_batched_tokens * self.strategy.dp
                for concurrency, query_len, _ in captured_calls
            )
        )
        self.assertEqual(row["prefill_num_chunks"], 3)
        # Each chunk fits in a single wave (concurrency=4, wave_size=4),
        # so TTFT equals the final chunk's completion time.
        self.assertEqual(row["ttft"], 5.0)
        self.assertEqual(row["prefill_phase_makespan_ms"], 5.0)
        self.assertEqual(row["token/s"], 8000.0)
        self.assertEqual(row["percentage_breakdowns"], "Mem 20.00 | Comm 80.00 | Cube 0.00 | Vec 0.00")
        self.assertEqual(row["profiling_sources"], "Measured 80.00 | Interpolated 0.00 | Analytic 20.00")
        self.assertEqual(row["profiling_result"], "hybrid")
        self.assertEqual(row["profiling_misses"], "outside_axis_boundary x3")

    def test_chunked_prefill_passes_prefill_phase_to_every_forward(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=1,
            input_length=8192,
            output_length=1,
            max_batched_tokens=4096,
            serving_cost=0,
        )
        captured_requests = []

        def fake_run_inference(requests, generate_inputs_func=None):
            captured_requests.extend(requests)
            return Mock(
                execution_time_s={"analytic": 0.001},
                device_memory_available_gb=1.0,
                breakdowns={},
            )

        with patch.object(self.strategy.model_runner, "run_inference", side_effect=fake_run_inference):
            self.strategy.get_inference_info(optimizer_data)

        self.assertEqual(
            [
                (request.query_len, request.seq_len, request.is_decode, request.concurrency)
                for request in captured_requests
            ],
            [
                (4096, 4096, False, 4),
                (4096, 8192, False, 4),
            ],
        )

    def test_single_chunk_prefill_uses_the_token_budget_on_every_dp_replica(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=1,
            input_length=4,
            output_length=16,
            max_batched_tokens=8,
            serving_cost=0,
        )
        captured_calls = []

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            captured_calls.append((concurrency, query_len, seq_len))

            class DummyMetrics:
                execution_time_s = {"analytic": 0.001}
                total_device_memory_gb = 64.0
                model_weight_size_gb = 20.0
                kv_cache_size_gb = 4.0
                model_activation_size_gb = 1.0
                reserved_memory_gb = 10.0
                device_memory_available_gb = 1.0
                breakdowns = {}

            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertEqual(captured_calls, [(4, None, None)])
        self.assertEqual(row["prefill_num_chunks"], 1)
        # Restore the TTFT assertion: the single forward's latency is
        # _select_latency_s({"analytic": 0.001}) * 1000 + serving_cost(0) = 1.0 ms.
        self.assertEqual(row["ttft"], 1.0)

    def test_chunked_prefill_stops_when_any_record_memory_is_negative(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=1,
            input_length=10,
            output_length=16,
            max_batched_tokens=4,
            serving_cost=2,
        )
        captured_calls = []

        def fake_forward(concurrency, optimizer_data, is_decode, *, query_len=None, seq_len=None):
            captured_calls.append((concurrency, query_len, seq_len))
            if seq_len == 10:
                raise AssertionError("chunk after negative memory should not be computed")

            class DummyMetrics:
                execution_time_s = {"analytic": 0.001}
                device_memory_available_gb = -1.0 if seq_len == 8 else 1.0
                breakdowns = {}

            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertEqual(captured_calls, [(4, 4, 4), (4, 4, 8)])
        self.assertTrue(result.check_early_stop_flag())
        self.assertLess(row["ttft"], 12.0)

    def test_prefix_cache_changes_prefill_shape_but_not_decode_shape(self):
        optimizer_data = OptimizerData(
            batch_size=2,
            input_length=200,
            output_length=32,
            max_batched_tokens=2048,
            prefix_cache_hit_rate=0.5,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        captured = []

        def fake_forward(concurrency, optimizer_data, is_decode):
            captured.append((is_decode, optimizer_data.get_effective_input_length(is_decode)))

            class DummyMetrics:
                total_device_memory_gb = 64.0
                model_weight_size_gb = 20.0
                kv_cache_size_gb = 4.0
                model_activation_size_gb = 1.0
                reserved_memory_gb = 10.0
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

    def test_get_inference_info_prefill_acc_search_records_search_info(self):
        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=16,
            input_length=100,
            output_length=10,
            max_batched_tokens=8192,
            serving_cost=0,
            concurrency_search_strategy="linear_exponential",
        )

        class DummyMetrics:
            total_device_memory_gb = 64.0
            model_weight_size_gb = 20.0
            kv_cache_size_gb = 4.0
            model_activation_size_gb = 1.0
            reserved_memory_gb = 10.0
            execution_time_s = {"analytic": 0.005}
            device_memory_available_gb = 2.0
            breakdowns = {}

        self.strategy.model_runner.total_device_memory_gb = 64.0
        self.strategy.model_runner.model_weight_size_gb = 20.0
        self.strategy.model_runner.user_input.reserved_memory_gb = 10.0

        captured = []

        def fake_forward(concurrency, optimizer_data, is_decode):
            captured.append((concurrency, is_decode))
            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        search_info = result.get_search_info()
        self.assertEqual(captured, [(64, False)])
        self.assertAlmostEqual(search_info["per_request_memory_gb"], 2.0)
        self.assertEqual(search_info["device_memory_available_gb"], 2.0)
        self.assertEqual(search_info["ttft"], 5.0)
        self.assertIsNone(search_info["tpot"])

    def test_get_inference_info_decode_acc_search_records_search_info(self):
        optimizer_data = OptimizerData(
            ttft_limits=None,
            tpot_limits=100,
            batch_size=16,
            input_length=100,
            output_length=10,
            serving_cost=0,
            num_mtp_tokens=2,
            mtp_acceptance_rate=[0.5, 0.3],
            concurrency_search_strategy="linear_exponential",
        )

        class DummyMetrics:
            total_device_memory_gb = 64.0
            model_weight_size_gb = 20.0
            kv_cache_size_gb = 4.0
            model_activation_size_gb = 1.0
            reserved_memory_gb = 10.0
            execution_time_s = {"analytic": 0.009}
            device_memory_available_gb = 2.0
            breakdowns = {}

        self.strategy.model_runner.total_device_memory_gb = 64.0
        self.strategy.model_runner.model_weight_size_gb = 20.0
        self.strategy.model_runner.user_input.reserved_memory_gb = 10.0

        captured = []

        def fake_forward(concurrency, optimizer_data, is_decode):
            captured.append((concurrency, is_decode))
            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        search_info = result.get_search_info()
        self.assertEqual(captured, [(64, True)])
        self.assertAlmostEqual(search_info["per_request_memory_gb"], 2.0)
        self.assertEqual(search_info["device_memory_available_gb"], 2.0)
        self.assertIsNone(search_info["ttft"])
        self.assertAlmostEqual(search_info["tpot"], 5.0)


class TestDisaggStrategyHermetic(unittest.TestCase):
    def test_decode_only_summary_hides_prefill_distribution_metadata(self):
        strategy = DisaggThroughputOptimizer()
        strategy.dp = 4
        strategy.tp = 1
        strategy.pp = 1
        strategy.is_moe_model = False
        strategy.num_mtp_tokens = 0
        strategy.model_runner = Mock()
        strategy.model_runner.user_input.device = "TEST_DEVICE"
        strategy.model_runner.user_input.model_id = "test-model"
        strategy.model_runner.user_input.quantize_linear_action = "DISABLED"
        strategy.model_runner.user_input.quantize_attention_action = "DISABLED"
        strategy.model_runner.model.model_config.parallel_config = Mock(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            decode_context_parallel_size=1,
        )

        optimizer_data = OptimizerData(
            ttft_limits=None,
            tpot_limits=50.0,
            batch_size=2,
            input_length=512,
            output_length=128,
            serving_cost=0,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )

        class DummyMetrics:
            execution_time_s = {"analytic": 0.004}
            device_memory_available_gb = 1.0
            breakdowns = {}

        with patch.object(strategy, "_get_forward_info", return_value=DummyMetrics()):
            result = strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertIsNone(row["ttft"])
        self.assertIsNone(row.get("input_length_mode"))
        self.assertIsNotNone(row["tpot"])

    def test_distribution_prefill_path_keeps_input_length_empty_in_base_row(self):
        strategy = DisaggThroughputOptimizer()
        strategy.dp = 4
        strategy.tp = 1
        strategy.pp = 1
        strategy.is_moe_model = False
        strategy.num_mtp_tokens = 0
        strategy.model_runner = Mock()
        strategy.model_runner.user_input.device = "TEST_DEVICE"
        strategy.model_runner.user_input.model_id = "test-model"
        strategy.model_runner.user_input.quantize_linear_action = "DISABLED"
        strategy.model_runner.user_input.quantize_attention_action = "DISABLED"
        strategy.model_runner.model.model_config.parallel_config = Mock(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            decode_context_parallel_size=1,
        )

        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=5,
            length_distribution=_simple_length_distribution(),
            output_length=50,
            serving_cost=0,
            max_batched_tokens=8192,
        )

        batch_result = Mock(
            execution_time_s={"analytic": 0.001},
            device_memory_available_gb=1.0,
            breakdowns={},
        )
        composition_rows = [
            {
                "num_input_tokens": 250,
                "query_len": 250,
                "request_ratio": 0.6,
                "samples": 3,
            },
            {
                "num_input_tokens": 1000,
                "query_len": 1000,
                "request_ratio": 0.4,
                "samples": 2,
            },
        ]

        with patch.object(
            strategy,
            "_get_batched_forward_info",
            return_value=([(batch_result, 5)], composition_rows),
        ):
            result = strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertTrue(pd.isna(row["input_length"]))
        self.assertIsNone(row["tpot"])

    def test_distribution_early_stop_uses_aggregated_ttft_not_p95(self):
        strategy = DisaggThroughputOptimizer()
        strategy.dp = 4
        strategy.tp = 1
        strategy.pp = 1
        strategy.is_moe_model = False
        strategy.num_mtp_tokens = 0
        strategy.model_runner = Mock()
        strategy.model_runner.user_input.device = "TEST_DEVICE"
        strategy.model_runner.user_input.model_id = "test-model"
        strategy.model_runner.user_input.quantize_linear_action = "DISABLED"
        strategy.model_runner.user_input.quantize_attention_action = "DISABLED"
        strategy.model_runner.model.model_config.parallel_config = Mock(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            decode_context_parallel_size=1,
        )

        optimizer_data = OptimizerData(
            ttft_limits=130.0,
            tpot_limits=None,
            batch_size=5,
            length_distribution=_simple_length_distribution(),
            output_length=50,
            serving_cost=7,
            max_batched_tokens=8192,
        )
        composition_rows = [
            {
                "num_input_tokens": 250,
                "query_len": 250,
                "request_ratio": 0.6,
                "samples": 3,
            },
            {
                "num_input_tokens": 1000,
                "query_len": 1000,
                "request_ratio": 0.4,
                "samples": 2,
            },
        ]

        with (
            patch.object(
                strategy,
                "_get_batched_forward_info",
                return_value=(
                    [
                        (
                            Mock(
                                execution_time_s={"analytic": 0.123},
                                device_memory_available_gb=2.0,
                                breakdowns={"prefill": {"Mem": 1.0}},
                            ),
                            5,
                        )
                    ],
                    composition_rows,
                ),
            ),
            patch.object(strategy, "_get_forward_info") as mock_forward,
        ):
            result = strategy.get_inference_info(optimizer_data)

        mock_forward.assert_not_called()
        self.assertFalse(result.check_early_stop_flag())

    def test_distribution_prefill_throughput_uses_global_tokens_instead_of_per_rank_tokens(
        self,
    ):
        strategy = DisaggThroughputOptimizer()
        strategy.dp = 4
        strategy.tp = 1
        strategy.pp = 1
        strategy.is_moe_model = False
        strategy.num_mtp_tokens = 0
        strategy.model_runner = Mock()
        strategy.model_runner.user_input.device = "TEST_DEVICE"
        strategy.model_runner.user_input.model_id = "test-model"
        strategy.model_runner.user_input.quantize_linear_action = "DISABLED"
        strategy.model_runner.user_input.quantize_attention_action = "DISABLED"
        strategy.model_runner.model.model_config.parallel_config = Mock(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            decode_context_parallel_size=1,
        )

        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=5,
            length_distribution=_simple_length_distribution(),
            output_length=50,
            serving_cost=0,
            max_batched_tokens=8192,
        )

        batch_result = Mock(
            execution_time_s={"analytic": 0.001},
            device_memory_available_gb=1.0,
            breakdowns={},
        )
        composition_rows = [
            {
                "num_input_tokens": 250,
                "query_len": 250,
                "request_ratio": 0.6,
                "samples": 3,
            },
            {
                "num_input_tokens": 1000,
                "query_len": 1000,
                "request_ratio": 0.4,
                "samples": 2,
            },
        ]

        with patch.object(
            strategy,
            "_get_batched_forward_info",
            return_value=([(batch_result, 5)], composition_rows),
        ):
            result = strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertEqual(row["concurrency"], 20)
        self.assertEqual(row["token/s"], 11000000.0)

    def test_distribution_chunked_prefill_uses_request_weighted_ttft(self):
        strategy = DisaggThroughputOptimizer()
        strategy.dp = 4
        strategy.tp = 1
        strategy.pp = 1
        strategy.is_moe_model = False
        strategy.num_mtp_tokens = 0
        strategy.model_runner = Mock()
        strategy.model_runner.user_input.device = "TEST_DEVICE"
        strategy.model_runner.user_input.model_id = "test-model"
        strategy.model_runner.user_input.quantize_linear_action = "DISABLED"
        strategy.model_runner.user_input.quantize_attention_action = "DISABLED"
        strategy.model_runner.model.model_config.parallel_config = Mock(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            decode_context_parallel_size=1,
        )

        optimizer_data = OptimizerData(
            ttft_limits=1000,
            tpot_limits=None,
            batch_size=5,
            length_distribution=_simple_length_distribution(),
            output_length=50,
            serving_cost=7,
            max_batched_tokens=200,
        )
        composition_rows = [
            {
                "num_input_tokens": 250,
                "query_len": 250,
                "request_ratio": 0.6,
                "samples": 3,
            },
            {
                "num_input_tokens": 1000,
                "query_len": 1000,
                "request_ratio": 0.4,
                "samples": 2,
            },
        ]
        first_chunk = Mock(
            execution_time_s={"analytic": 0.010},
            device_memory_available_gb=3.0,
            breakdowns={"prefill": {"Cube": 3.0, "Mem": 1.0}},
        )
        second_chunk = Mock(
            execution_time_s={"analytic": 0.020},
            device_memory_available_gb=2.0,
            breakdowns={"prefill": {"Cube": 1.0, "Mem": 1.0}},
        )

        with patch.object(
            strategy,
            "_get_batched_forward_info",
            return_value=([(first_chunk, 3), (second_chunk, 2)], composition_rows),
        ) as mock_get_batched:
            result = strategy.get_inference_info(optimizer_data)

        chunk_plan = mock_get_batched.call_args.args[2]
        self.assertGreater(len(chunk_plan), 1)
        row = result.get_summary_df().iloc[0]
        # Model-only completion times are 10 ms and 30 ms; serving cost is added once.
        self.assertEqual(row["ttft"], 25.0)
        self.assertEqual(row["prefill_phase_makespan_ms"], 37.0)
        self.assertAlmostEqual(row["token/s"], 11000 / 0.037, places=3)
        self.assertEqual(row["avail_GB"], 2.0)


def _pp_stage_profile(
    stage_id: int,
    *,
    compute_s: float,
    comm_s: float,
    payload_bytes: int = 4096,
    weight_bytes: int = 1_000,
    runtime_peak_bytes: int = 2_000,
    kv_cache_bytes: int = 0,
) -> PipelineStageProfile:
    return PipelineStageProfile(
        stage_id=stage_id,
        layer_start=stage_id,
        layer_end=stage_id + 1,
        compute_time_s_by_model={"analytic": compute_s},
        outgoing_comm_time_s_by_model={"analytic": comm_s},
        weight_bytes=weight_bytes,
        activation_bytes=0,
        kv_cache_bytes=kv_cache_bytes,
        kv_cache_per_token_bytes=0.0,
        indexer_cache_bytes=0,
        indexer_cache_per_token_bytes=0.0,
        runtime_peak_bytes=runtime_peak_bytes,
        outgoing_payload_bytes=payload_bytes if comm_s > 0 else 0,
    )


def _pp_profile(
    compute_times: tuple[float, ...],
    *,
    comm_times: tuple[float, ...] | None = None,
    payload_bytes: int = 4096,
    weight_bytes: int = 1_000,
    runtime_peak_bytes: int = 2_000,
    kv_cache_bytes: int = 0,
    include_transfers: bool = True,
) -> PipelineProfile:
    if comm_times is None:
        comm_times = (0.0,) * (len(compute_times) - 1)
    stages = tuple(
        _pp_stage_profile(
            stage_id,
            compute_s=compute,
            comm_s=comm_times[stage_id] if stage_id < len(comm_times) else 0.0,
            payload_bytes=payload_bytes,
            weight_bytes=weight_bytes,
            runtime_peak_bytes=runtime_peak_bytes,
            kv_cache_bytes=kv_cache_bytes,
        )
        for stage_id, compute in enumerate(compute_times)
    )
    transfers = (
        tuple(
            PipelineTransferProfile(
                source_stage_id=i,
                target_stage_id=i + 1,
                payload_bytes=payload_bytes,
                time_s_by_model={"analytic": comm_times[i]},
                bandwidth_bytes_ps=1.0e9,
                latency_s=0.0,
            )
            for i in range(len(comm_times))
        )
        if include_transfers
        else ()
    )
    return PipelineProfile(
        pp_size=len(compute_times),
        layer_partition=tuple(1 for _ in compute_times),
        stages=stages,
        transfers=transfers,
    )


class _PPMetrics:
    def __init__(
        self,
        profile: PipelineProfile,
        *,
        execution_time_s=None,
        device_memory_available_gb=10.0,
        profiling_source_times_s=None,
        profiling_miss_reasons=None,
    ):
        self.pipeline_profile = profile
        self.execution_time_s = execution_time_s or {"analytic": 0.0}
        self.device_memory_available_gb = device_memory_available_gb
        self.breakdowns = {}
        self.profiling_source_times_s = profiling_source_times_s or {}
        self.profiling_miss_reasons = profiling_miss_reasons or {}


def _make_pp_strategy(
    strategy_type,
    *,
    dp: int,
    pp: int,
    tp: int,
    microbatch_size: int = 1,
):
    from types import SimpleNamespace

    strategy = strategy_type()
    strategy.dp = dp
    strategy.tp = tp
    strategy.pp = pp
    strategy.is_moe_model = False
    strategy.num_mtp_tokens = 0
    strategy.model_runner = Mock()
    strategy.model_runner.perf_models = [SimpleNamespace(name="analytic")]
    strategy.model_runner.user_input.device = "TEST_DEVICE"
    strategy.model_runner.user_input.model_id = "test-model"
    strategy.model_runner.user_input.quantize_linear_action = "DISABLED"
    strategy.model_runner.user_input.quantize_attention_action = "DISABLED"
    strategy.model_runner.user_input.microbatch_size = microbatch_size
    strategy.model_runner.user_input.reserved_memory_gb = 0.0
    strategy.model_runner.total_device_memory_gb = 64.0
    strategy.model_runner.model_weight_size_gb = 0.0
    strategy.model_runner.model.model_config.parallel_config = Mock(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
        decode_context_parallel_size=1,
        expert_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_data_parallel_size=1,
    )
    return strategy


_PP_SCALING_MATRIX = (
    # concurrency = batch_size * dp (PP does not add request replicas)
    ("single-replica", 1, 2, 1, 2, 2),
    ("dp-replicated", 2, 2, 2, 2, 4),
)


def _pp_prefill_validation_cases(*, disaggregated: bool):
    """Cases that must still raise UnsupportedPPConfigurationError."""
    phase_kwargs = {"ttft_limits": 1000, "tpot_limits": None} if disaggregated else {}
    case_specs = (
        (
            "variable",
            {
                "batch_size": 2,
                "input_length": None,
                "length_distribution": _simple_length_distribution(),
                "max_batched_tokens": 2048,
            },
            "variable-length",
        ),
    )
    return tuple(
        (
            name,
            OptimizerData(output_length=8, **phase_kwargs, **optimizer_kwargs),
            expected,
        )
        for name, optimizer_kwargs, expected in case_specs
    )


def _run_pp_scaling_case(
    strategy_type,
    *,
    dp: int,
    pp: int,
    tp: int,
    batch_size: int,
):
    strategy = _make_pp_strategy(strategy_type, dp=dp, pp=pp, tp=tp)
    optimizer_kwargs = {
        "input_length": 4,
        "output_length": 8,
        "batch_size": batch_size,
        "max_batched_tokens": batch_size * 4,
        "serving_cost": 0,
        "num_mtp_tokens": 0,
        "mtp_acceptance_rate": [],
    }
    if strategy_type is DisaggThroughputOptimizer:
        optimizer_kwargs.update(ttft_limits=100000, tpot_limits=None)
    optimizer_data = OptimizerData(**optimizer_kwargs)
    profile = _pp_profile((2.0, 2.0), include_transfers=False)
    with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
        summary = strategy.get_inference_info(optimizer_data)
    return summary, summary.get_summary_df().iloc[0]


def _pp_schedule_estimates(*, makespan_s: float, interval_s: float, worst_tpot_s: float):
    from serving_cast.service.pipeline_schedule import (
        PipelineScheduleEstimate,
        RepeatedPipelineEstimate,
    )

    first_wave = PipelineScheduleEstimate(
        makespan_s=makespan_s,
        first_completion_s=makespan_s / 2,
        warmup_s=0.0,
        steady_s=makespan_s,
        cooldown_s=0.0,
        aggregate_bubble_s=0.0,
        bubble_ratio=0.0,
        bottleneck_stage_id=0,
        stage_busy_s=(makespan_s, makespan_s),
        stage_idle_s=(0.0, 0.0),
        microbatch_completion_s=(makespan_s / 2, makespan_s),
        stage_compute_intervals_s=((), ()),
        stage_outgoing_transfer_intervals_s=((), ()),
        stage_incoming_transfer_intervals_s=((), ()),
        stage_incoming_payload_bytes_s=(0, 0),
        stage_outgoing_payload_bytes_s=(0, 0),
        stage_communication_buffer_bytes_s=(0, 0),
    )
    repeated = RepeatedPipelineEstimate(
        first_wave=first_wave,
        repeated_makespan_s=makespan_s + interval_s,
        same_slot_period_s=(worst_tpot_s, worst_tpot_s),
        worst_tpot_s=worst_tpot_s,
        completed_tokens=2,
        measured_interval_s=interval_s,
    )
    return first_wave, repeated


_make_pp_disagg_strategy = partial(_make_pp_strategy, DisaggThroughputOptimizer)


class TestDisaggPipelineParallel(unittest.TestCase):
    def test_pp_summary_preserves_profiling_lookup_evidence(self):
        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        metrics = _PPMetrics(
            profile,
            profiling_source_times_s={"measured": 0.008, "analytic": 0.002},
            profiling_miss_reasons={"outside_axis_boundary": 3},
        )
        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=1,
            input_length=4,
            output_length=8,
            max_batched_tokens=4,
            serving_cost=0,
        )

        with patch.object(strategy, "_get_forward_info", return_value=metrics):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

        self.assertEqual(row["profiling_sources"], "Measured 80.00 | Analytic 20.00")
        self.assertEqual(row["profiling_source_scope"], "modeled_forward_lookup_latency")
        self.assertEqual(row["profiling_result"], "hybrid")
        self.assertEqual(row["profiling_misses"], "outside_axis_boundary x3")

    def test_prefill_formula_uses_wave_makespan_and_serving_cost(self):
        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=2,
            input_length=4,
            output_length=8,
            max_batched_tokens=8,
            serving_cost=5,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

        self.assertEqual(row["ttft"], 6005.0)
        self.assertIsNone(row["tpot"])
        self.assertAlmostEqual(row["token/s"], 2 * 4 * 1000.0 / 6005.0, places=3)

    def test_decode_formula_separates_worst_tpot_and_measured_interval(self):
        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        _, repeated = _pp_schedule_estimates(
            makespan_s=10.0,
            interval_s=6.0,
            worst_tpot_s=8.0,
        )
        optimizer_data = OptimizerData(
            ttft_limits=None,
            tpot_limits=10000,
            batch_size=2,
            input_length=512,
            output_length=128,
            max_batched_tokens=2048,
            serving_cost=2,
            num_mtp_tokens=0,
            mtp_acceptance_rate=[],
        )
        with (
            patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)),
            patch(
                "serving_cast.service.base_throughput_optimizer.estimate_repeated_pipeline",
                return_value=repeated,
            ),
        ):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

        self.assertIsNone(row["ttft"])
        self.assertEqual(row["tpot"], 8002.0)
        self.assertAlmostEqual(row["token/s"], 2.0 / 6.002, places=3)

    def test_pp_decode_applies_mtp_fold(self):
        """PP>1 decode TPOT and throughput are folded by (accept+1) when MTP is enabled."""
        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        _, repeated = _pp_schedule_estimates(
            makespan_s=10.0,
            interval_s=6.0,
            worst_tpot_s=8.0,
        )
        optimizer_data = OptimizerData(
            ttft_limits=None,
            tpot_limits=10000,
            batch_size=2,
            input_length=512,
            output_length=128,
            max_batched_tokens=2048,
            serving_cost=2,
            num_mtp_tokens=2,
            speculative_method="mtp",
            acceptance_length=1.5,
            mtp_acceptance_rate=[],
        )
        with (
            patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)),
            patch(
                "serving_cast.service.base_throughput_optimizer.estimate_repeated_pipeline",
                return_value=repeated,
            ),
        ):
            row = strategy.get_inference_info(optimizer_data).get_summary_df().iloc[0]

        # fold = clamp(1.5, 0, 2) + 1 = 2.5
        fold = 2.5
        expected_tpot = 8000.0 / fold + 2.0  # 3202.0
        expected_interval_s = 6.0 / fold + 0.002  # 2.402
        self.assertIsNone(row["ttft"])
        self.assertAlmostEqual(row["tpot"], expected_tpot, places=2)
        self.assertAlmostEqual(row["token/s"], 2.0 / expected_interval_s, places=3)

    def test_dp_scaling_uses_shared_matrix(self):
        for name, dp, pp, tp, batch_size, expected_concurrency in _PP_SCALING_MATRIX:
            with self.subTest(name=name):
                summary, row = _run_pp_scaling_case(
                    DisaggThroughputOptimizer,
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
                self.assertFalse(summary.check_early_stop_flag())

    def test_microbatch_profile_scales_concurrency_by_dp(self):
        strategy = _make_pp_disagg_strategy(dp=4, pp=2, tp=1, microbatch_size=2)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        captured = []

        def fake_forward(concurrency, optimizer_data, is_decode, **kwargs):
            captured.append(concurrency)
            return _PPMetrics(profile)

        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=2,
            input_length=4,
            output_length=8,
            max_batched_tokens=8,
            serving_cost=0,
        )
        with patch.object(strategy, "_get_forward_info", side_effect=fake_forward):
            strategy.get_inference_info(optimizer_data)

        self.assertEqual(captured, [2 * 4])

    def test_memory_gate_preserves_exact_oom_threshold(self):
        profile = _pp_profile((2.0, 2.0), runtime_peak_bytes=2000, include_transfers=False)
        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=1,
            input_length=4,
            output_length=8,
            max_batched_tokens=4,
            serving_cost=0,
        )
        for budget_bytes, expected_oom in ((2000, False), (1999, True)):
            with self.subTest(budget_bytes=budget_bytes):
                strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
                strategy.model_runner.total_device_memory_gb = budget_bytes / BYTES_TO_GB
                with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
                    summary = strategy.get_inference_info(optimizer_data)
                self.assertEqual(summary.check_early_stop_flag(), expected_oom)

    def test_prefill_validation_matrix(self):
        from serving_cast.service.utils import UnsupportedPPConfigurationError

        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        for name, optimizer_data, expected in _pp_prefill_validation_cases(disaggregated=True):
            with self.subTest(name=name):
                with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
                    with self.assertRaises(UnsupportedPPConfigurationError) as ctx:
                        strategy.get_inference_info(optimizer_data)
                self.assertIn(expected, str(ctx.exception))

    def test_prefill_budget_is_scoped_per_dp_replica(self):
        strategy = _make_pp_disagg_strategy(dp=2, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=2,
            input_length=1024,
            output_length=8,
            max_batched_tokens=2048,
            serving_cost=0,
        )
        with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
            summary = strategy.get_inference_info(optimizer_data)

        self.assertFalse(summary.check_early_stop_flag())
        # concurrency = batch_size * dp (PP does not add request replicas)
        self.assertEqual(summary.get_summary_df().iloc[0]["concurrency"], 4)

    def test_chunked_prefill_is_supported_for_pp(self):
        """PP>1 + chunked prefill is supported: each chunk is a pipeline microbatch."""
        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        # input_length=10 > max_batched_tokens=4 → 3 chunks (4,4,2)
        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=1,
            input_length=10,
            output_length=8,
            max_batched_tokens=4,
            serving_cost=0,
        )
        with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
            summary = strategy.get_inference_info(optimizer_data)
        # Should not raise; chunked prefill is now supported
        self.assertFalse(summary.check_early_stop_flag())
        # Confirm each chunk actually participated in the pipeline schedule:
        # input_length=10 split by max_batched_tokens=4 yields 3 microbatches.
        self.assertEqual(summary.get_summary_df().iloc[0]["prefill_num_chunks"], 3)

    def test_batch_budget_returns_early_stop_not_exception(self):
        """Batch exceeding token budget returns early-stop, not UnsupportedPPConfigurationError."""
        from serving_cast.service.optimizer_summary import EARLY_STOP_PREFILL_OOM

        strategy = _make_pp_disagg_strategy(dp=1, pp=2, tp=1)
        profile = _pp_profile((2.0, 2.0), include_transfers=False)
        # batch_size=4 * input_length=1024 = 4096 > max_batched_tokens=2048
        # but max_batch_by_tokens = 2048//1024 = 2 >= 1, so early-stop (shrink search)
        optimizer_data = OptimizerData(
            ttft_limits=100000,
            tpot_limits=None,
            batch_size=4,
            input_length=1024,
            output_length=8,
            max_batched_tokens=2048,
            serving_cost=0,
        )
        with patch.object(strategy, "_get_forward_info", return_value=_PPMetrics(profile)):
            summary = strategy.get_inference_info(optimizer_data)
        self.assertTrue(summary.check_early_stop_flag())
        self.assertEqual(summary.get_early_stop_reason(), EARLY_STOP_PREFILL_OOM)


if __name__ == "__main__":
    unittest.main()
