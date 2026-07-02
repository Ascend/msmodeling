# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from serving_cast.service.disagg_throughput_optimizer import DisaggThroughputOptimizer
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile

from .test_common import SimpleArgs


def _simple_length_distribution():
    return LengthDistribution(
        bins=[
            LengthBin(min_tokens=0, max_tokens=500, weight=0.6),
            LengthBin(min_tokens=500, max_tokens=1500, weight=0.4),
        ]
    )


class TestDisaggStrategy(unittest.TestCase):
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

    def test_chunked_prefill_splits_each_chunk_into_token_budget_waves(self):
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

            return DummyMetrics()

        with patch.object(self.strategy, "_get_forward_info", side_effect=fake_forward):
            result = self.strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertEqual(
            captured_calls,
            [
                (1, 4, 4),
                (1, 4, 8),
                (2, 2, 10),
            ],
        )
        self.assertTrue(
            all(
                concurrency * query_len <= optimizer_data.max_batched_tokens
                for concurrency, query_len, _ in captured_calls
            )
        )
        self.assertEqual(row["prefill_num_chunks"], 3)
        self.assertEqual(row["ttft"], 12.0)
        self.assertEqual(row["percentage_breakdowns"], "Mem 18.00 | Comm 82.00 | Cube 0.00 | Vec 0.00")

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
        self.assertEqual(captured_calls, [(1, 4, 4), (1, 4, 8)])
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
            max_batched_tokens=2048,
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
            return_value=(batch_result, composition_rows),
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

        with (
            patch.object(
                strategy,
                "_get_batched_forward_info",
                return_value=(
                    Mock(
                        execution_time_s={"analytic": 0.123},
                        device_memory_available_gb=2.0,
                        breakdowns={"prefill": {"Mem": 1.0}},
                    ),
                    [],
                ),
            ),
            patch.object(strategy, "_get_forward_info") as mock_forward,
        ):
            result = strategy.get_inference_info(optimizer_data)

        mock_forward.assert_not_called()
        self.assertFalse(result.check_early_stop_flag())

    def test_distribution_prefill_throughput_uses_global_tokens_instead_of_per_rank_tokens(self):
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
            return_value=(batch_result, composition_rows),
        ):
            result = strategy.get_inference_info(optimizer_data)

        row = result.get_summary_df().iloc[0]
        self.assertEqual(row["concurrency"], 20)
        self.assertEqual(row["token/s"], 11000000.0)


if __name__ == "__main__":
    unittest.main()
