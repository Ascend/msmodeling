# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock, Mock, patch

from serving_cast.parallel_runner import ParallelRunner
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import OptimizerData
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile

from .test_common import SimpleArgs


class RuntimeErrorExecutor:
    def __init__(self, max_workers=None, initializer=None):
        self.initializer = initializer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        if self.initializer is not None:
            self.initializer()

        class BrokenResultIterator:
            def __iter__(self_inner):
                return self_inner

            def __next__(self_inner):
                raise BrokenProcessPool

        return BrokenResultIterator()


class TestTaskRunner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.args = SimpleArgs()
        self.args.serving_cost = 0
        self.args.jobs = 4
        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]

    def test_get_user_config_multiple_tp(self):
        """Test _get_user_config with multiple TP values"""
        self.args.tp_sizes = [2, 4]
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)

        configs = list(task_runner._get_user_config())
        # Should only include TPs that divide evenly into num_devices
        self.assertEqual(len(configs), 2)  # TP=2, 4 all divide 4
        tps = [config.tp_size for config in configs]
        self.assertIn(2, tps)
        self.assertIn(4, tps)
        for config in configs:
            self.assertEqual(config.ep_size, 4)
            self.assertEqual(config.moe_dp_size, 1)

    def test_optimizer_data_loads_length_distribution_from_input_length_file(self):
        self.args.input_length = "serving_cast/example/length_distribution.yaml"

        task_runner = ParallelRunner(self.args)

        self.assertIsNone(task_runner.optimizer_data.input_length)
        self.assertIsNotNone(task_runner.optimizer_data.length_distribution)

    def test_get_user_config_default_tps(self):
        """Test _get_user_config with default TP values"""
        self.args.tp_sizes = []
        self.args.num_devices = 8

        task_runner = ParallelRunner(self.args)

        configs = list(task_runner._get_user_config())
        # Default TPs should be powers of 2 up to num_devices
        expected_tps = [1, 2, 4, 8]  # 2^0 to 2^3
        actual_tps = [config.tp_size for config in configs]
        for expected_tp in expected_tps:
            self.assertIn(expected_tp, actual_tps)

    def test_get_user_config_tp_ep_combinations(self):
        """Test searching TP/EP with fixed MOE-DP=1."""
        self.args.tp_sizes = [1, 2, 4]
        self.args.ep_sizes = [1, 2, 4]
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())
        self.assertEqual(len(configs), 9)

        target = next(config for config in configs if config.tp_size == 2 and config.ep_size == 2)
        self.assertEqual(target.dp_size, 2)
        self.assertEqual(target.moe_dp_size, 1)
        self.assertEqual(target.moe_tp_size, 2)

    def test_get_user_config_tp_ep_default_ranges(self):
        """Test TP/EP default ranges."""
        self.args.tp_sizes = []
        self.args.ep_sizes = []
        self.args.num_devices = 8

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())
        self.assertEqual(len(configs), 16)
        target = next(config for config in configs if config.tp_size == 8 and config.ep_size == 8)
        self.assertEqual(target.dp_size, 1)
        self.assertEqual(target.moe_dp_size, 1)
        self.assertEqual(target.moe_tp_size, 1)

    def test_get_user_config_tp_ep_moe_dp_combinations(self):
        """Test searching TP/EP/MOE-DP combinations."""
        self.args.tp_sizes = [1, 2]
        self.args.ep_sizes = [1, 2, 4]
        self.args.moe_dp_sizes = [1, 2, 4]
        self.args.num_devices = 8
        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())
        keys = {(config.tp_size, config.ep_size, config.moe_dp_size) for config in configs}
        self.assertIn((1, 2, 4), keys)
        self.assertIn((2, 4, 2), keys)
        for config in configs:
            self.assertEqual(
                config.moe_tp_size,
                self.args.num_devices // (config.ep_size * config.moe_dp_size),
            )

    def test_get_user_config_tp_ep_moe_dp_default_ranges(self):
        """Test TP/EP/MOE-DP default ranges."""
        self.args.tp_sizes = []
        self.args.ep_sizes = []
        self.args.moe_dp_sizes = []
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())
        self.assertEqual(len(configs), 18)
        keys = {(config.tp_size, config.ep_size, config.moe_dp_size) for config in configs}
        self.assertIn((4, 4, 1), keys)
        self.assertIn((2, 2, 2), keys)

    def test_get_user_config_num_mtp_tokens_combinations(self):
        """Test searching num_mtp_tokens together with parallel candidates."""
        self.args.tp_sizes = [1, 2]
        self.args.num_devices = 2
        self.args.num_mtp_token_sizes = [0, 2]

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())

        self.assertEqual(len(configs), 4)
        keys = {(config.tp_size, config.num_mtp_tokens) for config in configs}
        self.assertEqual(keys, {(1, 0), (1, 2), (2, 0), (2, 2)})

    def test_get_user_config_chrome_trace_names_include_num_mtp_tokens(self):
        """Test MTP search candidates do not overwrite the same chrome trace file."""
        self.args.tp_sizes = [1]
        self.args.num_devices = 1
        self.args.num_mtp_token_sizes = [0, 2]
        self.args.chrome_trace = "trace.json"

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())

        trace_names = {config.num_mtp_tokens: config.chrome_trace for config in configs}
        self.assertEqual(trace_names[0], "trace_tp1dp1mtp0.json")
        self.assertEqual(trace_names[2], "trace_tp1dp1mtp2.json")
        self.assertEqual(len(set(trace_names.values())), 2)

    def test_get_user_config_tp_ep_num_mtp_tokens_combinations(self):
        """Test TP/EP/MTP search combinations for the throughput optimizer CLI pattern."""
        self.args.tp_sizes = [1, 2]
        self.args.ep_sizes = [1, 2]
        self.args.num_mtp_token_sizes = [1, 2, 3]
        self.args.num_devices = 8

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())

        self.assertEqual(len(configs), 12)
        keys = {(config.tp_size, config.ep_size, config.num_mtp_tokens) for config in configs}
        self.assertIn((1, 1, 1), keys)
        self.assertIn((1, 2, 3), keys)
        self.assertIn((2, 1, 2), keys)
        self.assertIn((2, 2, 3), keys)

    def test_optimizer_data_uses_safe_num_mtp_tokens_for_multi_candidate_search(self):
        """Test base OptimizerData does not pin the first MTP candidate before task dispatch."""
        self.args.num_mtp_tokens = 1
        self.args.num_mtp_token_sizes = [1, 2, 3]

        task_runner = ParallelRunner(self.args)

        self.assertEqual(task_runner.optimizer_data.num_mtp_tokens, 0)
        configs = list(task_runner._get_user_config())
        self.assertEqual({config.num_mtp_tokens for config in configs}, {1, 2, 3})

    def test_run_with_tpot_limit(self):
        """Test run method with TPOT limit"""
        self.args.tpot_limits = 50
        self.args.batch_range = [2, 2]
        task_runner = ParallelRunner(self.args)
        result = task_runner.run_agg()

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], OptimizerSummary)

        summary_df = result[0].get_summary_df()
        row = summary_df.iloc[0]
        self.assertEqual(row["concurrency"], 2)

    def test_given_mocked_executor_when_called_then_returns_empty_list_and_verifies_executor_initialization(
        self,
    ):
        executor_cls = Mock()
        executor_inst = MagicMock()
        executor_cls.return_value = executor_inst
        executor_inst.__enter__.return_value = executor_inst
        executor_inst.__exit__.return_value = None
        initializer = Mock()

        def test_map(fn, *iterables, timeout=None, chunksize=1):
            initializer()
            return []

        executor_inst.map = test_map

        task_runner = ParallelRunner(self.args, executor_cls, initializer)
        df_list = task_runner._get_df_list(task_runner.optimizer_data)

        executor_cls.assert_called_once_with(max_workers=self.args.jobs, initializer=initializer)
        initializer.assert_called_once_with()

        self.assertEqual(df_list, [])

    def test_given_worker_initializer_raises_runtime_error_when_called_then_raises_and_logs_expected_errors(
        self,
    ):
        initializer = Mock()
        task_runner = ParallelRunner(
            self.args,
            executor_class=RuntimeErrorExecutor,
            worker_initializer=initializer,
        )

        with self.assertLogs("serving_cast.parallel_runner", "ERROR") as cm:
            self.assertRaises(RuntimeError, task_runner._get_df_list, task_runner.optimizer_data)
            self.assertTrue(len(cm.output), 3)
            self.assertRegex(
                cm.output[0],
                "ERROR:serving_cast.parallel_runner:A worker process crashed unexpectedly during execution. "
                "Common causes: memory issues, unpicklable objects, or unhandled exceptions in worker.",
            )
            self.assertRegex(
                cm.output[1],
                "ERROR:serving_cast.parallel_runner:Executor: RuntimeErrorExecutor, Workers: 4",
            )

    def test_run_disagg_with_ttft_and_tpot_limit(self):
        """Test run_disagg method with ttft and tpot limit"""
        self.args.ttft_limits = 1000
        self.args.tpot_limits = 50
        self.args.batch_range = [2, 2]
        self.args.disagg = True
        task_runner = ParallelRunner(self.args)
        result = task_runner.run_disagg()

        # Prefill and decode
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], OptimizerSummary)

        prefill_df = result[0].get_summary_df()
        row = prefill_df.iloc[0]
        self.assertEqual(row["concurrency"], 2)
        self.assertIsNone(row["tpot"])

        decode_df = result[1].get_summary_df()
        row = decode_df.iloc[0]
        self.assertEqual(row["concurrency"], 2)
        self.assertIsNone(row["ttft"])

    def test_submit_task(self):
        """Test _submit_task method"""
        user_config = UserInputConfig.from_args(self.args)
        optimizer_data = OptimizerData(
            input_length=self.args.input_length,
            output_length=self.args.output_length,
            ttft_limits=1000,
            tpot_limits=50,
            max_batched_tokens=self.args.max_batched_tokens,
            num_devices=self.args.num_devices,
            num_mtp_tokens=1,
            mtp_acceptance_rate=[0.9],
        )

        task_runner = ParallelRunner(self.args)
        result = task_runner._submit_task(user_config, optimizer_data)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, OptimizerSummary)
        row = result.get_summary_df().iloc[0]
        self.assertEqual(row["model_id"], self.args.model_id)
        self.assertEqual(row["parallel"], "TP=1 | PP=1 | DP=1")


class TestParallelRunnerPDMode(unittest.TestCase):
    """Test cases for ParallelRunner PD ratio mode."""

    def setUp(self):
        """Set up test fixtures for PD mode."""
        self.args = SimpleArgs()
        self.args.serving_cost = 0
        self.args.jobs = 4
        self.args.enable_optimize_prefill_decode_ratio = True
        self.args.prefill_devices_per_instance = 4
        self.args.decode_devices_per_instance = 2
        self.args.input_length = 1024
        self.args.output_length = 1024
        self.args.ttft_limits = 100
        self.args.tpot_limits = 10
        self.args.num_devices = 8
        self.args.batch_range = [1, 16]

    def test_add_summary_result_with_empty_list(self):
        """Test _add_summary_result with empty df_list."""
        task_runner = ParallelRunner(self.args)
        optimizer_data = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )

        task_runner._add_summary_result([], optimizer_data)
        self.assertEqual(len(task_runner.summary_result), 0)

    def test_add_summary_result_with_valid_df(self):
        """Test _add_summary_result with valid DataFrame."""
        import pandas as pd

        task_runner = ParallelRunner(self.args)
        optimizer_data = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )

        df = pd.DataFrame(
            {
                "ttft": [100.0],
                "tpot": [10.0],
                "concurrency": [10],
                "parallel": ["tp4pp1dp1"],
                "batch_size": [4],
            }
        )

        summary = OptimizerSummary(optimizer_data)
        summary.set_summary_df(df)

        task_runner._add_summary_result([summary], optimizer_data)
        self.assertEqual(len(task_runner.summary_result), 1)

    def test_add_summary_result_selects_tightest_memory_info(self):
        """Merged multi-TP summaries should keep the most constrained memory info."""
        import pandas as pd

        task_runner = ParallelRunner(self.args)
        optimizer_data = OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )
        df = pd.DataFrame({"ttft": [100.0], "tpot": [10.0], "token/s": [1.0]})
        loose_memory_info = {
            "total_device_memory_gb": 64.0,
            "reserved_memory_gb": 4.0,
            "device_memory_available_gb": 8.0,
        }
        tight_memory_info = {
            "total_device_memory_gb": 48.0,
            "reserved_memory_gb": 3.0,
            "device_memory_available_gb": 4.0,
        }

        first = OptimizerSummary(optimizer_data)
        first.set_summary_df(df)
        second = OptimizerSummary(optimizer_data)
        second.set_summary_df(df)
        second.set_memory_info(loose_memory_info)
        third = OptimizerSummary(optimizer_data)
        third.set_summary_df(df)
        third.set_memory_info(tight_memory_info)

        task_runner._add_summary_result([first, second, third], optimizer_data)

        self.assertEqual(task_runner.summary_result[0].get_memory_info(), tight_memory_info)

    def test_run_pd_ratio_combines_prefill_and_decode_results(self):
        """_run_pd_ratio should submit both phases and wrap the optimized result."""
        import pandas as pd

        class ImmediateFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class ImmediateThreadPool:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def submit(self, fn, *args, **kwargs):
                return ImmediateFuture(fn(*args, **kwargs))

        class RecordingPDRatioOptimizer:
            instances = []

            def __init__(self, output_length):
                self.output_length = output_length
                self.p_df = None
                self.d_df = None
                self.instances.append(self)

            def set_p_results(self, p_df):
                self.p_df = p_df

            def set_d_results(self, d_df):
                self.d_df = d_df

            def optimize(self):
                return pd.DataFrame(
                    {
                        "balanced_qps": [12.0],
                        "pd_ratio": [0.5],
                        "p_qps": [24.0],
                        "d_qps": [12.0],
                    }
                )

        task_runner = ParallelRunner(self.args)
        phase_calls = []

        def fake_run_pd_phase(devices_per_instance, is_prefill):
            phase_calls.append((devices_per_instance, is_prefill))
            if is_prefill:
                df = pd.DataFrame({"p_qps": [24.0]})
                df.attrs["memory_info"] = {
                    "total_device_memory_gb": 64.0,
                    "reserved_memory_gb": 4.0,
                    "device_memory_available_gb": 8.0,
                }
                return df
            df = pd.DataFrame({"d_qps": [12.0]})
            df.attrs["memory_info"] = {
                "total_device_memory_gb": 48.0,
                "reserved_memory_gb": 3.0,
                "device_memory_available_gb": 3.0,
            }
            return df

        task_runner._run_pd_phase = fake_run_pd_phase

        with (
            patch("serving_cast.parallel_runner.ThreadPoolExecutor", ImmediateThreadPool),
            patch("serving_cast.parallel_runner.PDRatioThroughputOptimizer", RecordingPDRatioOptimizer),
        ):
            result = task_runner._run_pd_ratio()

        self.assertEqual(
            phase_calls,
            [
                (self.args.prefill_devices_per_instance, True),
                (self.args.decode_devices_per_instance, False),
            ],
        )
        optimizer = RecordingPDRatioOptimizer.instances[0]
        self.assertEqual(optimizer.output_length, self.args.output_length)
        self.assertEqual(optimizer.p_df.iloc[0]["p_qps"], 24.0)
        self.assertEqual(optimizer.d_df.iloc[0]["d_qps"], 12.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].get_summary_df().iloc[0]["balanced_qps"], 12.0)
        self.assertEqual(result[0].get_memory_info()["total_device_memory_gb"], 48.0)
        self.assertEqual(result[0].get_memory_info()["device_memory_available_gb"], 3.0)

    def test_pd_phase_forces_disaggregation_strategy(self):
        """PD ratio sub-phases should use disaggregated optimizer semantics."""
        import pandas as pd

        class InlineExecutor:
            def __init__(self, max_workers=None, initializer=None):
                self.initializer = initializer

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def map(self, fn, *iterables, timeout=None, chunksize=1):
                if self.initializer is not None:
                    self.initializer()
                return [fn(item) for item in iterables[0]]

        class RecordingParallelRunner(ParallelRunner):
            def __init__(self, args):
                super().__init__(args, executor_class=InlineExecutor)
                self.disagg_modes = []

            def _submit_task(
                self,
                user_input,
                overwrite_optimizer_data,
                disagg_mode=None,
            ):
                self.disagg_modes.append(disagg_mode)
                summary = OptimizerSummary(overwrite_optimizer_data)
                summary.set_summary_df(pd.DataFrame({"ttft": [100.0], "concurrency": [user_input.tp_size]}))
                summary.set_memory_info(
                    {
                        "total_device_memory_gb": 64.0,
                        "reserved_memory_gb": 4.0,
                        "device_memory_available_gb": 8.0 / user_input.tp_size,
                    }
                )
                return summary

        self.args.disagg = False
        self.args.tp_sizes = [1, 2]
        self.args.num_devices = 2
        task_runner = RecordingParallelRunner(self.args)
        result_df = task_runner._run_pd_phase(
            devices_per_instance=self.args.prefill_devices_per_instance,
            is_prefill=True,
        )

        self.assertFalse(self.args.disagg)
        self.assertEqual(result_df.iloc[0]["ttft"], 100.0)
        self.assertEqual(result_df.attrs["memory_info"]["total_device_memory_gb"], 64.0)
        self.assertEqual(result_df.attrs["memory_info"]["device_memory_available_gb"], 4.0)
        self.assertTrue(task_runner.disagg_modes)
        self.assertTrue(all(task_runner.disagg_modes))


if __name__ == "__main__":
    unittest.main()
