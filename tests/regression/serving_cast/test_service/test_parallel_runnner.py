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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from serving_cast.parallel_runner import ParallelRunner
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import OptimizerData, UnsupportedPPConfigurationError
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from tests.helpers.model_assets import vendored_model_config_path

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

    def _assert_pp_schedule_schema(self, row, expected_pp_size):
        self.assertEqual(row["pp_size"], expected_pp_size)
        self.assertGreater(row["pp_makespan_ms"], 0.0)
        self.assertGreater(row["pp_bubble_ratio"], 0.0)
        self.assertIn(row["pp_bottleneck_stage"], (0, 1))

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

    def test_optimizer_data_omits_dflash_fields_when_disabled(self):
        task_runner = ParallelRunner(self.args)

        self.assertIsNone(task_runner.optimizer_data.dflash_block_size)
        self.assertIsNone(task_runner.optimizer_data.dflash_acceptance_length)
        self.assertIsNone(task_runner.optimizer_data.dspark_block_size)
        self.assertIsNone(task_runner.optimizer_data.dspark_acceptance_length)
        configs = list(task_runner._get_user_config())
        self.assertTrue(configs)
        self.assertIsNone(configs[0].speculative_method)
        self.assertFalse(configs[0].dflash)
        self.assertFalse(configs[0].dspark)

    def test_optimizer_data_fills_dflash_fields_when_enabled(self):
        # Fixture matches post-arg_parse args: block is already resolved (n+1),
        # and N candidates occupy the MTP search slot.
        self.args.speculative_method = "dflash"
        self.args.num_speculative_tokens = 15
        self.args.draft_block_size = 16
        self.args.acceptance_length = 5.0
        self.args.num_draft_layers = 0
        self.args.draft_model_config_path = None
        self.args.num_mtp_tokens = 0
        self.args.num_mtp_token_sizes = [15]
        self.args.chrome_trace = "trace.json"

        task_runner = ParallelRunner(self.args)

        self.assertEqual(task_runner.optimizer_data.dflash_block_size, 16)
        self.assertEqual(task_runner.optimizer_data.dflash_acceptance_length, 5.0)
        self.assertIsNone(task_runner.optimizer_data.dspark_block_size)
        configs = list(task_runner._get_user_config())
        self.assertTrue(configs)
        self.assertEqual(configs[0].speculative_method, "dflash")
        self.assertTrue(configs[0].dflash)
        self.assertFalse(configs[0].dspark)
        self.assertEqual(configs[0].num_mtp_tokens, 0)
        self.assertEqual(configs[0].draft_block_size(), 16)
        self.assertIn("dflash16", configs[0].chrome_trace)

    def test_pp_build_user_input_sets_num_mtp_zero_for_dflash(self):
        """PP path must set num_mtp_tokens=0 for DFlash (avoids MtpConfig/DflashConfig conflict)."""
        self.args.speculative_method = "dflash"
        self.args.num_speculative_tokens = 15
        self.args.draft_block_size = 16
        self.args.acceptance_length = 5.0
        self.args.num_draft_layers = 0
        self.args.draft_model_config_path = None
        self.args.num_mtp_tokens = 0
        self.args.num_mtp_token_sizes = [15]
        self.args.num_devices = 2
        self.args.tp_sizes = [1]
        self.args.ep_sizes = [1]
        self.args.pp_sizes = [2]
        self.args.num_hidden_layers_override = 2

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())

        self.assertTrue(configs)
        # PP path must reset num_mtp_tokens=0 for DFlash to avoid
        # ConfigResolver building MtpConfig alongside DflashConfig.
        self.assertEqual(configs[0].speculative_method, "dflash")
        self.assertEqual(configs[0].num_mtp_tokens, 0)
        self.assertTrue(configs[0].dflash)
        self.assertEqual(configs[0].num_speculative_tokens, 15)
        self.assertEqual(configs[0].draft_block_size(), 16)
        self.assertEqual(configs[0].pp_size, 2)

    def test_pp_build_user_input_multi_n_dflash_sets_per_candidate_spec_tokens(self):
        """PP multi-N DFlash: each candidate gets its own num_speculative_tokens."""
        self.args.speculative_method = "dflash"
        self.args.num_speculative_tokens = 3  # CLI first value (multi_n sets to n_candidates[0])
        self.args.draft_block_size = 0  # multi-n: resolved per-candidate, not at args level
        self.args.acceptance_length = 5.0
        self.args.num_draft_layers = 0
        self.args.draft_model_config_path = None
        self.args.num_mtp_tokens = 0
        self.args.num_mtp_token_sizes = [3, 7]
        self.args.num_devices = 4
        self.args.tp_sizes = [1]
        self.args.ep_sizes = [1]
        self.args.pp_sizes = [2]
        self.args.num_hidden_layers_override = 2

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())

        self.assertEqual(len(configs), 2)
        # Candidate 1: N=3
        self.assertEqual(configs[0].num_speculative_tokens, 3)
        self.assertEqual(configs[0].num_mtp_tokens, 0)
        self.assertEqual(configs[0].draft_block_size(), 4)
        # Candidate 2: N=7 — must NOT inherit base's num_speculative_tokens=3
        self.assertEqual(configs[1].num_speculative_tokens, 7)
        self.assertEqual(configs[1].num_mtp_tokens, 0)
        self.assertEqual(configs[1].draft_block_size(), 8)

    def test_pp_build_user_input_multi_n_dspark_sets_per_candidate_spec_tokens(self):
        """PP multi-N DSpark: each candidate gets its own num_speculative_tokens."""
        self.args.speculative_method = "dspark"
        self.args.num_speculative_tokens = 3
        self.args.draft_block_size = 0  # multi-n: resolved per-candidate, not at args level
        self.args.acceptance_length = 5.0
        self.args.num_draft_layers = 0
        self.args.draft_model_config_path = None
        self.args.num_mtp_tokens = 0
        self.args.num_mtp_token_sizes = [3, 7]
        self.args.num_devices = 4
        self.args.tp_sizes = [1]
        self.args.ep_sizes = [1]
        self.args.pp_sizes = [2]
        self.args.num_hidden_layers_override = 2

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())

        self.assertEqual(len(configs), 2)
        for config, n in zip(configs, [3, 7]):
            # DSpark reuses the MTP search slot but must carry num_mtp_tokens=0
            # (ConfigResolver would build MtpConfig alongside DsparkConfig).
            self.assertEqual(config.speculative_method, "dspark")
            self.assertEqual(config.num_speculative_tokens, n)
            self.assertEqual(config.num_mtp_tokens, 0)
            self.assertTrue(config.dspark)
            self.assertEqual(config.draft_block_size(), n + 1)

    def test_pp_build_user_input_prefill_forces_mtp_zero(self):
        """Disaggregated Prefill: legacy/new MTP candidates must be forced off under PP search.

        Mirrors the non-PP path's ``if is_prefill: mtp_list = [0]`` so TTFT
        stays a pure prefill measurement; DFlash/DSpark are excluded (their
        draft runs during Prefill by RFC design).
        """
        for speculative_method in ("mtp", None):
            with self.subTest(speculative_method=speculative_method):
                self.args.speculative_method = speculative_method
                self.args.num_speculative_tokens = 0
                self.args.draft_block_size = 0
                self.args.acceptance_length = 5.0
                self.args.num_draft_layers = 0
                self.args.draft_model_config_path = None
                self.args.num_mtp_tokens = 2
                self.args.num_mtp_token_sizes = [2]
                self.args.num_devices = 2
                self.args.tp_sizes = [1]
                self.args.ep_sizes = [1]
                self.args.pp_sizes = [2]
                self.args.num_hidden_layers_override = 2

                task_runner = ParallelRunner(self.args)
                prefill_configs = list(task_runner._get_user_config(is_prefill=True))
                decode_configs = list(task_runner._get_user_config(is_prefill=False))

                self.assertTrue(prefill_configs)
                for config in prefill_configs:
                    self.assertEqual(config.num_mtp_tokens, 0)
                    self.assertEqual(config.pp_size, 2)
                # Decode phase keeps the MTP search slot.
                self.assertTrue(decode_configs)
                for config in decode_configs:
                    self.assertEqual(config.num_mtp_tokens, 2)

    def test_pp_build_user_input_prefill_keeps_dflash_speculative_tokens(self):
        """Disaggregated Prefill keeps DFlash/DSpark draft candidates (RFC-defined behavior)."""
        self.args.speculative_method = "dflash"
        self.args.num_speculative_tokens = 3
        self.args.draft_block_size = 0
        self.args.acceptance_length = 5.0
        self.args.num_draft_layers = 0
        self.args.draft_model_config_path = None
        self.args.num_mtp_tokens = 0
        self.args.num_mtp_token_sizes = [3]
        self.args.num_devices = 2
        self.args.tp_sizes = [1]
        self.args.ep_sizes = [1]
        self.args.pp_sizes = [2]
        self.args.num_hidden_layers_override = 2

        task_runner = ParallelRunner(self.args)
        prefill_configs = list(task_runner._get_user_config(is_prefill=True))

        self.assertTrue(prefill_configs)
        for config in prefill_configs:
            self.assertEqual(config.speculative_method, "dflash")
            self.assertTrue(config.dflash)
            self.assertEqual(config.num_mtp_tokens, 0)
            # acceptance clamped per-candidate to n = block - 1 = 3
            self.assertEqual(config.acceptance_length, 3.0)
            self.assertEqual(config.num_speculative_tokens, 3)
            self.assertEqual(config.draft_block_size(), 4)

    def test_optimizer_data_fills_dspark_fields_when_enabled(self):
        # Fixture matches post-arg_parse args: block is already resolved (n+1),
        # and N candidates occupy the MTP search slot.
        self.args.speculative_method = "dspark"
        self.args.num_speculative_tokens = 7
        self.args.draft_block_size = 8
        self.args.acceptance_length = 5.0
        self.args.dspark_markov_rank = 256
        self.args.dspark_markov_head = "vanilla"
        self.args.num_draft_layers = 0
        self.args.draft_model_config_path = None
        self.args.num_mtp_tokens = 0
        self.args.num_mtp_token_sizes = [7]
        self.args.chrome_trace = "trace.json"

        task_runner = ParallelRunner(self.args)

        self.assertEqual(task_runner.optimizer_data.dspark_block_size, 8)
        self.assertEqual(task_runner.optimizer_data.dspark_acceptance_length, 5.0)
        self.assertEqual(task_runner.optimizer_data.dspark_markov_rank, 256)
        self.assertIsNone(task_runner.optimizer_data.dflash_block_size)
        configs = list(task_runner._get_user_config())
        self.assertTrue(configs)
        self.assertEqual(configs[0].speculative_method, "dspark")
        self.assertTrue(configs[0].dspark)
        self.assertFalse(configs[0].dflash)
        self.assertEqual(configs[0].num_mtp_tokens, 0)
        self.assertIn("dspark8", configs[0].chrome_trace)

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

    def test_get_user_config_prefill_disables_mtp(self):
        """Prefill must not inherit speculative Decode candidates."""
        self.args.num_mtp_tokens = 2
        self.args.num_mtp_token_sizes = [1, 2]

        task_runner = ParallelRunner(self.args)

        configs = list(task_runner._get_user_config(is_prefill=True))
        self.assertEqual({config.num_mtp_tokens for config in configs}, {0})

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

    def test_get_user_config_dcp_constrained_by_tp(self):
        """DCP search yields only (tp, dcp) pairs where dcp divides tp; it does not consume devices."""
        self.args.tp_sizes = [4]
        self.args.ep_sizes = [1]
        self.args.moe_dp_sizes = [1]
        self.args.dcp_sizes = [1, 2, 4, 8]
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())
        dcps = sorted(config.dcp_size for config in configs)
        # dcp=8 is pruned (8 does not divide tp=4); world_size is unchanged by dcp.
        self.assertEqual(dcps, [1, 2, 4])
        for config in configs:
            self.assertEqual(config.tp_size, 4)
            self.assertEqual(config.dp_size, 1)  # DCP reuses TP devices, dp = num_devices // tp

    def test_get_user_config_dcp_default_disabled(self):
        """Without --dcp-sizes, dcp stays 1 for every config."""
        self.args.tp_sizes = [1, 2, 4]
        self.args.dcp_sizes = None
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)
        configs = list(task_runner._get_user_config())
        self.assertTrue(configs)
        self.assertTrue(all(config.dcp_size == 1 for config in configs))

    def test_get_user_config_prefill_forces_dcp_one(self):
        """Prefill phase ignores the dcp search space (DCP is decode-only)."""
        self.args.tp_sizes = [4]
        self.args.dcp_sizes = [1, 2, 4]
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)
        prefill_configs = list(task_runner._get_user_config(is_prefill=True))
        self.assertTrue(prefill_configs)
        self.assertTrue(all(config.dcp_size == 1 for config in prefill_configs))
        # Decode (default) does search the dcp dimension.
        decode_configs = list(task_runner._get_user_config(is_prefill=False))
        self.assertEqual(sorted({c.dcp_size for c in decode_configs}), [1, 2, 4])

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
        self.assertEqual(row["tp_size"], 1)
        self.assertEqual(row["pp_size"], 1)
        self.assertEqual(row["dp_size"], 1)
        self.assertEqual(row["ep_size"], 1)
        self.assertEqual(row["moe_dp_size"], 1)
        self.assertEqual(row["pp_bubble_ratio"], 0.0)
        self.assertEqual(
            result[0].get_best_result_row()["parallel"],
            row["parallel"],
        )

    def test_run_pp2_agg_and_disagg_preserve_real_schedule_metrics(self):
        self.args.num_devices = 2
        self.args.tp_sizes = [1]
        self.args.ep_sizes = [1]
        self.args.pp_sizes = [2]
        self.args.num_hidden_layers_override = 2
        self.args.batch_range = [2, 2]
        self.args.ttft_limits = 1000
        self.args.tpot_limits = 50
        self.args.compile = False

        agg_result = ParallelRunner(self.args).run_agg()
        agg_row = agg_result[0].get_summary_df().iloc[0]
        self._assert_pp_schedule_schema(agg_row, expected_pp_size=2)

        self.args.disagg = True
        disagg_result = ParallelRunner(self.args).run_disagg()
        self.assertEqual(len(disagg_result), 2)
        prefill_summaries = [
            summary
            for summary in disagg_result
            if summary.data_config.ttft_limits is not None and summary.data_config.tpot_limits is None
        ]
        decode_summaries = [
            summary
            for summary in disagg_result
            if summary.data_config.ttft_limits is None and summary.data_config.tpot_limits is not None
        ]
        self.assertEqual(len(prefill_summaries), 1)
        self.assertEqual(len(decode_summaries), 1)
        for summary in [*prefill_summaries, *decode_summaries]:
            row = summary.get_summary_df().iloc[0]
            self._assert_pp_schedule_schema(row, expected_pp_size=2)

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

    def test_run_disagg_disables_mtp_for_prefill_only(self):
        """Standalone disaggregated Prefill must not model speculative decoding."""

        class RecordingParallelRunner(ParallelRunner):
            def __init__(self, args):
                super().__init__(args)
                self.recorded_mtp = []

            def _get_df_list(
                self,
                overwrite_optimizer_data,
                user_configs=None,
                disagg_mode=None,
                is_prefill=False,
            ):
                del disagg_mode
                effective_configs = (
                    list(user_configs)
                    if user_configs is not None
                    else list(self._get_user_config(is_prefill=is_prefill))
                )
                self.recorded_mtp.append(
                    (
                        overwrite_optimizer_data.num_mtp_tokens,
                        {config.num_mtp_tokens for config in effective_configs},
                    )
                )
                return []

        self.args.ttft_limits = 1000
        self.args.tpot_limits = 50
        self.args.num_mtp_tokens = 1
        task_runner = RecordingParallelRunner(self.args)

        task_runner.run_disagg()

        self.assertEqual(task_runner.recorded_mtp, [(0, {0}), (1, {1})])

    @pytest.mark.nightly
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
        self.assertEqual(row["model_id"], vendored_model_config_path(self.args.model_id))
        self.assertEqual(row["parallel"], "TP=1 | PP=1 | DP=1")

    def test_submit_task_skips_candidate_on_unsupported_pp_configuration(self):
        # Finding 1: _submit_task must catch ONLY UnsupportedPPConfigurationError
        # (skip the candidate) and let unrelated ValueError propagate (fail loud).
        user_config = UserInputConfig.from_args(self.args)
        optimizer_data = OptimizerData(
            input_length=self.args.input_length,
            output_length=self.args.output_length,
            max_batched_tokens=self.args.max_batched_tokens,
            num_devices=self.args.num_devices,
        )
        task_runner = ParallelRunner(self.args)

        class _RaisingStrategy:
            def __init__(self, exc):
                self._exc = exc

            def run(self, *args, **kwargs):
                raise self._exc

        with patch(
            "serving_cast.parallel_runner.OptimizerFactory.create_strategy",
            return_value=_RaisingStrategy(UnsupportedPPConfigurationError("unsupported PP config")),
        ):
            result = task_runner._submit_task(user_config, optimizer_data)
        # UnsupportedPPConfigurationError -> candidate skipped (None).
        self.assertIsNone(result)

    def test_submit_task_propagates_unrelated_value_error(self):
        # A generic ValueError (e.g. profile missing, stage inconsistency) must
        # NOT be swallowed by the PP-config catch; it should propagate.
        user_config = UserInputConfig.from_args(self.args)
        optimizer_data = OptimizerData(
            input_length=self.args.input_length,
            output_length=self.args.output_length,
            max_batched_tokens=self.args.max_batched_tokens,
            num_devices=self.args.num_devices,
        )
        task_runner = ParallelRunner(self.args)

        class _RaisingStrategy:
            def run(self, *args, **kwargs):
                raise ValueError("unrelated error: profile missing")

        with patch(
            "serving_cast.parallel_runner.OptimizerFactory.create_strategy",
            return_value=_RaisingStrategy(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                task_runner._submit_task(user_config, optimizer_data)
        self.assertIn("profile missing", str(ctx.exception))


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

    def _optimizer_data(self):
        return OptimizerData(
            input_length=1024,
            output_length=1024,
            ttft_limits=100,
            tpot_limits=10,
        )

    @staticmethod
    def _summary(optimizer_data, rows, memory_info=None):
        summary = OptimizerSummary(optimizer_data)
        summary.set_summary_df(pd.DataFrame(rows))
        if memory_info is not None:
            summary.set_memory_info(memory_info)
        return summary

    def test_add_summary_result_with_empty_list(self):
        """Test _add_summary_result with empty df_list."""
        task_runner = ParallelRunner(self.args)
        optimizer_data = self._optimizer_data()

        task_runner._add_summary_result([], optimizer_data)
        self.assertEqual(len(task_runner.summary_result), 0)

    def test_add_summary_result_with_valid_df(self):
        """Test _add_summary_result with valid DataFrame."""
        task_runner = ParallelRunner(self.args)
        optimizer_data = self._optimizer_data()
        summary = self._summary(
            optimizer_data,
            {
                "ttft": [100.0],
                "tpot": [10.0],
                "concurrency": [10],
                "parallel": ["tp4pp1dp1"],
                "batch_size": [4],
            },
        )

        task_runner._add_summary_result([summary], optimizer_data)
        self.assertEqual(len(task_runner.summary_result), 1)

    def test_add_summary_result_selects_tightest_memory_info(self):
        """Merged multi-TP summaries should keep the most constrained memory info."""
        task_runner = ParallelRunner(self.args)
        optimizer_data = self._optimizer_data()
        rows = {"ttft": [100.0], "tpot": [10.0], "token/s": [1.0]}
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

        first = self._summary(optimizer_data, rows)
        second = self._summary(optimizer_data, rows, loose_memory_info)
        third = self._summary(optimizer_data, rows, tight_memory_info)

        task_runner._add_summary_result([first, second, third], optimizer_data)

        self.assertEqual(task_runner.summary_result[0].get_memory_info(), tight_memory_info)

    def test_run_pd_ratio_combines_prefill_and_decode_results(self):
        """_run_pd_ratio should submit both phases and wrap the optimized result."""

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

        def fake_run_pd_phase(devices_per_instance, is_prefill, process_context=None):
            phase_calls.append((devices_per_instance, is_prefill, process_context))
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
            patch(
                "serving_cast.parallel_runner.PDRatioThroughputOptimizer",
                RecordingPDRatioOptimizer,
            ),
        ):
            result = task_runner._run_pd_ratio()

        self.assertEqual(
            [(devices_per_instance, is_prefill) for devices_per_instance, is_prefill, _ in phase_calls],
            [
                (self.args.prefill_devices_per_instance, True),
                (self.args.decode_devices_per_instance, False),
            ],
        )
        self.assertEqual(phase_calls[0][2].get_start_method(), "spawn")
        self.assertIs(phase_calls[0][2], phase_calls[1][2])
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

    def test_pd_phase_disables_mtp_for_prefill_only(self):
        """PD ratio uses the requested MTP depth only for Decode."""

        class RecordingParallelRunner(ParallelRunner):
            def __init__(self, args):
                super().__init__(args)
                self.recorded_optimizer_mtp = None
                self.recorded_user_mtp = []

            def _get_df_list(
                self,
                overwrite_optimizer_data,
                user_configs=None,
                disagg_mode=None,
                is_prefill=False,
                process_context=None,
            ):
                del disagg_mode, is_prefill, process_context
                self.recorded_optimizer_mtp = overwrite_optimizer_data.num_mtp_tokens
                self.recorded_user_mtp = [config.num_mtp_tokens for config in user_configs]
                return []

        self.args.num_mtp_tokens = 1
        prefill_runner = RecordingParallelRunner(self.args)
        prefill_runner._run_pd_phase(
            devices_per_instance=self.args.prefill_devices_per_instance,
            is_prefill=True,
        )
        self.assertEqual(prefill_runner.recorded_optimizer_mtp, 0)
        self.assertTrue(prefill_runner.recorded_user_mtp)
        self.assertEqual(set(prefill_runner.recorded_user_mtp), {0})

        decode_runner = RecordingParallelRunner(self.args)
        decode_runner._run_pd_phase(
            devices_per_instance=self.args.decode_devices_per_instance,
            is_prefill=False,
        )
        self.assertEqual(decode_runner.recorded_optimizer_mtp, 1)
        self.assertTrue(decode_runner.recorded_user_mtp)
        self.assertEqual(set(decode_runner.recorded_user_mtp), {1})

    def test_pd_prefill_without_ttft_limit_uses_unbounded_prefill(self):
        """An omitted TTFT SLO must not turn the PD Prefill phase into Decode."""

        class RecordingParallelRunner(ParallelRunner):
            def __init__(self, args):
                super().__init__(args)
                self.recorded_optimizer_data = None

            def _get_df_list(
                self,
                overwrite_optimizer_data,
                user_configs=None,
                disagg_mode=None,
                is_prefill=False,
                process_context=None,
            ):
                del user_configs, disagg_mode, is_prefill, process_context
                self.recorded_optimizer_data = overwrite_optimizer_data
                return []

        self.args.ttft_limits = None
        task_runner = RecordingParallelRunner(self.args)
        task_runner._run_pd_phase(
            devices_per_instance=self.args.prefill_devices_per_instance,
            is_prefill=True,
        )

        self.assertEqual(task_runner.recorded_optimizer_data.ttft_limits, float("inf"))
        self.assertIsNone(task_runner.recorded_optimizer_data.tpot_limits)

    def test_get_df_list_passes_pd_process_context_to_process_pool(self):
        """PD ratio sub-phases must construct process pools with the spawn context."""
        import multiprocessing as mp

        class ProcessContextRecordingExecutor(ProcessPoolExecutor):
            init_kwargs = None

            def __init__(self, **kwargs):
                type(self).init_kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def map(self, fn, *iterables, timeout=None, chunksize=1):
                return []

        task_runner = ParallelRunner(self.args, executor_class=ProcessContextRecordingExecutor)
        process_context = mp.get_context("spawn")

        result = task_runner._get_df_list(
            task_runner.optimizer_data,
            user_configs=[],
            process_context=process_context,
        )

        self.assertEqual(result, [])
        self.assertIs(ProcessContextRecordingExecutor.init_kwargs["mp_context"], process_context)

    def test_get_df_list_uses_spawn_for_process_pool_without_pd_context(self):
        """Default ProcessPoolExecutor must spawn, not inherit Linux fork."""
        import multiprocessing as mp

        class ProcessContextRecordingExecutor(ProcessPoolExecutor):
            init_kwargs = None

            def __init__(self, **kwargs):
                type(self).init_kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def map(self, fn, *iterables, timeout=None, chunksize=1):
                return []

        task_runner = ParallelRunner(self.args, executor_class=ProcessContextRecordingExecutor)
        result = task_runner._get_df_list(task_runner.optimizer_data, user_configs=[])

        self.assertEqual(result, [])
        mp_context = ProcessContextRecordingExecutor.init_kwargs["mp_context"]
        self.assertEqual(mp_context.get_start_method(), mp.get_context("spawn").get_start_method())

    def test_get_df_list_ignores_pd_process_context_for_injected_executor(self):
        """Injected non-process executors must not receive unsupported mp_context."""
        import multiprocessing as mp

        class ContextRecordingExecutor:
            init_kwargs = None

            def __init__(self, **kwargs):
                type(self).init_kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def map(self, fn, *iterables, timeout=None, chunksize=1):
                return []

        task_runner = ParallelRunner(self.args, executor_class=ContextRecordingExecutor)
        result = task_runner._get_df_list(
            task_runner.optimizer_data,
            user_configs=[],
            process_context=mp.get_context("spawn"),
        )

        self.assertEqual(result, [])
        self.assertNotIn("mp_context", ContextRecordingExecutor.init_kwargs)


if __name__ == "__main__":
    unittest.main()
