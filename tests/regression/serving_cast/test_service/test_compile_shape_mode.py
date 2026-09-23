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

import copy
from types import SimpleNamespace
import pickle
import unittest
from unittest.mock import ANY, Mock, patch

from serving_cast.parallel_runner import ParallelRunner
from serving_cast.service.compile_shape_mode import (
    COMPILE_MODE_RATIO_THRESHOLD,
    CompileDecisionKey,
    CompileModeDecision,
    CompileModeDecisionCache,
    decide_compile_shape_mode,
)
from serving_cast.service.workload_cache import WorkloadCache
from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData
from tensor_cast.core.user_config import UserInputConfig

from .test_common import SimpleArgs


class TestCompileShapeMode(unittest.TestCase):
    def setUp(self):
        self.user_input = UserInputConfig(
            model_id="Qwen/Qwen3-8B",
            device="TEST_DEVICE",
            do_compile=True,
            dp_size=2,
            pp_size=1,
            tp_size=1,
        )
        self.optimizer_data = OptimizerData(
            input_length=128,
            output_length=64,
            max_batched_tokens=1024,
            prefix_cache_hit_rate=0.0,
        )

    def test_ratio_threshold_keeps_dynamic_at_boundary(self):
        decision = decide_compile_shape_mode(2.0, 2.0 * COMPILE_MODE_RATIO_THRESHOLD)

        self.assertTrue(decision.dynamic_shapes)
        self.assertEqual(decision.reason, "dynamic_static_ratio_within_threshold")
        self.assertEqual(decision.ratio, COMPILE_MODE_RATIO_THRESHOLD)

    def test_ratio_above_threshold_selects_static(self):
        decision = decide_compile_shape_mode(2.0, 3.01)

        self.assertFalse(decision.dynamic_shapes)
        self.assertEqual(decision.reason, "dynamic_static_ratio_exceeds_threshold")

    def test_invalid_dynamic_probe_time_falls_back_to_dynamic(self):
        for dynamic_run_time_s in (0.0, float("nan")):
            with self.subTest(dynamic_run_time_s=dynamic_run_time_s):
                decision = decide_compile_shape_mode(2.0, dynamic_run_time_s)

                self.assertTrue(decision.dynamic_shapes)
                self.assertEqual(decision.reason, "invalid_dynamic_probe_time")
                self.assertIsNone(decision.ratio)

    def test_decision_key_changes_for_graph_affecting_fields_not_slo(self):
        base = CompileDecisionKey.from_inputs(
            self.user_input,
            self.optimizer_data,
            phase="decode",
            probe_batch_size=32,
            is_decode=True,
        )
        changed_slo = OptimizerData(
            input_length=128,
            output_length=64,
            max_batched_tokens=1024,
            tpot_limits=1.0,
            ttft_limits=2.0,
            prefix_cache_hit_rate=0.0,
        )
        changed_parallel = UserInputConfig(**{**self.user_input.__dict__, "tp_size": 2})
        changed_device = copy.copy(self.user_input)
        changed_device.device = "OTHER_DEVICE"

        self.assertEqual(
            base,
            CompileDecisionKey.from_inputs(
                self.user_input,
                changed_slo,
                phase="decode",
                probe_batch_size=32,
                is_decode=True,
            ),
        )
        self.assertNotEqual(
            base,
            CompileDecisionKey.from_inputs(
                changed_parallel,
                self.optimizer_data,
                phase="decode",
                probe_batch_size=32,
                is_decode=True,
            ),
        )
        self.assertEqual(
            base,
            CompileDecisionKey.from_inputs(
                changed_device,
                self.optimizer_data,
                phase="decode",
                probe_batch_size=32,
                is_decode=True,
            ),
        )

    def test_decision_cache_keeps_only_scalar_decision(self):
        cache = CompileModeDecisionCache()
        key = CompileDecisionKey.from_inputs(
            self.user_input,
            self.optimizer_data,
            phase="decode",
            probe_batch_size=32,
            is_decode=True,
        )
        decision = decide_compile_shape_mode(1.0, 2.0)

        cache.set(key, decision)

        self.assertIs(cache.get(key), decision)
        self.assertFalse(hasattr(cache.get(key), "model_runner"))

    def test_decision_cache_is_pickleable_for_process_pool_workers(self):
        cache = CompileModeDecisionCache()
        key = CompileDecisionKey.from_inputs(
            self.user_input,
            self.optimizer_data,
            phase="decode",
            probe_batch_size=32,
            is_decode=True,
        )
        decision = decide_compile_shape_mode(1.0, 2.0)
        cache.set(key, decision)

        restored_cache = pickle.loads(pickle.dumps(cache))

        self.assertEqual(restored_cache.get(key), decision)


class TestParallelRunnerCompileShapeMode(unittest.TestCase):
    def setUp(self):
        self.args = SimpleArgs()
        self.args.compile = True
        self.args.batch_range = [2, 2]
        self.args.jobs = 1
        self.runner = ParallelRunner(self.args)
        self.user_input = UserInputConfig.from_args(self.args)
        self.optimizer_data = OptimizerData(
            input_length=self.args.input_length,
            output_length=self.args.output_length,
            max_batched_tokens=self.args.max_batched_tokens,
            num_devices=self.args.num_devices,
            num_mtp_tokens=self.user_input.num_mtp_tokens,
            mtp_acceptance_rate=self.args.mtp_acceptance_rate,
        )

    @patch("serving_cast.parallel_runner.torch.compiler.reset")
    def test_calibration_rebuilds_selected_static_runner_and_caches_probe(self, _reset):
        static_probe_runner = Mock()
        dynamic_runner = Mock()
        selected_static_runner = Mock()
        static_metrics = SimpleNamespace(run_time_s=2.0)
        dynamic_metrics = SimpleNamespace(run_time_s=3.1)
        static_probe_runner.run_inference.return_value = static_metrics
        dynamic_runner.run_inference.return_value = dynamic_metrics
        static_probe_strategy = Mock()
        dynamic_strategy = Mock()
        selected_static_strategy = Mock()
        probe_key = SimpleNamespace(model_concurrency=2)
        request = object()
        static_probe_strategy.get_compile_calibration_probe.return_value = (probe_key, request)
        dynamic_strategy.get_compile_calibration_probe.return_value = (probe_key, request)

        with (
            patch.object(
                self.runner,
                "_build_model_runner",
                side_effect=[static_probe_runner, dynamic_runner, selected_static_runner],
            ),
            patch.object(
                self.runner,
                "_create_strategy",
                side_effect=[static_probe_strategy, dynamic_strategy, selected_static_strategy],
            ),
        ):
            selected_runner, selected_strategy = self.runner._resolve_compile_shape_mode(
                self.user_input,
                self.optimizer_data,
                disagg_mode=True,
            )

        self.assertIs(selected_runner, selected_static_runner)
        self.assertIs(selected_strategy, selected_static_strategy)
        static_probe_runner.run_inference.assert_called_once_with([request], generate_inputs_func=ANY)
        dynamic_runner.run_inference.assert_called_once_with([request], generate_inputs_func=ANY)
        selected_static_runner.run_inference.assert_not_called()
        selected_static_strategy.cache_compile_calibration_metrics.assert_called_once_with(probe_key, static_metrics)
        static_probe_strategy.cache_compile_calibration_metrics.assert_not_called()
        dynamic_strategy.cache_compile_calibration_metrics.assert_not_called()

    @patch("serving_cast.parallel_runner.torch.compiler.reset")
    def test_aggregation_calibrates_with_decode_request(self, _reset):
        static_runner = Mock()
        dynamic_runner = Mock()
        selected_runner = Mock()
        static_runner.run_inference.return_value = SimpleNamespace(run_time_s=2.0)
        dynamic_runner.run_inference.return_value = SimpleNamespace(run_time_s=2.1)
        static_strategy = Mock()
        dynamic_strategy = Mock()
        selected_strategy = Mock()
        probe_key = SimpleNamespace(model_concurrency=2)
        static_strategy.get_compile_calibration_probe.return_value = (probe_key, object())
        dynamic_strategy.get_compile_calibration_probe.return_value = (probe_key, object())

        with (
            patch.object(
                self.runner,
                "_build_model_runner",
                side_effect=[static_runner, dynamic_runner, selected_runner],
            ),
            patch.object(
                self.runner,
                "_create_strategy",
                side_effect=[static_strategy, dynamic_strategy, selected_strategy],
            ),
        ):
            self.runner._resolve_compile_shape_mode(
                self.user_input,
                self.optimizer_data,
                disagg_mode=False,
            )

        for strategy in (static_strategy, dynamic_strategy):
            self.assertTrue(strategy.get_compile_calibration_probe.call_args.kwargs["is_decode"])

    @patch("serving_cast.parallel_runner.torch.compiler.reset")
    def test_repeated_candidate_uses_in_process_decision_cache(self, _reset):
        static_runner = Mock()
        dynamic_runner = Mock()
        selected_probe_runner = Mock()
        cached_runner = Mock()
        static_runner.run_inference.return_value = SimpleNamespace(run_time_s=2.0)
        dynamic_runner.run_inference.return_value = SimpleNamespace(run_time_s=2.1)
        static_strategy = Mock()
        dynamic_strategy = Mock()
        selected_probe_strategy = Mock()
        cached_strategy = Mock()
        probe_key = SimpleNamespace(model_concurrency=2)
        static_strategy.get_compile_calibration_probe.return_value = (probe_key, object())
        dynamic_strategy.get_compile_calibration_probe.return_value = (probe_key, object())

        with (
            patch.object(
                self.runner,
                "_build_model_runner",
                side_effect=[static_runner, dynamic_runner, selected_probe_runner, cached_runner],
            ) as build_runner,
            patch.object(
                self.runner,
                "_create_strategy",
                side_effect=[static_strategy, dynamic_strategy, selected_probe_strategy, cached_strategy],
            ),
        ):
            self.runner._resolve_compile_shape_mode(
                self.user_input,
                self.optimizer_data,
                disagg_mode=True,
            )
            selected_runner, selected_strategy = self.runner._resolve_compile_shape_mode(
                self.user_input,
                self.optimizer_data,
                disagg_mode=True,
            )

        self.assertIs(selected_runner, cached_runner)
        self.assertIs(selected_strategy, cached_strategy)
        self.assertEqual(build_runner.call_count, 4)
        cached_runner.run_inference.assert_not_called()

    @patch("serving_cast.parallel_runner.torch.compiler.reset")
    def test_sequence_parallel_uses_static_without_probe(self, _reset):
        self.user_input.enable_sequence_parallel = True
        selected_runner = Mock()
        selected_strategy = Mock()

        with (
            patch.object(self.runner, "_build_model_runner", return_value=selected_runner) as build_runner,
            patch.object(self.runner, "_create_strategy", return_value=selected_strategy),
        ):
            resolved_runner, resolved_strategy = self.runner._resolve_compile_shape_mode(
                self.user_input,
                self.optimizer_data,
                disagg_mode=True,
            )

        self.assertIs(resolved_runner, selected_runner)
        self.assertIs(resolved_strategy, selected_strategy)
        self.assertFalse(build_runner.call_args.args[0].dynamic_shapes)

    @patch("serving_cast.parallel_runner.torch.compiler.reset")
    def test_variable_length_aggregation_falls_back_to_dynamic_without_crashing(self, _reset):
        optimizer_data = OptimizerData(
            input_length=None,
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=1000, weight=1.0)]),
            output_length=64,
        )
        selected_runner = Mock()
        selected_strategy = Mock()

        with (
            patch.object(self.runner, "_build_model_runner", return_value=selected_runner) as build_runner,
            patch.object(self.runner, "_create_strategy", return_value=selected_strategy),
        ):
            resolved_runner, resolved_strategy = self.runner._resolve_compile_shape_mode(
                self.user_input,
                optimizer_data,
                disagg_mode=False,
            )

        self.assertIs(resolved_runner, selected_runner)
        self.assertIs(resolved_strategy, selected_strategy)
        self.assertTrue(build_runner.call_args.args[0].dynamic_shapes)

    @patch("serving_cast.parallel_runner.torch.compiler.reset")
    def test_multi_device_runner_reuses_shared_compile_decision_without_probe(self, _reset):
        shared_cache = WorkloadCache()
        decision_key = CompileDecisionKey.from_inputs(
            self.user_input,
            self.optimizer_data,
            phase="decode",
            probe_batch_size=2,
            is_decode=True,
        )
        state, _, owner_token = shared_cache.claim_compile_mode_decision(decision_key.digest)
        self.assertEqual(state, "owner")
        shared_cache.publish_compile_mode_decision(
            decision_key.digest, CompileModeDecision(dynamic_shapes=False, reason="calibrated"), owner_token
        )

        runner = ParallelRunner(self.args, workload_cache=shared_cache)
        selected_runner = Mock()
        selected_strategy = Mock()

        with (
            patch.object(runner, "_build_model_runner", return_value=selected_runner) as build_runner,
            patch.object(runner, "_create_strategy", return_value=selected_strategy),
        ):
            runner._resolve_compile_shape_mode(
                self.user_input,
                self.optimizer_data,
                disagg_mode=True,
            )

        self.assertEqual(build_runner.call_count, 1)
        self.assertFalse(build_runner.call_args.args[0].dynamic_shapes)


if __name__ == "__main__":
    unittest.main()
