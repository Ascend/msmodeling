"""Regression tests for PSOOptimizer.prepare_plugin and run_plugin."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from optix.config.config import OptimizerConfigField, PerformanceIndex, map_param_with_value
from optix.optimizer.errors import BaselineRunError, NoFeasibleSolutionError
from tests.regression.optix.test_optimizer.test_pso_optimizer import _make_pso_optimizer


class _FakeSimulator:
    data_field = ()


class _FakeAisBench:
    data_field = ()


def _simulator_branch_optimizer(*, error_info=None, default_perf=None):
    scheduler = MagicMock()
    scheduler.error_info = error_info
    scheduler.simulator = MagicMock()
    scheduler.simulator.__class__ = _FakeSimulator
    scheduler.simulator.data_field = (
        OptimizerConfigField(
            name="max_batch_size",
            min=10,
            max=100,
            dtype="int",
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            value=50,
        ),
    )
    scheduler.simulator.default_config = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 50}}}
    scheduler.simulator.config.config_path = "/tmp/mindie/config.json"
    scheduler.benchmark = MagicMock()
    scheduler.benchmark.__class__ = _FakeAisBench
    scheduler.benchmark.data_field = (
        OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int", config_position="env"),
    )
    scheduler.benchmark.get_best_concurrency.return_value = 32
    perf = default_perf or PerformanceIndex(
        generate_speed=100.0,
        time_to_first_token=0.2,
        time_per_output_token=0.02,
        success_rate=1.0,
    )
    scheduler.run.return_value = perf
    scheduler.data_storage = MagicMock()
    scheduler.data_storage.get_run_info.return_value = {}
    scheduler.data_storage.get_best_result.return_value = pd.DataFrame()
    return _make_pso_optimizer(scheduler=scheduler, fine_tune=MagicMock(), pso_init_kwargs={})


class TestPreparePlugin(unittest.TestCase):
    @patch("optix.optimizer.optimizer.is_mindie", return_value=False)
    @patch("optix.optimizer.plugins.simulate.Simulator", _FakeSimulator)
    @patch("optix.optimizer.plugins.benchmark.AisBench", _FakeAisBench)
    @patch("optix.optimizer.optimizer.get_required_field_from_json", return_value=50)
    def test_prepare_plugin_success_sets_defaults(self, mock_get_field, mock_is_mindie):
        opt = _simulator_branch_optimizer()
        opt.prepare_plugin()
        self.assertIsNotNone(opt.default_fitness)
        self.assertEqual(opt.default_res.generate_speed, 100.0)
        scheduler = opt.scheduler
        scheduler.run.assert_called_once()
        scheduler.save_result.assert_called_once()

    @patch("optix.optimizer.optimizer.is_mindie", return_value=False)
    @patch("optix.optimizer.plugins.simulate.Simulator", _FakeSimulator)
    @patch("optix.optimizer.plugins.benchmark.AisBench", _FakeAisBench)
    @patch("optix.optimizer.optimizer.get_required_field_from_json", return_value=50)
    def test_prepare_plugin_raises_when_error_info_set(self, mock_get_field, mock_is_mindie):
        opt = _simulator_branch_optimizer(error_info=subprocess.SubprocessError("subprocess failed"))
        scheduler = opt.scheduler
        scheduler.simulator.command = ["vllm", "serve", "model_path"]
        scheduler.simulator.run_log = "/tmp/ms_serviceparam_optimizer__nvppydc"
        scheduler.simulator.process.returncode = 1
        with self.assertRaises(BaselineRunError) as ctx:
            opt.prepare_plugin()
        message = str(ctx.exception)
        self.assertIn("exit=1", message)
        self.assertIn("  log: /tmp/ms_serviceparam_optimizer__nvppydc", message)
        self.assertNotIn("Failed in run simulator", message)
        self.assertNotIn("Failed to start the default service", message)
        scheduler.save_result.assert_not_called()
        scheduler.stop_target_server.assert_called_once()

    @patch("optix.optimizer.optimizer.is_mindie", return_value=False)
    @patch("optix.optimizer.plugins.simulate.Simulator", _FakeSimulator)
    @patch("optix.optimizer.plugins.benchmark.AisBench", _FakeAisBench)
    @patch("optix.optimizer.optimizer.get_required_field_from_json", return_value=50)
    def test_prepare_plugin_keeps_baseline_only_enum_value_out_of_search_space(self, mock_get_field, mock_is_mindie):
        opt = _simulator_branch_optimizer()
        scheduler = opt.scheduler
        enum_field = OptimizerConfigField(
            name="ASCEND_RT_VISIBLE_DEVICES",
            config_position="env",
            dtype="enum",
            dtype_param=[1],
            value=2,
        )
        opt.target_field = (enum_field,)
        scheduler.simulator.data_field = (enum_field,)
        scheduler.benchmark.data_field = ()
        captured = {}

        def baseline_run(params, params_field):
            captured["params"] = params
            captured["params_field"] = params_field
            scheduler.simulator.data_field = params_field
            return PerformanceIndex(
                generate_speed=100.0,
                time_to_first_token=0.2,
                time_per_output_token=0.02,
                success_rate=1.0,
            )

        scheduler.run.side_effect = baseline_run

        opt.prepare_plugin()

        assert captured["params_field"][0].value == 2
        assert captured["params_field"][0].dtype_param == [1, 2]
        assert opt.target_field[0].dtype_param == [1]
        assert scheduler.simulator.data_field[0].dtype_param == [1]
        assert map_param_with_value(np.array([0.0]), tuple(opt.target_field))[0].value == 1


class TestRunPlugin(unittest.TestCase):
    @patch("optix.optimizer.optimizer.enable_simulate")
    @patch("optix.optimizer.optimizer.adapter_target_field")
    @patch("optix.optimizer.global_best_custom.CustomGlobalBestPSO")
    def test_run_plugin_loads_breakpoint_history(
        self,
        mock_pso_cls,
        mock_adapter,
        mock_enable_sim,
    ):
        opt = _simulator_branch_optimizer()
        opt.load_breakpoint = True
        opt.load_history_data = [{"fitness": 1.0, "f1": 10, "f2": 100}]
        opt.prepare_plugin = MagicMock()
        opt.computer_fitness = MagicMock(return_value=([np.array([1.0])], [1.0]))
        opt.refine_optimization_candidates = MagicMock(return_value=([1.0], [np.array([1.0])], [MagicMock()]))
        best_perf = PerformanceIndex(
            generate_speed=100.0,
            time_to_first_token=0.2,
            time_per_output_token=0.02,
            success_rate=1.0,
        )
        opt.best_params = MagicMock(return_value=(1.0, np.array([50.0, 25000.0]), best_perf))

        mock_adapter.return_value.__enter__ = MagicMock(return_value=None)
        mock_adapter.return_value.__exit__ = MagicMock(return_value=False)
        mock_enable_sim.return_value.__enter__ = MagicMock(return_value=False)
        mock_enable_sim.return_value.__exit__ = MagicMock(return_value=False)
        mock_pso_cls.return_value.optimize.return_value = (np.array([1.0]), np.array([[1.0]]))

        opt.run_plugin()
        opt.scheduler.data_storage.load_history_position.assert_called_once()
        mock_pso_cls.assert_called_once()
        _, kwargs = mock_pso_cls.call_args
        assert kwargs["breakpoint_pos"] == [np.array([1.0])]
        assert kwargs["breakpoint_cost"] == [1.0]

    @patch("optix.optimizer.optimizer.enable_simulate")
    @patch("optix.optimizer.optimizer.adapter_target_field")
    @patch("optix.optimizer.global_best_custom.CustomGlobalBestPSO")
    def test_run_plugin_stops_when_optimizer_has_no_feasible_solution(
        self,
        mock_pso_cls,
        mock_adapter,
        mock_enable_sim,
    ):
        opt = _simulator_branch_optimizer()
        opt.prepare_plugin = MagicMock()
        opt.refine_optimization_candidates = MagicMock()

        for cm in (mock_adapter, mock_enable_sim):
            cm.return_value.__enter__ = MagicMock(return_value=None)
            cm.return_value.__exit__ = MagicMock(return_value=False)

        no_feasible = NoFeasibleSolutionError("No feasible solution found after 2 optimization rounds")
        mock_pso_cls.return_value.optimize.side_effect = no_feasible

        with self.assertRaisesRegex(NoFeasibleSolutionError, "No feasible solution found"):
            opt.run_plugin()
        mock_pso_cls.return_value.optimize.assert_called_once_with(opt.op_func, iters=opt.iters)
        opt.refine_optimization_candidates.assert_not_called()

    @patch("optix.optimizer.optimizer.enable_simulate")
    @patch("optix.optimizer.optimizer.adapter_target_field")
    @patch("optix.optimizer.global_best_custom.CustomGlobalBestPSO")
    def test_run_plugin_raises_when_no_best_params(
        self,
        mock_pso_cls,
        mock_adapter,
        mock_enable_sim,
    ):
        opt = _simulator_branch_optimizer()
        opt.prepare_plugin = MagicMock()
        opt.refine_optimization_candidates = MagicMock(return_value=([], [], []))
        opt.best_params = MagicMock(return_value=(None, None, None))

        for cm in (mock_adapter, mock_enable_sim):
            cm.return_value.__enter__ = MagicMock(return_value=None)
            cm.return_value.__exit__ = MagicMock(return_value=False)
        mock_enable_sim.return_value.__enter__ = MagicMock(return_value=False)
        mock_pso_cls.return_value.optimize.return_value = (np.array([1.0]), np.array([[1.0]]))

        with self.assertRaises(NoFeasibleSolutionError):
            opt.run_plugin()
