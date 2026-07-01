# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
from math import inf
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optix.config.config import (
    DecodeContext,
    OptimizerConfigField,
    PerformanceIndex,
)
from optix.optimizer.optimizer import (
    PSOOptimizer,
    adapter_target_field,
    enable_simulate,
    sample,
)


def _make_pso_optimizer(**overrides):
    """Module-level factory for PSOOptimizer test instances.

    Creates a PSOOptimizer backed by mocked scheduler/data_storage.  All
    keyword arguments are forwarded to the PSOOptimizer constructor,
    overriding the sensible defaults below.

    Returns the fully constructed (but not yet run) PSOOptimizer instance.
    """
    scheduler = MagicMock()
    scheduler.error_info = None
    scheduler.data_storage = MagicMock()
    defaults = {
        "scheduler": scheduler,
        "n_particles": 3,
        "iters": 5,
        "target_field": (
            OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int"),
            OptimizerConfigField(name="max_prefill_token", min=1000, max=50000, dtype="int"),
        ),
        "ttft_penalty": 0,
        "tpot_penalty": 0,
        "success_rate_penalty": 0,
        "ttft_slo": 1.0,
        "tpot_slo": 0.1,
        "success_rate_slo": 0.9,
        "generate_speed_target": 100,
    }
    defaults.update(overrides)
    return PSOOptimizer(**defaults)


def _make_fine_tune_mock(**attrs):
    """Create a strict fine_tune mock that only exposes the specified attributes.

    Uses ``spec_set`` so that accessing any attribute not listed in *attrs*
    raises ``AttributeError``, catching logic regressions in ``best_params``
    that might silently pass with a plain ``MagicMock()``.
    """
    return MagicMock(spec_set=list(attrs.keys()), **attrs)


class TestPSOOptimizer:
    def _create_optimizer(self, **kwargs):
        overrides = {"n_particles": 5, "iters": 10}
        overrides.update(kwargs)
        return _make_pso_optimizer(**overrides)

    def test_init_basic(self):
        opt = self._create_optimizer()
        assert opt.n_particles == 5
        assert opt.iters == 10
        assert opt._iteration == 0

    def test_init_caps_at_max_iter_num(self):
        opt = self._create_optimizer(n_particles=500, iters=500)
        assert opt.n_particles == 200
        assert opt.iters == 200

    def test_is_within_boundary_true(self):
        assert PSOOptimizer.is_within_boundary([5, 50], (0, 0), (10, 100))

    def test_is_within_boundary_false(self):
        assert not PSOOptimizer.is_within_boundary([15, 50], (0, 0), (10, 100))

    def test_params_in_records_found(self):
        records = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        assert PSOOptimizer.params_in_records(np.array([1.0, 2.0]), records)

    def test_params_in_records_not_found(self):
        records = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        assert not PSOOptimizer.params_in_records(np.array([5.0, 6.0]), records)

    def test_constructing_bounds(self):
        opt = self._create_optimizer()
        min_b, max_b = opt.constructing_bounds()
        assert min_b == (10, 1000)
        assert max_b == (100, 50000)

    def test_constructing_bounds_excludes_constants(self):
        target_field = (
            OptimizerConfigField(name="f1", min=10, max=100, dtype="int"),
            OptimizerConfigField(name="f2", min=5, max=5, dtype="int", constant=5),
        )
        opt = self._create_optimizer(target_field=target_field)
        min_b, max_b = opt.constructing_bounds()
        assert min_b == (10,)
        assert max_b == (100,)

    def test_dimensions(self):
        opt = self._create_optimizer()
        assert opt.dimensions() == 2

    def test_dimensions_excludes_constants(self):
        target_field = (
            OptimizerConfigField(name="f1", min=10, max=100, dtype="int"),
            OptimizerConfigField(name="f2", min=5, max=5, dtype="int", constant=5),
        )
        opt = self._create_optimizer(target_field=target_field)
        assert opt.dimensions() == 1

    def test_get_max_generate_speed_index(self):
        opt = self._create_optimizer()
        perf_list = [
            MagicMock(generate_speed=10),
            MagicMock(generate_speed=50),
            MagicMock(generate_speed=30),
        ]
        slo_index = [0, 1, 2]
        assert opt.get_max_generate_speed_index(perf_list, slo_index) == 1

    def test_get_max_generate_speed_index_filtered(self):
        opt = self._create_optimizer()
        perf_list = [
            MagicMock(generate_speed=10),
            MagicMock(generate_speed=50),
            MagicMock(generate_speed=30),
        ]
        slo_index = [0, 2]
        assert opt.get_max_generate_speed_index(perf_list, slo_index) == 2

    def test_best_params_empty_input(self):
        opt = self._create_optimizer()
        f, p, pi = opt.best_params([], [], [])
        assert f is None
        assert p is None
        assert pi is None

    def test_best_params_no_penalty(self):
        opt = self._create_optimizer(ttft_penalty=0, tpot_penalty=0)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.01),
            PerformanceIndex(generate_speed=50, time_to_first_token=0.2, time_per_output_token=0.02),
        ]
        fitness_list = [1.0, 0.5]
        params_list = [np.array([10]), np.array([20])]
        f, p, pi = opt.best_params(fitness_list, params_list, perf_list)
        assert pi.generate_speed == 50

    def test_best_params_with_tpot_penalty(self):
        # When only tpot_penalty is active (ttft_penalty=0), best_params uses a
        # two-stage selection algorithm based on *relative* differences from the
        # TPOT upper bound:
        #
        #   relative_diff = (actual_tpot - threshold) / threshold
        #
        # Stage 1: If any candidates are within the threshold (relative_diff < 0),
        #   pick the one with the highest generate_speed among them.
        # Stage 2: If ALL candidates exceed the threshold, pick the one with the
        #   smallest relative_diff (i.e., closest to the threshold), NOT the
        #   smallest absolute difference. This ensures fair comparison when
        #   thresholds differ in magnitude across scenarios.
        #
        # Test data:
        #   Entry 0: tpot=0.04 → relative_diff = (0.04-0.05)/0.05 = -0.20 (within threshold)
        #   Entry 1: tpot=0.06 → relative_diff = (0.06-0.05)/0.05 = +0.20 (exceeds threshold)
        # Entry 0 is selected because it is the only candidate within the TPOT threshold.
        opt = self._create_optimizer(ttft_penalty=0, tpot_penalty=1.0)
        opt.fine_tune = _make_fine_tune_mock(tpot_upper_bound=0.05)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.04),
            PerformanceIndex(generate_speed=50, time_to_first_token=0.2, time_per_output_token=0.06),
        ]
        fitness_list = [1.0, 0.5]
        params_list = [np.array([10]), np.array([20])]
        f, p, pi = opt.best_params(fitness_list, params_list, perf_list)
        assert pi.generate_speed == 10

    def test_best_params_with_both_penalties(self):
        opt = self._create_optimizer(ttft_penalty=1.0, tpot_penalty=1.0)
        opt.fine_tune = _make_fine_tune_mock(tpot_upper_bound=0.05, ttft_upper_bound=0.15)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.04),
            PerformanceIndex(generate_speed=50, time_to_first_token=0.2, time_per_output_token=0.06),
        ]
        fitness_list = [1.0, 0.5]
        params_list = [np.array([10]), np.array([20])]
        f, p, pi = opt.best_params(fitness_list, params_list, perf_list)
        assert pi.generate_speed == 10

    def test_normalize_particle_position(self):
        opt = self._create_optimizer()
        position = np.array([50.0, 25000.0])
        corrected, ctx = opt._normalize_particle_position(position, 0, 5, 0)
        assert corrected is not None
        assert isinstance(ctx, DecodeContext)

    def test_skip_if_duplicate_first_time(self):
        opt = self._create_optimizer()
        position = np.array([50.0, 25000.0])
        ctx = DecodeContext(particle_index=0, n_particles=5, iteration=0)
        result = opt._skip_if_duplicate((50.0, 25000.0), 0, 0, position, ctx)
        assert result is False

    def test_skip_if_duplicate_second_time(self):
        opt = self._create_optimizer()
        position = np.array([50.0, 25000.0])
        ctx = DecodeContext(particle_index=0, n_particles=5, iteration=0)
        opt._seen_params[(50.0, 25000.0)] = (0, 0)
        result = opt._skip_if_duplicate((50.0, 25000.0), 1, 1, position, ctx)
        assert result is True

    def test_get_target_field_from_case_data(self):
        opt = self._create_optimizer()
        case_data = {"max_batch_size": 50, "max_prefill_token": 10000}
        fields = opt.get_target_field_from_case_data(case_data)
        assert fields[0].value == 50
        assert fields[1].value == 10000

    def test_get_target_field_from_case_data_missing_field(self):
        opt = self._create_optimizer()
        case_data = {"max_batch_size": 50}
        with pytest.raises(ValueError, match="Invalid data"):
            opt.get_target_field_from_case_data(case_data)


class TestAdapterTargetField:
    def test_context_manager_restores_field(self):
        scheduler = MagicMock()
        target_field = (
            OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int", value=50),
            OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int", value=32),
        )
        opt = PSOOptimizer(
            scheduler=scheduler,
            target_field=target_field,
            ttft_penalty=0,
            tpot_penalty=0,
            success_rate_penalty=0,
            ttft_slo=1.0,
            tpot_slo=0.1,
            success_rate_slo=0.9,
            generate_speed_target=100,
        )
        original_field = opt.target_field
        with adapter_target_field(opt):
            assert opt.target_field is not original_field
        assert opt.target_field is original_field


class TestSampleContextManager:
    @patch("optix.config.config.get_settings")
    def test_sample_no_sample_size(self, mock_settings):
        mock_settings.return_value.sample_size = None
        scheduler = MagicMock()
        with sample(scheduler):
            pass

    @patch("optix.config.config.get_settings")
    def test_sample_with_sample_size(self, mock_settings):
        mock_settings.return_value.sample_size = 50
        scheduler = MagicMock()
        scheduler.benchmark.num_prompts = 100
        scheduler.benchmark.__class__ = type("BenchmarkInterface", (), {})
        with sample(scheduler):
            pass


class TestEnableSimulate:
    @patch("optix.optimizer.optimizer.simulate_flag", False)
    def test_no_simulate_flag(self):
        scheduler = MagicMock()
        with enable_simulate(scheduler) as flag:
            assert flag is False


class TestOpFunc:
    """Test PSOOptimizer.op_func"""

    def _create_optimizer(self, **kwargs):
        return _make_pso_optimizer(**kwargs)

    def test_op_func_normal(self):
        opt = self._create_optimizer()
        perf = PerformanceIndex(
            generate_speed=200,
            time_to_first_token=0.1,
            time_per_output_token=0.01,
            success_rate=1.0,
        )
        opt.scheduler.run_with_request_rate.return_value = perf
        x = np.array([[50.0, 25000.0], [60.0, 30000.0], [70.0, 35000.0]])
        result = opt.op_func(x)
        assert len(result) == 3
        assert opt._iteration == 1

    def test_op_func_exception(self):
        opt = self._create_optimizer()
        opt.scheduler.run_with_request_rate.side_effect = Exception("service down")
        x = np.array([[50.0, 25000.0]])
        result = opt.op_func(x)
        assert result[0] == inf

    def test_op_func_duplicate_skipped(self):
        opt = self._create_optimizer()
        perf = PerformanceIndex(
            generate_speed=200,
            time_to_first_token=0.1,
            time_per_output_token=0.01,
            success_rate=1.0,
        )
        opt.scheduler.run_with_request_rate.return_value = perf
        # First call
        x = np.array([[50.0, 25000.0]])
        opt.op_func(x)
        # Second call with same params
        x2 = np.array([[50.0, 25000.0]])
        result = opt.op_func(x2)
        assert result[0] == inf


class TestComputerFitness:
    """Test PSOOptimizer.computer_fitness"""

    def _create_optimizer(self, **kwargs):
        overrides = dict(
            target_field=(
                OptimizerConfigField(name="f1", min=0, max=100, dtype="int"),
                OptimizerConfigField(name="f2", min=0, max=1000, dtype="int"),
            ),
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            success_rate_penalty=5.0,
            ttft_slo=0.5,
            tpot_slo=0.05,
            success_rate_slo=1.0,
            generate_speed_target=5000,
        )
        overrides.update(kwargs)
        return _make_pso_optimizer(**overrides)

    def test_computer_fitness_with_valid_data(self):
        opt = self._create_optimizer()
        opt.load_history_data = [
            {
                "f1": 50,
                "f2": 500,
                "fitness": 1.5,
            },
            {
                "f1": 60,
                "f2": 600,
                "fitness": 2.0,
            },
        ]
        positions, costs = opt.computer_fitness()
        assert len(positions) == 2
        assert len(costs) == 2
        assert costs[0] == 1.5
        assert costs[1] == 2.0

    def test_computer_fitness_nan_skipped(self):
        opt = self._create_optimizer()
        opt.load_history_data = [
            {
                "f1": 50,
                "f2": 500,
                "fitness": float("nan"),
            },
        ]
        positions, costs = opt.computer_fitness()
        assert len(positions) == 0

    def test_computer_fitness_out_of_bounds_skipped(self):
        opt = self._create_optimizer()
        opt.load_history_data = [
            {
                "f1": 200,  # exceeds max of 100
                "f2": 500,
                "fitness": 1.0,
            },
        ]
        positions, costs = opt.computer_fitness()
        assert len(positions) == 0

    def test_computer_fitness_computes_from_perf_index(self):
        opt = self._create_optimizer()
        opt.load_history_data = [
            {
                "f1": 50,
                "f2": 500,
                "generate_speed": 5000,
                "time_to_first_token": 0.1,
                "time_per_output_token": 0.01,
                "success_rate": 1.0,
            },
        ]
        positions, costs = opt.computer_fitness()
        assert len(positions) == 1
        assert costs[0] > 0


class TestBestParamsEdgeCases:
    """Test PSOOptimizer.best_params edge cases"""

    def _create_optimizer(self, **kwargs):
        overrides = dict(target_field=(OptimizerConfigField(name="f1", min=0, max=100, dtype="int"),))
        overrides.update(kwargs)
        return _make_pso_optimizer(**overrides)

    def test_mismatched_lengths(self):
        opt = self._create_optimizer()
        f, p, pi = opt.best_params([1.0], [np.array([10])], [])
        assert f is None

    def test_tpot_threshold_zero(self):
        opt = self._create_optimizer(ttft_penalty=0, tpot_penalty=1.0)
        opt.fine_tune = _make_fine_tune_mock(tpot_upper_bound=0)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.04),
        ]
        f, p, pi = opt.best_params([1.0], [np.array([10])], perf_list)
        assert f == 1.0

    def test_both_penalties_threshold_zero(self):
        opt = self._create_optimizer(ttft_penalty=1.0, tpot_penalty=1.0)
        opt.fine_tune = _make_fine_tune_mock(tpot_upper_bound=0, ttft_upper_bound=0.5)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.04),
        ]
        f, p, pi = opt.best_params([1.0], [np.array([10])], perf_list)
        assert f == 1.0

    def test_none_generate_speed_filled(self):
        opt = self._create_optimizer()
        perf_list = [
            PerformanceIndex(
                generate_speed=None,
                time_to_first_token=None,
                time_per_output_token=None,
            ),
        ]
        f, p, pi = opt.best_params([1.0], [np.array([10])], perf_list)
        assert pi.generate_speed == 0

    def test_tpot_penalty_no_slo_match(self):
        """All values exceed tpot threshold, pick smallest diff"""
        opt = self._create_optimizer(ttft_penalty=0, tpot_penalty=1.0)
        opt.fine_tune = _make_fine_tune_mock(tpot_upper_bound=0.01)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.02),
            PerformanceIndex(generate_speed=50, time_to_first_token=0.2, time_per_output_token=0.05),
        ]
        f, p, pi = opt.best_params([1.0, 0.5], [np.array([10]), np.array([20])], perf_list)
        # First has smallest diff: (0.02 - 0.01)/0.01 = 1.0 vs (0.05-0.01)/0.01 = 4.0
        assert pi.generate_speed == 10

    def test_both_penalties_no_slo_match(self):
        """All values exceed both thresholds, pick smallest sum"""
        opt = self._create_optimizer(ttft_penalty=1.0, tpot_penalty=1.0)
        opt.fine_tune = _make_fine_tune_mock(tpot_upper_bound=0.01, ttft_upper_bound=0.05)
        perf_list = [
            PerformanceIndex(generate_speed=10, time_to_first_token=0.1, time_per_output_token=0.02),
            PerformanceIndex(generate_speed=50, time_to_first_token=0.3, time_per_output_token=0.05),
        ]
        f, p, pi = opt.best_params([1.0, 0.5], [np.array([10]), np.array([20])], perf_list)
        # First: tpot_diff=(0.02-0.01)/0.01=1.0, ttft_diff=(0.1-0.05)/0.05=1.0, sum=2.0
        # Second: tpot_diff=(0.05-0.01)/0.01=4.0, ttft_diff=(0.3-0.05)/0.05=5.0, sum=9.0
        assert pi.generate_speed == 10


class TestRefineOptimizationCandidates:
    """Test PSOOptimizer.refine_optimization_candidates"""

    def _create_optimizer(self, **kwargs):
        overrides = dict(
            target_field=(
                OptimizerConfigField(name="CONCURRENCY", min=1, max=100, dtype="int", config_position="env"),
                OptimizerConfigField(name="REQUESTRATE", min=0.1, max=50, dtype="float", config_position="env"),
            ),
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            success_rate_penalty=5.0,
            ttft_slo=0.5,
            tpot_slo=0.05,
            success_rate_slo=1.0,
            generate_speed_target=5000,
            fine_tune=MagicMock(),
            max_fine_tune=3,
        )
        overrides.update(kwargs)
        opt = _make_pso_optimizer(**overrides)
        opt.default_run_param = np.array([50.0, 10.0])
        opt.default_res = PerformanceIndex(
            generate_speed=3000,
            time_to_first_token=0.3,
            time_per_output_token=0.04,
            success_rate=1.0,
        )
        opt.default_fitness = 1.5
        return opt

    def test_refine_runs_scheduler(self):
        opt = self._create_optimizer()
        perf = PerformanceIndex(
            generate_speed=4000,
            time_to_first_token=0.2,
            time_per_output_token=0.03,
            success_rate=1.0,
        )
        opt.scheduler.run.return_value = perf
        opt.fine_tune.reset_history = MagicMock()
        from optix.optimizer.experience_fine_tunning import StopFineTune

        opt.fine_tune.fine_tune_with_concurrency_and_request_rate.side_effect = StopFineTune()

        import pandas as pd

        best_results = pd.DataFrame([{"CONCURRENCY": 60, "REQUESTRATE": 15.0}])
        fitness_list, params_list, res_list = opt.refine_optimization_candidates(best_results)

        # refine_optimization_candidates always seeds the lists with the default
        # run, then appends one entry for each outer-loop iteration.  Because
        # StopFineTune breaks the inner fine-tune loop immediately, exactly
        # two entries are expected: [default, new_run].
        assert len(fitness_list) == 2
        assert len(params_list) == 2
        assert len(res_list) == 2

        # Entry 0 — the pre-seeded default values
        assert fitness_list[0] == opt.default_fitness
        assert np.array_equal(params_list[0], opt.default_run_param)
        assert res_list[0] == opt.default_res

        # Entry 1 — the run triggered by the best_results row (CONCURRENCY=60,
        # REQUESTRATE=15.0).  The fitness is computed by minimum_algorithm from
        # the mocked PerformanceIndex, so it must be a positive finite number.
        assert fitness_list[1] > 0
        assert fitness_list[1] < float("inf")
        assert len(params_list[1]) == 2
        assert isinstance(res_list[1], PerformanceIndex)
        assert res_list[1].generate_speed == perf.generate_speed

        opt.scheduler.run.assert_called()

    def test_refine_handles_runtime_exception(self):
        opt = self._create_optimizer()
        opt.scheduler.run.side_effect = Exception("service error")

        import pandas as pd

        best_results = pd.DataFrame([{"CONCURRENCY": 60, "REQUESTRATE": 15.0}])
        fitness_list, params_list, res_list = opt.refine_optimization_candidates(best_results)
        # Should contain at least the default
        assert len(fitness_list) >= 1


class TestOptimizerMain:
    """Test optimizer.main() function

    The decorators mock the heavy dependencies so that ``main()`` can be
    exercised end-to-end without real I/O or external processes.  The
    configuration flows from ``get_settings`` through ``PSOOptimizer``
    construction, so the assertions below verify that every relevant
    setting is forwarded correctly.
    """

    @patch("optix.deploy_env.validate_deploy_stack")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    @patch("optix.optimizer.optimizer.is_mindie", return_value=False)
    def test_main_basic_flow(
        self,
        mock_is_mindie,
        mock_is_root,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_validate_deploy_stack,
    ):
        import sys

        from optix.optimizer.optimizer import main as optix_main

        settings = MagicMock()
        settings.n_particles = 5
        settings.iters = 10
        settings.ttft_penalty = 3.0
        settings.tpot_penalty = 3.0
        settings.success_rate_penalty = 5.0
        settings.ttft_slo = 0.5
        settings.tpot_slo = 0.05
        settings.success_rate_slo = 1.0
        settings.generate_speed_target = 5000
        settings.max_fine_tune = 5
        settings.output = MagicMock()
        settings.step_size = 0.1
        settings.slo_coefficient = 1.2
        settings.ftol = 1e-3
        settings.ftol_iter = 5
        settings.data_storage = MagicMock()
        mock_get_settings.return_value = settings

        # Mock the simulators and benchmarks
        mock_simu = MagicMock()
        mock_simu.data_field = [
            OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int"),
        ]
        mock_bench = MagicMock()
        mock_bench.data_field = [
            OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int", config_position="env"),
        ]

        with (
            patch.dict(
                "optix.optimizer.register.simulates",
                {"mindie": lambda **kw: mock_simu},
            ),
            patch.dict(
                "optix.optimizer.register.benchmarks",
                {"ais_bench": lambda **kw: mock_bench},
            ),
            patch.object(sys, "argv", ["optix", "-e", "mindie", "-b", "ais_bench"]),
        ):
            optix_main()

        # ---- verify PSOOptimizer construction ----
        mock_pso.assert_called_once()
        _, pso_kwargs = mock_pso.call_args
        assert pso_kwargs["n_particles"] == 5
        assert pso_kwargs["iters"] == 10
        assert pso_kwargs["ttft_penalty"] == 3.0
        assert pso_kwargs["tpot_penalty"] == 3.0
        assert pso_kwargs["success_rate_penalty"] == 5.0
        assert pso_kwargs["ttft_slo"] == 0.5
        assert pso_kwargs["tpot_slo"] == 0.05
        assert pso_kwargs["success_rate_slo"] == 1.0
        assert pso_kwargs["generate_speed_target"] == 5000
        assert pso_kwargs["max_fine_tune"] == 5
        assert pso_kwargs["load_breakpoint"] is False
        assert pso_kwargs["fine_tune"] is mock_fine_tune.return_value
        assert pso_kwargs["pso_init_kwargs"] == {"ftol": 1e-3, "ftol_iter": 5}

        # ---- verify post-construction execution ----
        mock_pso.return_value.run_plugin.assert_called_once()
