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
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from math import inf, isclose, isinf, isnan
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from loguru import logger

from ..common import is_mindie
from ..config.base_config import (
    CONCURRENCYS,
    REAL_EVALUATION,
    REQUESTRATES,
    simulate_flag,
)
from ..config.config import DecodeContext, field_to_param, map_param_with_value
from ..logging import (
    LogStage,
    format_evaluation_failure,
    resolve_log_level as resolve_optix_env_log_level,
    set_log_level,
)
from ..optimizer.experience_fine_tunning import (
    FINE_TUNE_MODE_PD_DISAGGREGATION,
    FINE_TUNE_MODE_PD_MIXED,
)
from ..optimizer.errors import (
    BaselineRunError,
    ConfigFileNotFoundError,
    InvalidConfigError,
    NoFeasibleSolutionError,
    OptimizerError,
    PdDisaggError,
)
from ..optimizer.outcome import OptimizationResult, RunStatus
from ..optimizer.performance_tunner import PerformanceTuner
from ..optimizer.register import benchmarks, simulates
from ..optimizer.utils import get_required_field_from_json, is_root

MAX_ITER_NUM = 200


def _validate_optimization_mode_settings(
    mode: str,
    *,
    skip_pso: bool,
    manage_simulator_lifecycle: bool,
) -> None:
    if mode == "pd_disagg" and not manage_simulator_lifecycle:
        raise PdDisaggError(
            "pd_disagg mode requires manage_simulator_lifecycle=true because Prefill and Decode services must be "
            "managed during parameter search."
        )
    if mode == "pd_disagg" and skip_pso:
        raise PdDisaggError(
            "pd_disagg mode requires skip_pso=false because Prefill and Decode service parameters must be "
            "searched with PSO."
        )


class PSOOptimizer(PerformanceTuner):
    def __init__(
        self,
        scheduler,
        n_particles: int = 10,
        iters=100,
        pso_options=None,
        target_field: tuple | None = None,
        load_history_data: list | None = None,
        load_breakpoint: bool = False,
        pso_init_kwargs: dict | None = None,
        fine_tune=None,
        max_fine_tune: int = 30,
        skip_pso: bool = False,
        use_request_rate_calibration: bool = True,
        manage_simulator_lifecycle: bool = True,
        **kwargs,
    ):
        from ..config.config import PsoOptions, default_support_field

        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.scheduler.early_exit_fitness_evaluator = self
        self.n_particles = min(n_particles, MAX_ITER_NUM)
        self.iters = min(iters, MAX_ITER_NUM)
        self.target_field = target_field or default_support_field
        if not pso_options:
            self.pso_options = PsoOptions()
        else:
            self.pso_options = pso_options
        self.load_history_data = load_history_data
        self.load_breakpoint = load_breakpoint
        self.pso_init_kwargs = pso_init_kwargs or {}
        self.init_pos = None
        self.history_cost, self.history_pos = None, None
        self.default_fitness = None
        self.default_run_param = None
        self.default_res = None
        self.sample_data = None
        self.fine_tune = fine_tune
        self.max_fine_tune = min(max_fine_tune, MAX_ITER_NUM)
        self.skip_pso = skip_pso
        self.use_request_rate_calibration = use_request_rate_calibration
        self.manage_simulator_lifecycle = manage_simulator_lifecycle
        self._iteration = 0  # op_func call count, used for balanced strategy inter-iteration direction alternation
        self._refine_iter = 0  # +1 per refine candidate group, so backup dirs look like back_up/refine_1
        self._seen_params = {}

    def _run_benchmark_only_without_service_monitor(self, params, params_field):
        return self.scheduler.run_benchmark_only(params, params_field, monitor_service=False)

    def _run_reused_simulator_evaluation(self, params, params_field, decode_context=None):
        """Evaluate one candidate without taking ownership of the simulator lifecycle."""
        return self.scheduler.rerun_benchmark_only(
            params,
            params_field,
            decode_context=decode_context,
            with_request_rate=self.use_request_rate_calibration,
            monitor_service=False,
        )

    def _run_complete_reused_simulator_evaluation(self, params, target_field=None):
        """Run a complete benchmark-only refinement trial against the live simulator."""
        target_field = target_field or self.target_field
        with self.scheduler.disable_early_exit():
            return self.scheduler.rerun_benchmark_only(
                params,
                target_field,
                with_request_rate=False,
                monitor_service=False,
            )

    def _stop_after_fine_tune(self):
        if self.manage_simulator_lifecycle:
            self.scheduler.stop_target_server()
        else:
            self.scheduler.benchmark.stop()

    def _is_pd_mixed_fine_tune(self):
        return getattr(self.fine_tune, "fine_tune_mode", None) == FINE_TUNE_MODE_PD_MIXED

    def _should_keep_service_for_pd_mixed_fine_tune(self):
        return self.manage_simulator_lifecycle and self._is_pd_mixed_fine_tune()

    @staticmethod
    def _pso_state_allows_retry(optimizer) -> bool:
        is_global_best_missing = getattr(optimizer, "_is_global_best_missing", None)
        if not callable(is_global_best_missing):
            return False
        try:
            return bool(is_global_best_missing())
        except (TypeError, ValueError):
            return False

    def _run_pso_optimize(self, optimizer):
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                return optimizer.optimize(self.op_func, iters=self.iters)
            except ValueError as error:
                should_retry = attempt < max_attempts and self._pso_state_allows_retry(optimizer)
                if not should_retry:
                    if attempt > 1:
                        logger.warning("PSO optimization retry failed; aborting. error: {}", error)
                    raise
                logger.warning(
                    "PSO optimization failed before establishing a global best; retrying ({}/{}). error: {}",
                    attempt + 1,
                    max_attempts,
                    error,
                )
        raise RuntimeError("unreachable PSO retry state")

    def _run_complete_evaluation(self, params: np.ndarray, target_field=None):
        """Run a baseline or refinement trial to completion.

        Early exit is disabled because these runs require complete benchmark
        metrics for reliable fitness evaluation and candidate selection.
        """
        evaluation_target_field = self.target_field if target_field is None else target_field
        with self.scheduler.disable_early_exit():
            return self.scheduler.run(params, evaluation_target_field)

    @staticmethod
    def is_within_boundary(target_pos, min_bound, max_bound):
        for i, v in enumerate(target_pos):
            if min_bound[i] <= v <= max_bound[i]:
                continue
            return False
        return True

    @staticmethod
    def params_in_records(params, record_params):
        for _his_params in record_params:
            if (_his_params == params).all():
                return True
        return False

    def get_target_field_from_case_data(self, case_data):
        _target_field = deepcopy(self.target_field)
        for _field in _target_field:
            _case_value = case_data.get(_field.name, None)
            if _case_value is None:
                raise ValueError("Invalid data.")
            _field.value = _case_value
        return _target_field

    def computer_fitness(self) -> tuple:
        from ..config.config import PerformanceIndex

        all_position = []
        all_cost = []
        _min_bound, _max_bound = self.constructing_bounds()
        for case_data in self.load_history_data:
            _fitness = case_data.get("fitness")
            if not _fitness:
                _params = {}
                for k in PerformanceIndex.model_fields.keys():
                    if k in case_data:
                        _params[k] = case_data[k]
                performance_index = PerformanceIndex(**_params)
                _fitness = self.minimum_algorithm(performance_index)
            if isnan(_fitness) or isinf(_fitness):
                continue
            try:
                _target_field = self.get_target_field_from_case_data(case_data)
            except ValueError:
                continue
            _pos = field_to_param(_target_field)
            if not self.is_within_boundary(_pos, _min_bound, _max_bound):
                continue
            all_cost.append(_fitness)
            all_position.append(_pos)
        if len(all_position) != len(all_cost):
            raise ValueError("Failed in computer_fitness.")
        return all_position, all_cost

    def _normalize_particle_position(
        self,
        position: np.ndarray,
        particle_index: int,
        n_particles: int,
        iteration: int,
    ):
        """
        Apply constraint normalization to a single particle position: apply field derivation rule repairs
        (e.g., ternary_factories constraints), convert repaired actual params back to continuous-space position vector.

        Returns: (corrected_position, decode_context)
        - corrected_position: repaired position vector (may differ from original)
        - decode_context:     context used for this normalization, caller can reuse for scheduler
        """
        decode_context = DecodeContext(
            particle_index=particle_index,
            n_particles=n_particles,
            iteration=iteration,
        )
        try:
            actual_params = map_param_with_value(position, self.target_field, decode_context=decode_context)
            true_x = field_to_param(tuple(actual_params))
            if not np.allclose(position, true_x, atol=1e-9):
                logger.debug(
                    f"Particle {particle_index}: PSO position corrected {position} → {true_x} "
                    f"(ternary_factories repair)"
                )
            return true_x, decode_context
        except Exception as e:
            logger.warning(f"Position correction failed for particle {particle_index}: {e}")
            return position, decode_context

    def _skip_if_duplicate(
        self,
        param_key: tuple,
        particle_index: int,
        iteration: int,
        position: np.ndarray,
        decode_context,
    ) -> bool:
        from ..config.config import PerformanceIndex

        if param_key not in self._seen_params:
            self._seen_params[param_key] = (iteration, particle_index)
            return False
        prev_iter, prev_particle = self._seen_params[param_key]
        logger.bind(particle=particle_index, prev_iter=prev_iter, prev_particle=prev_particle).info(
            "Params already evaluated (iter={}, particle={}), skipping.",
            prev_iter,
            prev_particle,
        )
        self.scheduler.simulate_run_info = map_param_with_value(
            position, self.target_field, decode_context=decode_context
        )
        self.scheduler.error_info = f"skip: param type mapping same as iter={prev_iter} particle={prev_particle}"
        self.scheduler.performance_index = PerformanceIndex()
        self.scheduler.save_result(fitness=inf, stop_service=self.manage_simulator_lifecycle)
        return True

    def op_func(self, x) -> np.ndarray:
        n_particles = x.shape[0]
        current_iteration = self._iteration
        self._iteration += 1
        # pyswarms calls op_func once per iteration; use the incremented index so backup dirs look like back_up/pso_1
        self.scheduler.set_backup_phase("pso", self._iteration)
        generate_speed = []
        if self.manage_simulator_lifecycle:
            scheduler_run = (
                self.scheduler.run_with_request_rate if self.use_request_rate_calibration else self.scheduler.run
            )
        else:
            scheduler_run = self._run_reused_simulator_evaluation
        with logger.contextualize(iter=current_iteration, stage=LogStage.SEARCH.value):
            for i in range(n_particles):
                x[i], decode_context = self._normalize_particle_position(x[i], i, n_particles, current_iteration)
                param_key = tuple(np.round(x[i], decimals=6))
                if self._skip_if_duplicate(param_key, i, current_iteration, x[i], decode_context):
                    generate_speed.append(inf)
                    continue
                try:
                    _res = scheduler_run(x[i], self.target_field, decode_context=decode_context)
                    if self.scheduler.last_outcome and self.scheduler.last_outcome.status == RunStatus.FAILED:
                        logger.bind(particle=i).warning(
                            "Evaluation failed, fitness=inf: {}",
                            format_evaluation_failure(self.scheduler, self.scheduler.error_info),
                        )
                        _fitness = inf
                    else:
                        _fitness = self.minimum_algorithm(_res)
                except OptimizerError:
                    raise
                except Exception as e:
                    logger.bind(particle=i).warning("Evaluation failed, fitness=inf: {}", e)
                    _fitness = inf
                self.scheduler.save_result(fitness=_fitness, stop_service=self.manage_simulator_lifecycle)
                generate_speed.append(_fitness)
                logger.trace("Particle {} fitness {}", i, _fitness)
        return np.array(generate_speed)

    def constructing_bounds(self) -> tuple[tuple, tuple]:
        """
        Returns example: ((0, 10), (0, 10))
        """
        _min = []
        _max = []
        for _field in self.target_field:
            if _field.constant is not None or isclose(_field.min, _field.max, rel_tol=1e-5):
                continue
            _min.append(_field.min)
            _max.append(_field.max)
        return (tuple(_min), tuple(_max))

    def dimensions(self):
        d = 0
        for _field in self.target_field:
            if _field.constant is not None or isclose(_field.min, _field.max, rel_tol=1e-5):
                continue
            d += 1
        return d

    def _run_and_record_fine_tune_params(
        self,
        params,
        record_fitness,
        record_params,
        record_res,
        run_method=None,
        stop_service=True,
        force_run=False,
    ):
        params_recorded = self.params_in_records(params, record_params)
        if params_recorded and not force_run:
            return None, None, False
        if run_method is None:
            run_method = self.scheduler.run
        try:
            result = run_method(params, self.target_field)
            if self.scheduler.last_outcome and self.scheduler.last_outcome.status == RunStatus.FAILED:
                logger.error(
                    "Runtime exception. error: {}, please check.",
                    format_evaluation_failure(self.scheduler, self.scheduler.error_info),
                )
                fitness = inf
                self.scheduler.save_result(fitness=fitness, stop_service=stop_service)
                return None, None, False
            fitness = self.minimum_algorithm(result)
        except Exception as e:
            logger.error(
                "Runtime exception. error: {}, please check.",
                format_evaluation_failure(self.scheduler, e),
            )
            fitness = inf
            self.scheduler.save_result(fitness=fitness, stop_service=stop_service)
            return None, None, False
        self.scheduler.save_result(fitness=fitness, stop_service=stop_service)
        if not params_recorded:
            record_params.append(params)
            record_res.append(result)
            record_fitness.append(fitness)
        return result, fitness, True

    def _fine_tune_from_candidate(self, params, result, record_fitness, record_params, record_res):
        from ..optimizer.experience_fine_tunning import StopFineTune

        if self.fine_tune.fine_tune_mode == FINE_TUNE_MODE_PD_DISAGGREGATION:
            self._fine_tune_pd_disaggregation_from_candidate(params, result, record_fitness, record_params, record_res)
            return

        self.fine_tune.reset_history()
        try:
            for _ in range(self.max_fine_tune):
                try:
                    simulate_run_info = self.fine_tune.fine_tune_with_concurrency_and_request_rate(params, result)
                except ValueError as e:
                    logger.error("Failed in fine-tuning parameter. error: {}", e)
                    break
                except StopFineTune:
                    break
                params = field_to_param(simulate_run_info)
                fine_tune_run_method = (
                    self.scheduler.run_benchmark_only
                    if self.manage_simulator_lifecycle
                    else self._run_benchmark_only_without_service_monitor
                )
                result, _, was_recorded = self._run_and_record_fine_tune_params(
                    params,
                    record_fitness,
                    record_params,
                    record_res,
                    run_method=fine_tune_run_method,
                    stop_service=False,
                )
                if not was_recorded:
                    break
        finally:
            self._stop_after_fine_tune()

    def _fine_tune_pd_disaggregation_from_candidate(self, params, result, record_fitness, record_params, record_res):
        from ..optimizer.experience_fine_tunning import StopFineTune

        self.fine_tune.reset_history()
        try:
            try:
                probe_run_info = self.fine_tune.prepare_pd_disaggregation_probe(params)
            except ValueError as e:
                logger.error("Failed in PD disaggregation probe setup. error: {}", e)
                return
            probe_params = field_to_param(probe_run_info)
            probe_run_method = (
                self.scheduler.run
                if self.manage_simulator_lifecycle
                else self._run_benchmark_only_without_service_monitor
            )
            probe_result, _, was_recorded = self._run_and_record_fine_tune_params(
                probe_params,
                record_fitness,
                record_params,
                record_res,
                run_method=probe_run_method,
                stop_service=False,
                force_run=True,
            )
            if not was_recorded:
                return
            try:
                fixed_request_rate = self.fine_tune.init_pd_disaggregation_request_rate(probe_result)
            except ValueError as e:
                logger.error("Failed in PD disaggregation request rate setup. error: {}", e)
                return
            logger.info("PD disaggregation fixed request rate: {}", fixed_request_rate)

            params = probe_params
            result = probe_result
            for _ in range(self.max_fine_tune):
                try:
                    simulate_run_info = self.fine_tune.fine_tune_pd_disaggregation(params, result)
                except ValueError as e:
                    logger.error(
                        "Failed in PD disaggregation fine-tuning parameter. error: {}",
                        e,
                    )
                    break
                except StopFineTune:
                    break
                params = field_to_param(simulate_run_info)
                fine_tune_run_method = (
                    self.scheduler.run_benchmark_only
                    if self.manage_simulator_lifecycle
                    else self._run_benchmark_only_without_service_monitor
                )
                result, _, was_recorded = self._run_and_record_fine_tune_params(
                    params,
                    record_fitness,
                    record_params,
                    record_res,
                    run_method=fine_tune_run_method,
                    stop_service=False,
                )
                if not was_recorded:
                    break
        finally:
            self._stop_after_fine_tune()

    def refine_default_candidate(self):
        record_params = [self.default_run_param]
        record_res = [self.default_res]
        record_fitness = [self.default_fitness]
        self._fine_tune_from_candidate(
            self.default_run_param,
            self.default_res,
            record_fitness,
            record_params,
            record_res,
        )
        return record_fitness, record_params, record_res

    def refine_optimization_candidates(self, best_results: pd.DataFrame):
        _record_params = [self.default_run_param]
        _record_res = [self.default_res]
        _record_fitness = [self.default_fitness]
        fine_tune_enabled = self.scheduler.data_storage.config.pso_top_k > 0
        if not fine_tune_enabled:
            logger.info("pso_top_k=0; validating the best PSO candidate without running FineTune.")
        for _, _pso_info in best_results.iterrows():
            # The fine-tuning of each optimization candidate group is placed in one refine iteration dir, e.g. back_up/refine_1
            self._refine_iter += 1
            self.scheduler.set_backup_phase("refine", self._refine_iter)
            _target_field = self.get_target_field_from_case_data(_pso_info)
            for _field in _target_field:
                if _field.name in REQUESTRATES:
                    _field.value = _field.find_available_value(_field.value * 2)
            params = field_to_param(_target_field)
            run_method = (
                self._run_complete_evaluation
                if self.manage_simulator_lifecycle
                else self._run_complete_reused_simulator_evaluation
            )
            try:
                _res = run_method(params)
                if self.scheduler.last_outcome and self.scheduler.last_outcome.status == RunStatus.FAILED:
                    logger.error(
                        "Runtime exception. error: {}, please check.",
                        format_evaluation_failure(self.scheduler, self.scheduler.error_info),
                    )
                    _fitness = inf
                    self.scheduler.save_result(fitness=_fitness, stop_service=self.manage_simulator_lifecycle)
                    continue
                _fitness = self.minimum_algorithm(_res)
            except Exception as e:
                logger.error(
                    "Runtime exception. error: {}, please check.",
                    format_evaluation_failure(self.scheduler, e),
                )
                _fitness = inf
                self.scheduler.save_result(fitness=_fitness, stop_service=self.manage_simulator_lifecycle)
                continue
            self.scheduler.save_result(
                fitness=_fitness,
                stop_service=self.manage_simulator_lifecycle and not self._should_keep_service_for_pd_mixed_fine_tune(),
            )
            _record_params.append(params)
            _record_res.append(_res)
            _record_fitness.append(_fitness)
            if fine_tune_enabled:
                self._fine_tune_from_candidate(params, _res, _record_fitness, _record_params, _record_res)
        return _record_fitness, _record_params, _record_res

    def get_max_generate_speed_index(self, performance_index_list, slo_index):
        _best_index = 0
        _max = 0
        for i, v in enumerate(performance_index_list):
            if i not in slo_index:
                continue
            if v.generate_speed > _max:
                _max = v.generate_speed
                _best_index = i
        return _best_index

    @staticmethod
    def _performance_speed(performance_index):
        if performance_index.throughput is not None:
            return performance_index.throughput
        return performance_index.generate_speed

    def best_pd_disaggregation_params(self, fitnese_list, params_list, performance_index_list):
        _tpot_threshold = self.fine_tune.tpot_upper_bound
        if _tpot_threshold == 0:
            return fitnese_list[0], params_list[0], performance_index_list[0]
        _tpot_lt_slo_index = [
            i for i, p in enumerate(performance_index_list) if p.time_per_output_token <= _tpot_threshold
        ]
        if _tpot_lt_slo_index:
            _best_index = max(
                _tpot_lt_slo_index,
                key=lambda i: self._performance_speed(performance_index_list[i]),
            )
            return (
                fitnese_list[_best_index],
                params_list[_best_index],
                performance_index_list[_best_index],
            )
        _tpot_diff = [(p.time_per_output_token - _tpot_threshold) / _tpot_threshold for p in performance_index_list]
        _best_index = _tpot_diff.index(min(_tpot_diff))
        return (
            fitnese_list[_best_index],
            params_list[_best_index],
            performance_index_list[_best_index],
        )

    def best_params(self, fitnese_list, params_list, performance_index_list):
        if not performance_index_list or not fitnese_list or not params_list:
            logger.error(
                f"Input is empty."
                f"performance_index_list:{performance_index_list},"
                f"fitnese_list: {fitnese_list},"
                f"params_list: {params_list}"
            )
            return None, None, None
        if len(fitnese_list) != len(params_list) != len(performance_index_list):
            logger.error(
                f"The number of input elements does not match."
                f"performance_index_list:{len(performance_index_list)},"
                f"fitnese_list: {len(fitnese_list)},"
                f"params_list: {len(params_list)}"
            )
            return None, None, None
        for _p in performance_index_list:
            if _p.generate_speed is None:
                _p.generate_speed = 0
            if _p.time_to_first_token is None:
                _p.time_to_first_token = inf
            if _p.time_per_output_token is None:
                _p.time_per_output_token = inf

        if getattr(self.fine_tune, "fine_tune_mode", None) == FINE_TUNE_MODE_PD_DISAGGREGATION:
            return self.best_pd_disaggregation_params(fitnese_list, params_list, performance_index_list)

        if self.tpot_penalty == 0 and self.ttft_penalty == 0:
            _generate_speed = [p.generate_speed for p in performance_index_list]
            _best_index = _generate_speed.index(max(_generate_speed))
            return (
                fitnese_list[_best_index],
                params_list[_best_index],
                performance_index_list[_best_index],
            )
        if self.ttft_penalty == 0 and self.tpot_penalty != 0:
            _tpot_threshold = self.fine_tune.tpot_upper_bound
            if _tpot_threshold == 0:
                return fitnese_list[0], params_list[0], performance_index_list[0]
            _tpot_diff = [(p.time_per_output_token - _tpot_threshold) / _tpot_threshold for p in performance_index_list]
            _tpot_lt_slo_index = [i for i, v in enumerate(_tpot_diff) if v < 0]
            if _tpot_lt_slo_index:
                _best_index = self.get_max_generate_speed_index(performance_index_list, _tpot_lt_slo_index)
                return (
                    fitnese_list[_best_index],
                    params_list[_best_index],
                    performance_index_list[_best_index],
                )
            _best_index = _tpot_diff.index(min(_tpot_diff))
            return (
                fitnese_list[_best_index],
                params_list[_best_index],
                performance_index_list[_best_index],
            )
        if self.ttft_penalty != 0 and self.tpot_penalty != 0:
            _tpot_threshold = self.fine_tune.tpot_upper_bound
            _ttft_threshold = self.fine_tune.ttft_upper_bound
            if _tpot_threshold == 0 or _ttft_threshold == 0:
                return fitnese_list[0], params_list[0], performance_index_list[0]
            _performance_diff = [
                (
                    (p.time_per_output_token - _tpot_threshold) / _tpot_threshold,
                    (p.time_to_first_token - _ttft_threshold) / _ttft_threshold,
                )
                for p in performance_index_list
            ]
            _performance_lt_slo_index = [i for i, v in enumerate(_performance_diff) if all(kv < 0 for kv in v)]
            if _performance_lt_slo_index:
                _best_index = self.get_max_generate_speed_index(performance_index_list, _performance_lt_slo_index)
                return (
                    fitnese_list[_best_index],
                    params_list[_best_index],
                    performance_index_list[_best_index],
                )
            _performance_diff_sum = [sum(v) for v in _performance_diff]
            _best_index = _performance_diff_sum.index(min(_performance_diff_sum))
            return (
                fitnese_list[_best_index],
                params_list[_best_index],
                performance_index_list[_best_index],
            )
        return fitnese_list[0], params_list[0], performance_index_list[0]

    def mindie_prepare(self, mc):
        from ..config.config import get_settings

        settings = get_settings()
        if mc is None:
            return
        if not settings.theory_guided_enable:
            return
        mc.avg_input_length = self.scheduler.benchmark.get_performance_metric("InputTokens")
        mc.max_input_length = self.scheduler.benchmark.get_performance_metric("InputTokens", algorithm="max")
        mc.max_output_length = self.scheduler.benchmark.get_performance_metric("OutputTokens", algorithm="max")
        logger.debug(
            f"avg_input_length: {mc.avg_input_length}, max_input_length: {mc.max_input_length},"
            f"max_output_length: {mc.max_output_length}"
        )
        max_batch_size_lb, max_batch_size_ub = mc.get_max_batch_size_bound()
        if not isinf(max_batch_size_ub):
            scale_max_batch_size_ub = int(max_batch_size_ub * settings.scaling_coefficient)
        else:
            scale_max_batch_size_ub = inf
        if max_batch_size_lb >= max_batch_size_ub or max_batch_size_lb <= 0 or max_batch_size_ub <= 0:
            logger.warning(
                f"Theoretical derivation scope failure.max_batch_size_lb {max_batch_size_lb}, "
                f"max_batch_size_ub {max_batch_size_ub}, please check env"
            )
            return
        logger.debug(
            f"max_batch_size_lb {max_batch_size_lb}, max_batch_size_ub {max_batch_size_ub}. "
            f"scale_max_batch_size_ub {scale_max_batch_size_ub}"
        )

        for _field in self.target_field:
            if _field.name == "max_batch_size":
                if _field.min < max_batch_size_lb < _field.max:
                    _field.min = max_batch_size_lb
                if _field.min < scale_max_batch_size_ub < _field.max:
                    _field.max = scale_max_batch_size_ub
                    break
                if _field.min < max_batch_size_ub < _field.max:
                    _field.max = max_batch_size_ub
                break
        logger.debug(f"target_field: {self.target_field}")

    def _raise_if_baseline_failed(self) -> None:
        if not self.scheduler.error_info:
            return
        err = BaselineRunError.from_scheduler(self.scheduler)
        if self.manage_simulator_lifecycle:
            del_log = self.scheduler.del_log if self.scheduler.del_log is not None else False
            self.scheduler.stop_target_server(del_log)
        else:
            self.scheduler.benchmark.stop()
        raise err

    @staticmethod
    def _field_names(data_field) -> set[str]:
        return {field.name for field in data_field if hasattr(field, "name")}

    @staticmethod
    def _select_fields_by_name(target_field, names: set[str]):
        return tuple(field for field in target_field if field.name in names)

    def _restore_search_data_field(self, target_field, simulator_names: set[str], benchmark_names: set[str]) -> None:
        if hasattr(self.scheduler.simulator, "data_field"):
            self.scheduler.simulator.data_field = self._select_fields_by_name(target_field, simulator_names)
        if hasattr(self.scheduler.benchmark, "data_field"):
            self.scheduler.benchmark.data_field = self._select_fields_by_name(target_field, benchmark_names)

    def _run_baseline_preserving_search_space(self):
        """Run baseline evaluation while preserving search-space data fields.

        scheduler.run may overwrite simulator.data_field and benchmark.data_field
        as a side effect. Snapshot the original field names before the run and
        restore them in finally so the PSO search space defined by
        self.target_field remains intact for subsequent iterations.
        """
        simulator_names = self._field_names(getattr(self.scheduler.simulator, "data_field", ()))
        benchmark_names = self._field_names(getattr(self.scheduler.benchmark, "data_field", ()))
        search_target_field = tuple(self.target_field)
        baseline_target_field = tuple(deepcopy(self.target_field))
        self.default_run_param = field_to_param(baseline_target_field)
        try:
            return self._run_complete_evaluation(self.default_run_param, baseline_target_field)
        finally:
            self._restore_search_data_field(search_target_field, simulator_names, benchmark_names)

    def prepare_plugin(self):
        from ..config.config import get_settings
        from ..config.model_config import MindieModelConfig
        from ..optimizer.plugins.benchmark import AisBench
        from ..optimizer.plugins.simulate import Simulator

        with logger.contextualize(stage=LogStage.BASELINE.value):
            # The default-parameter baseline run is placed in the default phase, backup dir looks like back_up/default_1
            self.scheduler.set_backup_phase("default", 1)

            # Checkpoint: skip baseline run if results were loaded from a prior run
            if getattr(self, "_baseline_from_checkpoint", False):
                logger.info("Baseline restored from checkpoint — skipping evaluation run")
            elif isinstance(self.scheduler.simulator, Simulator):
                settings = get_settings()
                mc = None
                if is_mindie() and settings.theory_guided_enable:
                    mc = MindieModelConfig(self.scheduler.simulator.config.config_path)
                for _, _field in enumerate(self.target_field):
                    if _field.config_position.startswith("BackendConfig"):
                        _field.value = get_required_field_from_json(
                            self.scheduler.simulator.default_config,
                            _field.config_position,
                        )
                    elif _field.config_position == "env":
                        _field.value = os.getenv(_field.name, _field.value)
                if self.manage_simulator_lifecycle:
                    self.default_res = self._run_baseline_preserving_search_space()
                else:
                    self.default_run_param = field_to_param(self.target_field)
                    self.default_res = self.scheduler.run_benchmark_only(
                        self.default_run_param,
                        self.target_field,
                        monitor_service=False,
                    )
                self._raise_if_baseline_failed()
                if self.default_res.generate_speed:
                    self.gen_speed_target = 10 * self.default_res.generate_speed
                self.default_fitness = self.minimum_algorithm(self.default_res)
                self.scheduler.save_result(
                    fitness=self.default_fitness,
                    stop_service=self.manage_simulator_lifecycle
                    and not (self.skip_pso and self._is_pd_mixed_fine_tune()),
                )
                if is_mindie():
                    self.mindie_prepare(mc)
                if isinstance(self.scheduler.benchmark, AisBench):
                    _concurrency = self.scheduler.benchmark.get_best_concurrency()
                    for _field in self.target_field:
                        if _field.name in CONCURRENCYS and _field.min != _field.max:
                            if _field.min < _concurrency < _field.max:
                                _field.value = _field.max = _concurrency
                            elif _concurrency < _field.min:
                                _field.value = _field.max = _field.min
                            else:
                                _field.value = _field.max
            else:
                if self.manage_simulator_lifecycle:
                    self.default_res = self._run_baseline_preserving_search_space()
                else:
                    self.default_run_param = field_to_param(self.target_field)
                    self.default_res = self.scheduler.run_benchmark_only(
                        self.default_run_param,
                        self.target_field,
                        monitor_service=False,
                    )
                self._raise_if_baseline_failed()
                self.default_fitness = self.minimum_algorithm(self.default_res)
                self.scheduler.save_result(
                    fitness=self.default_fitness,
                    stop_service=self.manage_simulator_lifecycle
                    and not (self.skip_pso and self._is_pd_mixed_fine_tune()),
                )

            if (
                self.default_res.generate_speed is None
                or self.default_res.time_to_first_token is None
                or self.default_res.time_per_output_token is None
            ):
                logger.warning(
                    "Failed to obtain benchmark metric data. metric {}. "
                    "Please check if the benchmark is running successfully.",
                    self.default_res,
                )

            self.target_field = [
                *self.scheduler.simulator.data_field,
                *self.scheduler.benchmark.data_field,
            ]
            logger.success("Baseline established")

    def run_plugin(self):
        self.prepare_plugin()
        if self.skip_pso:
            logger.info("skip_pso is enabled; refining the baseline result without running PSO.")
            _record_fitness, _record_params, _record_res = self.refine_default_candidate()
        else:
            from ..optimizer.global_best_custom import CustomGlobalBestPSO

            with adapter_target_field(self):
                if self.load_breakpoint:
                    self.load_history_data = self.scheduler.data_storage.load_history_position(
                        self.scheduler.data_storage.config.store_dir,
                        filter_field={REAL_EVALUATION: True},
                    )
                if self.load_history_data and self.load_breakpoint:
                    self.history_pos, self.history_cost = self.computer_fitness()
                optimizer = CustomGlobalBestPSO(
                    n_particles=self.n_particles,
                    dimensions=self.dimensions(),
                    options=self.pso_options.model_dump(),
                    bounds=self.constructing_bounds(),
                    init_pos=self.init_pos,
                    breakpoint_pos=self.history_pos,
                    breakpoint_cost=self.history_cost,
                    **self.pso_init_kwargs,
                )
                with enable_simulate(self.scheduler):
                    cost, joint_vars = self._run_pso_optimize(optimizer)
                    best_results = self.scheduler.data_storage.get_best_result()
            _record_fitness, _record_params, _record_res = self.refine_optimization_candidates(best_results)
        best_fitness, best_param, best_performance_index = self.best_params(
            _record_fitness, _record_params, _record_res
        )
        if best_param is None or best_fitness is None or best_performance_index is None:
            raise NoFeasibleSolutionError()
        _position = {_field.name: _field.value for _field in map_param_with_value(best_param, self.target_field)}
        logger.success(
            "Optimization complete: fitness={} ttft={} tpot={} throughput={} params={}",
            best_fitness,
            best_performance_index.time_to_first_token,
            best_performance_index.time_per_output_token,
            best_performance_index.generate_speed,
            _position,
        )
        return OptimizationResult(
            fitness=best_fitness,
            params=_position,
            performance_index=best_performance_index,
        )


@contextmanager
def adapter_target_field(pso_optimizer: PSOOptimizer, *, on_pin_concurrency=None):
    """Temporarily install a mode-reduced deep copy of ``target_field`` for the ``with`` block.

    Shared by the PSO run plugin and the agent orchestrator: PSO internals read
    the search space from the instance attribute, so the reduced copy is
    installed for the duration of the block and the original list is restored
    afterwards (also on exception). ``on_pin_concurrency``, when given, is
    invoked with the field whenever request-rate calibration pins it to max.
    """
    _bak_target_field = pso_optimizer.target_field
    target_field = deepcopy(pso_optimizer.target_field)
    fix_concurrency = pso_optimizer.use_request_rate_calibration
    for _field in target_field:
        if _field.name in CONCURRENCYS and _field.constant is None and fix_concurrency:
            # True mode fixes CONCURRENCY at max while run_with_request_rate calibrates the request rate.
            _field.constant = _field.value = _field.convert_dtype(_field.max)
            if on_pin_concurrency is not None:
                on_pin_concurrency(_field)
        elif _field.name in REQUESTRATES and _field.constant is None:
            # REQUESTRATE=0 means unlimited traffic, so it is the semantic maximum
            # whenever the configured range contains zero. Otherwise use the
            # numeric upper bound. The field is fixed only for the PSO dimension;
            # request-rate calibration may still replace its value for a second run.
            maximum_rate = 0 if _field.min <= 0 <= _field.max else _field.max
            _field.constant = _field.value = _field.convert_dtype(maximum_rate)
        elif _field.constant is not None and _field.constant != _field.value:
            _field.value = _field.constant
    pso_optimizer.target_field = target_field
    try:
        yield
    finally:
        pso_optimizer.target_field = _bak_target_field


@contextmanager
def enable_simulate(scheduler):
    """
    Enter simulation mode
    Args: scheduler: The scheduler runner
    Returns
    """
    if simulate_flag:
        with scheduler.simulator.enable_simulation_model() as flag:
            yield flag
    else:
        yield False


def _run_optimizer(run_id: str | None = None) -> None:
    from ..config.config import Settings, get_settings, register_settings
    from ..optimizer.experience_fine_tunning import FineTune
    from ..optimizer.register import (
        DEFAULT_BENCHMARK_POLICY,
        register_ori_functions,
        validate_benchmark_policy,
        validate_simulator_policy,
    )
    from ..optimizer.register import (
        benchmarks as _benchmarks,
    )
    from ..optimizer.register import (
        simulates as _simulates,
    )
    from ..optimizer.scheduler import Scheduler
    from ..optimizer.store import DataStorage
    from ..plugins import load_general_plugins
    from cli.spec_cli import (
        METAVAR_FILE,
        SpecArgumentParser,
        add_log_options,
        add_option,
        add_version_option,
        make_token_type,
        parse_args as spec_parse_args,
    )

    register_ori_functions()
    load_general_plugins()

    parser = SpecArgumentParser(
        prog="msmodeling optix",
        description="Service parameter optimizer for LLM inference performance tuning.",
        examples=(
            "# Run with the default vLLM engine and AISBench\n"
            "msmodeling optix -e vllm -b ais_bench\n"
            "# Resume a previous search\n"
            "msmodeling optix --load-breakpoint --config ./config.toml"
        ),
    )
    add_version_option(parser)
    add_log_options(parser)
    parser.add_argument(
        "--mode",
        choices=("standard", "pd_disagg"),
        default="standard",
        help=(
            "Optimization workflow. standard runs the existing general optimization, including optional PSO "
            "and fine-tuning; "
            "pd_disagg runs separate Prefill and Decode searches followed by ratio recommendation. "
            "Defaults to standard."
        ),
    )
    parse_engine, engine_meta = make_token_type(
        list(_simulates.keys()),
        "--engine",
        store_canonical="snake",
        registered_names=True,
    )
    parse_bench, bench_meta = make_token_type(
        list(_benchmarks.keys()),
        "--benchmark-policy",
        store_canonical="snake",
        registered_names=True,
    )
    add_option(
        parser,
        "--load-breakpoint",
        dest="load_breakpoint",
        default=False,
        action="store_true",
        help="Continue from where the last optimization was aborted.",
        aliases=("--load_breakpoint", "-lb"),
    )
    parser.add_argument(
        "--backup",
        default=False,
        action="store_true",
        help="Back up optimizer data.",
    )
    add_option(
        parser,
        "-b",
        "--benchmark-policy",
        dest="benchmark_policy",
        default=DEFAULT_BENCHMARK_POLICY,
        type=parse_bench,
        metavar=bench_meta,
        help=(f"Benchmark used for custom performance indicators. [default: {DEFAULT_BENCHMARK_POLICY}]"),
        aliases=("--benchmark_policy",),
    )
    parser.add_argument(
        "-e",
        "--engine",
        default="vllm",
        type=parse_engine,
        metavar=engine_meta,
        help="Engine used for model evaluation.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        metavar=METAVAR_FILE,
        help="Custom configuration file (TOML). Absolute path, relative path, or filename in the current directory.",
    )

    args = spec_parse_args(parser)
    argv = sys.argv[1:]
    cli_set_log_level = any(token.split("=", 1)[0] in ("--log-level", "--log_level") for token in argv)
    if cli_set_log_level or getattr(args, "verbose", False) or getattr(args, "quiet", False):
        set_log_level(args.log_level)
    else:
        set_log_level(resolve_optix_env_log_level())
    from cli.logo import print_logo

    print_logo()
    if is_root():
        logger.warning(
            "Security Warning: Do not run this tool as root. "
            "Running with elevated privileges may compromise system security. "
            "Use a regular user account."
        )

    logger.info("Starting optix optimizer")
    with logger.contextualize(engine=args.engine):
        if args.config:
            custom_config_path = Path(args.config).expanduser().resolve()
            if not custom_config_path.is_file():
                raise ConfigFileNotFoundError(custom_config_path)

            tomllib = None
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    logger.warning("toml library not available, skipping TOML validation")
            if tomllib is not None:
                try:
                    with open(custom_config_path, "rb") as f:
                        tomllib.load(f)
                except Exception as e:
                    raise InvalidConfigError(custom_config_path, e) from e

            def create_custom_settings():
                from pydantic_settings import SettingsConfigDict

                default_toml_files = list(Settings.model_config.get("toml_file", ()))
                toml_files = default_toml_files + [custom_config_path]

                original_model_config = Settings.model_config

                custom_config_dict = dict(original_model_config)
                custom_config_dict["toml_file"] = toml_files
                custom_config_dict["extra"] = "allow"

                class CustomSettings(Settings):
                    model_config = SettingsConfigDict(**custom_config_dict)

                return CustomSettings()

            register_settings(create_custom_settings)
            logger.info("Using custom config file: {}", custom_config_path)
        settings = get_settings()
        from ..deploy_env import (
            emit_runtime_hints,
            resolve_deploy_context,
            validate_deploy_stack,
        )

        runtime_ctx, deploy_env = resolve_deploy_context()
        emit_runtime_hints(runtime_ctx, engine=args.engine)
        bak_path = None
        if args.backup:
            bak_path = settings.output.joinpath("back_up")
            if not bak_path.exists():
                bak_path.mkdir(parents=True, mode=0o750)
        mode = args.mode
        _validate_optimization_mode_settings(
            mode,
            skip_pso=settings.skip_pso,
            manage_simulator_lifecycle=settings.manage_simulator_lifecycle,
        )
        if mode == "pd_disagg":
            from ..optimizer.pd_disagg import OptixPhaseRunner, PdDisaggOrchestrator

            phase_runner = OptixPhaseRunner(
                settings,
                runtime_ctx=runtime_ctx,
                deploy_env=deploy_env,
                bak_path=bak_path,
                load_breakpoint=args.load_breakpoint,
                top_k=settings.pd_disagg.top_k,
            )
            orchestrator = PdDisaggOrchestrator(settings, run_id or uuid4().hex[:8], phase_runner)
            with logger.contextualize(stage=LogStage.SEARCH.value, engine="pd_disagg"):
                orchestrator.run()
            with logger.contextualize(stage=LogStage.DONE.value, engine="pd_disagg"):
                logger.success("Optimizer finished")
            return
        validate_deploy_stack(
            engine=args.engine,
            benchmark=args.benchmark_policy,
            env=deploy_env,
            ctx=runtime_ctx,
        )
        _simu = _bench = None
        _target_field = []
        if args.engine:
            validate_simulator_policy(args.engine)
            _simu = simulates[args.engine](
                bak_path=bak_path,
                runtime_ctx=runtime_ctx,
                deploy_env=deploy_env,
            )
            _target_field.extend(_simu.data_field)
        if args.benchmark_policy:
            validate_benchmark_policy(args.benchmark_policy)
            _bench = benchmarks[args.benchmark_policy](
                bak_path=bak_path,
                runtime_ctx=runtime_ctx,
                deploy_env=deploy_env,
            )
            _target_field.extend(_bench.data_field)
        _target_field = tuple(_target_field)
        if not _simu:
            raise ValueError("No available simulator object found.")
        if not _bench:
            raise ValueError("No available benchmark object found.")
        if len(_target_field) < 1:
            raise ValueError("No optimization fields were found. ")
        data_storage = DataStorage(settings.data_storage, _simu, _bench)
        scheduler = Scheduler(_simu, _bench, data_storage, bak_path=bak_path, engine=args.engine)
        fine_tune = FineTune(
            ttft_penalty=settings.ttft_penalty,
            tpot_penalty=settings.tpot_penalty,
            target_field=_target_field,
            ttft_slo=settings.ttft_slo,
            tpot_slo=settings.tpot_slo,
            slo_coefficient=settings.slo_coefficient,
            step_size=settings.step_size,
            fine_tune_mode=settings.fine_tune_mode,
        )
        pso = PSOOptimizer(
            scheduler,
            n_particles=settings.n_particles,
            iters=settings.iters,
            target_field=_target_field,
            ttft_penalty=settings.ttft_penalty,
            tpot_penalty=settings.tpot_penalty,
            success_rate_penalty=settings.success_rate_penalty,
            ttft_slo=settings.ttft_slo,
            tpot_slo=settings.tpot_slo,
            success_rate_slo=settings.success_rate_slo,
            generate_speed_target=settings.generate_speed_target,
            load_breakpoint=args.load_breakpoint,
            fine_tune=fine_tune,
            max_fine_tune=settings.max_fine_tune,
            skip_pso=settings.skip_pso,
            use_request_rate_calibration=settings.use_request_rate_calibration,
            manage_simulator_lifecycle=settings.manage_simulator_lifecycle,
            pso_init_kwargs={"ftol": settings.ftol, "ftol_iter": settings.ftol_iter},
        )
        if settings.optimizer_strategy == "agent":
            from ..optimizer.agentic import AgentOptimizer

            AgentOptimizer(pso=pso, agent_config=settings.agent_optimizer).run()
        else:
            with logger.contextualize(stage=LogStage.SEARCH.value):
                pso.run_plugin()
            with logger.contextualize(stage=LogStage.DONE.value):
                logger.success("Optimizer finished")


def _main() -> None:
    run_id = uuid4().hex[:8]
    with logger.contextualize(run_id=run_id, stage=LogStage.INIT.value, engine="-"):
        try:
            _run_optimizer(run_id)
        except OptimizerError as exc:
            logger.error("{}", exc)
            raise SystemExit(1) from None
        except Exception:
            logger.exception("Optimizer aborted")
            raise SystemExit(1) from None


def main() -> None:
    from optix import configure_logger

    configure_logger()
    _main()
