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
import argparse
import os
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
from ..logging import LogStage, format_evaluation_failure
from ..optimizer.errors import (
    BaselineRunError,
    ConfigFileNotFoundError,
    InvalidConfigError,
    NoFeasibleSolutionError,
    OptimizerError,
)
from ..optimizer.outcome import RunStatus
from ..optimizer.performance_tunner import PerformanceTuner
from ..optimizer.register import benchmarks, simulates
from ..optimizer.utils import get_required_field_from_json, is_root

MAX_ITER_NUM = 200


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
        max_fine_tune: int = 10,
        use_request_rate_calibration: bool = True,
        **kwargs,
    ):
        from ..config.config import PsoOptions, default_support_field

        super().__init__(**kwargs)
        self.scheduler = scheduler
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
        self.use_request_rate_calibration = use_request_rate_calibration
        self._iteration = 0  # op_func call count, used for balanced strategy inter-iteration direction alternation
        self._seen_params = {}

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
        self.scheduler.save_result(fitness=inf)
        return True

    def op_func(self, x) -> np.ndarray:
        n_particles = x.shape[0]
        current_iteration = self._iteration
        self._iteration += 1
        generate_speed = []
        scheduler_run = (
            self.scheduler.run_with_request_rate if self.use_request_rate_calibration else self.scheduler.run
        )
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
                self.scheduler.save_result(fitness=_fitness)
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

    def refine_optimization_candidates(self, best_results: pd.DataFrame):
        from ..optimizer.experience_fine_tunning import StopFineTune

        _record_params = [self.default_run_param]
        _record_res = [self.default_res]
        _record_fitness = [self.default_fitness]
        for _, _pso_info in best_results.iterrows():
            _target_field = self.get_target_field_from_case_data(_pso_info)
            for _field in _target_field:
                if _field.name in REQUESTRATES:
                    _field.value = _field.find_available_value(_field.value * 2)
            params = field_to_param(_target_field)
            try:
                _res = self.scheduler.run(params, self.target_field)
                if self.scheduler.last_outcome and self.scheduler.last_outcome.status == RunStatus.FAILED:
                    logger.error(
                        "Runtime exception. error: {}, please check.",
                        format_evaluation_failure(self.scheduler, self.scheduler.error_info),
                    )
                    _fitness = inf
                    self.scheduler.save_result(fitness=_fitness)
                    continue
                _fitness = self.minimum_algorithm(_res)
            except Exception as e:
                logger.error(
                    "Runtime exception. error: {}, please check.",
                    format_evaluation_failure(self.scheduler, e),
                )
                _fitness = inf
                self.scheduler.save_result(fitness=_fitness)
                continue
            self.scheduler.save_result(fitness=_fitness)
            _record_params.append(params)
            _record_res.append(_res)
            _record_fitness.append(_fitness)
            self.fine_tune.reset_history()
            for _ in range(self.max_fine_tune):
                try:
                    simulate_run_info = self.fine_tune.fine_tune_with_concurrency_and_request_rate(params, _res)
                except ValueError as e:
                    logger.error("Failed in fine-tuning parameter. error: {}", e)
                    break
                except StopFineTune:
                    break
                params = field_to_param(simulate_run_info)
                if self.params_in_records(params, _record_params):
                    break
                try:
                    _res = self.scheduler.run(params, self.target_field)
                    if self.scheduler.last_outcome and self.scheduler.last_outcome.status == RunStatus.FAILED:
                        logger.error(
                            "Runtime exception. error: {}, please check.",
                            format_evaluation_failure(self.scheduler, self.scheduler.error_info),
                        )
                        _fitness = inf
                        self.scheduler.save_result(fitness=_fitness)
                        break
                    _fitness = self.minimum_algorithm(_res)
                except Exception as e:
                    logger.error(
                        "Runtime exception. error: {}, please check.",
                        format_evaluation_failure(self.scheduler, e),
                    )
                    _fitness = inf
                    self.scheduler.save_result(fitness=_fitness)
                    break
                self.scheduler.save_result(fitness=_fitness)
                _record_params.append(params)
                _record_res.append(_res)
                _record_fitness.append(_fitness)
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
        del_log = self.scheduler.del_log if self.scheduler.del_log is not None else False
        self.scheduler.stop_target_server(del_log)
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
            return self.scheduler.run(self.default_run_param, baseline_target_field)
        finally:
            self._restore_search_data_field(search_target_field, simulator_names, benchmark_names)

    def prepare_plugin(self):
        from ..config.config import get_settings
        from ..config.model_config import MindieModelConfig
        from ..optimizer.plugins.benchmark import AisBench
        from ..optimizer.plugins.simulate import Simulator

        with logger.contextualize(stage=LogStage.BASELINE.value):
            if isinstance(self.scheduler.simulator, Simulator):
                settings = get_settings()
                mc = None
                if is_mindie() and settings.theory_guided_enable:
                    mc = MindieModelConfig(self.scheduler.simulator.config.config_path)
                for _, _field in enumerate(self.target_field):
                    if _field.config_position.startswith("BackendConfig"):
                        _field.value = get_required_field_from_json(
                            self.scheduler.simulator.default_config, _field.config_position
                        )
                    elif _field.config_position == "env":
                        _field.value = os.getenv(_field.name, _field.value)
                self.default_res = self._run_baseline_preserving_search_space()
                self._raise_if_baseline_failed()
                if self.default_res.generate_speed:
                    self.gen_speed_target = 10 * self.default_res.generate_speed
                self.default_fitness = self.minimum_algorithm(self.default_res)
                self.scheduler.save_result(fitness=self.default_fitness)
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
                self.default_res = self._run_baseline_preserving_search_space()
                self._raise_if_baseline_failed()
                self.default_fitness = self.minimum_algorithm(self.default_res)
                self.scheduler.save_result(fitness=self.default_fitness)

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
        from ..optimizer.global_best_custom import CustomGlobalBestPSO

        self.prepare_plugin()
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
                cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
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


@contextmanager
def adapter_target_field(pso_optimizer: PSOOptimizer):
    _bak_target_field = pso_optimizer.target_field
    target_field = deepcopy(pso_optimizer.target_field)
    fix_concurrency = pso_optimizer.use_request_rate_calibration
    for _field in target_field:
        if _field.name in CONCURRENCYS and _field.constant is None and fix_concurrency:
            # true 模式：CONCURRENCY 固定为 max，由 run_with_request_rate 内部校准请求速率
            _field.constant = _field.value = _field.convert_dtype(_field.max)
        elif _field.name in REQUESTRATES and _field.constant is None:
            # 两种模式均将 REQUESTRATE 固定为 max：
            #   true  → run_with_request_rate 需要最大速率上限
            #   false → scheduler.run 在最大请求速率下对每个候选并发做单次压测
            _field.constant = _field.convert_dtype(_field.max)
            _field.value = None
        elif _field.constant and _field.constant != _field.value:
            _field.value = _field.constant
    pso_optimizer.target_field = target_field
    yield
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


def _run_optimizer() -> None:
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

    register_ori_functions()
    load_general_plugins()

    parser = argparse.ArgumentParser(
        description="optix - Service Parameter Optimizer for LLM inference performance tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-lb",
        "--load_breakpoint",
        default=False,
        action="store_true",
        help="Continue from where the last optimization was aborted.",
    )
    parser.add_argument(
        "--backup",
        default=False,
        action="store_true",
        help="Whether to back up data.",
    )
    parser.add_argument(
        "-b",
        "--benchmark_policy",
        default=DEFAULT_BENCHMARK_POLICY,
        choices=list(_benchmarks.keys()),
        help="Whether to use custom performance indicators.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        default="vllm",
        choices=list(_simulates.keys()),
        help="The engine used for model evaluation.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="Path to custom configuration file (TOML format). "
        "Supports absolute path, relative path, or filename in current directory.",
    )

    args = parser.parse_args()
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
        bak_path = None
        if args.backup:
            bak_path = settings.output.joinpath("bak")
            if not bak_path.exists():
                bak_path.mkdir(parents=True, mode=0o750)
        _simu = _bench = None
        _target_field = []
        if args.engine:
            validate_simulator_policy(args.engine)
            _simu = simulates[args.engine](bak_path=bak_path)
            _target_field.extend(_simu.data_field)
        if args.benchmark_policy:
            validate_benchmark_policy(args.benchmark_policy)
            _bench = benchmarks[args.benchmark_policy](bak_path=bak_path)
            _target_field.extend(_bench.data_field)
        _target_field = tuple(_target_field)
        if not _simu:
            raise ValueError("No available simulator object found.")
        if not _bench:
            raise ValueError("No available benchmark object found.")
        if len(_target_field) < 1:
            raise ValueError("No optimization fields were found. ")
        data_storage = DataStorage(settings.data_storage, _simu, _bench)
        scheduler = Scheduler(_simu, _bench, data_storage, bak_path=bak_path)
        fine_tune = FineTune(
            ttft_penalty=settings.ttft_penalty,
            tpot_penalty=settings.tpot_penalty,
            target_field=_target_field,
            ttft_slo=settings.ttft_slo,
            tpot_slo=settings.tpot_slo,
            slo_coefficient=settings.slo_coefficient,
            step_size=settings.step_size,
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
            use_request_rate_calibration=settings.use_request_rate_calibration,
            pso_init_kwargs={"ftol": settings.ftol, "ftol_iter": settings.ftol_iter},
        )
        with logger.contextualize(stage=LogStage.SEARCH.value):
            pso.run_plugin()
        with logger.contextualize(stage=LogStage.DONE.value):
            logger.success("Optimizer finished")


def _main() -> None:
    run_id = uuid4().hex[:8]
    with logger.contextualize(run_id=run_id, stage=LogStage.INIT.value, engine="-"):
        try:
            _run_optimizer()
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
