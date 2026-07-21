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
import subprocess  # nosec B404
import time
from math import isclose
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..common import get_train_sub_path, is_mindie, is_vllm
from ..config.base_config import FOLDER_LIMIT_SIZE, REAL_EVALUATION, REQUESTRATES
from ..config.config import (
    DecodeContext,
    ErrorSeverity,
    OptimizerConfigField,
    PerformanceIndex,
    get_settings,
    map_param_with_value,
)
from ..config.constant import Stage
from ..logging import LogStage, format_subprocess_failure
from ..optimizer.errors import OptimizerError
from ..optimizer.health_check import (
    BenchmarkHealthCheckHook,
    BenchmarkHookPoint,
    ErrorContext,
    FatalError,
    HealthCheckContext,
    RetryableError,
    ServiceHealthCheckHook,
    ServiceHookPoint,
    benchmark_health_checks_hooks,
    service_health_checks_hooks,
)
from ..optimizer.outcome import RunOutcome, RunStatus
from ..optimizer.plugins.simulate import Simulator
from ..optimizer.protocols import (
    SupportsCheckSuccess,
    SupportsDataField,
    SupportsHealth,
    SupportsPrepare,
)
from ..optimizer.store import DataStorage
from ..optimizer.utils import get_folder_size


class Scheduler:
    def __init__(
        self,
        simulator,
        benchmark,
        data_storage: DataStorage,
        bak_path: Optional[Path] = None,
        retry_number: int = 3,
        wait_start_time: Optional[int] = None,
    ):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.bak_path = bak_path
        self.retry_number = retry_number
        self.wait_time = wait_start_time or get_settings().wait_start_time
        self.current_back_path = None
        # Backup phase marker: "pso" / "refine" / "default"; None means no optimization phase (falls back to the old plain-number dirs)
        self.backup_phase = None
        # Iteration index of the current phase; together with backup_phase forms a top-level dir under trial_logs, e.g. pso_001, refine_002
        self.backup_iter = 0
        self.simulate_run_info = None
        self.performance_index = None
        self._error_info = None
        self.last_outcome: Optional[RunOutcome] = None
        self.run_start_timestamp = None
        self.first_duration = None
        self.del_log = None
        self.service_checks = ServiceHealthCheckHook()
        self.benchmark_checks = BenchmarkHealthCheckHook()
        self._register_default_checks()

    @property
    def error_info(self):
        if self.last_outcome is not None and self.last_outcome.error_context is not None:
            return self.last_outcome.error_context
        return self._error_info

    @error_info.setter
    def error_info(self, value):
        self._error_info = value

    def _register_default_checks(self):
        """Register default health checks (can be overridden by subclasses)"""
        for name, func, priority in service_health_checks_hooks:
            self.service_checks.register(name, func, priority=priority)
        for name, func, priority in benchmark_health_checks_hooks:
            self.benchmark_checks.register(name, func, priority=priority)

    def _create_check_context(self, elapsed: float) -> HealthCheckContext:
        """Create check context"""
        return HealthCheckContext(
            simulator=self.simulator,
            benchmark=self.benchmark,
            scheduler=self,
            current_time=time.time(),
            elapsed_time=elapsed,
        )

    def _handle_error(self, error_context: ErrorContext) -> None:
        """Raise different exceptions based on error type"""
        if error_context.severity == ErrorSeverity.FATAL:
            raise FatalError(error_context.message)
        raise RetryableError(error_context.message)

    def _simulator_failure_message(self, return_code: int | None = None) -> str:
        command = list(getattr(self.simulator, "command", None) or [])
        log_path = getattr(self.simulator, "run_log", None)
        if return_code is None and hasattr(self.simulator, "process") and self.simulator.process is not None:
            return_code = self.simulator.process.returncode
        log_tail = None
        get_last_log = getattr(self.simulator, "get_last_log", None)
        if callable(get_last_log):
            log_tail = get_last_log(10)
        return format_subprocess_failure(command, return_code, log_path, log_tail=log_tail)

    def set_backup_phase(self, phase: Optional[str], iter_num: int):
        """Set the current backup phase and iteration index, used to create pso_N / refine_N / default_N top-level dirs under trial_logs.

        phase: "pso" / "refine" / "default"; None means no phase prefix (falls back to the old plain-number dirs).
        iter_num: iteration index of the current phase (starting from 1).
        """
        self.backup_phase = phase
        self.backup_iter = iter_num

    def _get_phase_bak_path(self) -> Optional[Path]:
        """Create/return the top-level phase dir under bak_path based on the current phase, e.g. trial_logs/pso_001.

        Returns bak_path itself when no phase is set, preserving the original behavior.
        """
        if not self.backup_phase:
            return self.bak_path
        phase_dir = self.bak_path.joinpath(f"{self.backup_phase}_{self.backup_iter:03d}")
        if not phase_dir.exists():
            phase_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        return phase_dir

    def set_back_up_path(self):
        if self.bak_path:
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                self.current_back_path = get_train_sub_path(self._get_phase_bak_path())
                self.simulator.bak_path = self.current_back_path
                self.benchmark.bak_path = self.current_back_path

    def wait_simulate(self):
        start_time = time.time()
        for _ in range(self.wait_time):
            time.sleep(1)
            elapsed = time.time() - start_time
            context = self._create_check_context(elapsed)
            result = self.service_checks.run(ServiceHookPoint.STARTUP_POLLING, context)
            if result.is_healthy:
                started = False
                if isinstance(self.simulator, SupportsCheckSuccess) and self.simulator.check_success():
                    started = True
                elif isinstance(self.simulator, SupportsHealth):
                    res = self.simulator.health()
                    if res.stage == Stage.running:
                        started = True
                    elif res.stage == Stage.start:
                        if int(elapsed) % 60 == 0:
                            logger.warning(
                                f"Check the service status at {elapsed} seconds. status: {res.stage}. info: {res.info}"
                            )
                        continue
                    elif res.stage in (Stage.stop, Stage.error):
                        return_code = None
                        if hasattr(self.simulator, "process") and self.simulator.process is not None:
                            return_code = self.simulator.process.returncode
                        logger.debug(
                            "Simulator subprocess exited during startup stage={} return_code={}",
                            res.stage,
                            return_code,
                        )
                        raise subprocess.SubprocessError(self._simulator_failure_message(return_code))
                    else:
                        logger.warning(f" Unknown Status. status: {res.stage}. info: {res.info}")
                else:
                    raise RuntimeError(
                        f"No actionable method found. the expected is check_success or health. "
                        f"simulator: {type(self.simulator)}"
                    )
                if started:
                    logger.success(f"Successfully started the {self.simulator.process} process.")
                    return
            else:
                self._handle_error(result.error_context)
        raise TimeoutError(self.wait_time)

    def run_simulate(self, params: np.ndarray, params_field: tuple[OptimizerConfigField]):
        if isinstance(self.benchmark, SupportsPrepare):
            self.benchmark.prepare()
        logger.debug("starting simulator subprocess")
        self.simulator.run(tuple(self.simulate_run_info))
        logger.debug("waiting for simulator startup")
        self.wait_simulate()
        logger.debug("simulator startup complete")

    def backup(self):
        self.simulator.backup()
        self.benchmark.backup()

    def monitoring_status(self):
        start_time = time.time()
        for _ in range(get_settings().particles_time_out):
            elapsed = time.time() - start_time
            context = self._create_check_context(elapsed)
            service_result = self.service_checks.run(ServiceHookPoint.RUNTIME_MONITOR, context)
            if not service_result.is_healthy:
                self._handle_error(service_result.error_context)
            benchmark_result = self.benchmark_checks.run(BenchmarkHookPoint.RUNTIME_MONITOR, context)
            if not benchmark_result.is_healthy:
                self._handle_error(benchmark_result.error_context)
            if isinstance(self.simulator, SupportsCheckSuccess):
                if is_mindie() or is_vllm():
                    if self.simulator.process.poll() is not None:
                        logger.debug(
                            "Simulator subprocess exited during runtime monitor return_code={}",
                            self.simulator.process.returncode,
                        )
                        raise subprocess.SubprocessError(self._simulator_failure_message())
                if self.benchmark.check_success():
                    return
            if isinstance(self.simulator, SupportsHealth):
                if not isinstance(self.simulator, Simulator):
                    res = self.simulator.health()
                    if res.stage != Stage.running:
                        logger.debug(
                            "Simulator health non-running during runtime monitor stage={}",
                            res.stage,
                        )
                        raise subprocess.SubprocessError(self._simulator_failure_message())
                res = self.benchmark.health()
                if res.stage != Stage.running:
                    return
            if self.run_start_timestamp and self.first_duration:
                _duration = time.time() - self.run_start_timestamp
                if _duration > 2 * self.first_duration:
                    logger.warning("The current runtime is more than twice the duration of the first run.")
            time.sleep(1)

        raise TimeoutError(f"{get_settings().particles_time_out}")

    def run_target_server(self, params: np.ndarray, params_field: tuple[OptimizerConfigField]):
        """
        1. Start mindie simulation
        2. Start benchmark test
        3. Check mindie status, check benchmark status
        """
        for attempt in range(self.retry_number):
            try:
                self.run_simulate(params, params_field)
                time.sleep(1)
                logger.debug("starting benchmark subprocess")
                self.benchmark.run(tuple(self.simulate_run_info))
                logger.debug("benchmark subprocess started")
                time.sleep(1)
                self.monitoring_status()
                return
            except FatalError as e:
                logger.debug(
                    "Fatal error in run_target_server (attempt {}/{}): {}, simulator log: {}, tail: {}",
                    attempt + 1,
                    self.retry_number,
                    e,
                    self.simulator.run_log,
                    self.simulator.get_last_log(),
                )
                self.stop_target_server(False)
                raise
            except RetryableError as e:
                logger.debug(
                    "Retryable error in run_target_server (attempt {}/{}): {}, simulator log: {}, tail: {}",
                    attempt + 1,
                    self.retry_number,
                    e,
                    self.simulator.run_log,
                    self.simulator.get_last_log(),
                )
                self.stop_target_server(False)
                continue
        raise ValueError(f"Failed in run_target_server after {self.retry_number} attempts")

    def stop_target_server(self, del_log: bool = False):
        self.simulator.stop(del_log)
        self.benchmark.stop(del_log)

    def save_result(self, **kwargs):
        duration = None
        if self.run_start_timestamp:
            duration = time.time() - self.run_start_timestamp
            if not self.first_duration:
                self.first_duration = duration
        real_evaluation = True
        if REAL_EVALUATION in kwargs:
            real_evaluation = kwargs.pop(REAL_EVALUATION)
        self.data_storage.save(
            self.performance_index,
            tuple(self.simulate_run_info),
            error=self.error_info,
            backup=self.current_back_path,
            duration=duration,
            real_evaluation=real_evaluation,
            **kwargs,
        )
        if self.bak_path:
            self.backup()
        del_log = self.del_log if self.del_log is not None else False
        self.stop_target_server(del_log)

    def update_data_field(self, params_field: tuple[OptimizerConfigField]):
        if isinstance(self.simulator, SupportsDataField):
            self.simulator.data_field = params_field
            self.simulator.update_command()
        if isinstance(self.benchmark, SupportsDataField):
            self.benchmark.data_field = params_field
            self.benchmark.update_command()

    def _apply_request_rate_second_run(self, params_field: tuple[OptimizerConfigField]) -> None:
        self.benchmark.stop()
        need_second_run = False
        for _field in self.simulate_run_info:
            if _field.name in REQUESTRATES:
                if not isclose(_field.min, _field.max):
                    _field.value = _field.find_available_value(self.performance_index.throughput * 1.05)
                    need_second_run = True
        if not need_second_run:
            logger.info("REQUESTRATE is fixed (min == max), skipping second run.")
            return
        logger.info(
            "second run param info {}",
            {v.name: v.value for v in self.simulate_run_info},
        )
        if isinstance(self.benchmark, SupportsDataField):
            self.benchmark.data_field = params_field
        self.benchmark.update_command()
        if isinstance(self.benchmark, SupportsPrepare):
            self.benchmark.prepare()
        self.benchmark.run(tuple(self.simulate_run_info))
        self.monitoring_status()
        time.sleep(1)
        self.performance_index = self.benchmark.get_performance_index()

    def _run_evaluation(
        self,
        params: np.ndarray,
        params_field: tuple[OptimizerConfigField],
        decode_context: Optional[DecodeContext],
        *,
        with_request_rate: bool = False,
    ) -> PerformanceIndex:
        with logger.contextualize(stage=LogStage.EVALUATE.value):
            self.run_start_timestamp = time.time()
            logger.debug("evaluation start param_count={} values={}", len(params), params.tolist())
            self.set_back_up_path()
            self.simulate_run_info = map_param_with_value(params, params_field, decode_context)
            logger.opt(lazy=True).trace("run param info {}", lambda: {v.name: v.value for v in self.simulate_run_info})
            self._error_info = None
            self.last_outcome = None
            self.del_log = True
            self.performance_index = PerformanceIndex()
            try:
                self.update_data_field(self.simulate_run_info)
                self.run_target_server(params, params_field)
                time.sleep(1)
                self.performance_index = self.benchmark.get_performance_index()
                if with_request_rate:
                    self._apply_request_rate_second_run(params_field)
            except OptimizerError:
                raise
            except Exception as e:
                self._error_info = e
                self.del_log = False
            status = RunStatus.FAILED if self._error_info else RunStatus.SUCCESS
            duration = time.time() - self.run_start_timestamp if self.run_start_timestamp else None
            error_type = type(self._error_info).__name__ if self._error_info else "-"
            logger.debug(
                "evaluation finished status={} duration={:.2f}s error_type={}",
                status.value,
                duration or 0.0,
                error_type,
            )
            self.last_outcome = RunOutcome(
                status=status,
                performance_index=self.performance_index,
                error_context=self._error_info,
            )
            return self.performance_index

    def run(
        self,
        params: np.ndarray,
        params_field: tuple[OptimizerConfigField],
        decode_context: Optional[DecodeContext] = None,
    ) -> PerformanceIndex:
        """
        1. Start mindie simulation
        2. Start benchmark test
        3. Get benchmark test results
        4. Stop mindie simulation
        5. Return benchmark test results
        params: 1D array whose values correspond to mindie related configurations.
        """
        return self._run_evaluation(params, params_field, decode_context, with_request_rate=False)

    def run_with_request_rate(
        self,
        params: np.ndarray,
        params_field: tuple[OptimizerConfigField],
        decode_context: Optional[DecodeContext] = None,
    ) -> PerformanceIndex:
        """
        Run the service: first run at max concurrency to get request rate,
        then run based on concurrency and request rate, using the last run as the evaluation result.
        params: 1D array whose values correspond to mindie related configurations.
        """
        return self._run_evaluation(params, params_field, decode_context, with_request_rate=True)
