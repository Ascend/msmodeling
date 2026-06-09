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
from pathlib import Path
from typing import Tuple, Optional
from math import isclose
import numpy as np
from loguru import logger

from ..common import get_train_sub_path, is_mindie, is_vllm
from ..config.base_config import REAL_EVALUATION
from ..config.config import (
    get_settings,
    DecodeContext,
    PerformanceIndex,
    OptimizerConfigField,
    map_param_with_value,
    ErrorSeverity,
)
from ..config.base_config import FOLDER_LIMIT_SIZE, REQUESTRATES
from ..config.constant import Stage
from ..optimizer.plugins.simulate import Simulator
from ..optimizer.store import DataStorage
from ..optimizer.utils import get_folder_size
from ..optimizer.health_check import (
    ErrorContext,
    BenchmarkHealthCheckHook,
    ServiceHealthCheckHook,
    ServiceHookPoint,
    BenchmarkHookPoint,
    HealthCheckContext,
    service_health_checks_hooks,
    benchmark_health_checks_hooks,
    FatalError,
    RetryableError,
)


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
        self.simulate_run_info = None
        self.performance_index = None
        self.error_info = None
        self.run_start_timestamp = None
        self.first_duration = None
        self.del_log = None
        self.service_checks = ServiceHealthCheckHook()
        self.benchmark_checks = BenchmarkHealthCheckHook()
        self._register_default_checks()

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
            logger.error(f"Fatal error: {error_context.message}")
            raise FatalError(error_context.message)
        else:  # RETRYABLE
            logger.warning(f"Retryable error: {error_context.message}")
            raise RetryableError(error_context.message)

    def set_back_up_path(self):
        if self.bak_path:
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                self.current_back_path = get_train_sub_path(self.bak_path)
                self.simulator.bak_path = self.current_back_path
                self.benchmark.bak_path = self.current_back_path

    def wait_simulate(self):
        logger.debug("wait run simulator")
        start_time = time.time()
        for _ in range(self.wait_time):
            time.sleep(1)
            elapsed = time.time() - start_time
            context = self._create_check_context(elapsed)
            result = self.service_checks.run(ServiceHookPoint.STARTUP_POLLING, context)
            if result.is_healthy:
                if hasattr(self.simulator, "check_success") and self.simulator.check_success():
                    logger.info(f"Successfully started the {self.simulator.process} process.")
                    return
                elif hasattr(self.simulator, "health"):
                    res = self.simulator.health()
                    if res.stage == Stage.running:
                        logger.info(f"Successfully started the {self.simulator.process} process.")
                        return
                    elif res.stage == Stage.start:
                        if int(elapsed) % 60 == 0:
                            logger.warning(
                                f"Check the service status at {elapsed} seconds. status: {res.stage}. info: {res.info}"
                            )
                        continue
                    elif res.stage in (Stage.stop, Stage.error):
                        logger.error(f"Failed to start the {self.simulator.process} process.")
                        raise subprocess.SubprocessError(
                            f"Failed in run simulator. status: {res.stage}. info: {res.info}"
                        )
                    else:
                        logger.warning(f" Unknown Status. status: {res.stage}. info: {res.info}")
                else:
                    raise RuntimeError(
                        f"No actionable method found. the expected is check_success or health. "
                        f"simulator: {type(self.simulator)}"
                    )
            else:
                self._handle_error(result.error_context)
        raise TimeoutError(self.wait_time)

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        if hasattr(self.benchmark, "prepare"):
            self.benchmark.prepare()
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()

    def backup(self):
        self.simulator.backup()
        self.benchmark.backup()

    def monitoring_status(self):
        logger.debug("monitor status")
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
            if hasattr(self.simulator, "check_success"):
                if is_mindie() or is_vllm():
                    if self.simulator.process.poll() is not None:
                        raise subprocess.SubprocessError(
                            f"Failed in run simulator. return code: {self.simulator.process.returncode}."
                        )
                if self.benchmark.check_success():
                    return
            if hasattr(self.simulator, "health"):
                if not isinstance(self.simulator, Simulator):
                    res = self.simulator.health()
                    if res.stage != Stage.running:
                        raise subprocess.SubprocessError(
                            f"Failed in run simulator. error: {res.stage} info: {res.info}."
                        )
                res = self.benchmark.health()
                if res.stage != Stage.running:
                    return
            if self.run_start_timestamp and self.first_duration:
                _duration = time.time() - self.run_start_timestamp
                if _duration > 2 * self.first_duration:
                    logger.warning("The current runtime is more than twice the duration of the first run.")
            time.sleep(1)

        raise TimeoutError(f"{get_settings().particles_time_out}")

    def run_target_server(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        """
        1. Start mindie simulation
        2. Start benchmark test
        3. Check mindie status, check benchmark status
        """
        for attempt in range(self.retry_number):
            try:
                self.run_simulate(params, params_field)
                time.sleep(1)
                self.benchmark.run(tuple(self.simulate_run_info))
                time.sleep(1)
                self.monitoring_status()
                return
            except FatalError as e:
                logger.error(
                    f"Fatal error in run_target_server (attempt {attempt + 1}/{self.retry_number}): {e}, \n"
                    f"simulator log: {self.simulator.run_log}, \n"
                    f"log last info: {self.simulator.get_last_log()}"
                )
                self.stop_target_server(False)
                raise
            except RetryableError as e:
                logger.warning(
                    f"Retryable error in run_target_server (attempt {attempt + 1}/{self.retry_number}): {e}, \n"
                    f"simulator log: {self.simulator.run_log}, \n"
                    f"log last info: {self.simulator.get_last_log()}"
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
        self.stop_target_server()

    def update_data_field(self, params_field: Tuple[OptimizerConfigField]):
        if hasattr(self.simulator, "data_field"):
            self.simulator.data_field = params_field
        if hasattr(self.simulator, "update_command"):
            self.simulator.update_command()
        if hasattr(self.benchmark, "data_field"):
            self.benchmark.data_field = params_field
        if hasattr(self.benchmark, "update_command"):
            self.benchmark.update_command()

    def run(
        self,
        params: np.ndarray,
        params_field: Tuple[OptimizerConfigField],
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
        self.run_start_timestamp = time.time()
        logger.debug("Start run in scheduler.")
        self.set_back_up_path()
        self.simulate_run_info = map_param_with_value(params, params_field, decode_context)
        logger.info("run param info {}", {v.name: v.value for v in self.simulate_run_info})
        self.error_info = None
        self.del_log = True
        self.performance_index = PerformanceIndex()
        try:
            self.update_data_field(self.simulate_run_info)
            self.run_target_server(params, params_field)
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(
                f"Failed running. bak path: {self.simulator.bak_path}. error {e}"
                f"simulator log {self.simulator.run_log}, benchmark log {self.benchmark.run_log}"
            )
            self.error_info = e
            self.del_log = False
        return self.performance_index

    def run_with_request_rate(
        self,
        params: np.ndarray,
        params_field: Tuple[OptimizerConfigField],
        decode_context: Optional[DecodeContext] = None,
    ) -> PerformanceIndex:
        """
        Run the service: first run at max concurrency to get request rate,
        then run based on concurrency and request rate, using the last run as the evaluation result.
        params: 1D array whose values correspond to mindie related configurations.
        """
        self.run_start_timestamp = time.time()
        self.set_back_up_path()
        self.simulate_run_info = map_param_with_value(params, params_field, decode_context)
        logger.info("run param info {}", {v.name: v.value for v in self.simulate_run_info})
        self.error_info = None
        self.del_log = True
        self.performance_index = PerformanceIndex()
        try:
            self.update_data_field(self.simulate_run_info)
            self.run_target_server(params, params_field)
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
            self.benchmark.stop()
            need_second_run = False
            for _field in self.simulate_run_info:
                if _field.name in REQUESTRATES:
                    if not isclose(_field.min, _field.max):
                        _field.value = _field.find_available_value(self.performance_index.throughput * 1.05)
                        need_second_run = True
            if not need_second_run:
                logger.info("REQUESTRATE is fixed (min == max), skipping second run.")
            else:
                logger.info(
                    "second run param info {}",
                    {v.name: v.value for v in self.simulate_run_info},
                )
                if hasattr(self.benchmark, "data_field"):
                    self.benchmark.data_field = params_field
                self.benchmark.update_command()
                try:
                    if hasattr(self.benchmark, "prepare"):
                        self.benchmark.prepare()
                    self.benchmark.run(tuple(self.simulate_run_info))
                except Exception as e:
                    logger.error(f"Failed in Benchmark Running. error: {e}, benchmark log {self.benchmark.run_log}")
                    raise e
                try:
                    self.monitoring_status()
                except Exception as e:
                    logger.error(
                        f"Failed in monitoring status. error: {e}, simulator log {self.simulator.run_log}, "
                        f"benchmark log {self.benchmark.run_log}"
                    )
                    raise e
                time.sleep(1)
                self.performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(
                f"Failed running. bak path: {self.simulator.bak_path}. error {e}"
                f"simulator log {self.simulator.run_log}, benchmark log {self.benchmark.run_log}"
            )
            self.error_info = e
            self.del_log = False
        return self.performance_index
