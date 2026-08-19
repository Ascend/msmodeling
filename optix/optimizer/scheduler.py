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
from contextlib import contextmanager
from math import isclose
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..common import get_train_sub_path, is_mindie, is_vllm
from ..config.base_config import CONCURRENCYS, FOLDER_LIMIT_SIZE, REAL_EVALUATION, REQUESTRATES
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
from ..optimizer.early_exit import (
    EarlyExitController,
    EarlyExitPhase,
    EarlyExitTriggered,
    MetricsUnavailableError,
    VllmMetricsClient,
)
from ..optimizer.outcome import RunOutcome, RunStatus
from ..optimizer.performance_tunner import PerformanceTuner
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
        engine: Optional[str] = None,
    ):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.engine = engine
        self.bak_path = bak_path
        self.retry_number = retry_number
        self.wait_time = wait_start_time or get_settings().wait_start_time
        self.current_back_path = None
        # Backup phase marker: "pso" / "refine" / "default"; None means no optimization phase (falls back to the old plain-number dirs)
        self.backup_phase = None
        # Iteration index of the current phase; together with backup_phase forms a top-level dir under back_up, e.g. pso_001, refine_002
        self.backup_iter = 0
        self.simulate_run_info = None
        self.performance_index = None
        self._error_info = None
        self.last_outcome: Optional[RunOutcome] = None
        self.run_start_timestamp = None
        self.first_duration = None
        self.del_log = None
        self.current_phase = EarlyExitPhase.EVALUATION
        self.early_exit_controller = None
        self.early_exit_fitness_evaluator: Optional[PerformanceTuner] = None
        self._auto_early_exit_controller = False
        self._early_exit_disabled_depth = 0
        self._metrics_unavailable_warned = False
        self.early_exit_info = None
        self.current_case_id: Optional[str] = None
        self.benchmark_pass = 1
        self._case_sequence = 0
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

    @contextmanager
    def disable_early_exit(self):
        self._early_exit_disabled_depth += 1
        try:
            yield
        finally:
            self._early_exit_disabled_depth -= 1

    @property
    def early_exit_disabled(self) -> bool:
        return self._early_exit_disabled_depth > 0

    @staticmethod
    def _has_fixed_request_rate(params_field: tuple[OptimizerConfigField, ...]) -> bool:
        request_rate_fields = [field for field in params_field or () if field.name in REQUESTRATES]
        if not request_rate_fields:
            return False
        for field in request_rate_fields:
            if field.constant is not None:
                continue
            try:
                if isclose(float(field.min), float(field.max)):
                    continue
            except (TypeError, ValueError):
                return False
            return False
        return True

    def _configure_early_exit_controller(self):
        if self.early_exit_controller is not None and not self._auto_early_exit_controller:
            return
        settings = get_settings()
        config = settings.benchmark_early_exit
        if not config.enabled:
            if self._auto_early_exit_controller:
                self.early_exit_controller = None
                self._auto_early_exit_controller = False
            return
        if self.engine != "vllm":
            if self._auto_early_exit_controller:
                self.early_exit_controller = None
                self._auto_early_exit_controller = False
            logger.warning(
                "Benchmark early exit is enabled but unsupported for engine '{}'; "
                "only the vllm engine exposes the required metrics.",
                self.engine,
            )
            return
        reference = None
        if hasattr(self.data_storage, "get_reference_performance_index"):
            reference = self.data_storage.get_reference_performance_index()
        fitness_evaluator = self.early_exit_fitness_evaluator
        if fitness_evaluator is None:
            fitness_evaluator = PerformanceTuner(
                ttft_penalty=settings.ttft_penalty,
                tpot_penalty=settings.tpot_penalty,
                success_rate_penalty=settings.success_rate_penalty,
                ttft_slo=settings.ttft_slo,
                tpot_slo=settings.tpot_slo,
                success_rate_slo=settings.success_rate_slo,
                generate_speed_target=settings.generate_speed_target,
            )
        max_num_seqs = self._run_parameter_value("MAX_NUM_SEQS")
        max_concurrency = self._run_parameter_value(*CONCURRENCYS)
        sample_sink = self._record_metrics_sample if config.action == "report" else None
        if self.early_exit_controller is not None:
            self.early_exit_controller.reference = reference
            self.early_exit_controller.fitness_evaluator = fitness_evaluator
            update_context = getattr(self.early_exit_controller, "update_runtime_context", None)
            if callable(update_context):
                update_context(max_num_seqs, max_concurrency, sample_sink)
            return
        self.early_exit_controller = EarlyExitController(
            config=config,
            metrics_client=VllmMetricsClient(
                metrics_url=config.metrics_url,
                timeout_seconds=config.timeout_seconds,
            ),
            reference=reference,
            fitness_evaluator=fitness_evaluator,
            max_num_seqs=max_num_seqs,
            max_concurrency=max_concurrency,
            sample_sink=sample_sink,
        )
        self._auto_early_exit_controller = True
        logger.info(
            "Benchmark early exit enabled for engine vllm; action={}, metrics_url={}, metrics_trace={}",
            config.action,
            config.metrics_url,
            "enabled" if sample_sink is not None else "disabled",
        )

    def _run_parameter_value(self, *names: str):
        for field in self.simulate_run_info or ():
            if field.name in names:
                return field.value
        return None

    def _new_case_id(self) -> str:
        allocator = getattr(self.data_storage, "next_case_id", None)
        if callable(allocator):
            case_id = allocator()
            if isinstance(case_id, str):
                return case_id
        self._case_sequence += 1
        return f"case_{self._case_sequence:06d}"

    def _record_metrics_sample(self, sample) -> None:
        saver = getattr(self.data_storage, "save_metrics_sample", None)
        if not callable(saver):
            return
        saver(
            sample,
            case_id=self.current_case_id,
            optimization_phase=self.backup_phase or "default",
            optimization_iteration=self.backup_iter,
            benchmark_phase=self.current_phase.value,
            benchmark_pass=self.benchmark_pass,
        )

    def _reset_early_exit_window(self):
        self._metrics_unavailable_warned = False
        if self.early_exit_controller is not None:
            self.early_exit_controller.reset()

    def _apply_early_exit_decision(self, decision):
        if self.early_exit_info is not None:
            return
        performance_index = decision.performance
        decision_elapsed = None
        if self.run_start_timestamp is not None:
            decision_elapsed = max(0.0, time.time() - self.run_start_timestamp)
        performance_index.would_early_exit = decision.would_early_exit
        performance_index.early_exit = decision.early_exit
        performance_index.early_exit_reason = decision.reason
        performance_index.early_exit_decision_elapsed_seconds = decision_elapsed
        performance_index.result_source = performance_index.result_source or "early_exit_metrics"
        performance_index.usable_as_best = False
        performance_index.reference_generate_speed = decision.reference_generate_speed
        performance_index.observed_generate_speed = decision.observed_generate_speed
        performance_index.reference_score = decision.reference_score
        performance_index.observed_score = decision.observed_score
        performance_index.slo_violations = dict(decision.slo_violations)
        self._attach_warmup_summary(performance_index, finalize=False)
        self.early_exit_info = decision
        self.performance_index = performance_index
        logger.info(
            "Benchmark early exit condition first met at {:.2f}s; action={}",
            decision_elapsed or 0.0,
            "terminate" if decision.early_exit else "report",
        )

    def _check_early_exit(self):
        if self.early_exit_controller is None:
            return
        try:
            if self.early_exit_disabled:
                observer = getattr(self.early_exit_controller, "observe", None)
                if callable(observer):
                    observer(self.current_phase)
                decision = None
            else:
                decision = self.early_exit_controller.check(self.current_phase)
        except MetricsUnavailableError as error:
            if not self._metrics_unavailable_warned:
                logger.warning(
                    "Benchmark early exit is temporarily disabled because metrics are unavailable: {}",
                    error,
                )
                self._metrics_unavailable_warned = True
            else:
                logger.debug("Skip benchmark early exit check because metrics are unavailable: {}", error)
            return
        self._metrics_unavailable_warned = False
        if decision is None:
            return
        self._apply_early_exit_decision(decision)
        if decision.early_exit:
            raise EarlyExitTriggered(decision)

    def _attach_warmup_summary(
        self,
        performance_index: PerformanceIndex,
        *,
        finalize: bool = True,
    ) -> PerformanceIndex:
        performance_index.case_id = self.current_case_id
        if self.early_exit_controller is None or self.current_phase != EarlyExitPhase.EVALUATION:
            return performance_index
        if finalize:
            finalizer = getattr(self.early_exit_controller, "finalize_warmup_on_case_end", None)
            if callable(finalizer):
                finalizer()
        result = getattr(self.early_exit_controller, "warmup_result", None)
        if result is None:
            return performance_index
        performance_index.warmup_end_elapsed_seconds = result.elapsed_seconds
        performance_index.warmup_end_reason = result.reason
        performance_index.warmup_sample_count = result.sample_count
        performance_index.warmup_effective_target = result.effective_target
        performance_index.warmup_load_threshold = result.load_threshold
        performance_index.warmup_running_requests = result.running_requests
        performance_index.warmup_waiting_requests = result.waiting_requests
        performance_index.warmup_load_ratio = result.load_ratio
        performance_index.warmup_forced = result.forced
        return performance_index

    def _attach_metrics_window_reference(self, performance_index: PerformanceIndex) -> PerformanceIndex:
        self._attach_warmup_summary(performance_index)
        if self.early_exit_controller is None:
            return performance_index
        reference_provider = getattr(self.early_exit_controller, "representative_window", None)
        if not callable(reference_provider):
            return performance_index
        reference_window, sample_count = reference_provider()
        if reference_window is None:
            return performance_index
        performance_index.metrics_window_generate_speed = reference_window.generate_speed
        performance_index.metrics_window_time_to_first_token = reference_window.time_to_first_token
        performance_index.metrics_window_time_per_output_token = reference_window.time_per_output_token
        performance_index.metrics_window_success_rate = reference_window.success_rate
        performance_index.metrics_window_sample_count = sample_count
        return performance_index

    def _merge_report_only_early_exit(self, performance_index: PerformanceIndex) -> PerformanceIndex:
        if self.early_exit_info is None or self.early_exit_info.early_exit:
            return performance_index
        performance_index.would_early_exit = self.early_exit_info.would_early_exit
        performance_index.early_exit_reason = self.early_exit_info.reason
        performance_index.early_exit_decision_elapsed_seconds = (
            self.early_exit_info.performance.early_exit_decision_elapsed_seconds
        )
        performance_index.reference_generate_speed = self.early_exit_info.reference_generate_speed
        performance_index.observed_generate_speed = self.early_exit_info.observed_generate_speed
        performance_index.reference_score = self.early_exit_info.reference_score
        performance_index.observed_score = self.early_exit_info.observed_score
        performance_index.slo_violations = dict(self.early_exit_info.slo_violations)
        return performance_index

    @staticmethod
    def _finalize_report_only_time_saving(performance_index: PerformanceIndex, duration: Optional[float]) -> None:
        if not performance_index.would_early_exit or performance_index.early_exit:
            return
        decision_elapsed = performance_index.early_exit_decision_elapsed_seconds
        if duration is None or duration <= 0 or decision_elapsed is None:
            return
        estimated_saved = max(0.0, duration - decision_elapsed)
        performance_index.estimated_time_saved_seconds = estimated_saved
        performance_index.estimated_time_saved_ratio = estimated_saved / duration

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
        """Set the current backup phase and iteration index, used to create pso_N / refine_N / default_N top-level dirs under back_up.

        phase: "pso" / "refine" / "default"; None means no phase prefix (falls back to the old plain-number dirs).
        iter_num: iteration index of the current phase (starting from 1).
        """
        self.backup_phase = phase
        self.backup_iter = iter_num

    def _get_phase_bak_path(self) -> Optional[Path]:
        """Create/return the top-level phase dir under bak_path based on the current phase, e.g. back_up/pso_001.

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
            self._check_early_exit()
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
                self._reset_early_exit_window()
                self.monitoring_status()
                return
            except EarlyExitTriggered as e:
                logger.warning(f"Early exit in run_target_server: {e}")
                raise
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

    def save_result(self, stop_simulator: bool = True, **kwargs):
        """Save the result of this run and clean up processes.
        stop_simulator: when True, stop both simulator and benchmark (default behavior).
        When False, stop only the benchmark and keep the simulator running
        """
        duration = None
        if self.run_start_timestamp:
            duration = time.time() - self.run_start_timestamp
            if not self.first_duration:
                self.first_duration = duration
        self._finalize_report_only_time_saving(self.performance_index, duration)
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
        if stop_simulator:
            self.stop_target_server(del_log)
        else:
            # Stop only the benchmark, keep the simulator running
            self.benchmark.stop(del_log=del_log)

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
        self.benchmark_pass += 1
        self.benchmark.run(tuple(self.simulate_run_info))
        self.current_phase = EarlyExitPhase.EVALUATION
        self._reset_early_exit_window()
        self.monitoring_status()
        time.sleep(1)
        performance_index = self._attach_metrics_window_reference(self.benchmark.get_performance_index())
        self.performance_index = self._merge_report_only_early_exit(performance_index)

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
            self.current_case_id = self._new_case_id()
            self.benchmark_pass = 1
            logger.opt(lazy=True).trace("run param info {}", lambda: {v.name: v.value for v in self.simulate_run_info})
            self._error_info = None
            self.last_outcome = None
            self.del_log = True
            self.performance_index = PerformanceIndex(case_id=self.current_case_id)
            self.early_exit_info = None
            self._configure_early_exit_controller()
            try:
                self.update_data_field(self.simulate_run_info)
                fixed_request_rate = with_request_rate and self._has_fixed_request_rate(tuple(self.simulate_run_info))
                if with_request_rate and not fixed_request_rate:
                    self.current_phase = EarlyExitPhase.CALIBRATION
                else:
                    self.current_phase = EarlyExitPhase.EVALUATION
                self.run_target_server(params, params_field)
                time.sleep(1)
                performance_index = self._attach_metrics_window_reference(self.benchmark.get_performance_index())
                self.performance_index = self._merge_report_only_early_exit(performance_index)
                if with_request_rate:
                    self._apply_request_rate_second_run(params_field)
            except EarlyExitTriggered as e:
                self._error_info = None
                self.del_log = False
                self.performance_index = e.decision.performance
                self.stop_target_server(False)
            except OptimizerError:
                raise
            except Exception as e:
                self._error_info = e
                self.del_log = False
            self._attach_warmup_summary(self.performance_index)
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

    def rerun_benchmark_only(
        self,
        params: np.ndarray,
        params_field: tuple[OptimizerConfigField],
        decode_context: Optional[DecodeContext] = None,
        *,
        with_request_rate: bool = False,
    ) -> PerformanceIndex:
        """
        Reuse the currently running simulator and rerun only the benchmark.
        """
        with logger.contextualize(stage=LogStage.BENCH_RUN.value):
            self.run_start_timestamp = time.time()
            self.set_back_up_path()
            self.simulate_run_info = map_param_with_value(params, params_field, decode_context)
            self.current_case_id = self._new_case_id()
            self.benchmark_pass = 1
            logger.debug(
                "rerun benchmark start (reuse running simulator) param info {}",
                {v.name: v.value for v in self.simulate_run_info},
            )
            self._error_info = None
            self.last_outcome = None
            self.del_log = True
            self.performance_index = PerformanceIndex(case_id=self.current_case_id)
            self.early_exit_info = None
            self._configure_early_exit_controller()
            try:
                # Stop any lingering benchmark process, then re-run it against the live simulator.
                self.benchmark.stop()
                # Update only the benchmark-side data field and command; the simulator is untouched.
                if isinstance(self.benchmark, SupportsDataField):
                    self.benchmark.data_field = self.simulate_run_info
                    self.benchmark.update_command()
                if isinstance(self.benchmark, SupportsPrepare):
                    self.benchmark.prepare()
                self.benchmark.run(tuple(self.simulate_run_info))
                time.sleep(1)
                self.current_phase = EarlyExitPhase.EVALUATION
                self._reset_early_exit_window()
                # monitoring_status runs health-check hooks on both the live simulator and benchmark.
                self.monitoring_status()
                time.sleep(1)
                performance_index = self._attach_metrics_window_reference(self.benchmark.get_performance_index())
                self.performance_index = self._merge_report_only_early_exit(performance_index)
                if with_request_rate:
                    self._apply_request_rate_second_run(params_field)
            except OptimizerError:
                raise
            except Exception as e:
                logger.error(
                    "Failed rerun benchmark. bak path: {}. error: {}, simulator log: {}, benchmark log: {}",
                    self.simulator.bak_path,
                    e,
                    self.simulator.run_log,
                    self.benchmark.run_log,
                )
                self._error_info = e
                self.del_log = False
                # The benchmark may have produced results before the server errored; try to fetch
                # the performance index to avoid losing valid data.
                try:
                    self.performance_index = self._attach_metrics_window_reference(
                        self.benchmark.get_performance_index()
                    )
                    logger.info("Successfully retrieved performance index despite server error.")
                except Exception as perf_err:
                    logger.warning("Failed to get performance index after server error: {}", perf_err)
            self._attach_warmup_summary(self.performance_index)
            status = RunStatus.FAILED if self._error_info else RunStatus.SUCCESS
            duration = time.time() - self.run_start_timestamp if self.run_start_timestamp else None
            error_type = type(self._error_info).__name__ if self._error_info else "-"
            logger.debug(
                "rerun benchmark finished status={} duration={:.2f}s error_type={}",
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
