"""Regression tests for optix structured loguru configuration."""

from __future__ import annotations

from io import StringIO
from math import inf
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from loguru import logger

from optix.config.config import ErrorSeverity, ErrorType
from optix.logging import (
    LOG_FORMAT,
    LOG_FORMAT_DEBUG,
    LOG_FORMAT_INFO,
    LogStage,
    configure_logger,
    format_command,
    format_subprocess_failure,
    format_subprocess_start,
    read_log_tail,
    resolve_log_format,
    resolve_log_level,
)
from optix.optimizer.health_check import ErrorContext, FatalError
from optix.optimizer.scheduler import Scheduler


@pytest.fixture(autouse=True)
def _reset_logger() -> None:
    configure_logger.cache_clear()
    logger.remove()
    yield
    configure_logger.cache_clear()
    logger.remove()


def _capture_logs(**add_kwargs) -> tuple[StringIO, int]:
    buffer = StringIO()
    capture_format = add_kwargs.pop("format", LOG_FORMAT)
    capture_level = add_kwargs.pop("level", "TRACE")
    handler_id = logger.add(buffer, format=capture_format, level=capture_level, **add_kwargs)
    return buffer, handler_id


def test_configure_logger_extra_fields_in_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIX_LOG_LEVEL", "INFO")
    configure_logger()

    buffer, handler_id = _capture_logs()
    logger.bind(run_id="run-abc", stage=LogStage.INIT.value).info("optimizer starting")
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "run-abc" in output
    assert LogStage.INIT.value in output
    assert "optimizer starting" in output


def test_optix_log_level_env_overrides_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIX_LOG_LEVEL", "warning")
    monkeypatch.setenv("MODELEVALSTATE_LEVEL", "DEBUG")
    assert resolve_log_level() == "WARNING"


def test_contextualize_child_inherits_run_id() -> None:
    buffer, handler_id = _capture_logs()

    with (
        logger.contextualize(run_id="parent-run", stage="-", engine="-"),
        logger.contextualize(stage=LogStage.SEARCH.value),
    ):
        logger.info("search iteration")
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "parent-run" in output
    assert LogStage.SEARCH.value in output


def test_run_context_propagates_through_global_logger() -> None:
    from tests.regression.optix.test_optimizer.test_pso_optimizer import _make_pso_optimizer

    run_id = "propagate-run-99"
    optimizer = _make_pso_optimizer(n_particles=1, iters=1)
    optimizer.scheduler.run_with_request_rate.side_effect = RuntimeError("bench failed")

    buffer, handler_id = _capture_logs()
    with logger.contextualize(run_id=run_id, engine="vllm", stage=LogStage.INIT.value):
        optimizer.op_func(np.array([[50.0, 25000.0]]))
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert run_id in output
    assert "Evaluation failed, fitness=inf" in output


def test_configure_logger_preserves_custom_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIX_LOG_LEVEL", "INFO")

    custom_buffer = StringIO()
    custom_id = logger.add(custom_buffer, format="{message}", level="INFO")

    configure_logger()

    logger.info("custom sink still receives logs")
    custom_output = custom_buffer.getvalue()

    logger.remove(custom_id)
    assert "custom sink still receives logs" in custom_output


def test_handle_error_raises_without_logging() -> None:
    scheduler = Scheduler(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    error_context = ErrorContext(
        error_type=ErrorType.OUT_OF_MEMORY,
        severity=ErrorSeverity.FATAL,
        message="OOM detected",
    )

    records: list[str] = []

    def _collect(message) -> None:
        records.append(str(message))

    handler_id = logger.add(_collect, level="TRACE")

    with pytest.raises(FatalError, match="OOM detected"):
        scheduler._handle_error(error_context)

    logger.remove(handler_id)
    assert records == []


def test_op_func_particle_failure_emits_single_warning() -> None:
    from tests.regression.optix.test_optimizer.test_pso_optimizer import _make_pso_optimizer

    optimizer = _make_pso_optimizer(n_particles=1, iters=1)
    optimizer.scheduler.run_with_request_rate.side_effect = RuntimeError("bench failed")

    records: list[tuple[str, str]] = []

    def _sink(message) -> None:
        record = message.record
        records.append((record["level"].name, record["message"]))

    handler_id = logger.add(_sink, level="TRACE")
    result = optimizer.op_func(np.array([[50.0, 25000.0]]))
    logger.remove(handler_id)

    assert result.shape == (1,)
    assert result[0] == inf
    warnings = [message for level, message in records if level == "WARNING"]
    assert len(warnings) == 1
    assert "Evaluation failed, fitness=inf" in warnings[0]
    assert all(level != "ERROR" for level, _ in records)


def test_exception_records_backtrace_in_debug_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIX_LOG_LEVEL", "DEBUG")
    configure_logger()

    buffer, handler_id = _capture_logs(diagnose=True, backtrace=True)
    try:
        raise ValueError("optimizer aborted")
    except ValueError:
        logger.exception("Optimizer aborted")
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "Optimizer aborted" in output
    assert "ValueError" in output
    assert "Traceback" in output or "optimizer aborted" in output


def test_main_logs_unexpected_error_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import patch

    monkeypatch.setenv("OPTIX_LOG_LEVEL", "DEBUG")
    configure_logger()

    buffer, handler_id = _capture_logs(diagnose=True, backtrace=True)
    with patch("optix.optimizer.optimizer._run_optimizer", side_effect=ValueError("boom")):
        from optix.optimizer.optimizer import main

        with pytest.raises(SystemExit) as exc_info:
            main()
    logger.remove(handler_id)

    assert exc_info.value.code == 1
    output = buffer.getvalue()
    assert output.count("Traceback (most recent call last):") == 1
    assert "Optimizer aborted" in output


def test_debug_format_includes_file_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIX_LOG_LEVEL", "DEBUG")
    configure_logger()

    buffer, handler_id = _capture_logs(format=LOG_FORMAT_DEBUG)
    logger.debug("debug probe")
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "test_logging.py:" in output


def test_trace_format_includes_file_line() -> None:
    buffer, handler_id = _capture_logs(format=LOG_FORMAT_DEBUG, level="TRACE")
    logger.trace("trace probe")
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "test_logging.py:" in output


def test_info_format_omits_file_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIX_LOG_LEVEL", "INFO")
    configure_logger()

    buffer, handler_id = _capture_logs(format=LOG_FORMAT_INFO)
    logger.bind(run_id="run-xyz", stage=LogStage.INIT.value).info("info probe")
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "info probe" in output
    assert "test_logging.py:" not in output


def test_resolve_log_format_debug_and_trace() -> None:
    assert resolve_log_format("DEBUG") == LOG_FORMAT_DEBUG
    assert resolve_log_format("trace") == LOG_FORMAT_DEBUG
    assert resolve_log_format("INFO") == LOG_FORMAT_INFO


def test_format_subprocess_start_multiline_readable() -> None:
    command = ["/usr/bin/vllm", "serve", "model_path", "--port", "8080"]
    log_path = "/tmp/ms_serviceparam_optimizer__abc"
    message = format_subprocess_start(command, log_path, pid=12345)

    assert message.startswith("Starting service subprocess")
    assert "  command: /usr/bin/vllm serve model_path --port 8080" in message
    assert f"  log: {log_path}" in message
    assert "  pid: 12345" in message


def test_format_command_uses_shlex_join() -> None:
    command = ["echo", "hello world"]
    assert format_command(command) == "echo 'hello world'"


def test_format_subprocess_failure_no_duplicate_command() -> None:
    command = ["vllm", "serve", "model_path"]
    log_path = "/tmp/run.log"
    message = format_subprocess_failure(command, 1, log_path, log_tail="error: port bind failed")

    assert message.count("vllm serve model_path") == 1
    assert "exit=1" in message
    assert f"  log: {log_path}" in message
    assert "  log tail:" in message
    assert "    error: port bind failed" in message
    assert "  hint:" not in message


def test_read_log_tail_returns_last_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    tail = read_log_tail(log_file, lines=2)

    assert tail == "line2\nline3"


def test_read_log_tail_large_file(tmp_path: Path) -> None:
    log_file = tmp_path / "large.log"
    with log_file.open("w", encoding="utf-8") as handle:
        for i in range(50_000):
            handle.write(f"line{i}\n")

    tail = read_log_tail(log_file, lines=2)

    assert tail == "line49998\nline49999"


def test_op_func_emits_warning_on_scheduler_internal_failure() -> None:
    from unittest.mock import patch

    from tests.regression.optix.test_optimizer.test_pso_optimizer import _make_pso_optimizer

    scheduler = Scheduler(MagicMock(), MagicMock(), MagicMock())
    scheduler.run_target_server = MagicMock(side_effect=RuntimeError("bench down"))

    optimizer = _make_pso_optimizer(scheduler=scheduler, n_particles=1, iters=1)
    params = np.array([50.0, 25000.0])
    fields = optimizer.target_field

    records: list[tuple[str, str]] = []

    def _sink(message) -> None:
        record = message.record
        records.append((record["level"].name, record["message"]))

    handler_id = logger.add(_sink, level="TRACE")
    with patch("optix.optimizer.scheduler.map_param_with_value") as mock_map:
        mock_map.return_value = fields
        result = optimizer.op_func(np.array([params]))
    logger.remove(handler_id)

    assert result.shape == (1,)
    assert result[0] == inf
    warnings = [message for level, message in records if level == "WARNING"]
    assert len(warnings) == 1
    assert "Evaluation failed, fitness=inf" in warnings[0]
    assert "bench down" in warnings[0]
    assert all(level != "ERROR" for level, _ in records)


def test_op_func_reraises_optimizer_error() -> None:
    from unittest.mock import patch

    from optix.optimizer.errors import BenchmarkResultError
    from tests.regression.optix.test_optimizer.test_pso_optimizer import _make_pso_optimizer

    scheduler = Scheduler(MagicMock(), MagicMock(), MagicMock())
    scheduler.run = MagicMock(
        side_effect=BenchmarkResultError("csv not unique"),
    )

    optimizer = _make_pso_optimizer(
        scheduler=scheduler,
        n_particles=1,
        iters=1,
        use_request_rate_calibration=False,
    )
    params = np.array([50.0, 25000.0])

    with patch("optix.optimizer.scheduler.map_param_with_value") as mock_map:
        mock_map.return_value = optimizer.target_field
        with pytest.raises(BenchmarkResultError, match="not unique"):
            optimizer.op_func(np.array([params]))


def test_validate_benchmark_policy_missing_executable() -> None:
    from unittest.mock import patch

    from optix.optimizer.errors import BenchmarkUnavailableError
    from optix.optimizer.register import DEFAULT_BENCHMARK_POLICY, register_ori_functions, validate_benchmark_policy

    register_ori_functions()
    with patch("optix.optimizer.register.shutil.which", return_value=None):
        with pytest.raises(BenchmarkUnavailableError, match="-b ais_bench") as exc_info:
            validate_benchmark_policy("ais_bench")
    assert f"default -b is {DEFAULT_BENCHMARK_POLICY}" in str(exc_info.value)


def test_validate_simulator_policy_missing_vllm() -> None:
    from unittest.mock import patch

    from optix.optimizer.errors import SimulatorUnavailableError
    from optix.optimizer.register import DEFAULT_SIMULATOR_POLICY, register_ori_functions, validate_simulator_policy

    register_ori_functions()
    with patch("optix.optimizer.register.shutil.which", return_value=None):
        with pytest.raises(SimulatorUnavailableError, match="-e vllm") as exc_info:
            validate_simulator_policy("vllm")
    assert f"default -e is {DEFAULT_SIMULATOR_POLICY}" in str(exc_info.value)


def test_validate_custom_benchmark_with_required_executable() -> None:
    from unittest.mock import patch

    from optix.optimizer.errors import BenchmarkUnavailableError
    from optix.optimizer.interfaces.benchmark import BenchmarkInterface
    from optix.optimizer.register import benchmarks, register_benchmarks, validate_benchmark_policy

    class CustomBench(BenchmarkInterface):
        required_executable = "custom_bench_tool"

        def update_command(self) -> None:
            pass

        def get_performance_index(self):
            from optix.config.config import PerformanceIndex

            return PerformanceIndex()

    policy = "_test_custom_bench_policy"
    register_benchmarks(policy, CustomBench)
    try:
        with patch("optix.optimizer.register.shutil.which", return_value=None):
            with pytest.raises(BenchmarkUnavailableError, match="-b " + policy):
                validate_benchmark_policy(policy)
    finally:
        benchmarks.pop(policy, None)


def test_validate_policy_skips_when_required_executable_none() -> None:
    from unittest.mock import patch

    from optix.optimizer.interfaces.benchmark import BenchmarkInterface
    from optix.optimizer.register import benchmarks, register_benchmarks, validate_benchmark_policy

    class NoExecBench(BenchmarkInterface):
        required_executable = None

        def update_command(self) -> None:
            pass

        def get_performance_index(self):
            from optix.config.config import PerformanceIndex

            return PerformanceIndex()

    policy = "_test_no_exec_bench_policy"
    register_benchmarks(policy, NoExecBench)
    try:
        with patch("optix.optimizer.register.shutil.which", return_value=None) as mock_which:
            validate_benchmark_policy(policy)
            mock_which.assert_not_called()
    finally:
        benchmarks.pop(policy, None)


def test_main_domain_error_preserves_run_id_and_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from optix.optimizer.errors import ConfigFileNotFoundError

    monkeypatch.setenv("OPTIX_LOG_LEVEL", "INFO")
    configure_logger()

    buffer, handler_id = _capture_logs(format=LOG_FORMAT_INFO)
    missing = Path("/tmp/optix-missing-config.toml")
    with patch(
        "optix.optimizer.register.register_ori_functions",
        side_effect=ConfigFileNotFoundError(missing),
    ):
        from optix.optimizer.optimizer import main

        with pytest.raises(SystemExit) as exc_info:
            main()
    logger.remove(handler_id)

    assert exc_info.value.code == 1
    output = buffer.getvalue()
    assert " | - | " not in output.split("Custom config file not found")[0][-40:]
    assert LogStage.INIT.value in output
    assert "Custom config file not found" in output
    run_id_match = [part for part in output.split(" | ") if len(part) == 8 and part.isalnum()]
    assert run_id_match, f"expected 8-char run_id in log output: {output!r}"


def test_baseline_run_error_message_structure() -> None:
    import subprocess
    from unittest.mock import MagicMock

    from optix.optimizer.errors import BaselineRunError

    scheduler = MagicMock()
    scheduler.simulator.command = ["vllm", "serve", "model_path"]
    scheduler.simulator.run_log = "/tmp/ms_serviceparam_optimizer__nvppydc"
    scheduler.simulator.process.returncode = 1
    scheduler.simulator.get_last_log = MagicMock(return_value="bind failed on port")
    scheduler.error_info = subprocess.SubprocessError("subprocess failed")

    err = BaselineRunError.from_scheduler(scheduler)
    message = str(err)

    assert "exit=1" in message
    assert "  log: /tmp/ms_serviceparam_optimizer__nvppydc" in message
    assert "vllm serve model_path" in message
    assert "  log tail:" in message
    assert "bind failed on port" in message
    assert "Failed in run simulator" not in message
    assert "Failed to start the default service" not in message


def test_baseline_run_error_reuses_formatted_subprocess_message() -> None:
    import subprocess
    from unittest.mock import MagicMock

    from optix.optimizer.errors import BaselineRunError

    formatted = (
        "Service subprocess failed (exit=2)\n"
        "  command: vllm serve model_path\n"
        "  log: /tmp/run.log\n"
        "  log tail:\n"
        "    error: invalid int value: 'port'"
    )
    scheduler = MagicMock()
    scheduler.simulator.command = ["vllm", "serve", "model_path"]
    scheduler.simulator.run_log = "/tmp/run.log"
    scheduler.simulator.process.returncode = 2
    scheduler.error_info = subprocess.SubprocessError(formatted)

    err = BaselineRunError.from_scheduler(scheduler)
    message = str(err)

    assert message == formatted
    scheduler.simulator.get_last_log.assert_not_called()


def test_format_evaluation_failure_reads_tail_before_stop() -> None:
    from unittest.mock import MagicMock

    from optix.logging import format_evaluation_failure

    simulator = MagicMock()
    simulator.command = ["vllm", "serve", "model_path", "--port", "port"]
    simulator.run_log = "/tmp/ms_serviceparam_optimizer__abc"
    simulator.process.returncode = 2
    simulator.get_last_log.return_value = "vllm serve: error: invalid int value: 'port'"

    scheduler = MagicMock()
    scheduler.simulator = simulator

    message = format_evaluation_failure(scheduler, TimeoutError("startup timed out"))

    assert "invalid int value: 'port'" in message
    assert "  log tail:" in message
    simulator.get_last_log.assert_called_once_with(10)


def test_main_domain_error_logs_message_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from optix.optimizer.errors import ConfigFileNotFoundError

    monkeypatch.setenv("OPTIX_LOG_LEVEL", "DEBUG")
    configure_logger()

    buffer, handler_id = _capture_logs(diagnose=True, backtrace=True)
    missing = Path("/tmp/optix-missing-config.toml")
    with patch("optix.optimizer.optimizer._run_optimizer", side_effect=ConfigFileNotFoundError(missing)):
        from optix.optimizer.optimizer import main

        with pytest.raises(SystemExit) as exc_info:
            main()
    logger.remove(handler_id)

    assert exc_info.value.code == 1
    output = buffer.getvalue()
    assert "Custom config file not found" in output
    assert "Traceback (most recent call last):" not in output
