# -------------------------------------------------------------------------
# This file is part of MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to terms and conditions of the Mulan PSL v2.
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
from unittest.mock import MagicMock, patch

from optix.optimizer.health_check import (
    FatalError,
    RetryableError,
    ServiceHookPoint,
    BenchmarkHookPoint,
    HealthCheckContext,
    HealthCheckResult,
    ServiceHealthCheckHook,
    BenchmarkHealthCheckHook,
    ServiceHealthChecks,
    BenchmarkHealthChecks,
)
from optix.config.config import ErrorSeverity, ErrorType


class TestHealthCheckExceptions(unittest.TestCase):
    def test_fatal_error(self):
        error = FatalError("OOM detected")
        self.assertEqual(str(error), "OOM detected")
        self.assertIsInstance(error, Exception)

    def test_retryable_error(self):
        error = RetryableError("Network error")
        self.assertEqual(str(error), "Network error")
        self.assertIsInstance(error, Exception)


class TestEnums(unittest.TestCase):
    def test_error_severity_values(self):
        self.assertEqual(ErrorSeverity.FATAL.value, "fatal")
        self.assertEqual(ErrorSeverity.RETRYABLE.value, "retryable")

    def test_error_type_values(self):
        self.assertEqual(ErrorType.OUT_OF_MEMORY.value, "out_of_memory")
        self.assertEqual(ErrorType.DEVICE_ERROR.value, "device_error")
        self.assertEqual(ErrorType.NETWORK_ERROR.value, "network_error")
        self.assertEqual(ErrorType.IO_ERROR.value, "io_error")


class TestDataClasses(unittest.TestCase):
    def test_health_check_result(self):
        result = HealthCheckResult(is_healthy=True)
        self.assertTrue(result.is_healthy)
        self.assertIsNone(result.error_context)

    def test_health_check_context(self):
        context = HealthCheckContext(
            simulator=MagicMock(),
            benchmark=MagicMock(),
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
            startup=False,
        )
        self.assertEqual(context.current_time, 100.0)
        self.assertEqual(context.elapsed_time, 10.0)
        self.assertFalse(context.startup)


class TestHealthCheckHooks(unittest.TestCase):
    def test_service_hook_register_and_run(self):
        hook = ServiceHealthCheckHook()
        hook.register(
            ServiceHookPoint.STARTUP_POLLING,
            lambda ctx: HealthCheckResult(is_healthy=True),
        )
        result = hook.run(ServiceHookPoint.STARTUP_POLLING, MagicMock())
        self.assertTrue(result.is_healthy)

    def test_benchmark_hook_register_and_run(self):
        hook = BenchmarkHealthCheckHook()
        hook.register(
            BenchmarkHookPoint.RUNTIME_MONITOR,
            lambda ctx: HealthCheckResult(is_healthy=True),
        )
        result = hook.run(BenchmarkHookPoint.RUNTIME_MONITOR, MagicMock())
        self.assertTrue(result.is_healthy)

    def test_hook_with_unhealthy_result(self):
        hook = ServiceHealthCheckHook()
        error_context = MagicMock()
        error_context.severity = ErrorSeverity.FATAL
        error_context.message = "Test error"
        hook.register(
            ServiceHookPoint.STARTUP_POLLING,
            lambda ctx: HealthCheckResult(is_healthy=False, error_context=error_context),
        )
        result = hook.run(ServiceHookPoint.STARTUP_POLLING, MagicMock())
        self.assertFalse(result.is_healthy)

    def test_hook_no_hooks_registered(self):
        hook = ServiceHealthCheckHook()
        result = hook.run(ServiceHookPoint.STARTUP_POLLING, MagicMock())
        self.assertTrue(result.is_healthy)

    def test_hook_with_exception(self):
        """Test handling when a hook function throws an exception"""
        hook = ServiceHealthCheckHook()

        def failing_hook(ctx):
            raise AttributeError("Missing attribute")

        hook.register(ServiceHookPoint.STARTUP_POLLING, failing_hook)
        result = hook.run(ServiceHookPoint.STARTUP_POLLING, MagicMock())

        # It should return an unhealthy result, with exceptions considered fatal errors
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.error_context.error_type, ErrorType.UNKNOWN)
        self.assertEqual(result.error_context.severity, ErrorSeverity.FATAL)
        self.assertIn("AttributeError", result.error_context.message)
        self.assertIn("Missing attribute", result.error_context.message)
        self.assertIn("failing_hook", result.error_context.message)


class TestServiceHealthChecks(unittest.TestCase):
    @patch("optix.optimizer.health_check.get_settings")
    def test_no_error(self, mock_get_settings):
        mock_config = MagicMock()
        mock_config.fatal_patterns = {}
        mock_config.retryable_patterns = {}
        mock_health_check = MagicMock()
        mock_health_check.service_errors = mock_config
        mock_health_check.log_snippet_length = 200
        mock_settings = MagicMock()
        mock_settings.health_check = mock_health_check
        mock_get_settings.return_value = mock_settings
        simulator = MagicMock()
        simulator.get_last_log.return_value = "INFO: Service started"
        context = HealthCheckContext(
            simulator=simulator,
            benchmark=MagicMock(),
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
        )
        result = ServiceHealthChecks.check_log_errors(context)
        self.assertTrue(result.is_healthy)
        simulator.get_last_log.assert_called_once_with(
            number=mock_health_check.log_snippet_length,
            retry=False,
        )

    @patch("optix.optimizer.health_check.get_settings")
    def test_detect_fatal_error(self, mock_get_settings):
        mock_config = MagicMock()
        mock_config.fatal_patterns = {
            ErrorType.OUT_OF_MEMORY: ["out of memory", "OOM"],
            ErrorType.DEVICE_ERROR: ["device error", "NPU error"],
        }
        mock_config.retryable_patterns = {}

        mock_health_check = MagicMock()
        mock_health_check.service_errors = mock_config
        mock_health_check.log_snippet_length = 200

        mock_settings = MagicMock()
        mock_settings.health_check = mock_health_check
        mock_get_settings.return_value = mock_settings

        simulator = MagicMock()
        simulator.get_last_log.return_value = "ERROR: out of memory, cannot allocate 1GB"

        context = HealthCheckContext(
            simulator=simulator,
            benchmark=MagicMock(),
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
        )

        result = ServiceHealthChecks.check_log_errors(context)
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.error_context.error_type, ErrorType.OUT_OF_MEMORY)
        self.assertEqual(result.error_context.severity, ErrorSeverity.FATAL)

    @patch("optix.optimizer.health_check.get_settings")
    def test_detect_retryable_error(self, mock_get_settings):
        mock_config = MagicMock()
        mock_config.fatal_patterns = {}
        mock_config.retryable_patterns = {
            ErrorType.NETWORK_ERROR: ["connection reset", "network unreachable"],
            ErrorType.IO_ERROR: ["file not found", "permission denied"],
        }

        mock_health_check = MagicMock()
        mock_health_check.service_errors = mock_config
        mock_health_check.log_snippet_length = 200

        mock_settings = MagicMock()
        mock_settings.health_check = mock_health_check
        mock_get_settings.return_value = mock_settings

        simulator = MagicMock()
        simulator.get_last_log.return_value = "ERROR: connection reset, network unreachable"

        context = HealthCheckContext(
            simulator=simulator,
            benchmark=MagicMock(),
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
        )

        result = ServiceHealthChecks.check_log_errors(context)
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.error_context.error_type, ErrorType.NETWORK_ERROR)
        self.assertEqual(result.error_context.severity, ErrorSeverity.RETRYABLE)

    def test_no_get_last_log_method(self):
        simulator = MagicMock(spec=[])
        context = HealthCheckContext(
            simulator=simulator,
            benchmark=MagicMock(),
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
        )
        result = ServiceHealthChecks.check_log_errors(context)
        self.assertTrue(result.is_healthy)

    @patch("optix.optimizer.health_check.get_settings")
    def test_empty_log_is_healthy(self, mock_get_settings):
        mock_health_check = MagicMock()
        mock_health_check.service_errors.fatal_patterns = {ErrorType.OUT_OF_MEMORY: ["oom"]}
        mock_health_check.service_errors.retryable_patterns = {}
        mock_health_check.log_snippet_length = 200
        mock_get_settings.return_value = MagicMock(health_check=mock_health_check)

        simulator = MagicMock()
        simulator.get_last_log.return_value = None

        context = HealthCheckContext(
            simulator=simulator,
            benchmark=MagicMock(),
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=1.0,
        )
        result = ServiceHealthChecks.check_log_errors(context)
        self.assertTrue(result.is_healthy)


class TestBenchmarkHealthChecks(unittest.TestCase):
    @patch("optix.optimizer.health_check.get_settings")
    def test_no_error(self, mock_get_settings):
        mock_config = MagicMock()
        mock_config.fatal_patterns = {}
        mock_config.retryable_patterns = {}

        mock_health_check = MagicMock()
        mock_health_check.benchmark_errors = mock_config
        mock_health_check.log_snippet_length = 200

        mock_settings = MagicMock()
        mock_settings.health_check = mock_health_check
        mock_get_settings.return_value = mock_settings

        benchmark = MagicMock()
        benchmark.get_last_log.return_value = "INFO: Benchmark started"

        context = HealthCheckContext(
            simulator=MagicMock(),
            benchmark=benchmark,
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
        )

        result = BenchmarkHealthChecks.check_log_errors(context)
        self.assertTrue(result.is_healthy)
        benchmark.get_last_log.assert_called_once_with(
            number=mock_health_check.log_snippet_length,
            retry=False,
        )

    @patch("optix.optimizer.health_check.get_settings")
    def test_detect_network_error(self, mock_get_settings):
        mock_config = MagicMock()
        mock_config.fatal_patterns = {}
        mock_config.retryable_patterns = {ErrorType.NETWORK_ERROR: ["connection refused", "timeout"]}

        mock_health_check = MagicMock()
        mock_health_check.benchmark_errors = mock_config
        mock_health_check.log_snippet_length = 200

        mock_settings = MagicMock()
        mock_settings.health_check = mock_health_check
        mock_get_settings.return_value = mock_settings

        benchmark = MagicMock()
        benchmark.get_last_log.return_value = "ERROR: connection refused"

        context = HealthCheckContext(
            simulator=MagicMock(),
            benchmark=benchmark,
            scheduler=MagicMock(),
            current_time=100.0,
            elapsed_time=10.0,
        )

        result = BenchmarkHealthChecks.check_log_errors(context)
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.error_context.error_type, ErrorType.NETWORK_ERROR)
        self.assertEqual(result.error_context.severity, ErrorSeverity.RETRYABLE)


if __name__ == "__main__":
    unittest.main()
