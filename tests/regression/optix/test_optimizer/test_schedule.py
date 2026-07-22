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
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from optix.config.base_config import FOLDER_LIMIT_SIZE
from optix.config.config import (
    ErrorSeverity,
    ErrorType,
    OptimizerConfigField,
    PerformanceIndex,
)
from optix.config.constant import ProcessState, Stage
from optix.optimizer.health_check import (
    BenchmarkHookPoint,
    ErrorContext,
    FatalError,
    RetryableError,
    ServiceHookPoint,
)
from optix.optimizer.scheduler import Scheduler


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.bak_path = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage, self.bak_path)
        # Initialize simulate_run_info to avoid repeated setup in each test method
        self.scheduler.simulate_run_info = ()

    @patch("optix.optimizer.utils.get_folder_size")
    def test_set_back_up_path_folder_size_exceeds_limit(self, mock_get_folder_size):
        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE + 1
        self.scheduler.set_back_up_path()

    @patch("optix.optimizer.utils.get_folder_size")
    @patch("optix.common.get_train_sub_path")
    def test_set_back_up_path_folder_size_within_limit(self, mock_get_train_sub_path, mock_get_folder_size):
        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE - 1
        mock_get_train_sub_path.return_value = "sub_path"
        self.scheduler.set_back_up_path()

    def test_backup_phase_defaults_to_none(self):
        """Default state: no phase set, _get_phase_bak_path returns bak_path unchanged"""
        self.assertIsNone(self.scheduler.backup_phase)
        self.assertEqual(self.scheduler.backup_iter, 0)
        self.assertIs(self.scheduler._get_phase_bak_path(), self.bak_path)

    def test_set_backup_phase_updates_state(self):
        """set_backup_phase records phase and iteration for later path building"""
        self.scheduler.set_backup_phase("pso", 3)
        self.assertEqual(self.scheduler.backup_phase, "pso")
        self.assertEqual(self.scheduler.backup_iter, 3)

    def test_get_phase_bak_path_builds_phase_dir(self):
        """_get_phase_bak_path creates a <phase>_<iter> subdir under bak_path"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            self.scheduler.bak_path = Path(tmp)
            self.scheduler.set_backup_phase("refine", 2)
            phase_dir = self.scheduler._get_phase_bak_path()
            self.assertEqual(phase_dir, Path(tmp) / "refine_002")
            self.assertTrue(phase_dir.is_dir())

    @patch("optix.optimizer.scheduler.get_train_sub_path")
    @patch("optix.optimizer.scheduler.get_folder_size")
    def test_set_back_up_path_uses_phase_dir(self, mock_get_folder_size, mock_get_train_sub_path):
        """set_back_up_path routes through the phase dir when a phase is active"""
        import tempfile
        from pathlib import Path

        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE - 1
        mock_get_train_sub_path.side_effect = lambda p: p
        with tempfile.TemporaryDirectory() as tmp:
            self.scheduler.bak_path = Path(tmp)
            self.scheduler.set_backup_phase("pso", 1)
            self.scheduler.set_back_up_path()
            mock_get_train_sub_path.assert_called_once_with(Path(tmp) / "pso_001")

    @patch("time.sleep", return_value=None)
    def test_wait_simulate_success(self, mock_sleep):
        self.simulator.health = MagicMock(return_value=ProcessState(stage=Stage.running))
        self.scheduler.wait_simulate()

    @patch("time.sleep", return_value=None)
    def test_wait_simulate_timeout(self, mock_sleep):
        self.simulator.health = MagicMock(return_value=ProcessState(stage=Stage.error))
        with self.assertRaises(Exception):
            self.scheduler.wait_simulate()

    @patch("time.sleep", return_value=None)
    def test_run_target_server_benchmark_error(self, _):
        self.benchmark.run.side_effect = Exception("Benchmark error")
        with self.assertRaises(Exception):
            self.scheduler.run_target_server(np.array([1, 2, 3]), ("field1", "field2"))

    @patch("time.sleep", return_value=None)
    def test_run_target_server_monitoring_error(self, _):
        self.scheduler.monitoring_status = MagicMock(side_effect=Exception("Monitoring error"))
        with self.assertRaises(Exception):
            self.scheduler.run_target_server(np.array([1, 2, 3]), ("field1", "field2"))

    def test_health_check_initialization(self):
        """Test health check hook initialization"""
        self.assertIsNotNone(self.scheduler.service_checks)
        self.assertIsNotNone(self.scheduler.benchmark_checks)
        self.assertIsInstance(self.scheduler.service_checks._hooks, dict)
        self.assertIsInstance(self.scheduler.benchmark_checks._hooks, dict)

    def test_register_default_checks(self):
        """Test default health check registration"""
        self.assertIn(ServiceHookPoint.STARTUP_POLLING, self.scheduler.service_checks._hooks)
        self.assertIn(ServiceHookPoint.RUNTIME_MONITOR, self.scheduler.service_checks._hooks)
        self.assertIn(BenchmarkHookPoint.RUNTIME_MONITOR, self.scheduler.benchmark_checks._hooks)

    def test_create_check_context(self):
        """Test health check context creation"""
        elapsed = 10.0
        context = self.scheduler._create_check_context(elapsed)
        self.assertEqual(context.simulator, self.simulator)
        self.assertEqual(context.benchmark, self.benchmark)
        self.assertEqual(context.scheduler, self.scheduler)
        self.assertEqual(context.elapsed_time, elapsed)
        self.assertFalse(context.startup)

    def test_handle_fatal_error(self):
        """Test fatal error handling (raises FatalError)"""
        error_context = ErrorContext(
            error_type=ErrorType.OUT_OF_MEMORY,
            severity=ErrorSeverity.FATAL,
            message="OOM detected",
        )
        with self.assertRaises(FatalError) as cm:
            self.scheduler._handle_error(error_context)
        self.assertEqual(str(cm.exception), "OOM detected")

    def test_handle_retryable_error(self):
        """Test retryable error handling (raises RetryableError)"""
        error_context = ErrorContext(
            error_type=ErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.RETRYABLE,
            message="Network error",
        )
        with self.assertRaises(RetryableError) as cm:
            self.scheduler._handle_error(error_context)
        self.assertEqual(str(cm.exception), "Network error")

    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.Scheduler.wait_simulate")
    def test_run_target_server_fatal_no_retry(self, mock_wait, _):
        """Test fatal error is raised immediately without retry"""
        mock_wait.side_effect = FatalError("OOM error")
        with self.assertRaises(FatalError):
            self.scheduler.run_target_server(np.array([1, 2, 3]), ("field1", "field2", "field3"))
        self.assertEqual(mock_wait.call_count, 1)

    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.Scheduler.monitoring_status")
    @patch("optix.optimizer.scheduler.Scheduler.wait_simulate")
    def test_run_target_server_retryable_with_retry(self, mock_wait, _, mock_sleep):
        """Test retryable error triggers retry"""
        mock_wait.side_effect = [
            RetryableError("Network error"),
            RetryableError("IO error"),
            None,
        ]
        self.simulator.health.return_value = MagicMock(stage=Stage.running)
        self.scheduler.wait_time = 1

        self.scheduler.run_target_server(np.array([1, 2, 3]), ("field1", "field2", "field3"))
        self.assertEqual(mock_wait.call_count, 3)


class TestSchedulerRunMethods(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(
            simulator=self.simulator,
            benchmark=self.benchmark,
            data_storage=self.data_storage,
        )

        self.params = np.array([1.0, 2.0, 3.0])
        self.field1 = OptimizerConfigField(name="param1", value=1.0, min=0.0, max=5.0)
        self.field2 = OptimizerConfigField(name="param2", value=2.0, min=0.0, max=5.0)
        self.field3 = OptimizerConfigField(name="param3", value=3.0, min=0.0, max=5.0)
        self.params_field = (self.field1, self.field2, self.field3)

        self.performance_index = PerformanceIndex()
        self.performance_index.throughput = 100.0
        self.benchmark.get_performance_index.return_value = self.performance_index

    @patch("time.time")
    def test_run_with_fixed_request_rate(self, mock_time):
        """Test run_with_request_rate behavior with fixed request rate"""
        mock_time.return_value = 1000.0

        req_rate_field = OptimizerConfigField(name="REQUESTRATE", value=50.0, min=50.0, max=50.0)
        params_field_with_fixed_req_rate = self.params_field + (req_rate_field,)

        self.scheduler.run_with_request_rate(self.params, params_field_with_fixed_req_rate)

        self.assertEqual(req_rate_field.value, 50.0)
        self.assertEqual(req_rate_field.min, 50.0)
        self.assertEqual(req_rate_field.max, 50.0)

    @patch("time.time")
    @patch("optix.optimizer.scheduler.logger")
    def test_run_logging(self, mock_logger, mock_time):
        """Test logging in run method"""
        mock_time.return_value = 1000.0

        self.scheduler.run(self.params, self.params_field)

    @patch("time.time")
    def test_run_returns_performance_index(self, mock_time):
        """Test run method returns PerformanceIndex"""
        mock_time.return_value = 1000.0
        result = self.scheduler.run(self.params, self.params_field)
        assert isinstance(result, PerformanceIndex)

    @patch("time.time")
    def test_run_handles_exception(self, mock_time):
        """Test run method handles exception gracefully"""
        mock_time.return_value = 1000.0
        self.scheduler.run_target_server = MagicMock(side_effect=RuntimeError("fail"))
        result = self.scheduler.run(self.params, self.params_field)
        assert self.scheduler.error_info is not None
        assert isinstance(result, PerformanceIndex)

    @patch("time.time")
    def test_save_result(self, mock_time):
        """Test save_result method"""
        mock_time.return_value = 1000.0
        self.scheduler.run_start_timestamp = 999.0
        self.scheduler.performance_index = PerformanceIndex()
        self.scheduler.simulate_run_info = self.params_field
        self.scheduler.error_info = None
        self.scheduler.current_back_path = None
        self.scheduler.save_result()
        self.data_storage.save.assert_called_once()

    @patch("time.time")
    def test_save_result_passes_del_log_flag(self, mock_time):
        mock_time.return_value = 1000.0
        self.scheduler.run_start_timestamp = 999.0
        self.scheduler.performance_index = PerformanceIndex()
        self.scheduler.simulate_run_info = self.params_field
        self.scheduler.del_log = True
        self.scheduler.save_result()
        self.simulator.stop.assert_called_with(True)
        self.benchmark.stop.assert_called_with(True)

    @patch("time.time")
    def test_save_result_with_first_duration(self, mock_time):
        """Test save_result sets first_duration on first call"""
        mock_time.return_value = 1010.0
        self.scheduler.run_start_timestamp = 1000.0
        self.scheduler.first_duration = None
        self.scheduler.performance_index = PerformanceIndex()
        self.scheduler.simulate_run_info = self.params_field
        self.scheduler.error_info = None
        self.scheduler.current_back_path = None
        self.scheduler.save_result()
        assert self.scheduler.first_duration is not None

    def test_stop_target_server(self):
        """Test stop_target_server delegates to simulator and benchmark"""
        self.scheduler.stop_target_server(del_log=True)
        self.simulator.stop.assert_called_once_with(True)
        self.benchmark.stop.assert_called_once_with(True)

    def test_update_data_field(self):
        """Test update_data_field updates simulator and benchmark"""
        self.simulator.data_field = None
        self.simulator.update_command = MagicMock()
        self.benchmark.data_field = None
        self.benchmark.update_command = MagicMock()
        self.scheduler.update_data_field(self.params_field)
        assert self.simulator.data_field == self.params_field
        assert self.benchmark.data_field == self.params_field

    def test_backup(self):
        """Test backup delegates to simulator and benchmark"""
        self.scheduler.backup()
        # pylint: disable=no-member
        self.simulator.backup.assert_called_once()
        self.benchmark.backup.assert_called_once()
        # pylint: enable=no-member


class TestSchedulerRunWithRequestRate(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(
            simulator=self.simulator,
            benchmark=self.benchmark,
            data_storage=self.data_storage,
        )
        self.params = np.array([1.0, 2.0])
        self.params_field = (
            OptimizerConfigField(name="max_batch_size", value=50.0, min=10, max=100, dtype="int"),
            OptimizerConfigField(name="REQUESTRATE", value=5.0, min=1.0, max=100.0, dtype="float"),
        )

    @patch("time.time")
    @patch("time.sleep")
    def test_run_with_request_rate_second_run(self, mock_sleep, mock_time):
        """Test run_with_request_rate triggers second run when REQUESTRATE not fixed"""
        mock_time.return_value = 1000.0
        perf = PerformanceIndex(throughput=4.0, generate_speed=100)
        self.benchmark.get_performance_index.return_value = perf
        self.benchmark.check_success.return_value = True
        self.simulator.check_success = MagicMock(return_value=True)

        result = self.scheduler.run_with_request_rate(self.params, self.params_field)
        assert isinstance(result, PerformanceIndex)
        # benchmark.run should be called twice (first run + second run)
        assert self.benchmark.run.call_count == 2, f"Expected 2 calls but got {self.benchmark.run.call_count}"

    @patch("time.time")
    @patch("time.sleep")
    def test_run_with_request_rate_fixed_rate_no_second_run(self, mock_sleep, mock_time):
        """Test run_with_request_rate skips second run when REQUESTRATE is fixed"""
        mock_time.return_value = 1000.0
        params_field_fixed = (
            OptimizerConfigField(name="max_batch_size", value=50.0, min=10, max=100, dtype="int"),
            OptimizerConfigField(
                name="REQUESTRATE",
                value=5.0,
                min=5.0,
                max=5.0,
                dtype="float",
                constant=5.0,
            ),
        )
        perf = PerformanceIndex(throughput=4.0, generate_speed=100)
        self.benchmark.get_performance_index.return_value = perf
        self.benchmark.check_success.return_value = True
        self.simulator.check_success = MagicMock(return_value=True)

        result = self.scheduler.run_with_request_rate(self.params, params_field_fixed)
        assert isinstance(result, PerformanceIndex)

    @patch("time.time")
    @patch("time.sleep")
    def test_run_with_request_rate_exception_sets_error(self, mock_sleep, mock_time):
        """Test run_with_request_rate handles exception"""
        mock_time.return_value = 1000.0
        self.scheduler.run_target_server = MagicMock(side_effect=RuntimeError("fail"))

        result = self.scheduler.run_with_request_rate(self.params, self.params_field)
        assert self.scheduler.error_info is not None
        assert isinstance(result, PerformanceIndex)


class TestSchedulerMonitoringStatus(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(
            simulator=self.simulator,
            benchmark=self.benchmark,
            data_storage=self.data_storage,
        )
        # Mock health check hooks to always return healthy
        self.scheduler.service_checks = MagicMock()
        self.scheduler.benchmark_checks = MagicMock()
        svc_result = MagicMock()
        svc_result.is_healthy = True
        self.scheduler.service_checks.run.return_value = svc_result
        bench_result = MagicMock()
        bench_result.is_healthy = True
        self.scheduler.benchmark_checks.run.return_value = bench_result

    @patch("time.time")
    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.get_settings")
    def test_monitoring_benchmark_stopped(self, mock_settings, mock_sleep, mock_time):
        """Test monitoring_status returns when benchmark.health stage != running"""
        mock_settings.return_value.particles_time_out = 5
        # start_time=0, iter1: elapsed+context
        mock_time.side_effect = [0, 1, 1]
        # Remove check_success to go directly to health path
        del self.simulator.check_success
        self.simulator.__class__ = type("OtherSim", (), {})
        self.simulator.health = MagicMock(return_value=MagicMock(stage=Stage.running))
        self.benchmark.health.return_value = MagicMock(stage=Stage.stop)

        self.scheduler.monitoring_status()

    @patch("time.time")
    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.get_settings")
    @patch("optix.optimizer.scheduler.logger")
    def test_monitoring_first_duration_warning(self, mock_logger, mock_settings, mock_sleep, mock_time):
        """Test monitoring_status logs warning when duration exceeds 2x first_duration"""
        mock_settings.return_value.particles_time_out = 3
        # start_time + iter1(elapsed, context, duration) + iter2(elapsed, context) = 6 calls
        mock_time.side_effect = [0, 100, 100, 100, 100, 100]
        # Remove check_success to go directly to health path
        del self.simulator.check_success
        self.simulator.__class__ = type("OtherSim", (), {})
        self.simulator.health = MagicMock(return_value=MagicMock(stage=Stage.running))
        # First iter: running so we reach duration check; second iter: stop so we return
        self.benchmark.health.side_effect = [
            MagicMock(stage=Stage.running),
            MagicMock(stage=Stage.stop),
        ]
        self.scheduler.run_start_timestamp = 1.0
        self.scheduler.first_duration = 1.0

        self.scheduler.monitoring_status()

        # Verify the warning was actually logged
        mock_logger.warning.assert_called_once_with(
            "The current runtime is more than twice the duration of the first run."
        )


class TestWaitSimulateHealthBranches(unittest.TestCase):
    """Test wait_simulate with various health() stage returns"""

    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage)
        self.scheduler.simulate_run_info = ()
        self.scheduler.wait_time = 3
        # Mock service_checks to always return healthy
        self.scheduler.service_checks = MagicMock()
        mock_result = MagicMock()
        mock_result.is_healthy = True
        self.scheduler.service_checks.run.return_value = mock_result

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_simulate_health_start_then_running(self, mock_time, mock_sleep):
        """Test wait_simulate with Stage.start followed by Stage.running"""
        # start_time=0, iter1: elapsed+context, iter2: elapsed+context
        mock_time.side_effect = [0, 1, 1, 2, 2]
        from optix.config.constant import ProcessState

        # First call: start (continue), second call: running (return)
        self.simulator.health = MagicMock(
            side_effect=[
                ProcessState(stage=Stage.start, info="loading"),
                ProcessState(stage=Stage.running),
            ]
        )

        self.scheduler.wait_simulate()

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_simulate_no_health_method_raises(self, mock_time, mock_sleep):
        """Test wait_simulate raises RuntimeError if no health/check_success method"""
        # start_time=0, iter1: elapsed+context
        mock_time.side_effect = [0, 1, 1]
        # Remove both health and check_success
        del self.simulator.health
        del self.simulator.check_success

        with self.assertRaises(RuntimeError):
            self.scheduler.wait_simulate()

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_simulate_service_error_fatal(self, mock_time, mock_sleep):
        """Test wait_simulate raises FatalError when service check returns unhealthy"""
        # start_time=0, iter1: elapsed+context
        mock_time.side_effect = [0, 1, 1]
        from optix.config.config import ErrorSeverity

        mock_result = MagicMock()
        mock_result.is_healthy = False
        mock_result.error_context = ErrorContext(
            error_type=ErrorType.OUT_OF_MEMORY,
            severity=ErrorSeverity.FATAL,
            message="OOM during startup",
        )
        self.scheduler.service_checks.run.return_value = mock_result

        with self.assertRaises(FatalError):
            self.scheduler.wait_simulate()


class TestMonitoringStatusBranches(unittest.TestCase):
    """Test monitoring_status with check_success and various health paths"""

    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage)
        # Mock health check hooks to always return healthy
        self.scheduler.service_checks = MagicMock()
        self.scheduler.benchmark_checks = MagicMock()
        svc_result = MagicMock()
        svc_result.is_healthy = True
        self.scheduler.service_checks.run.return_value = svc_result
        bench_result = MagicMock()
        bench_result.is_healthy = True
        self.scheduler.benchmark_checks.run.return_value = bench_result

    @patch("time.time")
    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.get_settings")
    @patch("optix.optimizer.scheduler.is_mindie")
    @patch("optix.optimizer.scheduler.is_vllm")
    def test_monitoring_check_success_returns(self, mock_vllm, mock_mindie, mock_settings, mock_sleep, mock_time):
        """Test monitoring_status returns when benchmark.check_success() is True"""
        mock_settings.return_value.particles_time_out = 5
        # start_time=0, iter1: elapsed+context
        mock_time.side_effect = [0, 1, 1]
        mock_mindie.return_value = True
        mock_vllm.return_value = False
        self.simulator.process.poll.return_value = None
        self.simulator.check_success = MagicMock(return_value=True)
        self.benchmark.check_success.return_value = True

        self.scheduler.monitoring_status()
        self.benchmark.check_success.assert_called()

    @patch("time.time")
    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.get_settings")
    @patch("optix.optimizer.scheduler.is_mindie")
    @patch("optix.optimizer.scheduler.is_vllm")
    def test_monitoring_simulator_poll_exited_raises(
        self, mock_vllm, mock_mindie, mock_settings, mock_sleep, mock_time
    ):
        """Test monitoring_status raises when simulator.process.poll() is not None (exited)"""
        import subprocess

        mock_settings.return_value.particles_time_out = 5
        # start_time=0, iter1: elapsed+context
        mock_time.side_effect = [0, 1, 1]
        mock_mindie.return_value = True
        mock_vllm.return_value = False
        self.simulator.check_success = MagicMock(return_value=True)
        self.simulator.process.poll.return_value = 1
        self.simulator.process.returncode = 1
        self.simulator.command = ["vllm", "serve", "model"]
        self.simulator.run_log = "/tmp/sim.log"

        with self.assertRaises(subprocess.SubprocessError) as ctx:
            self.scheduler.monitoring_status()
        message = str(ctx.exception)
        self.assertIn("exit=1", message)
        self.assertIn("command:", message)
        self.assertNotIn("Failed in run simulator", message)

    @patch("time.time")
    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.get_settings")
    def test_monitoring_simulator_health_error_raises(self, mock_settings, mock_sleep, mock_time):
        """Test monitoring_status raises when simulator.health() returns non-running"""
        import subprocess

        from optix.config.constant import ProcessState

        mock_settings.return_value.particles_time_out = 5
        # start_time=0, iter1: elapsed+context
        mock_time.side_effect = [0, 1, 1]
        # Remove check_success to go to health path
        del self.simulator.check_success
        # Make simulator NOT an instance of Simulator
        self.simulator.__class__ = type("OtherSim", (), {})
        self.simulator.health = MagicMock(return_value=ProcessState(stage=Stage.error, info="crashed"))
        self.simulator.command = ["vllm", "serve", "model"]
        self.simulator.run_log = "/tmp/sim.log"
        self.simulator.process.returncode = 1

        with self.assertRaises(subprocess.SubprocessError) as ctx:
            self.scheduler.monitoring_status()
        message = str(ctx.exception)
        self.assertIn("exit=1", message)
        self.assertIn("command:", message)
        self.assertNotIn("Failed in run simulator", message)

    @patch("time.time")
    @patch("time.sleep")
    @patch("optix.optimizer.scheduler.get_settings")
    def test_monitoring_timeout_raises(self, mock_settings, mock_sleep, mock_time):
        """Test monitoring_status raises TimeoutError when timeout reached"""
        mock_settings.return_value.particles_time_out = 2
        mock_time.side_effect = list(range(10))
        del self.simulator.check_success
        from optix.config.constant import ProcessState

        self.simulator.__class__ = type("OtherSim", (), {})
        self.simulator.health = MagicMock(return_value=ProcessState(stage=Stage.running))
        self.benchmark.health = MagicMock(return_value=ProcessState(stage=Stage.running))

        with self.assertRaises(TimeoutError):
            self.scheduler.monitoring_status()


class TestRunTargetServerRetry(unittest.TestCase):
    """Test run_target_server retry exhaustion"""

    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage)
        self.scheduler.simulate_run_info = ()

    @patch("time.sleep")
    def test_retry_exhaustion_raises_value_error(self, mock_sleep):
        """Test run_target_server raises ValueError after all retries exhausted"""
        self.scheduler.run_simulate = MagicMock()
        self.benchmark.run = MagicMock()
        self.scheduler.monitoring_status = MagicMock(side_effect=RetryableError("keep failing"))
        self.scheduler.retry_number = 2

        with self.assertRaises(ValueError, msg="Failed in run_target_server after 2 attempts"):
            self.scheduler.run_target_server(np.array([1.0]), (MagicMock(),))

    @patch("time.time")
    @patch("time.sleep")
    def test_save_result_with_backup(self, mock_sleep, mock_time):
        """Test save_result calls backup when bak_path is set"""
        mock_time.return_value = 1010.0
        self.scheduler.run_start_timestamp = 1000.0
        self.scheduler.first_duration = 5.0
        self.scheduler.performance_index = PerformanceIndex()
        self.scheduler.simulate_run_info = ()
        self.scheduler.error_info = None
        self.scheduler.current_back_path = "/tmp/bak/001"
        self.scheduler.bak_path = "/tmp/bak"
        self.scheduler.save_result(real_evaluation=False)
        self.data_storage.save.assert_called_once()
        # pylint: disable=no-member
        self.simulator.backup.assert_called_once()
        self.benchmark.backup.assert_called_once()
        # pylint: enable=no-member


class TestSchedulerRunEvaluation(unittest.TestCase):
    """Tests for _run_evaluation OptimizerError propagation."""

    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage)
        self.params_field = (
            OptimizerConfigField(
                name="CONCURRENCY",
                config_position="env",
                value=10,
                min=1,
                max=100,
                dtype="int",
            ),
        )

    @patch("time.sleep")
    def test_run_evaluation_reraises_benchmark_result_error(self, _mock_sleep):
        from optix.optimizer.errors import BenchmarkResultError

        self.scheduler.run_target_server = MagicMock()
        self.benchmark.get_performance_index.side_effect = BenchmarkResultError("csv not unique")

        with self.assertRaises(BenchmarkResultError):
            self.scheduler.run(np.array([10.0]), self.params_field)

        self.assertIsNone(self.scheduler.last_outcome)

    @patch("time.sleep")
    def test_run_evaluation_records_failure_for_runtime_error(self, _mock_sleep):
        from optix.optimizer.outcome import RunStatus

        self.scheduler.run_target_server = MagicMock(side_effect=RuntimeError("bench down"))

        result = self.scheduler.run(np.array([10.0]), self.params_field)

        self.assertIsNotNone(self.scheduler.last_outcome)
        self.assertEqual(self.scheduler.last_outcome.status, RunStatus.FAILED)
        self.assertIsInstance(result, PerformanceIndex)


class TestSchedulerRerunBenchmarkOnly(unittest.TestCase):
    """Tests for rerun_benchmark_only: reuse the running simulator, rerun only the benchmark."""

    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.benchmark.run_log = "/tmp/benchmark.log"
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(
            simulator=self.simulator,
            benchmark=self.benchmark,
            data_storage=self.data_storage,
        )
        # monitoring_status touches many collaborators; stub it out for these unit tests.
        self.scheduler.monitoring_status = MagicMock()

        self.params = np.array([64.0, 5.0])
        self.params_field = (
            OptimizerConfigField(name="CONCURRENCY", value=64.0, min=1, max=100, dtype="int"),
            OptimizerConfigField(name="REQUESTRATE", value=5.0, min=5.0, max=5.0, dtype="float"),
        )
        self.performance_index = PerformanceIndex(throughput=100.0, generate_speed=100)
        self.benchmark.get_performance_index.return_value = self.performance_index

    @patch("time.time")
    @patch("time.sleep")
    def test_rerun_success_returns_performance_index(self, mock_sleep, mock_time):
        """A successful rerun returns the performance index and records a SUCCESS outcome."""
        from optix.optimizer.outcome import RunStatus

        mock_time.return_value = 1000.0

        result = self.scheduler.rerun_benchmark_only(self.params, self.params_field)

        self.assertIsInstance(result, PerformanceIndex)
        self.assertEqual(result.throughput, 100.0)
        self.assertEqual(self.scheduler.last_outcome.status, RunStatus.SUCCESS)
        self.assertIsNone(self.scheduler.error_info)
        # The live simulator is never stopped; only the benchmark is restarted.
        self.simulator.stop.assert_not_called()
        self.benchmark.run.assert_called_once()

    @patch("time.time")
    @patch("time.sleep")
    def test_rerun_handles_exception_records_failure(self, mock_sleep, mock_time):
        """A runtime error records a FAILED outcome instead of raising."""
        from optix.optimizer.outcome import RunStatus

        mock_time.return_value = 1000.0
        self.benchmark.run.side_effect = RuntimeError("benchmark crashed")

        result = self.scheduler.rerun_benchmark_only(self.params, self.params_field)

        self.assertIsNotNone(self.scheduler.error_info)
        self.assertEqual(self.scheduler.last_outcome.status, RunStatus.FAILED)
        self.assertIsInstance(result, PerformanceIndex)
