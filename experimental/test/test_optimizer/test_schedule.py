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
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from experimental.optix.config.config import (
    CommunicationConfig,
    map_param_with_value,
    get_settings,
    default_support_field,
    OptimizerConfigField,
    PerformanceIndex,
    ErrorSeverity,
    ErrorType,
)
from experimental.optix.config.constant import Stage
from experimental.optix.optimizer.communication import (
    CommunicationForFile,
    CustomCommand,
)
from experimental.optix.optimizer.scheduler import ScheduleWithMultiMachine, Scheduler
from experimental.optix.config.base_config import FOLDER_LIMIT_SIZE
from experimental.optix.optimizer.health_check import (
    FatalError,
    ErrorContext,
    RetryableError,
    ServiceHookPoint,
    BenchmarkHookPoint,
)


class TestScheduleWithMultiMachine:
    @classmethod
    def test_schedule_with_multi_machine_init(cls):
        # Create mock objects
        mock_communication_config = MagicMock(spec=CommunicationConfig)
        mock_communication = MagicMock(spec=CommunicationForFile)
        mock_custom_command = MagicMock(spec=CustomCommand)

        # Mock constructors of CommunicationForFile and CustomCommand
        with (
            patch(
                "experimental.optix.optimizer.scheduler.CommunicationForFile",
                return_value=mock_communication,
            ) as mock_communication_class,
            patch(
                "experimental.optix.optimizer.scheduler.CustomCommand",
                return_value=mock_custom_command,
            ) as mock_custom_command_class,
        ):
            mock_communication_config.cmd_file = None
            mock_communication_config.res_file = None
            # Create ScheduleWithMultiMachine instance
            scheduler = ScheduleWithMultiMachine(mock_communication_config, MagicMock(), MagicMock(), MagicMock())

            # Verify CommunicationForFile and CustomCommand are correctly initialized
            mock_communication_class.assert_called_once_with(
                mock_communication_config.cmd_file, mock_communication_config.res_file
            )
            mock_custom_command_class.assert_called_once()

            # Verify communication attribute is correctly set
            assert scheduler.communication_config == mock_communication_config
            assert scheduler.communication == mock_communication

            # Verify cmd attribute is correctly set
            assert scheduler.cmd == mock_custom_command

            # Verify send_command and clear_command methods are correctly called
            mock_communication.send_command.assert_called_once_with(mock_custom_command.init)
            mock_communication.clear_command.assert_called_once_with(mock_custom_command.init)

    @classmethod
    def test_run_simulate(cls, schedule_with_multi_machine):
        # Mock methods
        schedule_with_multi_machine.cmd = CustomCommand()
        schedule_with_multi_machine.benchmark.prepare.return_value = None
        schedule_with_multi_machine.communication.send_command.return_value = None
        schedule_with_multi_machine.communication.clear_command.return_value = None
        schedule_with_multi_machine.simulator.run.return_value = None
        schedule_with_multi_machine.wait_simulate = MagicMock(return_value=None)
        _params = np.random.random(len(default_support_field))
        # Call run_simulate method
        schedule_with_multi_machine.simulate_run_info = map_param_with_value(_params, default_support_field)
        schedule_with_multi_machine.run_simulate(_params, default_support_field)

        # Verify method calls
        schedule_with_multi_machine.benchmark.prepare.assert_called_once()
        assert schedule_with_multi_machine.communication.send_command.call_count == 2
        assert schedule_with_multi_machine.communication.clear_command.call_count == 2
        schedule_with_multi_machine.wait_simulate.assert_called_once()

    @pytest.fixture
    def schedule_with_multi_machine(self):
        mock_communication_config = MagicMock(spec=CommunicationConfig)
        mock_communication = MagicMock(spec=CommunicationForFile)
        mock_custom_command = MagicMock(spec=CustomCommand)
        mock_communication_config.cmd_file = None
        mock_communication_config.res_file = None
        with (
            patch(
                "experimental.optix.optimizer.scheduler.CommunicationForFile",
                return_value=mock_communication,
                autospec=True,
            ),
            patch(
                "experimental.optix.optimizer.scheduler.CustomCommand",
                return_value=mock_custom_command,
                autospec=True,
            ),
        ):
            schedule = ScheduleWithMultiMachine(get_settings().communication, MagicMock(), MagicMock(), MagicMock())
            schedule.cmd = MagicMock()
            schedule.communication = MagicMock()
            yield schedule

    @patch("experimental.optix.optimizer.scheduler.get_train_sub_path")
    def test_set_back_up_path_with_bak_path(self, mock_get_train_sub_path, schedule_with_multi_machine, tmpdir):
        # Arrange
        bak_path = Path(tmpdir)
        schedule_with_multi_machine.bak_path = bak_path
        mock_get_train_sub_path.return_value = tmpdir
        # Act
        schedule_with_multi_machine.set_back_up_path()

        # Assert
        mock_get_train_sub_path.assert_called_once_with(bak_path)
        schedule_with_multi_machine.simulator.bak_path == bak_path
        schedule_with_multi_machine.benchmark.bak_path == bak_path
        schedule_with_multi_machine.communication.send_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.backup} params:{bak_path}"
        )
        schedule_with_multi_machine.communication.clear_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.backup} params:{bak_path}"
        )

    @patch("experimental.optix.optimizer.scheduler.time.sleep", return_value=None)
    def test_monitoring_status_success(self, mock_sleep, schedule_with_multi_machine):
        type(schedule_with_multi_machine.cmd).process_poll = PropertyMock(return_value="mocked process poll")
        schedule_with_multi_machine.communication.send_command = MagicMock()
        schedule_with_multi_machine.simulator.process.poll = MagicMock(return_value=None)
        schedule_with_multi_machine.communication.clear_command = MagicMock(return_value=None)
        schedule_with_multi_machine.benchmark.check_success = MagicMock(return_value=True)
        schedule_with_multi_machine.stop_target_server = MagicMock()

        schedule_with_multi_machine.monitoring_status()

        schedule_with_multi_machine.communication.send_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.simulator.process.poll.assert_called()
        schedule_with_multi_machine.communication.clear_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.benchmark.check_success.assert_called()
        schedule_with_multi_machine.stop_target_server.assert_not_called()

    @patch("experimental.optix.optimizer.scheduler.time.sleep", return_value=None)
    def test_monitoring_status_failure(self, mock_sleep, schedule_with_multi_machine):
        type(schedule_with_multi_machine.cmd).process_poll = PropertyMock(return_value="mocked process poll")
        schedule_with_multi_machine.communication.send_command = MagicMock()
        schedule_with_multi_machine.simulator.process.poll = MagicMock(return_value=1)
        schedule_with_multi_machine.communication.clear_command = MagicMock(return_value=None)
        schedule_with_multi_machine.benchmark.check_success = MagicMock(return_value=False)
        schedule_with_multi_machine.stop_target_server = MagicMock()

        with pytest.raises(subprocess.SubprocessError):
            schedule_with_multi_machine.monitoring_status()

        schedule_with_multi_machine.communication.send_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.simulator.process.poll.assert_called()
        schedule_with_multi_machine.communication.clear_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.benchmark.check_success.assert_not_called()
        schedule_with_multi_machine.stop_target_server.assert_called()

    @patch("experimental.optix.optimizer.scheduler.Scheduler.stop_target_server")
    def test_stop_target_server_with_del_log(self, mock_super_stop, schedule_with_multi_machine):
        # Test behavior when del_log is True
        schedule_with_multi_machine.cmd = CustomCommand()
        schedule_with_multi_machine.stop_target_server(del_log=True)

        # Verify Scheduler.stop_target_server is called
        mock_super_stop.assert_called_once_with(True)

        # Verify communication.send_command and communication.clear_command are correctly called
        assert "params:True" in schedule_with_multi_machine.cmd.history[-1]
        schedule_with_multi_machine.communication.send_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.history[-1]}"
        )
        schedule_with_multi_machine.communication.clear_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.history[-1]}"
        )


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.bak_path = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage, self.bak_path)
        # Initialize simulate_run_info to avoid repeated setup in each test method
        self.scheduler.simulate_run_info = ()

    @patch("experimental.optix.optimizer.utils.get_folder_size")
    def test_set_back_up_path_folder_size_exceeds_limit(self, mock_get_folder_size):
        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE + 1
        self.scheduler.set_back_up_path()

    @patch("experimental.optix.optimizer.utils.get_folder_size")
    @patch("experimental.optix.common.get_train_sub_path")
    def test_set_back_up_path_folder_size_within_limit(self, mock_get_train_sub_path, mock_get_folder_size):
        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE - 1
        mock_get_train_sub_path.return_value = "sub_path"
        self.scheduler.set_back_up_path()

    @patch("time.sleep", return_value=None)
    def test_wait_simulate_success(self, mock_sleep):
        self.simulator.health.return_value.stage = Stage.running
        self.scheduler.wait_simulate()

    @patch("time.sleep", return_value=None)
    def test_wait_simulate_timeout(self, mock_sleep):
        self.simulator.health.return_value.stage = Stage.error
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
    @patch("experimental.optix.optimizer.scheduler.Scheduler.wait_simulate")
    def test_run_target_server_fatal_no_retry(self, mock_wait, _):
        """Test fatal error is raised immediately without retry"""
        mock_wait.side_effect = FatalError("OOM error")
        with self.assertRaises(FatalError):
            self.scheduler.run_target_server(np.array([1, 2, 3]), ("field1", "field2", "field3"))
        self.assertEqual(mock_wait.call_count, 1)

    @patch("time.sleep")
    @patch("experimental.optix.optimizer.scheduler.Scheduler.monitoring_status")
    @patch("experimental.optix.optimizer.scheduler.Scheduler.wait_simulate")
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
    @patch("experimental.optix.optimizer.scheduler.logger")
    def test_run_logging(self, mock_logger, mock_time):
        """Test logging in run method"""
        mock_time.return_value = 1000.0

        self.scheduler.run(self.params, self.params_field)
