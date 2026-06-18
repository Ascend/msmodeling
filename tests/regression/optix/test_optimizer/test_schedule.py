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

from optix.config.config import (
    OptimizerConfigField,
    PerformanceIndex,
    ErrorSeverity,
    ErrorType,
)
from optix.config.constant import Stage
from optix.optimizer.scheduler import Scheduler
from optix.config.base_config import FOLDER_LIMIT_SIZE
from optix.optimizer.health_check import (
    FatalError,
    ErrorContext,
    RetryableError,
    ServiceHookPoint,
    BenchmarkHookPoint,
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
