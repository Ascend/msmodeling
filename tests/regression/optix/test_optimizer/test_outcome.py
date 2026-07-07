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
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import numpy as np

from optix.config.config import OptimizerConfigField, PerformanceIndex
from optix.optimizer.outcome import RunOutcome, RunStatus
from optix.optimizer.protocols import (
    SupportsDataField,
    SupportsHealth,
    SupportsPrepare,
)
from optix.optimizer.scheduler import Scheduler


class TestRunOutcome(unittest.TestCase):
    def test_frozen_dataclass(self):
        perf = PerformanceIndex()
        outcome = RunOutcome(status=RunStatus.SUCCESS, performance_index=perf)
        with self.assertRaises(FrozenInstanceError):
            outcome.status = RunStatus.FAILED

    def test_has_error_false_on_success(self):
        outcome = RunOutcome(status=RunStatus.SUCCESS, performance_index=PerformanceIndex())
        assert not outcome.has_error

    def test_has_error_true_on_failure(self):
        outcome = RunOutcome(
            status=RunStatus.FAILED,
            performance_index=PerformanceIndex(),
            error_context=RuntimeError("fail"),
        )
        assert outcome.has_error

    def test_has_error_true_when_failed_without_error_context(self):
        outcome = RunOutcome(
            status=RunStatus.FAILED,
            performance_index=PerformanceIndex(),
            error_context=None,
        )
        assert outcome.has_error

    def test_has_error_false_when_success_with_error_context(self):
        outcome = RunOutcome(
            status=RunStatus.SUCCESS,
            performance_index=PerformanceIndex(),
            error_context=RuntimeError("stale"),
        )
        assert not outcome.has_error


class TestSchedulerRunOutcome(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler(MagicMock(), MagicMock(), MagicMock())
        self.scheduler.simulator.check_success = MagicMock(return_value=True)

    @patch("time.time", return_value=1000.0)
    @patch("time.sleep")
    def test_run_sets_last_outcome_success(self, _sleep, _time):
        perf = PerformanceIndex(throughput=10.0)
        self.scheduler.benchmark.get_performance_index.return_value = perf
        params = np.array([1.0])
        fields = (OptimizerConfigField(name="p", value=1.0, min=0, max=5),)
        result = self.scheduler.run(params, fields)
        assert result.throughput == 10.0
        assert self.scheduler.last_outcome is not None
        assert self.scheduler.last_outcome.status == RunStatus.SUCCESS
        assert self.scheduler.error_info is None

    @patch("time.time", return_value=1000.0)
    def test_run_sets_last_outcome_failed(self, _time):
        self.scheduler.run_target_server = MagicMock(side_effect=RuntimeError("bench down"))
        params = np.array([1.0])
        fields = (OptimizerConfigField(name="p", value=1.0, min=0, max=5),)
        self.scheduler.run(params, fields)
        assert self.scheduler.last_outcome.status == RunStatus.FAILED
        assert isinstance(self.scheduler.error_info, RuntimeError)

    def test_error_info_manual_assignment(self):
        self.scheduler.error_info = "skip: duplicate particle"
        assert self.scheduler.error_info == "skip: duplicate particle"


class TestProtocols(unittest.TestCase):
    def test_supports_prepare_detects_prepare(self):
        bench = MagicMock()
        bench.prepare = MagicMock()
        assert isinstance(bench, SupportsPrepare)

    def test_supports_health_detects_health(self):
        sim = MagicMock()
        sim.health = MagicMock()
        assert isinstance(sim, SupportsHealth)

    def test_supports_data_field_requires_update_command(self):
        obj = MagicMock()
        obj.data_field = ()
        obj.update_command = MagicMock()
        assert isinstance(obj, SupportsDataField)
