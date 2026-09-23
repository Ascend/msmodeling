# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Regression tests for enable_simulate context manager."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from optix.optimizer.optimizer import enable_simulate


class TestEnableSimulate:
    @patch("optix.optimizer.optimizer.simulate_flag", False)
    def test_no_simulate_flag_yields_false(self):
        scheduler = MagicMock()
        with enable_simulate(scheduler) as flag:
            assert flag is False
        scheduler.simulator.enable_simulation_model.assert_not_called()

    @patch("optix.optimizer.optimizer.simulate_flag", True)
    def test_simulate_flag_enters_context_manager(self):
        scheduler = MagicMock()

        @contextmanager
        def _fake_enable():
            yield True

        scheduler.simulator.enable_simulation_model = MagicMock(side_effect=_fake_enable)
        with enable_simulate(scheduler) as flag:
            assert flag is True
        scheduler.simulator.enable_simulation_model.assert_called_once_with()
