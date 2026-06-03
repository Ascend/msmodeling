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
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest

from experimental.optix.config.config import CommunicationConfig, get_settings
from experimental.optix.optimizer.server import Scheduler
from experimental.optix.optimizer.communication import (
    CommunicationForFile,
    CustomCommand,
)


def test_scheduler_init(tmpdir):
    # Create a CommunicationConfig object
    work_path = Path(tmpdir)
    res_file = work_path.joinpath("res.txt")
    cmd_file = work_path.joinpath("cmd.txt")
    communication_config = CommunicationConfig(cmd_file=res_file, res_file=cmd_file)

    # Create Scheduler object
    scheduler = Scheduler(communication_config)

    # Check if communication_config is correctly passed
    assert scheduler.communication_config == communication_config

    # Check if simulator is initialized to None
    assert scheduler.simulator is None

    # Check if communication is correctly initialized
    assert isinstance(scheduler.communication, CommunicationForFile)
    assert scheduler.communication.res_file == res_file
    assert scheduler.communication.cmd_file == cmd_file

    # Check if cmd is correctly initialized
    assert isinstance(scheduler.cmd, CustomCommand)


def test_backup_path_exists():
    # Arrange
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = Mock()
    params = "/existing/path"
    _cmd = scheduler.cmd.start
    scheduler.cmd.history = _cmd
    with patch("experimental.optix.optimizer.server.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        scheduler.backup(params)

    # Assert
    scheduler.communication.send_command.assert_called_once_with(f"{scheduler.cmd.history[-1]}:done")
    scheduler.communication.clear_res.assert_called_once()


def test_backup_path_not_exists():
    # Arrange
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = Mock()
    params = "/non/existing/path"
    _cmd = scheduler.cmd.start
    scheduler.cmd.history = _cmd
    with patch("experimental.optix.optimizer.server.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.mkdir.return_value = None
        scheduler.backup(params)

    # Assert
    scheduler.communication.send_command.assert_called_once_with(f"{scheduler.cmd.history[-1]}:done")
    scheduler.communication.clear_res.assert_called_once()


def test_check_success_no_simulator():
    scheduler = Scheduler(get_settings().communication)
    scheduler.simulator = None
    assert scheduler.check_success() is None


def test_check_success_simulator_succeeds_immediately():
    scheduler = Scheduler(get_settings().communication)
    simulator = MagicMock()
    scheduler.simulator = simulator
    scheduler.cmd.history = "check success 1111111"
    simulator.check_success.return_value = True
    scheduler.communication = Mock()
    scheduler.check_success()
    scheduler.communication.send_command.assert_called_once_with("check success 1111111:True")


@patch("time.sleep")
def test_check_success_simulator_succeeds_after_retries(mock_sleep):
    scheduler = Scheduler(get_settings().communication)
    simulator = MagicMock()
    scheduler.simulator = simulator
    scheduler.cmd.history = "check success 1111111"
    simulator.check_success.side_effect = [False, False, True]
    scheduler.communication = Mock()
    scheduler.check_success()
    assert simulator.check_success.call_count == 3
    scheduler.communication.send_command.assert_called_once_with("check success 1111111:True")
    mock_sleep.call_count == 2


@patch("time.sleep")
def test_check_success_simulator_always_fails(mock_sleep):
    scheduler = Scheduler(get_settings().communication)
    simulator = MagicMock()
    scheduler.simulator = simulator
    scheduler.cmd.history = "check success 1111111"
    simulator.check_success.return_value = False
    scheduler.communication = Mock()
    scheduler.check_success()
    assert simulator.check_success.call_count == 10
    scheduler.communication.send_command.assert_called_once_with("check success 1111111:False")
    mock_sleep.call_count == 10


def test_stop_with_simulator():
    # Create Scheduler instance
    scheduler = Scheduler(get_settings().communication)
    # Mock simulator and communication objects
    scheduler.simulator = Mock()
    scheduler.communication = Mock()
    scheduler.cmd.history = "stop 1111111 params:True"
    # Test parameters
    params = "True"

    # Call stop method
    scheduler.stop(params)

    # Verify simulator.stop is called
    scheduler.simulator.stop.assert_called_once_with(True)

    # Verify communication.send_command is called
    scheduler.communication.send_command.assert_called_once()

    # Verify communication.clear_res is called
    scheduler.communication.clear_res.assert_called_once()


def test_stop_without_simulator():
    # Create Scheduler instance
    scheduler = Scheduler(get_settings().communication)

    # Set simulator to None
    scheduler.simulator = None

    # Mock communication object
    scheduler.communication = Mock()
    scheduler.cmd.history = "stop 1111111 params:True"

    # Test parameters
    params = "True"

    # Call stop method
    scheduler.stop(params)

    # Verify communication.send_command is not called
    scheduler.communication.send_command.assert_not_called()

    # Verify communication.clear_res is not called
    scheduler.communication.clear_res.assert_not_called()


def test_get_cmd_param_empty():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = ""
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_eof():
    scheduler = Scheduler(get_settings().communication)

    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "EOF"
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_history():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "cmd1"
    scheduler.cmd.history = ["cmd1"]
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_format_error():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "cmd1"
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_success():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "cmd1 params:123"
    assert scheduler.get_cmd_param() == ("cmd1", "123")


class TestSchedulerProcessPoll:
    @classmethod
    def test_process_poll_with_simulator(cls, scheduler):
        # Mock simulator.process.poll() return value
        scheduler.simulator.process.poll.return_value = True
        scheduler.cmd.history = "process_poll 1111111"
        # Call process_poll method
        scheduler.process_poll()

        # Verify related methods are called correctly
        scheduler.simulator.process.poll.assert_called_once()
        scheduler.communication.send_command.assert_called_once_with("process_poll 1111111:True")
        scheduler.communication.clear_res.assert_called_once()

    @classmethod
    def test_process_poll_without_simulator(cls, scheduler):
        # Set simulator to None
        scheduler.simulator = None
        scheduler.cmd.history = "process_poll 1111111"
        # Call process_poll method
        scheduler.process_poll()
        # Verify related methods are called correctly
        scheduler.communication.send_command.assert_called_once_with("process_poll 1111111:None")
        scheduler.communication.clear_res.assert_called_once()

    @pytest.fixture
    def scheduler(self):
        # Create Scheduler instance
        scheduler = Scheduler(get_settings().communication)
        scheduler.simulator = MagicMock()
        scheduler.communication = MagicMock()
        return scheduler


# Test case 1: test init returns False when get_cmd_param returns _cmd as None
def test_init_cmd_none():
    scheduler = Scheduler(get_settings().communication)
    scheduler.get_cmd_param = MagicMock(return_value=(None, None))
    assert scheduler.init() is False


# Test case 2: test init returns True when get_cmd_param returns _cmd as "init"
def test_init_cmd_init():
    scheduler = Scheduler(get_settings().communication)
    scheduler.get_cmd_param = MagicMock(return_value=("init", None))
    scheduler.cmd.history = "init 11111111"
    scheduler.communication = MagicMock()
    assert scheduler.init() is True
    scheduler.communication.send_command.assert_called_once_with("init 11111111:done")
    scheduler.communication.clear_res.assert_called_once()


# Test case 3: test init returns False when get_cmd_param returns _cmd not "init"
def test_init_cmd_not_init():
    scheduler = Scheduler(get_settings().communication)
    scheduler.get_cmd_param = MagicMock(return_value=("other_command", None))
    assert scheduler.init() is False


def test_run_no_cmd():
    scheduler = Scheduler(get_settings().communication)
    with patch.object(scheduler, "get_cmd_param", return_value=(None, None)):
        assert scheduler.run() == ""


def test_run_unknown_cmd():
    scheduler = Scheduler(get_settings().communication)
    with patch.object(scheduler, "get_cmd_param", return_value=("unknown_cmd", None)):
        with patch("experimental.optix.optimizer.server.logger") as mock_logger:
            assert scheduler.run() == ""
            mock_logger.error.assert_called_once()


def test_run_no_param():
    scheduler = Scheduler(get_settings().communication)
    scheduler.command = MagicMock(return_value="result")
    with patch.object(scheduler, "get_cmd_param", return_value=("command", None)):
        with patch(
            "experimental.optix.optimizer.server.getattr",
            return_value=scheduler.command,
        ):
            assert scheduler.run() == "result"
            scheduler.command.assert_called_once()


def test_run_with_param():
    scheduler = Scheduler(get_settings().communication)
    scheduler.command = MagicMock(return_value="result")
    with patch.object(scheduler, "get_cmd_param", return_value=("command", "param")):
        with patch(
            "experimental.optix.optimizer.server.getattr",
            return_value=scheduler.command,
        ):
            assert scheduler.run() == "result"
            scheduler.command.assert_called_once_with("param")
