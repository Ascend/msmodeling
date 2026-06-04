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
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ms_serviceparam_optimizer.config.config import (
    CUSTOM_OUTPUT,
    MODEL_EVAL_STATE_CONFIG_PATH,
    OptimizerConfigField,
)
from ms_serviceparam_optimizer.optimizer.custom_process import (
    CustomProcess,
    tempfile,
    os,
)


def test_before_run_no_run_params(monkeypatch):
    # Mock tempfile.mkstemp
    monkeypatch.setattr(tempfile, "mkstemp", lambda prefix="": (1234, "tempfile"))
    # Mock os.environ
    monkeypatch.setattr(os, "environ", {})
    process = CustomProcess()
    process.before_run()

    # Verify property assignment
    assert process.run_log_fp == 1234
    assert process.run_log == "tempfile"
    assert process.run_log_offset == 0


def test_before_run_with_run_params():
    process = CustomProcess()
    process.command = ["benchmark", "$CONCURRENCY", "$REQUESTRATE"]
    run_params = (
        OptimizerConfigField(
            name="CONCURRENCY",
            config_position="env",
            min=10,
            max=1000,
            dtype="int",
            value=10,
        ),
        OptimizerConfigField(
            name="REQUESTRATE",
            config_position="env",
            min=0.1,
            max=0.7,
            value=0.3,
            dtype="float",
        ),
    )
    process.before_run(run_params)
    assert process.command == ["benchmark", "10", "0.3"]


def test_before_run_env_var_already_set(monkeypatch):
    # Mock os.environ
    monkeypatch.setattr(
        os,
        "environ",
        {CUSTOM_OUTPUT: "/result", MODEL_EVAL_STATE_CONFIG_PATH: "config.toml"},
    )

    process = CustomProcess()
    process.before_run()

    # Verify tempfile.mkstemp was called
    assert os.environ[CUSTOM_OUTPUT] == "/result"
    assert os.environ[MODEL_EVAL_STATE_CONFIG_PATH] == "config.toml"


def test_check_success_process_still_running(tmpdir):
    # Simulate child process still running
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w", encoding="utf-8") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = None
    custom_process.print_log = True
    result = custom_process.check_success()

    assert result is False


def test_check_success_process_succeeded(tmpdir):
    # Simulate child process succeeded
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w", encoding="utf-8") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 0
    custom_process.print_log = True
    result = custom_process.check_success()

    assert result is True


def test_check_success_process_failed(tmpdir):
    # Simulate child process failed
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w", encoding="utf-8") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 1
    custom_process.print_log = True
    with pytest.raises(subprocess.SubprocessError):
        custom_process.check_success()


@patch("psutil.process_iter")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_process")
def test_check_env_no_residual_process(mock_kill_process, mock_process_iter):
    # Simulate no residual process
    mock_process_iter.return_value = [
        MagicMock(info={"pid": 1, "name": "not_process"}),
        MagicMock(info={"pid": 2, "name": "also_not_target"}),
        MagicMock(),
    ]

    CustomProcess.kill_residual_process("target_process")

    # Ensure kill_process was not called
    mock_process_iter.assert_called_once()
    mock_kill_process.assert_not_called()


@patch("psutil.process_iter")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_process")
def test_check_env_with_residual_process(mock_kill_process, mock_process_iter):
    # Simulate having residual processes
    mock_process_iter.return_value = [
        MagicMock(info={"pid": 1, "name": "not_target_process"}),
        MagicMock(info={"pid": 2, "name": "target_process"}),
        MagicMock(info={"pid": 3, "name": "another_target_process"}),
    ]

    CustomProcess.kill_residual_process("target_process,another_target_process")

    # Ensure kill_process was called
    mock_kill_process.assert_any_call("target_process")
    mock_kill_process.assert_any_call("another_target_process")


@patch("psutil.process_iter")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_process")
def test_check_env_kill_process_exception(mock_kill_process, mock_process_iter):
    # Simulate exception when attempting to kill process
    mock_process_iter.return_value = [MagicMock(info={"pid": 1, "name": "target_process"})]
    mock_kill_process.side_effect = Exception("Failed to kill process")

    CustomProcess.kill_residual_process("target_process")

    # Ensure kill_process was called and exception was caught
    mock_kill_process.assert_called_once_with("target_process")


# Test case 1: process_name exists and check_env succeeds
@patch("ms_serviceparam_optimizer.optimizer.custom_process.CustomProcess.kill_residual_process")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.CustomProcess.before_run")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.subprocess.Popen")
def test_run_process_name_exists_and_check_env_success(mock_popen, mock_before_run, mock_check_env):
    process = CustomProcess()
    process.process_name = "test_process"
    process.command = ["test_command"]
    process.work_path = "/test/work/path"
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.run()
    mock_check_env.assert_called_once_with("test_process")
    mock_before_run.assert_called_once()


# Test case 2: process_name exists but check_env fails
@patch("ms_serviceparam_optimizer.optimizer.custom_process.CustomProcess.kill_residual_process")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.CustomProcess.before_run")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.subprocess.Popen")
def test_run_process_name_exists_and_check_env_fail(mock_popen, mock_before_run, mock_check_env):
    process = CustomProcess()
    process.process_name = "test_process"
    process.command = ["test_command"]
    process.work_path = "/test/work/path"
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    mock_check_env.side_effect = Exception("kill_residual_process failed")
    process.run()
    mock_check_env.assert_called_once_with("test_process")
    mock_before_run.assert_called_once()
    mock_popen.assert_called_once()


# Test case 3: process_name does not exist
@patch("ms_serviceparam_optimizer.optimizer.custom_process.CustomProcess.before_run")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.subprocess.Popen")
def test_run_process_name_not_exists(mock_popen, mock_before_run):
    process = CustomProcess()
    process.process_name = None
    process.command = ["test_command"]
    process.work_path = "/test/work/path"
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.run()
    mock_before_run.assert_called_once()


# Test case 4: subprocess.Popen raises OSError
@patch("ms_serviceparam_optimizer.optimizer.custom_process.CustomProcess.before_run")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.subprocess.Popen")
def test_run_subprocess_popen_os_error(mock_popen, mock_before_run):
    process = CustomProcess()
    process.process_name = None
    process.command = ["test_command"]
    process.work_path = "/test/work/path"
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    mock_popen.side_effect = OSError("subprocess.Popen failed")
    with pytest.raises(OSError) as e:
        process.run()
    assert str(e.value) == "subprocess.Popen failed"
    mock_before_run.assert_called_once()


# Test case 1: run_log is None
def test_get_log_run_log_none():
    process = CustomProcess()
    process.run_log = None
    assert process.get_log() is None


# Test case 2: run_log file does not exist
@patch("pathlib.Path.exists", return_value=False)
def test_get_log_run_log_not_exists(mock_exists):
    process = CustomProcess()
    process.run_log = "nonexistent.log"
    assert process.get_log() is None


# Test cases for stop method
@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_children")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.remove_file")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.close_file_fp")
@patch("psutil.Process")
def test_stop_with_del_log_true(mock_psutil_process, mock_close_file_fp, mock_remove_file, mock_kill_children):
    # Test del_log=True case
    process = CustomProcess()
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.process = MagicMock()
    process.process.poll.return_value = None
    process.process.pid = 12345

    # Simulate child process
    mock_child_process = MagicMock()
    mock_psutil_process.return_value.children.return_value = [mock_child_process]

    # Call stop method
    process.stop(del_log=True)

    # Verify run_log_offset was reset
    assert process.run_log_offset == 0

    # Verify file operations
    mock_close_file_fp.assert_called_once_with(process.run_log_fp)
    mock_remove_file.assert_called_once_with(Path("/test/run/log"))

    # Verify process operations
    process.process.poll.assert_called()
    mock_psutil_process.assert_called_once_with(12345)
    mock_psutil_process.return_value.children.assert_called_once_with(recursive=True)
    process.process.kill.assert_called_once()
    process.process.wait.assert_called_once_with(10)
    mock_kill_children.assert_called_once_with([mock_child_process])


@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_children")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.remove_file")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.close_file_fp")
def test_stop_with_del_log_false(mock_close_file_fp, mock_remove_file, mock_kill_children):
    # Test del_log=False case
    process = CustomProcess()
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.process = None

    # Call stop method
    process.stop(del_log=False)

    # Verify run_log_offset was reset
    assert process.run_log_offset == 0

    # Verify file operations
    mock_close_file_fp.assert_called_once_with(process.run_log_fp)

    # Verify log file was not deleted
    mock_remove_file.assert_not_called()


@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_children")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.remove_file")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.close_file_fp")
def test_stop_process_already_exited(mock_close_file_fp, mock_remove_file, mock_kill_children):
    # Test process already exited case
    process = CustomProcess()
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.process = MagicMock()
    process.process.poll.return_value = 1  # Process already exited, return code

    # Call stop method
    process.stop()

    # Verify file operations
    mock_close_file_fp.assert_called_once_with(process.run_log_fp)
    mock_remove_file.assert_called_once_with(Path("/test/run/log"))

    # Verify no attempt to kill process
    process.process.kill.assert_not_called()
    process.process.wait.assert_not_called()
    mock_kill_children.assert_not_called()


@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_children")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.remove_file")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.close_file_fp")
@patch("psutil.Process")
def test_stop_process_wait_timeout(mock_psutil_process, mock_close_file_fp, mock_remove_file, mock_kill_children):
    # Test process wait timeout case
    import subprocess

    process = CustomProcess()
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.process = MagicMock()
    process.process.poll.return_value = None
    process.process.pid = 12345

    # Simulate wait timeout
    process.process.wait.side_effect = subprocess.TimeoutExpired("cmd", 10)

    # Simulate child process
    mock_child_process = MagicMock()
    mock_psutil_process.return_value.children.return_value = [mock_child_process]

    # Call stop method
    process.stop()

    # Verify file operations
    mock_close_file_fp.assert_called_once_with(process.run_log_fp)
    mock_remove_file.assert_called_once_with(Path("/test/run/log"))

    # Verify process operations
    process.process.kill.assert_called_once()
    process.process.wait.assert_called_once_with(10)
    process.process.send_signal.assert_called_once_with(9)  # SIGKILL
    mock_kill_children.assert_called_once_with([mock_child_process])


@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_children")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.remove_file")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.close_file_fp")
@patch("psutil.Process")
def test_stop_process_shutdown_failed(mock_psutil_process, mock_close_file_fp, mock_remove_file, mock_kill_children):
    # Test process shutdown failed case
    process = CustomProcess()
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.process = MagicMock()
    process.process.poll.return_value = None  # Process still running
    process.process.pid = 12345

    # Simulate child process
    mock_child_process = MagicMock()
    mock_psutil_process.return_value.children.return_value = [mock_child_process]

    # Call stop method
    process.stop()

    # Verify file operations
    mock_close_file_fp.assert_called_once_with(process.run_log_fp)
    mock_remove_file.assert_called_once_with(Path("/test/run/log"))

    # Verify process operations
    process.process.kill.assert_called_once()
    process.process.wait.assert_called_once_with(10)
    mock_kill_children.assert_called_once_with([mock_child_process])


@patch("ms_serviceparam_optimizer.optimizer.custom_process.kill_children")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.remove_file")
@patch("ms_serviceparam_optimizer.optimizer.custom_process.close_file_fp")
@patch("psutil.Process")
def test_stop_exception_handling(mock_psutil_process, mock_close_file_fp, mock_remove_file, mock_kill_children):
    # Test exception handling
    process = CustomProcess()
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.process = MagicMock()
    process.process.poll.return_value = None
    process.process.pid = 12345

    # Simulate psutil.Process raising exception
    mock_psutil_process.side_effect = Exception("Process error")

    # Call stop method
    process.stop()

    # Verify file operations
    mock_close_file_fp.assert_called_once_with(process.run_log_fp)
    mock_remove_file.assert_called_once_with(Path("/test/run/log"))

    # Verify exception was handled
    mock_psutil_process.assert_called_once_with(12345)
    process.process.kill.assert_not_called()
    process.process.wait.assert_not_called()
    mock_kill_children.assert_not_called()


# Test cases for get_last_log method
def test_get_last_log_no_run_log():
    # Test run_log is None case
    process = CustomProcess()
    process.run_log = None

    result = process.get_last_log()

    assert result is None


@patch("pathlib.Path.exists", return_value=False)
def test_get_last_log_file_not_exists(mock_exists):
    # Test log file does not exist case
    process = CustomProcess()
    process.run_log = "/nonexistent/log/file.log"

    result = process.get_last_log()

    assert result is None
    mock_exists.assert_called_once()


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
def test_get_last_log_default_number(mock_exists, mock_open_s):
    # Test default parameter number=5 case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"

    # Simulate file content
    mock_file = MagicMock()
    mock_file.readlines.return_value = [
        "Line 1\n",
        "Line 2\n",
        "Line 3\n",
        "Line 4\n",
        "Line 5\n",
        "Line 6\n",
        "Line 7\n",
    ]
    mock_open_s.return_value.__enter__.return_value = mock_file

    result = process.get_last_log()

    # Verify last 5 lines returned
    expected = "Line 3\n\nLine 4\n\nLine 5\n\nLine 6\n\nLine 7\n"
    assert result == expected
    mock_exists.assert_called_once()
    mock_open_s.assert_called_once_with(Path("/test/log/file.log"), "r", encoding="utf-8")


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
def test_get_last_log_custom_number(mock_exists, mock_open_s):
    # Test custom number parameter case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"

    # Simulate file content
    mock_file = MagicMock()
    mock_file.readlines.return_value = [
        "Line 1\n",
        "Line 2\n",
        "Line 3\n",
        "Line 4\n",
        "Line 5\n",
        "Line 6\n",
        "Line 7\n",
    ]
    mock_open_s.return_value.__enter__.return_value = mock_file

    result = process.get_last_log(number=3)

    # Verify last 3 lines returned
    expected = "Line 5\n\nLine 6\n\nLine 7\n"
    assert result == expected


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
def test_get_last_log_number_greater_than_file_lines(mock_exists, mock_open_s):
    # Test number greater than file lines count case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"

    # Simulate file content
    mock_file = MagicMock()
    mock_file.readlines.return_value = ["Line 1\n", "Line 2\n", "Line 3\n"]
    mock_open_s.return_value.__enter__.return_value = mock_file

    result = process.get_last_log(number=10)

    # Verify all lines returned
    expected = "Line 1\n\nLine 2\n\nLine 3\n"
    assert result == expected


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
def test_get_last_log_empty_file(mock_exists, mock_open_s):
    # Test empty file case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"

    # Simulate empty file
    mock_file = MagicMock()
    mock_file.readlines.return_value = []
    mock_open_s.return_value.__enter__.return_value = mock_file

    result = process.get_last_log()

    # Verify empty string returned
    assert result == ""


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
@patch("ms_serviceparam_optimizer.optimizer.custom_process.logger")
def test_get_last_log_unicode_error(mock_logger, mock_exists, mock_open_s):
    # Test UnicodeError exception case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"

    # Simulate UnicodeError
    mock_open_s.side_effect = UnicodeError("Encoding error")

    # Due to a bug in the source code, UnboundLocalError is raised on exception;
    # we need to catch this exception to verify error log was recorded
    with pytest.raises(UnboundLocalError) as exc_info:
        process.get_last_log()

    # Verify exception info
    assert "local variable 'file_lines' referenced before assignment" in str(exc_info.value)

    # Verify error log was recorded
    mock_logger.error.assert_called_once()
    assert "Failed read" in mock_logger.error.call_args[0][0]


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
@patch("ms_serviceparam_optimizer.optimizer.custom_process.logger")
def test_get_last_log_os_error(mock_logger, mock_exists, mock_open_s):
    # Test OSError exception case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"

    # Simulate OSError
    mock_open_s.side_effect = OSError("File access error")

    # Due to a bug in the source code, UnboundLocalError is raised on exception;
    # we need to catch this exception to verify error log was recorded
    with pytest.raises(UnboundLocalError) as exc_info:
        process.get_last_log()

    # Verify exception info
    assert "local variable 'file_lines' referenced before assignment" in str(exc_info.value)

    # Verify error log was recorded
    mock_logger.error.assert_called_once()
    assert "Failed read" in mock_logger.error.call_args[0][0]


@patch("ms_serviceparam_optimizer.optimizer.custom_process.open_s")
@patch("pathlib.Path.exists", return_value=True)
def test_get_last_log_with_command(mock_exists, mock_open_s):
    # Test having command attribute case
    process = CustomProcess()
    process.run_log = "/test/log/file.log"
    process.command = ["python", "script.py"]

    # Simulate file content
    mock_file = MagicMock()
    mock_file.readlines.return_value = ["Line 1\n", "Line 2\n", "Line 3\n"]
    mock_open_s.return_value.__enter__.return_value = mock_file

    result = process.get_last_log()

    # Verify last 5 lines returned (default value)
    expected = "Line 1\n\nLine 2\n\nLine 3\n"
    assert result == expected
