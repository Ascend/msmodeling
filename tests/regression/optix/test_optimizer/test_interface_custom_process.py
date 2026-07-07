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
import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from loguru import logger

from optix.config.config import (
    CUSTOM_OUTPUT,
    MODEL_EVAL_STATE_CONFIG_PATH,
    OptimizerConfigField,
)
from optix.optimizer.interfaces.custom_process import (
    BaseDataField,
    CustomProcess,
)


def test_before_run_no_run_params(monkeypatch):
    # Mock tempfile.mkstemp
    monkeypatch.setattr(tempfile, "mkstemp", lambda prefix="": (1234, "tempfile"))
    # Mock os.environ
    monkeypatch.setattr(os, "environ", {})
    process = CustomProcess()
    process.before_run()

    # Verify attributes are set
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
    process.stop(del_log=True)


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
    process.stop(del_log=True)


def test_check_success_process_still_running(tmpdir):
    # Mock subprocess still running
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w", encoding="utf-8") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = None
    custom_process.print_log = True


def test_check_success_process_succeeded(tmpdir):
    # Mock subprocess completed successfully
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w", encoding="utf-8") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 0
    custom_process.print_log = True


@patch("psutil.process_iter")
@patch("optix.optimizer.interfaces.custom_process.kill_process")
def test_check_env_no_residual_process(mock_kill_process, mock_process_iter):
    # Mock no residual processes
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
@patch("optix.optimizer.interfaces.custom_process.kill_process")
def test_check_env_with_residual_process(mock_kill_process, mock_process_iter):
    # Mock with residual processes
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
@patch("optix.optimizer.interfaces.custom_process.kill_process")
def test_check_env_kill_process_exception(mock_kill_process, mock_process_iter):
    # Mock exception when trying to kill process
    mock_process_iter.return_value = [MagicMock(info={"pid": 1, "name": "target_process"})]
    mock_kill_process.side_effect = Exception("Failed to kill process")

    CustomProcess.kill_residual_process("target_process")

    # Ensure kill_process was called, and exception was caught
    mock_kill_process.assert_called_once_with("target_process")


# Test case 1: process_name exists and check_env succeeds
@patch("optix.optimizer.interfaces.custom_process.CustomProcess.kill_residual_process")
@patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
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
@patch("optix.optimizer.interfaces.custom_process.CustomProcess.kill_residual_process")
@patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
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


@patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
def test_run_logs_multiline_command_and_log(mock_popen, mock_before_run):
    mock_popen.return_value = MagicMock(pid=4242)
    process = CustomProcess()
    process.process_name = None
    process.command = ["vllm", "serve", "model_path", "--port", "8080"]
    process.work_path = "/test/work/path"
    process.run_log_fp = MagicMock()
    process.run_log = "/tmp/ms_serviceparam_optimizer__abc"

    buffer = StringIO()
    handler_id = logger.add(buffer, level="TRACE")
    process.run()
    logger.remove(handler_id)

    output = buffer.getvalue()
    assert "  command:" in output
    assert "  log: /tmp/ms_serviceparam_optimizer__abc" in output
    assert "vllm serve model_path --port 8080" in output


# Test case 3: process_name does not exist
@patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
def test_run_process_name_not_exists(mock_popen, mock_before_run):
    process = CustomProcess()
    process.process_name = None
    process.command = ["test_command"]
    process.work_path = "/test/work/path"
    process.run_log_fp = MagicMock()
    process.run_log = "/test/run/log"
    process.run()
    mock_before_run.assert_called_once()


@patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
def test_run_popen_failure_closes_run_log_fd(mock_popen):
    """FD from mkstemp must close when Popen fails after before_run."""
    process = CustomProcess()
    process.process_name = None
    process.command = ["test_command"]
    process.work_path = "/tmp"
    mock_popen.side_effect = OSError("subprocess.Popen failed")
    with pytest.raises(OSError, match="subprocess.Popen failed"):
        process.run()
    assert process.run_log_fp is None


# Test case 4: subprocess.Popen raises OSError
@patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
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


class TestCustomProcessSupplement:
    def test_get_log_file_not_found(self):
        """Test get_log method when file does not exist"""
        process = CustomProcess()
        process.run_log = "/nonexistent/path/test.log"

        result = process.get_log()

        assert result is None

    def test_get_log_reads_content(self, tmp_path):
        """Test get_log reads file from offset"""
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3\n")
        process = CustomProcess()
        process.run_log = str(log_file)
        process.run_log_offset = 0
        result = process.get_log()
        assert "line1" in result
        assert process.run_log_offset > 0

    def test_get_last_log_returns_last_lines(self, tmp_path):
        """Test get_last_log returns the last N lines"""
        log_file = tmp_path / "test.log"
        lines = [f"line{i}\n" for i in range(20)]
        log_file.write_text("".join(lines))
        process = CustomProcess()
        process.run_log = str(log_file)
        result = process.get_last_log(number=3)
        assert "line17" in result
        assert "line18" in result
        assert "line19" in result

    def test_get_last_log_none_when_no_log(self):
        """Test get_last_log returns None when no run_log"""
        process = CustomProcess()
        process.run_log = None
        assert process.get_last_log() is None

    def test_get_last_log_nonexistent_file(self):
        """Test get_last_log returns None for nonexistent file"""
        process = CustomProcess()
        process.run_log = "/nonexistent/path.log"
        assert process.get_last_log() is None

    @patch("optix.optimizer.interfaces.custom_process.time.sleep")
    def test_get_last_log_retry_false_no_sleep(self, mock_sleep, tmp_path):
        """Health-check path must not block on empty logs."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("", encoding="utf-8")
        process = CustomProcess()
        process.run_log = str(log_file)
        process.process = Mock()
        process.process.poll.return_value = None

        result = process.get_last_log(number=3, retry=False)

        assert result is None
        mock_sleep.assert_not_called()

    def test_health_process_running(self):
        """Test health returns running when process.poll() is None"""
        process = CustomProcess()
        process.process = Mock()
        process.process.poll.return_value = None
        process.print_log = False
        result = process.health()
        assert result.stage.value == "running"

    def test_health_process_stopped(self):
        """Test health returns stop when process.poll() == 0"""
        process = CustomProcess()
        process.process = Mock()
        process.process.poll.return_value = 0
        process.print_log = False
        result = process.health()
        assert result.stage.value == "stop"

    def test_health_process_error(self):
        """Test health returns error when process has non-zero return code"""
        process = CustomProcess()
        process.process = Mock()
        process.process.poll.return_value = 1
        process.process.returncode = 1
        process.print_log = False
        process.command = ["vllm", "serve", "model_path"]
        process.run_log = "/tmp/test.log"
        result = process.health()
        assert result.stage.value == "error"
        assert "exit_code=1" in result.info
        assert "log=/tmp/test.log" in result.info
        assert "vllm" not in result.info
        assert "!r" not in result.info

    @patch("psutil.Process")
    def test_stop_running_process(self, mock_psutil_process):
        """Test stop kills running process"""
        process = CustomProcess()
        process.process = Mock()
        process.process.poll.return_value = None
        process.process.pid = 12345
        process.process.kill = Mock()
        process.process.wait = Mock()
        process.run_log_fp = None
        process.run_log = None
        mock_parent = Mock()
        mock_parent.children.return_value = []
        mock_psutil_process.return_value = mock_parent
        process.stop(del_log=False)
        process.process.kill.assert_called()

    def test_stop_already_exited_process(self):
        """Test stop when process already exited"""
        process = CustomProcess()
        process.process = Mock()
        process.process.poll.return_value = 0
        process.run_log_fp = None
        process.run_log = None
        process.stop(del_log=False)

    def test_stop_no_process(self):
        """Test stop when process is None"""
        process = CustomProcess()
        process.process = None
        process.run_log_fp = None
        process.run_log = None
        process.stop(del_log=False)

    def test_process_stage_property(self):
        """Test process_stage getter and setter"""
        from optix.config.constant import ProcessState, Stage

        process = CustomProcess()
        assert process.process_stage.stage == Stage.stop
        new_state = ProcessState(stage=Stage.running)
        process.process_stage = new_state
        assert process.process_stage.stage == Stage.running
        # Setting same stage should not change
        process.process_stage = ProcessState(stage=Stage.running)
        assert process.process_stage.stage == Stage.running

    def test_split_merged_args_json(self):
        """Test _split_merged_args splits JSON params correctly"""
        process = CustomProcess()
        process.command = [
            "vllm",
            "serve",
            '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'',
        ]
        process._split_merged_args()
        assert "--compilation-config" in process.command
        assert '{"cudagraph_mode": "FULL_DECODE_ONLY"}' in process.command

    def test_split_merged_args_no_json(self):
        """Test _split_merged_args leaves non-JSON args unchanged"""
        process = CustomProcess()
        process.command = ["vllm", "serve", "--port", "8000"]
        original = list(process.command)
        process._split_merged_args()
        assert process.command == original

    def test_split_merged_args_fullwidth_chars(self):
        """Test _split_merged_args handles fullwidth unicode characters"""
        process = CustomProcess()
        process.command = [
            "vllm",
            "serve",
            '--config \'{"key"\uff1a "value"}\'',
        ]
        process._split_merged_args()
        assert "--config" in process.command

    def test_split_merged_args_no_quotes_json(self):
        """Test _split_merged_args with no-quote JSON"""
        process = CustomProcess()
        process.command = [
            "vllm",
            '--config {"mode": "full"}',
        ]
        process._split_merged_args()
        assert "--config" in process.command

    def test_split_merged_args_non_string_element(self):
        """Test _split_merged_args with non-string elements"""
        process = CustomProcess()
        process.command = ["vllm", 123, "--port", "8000"]
        process._split_merged_args()
        assert 123 in process.command

    def test_split_merged_args_escaped_json(self):
        """Test _split_merged_args with escaped JSON string"""
        process = CustomProcess()
        process.command = [
            "vllm",
            '--compilation-config "{\\"mode\\": \\"FULL_DECODE_ONLY\\"}"',
        ]
        process._split_merged_args()
        assert "--compilation-config" in process.command

    def test_get_log_read_error(self, tmp_path):
        """Test get_log handles OSError during read"""
        log_file = tmp_path / "test.log"
        log_file.write_text("content")
        process = CustomProcess()
        process.run_log = str(log_file)
        process.run_log_offset = 0
        with patch(
            "optix.optimizer.interfaces.custom_process.open_file",
            side_effect=OSError("fail"),
        ):
            result = process.get_log()
        assert result is None

    def test_stop_timeout_sends_signal_9(self):
        """Test stop sends signal 9 on TimeoutExpired"""
        import subprocess as sp

        process = CustomProcess()
        process.process = MagicMock()
        process.process.poll.side_effect = [None, None, 0]
        process.process.pid = 12345
        process.process.kill = MagicMock()
        process.process.wait = MagicMock(side_effect=sp.TimeoutExpired("cmd", 10))
        process.process.send_signal = MagicMock()
        process.run_log_fp = None
        process.run_log = None
        mock_parent = MagicMock()
        mock_parent.children.return_value = []
        with patch("psutil.Process", return_value=mock_parent):
            process.stop(del_log=False)
        process.process.send_signal.assert_called_once_with(9)

    def test_stop_exception_sets_error_stage(self):
        """Test stop handles exception and sets error stage"""
        process = CustomProcess()
        process.process = MagicMock()
        process.process.poll.return_value = None
        process.process.pid = 12345
        process.run_log_fp = None
        process.run_log = None
        with patch("psutil.Process", side_effect=Exception("process not found")):
            process.stop(del_log=False)
        from optix.config.constant import Stage

        assert process.process_stage.stage == Stage.error

    def test_before_run_removes_empty_env_var(self):
        """Test before_run removes env var when value is NaN (invalid)"""
        process = CustomProcess()
        process.command = ["benchmark", "--request-rate", "$REQUESTRATE"]
        process.env = {"REQUESTRATE": "old_value"}
        run_params = (
            OptimizerConfigField(
                name="REQUESTRATE",
                config_position="env",
                min=0,
                max=100,
                dtype="float",
                value=float("nan"),
            ),
        )
        process.before_run(run_params)
        assert "REQUESTRATE" not in process.env
        assert "--request-rate" not in process.command
        assert "$REQUESTRATE" not in process.command
        process.stop(del_log=True)


class TestBaseDataField:
    @pytest.mark.parametrize(
        "target_field,expected",
        [
            ([], ()),
            (
                [
                    OptimizerConfigField(
                        name="field1",
                        config_position="pos1",
                        min=0,
                        max=100,
                        dtype="int",
                    )
                ],
                (
                    OptimizerConfigField(
                        name="field1",
                        config_position="pos1",
                        min=0,
                        max=100,
                        dtype="int",
                    ),
                ),
            ),
            (None, ()),
        ],
    )
    def test_data_field_property(self, target_field, expected):
        """Test data_field property getter"""
        mock_config = MagicMock()
        if target_field is not None:
            mock_config.target_field = target_field
        else:
            delattr(mock_config, "target_field")

        data_field = BaseDataField(config=mock_config)

        result = data_field.data_field

        assert result == expected

    def test_data_field_setter_add_new_field(self):
        """Test data_field property setter to add new field"""
        mock_config = MagicMock()
        mock_config.target_field = [
            OptimizerConfigField(
                name="existing_field",
                config_position="existing.position",
                min=0,
                max=100,
                dtype="int",
            )
        ]

        data_field = BaseDataField(config=mock_config)
        new_fields = (
            OptimizerConfigField(
                name="new_field",
                config_position="new.position",
                min=0,
                max=50,
                dtype="float",
            ),
            OptimizerConfigField(
                name="existing_field",
                config_position="updated.position",
                min=0,
                max=200,
                dtype="int",
            ),
        )

        # Set new field
        data_field.data_field = new_fields

        # Verify only existing_field was updated, new_field was ignored
        assert len(mock_config.target_field) == 1
        assert mock_config.target_field[0].name == "existing_field"
        assert mock_config.target_field[0].config_position == "updated.position"
        assert mock_config.target_field[0].max == 200

    def test_data_field_setter_empty_input(self):
        """Test data_field property setter with empty input"""
        mock_config = MagicMock()
        mock_config.target_field = [
            OptimizerConfigField(name="field1", config_position="pos1", min=0, max=100, dtype="int")
        ]

        data_field = BaseDataField(config=mock_config)

        # Set empty field
        data_field.data_field = ()

        # Verify original field was not modified
        assert len(mock_config.target_field) == 1
        assert mock_config.target_field[0].name == "field1"

    def test_data_field_setter_no_target_field(self):
        """Test data_field property setter when config has no target_field"""
        mock_config = MagicMock()
        # Mock config has no target_field attribute
        if hasattr(mock_config, "target_field"):
            delattr(mock_config, "target_field")

        data_field = BaseDataField(config=mock_config)
        new_fields = (
            OptimizerConfigField(
                name="test_field",
                config_position="test.position",
                min=0,
                max=100,
                dtype="int",
            ),
        )

        # Set new field (should have no effect)
        data_field.data_field = new_fields

        # Verify config was not modified
        assert not hasattr(mock_config, "target_field") or mock_config.target_field == []


class TestBeforeRunEnvHandling:
    """Test before_run env variable handling edge cases"""

    def test_non_positive_requestrate_removed_from_command(self):
        """Test NON_POSITIVE_INVALID_FIELDS with non-positive value removes CLI flag"""
        process = CustomProcess()
        process.command = ["benchmark", "--request-rate", "$REQUESTRATE"]
        run_params = (
            OptimizerConfigField(
                name="REQUESTRATE",
                config_position="env",
                min=0,
                max=100,
                dtype="float",
                value=-1.0,  # Non-positive value
            ),
        )
        process.before_run(run_params)
        # Non-positive REQUESTRATE should be removed along with --request-rate flag
        assert "--request-rate" not in process.command
        assert "$REQUESTRATE" not in process.command
        process.stop(del_log=True)

    def test_positive_requestrate_set_in_command(self):
        """Test positive REQUESTRATE value is set in command"""
        process = CustomProcess()
        process.command = ["benchmark", "--request-rate", "$REQUESTRATE"]
        run_params = (
            OptimizerConfigField(
                name="REQUESTRATE",
                config_position="env",
                min=0.1,
                max=100,
                dtype="float",
                value=5.0,
            ),
        )
        process.before_run(run_params)
        assert "5.0" in process.command
        assert "$REQUESTRATE" not in process.command
        process.stop(del_log=True)

    def test_env_nan_value_removed(self):
        """Test env with NaN value is considered invalid and removed"""
        process = CustomProcess()
        process.command = ["benchmark", "$MYVAR"]
        process.env = {"MYVAR": "old_value"}
        run_params = (
            OptimizerConfigField(
                name="MYVAR",
                config_position="env",
                min=0,
                max=100,
                dtype="float",
                value=float("nan"),
            ),
        )
        process.before_run(run_params)
        assert "MYVAR" not in process.env
        assert "$MYVAR" not in process.command
        process.stop(del_log=True)

    def test_env_inf_value_removed(self):
        """Test env with Inf value is considered invalid and removed"""
        process = CustomProcess()
        process.command = ["benchmark", "$MYVAR"]
        process.env = {"MYVAR": "old_value"}
        run_params = (
            OptimizerConfigField(
                name="MYVAR",
                config_position="env",
                min=0,
                max=1000,
                dtype="float",
                value=float("inf"),
            ),
        )
        process.before_run(run_params)
        assert "MYVAR" not in process.env
        process.stop(del_log=True)

    def test_env_string_empty_removed(self):
        """Test env with empty string value is considered invalid"""
        process = CustomProcess()
        process.command = ["benchmark", "$MYVAR"]
        process.env = {"MYVAR": "old_value"}
        run_params = (
            OptimizerConfigField(
                name="MYVAR",
                config_position="env",
                min=0,
                max=100,
                dtype="str",
                value="   ",  # whitespace-only string
            ),
        )
        process.before_run(run_params)
        assert "MYVAR" not in process.env
        process.stop(del_log=True)

    def test_variable_substitution_in_json_string(self):
        """Test variable substitution inside JSON string parameters"""
        process = CustomProcess()
        process.command = [
            "vllm",
            "--config",
            '{"num_tokens": $NUM_TOKENS, "mode": "full"}',
        ]
        run_params = (
            OptimizerConfigField(
                name="NUM_TOKENS",
                config_position="env",
                min=1,
                max=1000,
                dtype="int",
                value=128,
            ),
        )
        process.before_run(run_params)
        assert '{"num_tokens": 128, "mode": "full"}' in process.command
        process.stop(del_log=True)


class TestRunCommandChecks:
    """Test run() method command validation and edge cases"""

    @patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
    @patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
    def test_run_duplicate_flag_logs_warning(self, mock_popen, mock_before_run):
        """Test run() logs warning when command has duplicate flags"""
        process = CustomProcess()
        process.process_name = None
        process.command = ["vllm", "--port", "8000", "--port", "9000"]
        process.work_path = "/test"
        process.run_log_fp = MagicMock()
        process.run_log = "/tmp/log"
        # Should not raise, just log warning
        process.run()
        mock_popen.assert_called_once()

    @patch("optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
    @patch("optix.optimizer.interfaces.custom_process.subprocess.Popen")
    def test_run_non_string_env_logs_error(self, mock_popen, mock_before_run):
        """Test run() logs error for non-string env values"""
        process = CustomProcess()
        process.process_name = None
        process.command = ["test_cmd"]
        process.work_path = "/test"
        process.run_log_fp = MagicMock()
        process.run_log = "/tmp/log"
        process.env = {"KEY": 123}  # Non-string value
        process.run()
        mock_popen.assert_called_once()

    def test_backup_calls_utility(self, tmp_path):
        """Test backup method delegates to backup utility"""
        process = CustomProcess()
        process.run_log = str(tmp_path / "test.log")
        process.bak_path = str(tmp_path / "backup")
        with patch("optix.optimizer.interfaces.custom_process.backup") as mock_backup:
            process.backup()
            mock_backup.assert_called_once_with(process.run_log, process.bak_path, "CustomProcess")

    def test_get_last_log_encoding_fallback(self, tmp_path):
        """Test get_last_log tries multiple encodings"""
        log_file = tmp_path / "test.log"
        # Write content with utf-8
        log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
        process = CustomProcess()
        process.run_log = str(log_file)
        result = process.get_last_log(number=2)
        assert "line2" in result
        assert "line3" in result

    def test_get_last_log_reads_after_subprocess_write(self, tmp_path):
        import subprocess

        log_file = tmp_path / "subprocess.log"
        process = CustomProcess()
        process.run_log = str(log_file)
        fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        process.run_log_fp = fd
        subprocess.run(
            ["/usr/bin/env", "python3", "-c", "import sys; print('vllm error line', file=sys.stderr)"],
            stdout=fd,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
        tail = process.get_last_log(number=5)
        os.close(fd)
        process.run_log_fp = None
        assert tail is not None
        assert "vllm error line" in tail
