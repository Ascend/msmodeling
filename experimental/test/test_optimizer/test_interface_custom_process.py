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
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from experimental.optix.config.config import (
    CUSTOM_OUTPUT,
    MODEL_EVAL_STATE_CONFIG_PATH,
    OptimizerConfigField,
)
from experimental.optix.optimizer.interfaces.custom_process import (
    CustomProcess,
    BaseDataField,
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
@patch("experimental.optix.optimizer.interfaces.custom_process.kill_process")
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
@patch("experimental.optix.optimizer.interfaces.custom_process.kill_process")
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
@patch("experimental.optix.optimizer.interfaces.custom_process.kill_process")
def test_check_env_kill_process_exception(mock_kill_process, mock_process_iter):
    # Mock exception when trying to kill process
    mock_process_iter.return_value = [MagicMock(info={"pid": 1, "name": "target_process"})]
    mock_kill_process.side_effect = Exception("Failed to kill process")

    CustomProcess.kill_residual_process("target_process")

    # Ensure kill_process was called, and exception was caught
    mock_kill_process.assert_called_once_with("target_process")


# Test case 1: process_name exists and check_env succeeds
@patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.kill_residual_process")
@patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("experimental.optix.optimizer.interfaces.custom_process.subprocess.Popen")
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
@patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.kill_residual_process")
@patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("experimental.optix.optimizer.interfaces.custom_process.subprocess.Popen")
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
@patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("experimental.optix.optimizer.interfaces.custom_process.subprocess.Popen")
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
@patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.before_run")
@patch("experimental.optix.optimizer.interfaces.custom_process.subprocess.Popen")
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
