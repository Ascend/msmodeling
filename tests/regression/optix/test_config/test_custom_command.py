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
from unittest.mock import patch
import pytest

from optix.config.custom_command import (
    VllmCommand,
    VllmCommandConfig,
    VllmBenchmarkCommand,
    VllmBenchmarkCommandConfig,
    AisBenchCommand,
    AisBenchCommandConfig,
    MindieCommand,
    MindieCommandConfig,
)


class TestVllmCommand:
    @patch("shutil.which")
    def test_init_success(self, mock_which):
        """Test successful initialization when vllm is found in PATH"""
        mock_which.return_value = "/usr/bin/vllm"
        config = VllmCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            others="",
        )
        command = VllmCommand(config)
        assert command.process == "/usr/bin/vllm"
        assert command.command_config == config

    @patch("shutil.which")
    def test_init_failure_vllm_not_found(self, mock_which):
        """Test initialization fails when vllm is not found in PATH"""
        mock_which.return_value = None
        config = VllmCommandConfig()
        with pytest.raises(ValueError) as excinfo:
            VllmCommand(config)
        assert "Error: The 'vllm' executable was not found in the system PATH." in str(excinfo.value)

    @patch("shutil.which")
    def test_command_property(self, mock_which):
        """Test command property generates correct command list"""
        mock_which.return_value = "/usr/bin/vllm"
        config = VllmCommandConfig(
            host="127.0.0.1",
            port="8080",
            model="/path/to/model",
            served_model_name="my-model",
            others="--gpu-memory-utilization 0.9",
        )
        cmd_obj = VllmCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "/usr/bin/vllm"
        assert "serve" in cmd
        assert "/path/to/model" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd

    @patch("shutil.which")
    def test_command_no_others(self, mock_which):
        """Test command when others is empty"""
        mock_which.return_value = "/usr/bin/vllm"
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        cmd_obj = VllmCommand(config)
        cmd = cmd_obj.command
        assert "--gpu-memory-utilization" not in cmd


class TestVllmBenchmarkCommand:
    @patch("shutil.which")
    def test_init_success(self, mock_which):
        mock_which.return_value = "/usr/bin/vllm"
        config = VllmBenchmarkCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            dataset_name="sharegpt",
            num_prompts=100,
            result_dir="/tmp/results",
            others="",
        )
        cmd_obj = VllmBenchmarkCommand(config)
        assert cmd_obj.process == "/usr/bin/vllm"

    @patch("shutil.which")
    def test_init_failure(self, mock_which):
        mock_which.return_value = None
        config = VllmBenchmarkCommandConfig(num_prompts=10)
        with pytest.raises(ValueError):
            VllmBenchmarkCommand(config)

    @patch("shutil.which")
    def test_command_property(self, mock_which):
        mock_which.return_value = "/usr/bin/vllm"
        config = VllmBenchmarkCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            dataset_name="sharegpt",
            num_prompts=100,
            result_dir="/tmp/results",
            others="--extra-arg value",
        )
        cmd_obj = VllmBenchmarkCommand(config)
        cmd = cmd_obj.command
        assert "bench" in cmd
        assert "serve" in cmd
        assert "--num-prompts" in cmd
        assert "100" in cmd
        assert "--save-result" in cmd
        assert "--extra-arg" in cmd
        assert "value" in cmd
        assert "$CONCURRENCY" in cmd
        assert "$REQUESTRATE" in cmd


class TestAisBenchCommand:
    @patch("shutil.which")
    def test_init_success(self, mock_which):
        mock_which.return_value = "/usr/bin/ais_bench"
        config = AisBenchCommandConfig(
            models="model1",
            datasets="ds1",
            mode="perf",
            num_prompts=50,
            work_dir="/work",
        )
        cmd_obj = AisBenchCommand(config)
        assert cmd_obj.process == "/usr/bin/ais_bench"

    @patch("shutil.which")
    def test_init_failure(self, mock_which):
        mock_which.return_value = None
        config = AisBenchCommandConfig(num_prompts=50)
        with pytest.raises(ValueError, match="ais_bench"):
            AisBenchCommand(config)

    @patch("shutil.which")
    def test_command_property(self, mock_which):
        mock_which.return_value = "/usr/bin/ais_bench"
        config = AisBenchCommandConfig(
            models="model1",
            datasets="ds1",
            mode="perf",
            num_prompts=50,
            work_dir="/work",
        )
        cmd_obj = AisBenchCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "/usr/bin/ais_bench"
        assert "--models" in cmd
        assert "model1" in cmd
        assert "--datasets" in cmd
        assert "--mode" in cmd
        assert "--num-prompts" in cmd
        assert "50" in cmd
        assert "--debug" in cmd


class TestMindieCommand:
    @patch("os.path.isfile")
    @patch("shutil.which")
    def test_default_path_exists(self, mock_which, mock_isfile):
        mock_isfile.return_value = True
        config = MindieCommandConfig()
        cmd_obj = MindieCommand(config)
        cmd = cmd_obj.command
        assert "mindieservice_daemon" in cmd[0]

    @patch("os.path.isfile")
    @patch("shutil.which")
    def test_fallback_to_new_command(self, mock_which, mock_isfile):
        mock_isfile.return_value = False
        mock_which.return_value = "/usr/bin/mindie_llm_server"
        config = MindieCommandConfig()
        cmd_obj = MindieCommand(config)
        cmd = cmd_obj.command
        assert cmd == ["mindie_llm_server"]

    @patch("os.path.isfile")
    @patch("shutil.which")
    def test_raises_when_no_command_found(self, mock_which, mock_isfile):
        mock_isfile.return_value = False
        mock_which.return_value = None
        config = MindieCommandConfig()
        cmd_obj = MindieCommand(config)
        with pytest.raises(FileNotFoundError):
            _ = cmd_obj.command

    @patch("os.path.isfile")
    def test_custom_install_path(self, mock_isfile, monkeypatch):
        monkeypatch.setenv("MIES_INSTALL_PATH", "/custom/path")
        mock_isfile.return_value = True
        config = MindieCommandConfig()
        cmd_obj = MindieCommand(config)
        cmd = cmd_obj.command
        assert "/custom/path" in cmd[0]
