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
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

from optix.config.custom_command import (
    AisBenchCommand,
    AisBenchCommandConfig,
    VllmBenchmarkCommand,
    VllmBenchmarkCommandConfig,
    VllmCommand,
    VllmCommandConfig,
)


def _require_resolve_mindie_argv() -> Callable[[Mapping[str, str]], list[str]]:
    import optix.deploy_env as deploy_env_module

    resolve = getattr(deploy_env_module, "resolve_mindie_argv", None)
    assert resolve is not None, "resolve_mindie_argv is not implemented in optix.deploy_env"
    return cast("Callable[[Mapping[str, str]], list[str]]", resolve)


class TestVllmCommand:
    def test_init_success_without_which(self) -> None:
        config = VllmCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            others="",
        )
        command = VllmCommand(config)
        assert command.command_config == config

    def test_command_property(self) -> None:
        config = VllmCommandConfig(
            host="127.0.0.1",
            port="8080",
            model="/path/to/model",
            served_model_name="my-model",
            others="--gpu-memory-utilization 0.9",
        )
        cmd_obj = VllmCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "vllm"
        assert "serve" in cmd
        assert "/path/to/model" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd

    def test_command_no_others(self) -> None:
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        cmd_obj = VllmCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "vllm"
        assert "--gpu-memory-utilization" not in cmd


class TestVllmBenchmarkCommand:
    def test_init_success_without_which(self) -> None:
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
        assert cmd_obj.benchmark_command_config == config

    def test_command_property(self) -> None:
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
        assert cmd[0] == "vllm"
        assert "bench" in cmd
        assert "serve" in cmd
        assert "--save-result" in cmd
        assert "--extra-arg" in cmd
        assert "value" in cmd
        assert "$CONCURRENCY" in cmd
        assert "$REQUESTRATE" in cmd


class TestAisBenchCommand:
    def test_init_success_without_which(self) -> None:
        config = AisBenchCommandConfig(
            models="model1",
            mode="perf",
            work_dir="/work",
            others="",
        )
        cmd_obj = AisBenchCommand(config)
        assert cmd_obj.aisbench_command_config == config

    def test_command_property(self) -> None:
        config = AisBenchCommandConfig(
            models="model1",
            mode="perf",
            work_dir="/work",
            others="--extra-arg value",
        )
        cmd_obj = AisBenchCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "ais_bench"
        assert "--models" in cmd
        assert "model1" in cmd
        assert "--mode" in cmd
        assert "perf" in cmd
        assert "--work-dir" in cmd
        assert "/work" in cmd
        assert "--debug" in cmd
        assert "--extra-arg" in cmd
        assert "value" in cmd

    def test_command_no_others(self):
        """Test command when others is empty"""
        config = AisBenchCommandConfig(
            models="model1",
            mode="perf",
            work_dir="/work",
            others="",
        )
        cmd_obj = AisBenchCommand(config)
        cmd = cmd_obj.command
        assert "--extra-arg" not in cmd


class TestResolveMindieArgv:
    @patch("optix.deploy_env.os.path.isfile")
    def test_default_path_exists(self, mock_isfile: Any) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        mock_isfile.return_value = True
        argv = resolve_mindie_argv({})
        assert "mindieservice_daemon" in argv[0]

    @patch("optix.deploy_env.shutil.which")
    @patch("optix.deploy_env.os.path.isfile")
    def test_fallback_to_mindie_llm_server_in_deploy_path(
        self, mock_isfile: Any, mock_which: Any, tmp_path: Path
    ) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        deploy_bin = tmp_path / "deploy" / "bin"
        deploy_bin.mkdir(parents=True)
        mindie_server = deploy_bin / "mindie_llm_server"
        mindie_server.write_text("#!/bin/sh\n", encoding="utf-8")
        mindie_server.chmod(0o755)

        mock_isfile.return_value = False
        mock_which.return_value = str(mindie_server.resolve())
        env = {"PATH": f"{deploy_bin}:/usr/bin"}
        argv = resolve_mindie_argv(env)
        assert argv == [str(mindie_server.resolve())]

    @patch("optix.deploy_env.shutil.which")
    @patch("optix.deploy_env.os.path.isfile")
    def test_raises_when_no_command_found(self, mock_isfile: Any, mock_which: Any) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        mock_isfile.return_value = False
        mock_which.return_value = None
        with pytest.raises(FileNotFoundError):
            resolve_mindie_argv({"PATH": "/usr/bin"})

    @patch("optix.deploy_env.os.path.isfile")
    def test_custom_install_path(self, mock_isfile: Any, tmp_path: Path) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        custom_root = tmp_path / "custom" / "mindie-service"
        custom_daemon = custom_root / "bin" / "mindieservice_daemon"
        custom_daemon.parent.mkdir(parents=True)
        custom_daemon.write_text("#!/bin/sh\n", encoding="utf-8")
        custom_daemon.chmod(0o755)

        mock_isfile.side_effect = lambda path: Path(path) == custom_daemon
        env = {"MIES_INSTALL_PATH": str(custom_root), "PATH": "/usr/bin"}
        argv = resolve_mindie_argv(env)
        assert str(custom_daemon) in argv[0]
