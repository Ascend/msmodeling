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

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

from optix.config.base_config import CUSTOM_OUTPUT
from optix.deploy_env import (
    OptixDeployEnvError,
    RuntimeContext,
    build_deploy_env,
    detect_runtime_context,
    emit_runtime_hints,
    materialize_command,
    resolve_deploy_path_prefix,
    validate_deploy_stack,
)
from optix.optimizer.interfaces.custom_process import CustomProcess


def _write_executable_stub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def test_custom_process_env_is_deploy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    venv_root = tmp_path / ".venv"
    parent_env = {**os.environ, "VIRTUAL_ENV": str(venv_root), "PATH": f"{venv_root / 'bin'}:/usr/bin"}
    monkeypatch.setattr(os, "environ", parent_env)

    process = CustomProcess()

    assert "VIRTUAL_ENV" not in process.env
    assert os.environ.get("VIRTUAL_ENV") == str(venv_root)


def test_vllm_only_in_deploy_path_passes_validate_and_materialize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv_root = tmp_path / ".venv"
    venv_bin = venv_root / "bin"
    venv_bin.mkdir(parents=True)

    deploy_root = tmp_path / "deploy"
    deploy_vllm = _write_executable_stub(deploy_root / "bin" / "vllm")
    _write_executable_stub(deploy_root / "bin" / "ais_bench")

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_root))
    monkeypatch.setenv("OPTIX_DEPLOY_PATH", str(deploy_root))
    monkeypatch.setenv("PATH", f"{venv_bin}:/usr/bin:/bin")

    from optix.config.custom_command import VllmCommand, VllmCommandConfig

    ctx = detect_runtime_context()
    deploy_env = build_deploy_env(os.environ, deploy_path_prefix=resolve_deploy_path_prefix())

    validate_deploy_stack(engine="vllm", benchmark="ais_bench", env=deploy_env, ctx=ctx)

    cmd_obj = VllmCommand(
        VllmCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            others="",
        )
    )
    assert cmd_obj.command[0] == "vllm"

    materialized = materialize_command(cmd_obj.command, deploy_env, ctx)
    assert materialized[0] == str(deploy_vllm)


def test_ais_bench_get_models_config_path_uses_deploy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    venv_root = tmp_path / ".venv"
    deploy_root = tmp_path / "deploy"
    deploy_ais_bench = _write_executable_stub(deploy_root / "bin" / "ais_bench")

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_root))
    monkeypatch.setenv("OPTIX_DEPLOY_PATH", str(deploy_root))
    monkeypatch.setenv("PATH", f"{venv_root / 'bin'}:/usr/bin")

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return MagicMock(
            returncode=0,
            stdout="Type │ --models │ /path/config.py │ extra\n",
            stderr="",
        )

    with (
        patch("optix.optimizer.plugins.benchmark.subprocess.run", side_effect=_fake_run),
        patch("optix.optimizer.plugins.benchmark.open_file") as mock_open_file,
    ):
        mock_open_file.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value="data")))
        mock_open_file.return_value.__exit__ = MagicMock(return_value=False)

        from optix.optimizer.plugins.benchmark import AisBench

        mock_config = MagicMock()
        mock_config.work_path = "/work"
        mock_config.command.models = "model1"
        mock_config.command.datasets = "ds1"
        mock_config.command.mode = "perf"
        mock_config.command.num_prompts = 100
        mock_config.command.work_dir = "/work"
        mock_config.output_path = "/output"

        bench = AisBench(config=mock_config)

    captured_env = cast("Mapping[str, str]", captured["env"])
    captured_cmd = cast("list[str]", captured["cmd"])
    assert "VIRTUAL_ENV" not in captured_env
    assert os.environ.get("VIRTUAL_ENV") == str(venv_root)
    assert captured_cmd[0] == str(deploy_ais_bench)
    deploy_path = captured_env["PATH"].split(os.pathsep)[0]
    assert deploy_path == str(deploy_root / "bin")
    assert bench.command is not None
    assert bench.command[0] == str(deploy_ais_bench)


def test_vllm_simulator_pkill_uses_deploy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    venv_root = tmp_path / ".venv"
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_root))
    monkeypatch.setenv("PATH", f"{venv_root / 'bin'}:/usr/bin")

    captured_envs: list[dict[str, str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        captured_envs.append(kwargs.get("env", {}))
        return MagicMock(returncode=0, stdout="0", stderr="")

    mock_config = MagicMock()
    mock_config.process_name = "vllm"
    mock_config.command = MagicMock()
    mock_config.command.host = "localhost"
    mock_config.command.port = "8000"
    mock_config.command.model = "gpt2"
    mock_config.command.served_model_name = "gpt2"
    mock_config.command.others = ""

    def _which(name: str, path: str | None = None) -> str | None:
        if name in ("vllm", "pkill", "pgrep"):
            return f"/usr/bin/{name}"
        return None

    with (
        patch("optix.deploy_env.shutil.which", side_effect=_which),
        patch("optix.optimizer.plugins.simulate.shutil.which", side_effect=_which),
        patch("optix.optimizer.plugins.simulate.subprocess.run", side_effect=_fake_run),
    ):
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(mock_config)
        with patch.object(simulator, "_is_vllm_running", side_effect=[True, False]):
            simulator._stop_vllm_process(max_attempts=1, timeout=0)

    assert captured_envs
    for env in captured_envs:
        assert env is not None
        assert env == simulator.env
        assert "VIRTUAL_ENV" not in env
    assert os.environ.get("VIRTUAL_ENV") == str(venv_root)


def _run_optimizer_main_expect_deploy_error(monkeypatch: pytest.MonkeyPatch, argv: list[str], match: str) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    pso_called: list[bool] = []

    class _FakePSO:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run_plugin(self) -> None:
            pso_called.append(True)

    with (
        patch("optix.configure_logger", lambda: None),
        patch("cli.logo.print_logo", lambda: None),
        patch("optix.optimizer.optimizer.is_root", return_value=False),
        patch("optix.deploy_env.shutil.which", return_value=None),
        patch("optix.optimizer.optimizer.PSOOptimizer", _FakePSO),
    ):
        from optix.optimizer.optimizer import main

        with pytest.raises(OptixDeployEnvError, match=match):
            main()

    assert pso_called == []


def test_optimizer_main_fail_fast_on_missing_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_optimizer_main_expect_deploy_error(
        monkeypatch,
        ["msmodeling-optix", "-e", "vllm", "-b", "ais_bench"],
        "找不到部署命令",
    )


def test_optimizer_main_fail_fast_on_missing_mindie(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_optimizer_main_expect_deploy_error(
        monkeypatch,
        ["msmodeling-optix", "-e", "mindie", "-b", "ais_bench"],
        "mindie",
    )


def test_ais_bench_init_fails_fast_when_benchmark_missing_in_deploy_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_root = tmp_path / "deploy"
    _write_executable_stub(deploy_root / "bin" / "vllm")

    monkeypatch.setenv("OPTIX_DEPLOY_PATH", str(deploy_root))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    monkeypatch.setattr(
        sys,
        "argv",
        ["msmodeling-optix", "-e", "vllm", "-b", "ais_bench"],
    )
    pso_called: list[bool] = []

    class _FakePSO:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run_plugin(self) -> None:
            pso_called.append(True)

    def _which(name: str, path: str | None = None) -> str | None:
        path_value = path or os.environ.get("PATH", "")
        for segment in path_value.split(os.pathsep):
            candidate = Path(segment) / name
            if candidate.is_file():
                return str(candidate.resolve())
        return None

    with (
        patch("optix.configure_logger", lambda: None),
        patch("cli.logo.print_logo", lambda: None),
        patch("optix.optimizer.optimizer.is_root", return_value=False),
        patch("optix.deploy_env.shutil.which", side_effect=_which),
        patch("optix.optimizer.optimizer.PSOOptimizer", _FakePSO),
    ):
        from optix.optimizer.optimizer import main

        with pytest.raises(OptixDeployEnvError, match="ais_bench"):
            main()

    assert pso_called == []


def test_before_run_still_sets_custom_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from optix.config.config import OptimizerConfigField

    venv_root = tmp_path / ".venv"
    parent_env = {**os.environ, "VIRTUAL_ENV": str(venv_root)}
    monkeypatch.setattr(os, "environ", parent_env)

    process = CustomProcess()
    process.command = ["bench", "$CONCURRENCY"]
    run_params = (
        OptimizerConfigField(
            name="CONCURRENCY",
            config_position="env",
            min=1,
            max=10,
            dtype="int",
            value=1,
        ),
    )
    process.before_run(run_params)

    assert CUSTOM_OUTPUT in process.env
    assert process.env[CUSTOM_OUTPUT]
    assert "VIRTUAL_ENV" not in process.env
    assert os.environ.get("VIRTUAL_ENV") == str(venv_root)


def test_warn_when_not_in_venv() -> None:
    messages: list[str] = []

    def _capture(msg: str) -> None:
        messages.append(msg)

    sink_id = logger.add(_capture, level="WARNING")
    try:
        ctx = RuntimeContext(
            in_virtualenv=False,
            virtualenv_root=None,
            python_executable=Path(sys.executable),
            msmodeling_install_editable=False,
        )
        emit_runtime_hints(ctx, engine="vllm")
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages)
    assert "[optix/env]" in combined
    assert "虚拟环境" in combined
    assert "uv venv" in combined


def test_error_message_includes_optix_deploy_path_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("optix.deploy_env.shutil.which", lambda *_a, **_k: None)
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=tmp_path / ".venv",
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    with pytest.raises(OptixDeployEnvError) as exc_info:
        validate_deploy_stack(engine="vllm", benchmark="ais_bench", env={"PATH": "/usr/bin"}, ctx=ctx)

    message = str(exc_info.value)
    assert "OPTIX_DEPLOY_PATH" in message
    assert "path_prefix" in message
