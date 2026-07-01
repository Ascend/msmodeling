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
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from loguru import logger

from optix.config.config import get_settings

_OPTIX_ENV_LOG_PREFIX = "[optix/env]"
_OPTIX_DEPLOY_PATH_ENV = "OPTIX_DEPLOY_PATH"

_ENGINE_EXECUTABLES: dict[str, str] = {
    "vllm": "vllm",
    "mindie": "mindieservice_daemon",
}

_BENCHMARK_EXECUTABLES: dict[str, str] = {
    "ais_bench": "ais_bench",
    "vllm_benchmark": "vllm",
}

_MATERIALIZE_COMMAND_NAMES = frozenset({"vllm", "ais_bench"})

_MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"
_MINDIE_DAEMON_NAME = "mindieservice_daemon"
_MINDIE_LLM_SERVER_NAME = "mindie_llm_server"


class OptixDeployEnvError(RuntimeError):
    """Raised when deploy stack validation or path resolution fails."""


@dataclass(frozen=True)
class RuntimeContext:
    in_virtualenv: bool
    virtualenv_root: Path | None
    python_executable: Path
    msmodeling_install_editable: bool = False


def detect_runtime_context() -> RuntimeContext:
    virtualenv_root: Path | None = None
    in_virtualenv = False

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        in_virtualenv = True
        virtualenv_root = Path(virtual_env).resolve()

    if not in_virtualenv and getattr(sys, "real_prefix", None) is not None:
        in_virtualenv = True
        virtualenv_root = Path(sys.prefix).resolve()

    if not in_virtualenv and sys.prefix != sys.base_prefix:
        in_virtualenv = True
        virtualenv_root = Path(sys.prefix).resolve()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not in_virtualenv and conda_prefix and conda_env and conda_env != "base":
        in_virtualenv = True
        virtualenv_root = Path(conda_prefix).resolve()

    return RuntimeContext(
        in_virtualenv=in_virtualenv,
        virtualenv_root=virtualenv_root,
        python_executable=Path(sys.executable).resolve(),
        msmodeling_install_editable=False,
    )


def emit_runtime_hints(ctx: RuntimeContext, *, engine: str) -> None:
    if not ctx.in_virtualenv:
        logger.warning(
            f"{_OPTIX_ENV_LOG_PREFIX} 当前未在虚拟环境中运行。\n"
            "安装 msmodeling 会带上 torch、transformers 等包，它们给 TensorCast 仿真用，不是 OptiX 真机寻优用的。\n"
            "装到系统 Python 里会改掉 vLLM、MindIE 依赖的版本，服务可能起不来或推理出错。\n"
            "请改用虚拟环境：\n"
            "  1. uv venv .venv && source .venv/bin/activate\n"
            "  2. uv sync && pip install -e .\n"
            f"  3. msmodeling optix -e {engine} ..."
        )


def resolve_deploy_path_prefix() -> str | None:
    env_path = os.environ.get(_OPTIX_DEPLOY_PATH_ENV)
    if env_path:
        resolved = Path(env_path).expanduser().resolve()
        if not resolved.is_dir():
            raise OptixDeployEnvError(f"{_OPTIX_ENV_LOG_PREFIX} OPTIX_DEPLOY_PATH 不是有效目录：{resolved}")
        return str(resolved)

    config_prefix = get_settings().deploy.path_prefix
    if config_prefix:
        resolved = Path(config_prefix).expanduser().resolve()
        if not resolved.is_dir():
            raise OptixDeployEnvError(
                f"{_OPTIX_ENV_LOG_PREFIX} config.toml [deploy] path_prefix 不是有效目录：{resolved}"
            )
        return str(resolved)
    return None


def _resolve_isolation_root(parent: Mapping[str, str]) -> Path | None:
    virtual_env = parent.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env).resolve()

    conda_prefix = parent.get("CONDA_PREFIX")
    conda_env = parent.get("CONDA_DEFAULT_ENV")
    if conda_prefix and conda_env and conda_env != "base":
        return Path(conda_prefix).resolve()
    return None


def _is_under_venv(segment: str, venv_root: Path) -> bool:
    try:
        resolved = Path(segment).resolve()
    except OSError:
        logger.warning(f"{_OPTIX_ENV_LOG_PREFIX} 无法解析路径段 {segment!r}，保守视为非 venv 路径")
        return False
    try:
        resolved.relative_to(venv_root)
        return True
    except ValueError:
        return False


def _filter_path_segments(value: str, venv_root: Path | None, *, separator: str | None = None) -> str:
    sep = separator if separator is not None else os.pathsep
    segments = value.split(sep)
    if venv_root is None:
        return value
    kept = [segment for segment in segments if segment and not _is_under_venv(segment, venv_root)]
    return sep.join(kept)


def _filter_pythonpath(value: str, venv_root: Path | None) -> str:
    return _filter_path_segments(value, venv_root, separator=os.pathsep)


def build_deploy_env(
    parent: Mapping[str, str],
    *,
    deploy_path_prefix: str | None,
    isolation_root: Path | None = None,
) -> dict[str, str]:
    env = dict(parent)
    venv_root = isolation_root if isolation_root is not None else _resolve_isolation_root(parent)

    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    env.pop(_OPTIX_DEPLOY_PATH_ENV, None)

    path_sep = os.pathsep
    if "PATH" in env:
        env["PATH"] = _filter_path_segments(env["PATH"], venv_root, separator=path_sep)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = _filter_pythonpath(env["PYTHONPATH"], venv_root)
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = _filter_path_segments(env["LD_LIBRARY_PATH"], venv_root)

    if deploy_path_prefix:
        deploy_bin = str(Path(deploy_path_prefix).resolve() / "bin")
        current_path = env.get("PATH", "")
        env["PATH"] = deploy_bin if not current_path else f"{deploy_bin}{os.pathsep}{current_path}"

    return env


def resolve_deploy_context() -> tuple[RuntimeContext, dict[str, str]]:
    ctx = detect_runtime_context()
    deploy_env = build_deploy_env(
        os.environ,
        deploy_path_prefix=resolve_deploy_path_prefix(),
        isolation_root=ctx.virtualenv_root,
    )
    return ctx, deploy_env


def resolve_deploy_executable(
    name: str,
    env: Mapping[str, str],
    *,
    msmodeling_venv: Path | None,
) -> Path:
    path_value = env.get("PATH", "")
    resolved = shutil.which(name, path=path_value or None)
    if resolved is None:
        raise OptixDeployEnvError(f"{_OPTIX_ENV_LOG_PREFIX} 找不到部署命令：{name}。")
    executable = Path(resolved).resolve()
    if msmodeling_venv is not None and _is_under_venv(str(executable), msmodeling_venv.resolve()):
        raise OptixDeployEnvError(f"{_OPTIX_ENV_LOG_PREFIX} 命令 {name} 解析到 msmodeling 虚拟环境：{executable}")
    return executable


def resolve_path_executable(name: str, env: Mapping[str, str], ctx: RuntimeContext) -> Path:
    return resolve_deploy_executable(name, env, msmodeling_venv=ctx.virtualenv_root)


def resolve_mindie_argv(env: Mapping[str, str]) -> list[str]:
    mindie_service_path = env.get("MIES_INSTALL_PATH", _MINDIE_SERVICE_DEFAULT_PATH)
    mindie_command_path = os.path.join(mindie_service_path, "bin", _MINDIE_DAEMON_NAME)
    if not os.path.isfile(mindie_command_path):
        path_value = env.get("PATH", "")
        if shutil.which(_MINDIE_LLM_SERVER_NAME, path=path_value or None) is None:
            raise FileNotFoundError(f"Command {_MINDIE_LLM_SERVER_NAME} is not available")
        return [_MINDIE_LLM_SERVER_NAME]
    return [mindie_command_path]


def materialize_command(argv: list[str], env: Mapping[str, str], ctx: RuntimeContext) -> list[str]:
    if not argv:
        return argv

    first = argv[0]
    if first in _MATERIALIZE_COMMAND_NAMES:
        executable = resolve_deploy_executable(first, env, msmodeling_venv=ctx.virtualenv_root)
        return [str(executable), *argv[1:]]

    if (
        os.path.isabs(first)
        and os.path.isfile(first)
        and ctx.virtualenv_root is not None
        and _is_under_venv(first, ctx.virtualenv_root)
    ):
        raise OptixDeployEnvError(
            f"{_OPTIX_ENV_LOG_PREFIX} 命令落在 msmodeling 虚拟环境里：{first}\n"
            "不要在 msmodeling 环境里安装部署命令，会和仿真依赖搅在一起。"
        )

    return list(argv)


def _skip_unless_in_registry(name: str, registry: Mapping[str, str]) -> bool:
    if name not in registry:
        logger.info(f"{_OPTIX_ENV_LOG_PREFIX} 跳过内置 deploy 校验：{name} 不在 registry 中")
        return True
    return False


def _raise_missing_executable(executable_name: str, *, engine: str | None = None) -> NoReturn:
    context = f"，当前引擎为 {engine}" if engine else ""
    raise OptixDeployEnvError(
        f"{_OPTIX_ENV_LOG_PREFIX} 找不到部署命令：{executable_name}{context}。\n"
        "子进程已去掉 msmodeling 虚拟环境的 PATH，系统 PATH 里还没有可用的 "
        f"{executable_name}。\n"
        f"  1. 先确认系统已安装：which {executable_name}\n"
        "  2. 若命令不在系统 PATH，可 export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root\n"
        "     或在 config.toml 的 [deploy] 里设置 path_prefix\n"
        f"  3. msmodeling optix -e {engine or 'vllm'} ..."
    )


def _validate_resolved_executable(
    executable_name: str,
    resolved: str,
    *,
    ctx: RuntimeContext,
    engine: str | None = None,
) -> None:
    executable = Path(resolved).resolve()
    if ctx.virtualenv_root is not None and _is_under_venv(str(executable), ctx.virtualenv_root):
        raise OptixDeployEnvError(
            f"{_OPTIX_ENV_LOG_PREFIX} 命令 {executable_name} 落在 msmodeling 虚拟环境里：{executable}\n"
            f"不要在 msmodeling 环境里安装 {executable_name}，会和仿真依赖搅在一起。\n"
            f"  1. 在该 venv 里执行：pip uninstall {executable_name}\n"
            "  2. 改用系统里已部署的 vLLM 或 MindIE；PATH 特殊时再设 OPTIX_DEPLOY_PATH"
        )
    logger.info(
        f"{_OPTIX_ENV_LOG_PREFIX} msmodeling 运行于虚拟环境 {ctx.virtualenv_root or '系统 Python'}；"
        f"部署命令 {executable_name} → {executable}"
    )


def _validate_engine(engine: str, env: dict[str, str], ctx: RuntimeContext) -> None:
    if _skip_unless_in_registry(engine, _ENGINE_EXECUTABLES):
        return

    path_value = env.get("PATH", "")
    if not path_value.strip():
        raise OptixDeployEnvError(f"{_OPTIX_ENV_LOG_PREFIX} 找不到部署命令：{_ENGINE_EXECUTABLES[engine]}；PATH 为空。")

    if engine == "mindie":
        try:
            argv = resolve_mindie_argv(env)
        except FileNotFoundError as exc:
            raise OptixDeployEnvError(
                f"{_OPTIX_ENV_LOG_PREFIX} 找不到部署命令：mindie，当前引擎为 mindie。\n  {exc}"
            ) from exc
        first = argv[0]
        resolved: str
        if os.path.isabs(first):
            if not os.path.isfile(first):
                _raise_missing_executable(_MINDIE_DAEMON_NAME, engine=engine)
            resolved = first
        else:
            found = shutil.which(first, path=path_value or None)
            if found is None:
                _raise_missing_executable(first, engine=engine)
            resolved = found
        _validate_resolved_executable(first, resolved, ctx=ctx, engine=engine)
        return

    executable_name = _ENGINE_EXECUTABLES[engine]
    found = shutil.which(executable_name, path=path_value or None)
    if found is None:
        _raise_missing_executable(executable_name, engine=engine)
    _validate_resolved_executable(executable_name, found, ctx=ctx, engine=engine)


def _validate_benchmark(benchmark: str, env: dict[str, str], ctx: RuntimeContext) -> None:
    if _skip_unless_in_registry(benchmark, _BENCHMARK_EXECUTABLES):
        return

    executable_name = _BENCHMARK_EXECUTABLES[benchmark]
    path_value = env.get("PATH", "")
    found = shutil.which(executable_name, path=path_value or None)
    if found is None:
        _raise_missing_executable(executable_name)
    _validate_resolved_executable(executable_name, found, ctx=ctx)


def validate_deploy_stack(
    *,
    engine: str,
    benchmark: str,
    env: dict[str, str],
    ctx: RuntimeContext,
) -> None:
    _validate_engine(engine, env, ctx)
    _validate_benchmark(benchmark, env, ctx)
