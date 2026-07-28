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
    virtualenv_root = _resolve_isolation_root(os.environ)

    return RuntimeContext(
        in_virtualenv=virtualenv_root is not None,
        virtualenv_root=virtualenv_root,
        python_executable=Path(sys.executable).resolve(),
        msmodeling_install_editable=False,
    )


def emit_runtime_hints(ctx: RuntimeContext, *, engine: str) -> None:
    if not ctx.in_virtualenv:
        logger.warning(
            f"{_OPTIX_ENV_LOG_PREFIX} msmodeling is not running in a virtual environment.\n"
            "Installing msmodeling also installs packages such as torch and transformers. "
            "These packages are intended for TensorCast simulation, not OptiX optimization on real hardware.\n"
            "Installing them into the system Python may replace versions required by vLLM or MindIE, "
            "which can prevent services from starting or cause inference errors.\n"
            "Use a virtual environment instead:\n"
            "  1. uv sync\n"
            "  2. source .venv/bin/activate\n"
            f"  3. msmodeling optix -e {engine} ..."
        )


def resolve_deploy_path_prefix() -> str | None:
    env_path = os.environ.get(_OPTIX_DEPLOY_PATH_ENV)
    if env_path:
        resolved = Path(env_path).expanduser().resolve()
        if not resolved.is_dir():
            raise OptixDeployEnvError(f"{_OPTIX_ENV_LOG_PREFIX} OPTIX_DEPLOY_PATH is not a valid directory: {resolved}")
        return str(resolved)

    config_prefix = get_settings().deploy.path_prefix
    if config_prefix:
        resolved = Path(config_prefix).expanduser().resolve()
        if not resolved.is_dir():
            raise OptixDeployEnvError(
                f"{_OPTIX_ENV_LOG_PREFIX} config.toml [deploy] path_prefix is not a valid directory: {resolved}"
            )
        return str(resolved)
    return None


def _resolve_isolation_root(parent: Mapping[str, str]) -> Path | None:
    virtual_env = parent.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env).resolve()

    if getattr(sys, "real_prefix", None) is not None:
        return Path(sys.prefix).resolve()

    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix).resolve()

    conda_prefix = parent.get("CONDA_PREFIX")
    conda_env = parent.get("CONDA_DEFAULT_ENV")
    if conda_prefix and conda_env and conda_env != "base":
        return Path(conda_prefix).resolve()
    return None


def _is_conda_isolation_root(parent: Mapping[str, str], isolation_root: Path | None) -> bool:
    conda_prefix = parent.get("CONDA_PREFIX")
    conda_env = parent.get("CONDA_DEFAULT_ENV")
    if isolation_root is None or not conda_prefix or not conda_env or conda_env == "base":
        return False
    return isolation_root == Path(conda_prefix).resolve()


def _is_under_venv(segment: str, venv_root: Path) -> bool:
    try:
        resolved = Path(segment).resolve()
    except OSError:
        logger.warning(
            f"{_OPTIX_ENV_LOG_PREFIX} Failed to resolve path segment {segment!r}; "
            "treating it as outside the virtual environment"
        )
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
    if _is_conda_isolation_root(parent, venv_root):
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
        raise OptixDeployEnvError(f"{_OPTIX_ENV_LOG_PREFIX} Deployment command not found: {name}.")
    executable = Path(resolved).resolve()
    if msmodeling_venv is not None and _is_under_venv(str(executable), msmodeling_venv.resolve()):
        raise OptixDeployEnvError(
            f"{_OPTIX_ENV_LOG_PREFIX} Command {name} resolves inside the msmodeling virtual environment: {executable}"
        )
    return executable


def resolve_path_executable(name: str, env: Mapping[str, str], ctx: RuntimeContext) -> Path:
    return resolve_deploy_executable(name, env, msmodeling_venv=ctx.virtualenv_root)


def resolve_mindie_argv(env: Mapping[str, str]) -> list[str]:
    mindie_service_path = env.get("MIES_INSTALL_PATH", _MINDIE_SERVICE_DEFAULT_PATH)
    mindie_command_path = os.path.join(mindie_service_path, "bin", _MINDIE_DAEMON_NAME)
    if not os.path.isfile(mindie_command_path):
        path_value = env.get("PATH", "")
        resolved = shutil.which(_MINDIE_LLM_SERVER_NAME, path=path_value or None)
        if resolved is None:
            raise FileNotFoundError(f"Command {_MINDIE_LLM_SERVER_NAME} is not available")
        return [str(Path(resolved).resolve())]
    return [mindie_command_path]


def materialize_command(
    argv: list[str],
    env: Mapping[str, str],
    ctx: RuntimeContext,
    *,
    cwd: str | Path | None = None,
) -> list[str]:
    if not argv:
        return argv

    first = argv[0]
    if first in _MATERIALIZE_COMMAND_NAMES:
        executable = resolve_deploy_executable(first, env, msmodeling_venv=ctx.virtualenv_root)
        return [str(executable), *argv[1:]]

    command_path = Path(first)
    has_path_separator = os.sep in first or (os.altsep is not None and os.altsep in first)
    if not command_path.is_absolute() and not has_path_separator:
        return list(argv)

    if not command_path.is_absolute():
        command_path = Path(cwd or os.getcwd()) / command_path
    try:
        resolved_command = command_path.resolve()
    except OSError:
        return list(argv)
    if not resolved_command.is_file():
        return list(argv)

    if ctx.virtualenv_root is not None and _is_under_venv(str(resolved_command), ctx.virtualenv_root.resolve()):
        raise OptixDeployEnvError(
            f"{_OPTIX_ENV_LOG_PREFIX} Command is inside the msmodeling virtual environment: {resolved_command}\n"
            "Do not install deployment commands in the msmodeling environment because they may conflict "
            "with simulation dependencies."
        )

    return [str(resolved_command), *argv[1:]]


def _skip_unless_in_registry(name: str, registry: Mapping[str, str]) -> bool:
    if name not in registry:
        logger.info(f"{_OPTIX_ENV_LOG_PREFIX} Skipping built-in deployment validation: {name} is not registered")
        return True
    return False


def _raise_missing_executable(executable_name: str, *, engine: str | None = None) -> NoReturn:
    context = f", current engine: {engine}" if engine else ""
    raise OptixDeployEnvError(
        f"{_OPTIX_ENV_LOG_PREFIX} Deployment command not found: {executable_name}{context}.\n"
        "The subprocess PATH excludes the msmodeling virtual environment, and no usable "
        f"{executable_name} was found on the system PATH.\n"
        f"  1. Verify that it is installed on the system: which {executable_name}\n"
        "  2. If the command is not on the system PATH, export "
        "OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root\n"
        "     or set path_prefix in the [deploy] section of config.toml\n"
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
            f"{_OPTIX_ENV_LOG_PREFIX} Command {executable_name} is inside the msmodeling virtual environment: "
            f"{executable}\n"
            f"Do not install {executable_name} in the msmodeling environment because it may conflict "
            "with simulation dependencies.\n"
            f"  1. Run this command in that virtual environment: pip uninstall {executable_name}\n"
            "  2. Use the system deployment of vLLM or MindIE; set OPTIX_DEPLOY_PATH only when "
            "the system PATH needs an override"
        )
    logger.info(
        f"{_OPTIX_ENV_LOG_PREFIX} msmodeling runtime: {ctx.virtualenv_root or 'system Python'}; "
        f"deployment command {executable_name}: {executable}"
    )


def _validate_engine(engine: str, env: dict[str, str], ctx: RuntimeContext) -> None:
    if _skip_unless_in_registry(engine, _ENGINE_EXECUTABLES):
        return

    path_value = env.get("PATH", "")
    if not path_value.strip():
        raise OptixDeployEnvError(
            f"{_OPTIX_ENV_LOG_PREFIX} Deployment command not found: {_ENGINE_EXECUTABLES[engine]}; PATH is empty."
        )

    if engine == "mindie":
        try:
            argv = resolve_mindie_argv(env)
        except FileNotFoundError as exc:
            raise OptixDeployEnvError(
                f"{_OPTIX_ENV_LOG_PREFIX} Deployment command not found: mindie, current engine: mindie.\n  {exc}"
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
