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

import shutil
from typing import Literal

from loguru import logger

from .errors import BenchmarkUnavailableError, SimulatorUnavailableError

simulates = {}
benchmarks = {}

DEFAULT_BENCHMARK_POLICY = "ais_bench"
DEFAULT_SIMULATOR_POLICY = "vllm"

_BENCHMARK_GUIDANCE = {
    "ais_bench": "Install AIS Bench or run with -b vllm_benchmark.",
    "vllm_benchmark": "Install the vLLM CLI or choose another -b benchmark policy.",
}

_SIMULATOR_GUIDANCE = {
    "vllm": "Install the vLLM CLI or choose another -e engine policy.",
}


def _validate_required_executable(
    role: Literal["b", "e"],
    policy: str,
    plugin_cls: type,
    *,
    default_policy: str,
    guidance_map: dict[str, str],
) -> None:
    executable = getattr(plugin_cls, "required_executable", None)
    if not executable:
        return
    if shutil.which(executable) is not None:
        return
    guidance = guidance_map.get(policy, f"Install '{executable}' or choose another -{role} policy.")
    if role == "b":
        raise BenchmarkUnavailableError(
            policy,
            executable,
            guidance,
            default_policy=default_policy,
        )
    raise SimulatorUnavailableError(
        policy,
        executable,
        guidance,
        default_policy=default_policy,
    )


def validate_benchmark_policy(policy: str) -> None:
    """Fail fast when the benchmark executable for ``policy`` is not on PATH."""
    plugin_cls = benchmarks.get(policy)
    if plugin_cls is None:
        return
    _validate_required_executable(
        "b",
        policy,
        plugin_cls,
        default_policy=DEFAULT_BENCHMARK_POLICY,
        guidance_map=_BENCHMARK_GUIDANCE,
    )


def validate_simulator_policy(policy: str) -> None:
    """Fail fast when the simulator executable for ``policy`` is not on PATH."""
    plugin_cls = simulates.get(policy)
    if plugin_cls is None:
        return
    _validate_required_executable(
        "e",
        policy,
        plugin_cls,
        default_policy=DEFAULT_SIMULATOR_POLICY,
        guidance_map=_SIMULATOR_GUIDANCE,
    )


def register_simulator(model_arch: str, model_cls) -> None:
    """
    Register an external model to be used in ..

    :code:`model_cls` can be either:

    - A :class:`SimulatorInterface` class directly referencing the model.
    """
    from ..optimizer.interfaces.simulator import SimulatorInterface

    if not isinstance(model_arch, str):
        msg = f"`model_arch` should be a string, not a {type(model_arch)}"
        raise TypeError(msg)

    if model_arch in simulates:
        logger.warning(
            "Model architecture {} is already registered, and will be overwritten by the new model class {}.",
            model_arch,
            model_cls,
        )
    if isinstance(model_cls, type) and issubclass(model_cls, SimulatorInterface):
        simulates[model_arch] = model_cls
    else:
        msg = f"`model_cls` should be a SimulatorInterface class, not a {type(model_cls)}"
        raise TypeError(msg)


def register_benchmarks(model_arch: str, model_cls) -> None:
    """
    Register an external model to be used in ..

    :code:`model_cls` can be either:

    - A :class:`BenchmarkInterface` class directly referencing the model.
    """
    from ..optimizer.interfaces.benchmark import BenchmarkInterface

    if not isinstance(model_arch, str):
        msg = f"`model_arch` should be a string, not a {type(model_arch)}"
        raise TypeError(msg)

    if model_arch in benchmarks:
        logger.warning(
            "Model architecture {} is already registered, and will be overwritten by the new model class {}.",
            model_arch,
            model_cls,
        )
    if isinstance(model_cls, type) and issubclass(model_cls, BenchmarkInterface):
        benchmarks[model_arch] = model_cls
    else:
        msg = f"`model_cls` should be a BenchmarkInterface class, not a {type(model_cls)}"
        raise TypeError(msg)


def register_ori_functions():
    from ..optimizer.plugins.benchmark import AisBench, VllmBenchMark
    from ..optimizer.plugins.simulate import Simulator, VllmSimulator

    register_benchmarks("vllm_benchmark", VllmBenchMark)
    register_benchmarks("ais_bench", AisBench)
    register_simulator("vllm", VllmSimulator)
    register_simulator("mindie", Simulator)
