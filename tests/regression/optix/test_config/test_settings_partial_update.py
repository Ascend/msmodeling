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

from optix.config.config import AisBenchConfig, Settings, VllmBenchmarkConfig, VllmConfig
from optix.config.custom_command import VllmCommandConfig


def test_partial_update_vllm_syncs_host_without_control_vllm_package(tmp_path: Path) -> None:
    """partial_update_vllm must not depend on importlib.find_spec('vllm') in control env."""
    output = tmp_path / "output"
    settings = Settings(
        output=output,
        vllm=VllmConfig(
            command=VllmCommandConfig(
                host="127.0.0.1",
                port="8000",
                model="test-model",
                served_model_name="test-served",
            ),
        ),
        vllm_benchmark=VllmBenchmarkConfig(),
    )
    assert settings.vllm_benchmark.command.host == "127.0.0.1"
    assert settings.vllm_benchmark.command.port == "8000"
    assert settings.vllm_benchmark.command.model == "test-model"
    assert settings.vllm_benchmark.command.served_model_name == "test-served"
    assert settings.vllm.output == output.joinpath("vllm")
    assert settings.vllm_benchmark.output_path == output.joinpath("vllm")


def test_partial_update_aisbench_sets_work_dir_without_control_ais_bench(tmp_path: Path) -> None:
    """partial_update_aisbench must not depend on importlib.find_spec('ais_bench') in control env."""
    output = tmp_path / "output"
    settings = Settings(output=output, ais_bench=AisBenchConfig())
    expected_work_dir = str(settings.ais_bench.output_path)
    assert settings.ais_bench.command.work_dir == expected_work_dir
    assert settings.ais_bench.output_path == output.joinpath("ais_bench")
    assert settings.ais_bench.command.work_dir
    assert output.exists()


def test_partial_update_creates_output_paths(tmp_path: Path) -> None:
    output = tmp_path / "result"
    settings = Settings(output=output)
    assert output.exists()
    assert settings.output == output.resolve()
    assert settings.vllm_benchmark.command.result_dir
    assert Path(settings.vllm_benchmark.command.result_dir).exists()
    assert settings.ais_bench.command.work_dir
