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
import shlex
from pydantic import BaseModel, field_validator

from ..serve_plan import (
    BENCHMARK_ONLY_FIELDS as BENCHMARK_ONLY_FIELDS,
    CONTAINER_FLAGS as CONTAINER_FLAGS,
    FLAG_NAME_MAP as FLAG_NAME_MAP,
    JSON_SUBKEY_MAP as JSON_SUBKEY_MAP,
    MAX_REQUEST_NUM as MAX_REQUEST_NUM,
    VLLM_SERVE_CAPACITY_FIELDS as VLLM_SERVE_CAPACITY_FIELDS,
    _field_to_cli_flag as _field_to_cli_flag,
    _split_container_flags as _split_container_flags,
    param_to_cli_flag as param_to_cli_flag,
    render_serve_flags,
)


class AisBenchCommandConfig(BaseModel):
    models: str = ""
    mode: str = ""
    work_dir: str = ""
    others: str = ""


class AisBenchCommand:
    def __init__(self, aisbench_command_config: AisBenchCommandConfig):
        self.aisbench_command_config = aisbench_command_config

    @property
    def command(self):
        _cmd = [
            "ais_bench",
            "--models",
            self.aisbench_command_config.models,
            "--mode",
            self.aisbench_command_config.mode,
            "--work-dir",
            self.aisbench_command_config.work_dir,
            "--debug",
        ]
        if self.aisbench_command_config.others:
            _cmd.extend(shlex.split(self.aisbench_command_config.others))
        return _cmd


class VllmBenchmarkCommandConfig(BaseModel):
    serving: str = ""
    backend: str = "vllm"
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    dataset_name: str = ""
    dataset_path: str = ""
    num_prompts: str = ""
    result_dir: str = ""
    others: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_scalars_to_str(cls, value):
        """Coerce TOML int/float (e.g. num_prompts = 500) to str before validation."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value


class VllmBenchmarkCommand:
    def __init__(self, benchmark_command_config: VllmBenchmarkCommandConfig):
        self.benchmark_command_config = benchmark_command_config

    @property
    def command(self):
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--host",
            self.benchmark_command_config.host,
            "--port",
            self.benchmark_command_config.port,
            "--model",
            self.benchmark_command_config.model,
            "--served-model-name",
            self.benchmark_command_config.served_model_name,
            "--dataset-name",
            self.benchmark_command_config.dataset_name,
            "--max-concurrency",
            "$CONCURRENCY",
            "--request-rate",
            "$REQUESTRATE",
            "--result-dir",
            self.benchmark_command_config.result_dir,
            "--save-result",
        ]
        if self.benchmark_command_config.num_prompts:
            cmd.extend(["--num-prompts", self.benchmark_command_config.num_prompts])
        if self.benchmark_command_config.others:
            cmd.extend(shlex.split(self.benchmark_command_config.others))
        return cmd


class MindieCommandConfig(BaseModel):
    pass


class VllmCommandConfig(BaseModel):
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    others: str = ""


class VllmCommand:
    def __init__(self, command_config: VllmCommandConfig, resolved_field: list = None):
        self.command_config = command_config
        self._resolved_field = resolved_field or []

    @property
    def command(self):
        cmd = [
            "vllm",
            "serve",
            self.command_config.model,
            "--served-model-name",
            self.command_config.served_model_name,
            "--host",
            self.command_config.host,
            "--port",
            self.command_config.port,
        ]

        # Dynamic flags from resolved_field (run-type candidate fields passed via
        # the constructor) merged with `others` — see render_serve_flags for the
        # precedence/dedupe rules, shared with external simulator plugins.
        cmd.extend(render_serve_flags(self._resolved_field, self.command_config.others))
        return cmd
