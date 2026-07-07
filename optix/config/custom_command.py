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
import shutil
import shlex
from pydantic import BaseModel


MAX_REQUEST_NUM = 1e6


class AisBenchCommandConfig(BaseModel):
    models: str = ""
    mode: str = ""
    work_dir: str = ""
    others: str = ""


class AisBenchCommand:
    def __init__(self, aisbench_command_config: AisBenchCommandConfig):
        self.process = shutil.which("ais_bench")
        if self.process is None:
            raise ValueError("Error: The 'ais_bench' executable was not found in the system PATH.")
        self.aisbench_command_config = aisbench_command_config

    @property
    def command(self):
        _cmd = [
            self.process,
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
    result_dir: str = ""
    others: str = ""


class VllmBenchmarkCommand:
    def __init__(self, benchmark_command_config: VllmBenchmarkCommandConfig):
        self.process = shutil.which("vllm")
        if self.process is None:
            raise ValueError("Error: The 'vllm' executable was not found in the system PATH.")
        self.benchmark_command_config = benchmark_command_config

    @property
    def command(self):
        cmd = [
            self.process,
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
        if self.benchmark_command_config.others:
            cmd.extend(shlex.split(self.benchmark_command_config.others))
        return cmd


class MindieCommandConfig(BaseModel):
    pass


class MindieCommand:
    def __init__(self, command_config: MindieCommandConfig):
        self.command_config = command_config

    @property
    def command(self):
        mindie_service_default_path: str = "/usr/local/Ascend/mindie/latest/mindie-service"
        mindie_service_path: str = os.getenv("MIES_INSTALL_PATH", mindie_service_default_path)
        mindie_command_path: str = os.path.join(mindie_service_path, "bin", "mindieservice_daemon")
        new_mindie_command: str = "mindie_llm_server"
        if not os.path.isfile(mindie_command_path):
            if shutil.which(new_mindie_command) is None:
                raise FileNotFoundError(f"Command {new_mindie_command} is not available")
            return [new_mindie_command]
        return [mindie_command_path]


class VllmCommandConfig(BaseModel):
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    others: str = ""


class VllmCommand:
    def __init__(self, command_config: VllmCommandConfig):
        self.process = shutil.which("vllm")
        if self.process is None:
            raise ValueError("Error: The 'vllm' executable was not found in the system PATH.")
        self.command_config = command_config

    @property
    def command(self):
        cmd = [
            self.process,
            "serve",
            self.command_config.model,
            "--served-model-name",
            self.command_config.served_model_name,
            "--host",
            self.command_config.host,
            "--port",
            self.command_config.port,
            "--max-num-batched-tokens",
            "$MAX_NUM_BATCHED_TOKENS",
            "--max-num-seqs",
            "$MAX_NUM_SEQS",
        ]
        if self.command_config.others:
            cmd.extend(shlex.split(self.command_config.others))
        return cmd
