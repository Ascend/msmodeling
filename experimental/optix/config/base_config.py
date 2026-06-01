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
import time
from enum import Enum
from pathlib import Path

import optix

RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
INSTALL_PATH = Path(optix.__path__[0])
RUN_PATH = Path(os.getcwd())
MODEL_EVAL_STATE_CONFIG_PATH = "MODEL_EVAL_STATE_CONFIG_PATH"
ms_serviceparam_optimizer_config_path = os.getenv(MODEL_EVAL_STATE_CONFIG_PATH) or os.getenv(MODEL_EVAL_STATE_CONFIG_PATH.lower())
if not ms_serviceparam_optimizer_config_path:
    ms_serviceparam_optimizer_config_path = RUN_PATH.joinpath("config.toml")
ms_serviceparam_optimizer_config_path = Path(ms_serviceparam_optimizer_config_path).absolute().resolve()

CUSTOM_OUTPUT = "MODEL_EVAL_STATE_OUTPUT"
custom_output = os.getenv(CUSTOM_OUTPUT) or os.getenv(CUSTOM_OUTPUT.lower())
if custom_output:
    custom_output = Path(custom_output).resolve()
else:
    custom_output = RUN_PATH
VLLM_CUSTOM_OUTPUT = "MODEL_EVAL_STATE_VLLM_CUSTOM_OUTPUT"
MODEL_EVAL_STATE_SIMULATE = "MODEL_EVAL_STATE_SIMULATE"
MODEL_EVAL_STATE_ALL = "MODEL_EVAL_STATE_ALL"
SIMULATE = "simulate"
REAL_EVALUATION = "real_evaluation"
REQUESTRATES = ("REQUESTRATE",)
CONCURRENCYS = ("CONCURRENCY", "MAXCONCURRENCY")
simulate_env = os.getenv(MODEL_EVAL_STATE_SIMULATE) or os.getenv(MODEL_EVAL_STATE_SIMULATE.lower())
simulate_flag = simulate_env and (simulate_env.lower() == "true" or simulate_env.lower() != "false")
optimizer_env = os.getenv(MODEL_EVAL_STATE_ALL) or os.getenv(MODEL_EVAL_STATE_ALL.lower())
optimizer_flag = optimizer_env and (optimizer_env.lower() == "true" or optimizer_env.lower() != "false")


MINDIE_BENCHMARK_PERF_COLUMNS = ["average", "max", "min", "p75", "p90", "slo_p90", "p99", "n"]
FOLDER_LIMIT_SIZE = 1024 * 1024 * 1024  # 1GB


class EnginePolicy(Enum):
    mindie = "mindie"
    vllm = "vllm"


class AnalyzeTool(Enum):
    default = "default"
    profiler = "profiler"
    vllm_benchmark = "vllm"


class BenchMarkPolicy(Enum):
    benchmark = "benchmark"
    profiler_benchmark = "profiler_benchmark"
    vllm_benchmark = "vllm_benchmark"
    ais_bench = "ais_bench"


class DeployPolicy(Enum):
    single = "single"
    multiple = "multiple"


class PDPolicy(Enum):
    competition = "competition"
    disaggregation = "disaggregation"

    
class ServiceType(Enum):
    master = "master"
    slave = "slave"