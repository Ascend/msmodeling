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
from enum import Enum
from pydantic import BaseModel

MODEL_EVAL_STATE_SIMULATE = "MODEL_EVAL_STATE_SIMULATE"
MODEL_EVAL_STATE_ALL = "MODEL_EVAL_STATE_ALL"
SIMULATE = "simulate"

simulate_env = os.getenv(MODEL_EVAL_STATE_SIMULATE) or os.getenv(MODEL_EVAL_STATE_SIMULATE.lower())
simulate_flag = simulate_env and (simulate_env.lower() == "true" or simulate_env.lower() != "false")

REAL_EVALUATION = "real_evaluation"

REQUESTRATES = ("REQUESTRATE",)
CONCURRENCYS = ("CONCURRENCY", "MAXCONCURRENCY")
METRIC_TTFT = 'ttft'
METRIC_TPOT = 'tpot'

class Stage(Enum):
    start = "start"
    running = "running"
    error = "error"
    stop = "stop"

class ProcessState(BaseModel):
    stage: Stage = Stage.start
    info: str = ""