# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
from optix.config.config import Settings, OptimizerConfigField


class MultiHostCommandConfig(BaseModel):
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    others: str = ""


class MultiHostConfig(BaseModel):
    output: Path = Path("vllm")
    process_name: str = "vllm"
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: MultiHostCommandConfig = MultiHostCommandConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


class CusSettings(Settings):
    name: str = "multihost_inference_optimization"
    multihost: MultiHostConfig = Field(
        default_factory=lambda data: MultiHostConfig(output=data["output"].joinpath("vllm")), validate_default=True
    )
