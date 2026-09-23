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

from optix.config.config import Settings
from pydantic import Field
from pathlib import Path
from pydantic_settings import SettingsConfigDict

# TODO: import your benchmark's config from benchmark.py
from evalscopeperf.benchmark import EvalscopePerfConfig


# TODO: set your benchmark's basic config
class CusSettings(Settings):
    model_config = SettingsConfigDict(extra="ignore")
    name: str = "evalscopeperf"
    evalscopeperf: EvalscopePerfConfig = Field(
        default_factory=lambda data: EvalscopePerfConfig(output_path=Path.cwd().joinpath("evalscopeperf")),
        validate_default=True,
    )
