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
"""Only the OptiX-facing configuration for the AISBench adapter."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import SettingsConfigDict

from optix.config.config import OptimizerConfigField, PerformanceConfig, Settings, range_to_enum


class AisBenchAutoCommandConfig(BaseModel):
    executable: str = "python3"
    script: str = Field(min_length=1)
    others: str = ""


class AisBenchAutoConfig(BaseModel):
    process_name: str = ""
    output_path: Path = Path("outputs/default")
    work_path: Path = Field(default_factory=lambda: Path.cwd().resolve())
    command: AisBenchAutoCommandConfig
    performance_config: PerformanceConfig = Field(default_factory=PerformanceConfig)
    target_field: list[OptimizerConfigField] = Field(default_factory=list)
    best_concurrency_coefficient: int = 3
    best_concurrency_threshold: int = 200


class AisBenchAutoSettings(Settings):
    model_config = SettingsConfigDict(extra="ignore")
    aisbench_auto: AisBenchAutoConfig | None = None

    @model_validator(mode="after")
    def normalize_benchmark(self):
        # Pylint cannot infer fields inherited through Pydantic models.
        # pylint: disable=no-member
        config = self.aisbench_auto
        if config is not None and config.target_field:
            range_to_enum(config.target_field)
        return self
