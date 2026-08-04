# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Immutable requests and records for source execution organization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tools.model_diagnostics.domain.models import ModelExecutionRecord, OperatorCallRecord
from tools.model_diagnostics.domain.selection import (
    normalize_selected_layers,
    normalize_selected_stage_regions,
)
from tools.model_diagnostics.domain.specification import ModelDiagnosticsSpec


@dataclass(frozen=True)
class StageExecutionRecord:
    stage_id: str
    operator_calls: tuple[OperatorCallRecord, ...]


@dataclass(frozen=True)
class LayerExecutionRecord:
    layer_index: int
    layer_kind: str
    stages: tuple[StageExecutionRecord, ...]


@dataclass(frozen=True)
class RegionExecutionRecord:
    region_id: str
    stages: tuple[StageExecutionRecord, ...] = ()
    layers: tuple[LayerExecutionRecord, ...] = ()


@dataclass(frozen=True)
class ExecutionOrganizationRequest:
    execution: ModelExecutionRecord
    spec: ModelDiagnosticsSpec
    selected_layers: Mapping[str, tuple[int, ...]]
    selected_stage_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_layers", normalize_selected_layers(self.selected_layers))
        object.__setattr__(
            self,
            "selected_stage_regions",
            normalize_selected_stage_regions(self.selected_stage_regions),
        )
