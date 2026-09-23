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
