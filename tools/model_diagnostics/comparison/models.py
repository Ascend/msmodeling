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
"""Requests and contracts for one pairwise stage comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from tools.model_diagnostics.domain import ComparisonSpec, Finding, StageExecutionRecord


@dataclass(frozen=True)
class StageComparisonRequest:
    region_id: str
    layer_index: int | None
    left_stage: StageExecutionRecord
    right_stage: StageExecutionRecord
    comparison: ComparisonSpec
    operator_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if self.layer_index is not None:
            if isinstance(self.layer_index, bool) or not isinstance(self.layer_index, int):
                raise TypeError("layer_index must be an integer")
            if self.layer_index < 0:
                raise ValueError("layer_index must be non-negative")
        if self.left_stage.stage_id != self.right_stage.stage_id:
            raise ValueError("paired stages must have the same stage_id")
        object.__setattr__(self, "operator_aliases", MappingProxyType(dict(self.operator_aliases)))


class StageComparisonStrategy(Protocol):
    strategy_id: str

    def execute(self, request: StageComparisonRequest) -> tuple[Finding, ...]: ...
