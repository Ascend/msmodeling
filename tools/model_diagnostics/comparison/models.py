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
