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

"""Narrow source, specification and organization ports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from tools.model_diagnostics.domain import (
    ExecutionOrganizationRequest,
    ModelDiagnosticsSpec,
    ModelExecutionRecord,
    ModelRunContext,
    RegionExecutionRecord,
    SourceDescription,
    SourceKind,
)


class ModelDiagnosticsSpecProvider(Protocol):
    def get(self, context: ModelRunContext) -> ModelDiagnosticsSpec: ...


class OperatorRecordSource(Protocol):
    source_kind: SourceKind

    def describe(self) -> SourceDescription: ...

    def load_execution(
        self,
        context: ModelRunContext,
        spec: ModelDiagnosticsSpec,
        selected_layers: Mapping[str, tuple[int, ...]],
        selected_stage_regions: tuple[str, ...],
    ) -> ModelExecutionRecord: ...


class ExecutionOrganizationStrategy(Protocol):
    strategy_id: str

    def execute(
        self,
        request: ExecutionOrganizationRequest,
    ) -> tuple[RegionExecutionRecord, ...]: ...
