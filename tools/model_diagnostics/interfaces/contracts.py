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
