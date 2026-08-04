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
"""Immutable values shared by diagnostics sources and strategies."""

from tools.model_diagnostics.domain.artifact import ProducerInfo, SimulationExecutionArtifact
from tools.model_diagnostics.domain.models import (
    INPUT,
    OUTPUT,
    DiagnosticsRequest,
    ExecutionPhase,
    ModelExecutionRecord,
    ModelRunContext,
    OperatorCallRecord,
    ParallelContext,
    SourceKind,
    TensorDirection,
    TensorInfo,
    TensorSlot,
)
from tools.model_diagnostics.domain.organization import (
    ExecutionOrganizationRequest,
    LayerExecutionRecord,
    RegionExecutionRecord,
    StageExecutionRecord,
)
from tools.model_diagnostics.domain.results import (
    DiagnosticValue,
    DiagnosticsResult,
    DiagnosticsSummary,
    EvidenceRef,
    Finding,
    FindingStatus,
    Limitation,
    ModelRunContextSummary,
    SourceDescription,
    summarize_findings,
)
from tools.model_diagnostics.domain.selection import (
    normalize_selected_layers,
    normalize_selected_stage_regions,
)
from tools.model_diagnostics.domain.specification import (
    BoundaryEqualOptions,
    ComparisonSpec,
    ComparisonOptions,
    ConcatOptions,
    DTypeExpr,
    LayerSpec,
    ModelDiagnosticsSpec,
    OneToOneOptions,
    RegionSpec,
    RuntimeStageOptions,
    ShapeExpr,
    SpecMatchCriteria,
    StageSpec,
    TensorMapping,
    TensorMappingMode,
    TensorRelation,
    TensorSlotPair,
    TensorSlotRef,
    TheoryOperatorSpec,
    TheoryStageOptions,
    TheoryTensorSpec,
)

__all__ = [
    "INPUT",
    "OUTPUT",
    "DiagnosticsRequest",
    "ExecutionPhase",
    "ModelExecutionRecord",
    "ModelRunContext",
    "OperatorCallRecord",
    "ParallelContext",
    "SourceKind",
    "TensorDirection",
    "TensorInfo",
    "TensorSlot",
    "ExecutionOrganizationRequest",
    "LayerExecutionRecord",
    "RegionExecutionRecord",
    "StageExecutionRecord",
    "BoundaryEqualOptions",
    "ComparisonSpec",
    "ComparisonOptions",
    "ConcatOptions",
    "DTypeExpr",
    "LayerSpec",
    "ModelDiagnosticsSpec",
    "OneToOneOptions",
    "RegionSpec",
    "RuntimeStageOptions",
    "ShapeExpr",
    "SpecMatchCriteria",
    "StageSpec",
    "TensorMapping",
    "TensorMappingMode",
    "TensorRelation",
    "TensorSlotPair",
    "TensorSlotRef",
    "TheoryOperatorSpec",
    "TheoryStageOptions",
    "TheoryTensorSpec",
    "ProducerInfo",
    "SimulationExecutionArtifact",
    "DiagnosticValue",
    "DiagnosticsResult",
    "DiagnosticsSummary",
    "EvidenceRef",
    "Finding",
    "FindingStatus",
    "Limitation",
    "ModelRunContextSummary",
    "SourceDescription",
    "summarize_findings",
    "normalize_selected_layers",
    "normalize_selected_stage_regions",
]
