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

"""Immutable values shared by diagnostics sources and strategies."""

from tools.model_diagnostics.domain.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ProducerInfo,
    SimulationExecutionArtifact,
)
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
    validate_expert_parallel_features,
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
    "validate_expert_parallel_features",
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
    "ARTIFACT_SCHEMA_VERSION",
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
