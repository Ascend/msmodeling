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

"""Project a stable simulation artifact into diagnostics domain records."""

from __future__ import annotations

from tools.model_diagnostics.domain import (
    ModelExecutionRecord,
    SimulationExecutionArtifact,
    SourceDescription,
    SourceKind,
)
from tools.model_diagnostics.errors import SourceLoadError


def execution_from_runtime_artifact(artifact: SimulationExecutionArtifact) -> ModelExecutionRecord:
    """Expose a captured artifact through the shared in-memory source record."""

    return ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=artifact.run_context,
        operator_calls=artifact.operator_calls,
    )


class SimulationArtifactSource:
    """Expose one in-memory Runtime artifact through the source port."""

    source_kind = SourceKind.RUNTIME

    def __init__(
        self,
        artifact: SimulationExecutionArtifact,
        *,
        artifact_reference: str | None = None,
    ) -> None:
        self._artifact = artifact
        self._artifact_reference = artifact_reference

    def describe(self) -> SourceDescription:
        return SourceDescription(
            source_kind=self.source_kind,
            artifact_reference=self._artifact_reference,
            producer=self._artifact.producer,
        )

    def load_execution(
        self,
        context,
        spec,
        selected_layers,
        selected_stage_regions,
    ) -> ModelExecutionRecord:
        if self._artifact.run_context != context:
            raise SourceLoadError("artifact context does not match the requested context")
        return execution_from_runtime_artifact(self._artifact)
