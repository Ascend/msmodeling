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
