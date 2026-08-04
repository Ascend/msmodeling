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
"""Stable producer-to-diagnostics artifact values."""

from dataclasses import dataclass

from tools.model_diagnostics.domain.models import ModelRunContext, OperatorCallRecord


@dataclass(frozen=True)
class ProducerInfo:
    package_version: str
    git_revision: str | None
    capture_backend: str

    def __post_init__(self) -> None:
        if not self.package_version.strip():
            raise ValueError("package_version must not be empty")
        if not self.capture_backend.strip():
            raise ValueError("capture_backend must not be empty")


@dataclass(frozen=True)
class SimulationExecutionArtifact:
    schema_version: str
    producer: ProducerInfo
    run_context: ModelRunContext
    operator_calls: tuple[OperatorCallRecord, ...]

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        object.__setattr__(self, "operator_calls", tuple(self.operator_calls))
        indices = tuple(call.call_index for call in self.operator_calls)
        if indices != tuple(range(len(indices))):
            raise ValueError("artifact operator call indices must be contiguous from zero")
