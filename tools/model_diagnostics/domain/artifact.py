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

"""Stable producer-to-diagnostics artifact values."""

from dataclasses import dataclass

from tools.model_diagnostics.domain.models import ModelRunContext, OperatorCallRecord

ARTIFACT_SCHEMA_VERSION = "1"


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
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact schema_version {self.schema_version!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "operator_calls", tuple(self.operator_calls))
        indices = tuple(call.call_index for call in self.operator_calls)
        if indices != tuple(range(len(indices))):
            raise ValueError("artifact operator call indices must be contiguous from zero")
