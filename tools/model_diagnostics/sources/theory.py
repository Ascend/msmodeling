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
"""Theory OperatorRecordSource: materialize selected Spec modules as calls."""

from __future__ import annotations

from typing import Mapping

from tools.model_diagnostics.domain.models import (
    ModelExecutionRecord,
    ModelRunContext,
    SourceKind,
)
from tools.model_diagnostics.domain.results import SourceDescription
from tools.model_diagnostics.domain.specification import ModelDiagnosticsSpec
from tools.model_diagnostics.organization.theory import (
    TheoryExecutionOrganizationStrategy,
    build_theory_regions,
    flatten_theory_calls,
)

# Re-export organizer next to the shared builder for package consumers.
__all__ = [
    "TheoryOperatorRecordSource",
    "TheoryExecutionOrganizationStrategy",
]


class TheoryOperatorRecordSource:
    """Build Theory calls only for requested regions and physical layers."""

    source_kind = SourceKind.THEORY

    def describe(self) -> SourceDescription:
        return SourceDescription(source_kind=self.source_kind)

    def load_execution(
        self,
        context: ModelRunContext,
        spec: ModelDiagnosticsSpec,
        selected_layers: Mapping[str, tuple[int, ...]],
        selected_stage_regions: tuple[str, ...],
    ) -> ModelExecutionRecord:
        regions = build_theory_regions(
            context,
            spec,
            selected_layers,
            selected_stage_regions,
        )
        return ModelExecutionRecord(
            source_kind=SourceKind.THEORY,
            run_context=context,
            operator_calls=flatten_theory_calls(regions),
        )
