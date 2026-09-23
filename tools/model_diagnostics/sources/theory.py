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
