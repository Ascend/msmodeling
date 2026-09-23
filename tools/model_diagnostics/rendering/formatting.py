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

"""Shared human-readable formatting for diagnostics renderers."""

from __future__ import annotations

from tools.model_diagnostics.domain import DiagnosticsResult, Finding


def context_line(result: DiagnosticsResult) -> str:
    context = result.context
    phase = context.phase.value if context.phase is not None else "?"
    batch = context.batch_size if context.batch_size is not None else "?"
    query = context.query_length if context.query_length is not None else "?"
    ctx = 0 if context.context_length is None else context.context_length
    tp = 1 if context.tensor_parallel_size is None else context.tensor_parallel_size
    return f"{context.model_name} | {phase} | batch={batch} query={query} context={ctx} | TP={tp}"


def finding_location(finding: Finding) -> str:
    layer = "" if finding.layer_index is None else f"layer[{finding.layer_index}]/"
    theory_ops = unique_operator_names(finding.left_evidence)
    if len(theory_ops) == 1:
        return f"{layer}{theory_ops[0]}"
    if theory_ops:
        return f"{layer}{'+'.join(theory_ops)}"
    return f"{layer}{finding.stage_id}"


def display_expected(finding: Finding) -> str:
    return "" if finding.expected is None else str(finding.expected)


def display_actual(finding: Finding) -> str:
    shapes = "" if finding.actual is None else str(finding.actual)
    runtime_ops = unique_operator_names(finding.right_evidence)
    if not runtime_ops:
        return shapes
    operator = runtime_ops[0] if len(runtime_ops) == 1 else "+".join(runtime_ops)
    return operator if not shapes else f"{operator} | {shapes}"


def unique_operator_names(evidence) -> list[str]:
    names: list[str] = []
    for item in evidence:
        if item.operator_name not in names:
            names.append(item.operator_name)
    return names
