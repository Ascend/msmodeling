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

"""Minimal pytest-compatible assertion adapter."""

from tools.model_diagnostics.domain import DiagnosticsResult, Finding, FindingStatus


def assert_diagnostics_passed(result: DiagnosticsResult) -> None:
    """Explain non-PASS findings without rendering reports or writing files."""
    __tracebackhide__ = True
    if result.summary.overall_status is FindingStatus.PASS:
        return
    failing_count = 0
    previews: list[str] = []
    for finding in result.findings:
        if finding.status is FindingStatus.PASS:
            continue
        failing_count += 1
        if len(previews) < 5:
            previews.append(_finding_preview(finding))
    preview = "\n".join(previews)
    suffix = "" if failing_count <= 5 else f"\n... {failing_count - 5} more finding(s)"
    raise AssertionError(
        f"model diagnostics {result.summary.overall_status.value} for "
        f"{result.context.model_name} ({result.context.phase.value if result.context.phase else 'unknown'}): "
        f"{failing_count} non-pass finding(s)\n{preview}{suffix}"
    )


def _finding_preview(finding: Finding) -> str:
    layer = "" if finding.layer_index is None else f"/layer[{finding.layer_index}]"
    difference = ""
    if finding.expected is not None or finding.actual is not None:
        difference = f"; expected {finding.expected!r}, got {finding.actual!r}"
    evidence = finding.left_evidence or finding.right_evidence
    slot = next((item.tensor_slot for item in evidence if item.tensor_slot is not None), None)
    tensor = f" {slot}" if slot is not None else ""
    return (
        f"- {finding.region_id}{layer}/{finding.stage_id}{tensor}: "
        f"{finding.message}{difference}"
    )
