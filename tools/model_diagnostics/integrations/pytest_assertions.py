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
