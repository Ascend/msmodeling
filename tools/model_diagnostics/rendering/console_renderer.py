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

"""Readable terminal rendering with detailed findings and compact limitations."""

from __future__ import annotations

from tools.model_diagnostics.domain import DiagnosticsResult, Finding, FindingStatus

from .formatting import context_line, display_actual, display_expected, finding_location


class ConsoleResultRenderer:
    media_type = "text/plain"

    def __init__(self, *, show_all: bool = False, fail_only: bool = False) -> None:
        self._show_all = show_all
        self._fail_only = fail_only

    def render(self, result: DiagnosticsResult) -> str:
        lines: list[str] = []
        findings = self._visible_findings(result)
        if findings:
            lines.append("")
        for index, finding in enumerate(findings):
            if index:
                lines.append("")
            lines.extend(_detailed_finding(finding))
        if result.limitations:
            if self._show_all:
                lines.extend(("", "Limitations:"))
                lines.extend(f"  {item.code}: {item.message}" for item in result.limitations)
            else:
                lines.extend(("", "Use --show-all to display details."))
        failures = tuple(finding for finding in result.findings if finding.status is not FindingStatus.PASS)
        outcome = "PASS" if result.summary.overall_status is FindingStatus.PASS else "FAIL"
        lines.extend(("", f"Model diagnostics: {outcome}", context_line(result)))
        lines.append(_summary_line(result))
        if failures:
            lines.append("Failure list:")
            lines.extend(f"- {_failure_location(finding)}" for finding in failures)
        return "\n".join(lines).lstrip("\n") + "\n"

    def _visible_findings(self, result: DiagnosticsResult) -> tuple[Finding, ...]:
        if self._show_all or not self._fail_only:
            return result.findings
        return tuple(finding for finding in result.findings if finding.status is not FindingStatus.PASS)


def _summary_line(result: DiagnosticsResult) -> str:
    counts = result.summary.counts_by_status
    passed = counts[FindingStatus.PASS]
    failed = sum(counts[status] for status in FindingStatus if status is not FindingStatus.PASS)
    return f"Summary: {passed} pass, {failed} fail"


def _detailed_finding(finding: Finding) -> list[str]:
    return [
        f"{finding.status.value.upper()}  {finding_location(finding)}",
        f"  Expected: {display_expected(finding) or '-'}",
        f"  Actual:   {display_actual(finding) or '-'}",
        f"  Message:  {finding.message}",
    ]


def _failure_location(finding: Finding) -> str:
    parts = [finding.region_id]
    if finding.layer_index is not None:
        parts.append(f"layer[{finding.layer_index}]")
    parts.append(finding.stage_id)
    evidence = (*finding.left_evidence, *finding.right_evidence)
    slot = next((item.tensor_slot for item in evidence if item.tensor_slot is not None), None)
    if slot is not None:
        parts.append(f"{slot.direction.value.upper()}[{slot.index}]")
    return "/".join(parts)
