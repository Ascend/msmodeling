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
import pytest

from tools.model_diagnostics.domain import (
    DiagnosticsSummary,
    EvidenceRef,
    Finding,
    FindingStatus,
    SourceKind,
    summarize_findings,
)


def _finding(status: FindingStatus) -> Finding:
    return Finding(
        region_id="language",
        layer_index=0,
        stage_id="attention",
        rule_id="shape_equal",
        comparison_kind="one_to_one",
        status=status,
        message_code=f"shape.{status.value}",
        message=f"shape check is {status.value}",
    )


@pytest.mark.parametrize(
    ("statuses", "overall"),
    (
        ((), FindingStatus.PASS),
        ((FindingStatus.PASS,), FindingStatus.PASS),
        ((FindingStatus.PASS, FindingStatus.SKIP), FindingStatus.SKIP),
        ((FindingStatus.UNSUPPORTED, FindingStatus.SKIP), FindingStatus.UNSUPPORTED),
        ((FindingStatus.INCOMPLETE, FindingStatus.UNSUPPORTED), FindingStatus.INCOMPLETE),
        ((FindingStatus.FAIL, FindingStatus.INCOMPLETE), FindingStatus.FAIL),
    ),
)
def test_summary_uses_deterministic_worst_status(statuses, overall: FindingStatus) -> None:
    summary = summarize_findings(tuple(_finding(status) for status in statuses))

    assert summary.overall_status is overall
    assert sum(summary.counts_by_status.values()) == len(statuses)
    assert set(summary.counts_by_status) == set(FindingStatus)


def test_summary_mapping_is_immutable() -> None:
    summary = DiagnosticsSummary(
        overall_status=FindingStatus.PASS,
        counts_by_status={FindingStatus.PASS: 1},
    )

    with pytest.raises(TypeError):
        summary.counts_by_status[FindingStatus.FAIL] = 1


def test_evidence_rejects_negative_positions() -> None:
    with pytest.raises(ValueError, match="stage_call_position"):
        EvidenceRef(
            source_kind=SourceKind.RUNTIME,
            call_index=4,
            stage_call_position=-1,
            operator_name="aten.mm.default",
        )
