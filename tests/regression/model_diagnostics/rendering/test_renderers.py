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
from dataclasses import replace

import pytest

from tools.model_diagnostics.domain import (
    DiagnosticsResult,
    EvidenceRef,
    ExecutionPhase,
    Finding,
    FindingStatus,
    Limitation,
    ModelRunContextSummary,
    ModelRunContext,
    OperatorCallRecord,
    ParallelContext,
    ProducerInfo,
    SimulationExecutionArtifact,
    SourceDescription,
    SourceKind,
    summarize_findings,
)
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.rendering import (
    ComparisonHtmlRenderer,
    ConsoleResultRenderer,
    RuntimeHtmlRenderer,
    write_html_report,
)


def _result(status: FindingStatus, *, message="shape differs") -> DiagnosticsResult:
    finding = Finding(
        region_id="language",
        layer_index=0,
        stage_id="attention",
        rule_id="output.shape",
        comparison_kind="shape",
        status=status,
        message_code=f"tensor.{status.value}",
        message=message,
        expected=(2, 4),
        actual=(2, 5),
        left_evidence=(EvidenceRef(SourceKind.THEORY, 1, 0, "attention"),),
        right_evidence=(EvidenceRef(SourceKind.RUNTIME, 3, 0, "tensor_cast.attention.default"),),
    )
    findings = (finding,)
    return DiagnosticsResult(
        schema_version="1",
        spec_id="qwen3",
        spec_version="1.0.0",
        context=ModelRunContextSummary(
            model_name="Qwen/Qwen3",
            entrypoint="test",
            phase=ExecutionPhase.PREFILL,
            batch_size=1,
            query_length=2,
            context_length=2,
            tensor_parallel_size=1,
        ),
        left_source=SourceDescription(SourceKind.THEORY),
        right_source=SourceDescription(SourceKind.RUNTIME),
        selected_layers={"language": (0,)},
        selected_stage_regions=(),
        findings=findings,
        summary=summarize_findings(findings),
    )


def test_comparison_html_uses_operator_location_and_context_line() -> None:
    rendered = ComparisonHtmlRenderer().render(_result(FindingStatus.FAIL, message="left | right"))

    assert "Model diagnostics: FAIL" in rendered
    assert "Qwen/Qwen3 | prefill | batch=1 query=2 context=2 | TP=1" in rendered
    assert "layer[0]/attention" in rendered
    assert "tensor_cast.attention.default | (2, 5)" in rendered
    assert "left | right" in rendered


def test_runtime_html_lists_every_call_and_writes_atomically(tmp_path) -> None:
    artifact = SimulationExecutionArtifact(
        schema_version="1",
        producer=ProducerInfo("1.0", None, "test-backend"),
        run_context=ModelRunContext(
            model_name="model<&>",
            entrypoint="test",
            phase=ExecutionPhase.DECODE,
            batch_size=1,
            query_length=1,
            context_length=8,
            parallel=ParallelContext(),
            model_config={},
            quantization_config={},
        ),
        operator_calls=(OperatorCallRecord(0, "aten.mm<&>", None, ()),),
    )
    rendered = RuntimeHtmlRenderer().render(artifact)
    report = tmp_path / "nested" / "runtime.html"
    write_html_report(report, rendered)

    assert "1 operator calls" in rendered
    assert "aten.mm&lt;&amp;&gt;" in rendered
    assert report.read_text(encoding="utf-8") == rendered


def test_console_renderer_expands_problem_details() -> None:
    rendered = ConsoleResultRenderer().render(_result(FindingStatus.FAIL))

    assert rendered.endswith("Failure list:\n- language/layer[0]/attention\n")
    assert "Summary: 0 pass, 1 fail" in rendered
    assert "FAIL  layer[0]/attention" in rendered
    assert "  Expected: (2, 4)" in rendered
    assert "  Actual:   tensor_cast.attention.default | (2, 5)" in rendered
    assert "  Message:  shape differs" in rendered
    assert "|" in rendered
    assert "| Status |" not in rendered


def test_console_renderer_expands_passing_finding_details_by_default() -> None:
    result = _result(FindingStatus.PASS)

    rendered = ConsoleResultRenderer().render(result)

    assert "PASS  layer[0]/attention" in rendered
    assert "  Expected: (2, 4)" in rendered
    assert "  Actual:   tensor_cast.attention.default | (2, 5)" in rendered
    assert "Failure list:" not in rendered
    assert "- none" not in rendered


@pytest.mark.parametrize(
    "status",
    (
        FindingStatus.FAIL,
        FindingStatus.INCOMPLETE,
        FindingStatus.UNSUPPORTED,
        FindingStatus.SKIP,
    ),
)
def test_console_renderer_failure_list_includes_every_non_pass_status(status: FindingStatus) -> None:
    rendered = ConsoleResultRenderer().render(_result(status))

    assert "Failure list:" in rendered
    assert "- language/layer[0]/attention" in rendered


def test_console_renderer_compacts_limitations_unless_show_all() -> None:
    base = _result(FindingStatus.PASS)
    result = replace(
        base,
        limitations=(Limitation(code="known.limit", message="details"),),
    )

    compact = ConsoleResultRenderer().render(result)
    expanded = ConsoleResultRenderer(show_all=True).render(result)

    assert "Use --show-all to display details." in compact
    assert "Limitations:" not in compact
    assert "known.limit" not in compact
    assert "Limitations:" in expanded
    assert "known.limit: details" in expanded


def test_console_renderer_fail_only_hides_pass() -> None:
    rendered = ConsoleResultRenderer(fail_only=True).render(_result(FindingStatus.PASS))

    assert "Summary: 1 pass" in rendered
    assert "layer[0]/attention" not in rendered


def test_console_show_all_overrides_fail_only() -> None:
    rendered = ConsoleResultRenderer(show_all=True, fail_only=True).render(_result(FindingStatus.PASS))
    assert "PASS  layer[0]/attention" in rendered


def test_pytest_adapter_returns_only_for_pass() -> None:
    assert_diagnostics_passed(_result(FindingStatus.PASS))

    with pytest.raises(AssertionError, match="tensor.fail"):
        assert_diagnostics_passed(_result(FindingStatus.FAIL))
