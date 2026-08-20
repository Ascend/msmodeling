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
"""Real Qwen3 Dense model diagnostics end-to-end cases."""

from __future__ import annotations

import pytest

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import (
    ExecutionOrganizationRequest,
    FindingStatus,
    SourceKind,
)
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.organization import RuntimeArtifactOrganizer
from tools.model_diagnostics.sources import SimulationArtifactSource


@pytest.mark.nightly
def test_qwen3_dense_capture_organize_and_compare(qwen3_dense_case) -> None:
    """Validate every Qwen3 Dense size through capture and comparison."""

    profile, artifact = qwen3_dense_case
    assert artifact.run_context.quantization_config["action"] == profile.quantize_linear_action
    if profile.quantize_linear_action == "DISABLED":
        assert artifact.run_context.quantization_config["enabled"] is False
    else:
        assert artifact.run_context.quantization_config["enabled"] is True
        assert artifact.run_context.quantization_config["linear_input_dtype"] == "int8"

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)
    assert result.spec_id == "qwen3_dense_v1"
    assert result.left_source.source_kind is SourceKind.THEORY
    assert result.right_source.source_kind is SourceKind.RUNTIME
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
    expected_stages = {
        "embedding",
        "attention_qkv",
        "attention",
        "dense_ffn",
        "lm_head",
    }
    assert {finding.stage_id for finding in result.findings} >= expected_stages
    assert all(finding.expected is not None and finding.actual is not None for finding in result.findings)

    operator_names = tuple(call.operator_name for call in artifact.operator_calls)
    quant_linear = "tensor_cast.static_quant_linear.default"
    if profile.quantize_linear_action == "DISABLED":
        assert quant_linear not in operator_names
    else:
        assert quant_linear in operator_names
        execution = SimulationArtifactSource(artifact).load_execution(
            artifact.run_context,
            spec,
            request.selected_layers,
            request.selected_stage_regions,
        )
        regions = RuntimeArtifactOrganizer().execute(
            ExecutionOrganizationRequest(
                execution=execution,
                spec=spec,
                selected_layers=request.selected_layers,
                selected_stage_regions=request.selected_stage_regions,
            )
        )
        language = regions[1].layers[0]
        assert [call.operator_name for call in language.stages[0].operator_calls] == [quant_linear]
        assert [call.operator_name for call in language.stages[1].operator_calls] == [
            "tensor_cast.attention.default",
            quant_linear,
        ]
        assert [call.operator_name for call in language.stages[2].operator_calls] == [
            quant_linear,
            "tensor_cast.swiglu.default",
            quant_linear,
        ]
        assert [stage.stage_id for stage in regions[2].stages] == ["lm_head"]
        output_operators = [call.operator_name for call in regions[2].stages[0].operator_calls]
        if profile.phase.value == "decode":
            assert output_operators == ["aten.mm.default"]
        else:
            assert output_operators == ["aten.index.Tensor", "aten.mm.default"]


@pytest.mark.nightly
def test_qwen3_dense_mtp_decode_passes_diagnostics(qwen3_dense_mtp_case) -> None:
    """Cover every Qwen3 Dense size, including quantized MTP Runtime."""

    profile, artifact = qwen3_dense_mtp_case
    assert profile.phase.value == "decode"
    assert artifact.run_context.query_length == 3
    assert artifact.run_context.model_config["num_mtp_tokens"] == 2
    operator_names = [call.operator_name for call in artifact.operator_calls]
    assert operator_names.count("tensor_cast.shift_and_update_input_ids.default") == 2
    quant_linear = "tensor_cast.static_quant_linear.default"
    if profile.quantize_linear_action == "DISABLED":
        assert artifact.run_context.quantization_config["enabled"] is False
        assert quant_linear not in operator_names
    else:
        assert artifact.run_context.quantization_config["enabled"] is True
        assert artifact.run_context.quantization_config["linear_input_dtype"] == "int8"
        assert quant_linear in operator_names

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
