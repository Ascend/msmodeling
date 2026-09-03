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
"""Offline DeepSeek V4 Theory-to-Runtime diagnostics."""

from __future__ import annotations

import pytest
import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile

_MODEL = "tests/assets/model_config/deepseek_v4_flash"


@pytest.mark.parametrize("phase", (ExecutionPhase.PREFILL, ExecutionPhase.DECODE))
def test_deepseek_v4_offline_capture_and_compare(phase: ExecutionPhase) -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=phase,
        batch_size=80,
        query_length=1,
        context_length=65500,
        num_mtp_tokens=0,
        parallel=ParallelContext(
            tensor_parallel_size=4,
            data_parallel_size=16,
            expert_parallel_size=64,
        ),
        selected_language_layers=(0,),
        selected_stage_regions=("input", "output"),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="W8A8_DYNAMIC",
        word_embedding_tp=None,
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert artifact.run_context.model_config["model_type"] == "deepseek_v4"
    assert result.spec_id == "deepseek_v4_v1"
    assert_diagnostics_passed(result)


def test_deepseek_v4_mtp_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
        parallel=ParallelContext(),
        selected_language_layers=(0,),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert artifact.run_context.model_config["num_mtp_tokens"] == 1
    assert (
        sum(call.operator_name == "tensor_cast.shift_and_update_input_ids.default" for call in artifact.operator_calls)
        == 1
    )

    mtp_region = next(region for region in spec.regions if region.region_id == "mtp")
    assert len(mtp_region.layer_layout) == profile.num_mtp_tokens
    assert mtp_region.layer_layout == ("deepseek_v4_mtp",)
    predictor_stages = {stage.stage_id for stage in mtp_region.layer_specs["deepseek_v4_mtp"].stages}
    assert {
        "input_shift",
        "hc_expand",
        "hc_pre_attention",
        "sparse_attention",
        "moe_gate",
        "moe_experts",
        "hc_post_moe",
        "hc_reduce",
        "proposal_selection",
    }.issubset(predictor_stages)

    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
    mtp_findings = tuple(finding for finding in result.findings if finding.region_id == "mtp")
    assert mtp_findings
    assert all(finding.status is FindingStatus.PASS for finding in mtp_findings)


def test_deepseek_v4_compressed_attention_paths_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=2,
        context_length=128,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=(2, 3),
        selected_stage_regions=(),
        num_hidden_layers_override=4,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert artifact.run_context.model_config["layer_types"][:4] == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert sum(call.operator_name == "tensor_cast.compressor.default" for call in artifact.operator_calls) == 3
    assert (
        sum(call.operator_name == "tensor_cast.quant_lightning_indexer.default" for call in artifact.operator_calls)
        == 1
    )
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
