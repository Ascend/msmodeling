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
"""Offline MiniMax-M2 MTP Theory-to-Runtime diagnostics."""

from __future__ import annotations

import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile

_MODEL = "tests/assets/model_config/minimax_m2"


def test_minimax_m2_mtp_capture_and_compare_with_tp() -> None:
    """Protect the optional MTP Region and MTP predictor TP sharding."""

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=2,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
        parallel=ParallelContext(
            tensor_parallel_size=2,
            data_parallel_size=1,
            expert_parallel_size=2,
            moe_data_parallel_size=1,
        ),
        selected_language_layers=(0,),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="W8A8_STATIC",
        word_embedding_tp=None,
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert result.spec_id == "minimax_m2_v1"
    assert artifact.run_context.model_config["num_mtp_tokens"] == 1
    mtp = next(region for region in spec.regions if region.region_id == "mtp")
    assert mtp.layer_layout == ("minimax_m2_mtp",)
    assert (
        sum(call.operator_name == "tensor_cast.shift_and_update_input_ids.default" for call in artifact.operator_calls)
        == 1
    )
    mtp_findings = tuple(finding for finding in result.findings if finding.region_id == "mtp")
    assert mtp_findings
    expected_mtp_stages = {
        "input_shift",
        "q_norm",
        "k_norm",
        "attention",
        "moe_experts",
        "proposal_lm_head",
    }
    assert expected_mtp_stages <= {finding.stage_id for finding in mtp_findings}
    assert {finding.layer_index for finding in mtp_findings if finding.stage_id in expected_mtp_stages} == {0}
    assert all(finding.status is FindingStatus.PASS for finding in mtp_findings)
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
