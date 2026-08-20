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
"""DeepSeek-V3 Theory-to-Runtime end-to-end diagnostics."""

from __future__ import annotations

import pytest
import torch

from conftest import assert_parallel_contract
from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile
from tools.model_diagnostics.specification.context_env import build_theory_env


@pytest.fixture(
    scope="module",
    params=(
        pytest.param(
            ("tests/assets/model_config/deepseek_v32", ExecutionPhase.PREFILL, 2, None, "DISABLED"), id="v32_prefill"
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v32", ExecutionPhase.DECODE, 1, 128, "DISABLED"), id="v32_decode"
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v32", ExecutionPhase.PREFILL, 2, None, "W8A8_DYNAMIC"),
            id="v32_w8a8",
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v32", ExecutionPhase.PREFILL, 2, None, "W4A8_DYNAMIC"),
            id="v32_w4a8",
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v3", ExecutionPhase.PREFILL, 2, None, "DISABLED"), id="v3_prefill"
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v3", ExecutionPhase.DECODE, 1, 128, "DISABLED"), id="v3_decode"
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v3", ExecutionPhase.PREFILL, 2, None, "W8A8_DYNAMIC"),
            id="v3_w8a8",
            marks=pytest.mark.nightly,
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v31", ExecutionPhase.PREFILL, 2, None, "DISABLED"),
            id="v31_prefill",
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v31", ExecutionPhase.DECODE, 1, 128, "DISABLED"),
            id="v31_decode",
            marks=pytest.mark.nightly,
        ),
        pytest.param(
            ("tests/assets/model_config/deepseek_v31", ExecutionPhase.PREFILL, 2, None, "W8A8_DYNAMIC"),
            id="v31_w8a8",
            marks=pytest.mark.nightly,
        ),
    ),
)
def deepseek_v3_case(request):
    """Capture Dense-prefix and first MoE/shared-expert layer."""

    model_name, phase, query_length, context_length, quantization = request.param
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
        entrypoint="text_generate",
        phase=phase,
        batch_size=1,
        query_length=query_length,
        context_length=context_length,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=None,
        selected_stage_regions=(),
        num_hidden_layers_override=4,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action=quantization,
        word_embedding_tp=None,
    )
    torch.compiler.reset()
    return profile, capture_artifact_for_profile(profile)


def test_deepseek_v3_capture_organize_and_compare(deepseek_v3_case) -> None:
    profile, artifact = deepseek_v3_case
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert result.spec_id == "deepseek_v3_v1"
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
    stage_ids = {finding.stage_id for finding in result.findings}
    assert stage_ids >= {
        "embedding",
        "mla_projection",
        "sparse_attention",
        "dense_ffn",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
        "lm_head",
    }
    assert ("dsa_indexer" in stage_ids) is (artifact.run_context.model_config["model_type"] == "deepseek_v32")
    operator_names = {call.operator_name for call in artifact.operator_calls}
    if profile.quantize_linear_action == "W8A8_DYNAMIC":
        assert "tensor_cast.mlapo_quant.default" in operator_names
        assert "tensor_cast.static_quant_linear.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant_swiglu.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant.default" in operator_names
    elif profile.quantize_linear_action == "W4A8_DYNAMIC":
        assert "tensor_cast.mlapo_quant.default" in operator_names
        assert "tensor_cast.static_quant_linear_int4.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant_int4_swiglu.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant_int4.default" in operator_names


@pytest.mark.parametrize(
    "model_name",
    (
        pytest.param("tests/assets/model_config/deepseek_v32", id="v32"),
        pytest.param("tests/assets/model_config/deepseek_v3", id="v3", marks=pytest.mark.nightly),
        pytest.param("tests/assets/model_config/deepseek_v31", id="v31", marks=pytest.mark.nightly),
    ),
)
def test_deepseek_v3_mtp_capture_organize_and_compare(model_name: str) -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
        parallel=ParallelContext(),
        selected_language_layers=None,
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

    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
    mtp_stages = {finding.stage_id for finding in result.findings if finding.region_id == "mtp"}
    expected_mtp_stages = {
        "input_shift",
        "mla_projection",
        "sparse_attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_ffn",
        "proposal_selection",
    }
    assert expected_mtp_stages.issubset(mtp_stages)
    assert ("dsa_indexer" in mtp_stages) is (artifact.run_context.model_config["model_type"] == "deepseek_v32")


@pytest.mark.parametrize(
    ("parallel", "query_length", "expected_tmoe", "expected_local_heads"),
    (
        (
            ParallelContext(
                tensor_parallel_size=2,
                data_parallel_size=1,
                expert_parallel_size=2,
                moe_data_parallel_size=1,
            ),
            3,
            2,
            64,
        ),
        (
            ParallelContext(
                tensor_parallel_size=1,
                data_parallel_size=2,
                expert_parallel_size=1,
                moe_data_parallel_size=2,
            ),
            2,
            4,
            128,
        ),
    ),
    ids=("tp2_ep2", "dp2_mdp2"),
)
def test_deepseek_v3_parallel_shapes_compare(
    parallel: ParallelContext,
    query_length: int,
    expected_tmoe: int,
    expected_local_heads: int,
) -> None:
    """Validate rank-local MLA heads and MoE token domains under parallel layouts."""

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="tests/assets/model_config/deepseek_v32",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=query_length,
        context_length=None,
        num_mtp_tokens=0,
        parallel=parallel,
        selected_language_layers=None,
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

    env = build_theory_env(artifact.run_context)
    assert env["Lh"] == expected_local_heads
    assert env["Tmoe"] == expected_tmoe
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS


@pytest.mark.parametrize(
    "model_name",
    ("tests/assets/model_config/deepseek_v3", "tests/assets/model_config/deepseek_v31"),
    ids=("v3", "v31"),
)
@pytest.mark.nightly
@pytest.mark.parametrize(
    ("parallel", "query_length"),
    (
        (
            ParallelContext(
                tensor_parallel_size=2,
                data_parallel_size=1,
                expert_parallel_size=2,
                moe_data_parallel_size=1,
            ),
            3,
        ),
        (
            ParallelContext(
                tensor_parallel_size=1,
                data_parallel_size=2,
                expert_parallel_size=1,
                moe_data_parallel_size=2,
            ),
            2,
        ),
    ),
    ids=("tp2_ep2", "dp2_mdp2"),
)
def test_deepseek_v3_legacy_parallel_shapes_compare(
    model_name: str,
    parallel: ParallelContext,
    query_length: int,
) -> None:
    """Validate rank-local MLA heads and MoE token domains for V3/V3.1 under parallel layouts."""

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=query_length,
        context_length=None,
        num_mtp_tokens=0,
        parallel=parallel,
        selected_language_layers=None,
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
    result = application.run_against_artifact(
        profile.to_request(context=artifact.run_context, spec=spec),
        artifact,
    )
    env = build_theory_env(artifact.run_context)
    assert_parallel_contract(env, parallel, raw_logits_gate=True)
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
