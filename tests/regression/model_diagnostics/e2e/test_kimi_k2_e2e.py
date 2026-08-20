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
"""Kimi-K2 text-path Theory-to-Runtime end-to-end diagnostics."""

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

# Kimi-K2/K2.5/K2.6 configs carry auto_map remote code and cannot be vendored
# offline as config.json only; these cases need live Hub access and therefore
# run only in the nightly network phase (never in the default/gate path).
pytestmark = pytest.mark.network


def _run_kimi_k2_case(
    *,
    model_name: str = "moonshotai/Kimi-K2-Base",
    phase: ExecutionPhase,
    query_length: int,
    context_length: int | None,
    quantization: str = "DISABLED",
    num_mtp_tokens: int = 0,
    parallel: ParallelContext = ParallelContext(),
):
    if model_name in {"moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6"}:
        # TensorCast supports one model per process. Pytest intentionally
        # exercises both compatible model ids in one worker, so reset only
        # the process-level remote-class patch guard to emulate separate CLI runs.
        import tensor_cast.transformers.builtin_model.kimi_k25 as kimi_k25

        kimi_k25._patched_kimi_k25 = False

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
        entrypoint="text_generate",
        phase=phase,
        batch_size=1,
        query_length=query_length,
        context_length=context_length,
        num_mtp_tokens=num_mtp_tokens,
        parallel=parallel,
        selected_language_layers=None,
        selected_stage_regions=(),
        # Kimi-K2 starts with one Dense layer; include layer[1] to cover MoE.
        num_hidden_layers_override=2,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action=quantization,
        word_embedding_tp=None,
    )
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    return profile, artifact, application.run_against_artifact(request, artifact)


@pytest.mark.parametrize(
    ("phase", "query_length", "context_length", "quantization"),
    (
        pytest.param(ExecutionPhase.PREFILL, 2, None, "DISABLED", id="prefill"),
        pytest.param(ExecutionPhase.DECODE, 1, 128, "DISABLED", id="decode"),
        pytest.param(
            ExecutionPhase.PREFILL,
            2,
            None,
            "W8A8_DYNAMIC",
            id="w8a8",
            marks=pytest.mark.nightly,
        ),
    ),
)
def test_kimi_k2_base_capture_organize_and_compare(
    phase: ExecutionPhase,
    query_length: int,
    context_length: int | None,
    quantization: str,
) -> None:
    profile, artifact, result = _run_kimi_k2_case(
        phase=phase,
        query_length=query_length,
        context_length=context_length,
        quantization=quantization,
    )

    assert artifact.run_context.model_config["model_type"] == "deepseek_v3"
    assert result.spec_id == "deepseek_v3_v1"
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
    stage_ids = {finding.stage_id for finding in result.findings}
    assert "dsa_indexer" not in stage_ids
    assert {"sparse_attention", "moe_experts", "moe_combine", "shared_ffn"}.issubset(stage_ids)
    if profile.quantize_linear_action == "W8A8_DYNAMIC":
        operator_names = {call.operator_name for call in artifact.operator_calls}
        assert "tensor_cast.mlapo_quant.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant_swiglu.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant.default" in operator_names


@pytest.mark.nightly
def test_kimi_k2_base_mtp_capture_organize_and_compare() -> None:
    _, _, result = _run_kimi_k2_case(
        phase=ExecutionPhase.DECODE,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
    )

    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
    mtp_stages = {finding.stage_id for finding in result.findings if finding.region_id == "mtp"}
    assert {"sparse_attention", "moe_experts", "moe_combine", "shared_ffn", "proposal_selection"}.issubset(mtp_stages)


@pytest.mark.parametrize(
    "model_name",
    (
        pytest.param("moonshotai/Kimi-K2.5", id="kimi-k2.5"),
        pytest.param("moonshotai/Kimi-K2.6", id="kimi-k2.6", marks=pytest.mark.nightly),
    ),
)
@pytest.mark.parametrize(
    ("phase", "query_length", "context_length", "quantization"),
    (
        pytest.param(ExecutionPhase.PREFILL, 2, None, "DISABLED", id="prefill"),
        pytest.param(ExecutionPhase.DECODE, 1, 128, "DISABLED", id="decode"),
        pytest.param(
            ExecutionPhase.PREFILL,
            2,
            None,
            "W8A8_DYNAMIC",
            id="w8a8",
            marks=pytest.mark.nightly,
        ),
    ),
)
def test_kimi_k25_compatible_text_capture_organize_and_compare(
    model_name: str,
    phase: ExecutionPhase,
    query_length: int,
    context_length: int | None,
    quantization: str,
) -> None:
    profile, artifact, result = _run_kimi_k2_case(
        model_name=model_name,
        phase=phase,
        query_length=query_length,
        context_length=context_length,
        quantization=quantization,
    )

    assert artifact.run_context.model_config["model_type"] == "kimi_k2"
    assert result.spec_id == "deepseek_v3_v1"
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
    if profile.quantize_linear_action == "W8A8_DYNAMIC":
        operator_names = {call.operator_name for call in artifact.operator_calls}
        assert "tensor_cast.mlapo_quant.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant_swiglu.default" in operator_names
        assert "tensor_cast.grouped_matmul_quant.default" in operator_names


@pytest.mark.parametrize(
    "model_name",
    ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6"),
    ids=("kimi-k2.5", "kimi-k2.6"),
)
@pytest.mark.nightly
def test_kimi_k25_compatible_text_mtp_capture_organize_and_compare(model_name: str) -> None:
    _, artifact, result = _run_kimi_k2_case(
        model_name=model_name,
        phase=ExecutionPhase.DECODE,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
    )

    assert artifact.run_context.model_config["model_type"] == "kimi_k2"
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
    mtp_stages = {finding.stage_id for finding in result.findings if finding.region_id == "mtp"}
    assert {"sparse_attention", "moe_experts", "moe_combine", "proposal_selection"}.issubset(mtp_stages)


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
@pytest.mark.nightly
def test_kimi_k2_parallel_shapes_compare(
    parallel: ParallelContext,
    query_length: int,
) -> None:
    """Validate rank-local MLA heads and MoE token domains for Kimi-K2-Base under parallel layouts."""

    _, artifact, result = _run_kimi_k2_case(
        model_name="moonshotai/Kimi-K2-Base",
        phase=ExecutionPhase.PREFILL,
        query_length=query_length,
        context_length=None,
        parallel=parallel,
    )

    env = build_theory_env(artifact.run_context)
    assert_parallel_contract(env, parallel, raw_logits_gate=True)
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS


@pytest.mark.parametrize(
    "model_name",
    ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6"),
    ids=("kimi-k2.5", "kimi-k2.6"),
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
def test_kimi_k25_compatible_parallel_shapes_compare(
    model_name: str,
    parallel: ParallelContext,
    query_length: int,
) -> None:
    """Validate rank-local MLA heads and MoE token domains for Kimi K2.5/K2.6 under parallel layouts."""

    _, artifact, result = _run_kimi_k2_case(
        model_name=model_name,
        phase=ExecutionPhase.PREFILL,
        query_length=query_length,
        context_length=None,
        parallel=parallel,
    )

    env = build_theory_env(artifact.run_context)
    assert_parallel_contract(env, parallel, raw_logits_gate=False)
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
