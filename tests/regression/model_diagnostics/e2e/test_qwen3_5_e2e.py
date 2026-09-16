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
"""Qwen3.5 hybrid (full + linear attention) Theory-to-Runtime E2E diagnostics."""

from __future__ import annotations

import pytest
import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile
from tools.model_diagnostics.specification.context_env import build_theory_env

# Vendored offline configs (transformers 5.13.x natively supports these
# architectures, so config.json is sufficient; no remote-code .py needed).
_QWEN35_DENSE = "tests/assets/model_config/qwen3_5_27b"
_QWEN35_MOE = "tests/assets/model_config/qwen3_5_moe_397b_a17b"
_QWEN3_NEXT = "tests/assets/model_config/qwen3_next_80b_a3b"


def _run_qwen35_case(
    *,
    model_name: str,
    phase: ExecutionPhase,
    query_length: int,
    context_length: int | None,
    quantization: str = "DISABLED",
    parallel: ParallelContext = ParallelContext(),
    num_mtp_tokens: int = 0,
):
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
        # Six layers cover the [linear x3, full, linear, linear] hybrid prefix.
        num_hidden_layers_override=6,
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
    "model_name",
    (
        pytest.param(_QWEN35_DENSE, id="dense"),
        pytest.param(_QWEN35_MOE, id="moe", marks=pytest.mark.nightly),
        pytest.param(_QWEN3_NEXT, id="next", marks=pytest.mark.nightly),
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
def test_qwen35_capture_organize_and_compare(
    model_name: str,
    phase: ExecutionPhase,
    query_length: int,
    context_length: int | None,
    quantization: str,
) -> None:
    profile, artifact, result = _run_qwen35_case(
        model_name=model_name,
        phase=phase,
        query_length=query_length,
        context_length=context_length,
        quantization=quantization,
    )

    assert result.spec_id == "qwen3_5_next_v1"
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
    stage_ids = {finding.stage_id for finding in result.findings}
    assert "attention" in stage_ids
    if model_name == _QWEN3_NEXT:
        assert {"linear_projection", "linear_delta_rule", "linear_output"}.issubset(stage_ids)
        assert {"moe_gate", "moe_experts", "moe_combine", "shared_ffn"}.issubset(stage_ids)
    else:
        assert {"linear_projection", "linear_delta_rule", "linear_output"}.issubset(stage_ids)
        if model_name == _QWEN35_DENSE:
            assert "dense_ffn" in stage_ids
        else:
            assert {"moe_gate", "moe_experts", "moe_combine", "shared_ffn"}.issubset(stage_ids)
    if profile.quantize_linear_action == "W8A8_DYNAMIC":
        operator_names = {call.operator_name for call in artifact.operator_calls}
        assert "tensor_cast.static_quant_linear.default" in operator_names


@pytest.mark.nightly
@pytest.mark.parametrize(
    "model_name",
    (
        pytest.param(_QWEN35_DENSE, id="dense"),
        pytest.param(_QWEN35_MOE, id="moe"),
        pytest.param(_QWEN3_NEXT, id="next"),
    ),
)
def test_qwen35_tp2_parallel_shapes_compare(model_name: str) -> None:
    """Validate rank-local full/linear attention head sharding under TP=2."""

    # Qwen3.5-MoE / Qwen3-Next are MoE models: diagnostics requires
    # MDP*EP == world_size (MoE tensor parallel is fixed at 1), so TP=2 needs
    # MoE-DP=2 to satisfy the layout check while still sharding attention
    # heads by TP. Runtime keeps experts on the full routed domain (MoE TP=1).
    parallel = ParallelContext(
        tensor_parallel_size=2,
        moe_data_parallel_size=2 if model_name in {_QWEN35_MOE, _QWEN3_NEXT} else 1,
    )
    _, artifact, result = _run_qwen35_case(
        model_name=model_name,
        phase=ExecutionPhase.PREFILL,
        query_length=3,
        context_length=None,
        parallel=parallel,
    )
    env = build_theory_env(artifact.run_context)
    model_config = artifact.run_context.model_config
    assert env["Lh"] == int(model_config["num_attention_heads"]) // parallel.tensor_parallel_size
    assert env["Lk_lin"] == int(model_config["linear_num_key_heads"]) // parallel.tensor_parallel_size
    assert env["Lv_lin"] == int(model_config["linear_num_value_heads"]) // parallel.tensor_parallel_size
    if model_name == _QWEN35_MOE:
        # MoE tensor parallel is fixed at 1: dispatch and experts stay on the
        # full routed domain even under TP=2 (verified against the compiled
        # graph on transformers 5.13.x).
        assert env["Tmoe"] == env["T"]
        assert env["Fe"] == env["Fmoe"]
        assert env["Te"] == env["Tmoe"] * int(model_config["num_experts_per_tok"])
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS


@pytest.mark.nightly
@pytest.mark.parametrize(
    "model_name",
    (_QWEN35_DENSE, _QWEN35_MOE, _QWEN3_NEXT),
    ids=("dense", "moe", "next"),
)
def test_qwen35_mtp_decode_capture_organize_and_compare(model_name: str) -> None:
    """Validate the MTP region on every classification-4 model."""

    profile, artifact, result = _run_qwen35_case(
        model_name=model_name,
        phase=ExecutionPhase.DECODE,
        query_length=3,
        context_length=128,
        num_mtp_tokens=1,
    )

    assert result.spec_id == "qwen3_5_next_v1"
    mtp_stages = {finding.stage_id for finding in result.findings if finding.region_id == "mtp"}
    assert {
        "input_shift",
        "embedding",
        "input_fusion",
        "proposal_selection",
        "proposal_lm_head",
        "proposal_sampler",
    }.issubset(mtp_stages)
    # Six-layer override trims layer_types to [lin, lin, lin, full, lin, lin];
    # last_kind_from therefore repeats linear_attention for every MTP layer.
    if model_name == _QWEN3_NEXT:
        assert "linear_attention" in mtp_stages
        assert {"moe_gate", "moe_experts", "moe_combine", "shared_ffn"}.issubset(mtp_stages)
    else:
        assert {"linear_projection", "linear_delta_rule", "linear_output"}.issubset(mtp_stages)
        if model_name == _QWEN35_DENSE:
            assert "dense_ffn" in mtp_stages
        else:
            assert {"moe_gate", "moe_experts", "moe_combine", "shared_ffn"}.issubset(mtp_stages)
    assert_diagnostics_passed(result)
    assert result.summary.overall_status is FindingStatus.PASS
