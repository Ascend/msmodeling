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
"""Real Qwen3 MoE model diagnostics end-to-end cases."""

from __future__ import annotations

import pytest
import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext, SourceKind
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile

_QUANT_LINEAR = "tensor_cast.static_quant_linear.default"
_GMM_SWIGLU = "tensor_cast.grouped_matmul_swiglu.default"
_GMM_QUANT_SWIGLU = "tensor_cast.grouped_matmul_quant_swiglu.default"
_GMM = "tensor_cast.grouped_matmul.default"
_GMM_QUANT = "tensor_cast.grouped_matmul_quant.default"
_MODELS = ("tests/assets/model_config/qwen3_moe_30b_a3b", "tests/assets/model_config/qwen3_moe_235b_a22b")
_ROUTINE_CASES = (
    (_MODELS[0], ExecutionPhase.PREFILL, "DISABLED", 2, None),
    (_MODELS[0], ExecutionPhase.DECODE, "DISABLED", 1, 128),
    (_MODELS[0], ExecutionPhase.PREFILL, "W8A8_DYNAMIC", 2, None),
)
_FULL_CASES = tuple(
    (model, phase, quantization, query_length, context_length)
    for model in _MODELS
    for phase, quantization, query_length, context_length in (
        (ExecutionPhase.PREFILL, "DISABLED", 2, None),
        (ExecutionPhase.DECODE, "DISABLED", 1, 128),
        (ExecutionPhase.PREFILL, "W8A8_DYNAMIC", 2, None),
    )
)
_NIGHTLY_CASES = tuple(case for case in _FULL_CASES if case not in _ROUTINE_CASES)
_ROUTINE_MTP_CASES = ((_MODELS[0], "DISABLED"), (_MODELS[0], "W8A8_DYNAMIC"))
_FULL_MTP_CASES = tuple((model, quantization) for model in _MODELS for quantization in ("DISABLED", "W8A8_DYNAMIC"))
_NIGHTLY_MTP_CASES = tuple(case for case in _FULL_MTP_CASES if case not in _ROUTINE_MTP_CASES)


def _case_id(case) -> str:
    model, phase, quantization, _, _ = case
    size = model.rsplit("/", maxsplit=1)[-1].removeprefix("Qwen3-").lower().replace("-", "_")
    return f"{size}_{phase.value}_{quantization.lower()}"


def _mtp_case_id(case) -> str:
    model, quantization = case
    size = model.rsplit("/", maxsplit=1)[-1].removeprefix("Qwen3-").lower().replace("-", "_")
    return f"{size}_mtp_{quantization.lower()}"


def _profile(case, *, num_mtp_tokens: int = 0) -> DiagnosticsRunProfile:
    model, phase, quantization, query_length, context_length = case
    return DiagnosticsRunProfile(
        schema_version="1",
        model_name=model,
        entrypoint="text_generate",
        phase=phase,
        batch_size=1,
        query_length=query_length,
        context_length=context_length,
        num_mtp_tokens=num_mtp_tokens,
        parallel=ParallelContext(),
        selected_language_layers=None,
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action=quantization,
        word_embedding_tp=None,
    )


def _capture(case, *, num_mtp_tokens: int = 0):
    profile = _profile(case, num_mtp_tokens=num_mtp_tokens)
    torch.compiler.reset()
    return profile, capture_artifact_for_profile(profile)


def _assert_quantization_context(profile, artifact) -> None:
    assert artifact.run_context.quantization_config["action"] == profile.quantize_linear_action
    if profile.quantize_linear_action == "DISABLED":
        assert artifact.run_context.quantization_config["enabled"] is False
    else:
        assert artifact.run_context.quantization_config["enabled"] is True
        assert artifact.run_context.quantization_config["linear_input_dtype"] == "int8"


def _assert_moe_runtime_kernels(profile, artifact) -> None:
    operator_names = tuple(call.operator_name for call in artifact.operator_calls)
    if profile.quantize_linear_action == "DISABLED":
        assert _QUANT_LINEAR not in operator_names
        assert _GMM_QUANT_SWIGLU not in operator_names
        assert _GMM_QUANT not in operator_names
        assert _GMM_SWIGLU in operator_names
        assert _GMM in operator_names
    else:
        assert _QUANT_LINEAR in operator_names
        assert _GMM_QUANT_SWIGLU in operator_names
        assert _GMM_QUANT in operator_names
        assert _GMM_SWIGLU not in operator_names
        assert _GMM not in operator_names


def _assert_qwen3_moe_case(case) -> None:
    """Validate every public Qwen3 MoE size through capture and comparison."""

    profile, artifact = _capture(case)
    assert artifact.run_context.model_config.get("model_type") == "qwen3_moe"
    assert artifact.run_context.parallel.expert_parallel_size == 1
    _assert_quantization_context(profile, artifact)
    _assert_moe_runtime_kernels(profile, artifact)

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)
    assert result.spec_id == "qwen3_moe_v1"
    assert result.left_source.source_kind is SourceKind.THEORY
    assert result.right_source.source_kind is SourceKind.RUNTIME
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
    expected_stages = {
        "embedding",
        "attention_qkv",
        "attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "lm_head",
    }
    assert {finding.stage_id for finding in result.findings} >= expected_stages


@pytest.mark.parametrize("case", _ROUTINE_CASES, ids=tuple(_case_id(case) for case in _ROUTINE_CASES))
def test_qwen3_moe_capture_organize_and_compare(case) -> None:
    """Keep representative prefill, decode and quantized MoE E2E in routine gates."""

    _assert_qwen3_moe_case(case)


@pytest.mark.nightly
@pytest.mark.parametrize("case", _NIGHTLY_CASES, ids=tuple(_case_id(case) for case in _NIGHTLY_CASES))
def test_qwen3_moe_full_model_matrix(case) -> None:
    """Validate the remaining public Qwen3 MoE model matrix nightly."""

    _assert_qwen3_moe_case(case)


def _assert_qwen3_moe_mtp_case(case) -> None:
    """Validate Qwen3 MoE MTP decode, including quantized MTP combinations."""

    model, quantization = case
    profile, artifact = _capture(
        (model, ExecutionPhase.DECODE, quantization, 3, 128),
        num_mtp_tokens=2,
    )
    assert profile.phase.value == "decode"
    assert profile.num_mtp_tokens == 2
    assert artifact.run_context.query_length == 3
    assert artifact.run_context.model_config.get("num_mtp_tokens") == 2
    _assert_quantization_context(profile, artifact)
    _assert_moe_runtime_kernels(profile, artifact)
    operator_names = [call.operator_name for call in artifact.operator_calls]
    assert operator_names.count("tensor_cast.shift_and_update_input_ids.default") == 2

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)
    assert result.spec_id == "qwen3_moe_v1"
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)


@pytest.mark.parametrize("case", _ROUTINE_MTP_CASES, ids=tuple(_mtp_case_id(case) for case in _ROUTINE_MTP_CASES))
def test_qwen3_moe_mtp_capture_organize_and_compare(case) -> None:
    """Keep representative unquantized and quantized MTP E2E in routine gates."""

    _assert_qwen3_moe_mtp_case(case)


@pytest.mark.nightly
@pytest.mark.parametrize("case", _NIGHTLY_MTP_CASES, ids=tuple(_mtp_case_id(case) for case in _NIGHTLY_MTP_CASES))
def test_qwen3_moe_full_mtp_matrix(case) -> None:
    """Validate the remaining public Qwen3 MoE MTP matrix nightly."""

    _assert_qwen3_moe_mtp_case(case)
