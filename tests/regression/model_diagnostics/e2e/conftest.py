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
"""Fixtures for real-model diagnostics end-to-end cases."""

from __future__ import annotations

import math

import pytest
import torch

from tools.model_diagnostics.domain import ExecutionPhase, ParallelContext
from tools.model_diagnostics.specification import DiagnosticsRunProfile
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile


def assert_parallel_contract(env: dict[str, object], parallel: ParallelContext, *, raw_logits_gate: bool) -> None:
    """Validate rank-local heads and MoE token domains for a parallel layout."""

    assert env["Lh"] == env["Nh"] // parallel.tensor_parallel_size
    if parallel.expert_parallel_size > 1:
        assert env["Tmoe"] == math.ceil(env["T"] / parallel.tensor_parallel_size)
    else:
        assert env["Tmoe"] == env["T"] * parallel.data_parallel_size
    if raw_logits_gate and parallel.expert_parallel_size > 1:
        assert env["MOE_GATE_TOKENS"] == env["T"]
    else:
        assert env["MOE_GATE_TOKENS"] == env["Tmoe"]


_QWEN3_DENSE_MODELS = (
    "tests/assets/model_config/qwen3_dense_0_6b",
    "tests/assets/model_config/qwen3_dense_1_7b",
    "tests/assets/model_config/qwen3_dense_4b",
    "tests/assets/model_config/qwen3_dense_8b",
    "tests/assets/model_config/qwen3_dense_14b",
    "tests/assets/model_config/qwen3_dense_32b",
)
_REQUIRED_CASES = (
    (ExecutionPhase.PREFILL, "DISABLED", 2, None),
    (ExecutionPhase.DECODE, "DISABLED", 1, 128),
    (ExecutionPhase.PREFILL, "W8A8_DYNAMIC", 2, None),
)
_ADDITIONAL_CASES = (
    ("tests/assets/model_config/qwen3_dense_8b", ExecutionPhase.PREFILL, "W8A8_STATIC", 2, None),
    ("tests/assets/model_config/qwen3_dense_8b", ExecutionPhase.DECODE, "W8A8_DYNAMIC", 1, 128),
)
_QWEN3_DENSE_CASES = (
    tuple(
        (model_name, phase, quantization, query_length, context_length)
        for model_name in _QWEN3_DENSE_MODELS
        for phase, quantization, query_length, context_length in _REQUIRED_CASES
    )
    + _ADDITIONAL_CASES
)
_QWEN3_DENSE_MTP_CASES = (
    ("tests/assets/model_config/qwen3_dense_0_6b", "DISABLED"),
    ("tests/assets/model_config/qwen3_dense_1_7b", "W8A8_DYNAMIC"),
    ("tests/assets/model_config/qwen3_dense_4b", "DISABLED"),
    ("tests/assets/model_config/qwen3_dense_8b", "W8A8_DYNAMIC"),
    ("tests/assets/model_config/qwen3_dense_14b", "DISABLED"),
    ("tests/assets/model_config/qwen3_dense_32b", "W8A8_DYNAMIC"),
)


def _case_id(case: tuple[str, ExecutionPhase, str, int, int | None]) -> str:
    model_name, phase, quantization, _, _ = case
    model_size = model_name.rsplit("/", maxsplit=1)[-1].lower()
    return f"{model_size}_{phase.value}_{quantization.lower()}"


def _mtp_case_id(case: tuple[str, str]) -> str:
    model_name, quantization = case
    model_size = model_name.rsplit("/", maxsplit=1)[-1].lower()
    return f"{model_size}_mtp_{quantization.lower()}"


def _profile(
    *,
    model_name: str,
    phase: ExecutionPhase,
    quantization: str,
    query_length: int,
    context_length: int | None,
    num_mtp_tokens: int = 0,
) -> DiagnosticsRunProfile:
    """Build one complete E2E input without depending on example YAML defaults."""

    return DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
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


@pytest.fixture(
    scope="module",
    params=_QWEN3_DENSE_CASES,
    ids=tuple(_case_id(case) for case in _QWEN3_DENSE_CASES),
)
def qwen3_dense_case(request):
    """Capture required prefill, decode and quantized Qwen3 Dense variants."""

    model_name, phase, quantization, query_length, context_length = request.param
    profile = _profile(
        model_name=model_name,
        phase=phase,
        quantization=quantization,
        query_length=query_length,
        context_length=context_length,
    )
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    return profile, artifact


@pytest.fixture(
    scope="module",
    params=_QWEN3_DENSE_MTP_CASES,
    ids=tuple(_mtp_case_id(case) for case in _QWEN3_DENSE_MTP_CASES),
)
def qwen3_dense_mtp_case(request):
    """Capture every Qwen3 Dense size with two MTP tokens."""

    model_name, quantization = request.param
    profile = _profile(
        model_name=model_name,
        phase=ExecutionPhase.DECODE,
        quantization=quantization,
        query_length=3,
        context_length=128,
        num_mtp_tokens=2,
    )
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    return profile, artifact
