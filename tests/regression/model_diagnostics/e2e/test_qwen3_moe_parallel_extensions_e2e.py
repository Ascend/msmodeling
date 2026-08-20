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
"""Qwen3 MoE E2E for MoE-DP / redundant-expert profile switches."""

from __future__ import annotations

import pytest
import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext, SourceKind
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

_MODEL = "tests/assets/model_config/qwen3_moe_30b_a3b"


def _profile(
    *,
    parallel: ParallelContext,
    batch_size: int = 1,
    query_length: int = 2,
    enable_redundant_experts: bool = False,
    enable_external_shared_experts: bool = False,
) -> DiagnosticsRunProfile:
    return DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=batch_size,
        query_length=query_length,
        context_length=None,
        num_mtp_tokens=0,
        parallel=parallel,
        selected_language_layers=None,
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        enable_redundant_experts=enable_redundant_experts,
        enable_external_shared_experts=enable_external_shared_experts,
    )


def test_qwen3_moe_e2e_with_moe_dp_size() -> None:
    """MDP=2 with matching DP=2 keeps fused MoE Spec path green."""

    profile = _profile(
        parallel=ParallelContext(
            data_parallel_size=2,
            moe_data_parallel_size=2,
            expert_parallel_size=1,
        )
    )
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    assert artifact.run_context.parallel.moe_data_parallel_size == 2
    assert artifact.run_context.model_config.get("moe_dp_size") == 2

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    result = application.run_against_artifact(
        profile.to_request(context=artifact.run_context, spec=spec),
        artifact,
    )
    assert_diagnostics_passed(result)
    assert result.left_source.source_kind is SourceKind.THEORY
    assert result.right_source.source_kind is SourceKind.RUNTIME


def test_qwen3_moe_e2e_with_enable_redundant_experts() -> None:
    """EP=2 + enable_redundant_experts: Theory Te follows global experts; Spec still PASS."""

    profile = _profile(
        parallel=ParallelContext(
            data_parallel_size=2,
            moe_data_parallel_size=1,
            expert_parallel_size=2,
        ),
        enable_redundant_experts=True,
    )
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    assert artifact.run_context.model_config.get("enable_redundant_experts") is True

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    result = application.run_against_artifact(
        profile.to_request(context=artifact.run_context, spec=spec),
        artifact,
    )
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)


def test_qwen3_moe_e2e_with_ep_dp_token_transform() -> None:
    """EP=2, DP=1 exercises Runtime pad/slice while Theory stays aligned."""

    profile = _profile(
        parallel=ParallelContext(
            tensor_parallel_size=2,
            data_parallel_size=1,
            expert_parallel_size=2,
            moe_data_parallel_size=1,
        ),
        query_length=3,
    )
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    env = build_theory_env(artifact.run_context)
    assert env["T"] == 3
    assert env["Tmoe"] == 2
    assert env["Te"] == 32

    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    result = application.run_against_artifact(
        profile.to_request(context=artifact.run_context, spec=spec),
        artifact,
    )
    assert_diagnostics_passed(result)


def test_qwen3_moe_rejects_external_shared_without_shared_experts() -> None:
    """Qwen3 MoE has no shared experts; Runtime must fail fast (not silent PASS)."""

    profile = _profile(
        parallel=ParallelContext(
            data_parallel_size=2,
            expert_parallel_size=2,
        ),
        enable_external_shared_experts=True,
    )
    torch.compiler.reset()
    with pytest.raises(AssertionError):
        capture_artifact_for_profile(profile)
