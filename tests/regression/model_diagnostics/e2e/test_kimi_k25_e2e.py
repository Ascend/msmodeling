# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""Offline Kimi-K2.5 Theory-to-Runtime diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile

_MODEL = "tests/assets/model_config/kimi_k2_5"


def test_kimi_k25_padding_metadata_uses_int32_cumulative_lengths() -> None:
    source = Path(_MODEL, "modeling_deepseek.py").read_text(encoding="utf-8")
    helper = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_get_unpad_data"
    )
    dtype_attributes = [node for node in ast.walk(helper) if isinstance(node, ast.Attribute) and node.attr == "int32"]

    assert dtype_attributes
    assert all(isinstance(attribute.value, ast.Name) for attribute in dtype_attributes)
    assert all(attribute.value.id == "torch" for attribute in dtype_attributes)


def test_kimi_k25_offline_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=30,
        context_length=0,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=(0, 1),
        selected_stage_regions=("input", "vision_prelude", "vision_postlude", "output"),
        num_hidden_layers_override=2,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        image_batch_size=1,
        image_height=1080,
        image_width=1920,
        selected_region_layers={"vision_encoder": (0, 26)},
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert artifact.run_context.model_config["model_type"] == "kimi_k25"
    assert artifact.run_context.model_config["vision_num_hidden_layers"] == 27
    assert result.spec_id == "kimi_k25_v1"
    assert_diagnostics_passed(result)


def test_kimi_k25_mtp_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
        parallel=ParallelContext(tensor_parallel_size=2, expert_parallel_size=2),
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
    assert mtp_region.layer_layout == ("kimi_k25_mtp",)
    predictor_stages = {stage.stage_id for stage in mtp_region.layer_specs["kimi_k25_mtp"].stages}
    assert {
        "input_shift",
        "embedding",
        "input_fusion",
        "mla_preprocess",
        "mla_attention",
        "moe_gate",
        "moe_dispatch",
        "moe_experts",
        "moe_combine",
        "shared_expert",
        "proposal_selection",
        "proposal_lm_head",
        "proposal_sampler",
    }.issubset(predictor_stages)

    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
    mtp_findings = tuple(finding for finding in result.findings if finding.region_id == "mtp")
    assert mtp_findings
    assert all(finding.status is FindingStatus.PASS for finding in mtp_findings)


def test_kimi_k25_packed_batch_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=2,
        query_length=30,
        context_length=0,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=(0,),
        selected_stage_regions=("input", "output"),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        image_batch_size=1,
        image_height=1080,
        image_width=1920,
        selected_region_layers={"vision_encoder": (0, 26)},
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    result = application.run_against_artifact(
        profile.to_request(context=artifact.run_context, spec=spec),
        artifact,
    )

    embedding = next(call for call in artifact.operator_calls if call.operator_name == "aten.embedding.default")
    assert embedding.tensors[-1].shape == (1, 5446, 7168)
    assert_diagnostics_passed(result)
