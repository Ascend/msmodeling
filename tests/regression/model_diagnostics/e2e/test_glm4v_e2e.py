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

"""Offline GLM-4V Theory-to-Runtime diagnostics."""

from __future__ import annotations

import pytest
import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile


@pytest.mark.parametrize(
    (
        "model_name",
        "expected_model_type",
        "expected_spec_id",
        "selected_layer",
        "layer_count",
        "image_height",
        "image_width",
        "do_compile",
    ),
    (
        pytest.param(
            "tests/assets/model_config/glm4v_9b_thinking",
            "glm4v",
            "glm4v_v1",
            0,
            1,
            224,
            224,
            False,
            id="dense",
        ),
        pytest.param(
            "tests/assets/model_config/glm4v_moe_4_5v",
            "glm4v_moe",
            "glm4v_moe_v1",
            1,
            2,
            1080,
            1920,
            True,
            id="moe",
        ),
    ),
)
def test_glm4v_offline_capture_and_compare(
    model_name: str,
    expected_model_type: str,
    expected_spec_id: str,
    selected_layer: int,
    layer_count: int,
    image_height: int,
    image_width: int,
    do_compile: bool,
) -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=30,
        context_length=0,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=(selected_layer,),
        selected_region_layers={"vision_encoder": (0, 12, 23)},
        selected_stage_regions=("input", "vision_prelude", "vision_merger", "output"),
        num_hidden_layers_override=layer_count,
        do_compile=do_compile,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        image_batch_size=1,
        image_height=image_height,
        image_width=image_width,
    )

    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)

    assert artifact.run_context.model_config["model_type"] == expected_model_type
    assert result.spec_id == expected_spec_id
    assert_diagnostics_passed(result)


@pytest.mark.parametrize(
    (
        "model_name",
        "selected_layer",
        "layer_count",
        "do_compile",
        "mtp_layer_kind",
        "parallel",
    ),
    (
        pytest.param(
            "tests/assets/model_config/glm4v_9b_thinking",
            0,
            1,
            False,
            "glm4v_dense_mtp",
            ParallelContext(tensor_parallel_size=2),
            id="dense-mtp",
        ),
        pytest.param(
            "tests/assets/model_config/glm4v_moe_4_5v",
            1,
            2,
            True,
            "glm4v_moe_mtp",
            ParallelContext(tensor_parallel_size=2, expert_parallel_size=2),
            id="moe-mtp",
        ),
    ),
)
def test_glm4v_mtp_capture_and_compare(
    model_name: str,
    selected_layer: int,
    layer_count: int,
    do_compile: bool,
    mtp_layer_kind: str,
    parallel: ParallelContext,
) -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=model_name,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=2,
        context_length=128,
        num_mtp_tokens=1,
        parallel=parallel,
        selected_language_layers=(selected_layer,),
        selected_stage_regions=(),
        num_hidden_layers_override=layer_count,
        do_compile=do_compile,
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
    assert mtp_region.layer_layout == (mtp_layer_kind,)
    assert result.summary.overall_status is FindingStatus.PASS
    assert_diagnostics_passed(result)
    mtp_findings = tuple(finding for finding in result.findings if finding.region_id == "mtp")
    assert mtp_findings
    assert all(finding.status is FindingStatus.PASS for finding in mtp_findings)
