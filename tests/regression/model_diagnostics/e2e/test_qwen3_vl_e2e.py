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
"""Offline Qwen3-VL Theory-to-Runtime diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, FindingStatus, ParallelContext
from tools.model_diagnostics.integrations import assert_diagnostics_passed
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
from tools.model_diagnostics.specification import DiagnosticsRunProfile

_DENSE_MODEL = "tests/assets/model_config/qwen3_vl_8b_instruct"
_MOE_MODEL = "tests/assets/model_config/qwen3_vl_moe_235b_a22b"


def _run(profile: DiagnosticsRunProfile):
    torch.compiler.reset()
    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)
    result = application.run_against_artifact(request, artifact)
    assert_diagnostics_passed(result)
    return artifact, result


def _assert_mtp_capture(profile: DiagnosticsRunProfile, *, layer_kind: str) -> None:
    artifact, result = _run(profile)
    spec = create_model_diagnostics_application().spec_provider.get(artifact.run_context)

    assert artifact.run_context.model_config["num_mtp_tokens"] == 1
    assert (
        sum(call.operator_name == "tensor_cast.shift_and_update_input_ids.default" for call in artifact.operator_calls)
        == 1
    )
    mtp_region = next(region for region in spec.regions if region.region_id == "mtp")
    assert mtp_region.layer_layout == (layer_kind,)
    assert result.summary.overall_status is FindingStatus.PASS
    mtp_findings = tuple(finding for finding in result.findings if finding.region_id == "mtp")
    assert mtp_findings
    assert all(finding.status is FindingStatus.PASS for finding in mtp_findings)


def test_qwen3_vl_dense_offline_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_DENSE_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=128,
        context_length=0,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=(0,),
        selected_region_layers={"vision_encoder": (0, 8, 16, 24, 26)},
        selected_stage_regions=("input", "vision_prelude", "vision_merger", "output"),
        num_hidden_layers_override=1,
        do_compile=True,
        device="TEST_DEVICE",
        quantize_linear_action="W8A8_DYNAMIC",
        word_embedding_tp=None,
        image_batch_size=1,
        image_height=224,
        image_width=224,
    )

    artifact, result = _run(profile)

    assert artifact.run_context.model_config["model_type"] == "qwen3_vl"
    assert result.spec_id == "qwen3_vl_v1"


def test_qwen3_vl_moe_offline_capture_and_compare() -> None:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=_MOE_MODEL,
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=4,
        query_length=16,
        context_length=200,
        num_mtp_tokens=0,
        parallel=ParallelContext(tensor_parallel_size=8, expert_parallel_size=8),
        selected_language_layers=(0,),
        selected_stage_regions=("input", "output"),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="W8A8_DYNAMIC",
        word_embedding_tp=None,
        image_batch_size=1,
        image_height=720,
        image_width=1080,
    )

    artifact, result = _run(profile)

    assert artifact.run_context.model_config["model_type"] == "qwen3_vl_moe"
    assert result.spec_id == "qwen3_vl_moe_v1"


def test_qwen3_vl_moe_mixed_dense_and_moe_layers_capture_and_compare(tmp_path: Path) -> None:
    source_config = Path(_MOE_MODEL) / "config.json"
    mixed_model = tmp_path / "qwen3_vl_moe_mixed"
    mixed_model.mkdir()
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    payload["text_config"]["decoder_sparse_step"] = 2
    payload["text_config"]["mlp_only_layers"] = []
    (mixed_model / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=str(mixed_model),
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=4,
        query_length=16,
        context_length=200,
        num_mtp_tokens=0,
        parallel=ParallelContext(tensor_parallel_size=8, expert_parallel_size=8),
        selected_language_layers=(0, 1),
        selected_stage_regions=("input", "output"),
        num_hidden_layers_override=2,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="W8A8_DYNAMIC",
        word_embedding_tp=None,
    )

    artifact, result = _run(profile)
    spec = create_model_diagnostics_application().spec_provider.get(artifact.run_context)
    language = next(region for region in spec.regions if region.region_id == "language")

    assert language.layer_layout == (
        "qwen3_vl_moe_dense_text_decoder",
        "qwen3_vl_moe_text_decoder",
    )
    assert result.spec_id == "qwen3_vl_moe_v1"


def test_qwen3_vl_dense_mtp_capture_and_compare() -> None:
    _assert_mtp_capture(
        DiagnosticsRunProfile(
            schema_version="1",
            model_name=_DENSE_MODEL,
            entrypoint="text_generate",
            phase=ExecutionPhase.DECODE,
            batch_size=1,
            query_length=2,
            context_length=128,
            num_mtp_tokens=1,
            parallel=ParallelContext(tensor_parallel_size=2),
            selected_language_layers=(0,),
            selected_stage_regions=(),
            num_hidden_layers_override=1,
            do_compile=True,
            device="TEST_DEVICE",
            quantize_linear_action="DISABLED",
            word_embedding_tp=None,
        ),
        layer_kind="qwen3_vl_dense_mtp",
    )


def test_qwen3_vl_moe_mtp_capture_and_compare() -> None:
    _assert_mtp_capture(
        DiagnosticsRunProfile(
            schema_version="1",
            model_name=_MOE_MODEL,
            entrypoint="text_generate",
            phase=ExecutionPhase.DECODE,
            batch_size=4,
            query_length=16,
            context_length=200,
            num_mtp_tokens=1,
            parallel=ParallelContext(tensor_parallel_size=8, expert_parallel_size=8),
            selected_language_layers=(0,),
            selected_stage_regions=(),
            num_hidden_layers_override=1,
            do_compile=False,
            device="TEST_DEVICE",
            quantize_linear_action="W8A8_DYNAMIC",
            word_embedding_tp=None,
        ),
        layer_kind="qwen3_vl_moe_mtp",
    )
