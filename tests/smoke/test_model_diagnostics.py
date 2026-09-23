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

"""Quick end-to-end guard for model diagnostics."""

import os
from pathlib import Path

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import ExecutionPhase, ParallelContext
from tools.model_diagnostics.specification import DiagnosticsRunProfile
from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile

_QWEN3_DENSE_CONFIG = Path(__file__).resolve().parents[1] / "assets" / "model_config" / "qwen3_dense_0_6b"


def test_model_diagnostics_capture_and_theory_resolution_smoke(monkeypatch) -> None:
    """Exercise eager capture and Theory request resolution on one small model."""

    if os.name == "nt":
        # The sandbox exposes synthetic POSIX mode bits for this reviewed asset.
        monkeypatch.setattr("tensor_cast.core.model_source_security._find_mount_type", lambda _path: "drvfs")

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name=str(_QWEN3_DENSE_CONFIG),
        entrypoint="text_generate",
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=1,
        context_length=16,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=None,
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )

    artifact = capture_artifact_for_profile(profile)
    application = create_model_diagnostics_application()
    spec = application.spec_provider.get(artifact.run_context)
    request = profile.to_request(context=artifact.run_context, spec=spec)

    assert artifact.operator_calls
    assert spec.spec_id == "qwen3_dense_v1"
    assert request.selected_layers == {"language": (0,)}
