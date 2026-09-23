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

from pathlib import Path

import pytest

from tools.model_diagnostics import (
    InvalidDiagnosticsRequest,
    ModelDiagnosticsApplication,
    SourceLoadError,
    create_model_diagnostics_application,
)
from tools.model_diagnostics.application import SourceLoadError as ApplicationSourceLoadError
from tools.model_diagnostics.organization import (
    RuntimeArtifactOrganizer,
    TheoryExecutionOrganizationStrategy,
)
from tools.model_diagnostics.sources import (
    SimulationArtifactSource,
    TheoryOperatorRecordSource,
)
from tools.model_diagnostics.specification import (
    SourceLoadError as SpecificationSourceLoadError,
    SpecificationLoadError,
)


def test_public_api_exports_both_sources_and_organizers() -> None:
    assert TheoryOperatorRecordSource.source_kind.value == "theory"
    assert SimulationArtifactSource.source_kind.value == "runtime"
    assert TheoryExecutionOrganizationStrategy.strategy_id == "theory_organization"
    assert RuntimeArtifactOrganizer.strategy_id == "runtime_artifact"


def test_all_layers_export_the_same_source_load_error() -> None:
    assert ApplicationSourceLoadError is SourceLoadError
    assert SpecificationSourceLoadError is SourceLoadError
    assert issubclass(InvalidDiagnosticsRequest, ValueError)


def test_composition_root_shares_one_registry_between_loader_and_runner() -> None:
    application = create_model_diagnostics_application()

    assert isinstance(application, ModelDiagnosticsApplication)
    runner = application.runner
    loader = runner._spec_provider.loader
    assert loader._comparison_registry is runner._comparison_strategies
    assert set(runner._organization_by_source) == {
        TheoryOperatorRecordSource.source_kind,
        SimulationArtifactSource.source_kind,
    }


def test_composition_root_rejects_an_empty_spec_catalog(tmp_path: Path) -> None:
    with pytest.raises(SpecificationLoadError, match="no model diagnostics specs"):
        create_model_diagnostics_application(specs_dir=tmp_path)


def test_application_runs_from_profile_path(monkeypatch, tmp_path: Path) -> None:
    profile = object()
    artifact = object()
    expected = object()
    application = ModelDiagnosticsApplication(
        runner=object(),
        theory_source=object(),
        spec_provider=object(),
    )
    monkeypatch.setattr(
        "tools.model_diagnostics.application.composition.load_diagnostics_run_profile",
        lambda path: profile,
    )
    monkeypatch.setattr(
        "tools.model_diagnostics.sources.runtime_capture.capture_artifact_for_profile",
        lambda loaded_profile: artifact,
    )
    monkeypatch.setattr(
        ModelDiagnosticsApplication,
        "run_profile_against_artifact",
        lambda self, loaded_profile, captured_artifact: expected,
    )

    assert application.run_from_profile(tmp_path / "profile.yaml") is expected


def test_runtime_source_description_preserves_artifact_provenance(
    tmp_path: Path,
) -> None:
    from tools.model_diagnostics.domain import (
        ExecutionPhase,
        ModelRunContext,
        ParallelContext,
        ProducerInfo,
        SimulationExecutionArtifact,
    )

    artifact = SimulationExecutionArtifact(
        schema_version="1",
        producer=ProducerInfo(
            package_version="0.1.0",
            git_revision=None,
            capture_backend="synthetic",
        ),
        run_context=ModelRunContext(
            model_name="synthetic/model",
            entrypoint="test",
            phase=ExecutionPhase.PREFILL,
            batch_size=1,
            query_length=1,
            context_length=1,
            parallel=ParallelContext(),
            model_config={},
            quantization_config={},
        ),
        operator_calls=(),
    )
    in_memory = SimulationArtifactSource(artifact).describe()

    assert in_memory.producer == artifact.producer
    assert in_memory.artifact_reference is None
