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

"""Default composition root for the builtin Theory-to-Runtime application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import (
    DiagnosticsRequest,
    DiagnosticsResult,
    ModelDiagnosticsSpec,
    SimulationExecutionArtifact,
    SourceKind,
)
from tools.model_diagnostics.organization import (
    RuntimeArtifactOrganizer,
    TheoryExecutionOrganizationStrategy,
)
from tools.model_diagnostics.sources import SimulationArtifactSource, TheoryOperatorRecordSource
from tools.model_diagnostics.specification import (
    DiagnosticsRunProfile,
    LoadedSpecCatalogResolver,
    ResolvingSpecProvider,
    SpecificationLoadError,
    YamlModelDiagnosticsSpecLoader,
    create_builtin_operator_activation_registry,
    create_builtin_source_options_parsers,
    load_diagnostics_run_profile,
)
from tools.model_diagnostics.specification.theory_fragments import (
    load_builtin_theory_fragment_registry,
)

from .runner import ModelDiagnosticsRunner


@dataclass(frozen=True)
class ModelDiagnosticsApplication:
    """Ready-to-run builtin Theory versus Runtime diagnostics."""

    runner: ModelDiagnosticsRunner
    theory_source: TheoryOperatorRecordSource
    spec_provider: ResolvingSpecProvider

    def run_against_artifact(
        self,
        request: DiagnosticsRequest,
        artifact: SimulationExecutionArtifact,
        *,
        spec: ModelDiagnosticsSpec | None = None,
    ) -> DiagnosticsResult:
        """Compare Theory expectations with one in-memory Runtime Artifact."""

        return self.runner.run(
            request,
            self.theory_source,
            SimulationArtifactSource(artifact),
            spec=spec,
        )

    def run_profile_against_artifact(
        self,
        profile: DiagnosticsRunProfile,
        artifact: SimulationExecutionArtifact,
    ) -> DiagnosticsResult:
        """Resolve Theory and compare it with an already captured Runtime Artifact."""

        spec = self.spec_provider.get(artifact.run_context)
        request = profile.to_request(context=artifact.run_context, spec=spec)
        return self.run_against_artifact(request, artifact, spec=spec)

    def run_from_profile(self, profile_path: Path) -> DiagnosticsResult:
        """End-to-end: load run profile → capture Runtime → Theory↔Runtime compare."""

        from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile

        profile = load_diagnostics_run_profile(profile_path)
        artifact = capture_artifact_for_profile(profile)
        return self.run_profile_against_artifact(profile, artifact)


def create_model_diagnostics_application(
    *,
    specs_dir: Path | None = None,
) -> ModelDiagnosticsApplication:
    """Build one coherent application with a single shared comparison registry."""

    builtin_specs_dir = specs_dir or Path(__file__).resolve().parents[1] / "specs"
    registry = create_stage_comparison_registry()
    fragment_registry = load_builtin_theory_fragment_registry()
    source_options_parsers = create_builtin_source_options_parsers(
        fragment_registry=fragment_registry,
    )
    loader = YamlModelDiagnosticsSpecLoader(
        comparison_registry=registry,
        activation_registry=create_builtin_operator_activation_registry(),
        source_options_parsers=source_options_parsers,
        specs_dir=builtin_specs_dir,
        fragment_registry=fragment_registry,
    )
    spec_ids = tuple(path.stem for path in sorted(builtin_specs_dir.glob("*.yaml")))
    if not spec_ids:
        raise SpecificationLoadError(f"no model diagnostics specs found in {builtin_specs_dir}")
    documents = {spec_id: loader.load(spec_id) for spec_id in spec_ids}
    provider = ResolvingSpecProvider(
        resolver=LoadedSpecCatalogResolver(specs=tuple(doc.spec for doc in documents.values())),
        loader=loader,
        documents=documents,
    )
    runner = ModelDiagnosticsRunner(
        spec_provider=provider,
        organization_by_source={
            SourceKind.THEORY: TheoryExecutionOrganizationStrategy(),
            SourceKind.RUNTIME: RuntimeArtifactOrganizer(),
        },
        comparison_strategies=registry,
    )
    return ModelDiagnosticsApplication(
        runner=runner,
        theory_source=TheoryOperatorRecordSource(),
        spec_provider=provider,
    )
