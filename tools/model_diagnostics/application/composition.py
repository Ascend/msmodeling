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
"""Default composition root for the builtin Theory-to-Runtime application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.domain import (
    DiagnosticsRequest,
    DiagnosticsResult,
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
    ) -> DiagnosticsResult:
        """Compare Theory expectations with one in-memory Runtime Artifact."""

        return self.runner.run(
            request,
            self.theory_source,
            SimulationArtifactSource(artifact),
        )

    def run_profile_against_artifact(
        self,
        profile: DiagnosticsRunProfile,
        artifact: SimulationExecutionArtifact,
    ) -> DiagnosticsResult:
        """Resolve Theory and compare it with an already captured Runtime Artifact."""

        spec = self.spec_provider.get(artifact.run_context)
        request = profile.to_request(context=artifact.run_context, spec=spec)
        return self.run_against_artifact(request, artifact)

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
