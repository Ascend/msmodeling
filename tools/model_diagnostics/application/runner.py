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
"""Fixed orchestration for one pairwise model diagnostic."""

from __future__ import annotations

from collections.abc import Mapping

from tools.model_diagnostics.comparison import (
    StageComparisonRegistry,
    StageComparisonRequest,
)
from tools.model_diagnostics.comparison.operator_policy import resolve_operator_aliases
from tools.model_diagnostics.domain import (
    ComparisonSpec,
    DiagnosticsRequest,
    DiagnosticsResult,
    EvidenceRef,
    ExecutionOrganizationRequest,
    Finding,
    FindingStatus,
    Limitation,
    ModelDiagnosticsSpec,
    ModelRunContext,
    ModelRunContextSummary,
    OneToOneOptions,
    RegionExecutionRecord,
    RuntimeStageOptions,
    SourceDescription,
    SourceKind,
    StageExecutionRecord,
    StageSpec,
    TensorMapping,
    TensorMappingMode,
    summarize_findings,
)
from tools.model_diagnostics.interfaces import (
    ExecutionOrganizationStrategy,
    ModelDiagnosticsSpecProvider,
    OperatorRecordSource,
)
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest, SourceLoadError

_RESULT_SCHEMA_VERSION = "1"
_DEFAULT_COMPARISON = ComparisonSpec(
    "one_to_one",
    OneToOneOptions(TensorMapping(mode=TensorMappingMode.POSITIONAL)),
)


class ModelDiagnosticsRunner:
    def __init__(
        self,
        spec_provider: ModelDiagnosticsSpecProvider,
        organization_by_source: Mapping[SourceKind, ExecutionOrganizationStrategy],
        comparison_strategies: StageComparisonRegistry,
    ) -> None:
        self._spec_provider = spec_provider
        self._organization_by_source = dict(organization_by_source)
        self._comparison_strategies = comparison_strategies

    def run(
        self,
        request: DiagnosticsRequest,
        left_source: OperatorRecordSource,
        right_source: OperatorRecordSource,
    ) -> DiagnosticsResult:
        if left_source.source_kind is right_source.source_kind:
            raise InvalidDiagnosticsRequest("diagnostics requires two distinct source kinds")
        left_description = _describe_source(left_source)
        right_description = _describe_source(right_source)
        spec = self._spec_provider.get(request.context)
        _validate_request_against_spec(request, spec.regions)
        left_regions = self._load_and_organize(request, spec, left_source)
        right_regions = self._load_and_organize(request, spec, right_source)
        findings = self._compare_selected(
            request,
            spec,
            left_source.source_kind,
            right_source.source_kind,
            left_regions,
            right_regions,
        )
        context = request.context
        return DiagnosticsResult(
            schema_version=_RESULT_SCHEMA_VERSION,
            spec_id=spec.spec_id,
            spec_version=spec.spec_version,
            context=ModelRunContextSummary(
                model_name=context.model_name,
                entrypoint=context.entrypoint,
                phase=context.phase,
                batch_size=context.batch_size,
                query_length=context.query_length,
                context_length=context.context_length,
                tensor_parallel_size=context.parallel.tensor_parallel_size,
            ),
            left_source=left_description,
            right_source=right_description,
            selected_layers=request.selected_layers,
            selected_stage_regions=request.selected_stage_regions,
            findings=findings,
            summary=summarize_findings(findings),
            limitations=_known_limitations(context, spec),
        )

    def _load_and_organize(self, request, spec, source) -> tuple[RegionExecutionRecord, ...]:
        execution = source.load_execution(
            request.context,
            spec,
            request.selected_layers,
            request.selected_stage_regions,
        )
        if execution.source_kind is not source.source_kind:
            raise SourceLoadError("source returned an execution with the wrong source_kind")
        if execution.run_context != request.context:
            raise SourceLoadError("source execution context does not match the diagnostics request")
        try:
            organizer = self._organization_by_source[source.source_kind]
        except KeyError as error:
            raise SourceLoadError(f"no organizer registered for source {source.source_kind.value!r}") from error
        return organizer.execute(
            ExecutionOrganizationRequest(
                execution=execution,
                spec=spec,
                selected_layers=request.selected_layers,
                selected_stage_regions=request.selected_stage_regions,
            )
        )

    def _compare_selected(
        self,
        request,
        spec: ModelDiagnosticsSpec,
        left_kind,
        right_kind,
        left_regions,
        right_regions,
    ) -> tuple[Finding, ...]:
        left_index = {region.region_id: region for region in left_regions}
        right_index = {region.region_id: region for region in right_regions}
        findings: list[Finding] = []
        for region_spec in spec.regions:
            region_id = region_spec.region_id
            if region_id in request.selected_stage_regions:
                for stage_spec in region_spec.stages:
                    findings.extend(
                        self._compare_stage(
                            region_id,
                            None,
                            stage_spec,
                            spec,
                            left_kind,
                            right_kind,
                            _region_stage(left_index.get(region_id), stage_spec.stage_id),
                            _region_stage(right_index.get(region_id), stage_spec.stage_id),
                        )
                    )
            for layer_index in request.selected_layers.get(region_id, ()):
                if layer_index < 0 or layer_index >= len(region_spec.layer_layout):
                    raise AssertionError("request layer validation drifted after Source loading")
                layer_kind = region_spec.layer_layout[layer_index]
                for stage_spec in region_spec.layer_specs[layer_kind].stages:
                    findings.extend(
                        self._compare_stage(
                            region_id,
                            layer_index,
                            stage_spec,
                            spec,
                            left_kind,
                            right_kind,
                            _layer_stage(left_index.get(region_id), layer_index, stage_spec.stage_id),
                            _layer_stage(right_index.get(region_id), layer_index, stage_spec.stage_id),
                        )
                    )
        return tuple(findings)

    def _compare_stage(
        self,
        region_id: str,
        layer_index: int | None,
        stage_spec: StageSpec,
        spec: ModelDiagnosticsSpec,
        left_kind: SourceKind,
        right_kind: SourceKind,
        left_stage: StageExecutionRecord | None,
        right_stage: StageExecutionRecord | None,
    ) -> tuple[Finding, ...]:
        if left_stage is None or right_stage is None:
            return (
                Finding(
                    region_id=region_id,
                    layer_index=layer_index,
                    stage_id=stage_spec.stage_id,
                    rule_id="stage_presence",
                    comparison_kind="stage_presence",
                    status=FindingStatus.INCOMPLETE,
                    message_code="stage.missing",
                    message="required organized stage is missing",
                    left_evidence=_stage_evidence(left_kind, left_stage),
                    right_evidence=_stage_evidence(right_kind, right_stage),
                ),
            )
        comparison = _comparison_for(stage_spec, left_kind, right_kind)
        strategy = self._comparison_strategies.resolve(comparison.strategy_id)
        operator_aliases = resolve_operator_aliases(spec.operator_aliases)
        return strategy.execute(
            StageComparisonRequest(
                region_id=region_id,
                layer_index=layer_index,
                left_stage=left_stage,
                right_stage=right_stage,
                comparison=comparison,
                operator_aliases=operator_aliases,
            )
        )


def _known_limitations(
    context: ModelRunContext,
    spec: ModelDiagnosticsSpec,
) -> tuple[Limitation, ...]:
    """Surface known, approved diagnostic limitations instead of silently passing.

    These are structural facts about the current comparison/organization
    behavior (not a Spec-level SKIP DSL): they make explicit what is *not*
    exhaustively checked, so PASS never hides a silently narrowed comparison.
    """

    limitations: list[Limitation] = []
    declared_dtype = context.model_config.get("declared_torch_dtype")
    if isinstance(declared_dtype, str) and declared_dtype in {"bfloat16", "bf16"}:
        limitations.append(
            Limitation(
                code="theory.dtype.runtime_fp16_binding",
                message=(
                    f"model_config.declared_torch_dtype={declared_dtype!r} is bound to "
                    "the Runtime-executed float16 dtype; Theory tensor dtype expectations "
                    "compare against float16, not the HF-declared dtype."
                ),
            )
        )
    ignored_operators = _configured_ignored_operators(spec)
    if ignored_operators:
        limitations.append(
            Limitation(
                code="runtime.mechanical_ops_ignored",
                message=(
                    "Runtime stage-boundary scanning ignores configured mechanical "
                    "operators that carry no independent Theory expectation: " + ", ".join(ignored_operators)
                ),
            )
        )
    return tuple(limitations)


def _configured_ignored_operators(spec: ModelDiagnosticsSpec) -> tuple[str, ...]:
    ignored: set[str] = set()
    for region in spec.regions:
        for stage in region.stages:
            _collect_ignored_operators(stage, ignored)
        for layer_spec in region.layer_specs.values():
            for stage in layer_spec.stages:
                _collect_ignored_operators(stage, ignored)
    return tuple(sorted(ignored))


def _collect_ignored_operators(stage: StageSpec, ignored: set[str]) -> None:
    options = stage.source_options.get(SourceKind.RUNTIME)
    if isinstance(options, RuntimeStageOptions):
        ignored.update(options.ignored_operators)


def _comparison_for(
    stage_spec: StageSpec,
    left_kind: SourceKind,
    right_kind: SourceKind,
) -> ComparisonSpec:
    return stage_spec.comparisons.get(
        (left_kind, right_kind),
        _DEFAULT_COMPARISON,
    )


def _describe_source(source: OperatorRecordSource) -> SourceDescription:
    description = source.describe()
    if description.source_kind is not source.source_kind:
        raise SourceLoadError("source description has the wrong source_kind")
    return description


def _validate_request_against_spec(
    request: DiagnosticsRequest,
    region_specs,
) -> None:
    regions = {region.region_id: region for region in region_specs}
    requested = set(request.selected_layers).union(request.selected_stage_regions)
    unknown = requested.difference(regions)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidDiagnosticsRequest(f"unknown selected region id(s): {names}")

    for region_id in request.selected_stage_regions:
        if not regions[region_id].stages:
            raise InvalidDiagnosticsRequest(f"selected stage region {region_id!r} has no region-level stages")

    for region_id, layer_indices in request.selected_layers.items():
        region = regions[region_id]
        if not region.layer_layout:
            raise InvalidDiagnosticsRequest(f"selected layer region {region_id!r} has no layer layout")
        out_of_range = tuple(
            index for index in layer_indices if index < 0 or index >= len(region.layer_layout)
        )
        if out_of_range:
            raise InvalidDiagnosticsRequest(f"selected layer {out_of_range[0]} exceeds region {region_id!r}")


def _region_stage(
    region: RegionExecutionRecord | None,
    stage_id: str,
) -> StageExecutionRecord | None:
    if region is None:
        return None
    return next((stage for stage in region.stages if stage.stage_id == stage_id), None)


def _layer_stage(
    region: RegionExecutionRecord | None,
    layer_index: int,
    stage_id: str,
) -> StageExecutionRecord | None:
    if region is None:
        return None
    layer = next((candidate for candidate in region.layers if candidate.layer_index == layer_index), None)
    if layer is None:
        return None
    return next((stage for stage in layer.stages if stage.stage_id == stage_id), None)


def _stage_evidence(
    source_kind: SourceKind,
    stage: StageExecutionRecord | None,
) -> tuple[EvidenceRef, ...]:
    if stage is None:
        return ()
    return tuple(
        EvidenceRef(
            source_kind=source_kind,
            call_index=call.call_index,
            stage_call_position=position,
            operator_name=call.operator_name,
            source_reference=call.source_reference,
        )
        for position, call in enumerate(stage.operator_calls)
    )
