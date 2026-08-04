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
"""Sequential Runtime Artifact organization driven only by a model specification."""

from __future__ import annotations

from dataclasses import dataclass

from tools.model_diagnostics.domain import (
    ExecutionOrganizationRequest,
    LayerExecutionRecord,
    RegionExecutionRecord,
    RuntimeStageOptions,
    SourceKind,
    StageExecutionRecord,
    StageSpec,
)
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest, SourceLoadError
from tools.model_diagnostics.specification.errors import SpecificationLoadError


@dataclass(frozen=True)
class _ExpectedStage:
    region_id: str
    layer_index: int | None
    layer_kind: str | None
    stage: StageSpec
    emit: bool


class RuntimeArtifactOrganizer:
    """Split ordered Runtime calls into requested layers and stages."""

    strategy_id = "runtime_artifact"

    def execute(self, request: ExecutionOrganizationRequest) -> tuple[RegionExecutionRecord, ...]:
        if request.execution.source_kind is not SourceKind.RUNTIME:
            raise SourceLoadError("RuntimeArtifactOrganizer requires a runtime execution")

        expected = self._build_scan_plan(request)
        calls = request.execution.operator_calls
        starts: list[int | None] = []
        cursor = 0
        evidence_exhausted = False
        for item in expected:
            if evidence_exhausted:
                starts.append(None)
                continue
            options = self._runtime_options(item.stage)
            start = self._find_boundary(
                calls,
                cursor,
                options.boundary_operators,
            )
            if start is None:
                starts.append(None)
                evidence_exhausted = True
                continue
            starts.append(start)
            cursor = start + 1

        region_stages: dict[str, list[StageExecutionRecord]] = {}
        layer_stages: dict[str, dict[int, list[StageExecutionRecord]]] = {}
        for expected_index, item in enumerate(expected):
            start = starts[expected_index]
            if not item.emit or start is None:
                continue
            # If the next expected boundary (including emit=False lookahead) is
            # missing, close the already-located stage at end-of-stream instead
            # of dropping it. Missing later stages still remain absent.
            next_start = starts[expected_index + 1] if expected_index + 1 < len(starts) else None
            end = next_start if next_start is not None else len(calls)
            options = self._runtime_options(item.stage)
            stage_calls = tuple(
                call
                for call in calls[start:end]
                if not any(_matches_operator_name(call.operator_name, ignored) for ignored in options.ignored_operators)
            )
            record = StageExecutionRecord(stage_id=item.stage.stage_id, operator_calls=stage_calls)
            if item.layer_index is None:
                region_stages.setdefault(item.region_id, []).append(record)
            else:
                layer_stages.setdefault(item.region_id, {}).setdefault(item.layer_index, []).append(record)

        organized: list[RegionExecutionRecord] = []
        for region in request.spec.regions:
            selected_indices = request.selected_layers.get(region.region_id, ())
            selected_region = region.region_id in request.selected_stage_regions
            if not selected_indices and not selected_region:
                continue
            organized.append(
                RegionExecutionRecord(
                    region_id=region.region_id,
                    stages=tuple(region_stages.get(region.region_id, ())),
                    layers=tuple(
                        LayerExecutionRecord(
                            layer_index=layer_index,
                            layer_kind=region.layer_layout[layer_index],
                            stages=tuple(layer_stages.get(region.region_id, {}).get(layer_index, ())),
                        )
                        for layer_index in selected_indices
                    ),
                )
            )
        return tuple(organized)

    def _build_scan_plan(self, request: ExecutionOrganizationRequest) -> list[_ExpectedStage]:
        regions = request.spec.regions
        known_region_ids = {region.region_id for region in regions}
        requested_region_ids = set(request.selected_layers).union(request.selected_stage_regions)
        unknown = requested_region_ids.difference(known_region_ids)
        if unknown:
            raise InvalidDiagnosticsRequest(f"unknown selected region {sorted(unknown)[0]!r}")
        if not requested_region_ids:
            return []

        last_requested_index = max(
            index for index, region in enumerate(regions) if region.region_id in requested_region_ids
        )
        expected: list[_ExpectedStage] = []
        for region_index, region in enumerate(regions[: last_requested_index + 1]):
            selected_indices = request.selected_layers.get(region.region_id, ())
            selected_region = region.region_id in request.selected_stage_regions
            if region.stages:
                expected.extend(
                    _ExpectedStage(
                        region_id=region.region_id,
                        layer_index=None,
                        layer_kind=None,
                        stage=stage,
                        emit=selected_region,
                    )
                    for stage in region.stages
                )

            if not region.layer_layout:
                if selected_indices:
                    raise InvalidDiagnosticsRequest(f"selected region {region.region_id!r} has no layer layout")
                continue

            if selected_indices and max(selected_indices) >= len(region.layer_layout):
                raise InvalidDiagnosticsRequest(
                    f"selected layer index {max(selected_indices)} exceeds region {region.region_id!r} layout"
                )
            must_scan_full_region = region_index < last_requested_index
            scan_through = len(region.layer_layout) - 1 if must_scan_full_region else max(selected_indices, default=-1)
            # Region-stage-only selection leaves selected_indices empty (scan_through
            # == -1). Still add the first layer boundary as a non-emit lookahead so
            # region stages are closed before subsequent layer operators.
            if scan_through < 0 and not selected_region:
                continue
            for layer_index in range(scan_through + 1):
                layer_kind = region.layer_layout[layer_index]
                for stage in region.layer_specs[layer_kind].stages:
                    expected.append(
                        _ExpectedStage(
                            region_id=region.region_id,
                            layer_index=layer_index,
                            layer_kind=layer_kind,
                            stage=stage,
                            emit=layer_index in selected_indices,
                        )
                    )
            if not must_scan_full_region and scan_through + 1 < len(region.layer_layout):
                lookahead_index = scan_through + 1
                lookahead_kind = region.layer_layout[lookahead_index]
                expected.append(
                    _ExpectedStage(
                        region_id=region.region_id,
                        layer_index=lookahead_index,
                        layer_kind=lookahead_kind,
                        stage=region.layer_specs[lookahead_kind].stages[0],
                        emit=False,
                    )
                )
        return expected

    @staticmethod
    def _runtime_options(stage: StageSpec) -> RuntimeStageOptions:
        options = stage.source_options.get(SourceKind.RUNTIME)
        if not isinstance(options, RuntimeStageOptions):
            raise SpecificationLoadError(f"stage {stage.stage_id!r} has no runtime options")
        return options

    @staticmethod
    def _find_boundary(
        calls,
        cursor: int,
        boundary_operators: tuple[str, ...],
    ) -> int | None:
        for index in range(cursor, len(calls)):
            if any(_matches_operator_name(calls[index].operator_name, boundary) for boundary in boundary_operators):
                return index
        return None


def _matches_operator_name(operator_name: str, pattern: str) -> bool:
    """Match an exact full name or the exact operator field of a full name."""

    if operator_name == pattern:
        return True
    if "." in pattern:
        return False
    parts = operator_name.split(".")
    return len(parts) >= 3 and parts[-2] == pattern
