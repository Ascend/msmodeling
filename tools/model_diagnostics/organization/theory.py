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
"""Shared Theory organization builder (Source flattens; Organizer reuses this)."""

from __future__ import annotations

from typing import Mapping

from tools.model_diagnostics.domain.models import (
    ModelRunContext,
    OperatorCallRecord,
    SourceKind,
    TensorInfo,
)
from tools.model_diagnostics.domain.organization import (
    ExecutionOrganizationRequest,
    LayerExecutionRecord,
    RegionExecutionRecord,
    StageExecutionRecord,
)
from tools.model_diagnostics.domain.specification import (
    ModelDiagnosticsSpec,
    TheoryOperatorSpec,
    TheoryStageOptions,
)
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest, SourceLoadError
from tools.model_diagnostics.specification.context_env import build_theory_env
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.expressions import evaluate_dtype, evaluate_shape


def _tensors_for_operator(
    operator: TheoryOperatorSpec,
    env: Mapping[str, object],
) -> tuple[TensorInfo, ...]:
    tensors: list[TensorInfo] = []
    for slot, tensor_spec in sorted(
        operator.tensors.items(),
        key=lambda item: (item[0].direction.value, item[0].index),
    ):
        shape = evaluate_shape(tensor_spec.shape.expression, env) if tensor_spec.shape is not None else None
        dtype = evaluate_dtype(tensor_spec.dtype.expression, env) if tensor_spec.dtype is not None else None
        tensors.append(TensorInfo(slot=slot, shape=shape, dtype=dtype))
    return tuple(tensors)


def _stage_calls(
    *,
    theory: TheoryStageOptions,
    env: Mapping[str, object],
    call_index_start: int,
    source_prefix: str,
) -> tuple[tuple[OperatorCallRecord, ...], int]:
    calls: list[OperatorCallRecord] = []
    call_index = call_index_start
    for operator in theory.operators:
        calls.append(
            OperatorCallRecord(
                call_index=call_index,
                operator_name=operator.operator_name,
                original_operator_name=None,
                tensors=_tensors_for_operator(operator, env),
                source_reference=f"{source_prefix}:{operator.operator_name}",
            )
        )
        call_index += 1
    return tuple(calls), call_index


def build_theory_regions(
    context: ModelRunContext,
    spec: ModelDiagnosticsSpec,
    selected_layers: Mapping[str, tuple[int, ...]],
    selected_stage_regions: tuple[str, ...],
) -> tuple[RegionExecutionRecord, ...]:
    """Build organized Theory regions directly from the materialized Spec."""

    env = build_theory_env(context)
    selected_stage_set = set(selected_stage_regions)
    region_by_id = {region.region_id: region for region in spec.regions}
    unknown = set(selected_layers).union(selected_stage_set).difference(region_by_id)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidDiagnosticsRequest(f"unknown selected region id(s): {names}")

    records: list[RegionExecutionRecord] = []
    call_index = 0
    for region in spec.regions:
        emit_stages = region.region_id in selected_stage_set
        layer_selection = selected_layers.get(region.region_id)
        if not emit_stages and layer_selection is None:
            continue

        region_stages: list[StageExecutionRecord] = []
        if emit_stages:
            for stage in region.stages:
                theory = stage.source_options.get(SourceKind.THEORY)
                if not isinstance(theory, TheoryStageOptions):
                    raise SpecificationLoadError(f"stage {stage.stage_id!r} missing theory source_options")
                calls, call_index = _stage_calls(
                    theory=theory,
                    env=env,
                    call_index_start=call_index,
                    source_prefix=f"theory:{region.region_id}:{stage.stage_id}",
                )
                region_stages.append(StageExecutionRecord(stage_id=stage.stage_id, operator_calls=calls))

        layers: list[LayerExecutionRecord] = []
        if layer_selection is not None:
            if not region.layer_layout:
                raise SpecificationLoadError(
                    f"region {region.region_id!r} has selected layers but empty layer_layout; "
                    "Spec must be materialized with Context before Theory organization"
                )
            for layer_index in layer_selection:
                if layer_index >= len(region.layer_layout):
                    raise InvalidDiagnosticsRequest(
                        f"selected layer {layer_index} out of range for region {region.region_id!r}"
                    )
                layer_kind = region.layer_layout[layer_index]
                layer_spec = region.layer_specs[layer_kind]
                layer_env = {**env, "LAYER": layer_index}
                stage_records: list[StageExecutionRecord] = []
                for stage in layer_spec.stages:
                    theory = stage.source_options.get(SourceKind.THEORY)
                    if not isinstance(theory, TheoryStageOptions):
                        raise SpecificationLoadError(f"stage {stage.stage_id!r} missing theory source_options")
                    calls, call_index = _stage_calls(
                        theory=theory,
                        env=layer_env,
                        call_index_start=call_index,
                        source_prefix=(f"theory:{region.region_id}:L{layer_index}:{stage.stage_id}"),
                    )
                    stage_records.append(StageExecutionRecord(stage_id=stage.stage_id, operator_calls=calls))
                layers.append(
                    LayerExecutionRecord(
                        layer_index=layer_index,
                        layer_kind=layer_kind,
                        stages=tuple(stage_records),
                    )
                )

        if not region_stages and not layers:
            continue

        records.append(
            RegionExecutionRecord(
                region_id=region.region_id,
                stages=tuple(region_stages),
                layers=tuple(layers),
            )
        )
    return tuple(records)


def flatten_theory_calls(
    regions: tuple[RegionExecutionRecord, ...],
) -> tuple[OperatorCallRecord, ...]:
    """Flatten organized Theory regions into the Source Protocol call stream."""

    calls: list[OperatorCallRecord] = []
    for region in regions:
        for stage in region.stages:
            calls.extend(stage.operator_calls)
        for layer in region.layers:
            for stage in layer.stages:
                calls.extend(stage.operator_calls)
    return tuple(calls)


class TheoryExecutionOrganizationStrategy:
    """Organize Theory by rebuilding from Spec, then checking the Source stream.

    Rebuild avoids Source→Organizer round-trip redundancy. The cheap assertion
    against ``execution.operator_calls`` keeps third-party Theory Sources honest:
    call count and ordered operator names must still match the Spec materialization.
    """

    strategy_id = "theory_organization"

    def execute(self, request: ExecutionOrganizationRequest) -> tuple[RegionExecutionRecord, ...]:
        if request.execution.source_kind is not SourceKind.THEORY:
            raise SourceLoadError("Theory organizer requires THEORY execution records")
        regions = build_theory_regions(
            request.execution.run_context,
            request.spec,
            request.selected_layers,
            request.selected_stage_regions,
        )
        _assert_execution_matches_regions(request.execution.operator_calls, regions)
        return regions


def _assert_execution_matches_regions(
    execution_calls: tuple[OperatorCallRecord, ...],
    regions: tuple[RegionExecutionRecord, ...],
) -> None:
    expected = flatten_theory_calls(regions)
    if len(execution_calls) != len(expected):
        raise SourceLoadError(
            "theory execution call count diverges from Spec organization: "
            f"execution has {len(execution_calls)}, Spec expects {len(expected)}"
        )
    for index, (actual, want) in enumerate(zip(execution_calls, expected)):
        if actual.operator_name != want.operator_name:
            raise SourceLoadError(
                f"theory execution diverges from Spec at call[{index}]: "
                f"expected {want.operator_name!r}, got {actual.operator_name!r}"
            )
