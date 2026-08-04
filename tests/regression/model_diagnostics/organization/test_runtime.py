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
from tools.model_diagnostics.domain import (
    INPUT,
    OUTPUT,
    ExecutionOrganizationRequest,
    LayerSpec,
    ModelDiagnosticsSpec,
    ModelExecutionRecord,
    ModelRunContext,
    OperatorCallRecord,
    ParallelContext,
    ProducerInfo,
    RegionSpec,
    RuntimeStageOptions,
    SourceKind,
    SpecMatchCriteria,
    StageSpec,
)
from tools.model_diagnostics.organization.runtime import RuntimeArtifactOrganizer
from tools.model_diagnostics.sources.runtime_capture import RuntimeArtifactCapture
from tools.model_diagnostics.sources.simulation_artifact import execution_from_runtime_artifact


def _context() -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-test",
        entrypoint="capture",
        phase=None,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(),
        model_config={},
        quantization_config={},
    )


def _stage(
    stage_id: str,
    boundary: str,
    *ignored: str,
) -> StageSpec:
    return StageSpec(
        stage_id=stage_id,
        source_options={
            SourceKind.RUNTIME: RuntimeStageOptions(
                boundary_operators=(boundary,),
                ignored_operators=ignored,
            )
        },
        comparisons={},
    )


def _spec() -> ModelDiagnosticsSpec:
    dense = LayerSpec(
        layer_kind="dense",
        stages=(
            _stage("attention", "rms_norm", "view"),
            _stage("dense_ffn", "ffn_norm", "view"),
        ),
    )
    return ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="qwen3_test",
        spec_version="1.0.0",
        model_category="qwen3_dense",
        matches=SpecMatchCriteria(model_types=("qwen3",)),
        regions=(
            RegionSpec(
                region_id="language",
                layer_layout=("dense", "dense"),
                layer_specs={"dense": dense},
            ),
        ),
    )


def _execution() -> ModelExecutionRecord:
    names = (
        "rms_norm",
        "q_proj",
        "view",
        "ffn_norm",
        "up_proj",
        "rms_norm",
        "q_proj",
        "view",
        "ffn_norm",
        "up_proj",
    )
    return ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=_context(),
        operator_calls=tuple(
            OperatorCallRecord(
                call_index=index,
                operator_name=name,
                original_operator_name=None,
                tensors=(),
            )
            for index, name in enumerate(names)
        ),
    )


def test_runtime_organizer_scans_prior_layers_but_materializes_only_selected_layers() -> None:
    request = ExecutionOrganizationRequest(
        execution=_execution(),
        spec=_spec(),
        selected_layers={"language": (1,)},
        selected_stage_regions=(),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    assert len(regions) == 1
    assert regions[0].region_id == "language"
    assert [layer.layer_index for layer in regions[0].layers] == [1]
    assert [stage.stage_id for stage in regions[0].layers[0].stages] == ["attention", "dense_ffn"]
    assert [[call.call_index for call in stage.operator_calls] for stage in regions[0].layers[0].stages] == [
        [5, 6],
        [8, 9],
    ]


def test_runtime_organizer_only_filters_operators_declared_by_the_stage() -> None:
    request = ExecutionOrganizationRequest(
        execution=_execution(),
        spec=_spec(),
        selected_layers={"language": (0,)},
        selected_stage_regions=(),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    attention = regions[0].layers[0].stages[0]
    assert [call.operator_name for call in attention.operator_calls] == ["rms_norm", "q_proj"]


def test_runtime_operator_patterns_match_exact_operator_field_not_substrings() -> None:
    spec = ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="operator_field_match",
        spec_version="1.0.0",
        model_category="qwen3_dense",
        matches=SpecMatchCriteria(model_types=("qwen3",)),
        regions=(
            RegionSpec(
                region_id="input",
                stages=(_stage("attention", "rms_norm", "view"),),
            ),
        ),
    )
    names = (
        "tensor_cast.rms_norm.default",
        "aten.view.default",
        "aten.preview.default",
    )
    execution = ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=_context(),
        operator_calls=tuple(OperatorCallRecord(index, name, None, ()) for index, name in enumerate(names)),
    )

    regions = RuntimeArtifactOrganizer().execute(
        ExecutionOrganizationRequest(
            execution=execution,
            spec=spec,
            selected_layers={},
            selected_stage_regions=("input",),
        )
    )

    assert [call.operator_name for call in regions[0].stages[0].operator_calls] == [
        "tensor_cast.rms_norm.default",
        "aten.preview.default",
    ]


def test_runtime_organizer_uses_next_layer_boundary_to_stop_selected_layer() -> None:
    request = ExecutionOrganizationRequest(
        execution=_execution(),
        spec=_spec(),
        selected_layers={"language": (0,)},
        selected_stage_regions=(),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    dense_ffn = regions[0].layers[0].stages[1]
    assert [call.call_index for call in dense_ffn.operator_calls] == [3, 4]


def test_runtime_organizer_closes_region_stage_at_first_layer_when_only_region_selected() -> None:
    dense = LayerSpec(
        layer_kind="dense",
        stages=(
            _stage("attention", "rms_norm", "view"),
            _stage("dense_ffn", "ffn_norm", "view"),
        ),
    )
    spec = ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="region_then_layers",
        spec_version="1.0.0",
        model_category="qwen3_dense",
        matches=SpecMatchCriteria(model_types=("qwen3",)),
        regions=(
            RegionSpec(
                region_id="language",
                stages=(_stage("pre_layer", "embedding"),),
                layer_layout=("dense", "dense"),
                layer_specs={"dense": dense},
            ),
        ),
    )
    names = (
        "embedding",
        "rms_norm",
        "q_proj",
        "view",
        "ffn_norm",
        "up_proj",
        "rms_norm",
        "q_proj",
    )
    execution = ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=_context(),
        operator_calls=tuple(OperatorCallRecord(index, name, None, ()) for index, name in enumerate(names)),
    )

    regions = RuntimeArtifactOrganizer().execute(
        ExecutionOrganizationRequest(
            execution=execution,
            spec=spec,
            selected_layers={},
            selected_stage_regions=("language",),
        )
    )

    assert regions[0].layers == ()
    assert [stage.stage_id for stage in regions[0].stages] == ["pre_layer"]
    assert [call.operator_name for call in regions[0].stages[0].operator_calls] == ["embedding"]


def test_runtime_organizer_closes_located_stage_when_next_boundary_is_missing() -> None:
    execution = ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=_context(),
        operator_calls=(
            OperatorCallRecord(0, "rms_norm", None, ()),
            OperatorCallRecord(1, "q_proj", None, ()),
        ),
    )
    request = ExecutionOrganizationRequest(
        execution=execution,
        spec=_spec(),
        selected_layers={"language": (0,)},
        selected_stage_regions=(),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    assert len(regions) == 1
    assert len(regions[0].layers) == 1
    assert [stage.stage_id for stage in regions[0].layers[0].stages] == ["attention"]
    assert [call.call_index for call in regions[0].layers[0].stages[0].operator_calls] == [0, 1]


def test_runtime_organizer_returns_missing_stage_when_first_boundary_is_absent() -> None:
    execution = ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=_context(),
        operator_calls=(
            OperatorCallRecord(0, "unrelated", None, ()),
            OperatorCallRecord(1, "also_unrelated", None, ()),
        ),
    )
    request = ExecutionOrganizationRequest(
        execution=execution,
        spec=_spec(),
        selected_layers={"language": (0,)},
        selected_stage_regions=(),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    assert len(regions) == 1
    assert len(regions[0].layers) == 1
    assert regions[0].layers[0].stages == ()


def test_runtime_organizer_preserves_spec_region_order_and_scans_to_selected_output() -> None:
    base_spec = _spec()
    full_spec = ModelDiagnosticsSpec(
        schema_version=base_spec.schema_version,
        spec_id=base_spec.spec_id,
        spec_version=base_spec.spec_version,
        model_category=base_spec.model_category,
        matches=base_spec.matches,
        regions=(
            RegionSpec(region_id="input", stages=(_stage("embedding", "embedding"),)),
            base_spec.regions[0],
            RegionSpec(region_id="output", stages=(_stage("lm_head", "lm_head"),)),
        ),
    )
    names = (
        "embedding",
        *(call.operator_name for call in _execution().operator_calls),
        "lm_head",
    )
    execution = ModelExecutionRecord(
        source_kind=SourceKind.RUNTIME,
        run_context=_context(),
        operator_calls=tuple(OperatorCallRecord(index, name, None, ()) for index, name in enumerate(names)),
    )
    request = ExecutionOrganizationRequest(
        execution=execution,
        spec=full_spec,
        selected_layers={"language": (0,)},
        selected_stage_regions=("output", "input"),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    assert [region.region_id for region in regions] == ["input", "language", "output"]
    assert [call.operator_name for call in regions[0].stages[0].operator_calls] == ["embedding"]
    assert [call.operator_name for call in regions[2].stages[0].operator_calls] == ["lm_head"]


def test_real_runtime_artifact_flows_directly_into_stage_organization() -> None:
    import torch
    from tensor_cast.runtime import Runtime

    left = torch.ones((2, 4), dtype=torch.float16)
    right = torch.ones((4, 3), dtype=torch.float16)
    with Runtime([], None) as runtime:
        capture_start = len(runtime.op_invoke_infos)
        output = torch.mm(left, right)
        torch.relu(output)

    artifact = RuntimeArtifactCapture.snapshot(
        runtime,
        start_invocation_index=capture_start,
        run_context=_context(),
        producer=ProducerInfo(
            package_version="0.2.0",
            git_revision=None,
            capture_backend="tensor_cast.runtime_observer",
        ),
    )
    assert [call.operator_name for call in artifact.operator_calls] == [
        "aten.mm.default",
        "aten.relu.default",
    ]
    assert [(tensor.slot, tensor.shape, tensor.dtype) for tensor in artifact.operator_calls[0].tensors] == [
        (INPUT[0], (2, 4), "float16"),
        (INPUT[1], (4, 3), "float16"),
        (OUTPUT[0], (2, 3), "float16"),
    ]
    execution = execution_from_runtime_artifact(artifact)
    spec = ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="runtime_smoke",
        spec_version="1.0.0",
        model_category="smoke",
        matches=SpecMatchCriteria(model_types=("smoke",)),
        regions=(
            RegionSpec(
                region_id="smoke",
                stages=(
                    _stage("matmul", "aten.mm.default"),
                    _stage("activation", "aten.relu.default"),
                ),
            ),
        ),
    )

    regions = RuntimeArtifactOrganizer().execute(
        ExecutionOrganizationRequest(
            execution=execution,
            spec=spec,
            selected_layers={},
            selected_stage_regions=("smoke",),
        )
    )

    assert execution.source_kind is SourceKind.RUNTIME
    assert [[call.operator_name for call in stage.operator_calls] for stage in regions[0].stages] == [
        ["aten.mm.default"],
        ["aten.relu.default"],
    ]
