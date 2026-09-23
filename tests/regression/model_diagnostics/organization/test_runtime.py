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
from dataclasses import replace

from tools.model_diagnostics.specification.ignore_groups import load_builtin_ignore_groups

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


def test_runtime_organizer_preserves_calls_before_first_boundary() -> None:
    execution = _execution()
    prefixed_calls = (
        OperatorCallRecord(
            call_index=0,
            operator_name="unexpected_prefix",
            original_operator_name=None,
            tensors=(),
        ),
        *(
            OperatorCallRecord(
                call_index=call.call_index + 1,
                operator_name=call.operator_name,
                original_operator_name=call.original_operator_name,
                tensors=call.tensors,
            )
            for call in execution.operator_calls
        ),
    )
    request = ExecutionOrganizationRequest(
        execution=ModelExecutionRecord(
            source_kind=SourceKind.RUNTIME,
            run_context=execution.run_context,
            operator_calls=prefixed_calls,
        ),
        spec=_spec(),
        selected_layers={"language": (0,)},
        selected_stage_regions=(),
    )

    regions = RuntimeArtifactOrganizer().execute(request)

    attention = regions[0].layers[0].stages[0]
    assert [call.operator_name for call in attention.operator_calls] == [
        "unexpected_prefix",
        "rms_norm",
        "q_proj",
    ]


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


def test_runtime_boundary_matches_aten_rmsnorm_sequence() -> None:
    from types import SimpleNamespace

    from tools.model_diagnostics.organization.runtime import _matches_composite_boundary

    calls = [
        SimpleNamespace(operator_name="prims.convert_element_type.default"),
        SimpleNamespace(operator_name="aten.pow.Tensor_Scalar"),
        SimpleNamespace(operator_name="aten.mean.dim"),
        SimpleNamespace(operator_name="aten.add.Tensor"),
        SimpleNamespace(operator_name="aten.rsqrt.default"),
        SimpleNamespace(operator_name="aten.mul.Tensor"),
        SimpleNamespace(operator_name="aten.mul.Tensor"),
        SimpleNamespace(operator_name="prims.convert_element_type.default"),
    ]

    assert _matches_composite_boundary(calls, 1, "rms_norm") is True
    assert _matches_composite_boundary(calls, 0, "rms_norm") is False
    assert _matches_composite_boundary(calls, 3, "rms_norm") is False
    assert _matches_composite_boundary(calls, 0, "attention") is False


def test_runtime_boundary_matches_cast_wrapped_aten_rmsnorm_sequence() -> None:
    from types import SimpleNamespace

    from tools.model_diagnostics.organization.runtime import _matches_composite_boundary

    calls = [
        SimpleNamespace(operator_name="aten._to_copy.default"),
        SimpleNamespace(operator_name="aten.pow.Tensor_Scalar"),
        SimpleNamespace(operator_name="aten.mean.dim"),
        SimpleNamespace(operator_name="aten.add.Tensor"),
        SimpleNamespace(operator_name="aten.rsqrt.default"),
        SimpleNamespace(operator_name="aten.mul.Tensor"),
        SimpleNamespace(operator_name="aten._to_copy.default"),
        SimpleNamespace(operator_name="aten.mul.Tensor"),
    ]

    assert _matches_composite_boundary(calls, 0, "rms_norm") is True
    assert _matches_composite_boundary(calls, 1, "rms_norm") is False
    from tools.model_diagnostics.organization.runtime import _composite_boundary_width

    assert _composite_boundary_width(calls, 0, "rms_norm") == 8
    assert _composite_boundary_width(calls, 1, "rms_norm") == 0


def test_semantic_ignore_families_preserve_boundaries_and_core_evidence() -> None:
    groups = load_builtin_ignore_groups()
    ignored = tuple(
        name
        for group in (
            "collective_communication",
            "rms_norm_kernels",
            "quantization",
            "dtype_cast",
            "layout_permute",
            "shape_views",
            "tensor_storage",
            "tensor_split",
            "rotary_embedding",
        )
        for name in groups[group]
    )
    spec = replace(
        _spec(),
        regions=(
            RegionSpec(
                region_id="language",
                stages=(
                    _stage("projection", "rms_norm", *ignored),
                    _stage("output", "linear_attn_gated_rmsnorm", *ignored),
                ),
            ),
        ),
    )
    names = (
        "rms_norm",
        *ignored,
        "prims.convert_element_type.default",
        "q_proj",
        "linear_attn_gated_rmsnorm",
        "native_layer_norm",
        "quant_lightning_indexer",
        "permute_tokens",
        "unpermute_tokens",
        "cat",
        "mul",
        "sum",
    )
    execution = replace(
        _execution(),
        operator_calls=tuple(
            OperatorCallRecord(call_index=i, operator_name=name, original_operator_name=None, tensors=())
            for i, name in enumerate(names)
        ),
    )
    organized = RuntimeArtifactOrganizer().execute(
        ExecutionOrganizationRequest(
            execution=execution,
            spec=spec,
            selected_layers={},
            selected_stage_regions=("language",),
        )
    )
    first, second = organized[0].stages
    assert tuple(call.operator_name for call in first.operator_calls) == ("q_proj",)
    assert tuple(call.operator_name for call in second.operator_calls) == (
        "linear_attn_gated_rmsnorm",
        "native_layer_norm",
        "quant_lightning_indexer",
        "permute_tokens",
        "unpermute_tokens",
        "cat",
        "mul",
        "sum",
    )
    assert first.operator_calls[0] is execution.operator_calls[names.index("q_proj")]
