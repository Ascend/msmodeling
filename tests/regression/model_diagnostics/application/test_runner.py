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
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import pytest

from tools.model_diagnostics.application import (
    InvalidDiagnosticsRequest,
    ModelDiagnosticsRunner,
    SourceLoadError,
)
from tools.model_diagnostics.application.runner import _validate_request_against_spec
from tools.model_diagnostics.comparison import OneToOneEqualStrategy, StageComparisonRegistry
from tools.model_diagnostics.domain import (
    INPUT,
    OUTPUT,
    ComparisonSpec,
    DiagnosticsRequest,
    ExecutionPhase,
    FindingStatus,
    LayerExecutionRecord,
    LayerSpec,
    ModelDiagnosticsSpec,
    ModelExecutionRecord,
    ModelRunContext,
    OneToOneOptions,
    OperatorCallRecord,
    ParallelContext,
    ProducerInfo,
    RegionExecutionRecord,
    RegionSpec,
    RuntimeStageOptions,
    SourceKind,
    SourceDescription,
    SpecMatchCriteria,
    StageExecutionRecord,
    StageSpec,
    TensorInfo,
    TensorMapping,
    TensorMappingMode,
)
from tools.model_diagnostics.organization import RuntimeArtifactOrganizer


def _context() -> ModelRunContext:
    return ModelRunContext(
        model_name="independent/test",
        entrypoint="test",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=2,
        parallel=ParallelContext(),
        model_config={},
        quantization_config={},
    )


def _comparison() -> ComparisonSpec:
    return ComparisonSpec(
        "one_to_one",
        OneToOneOptions(TensorMapping(TensorMappingMode.POSITIONAL)),
    )


def _spec(*, supported=True) -> ModelDiagnosticsSpec:
    stage = StageSpec(
        stage_id="attention",
        source_options={},
        comparisons=({(SourceKind.THEORY, SourceKind.RUNTIME): _comparison()} if supported else {}),
    )
    return ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="test",
        spec_version="1.0.0",
        model_category="test",
        matches=SpecMatchCriteria(model_types=("test",)),
        regions=(
            RegionSpec(
                region_id="language",
                layer_layout=("dense",),
                layer_specs={"dense": LayerSpec("dense", (stage,))},
            ),
        ),
    )


def _call(index: int) -> OperatorCallRecord:
    return OperatorCallRecord(
        call_index=index,
        operator_name="mm",
        original_operator_name=None,
        tensors=(
            TensorInfo(INPUT[0], (2, 4), "float16"),
            TensorInfo(OUTPUT[0], (2, 4), "float16"),
        ),
    )


@dataclass
class _Provider:
    spec: ModelDiagnosticsSpec

    def get(self, context):
        return self.spec


@dataclass
class _Source:
    source_kind: SourceKind
    execution: ModelExecutionRecord
    description: SourceDescription | None = None

    def describe(self):
        return self.description or SourceDescription(self.source_kind)

    def load_execution(self, context, spec, selected_layers, selected_stage_regions):
        return self.execution


@dataclass
class _FailIfLoadedSource:
    source_kind: SourceKind

    def describe(self):
        return SourceDescription(self.source_kind)

    def load_execution(self, context, spec, selected_layers, selected_stage_regions):
        raise AssertionError("Source must not load before request validation")


@dataclass
class _Organizer:
    regions: tuple[RegionExecutionRecord, ...]
    strategy_id: str = "synthetic"

    def execute(self, request):
        return self.regions


@dataclass(frozen=True)
class _Parser:
    def parse(self, raw):
        return _comparison().options


def _regions(call: OperatorCallRecord, *, include_stage=True):
    stages = (StageExecutionRecord("attention", (call,)),) if include_stage else ()
    return (
        RegionExecutionRecord(
            region_id="language",
            layers=(LayerExecutionRecord(0, "dense", stages),),
        ),
    )


def _runner(spec, left_regions, right_regions):
    registry = StageComparisonRegistry()
    registry.register(
        "one_to_one",
        option_parser=_Parser(),
        strategy=OneToOneEqualStrategy(),
    )
    return ModelDiagnosticsRunner(
        _Provider(spec),
        {
            SourceKind.THEORY: _Organizer(left_regions),
            SourceKind.RUNTIME: _Organizer(right_regions),
        },
        registry,
    )


def _sources(context):
    return (
        _Source(SourceKind.THEORY, ModelExecutionRecord(SourceKind.THEORY, context, (_call(10),))),
        _Source(SourceKind.RUNTIME, ModelExecutionRecord(SourceKind.RUNTIME, context, (_call(20),))),
    )


def test_runner_pairs_declared_stage_and_aggregates_result() -> None:
    context = _context()
    left, right = _sources(context)
    runner = _runner(_spec(), _regions(_call(10)), _regions(_call(20)))

    result = runner.run(DiagnosticsRequest(context, {"language": (0,)}), left, right)

    assert result.summary.overall_status is FindingStatus.PASS
    assert result.findings[0].status is FindingStatus.PASS
    assert result.left_source.source_kind is SourceKind.THEORY
    assert result.right_source.source_kind is SourceKind.RUNTIME


def test_runner_preserves_source_description_provenance() -> None:
    context = _context()
    left, right = _sources(context)
    producer = ProducerInfo("0.2.0", "abc123", "runtime_observer")
    right.description = SourceDescription(
        SourceKind.RUNTIME,
        artifact_reference="runtime.json",
        producer=producer,
    )
    runner = _runner(_spec(), _regions(_call(10)), _regions(_call(20)))

    result = runner.run(DiagnosticsRequest(context, {"language": (0,)}), left, right)

    assert result.right_source.artifact_reference == "runtime.json"
    assert result.right_source.producer == producer


def test_runner_reports_missing_stage_without_truncating_pairing() -> None:
    context = _context()
    left, right = _sources(context)
    runner = _runner(_spec(), _regions(_call(10)), _regions(_call(20), include_stage=False))

    result = runner.run(DiagnosticsRequest(context, {"language": (0,)}), left, right)

    assert result.summary.overall_status is FindingStatus.INCOMPLETE
    assert result.findings[0].message_code == "stage.missing"
    assert result.findings[0].left_evidence
    assert not result.findings[0].right_evidence


def test_runner_reports_incomplete_for_valid_runtime_evidence_without_boundary() -> None:
    context = _context()
    stage = StageSpec(
        stage_id="attention",
        source_options={
            SourceKind.RUNTIME: RuntimeStageOptions(("attention_boundary",)),
        },
        comparisons={(SourceKind.THEORY, SourceKind.RUNTIME): _comparison()},
    )
    spec = ModelDiagnosticsSpec(
        schema_version="1",
        spec_id="test",
        spec_version="1.0.0",
        model_category="test",
        matches=SpecMatchCriteria(model_types=("test",)),
        regions=(
            RegionSpec(
                region_id="language",
                layer_layout=("dense",),
                layer_specs={"dense": LayerSpec("dense", (stage,))},
            ),
        ),
    )
    registry = StageComparisonRegistry()
    registry.register(
        "one_to_one",
        option_parser=_Parser(),
        strategy=OneToOneEqualStrategy(),
    )
    runner = ModelDiagnosticsRunner(
        _Provider(spec),
        {
            SourceKind.THEORY: _Organizer(_regions(_call(10))),
            SourceKind.RUNTIME: RuntimeArtifactOrganizer(),
        },
        registry,
    )
    theory = _Source(
        SourceKind.THEORY,
        ModelExecutionRecord(SourceKind.THEORY, context, (_call(10),)),
    )
    runtime = _Source(
        SourceKind.RUNTIME,
        ModelExecutionRecord(
            SourceKind.RUNTIME,
            context,
            (OperatorCallRecord(0, "unrelated", None, ()),),
        ),
    )

    result = runner.run(
        DiagnosticsRequest(context, {"language": (0,)}),
        theory,
        runtime,
    )

    assert result.summary.overall_status is FindingStatus.INCOMPLETE
    assert result.findings[0].message_code == "stage.missing"


def test_runner_defaults_to_one_to_one_when_pair_is_unconfigured() -> None:
    context = _context()
    left, right = _sources(context)
    runner = _runner(_spec(supported=False), _regions(_call(10)), _regions(_call(20)))

    result = runner.run(DiagnosticsRequest(context, {"language": (0,)}), left, right)

    assert result.summary.overall_status is FindingStatus.PASS
    assert len(result.findings) == 1
    assert result.findings[0].status is FindingStatus.PASS
    assert "/float16" in str(result.findings[0].expected)
    assert result.findings[0].expected == result.findings[0].actual


def test_runner_rejects_source_context_drift() -> None:
    context = _context()
    different = ModelRunContext(
        model_name="different",
        entrypoint=context.entrypoint,
        phase=context.phase,
        batch_size=context.batch_size,
        query_length=context.query_length,
        context_length=context.context_length,
        parallel=context.parallel,
        model_config={},
        quantization_config={},
    )
    left, right = _sources(context)
    right.execution = ModelExecutionRecord(SourceKind.RUNTIME, different, (_call(20),))
    runner = _runner(_spec(), _regions(_call(10)), _regions(_call(20)))

    with pytest.raises(SourceLoadError, match="context"):
        runner.run(DiagnosticsRequest(context, {"language": (0,)}), left, right)


def test_runner_validates_unknown_region_before_loading_sources() -> None:
    runner = _runner(_spec(), (), ())
    request = DiagnosticsRequest(_context(), {"unknown": (0,)})

    with pytest.raises(InvalidDiagnosticsRequest, match="unknown selected region"):
        runner.run(
            request,
            _FailIfLoadedSource(SourceKind.THEORY),
            _FailIfLoadedSource(SourceKind.RUNTIME),
        )


def test_runner_validates_layer_range_before_loading_sources() -> None:
    runner = _runner(_spec(), (), ())
    request = DiagnosticsRequest(_context(), {"language": (1,)})

    with pytest.raises(InvalidDiagnosticsRequest, match="selected layer 1"):
        runner.run(
            request,
            _FailIfLoadedSource(SourceKind.THEORY),
            _FailIfLoadedSource(SourceKind.RUNTIME),
        )


def test_runner_rejects_negative_layer_indices_against_spec() -> None:
    # DiagnosticsRequest already rejects negatives; exercise the runner gate directly
    # so a bypass cannot silently map -1 onto the last layout entry.
    request = SimpleNamespace(
        selected_layers=MappingProxyType({"language": (-1,)}),
        selected_stage_regions=(),
    )

    with pytest.raises(InvalidDiagnosticsRequest, match="selected layer -1"):
        _validate_request_against_spec(request, _spec().regions)
