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
from dataclasses import replace
from types import MappingProxyType

import pytest

from tools.model_diagnostics.domain import (
    INPUT,
    DiagnosticsRequest,
    ExecutionPhase,
    ModelExecutionRecord,
    ModelRunContext,
    OperatorCallRecord,
    ParallelContext,
    ProducerInfo,
    SimulationExecutionArtifact,
    SourceKind,
    TensorDirection,
    TensorInfo,
    TensorSlot,
    validate_expert_parallel_features,
)
from tools.model_diagnostics.domain.models import _TensorSlots
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest


def _context() -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=128,
        context_length=128,
        parallel=ParallelContext(tensor_parallel_size=1),
        model_config={"num_hidden_layers": 36},
        quantization_config={},
    )


@pytest.mark.parametrize("external,redundant", [(True, False), (False, True), (True, True)])
def test_expert_features_require_expert_parallel(external: bool, redundant: bool) -> None:
    with pytest.raises(ValueError, match="require expert_parallel_size > 1"):
        validate_expert_parallel_features(
            1,
            enable_external_shared_experts=external,
            enable_redundant_experts=redundant,
        )


def test_expert_feature_validation_accepts_disabled_features_or_ep() -> None:
    validate_expert_parallel_features(
        1,
        enable_external_shared_experts=False,
        enable_redundant_experts=False,
    )
    validate_expert_parallel_features(
        2,
        enable_external_shared_experts=True,
        enable_redundant_experts=True,
    )


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "expert_parallel_size",
        "moe_data_parallel_size",
    ],
)
def test_parallel_context_rejects_non_positive_sizes(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        ParallelContext(**{field: 0})


def test_model_run_context_freezes_configuration_mappings() -> None:
    model_config = {"num_hidden_layers": 36}
    context = ModelRunContext(
        model_name="Qwen3-8B",
        entrypoint=None,
        phase=None,
        batch_size=None,
        query_length=None,
        context_length=None,
        parallel=ParallelContext(),
        model_config=model_config,
        quantization_config={},
    )

    model_config["num_hidden_layers"] = 40

    assert isinstance(context.model_config, MappingProxyType)
    assert context.model_config["num_hidden_layers"] == 36


@pytest.mark.parametrize("field", ["batch_size", "query_length"])
def test_model_run_context_rejects_non_positive_dimensions(field: str) -> None:
    values = {
        "model_name": "Qwen3-8B",
        "entrypoint": None,
        "phase": None,
        "batch_size": None,
        "query_length": None,
        "context_length": None,
        "parallel": ParallelContext(),
        "model_config": {},
        "quantization_config": {},
    }
    values[field] = 0

    with pytest.raises(ValueError, match=field):
        ModelRunContext(**values)


def test_model_run_context_accepts_zero_context_length() -> None:
    context = replace(_context(), context_length=0)

    assert context.context_length == 0


def test_model_run_context_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        ModelRunContext(
            model_name=" ",
            entrypoint=None,
            phase=None,
            batch_size=None,
            query_length=None,
            context_length=None,
            parallel=ParallelContext(),
            model_config={},
            quantization_config={},
        )


def test_model_run_context_rejects_empty_business_entrypoint() -> None:
    with pytest.raises(ValueError, match="business entrypoint"):
        ModelRunContext(
            model_name="Qwen3-8B",
            entrypoint=" ",
            phase=None,
            batch_size=None,
            query_length=None,
            context_length=None,
            parallel=ParallelContext(),
            model_config={},
            quantization_config={},
        )


def test_tensor_slot_rejects_negative_or_boolean_index() -> None:
    with pytest.raises(ValueError, match="index"):
        TensorSlot(TensorDirection.INPUT, -1)
    with pytest.raises(TypeError, match="index"):
        TensorSlot(TensorDirection.INPUT, True)


def test_tensor_slot_helper_constructs_input_slot() -> None:
    assert INPUT[2] == TensorSlot(TensorDirection.INPUT, 2)


def test_tensor_slot_helper_supports_each_direction() -> None:
    outputs = _TensorSlots(TensorDirection.OUTPUT)

    assert outputs[1] == TensorSlot(TensorDirection.OUTPUT, 1)


def test_operator_call_requires_non_negative_index_and_name() -> None:
    tensor = TensorInfo(INPUT[0], (1, 128, 4096), "bfloat16")

    with pytest.raises(ValueError, match="call_index"):
        OperatorCallRecord(-1, "linear", None, (tensor,))
    with pytest.raises(ValueError, match="operator_name"):
        OperatorCallRecord(0, "", None, (tensor,))


def test_tensor_info_validates_shape_dimensions() -> None:
    with pytest.raises(ValueError, match="shape"):
        TensorInfo(INPUT[0], (1, -1), "bfloat16")
    with pytest.raises(TypeError, match="shape"):
        TensorInfo(INPUT[0], (1, True), "bfloat16")


def test_diagnostics_request_normalizes_selected_layers() -> None:
    request = DiagnosticsRequest(
        context=_context(),
        selected_layers={"language": (5, 0, 5, 1)},
    )

    assert request.selected_layers == {"language": (0, 1, 5)}


def test_diagnostics_request_requires_a_selection() -> None:
    with pytest.raises(InvalidDiagnosticsRequest, match="selection"):
        DiagnosticsRequest(context=_context(), selected_layers={})


def test_diagnostics_request_accepts_stage_regions_only_and_deduplicates() -> None:
    request = DiagnosticsRequest(
        context=_context(),
        selected_layers={},
        selected_stage_regions=("input", "output", "input"),
    )

    assert request.selected_stage_regions == ("input", "output")


@pytest.mark.parametrize("layer_indices", [(), (-1,), (True,)])
def test_diagnostics_request_rejects_invalid_layer_selection(layer_indices: tuple[int, ...]) -> None:
    with pytest.raises(InvalidDiagnosticsRequest, match="selected layer"):
        DiagnosticsRequest(context=_context(), selected_layers={"language": layer_indices})


def test_model_execution_source_kind_is_explicit() -> None:
    assert SourceKind.THEORY.value == "theory"
    assert SourceKind.RUNTIME.value == "runtime"


def test_model_execution_record_requires_unique_ordered_calls() -> None:
    tensor = TensorInfo(INPUT[0], (1, 128, 4096), "bfloat16")
    first = OperatorCallRecord(0, "rms_norm", None, (tensor,))
    second = OperatorCallRecord(1, "linear", None, (tensor,))

    record = ModelExecutionRecord(SourceKind.RUNTIME, _context(), (first, second))

    assert record.operator_calls == (first, second)
    with pytest.raises(ValueError, match="unique and ordered"):
        ModelExecutionRecord(SourceKind.RUNTIME, _context(), (second, first))
    with pytest.raises(ValueError, match="unique and ordered"):
        ModelExecutionRecord(SourceKind.RUNTIME, _context(), (first, first))


def test_operator_call_record_freezes_nested_sequences() -> None:
    shape = [1, 128, 4096]
    tensors = [TensorInfo(INPUT[0], shape, "bfloat16")]
    call = OperatorCallRecord(0, "rms_norm", None, tensors)

    shape[0] = 8
    tensors.append(TensorInfo(INPUT[1], (1,), "bfloat16"))

    assert isinstance(call.tensors, tuple)
    assert call.tensors[0].shape == (1, 128, 4096)
    assert len(call.tensors) == 1


def test_model_execution_record_freezes_operator_calls() -> None:
    call = OperatorCallRecord(0, "rms_norm", None, (TensorInfo(INPUT[0], (1,), "bfloat16"),))
    calls = [call]
    record = ModelExecutionRecord(SourceKind.RUNTIME, _context(), calls)

    calls.append(OperatorCallRecord(1, "linear", None, ()))

    assert isinstance(record.operator_calls, tuple)
    assert record.operator_calls == (call,)


def test_simulation_execution_artifact_freezes_operator_calls() -> None:
    call = OperatorCallRecord(0, "rms_norm", None, (TensorInfo(INPUT[0], (1,), "bfloat16"),))
    calls = [call]
    artifact = SimulationExecutionArtifact(
        schema_version="1",
        producer=ProducerInfo(package_version="0.0.0", git_revision=None, capture_backend="test"),
        run_context=_context(),
        operator_calls=calls,
    )

    calls.append(OperatorCallRecord(1, "linear", None, ()))

    assert isinstance(artifact.operator_calls, tuple)
    assert artifact.operator_calls == (call,)
