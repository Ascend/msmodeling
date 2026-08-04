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
from types import SimpleNamespace

import pytest

from tools.model_diagnostics.domain import (
    ExecutionPhase,
    ModelRunContext,
    ParallelContext,
    ProducerInfo,
    TensorDirection,
)
from tools.model_diagnostics.sources.runtime_capture import RuntimeArtifactCapture


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: str


class _FakeOp:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name


def _context() -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=2,
        parallel=ParallelContext(),
        model_config={"num_hidden_layers": 36},
        quantization_config={},
    )


def _event(name: str, args: tuple, kwargs: dict, out: object) -> SimpleNamespace:
    invocation = SimpleNamespace(func=_FakeOp(name), args=args, kwargs=kwargs, out=out)
    return SimpleNamespace(op_invoke_info=invocation)


def test_capture_preserves_event_order_and_flattens_tensor_leaves() -> None:
    hidden = _FakeTensor((1, 2, 4096), "torch.bfloat16")
    weight = _FakeTensor((4096, 4096), "torch.bfloat16")
    output = _FakeTensor((1, 2, 4096), "torch.bfloat16")
    runtime = SimpleNamespace(
        event_list=[
            _event("tensor_cast.rms_norm.default", (hidden,), {}, hidden),
            _event(
                "aten.mm.default",
                ([hidden],),
                {"weight": {"value": weight}, "ignored": 1},
                (output,),
            ),
        ]
    )

    artifact = RuntimeArtifactCapture.snapshot(
        runtime,
        run_context=_context(),
        producer=_producer(),
    )

    assert tuple(call.call_index for call in artifact.operator_calls) == (0, 1)
    assert tuple(call.operator_name for call in artifact.operator_calls) == (
        "tensor_cast.rms_norm.default",
        "aten.mm.default",
    )
    mm_tensors = artifact.operator_calls[1].tensors
    assert [(tensor.slot.direction, tensor.slot.index) for tensor in mm_tensors] == [
        (TensorDirection.INPUT, 0),
        (TensorDirection.INPUT, 1),
        (TensorDirection.OUTPUT, 0),
    ]
    assert [tensor.shape for tensor in mm_tensors] == [hidden.shape, weight.shape, output.shape]
    assert all(tensor.dtype == "bfloat16" for tensor in mm_tensors)


def test_capture_records_operator_without_tensor_leaves() -> None:
    runtime = SimpleNamespace(event_list=[_event("tensor_cast.barrier.default", (1,), {}, None)])

    artifact = RuntimeArtifactCapture.snapshot(runtime, run_context=_context(), producer=_producer(git_revision=None))

    assert artifact.operator_calls[0].tensors == ()


def test_capture_fails_fast_on_malformed_runtime_event() -> None:
    runtime = SimpleNamespace(
        event_list=[SimpleNamespace(op_invoke_info=SimpleNamespace(args=(), kwargs={}, out=None))]
    )

    with pytest.raises(ValueError, match="func"):
        RuntimeArtifactCapture.snapshot(runtime, run_context=_context(), producer=_producer(git_revision=None))


def test_capture_does_not_mutate_runtime_invocations() -> None:
    args = ([_FakeTensor((2, 2), "float16")],)
    invocation = SimpleNamespace(func=_FakeOp("aten.mm.default"), args=args, kwargs={}, out=None)
    runtime = SimpleNamespace(event_list=[SimpleNamespace(op_invoke_info=invocation)])

    RuntimeArtifactCapture.snapshot(runtime, run_context=_context(), producer=_producer(git_revision=None))

    assert invocation.args is args


def test_capture_starts_at_explicit_runtime_invocation_boundary() -> None:
    events = [
        _event("aten.patch_initialization.default", (), {}, None),
        _event("aten.mm.default", (), {}, None),
        _event("aten.relu.default", (), {}, None),
    ]
    runtime = SimpleNamespace(
        event_list=events,
        op_invoke_infos=[event.op_invoke_info for event in events],
    )

    artifact = RuntimeArtifactCapture.snapshot(
        runtime,
        start_invocation_index=1,
        run_context=_context(),
        producer=_producer(git_revision=None),
    )

    assert tuple(call.operator_name for call in artifact.operator_calls) == (
        "aten.mm.default",
        "aten.relu.default",
    )
    assert tuple(call.call_index for call in artifact.operator_calls) == (0, 1)
    assert tuple(call.source_reference for call in artifact.operator_calls) == (
        "runtime.event_list[1]",
        "runtime.event_list[2]",
    )


@pytest.mark.parametrize("start_invocation_index", [-1, 2, True, 1.5])
def test_capture_rejects_invalid_runtime_invocation_boundary(start_invocation_index: object) -> None:
    event = _event("aten.mm.default", (), {}, None)
    runtime = SimpleNamespace(event_list=[event], op_invoke_infos=[event.op_invoke_info])

    with pytest.raises((TypeError, ValueError), match="start_invocation_index"):
        RuntimeArtifactCapture.snapshot(
            runtime,
            start_invocation_index=start_invocation_index,
            run_context=_context(),
            producer=_producer(git_revision=None),
        )


def test_real_runtime_captures_only_events_after_explicit_boundary() -> None:
    torch = pytest.importorskip("torch")
    from tensor_cast.runtime import Runtime

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    weight = torch.ones((4, 3), dtype=torch.bfloat16)

    with Runtime([], None) as runtime:
        capture_start = len(runtime.op_invoke_infos)
        output = torch.mm(x, weight)
        torch.relu(output)

    artifact = RuntimeArtifactCapture.snapshot(
        runtime,
        start_invocation_index=capture_start,
        run_context=_context(),
        producer=_producer(git_revision=None),
    )

    assert tuple(call.operator_name for call in artifact.operator_calls) == (
        "aten.mm.default",
        "aten.relu.default",
    )


def test_capture_session_owns_invocation_boundary_and_builds_artifact() -> None:
    initialization = _event("aten.patch_initialization.default", (), {}, None)
    runtime = SimpleNamespace(event_list=[], op_invoke_infos=[initialization.op_invoke_info])
    capture = RuntimeArtifactCapture.begin(runtime, run_context=_context())
    mm = _event("aten.mm.default", (), {}, None)
    runtime.op_invoke_infos.append(mm.op_invoke_info)
    runtime.event_list.extend((initialization, mm))

    artifact = capture.finish(runtime, producer=_producer())

    assert tuple(call.operator_name for call in artifact.operator_calls) == ("aten.mm.default",)
    assert artifact.producer == _producer()


def test_capture_session_rejects_a_different_runtime() -> None:
    runtime = SimpleNamespace(event_list=[], op_invoke_infos=[])
    capture = RuntimeArtifactCapture.begin(runtime, run_context=_context())
    other_runtime = SimpleNamespace(event_list=[], op_invoke_infos=[])

    with pytest.raises(ValueError, match="same runtime"):
        capture.finish(other_runtime, producer=_producer())


def test_capture_session_can_only_finish_once() -> None:
    runtime = SimpleNamespace(event_list=[], op_invoke_infos=[])
    capture = RuntimeArtifactCapture.begin(runtime, run_context=_context())
    capture.finish(runtime, producer=_producer())

    with pytest.raises(RuntimeError, match="already finished"):
        capture.finish(runtime, producer=_producer())


def test_capture_session_requires_runtime_invocations_at_begin() -> None:
    with pytest.raises(ValueError, match="op_invoke_infos"):
        RuntimeArtifactCapture.begin(SimpleNamespace(event_list=[]), run_context=_context())


def _producer(*, package_version: str = "0.2.0", git_revision: str | None = "abc1234") -> ProducerInfo:
    return ProducerInfo(
        package_version=package_version,
        git_revision=git_revision,
        capture_backend="tensor_cast.runtime_observer",
    )
