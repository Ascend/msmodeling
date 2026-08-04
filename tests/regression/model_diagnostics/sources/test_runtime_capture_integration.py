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
from types import SimpleNamespace

import pytest

from tools.model_diagnostics.domain import ExecutionPhase, ModelRunContext, ParallelContext, ProducerInfo
from tools.model_diagnostics.sources.runtime_capture import capture_model_runner_artifact


def _context() -> ModelRunContext:
    return ModelRunContext(
        model_name="Qwen3-test",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        parallel=ParallelContext(),
        model_config={},
        quantization_config={},
    )


def _producer() -> ProducerInfo:
    return ProducerInfo("0.2.0", "abc1234+dirty", "tensor_cast.runtime_observer")


def test_capture_runner_uses_existing_model_and_original_runtime_without_model_runner_hooks() -> None:
    torch = pytest.importorskip("torch")
    x = torch.ones((2, 4), dtype=torch.float16)
    weight = torch.ones((4, 3), dtype=torch.float16)
    generated: list[tuple[object, object, int]] = []

    class _Model:
        def forward(self, *, x, weight):
            return torch.relu(torch.mm(x, weight))

    model = _Model()
    requests = [object()]
    runner = SimpleNamespace(
        model=model,
        perf_models=[],
        device_profile=None,
        request_info_default=requests,
        user_input=SimpleNamespace(block_size=128),
    )

    def generate_inputs(model_arg, requests_arg, *, block_size):
        generated.append((model_arg, requests_arg, block_size))
        return {"x": x, "weight": weight}

    artifact = capture_model_runner_artifact(
        runner,
        generate_inputs_func=generate_inputs,
        run_context=_context(),
        producer=_producer(),
    )

    assert generated == [(model, requests, 128)]
    assert tuple(call.operator_name for call in artifact.operator_calls) == (
        "aten.mm.default",
        "aten.relu.default",
    )


def test_replay_preserves_op_invoke_info_identity_so_id_based_filtering_is_safe() -> None:
    """Capture reads ``event_list`` only after ``Runtime.__exit__``.

    ``Runtime.event_list`` is populated exclusively inside ``__exit__`` (via
    ``repeat_op_invoke_infos``/``replay_op_invoke_infos``); it is always empty
    while inside the ``with Runtime(...)`` block, so ``finish()`` cannot run
    before exit. ``_build_runtime_artifact`` filters events by
    ``id(invocation)`` computed from ``op_invoke_infos`` at ``begin()`` time, so
    that filtering is only correct if replay reuses the exact same
    ``OpInvokeInfo`` objects rather than rebuilding new ones. This test pins
    that invariant directly against the real ``Runtime``.
    """

    torch = pytest.importorskip("torch")
    from tensor_cast.runtime import Runtime

    x = torch.ones((2, 4), dtype=torch.float16)
    weight = torch.ones((4, 3), dtype=torch.float16)

    with Runtime([], None) as runtime:
        assert runtime.event_list == []
        torch.mm(x, weight)
        assert runtime.event_list == []  # not populated until __exit__

    invocation_by_id = {id(invocation): invocation for invocation in runtime.op_invoke_infos}
    assert runtime.event_list
    assert all(
        id(event.op_invoke_info) in invocation_by_id
        and event.op_invoke_info is invocation_by_id[id(event.op_invoke_info)]
        for event in runtime.event_list
    )


def test_run_context_binds_theory_dtype_to_runtime_fp16_when_hf_declares_bf16() -> None:
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile
    from tools.model_diagnostics.sources.runtime_capture import _run_context_after_model_load

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen/Qwen3-8B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_language_layers=None,
        selected_stage_regions=("input", "output"),
        num_hidden_layers_override=1,
        do_compile=True,
        device="npu",
        quantize_linear_action="W8A8_DYNAMIC",
        word_embedding_tp="row",
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4096,
                intermediate_size=12288,
                num_attention_heads=32,
                num_key_value_heads=8,
                num_hidden_layers=36,
                vocab_size=151936,
                head_dim=128,
                model_type="qwen3",
                torch_dtype="bfloat16",
            )
        ),
        user_input=SimpleNamespace(num_mtp_tokens=2),
    )

    context = _run_context_after_model_load(profile, runner)

    assert context.model_config["declared_torch_dtype"] == "bfloat16"
    assert context.model_config["torch_dtype"] == "float16"
    assert context.model_config["word_embedding_tp"] == "row"
    assert context.model_config["num_mtp_tokens"] == 2
    assert context.quantization_config["enabled"] is True
    assert context.quantization_config["action"] == "W8A8_DYNAMIC"
    assert context.quantization_config["linear_input_dtype"] == "int8"

    class _Model:
        def forward(self):
            raise RuntimeError("model failed")

    runner = SimpleNamespace(
        model=_Model(),
        perf_models=[],
        device_profile=None,
        request_info_default=[],
        user_input=SimpleNamespace(block_size=128),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        capture_model_runner_artifact(
            runner,
            generate_inputs_func=lambda *_args, **_kwargs: {},
            run_context=_context(),
            producer=_producer(),
        )
