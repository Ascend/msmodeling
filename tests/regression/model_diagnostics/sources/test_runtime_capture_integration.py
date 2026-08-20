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


@pytest.mark.parametrize("quantization", ("W8A8_DYNAMIC", "W4A8_DYNAMIC"))
def test_run_context_binds_theory_dtype_to_runtime_fp16_when_hf_declares_bf16(quantization) -> None:
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
        quantize_linear_action=quantization,
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
    assert context.quantization_config["action"] == quantization
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


def test_dense_profile_keeps_tp_layout_independent_of_moe_defaults(monkeypatch) -> None:
    from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    captured: dict[str, object] = {}

    class _Runner:
        def __init__(self, user_input):
            captured["user_input"] = user_input
            parallel = user_input.get_parallel_config()
            captured["parallel"] = parallel
            self.user_input = user_input
            self.model = SimpleNamespace(
                config=SimpleNamespace(
                    hidden_size=1024,
                    intermediate_size=3072,
                    num_attention_heads=16,
                    num_key_value_heads=8,
                    num_hidden_layers=1,
                    vocab_size=151936,
                    head_dim=64,
                    model_type="qwen3",
                    torch_dtype="float16",
                )
            )

    monkeypatch.setattr("tensor_cast.core.model_runner.ModelRunner", _Runner)
    monkeypatch.setattr(
        "tensor_cast.transformers.utils.AutoModelConfigLoader.load_config",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen3"),
    )
    monkeypatch.setattr(
        "tools.model_diagnostics.sources.runtime_capture.capture_model_runner_artifact",
        lambda runner, **kwargs: kwargs["run_context"],
    )
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen/Qwen3-0.6B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(tensor_parallel_size=2, data_parallel_size=1),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )

    captured_result = capture_artifact_for_profile(profile)
    context = getattr(captured_result, "run_context", captured_result)

    assert isinstance(context, ModelRunContext)
    assert context.model_config["model_type"] == "qwen3"
    assert captured["user_input"].moe_tp_size is None
    assert captured["parallel"].moe_tensor_parallel_size == 2
    assert "moe_tp_size" not in context.model_config
    assert "moe_dp_size" not in context.model_config
    assert "enable_redundant_experts" not in context.model_config
    assert "enable_external_shared_experts" not in context.model_config


def test_moe_profile_records_diagnostics_moe_execution_settings() -> None:
    from tools.model_diagnostics.sources.runtime_capture import _run_context_after_model_load
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="Qwen/Qwen3-30B-A3B",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(moe_data_parallel_size=2),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=1024,
                intermediate_size=3072,
                num_attention_heads=16,
                num_key_value_heads=8,
                num_hidden_layers=1,
                vocab_size=151936,
                head_dim=64,
                model_type="qwen3_moe",
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=768,
                torch_dtype="float16",
            )
        ),
        user_input=SimpleNamespace(num_mtp_tokens=0, block_size=128),
    )

    context = _run_context_after_model_load(profile, runner)

    assert context.model_config["moe_tp_size"] == 1
    assert context.model_config["moe_dp_size"] == 2
    assert context.model_config["enable_redundant_experts"] is False
    assert context.model_config["enable_external_shared_experts"] is False


def test_run_context_normalizes_dsa_topk_internal_alias() -> None:
    from tools.model_diagnostics.sources.runtime_capture import _run_context_after_model_load
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="deepseek-ai/DeepSeek-V3.2",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )
    config = SimpleNamespace(
        hidden_size=7168,
        intermediate_size=18432,
        num_attention_heads=128,
        num_key_value_heads=128,
        num_hidden_layers=61,
        vocab_size=129280,
        model_type="deepseek_v32",
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        topk_limit=2048,
        torch_dtype="float16",
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(text_config=config),
        user_input=SimpleNamespace(num_mtp_tokens=0, block_size=128),
    )

    context = _run_context_after_model_load(profile, runner)

    assert context.model_config["index_topk"] == 2048


def test_model_runner_assertion_is_not_reclassified_as_external_shared_error(monkeypatch) -> None:
    from tools.model_diagnostics.sources.runtime_capture import capture_artifact_for_profile
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    class _FailingRunner:
        def __init__(self, _user_input):
            raise AssertionError("unrelated model initialization failure")

    monkeypatch.setattr("tensor_cast.core.model_runner.ModelRunner", _FailingRunner)
    monkeypatch.setattr(
        "tools.model_diagnostics.sources.runtime_capture._model_is_moe",
        lambda _model_name: True,
    )
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="test/moe-assertion-model",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(data_parallel_size=2, expert_parallel_size=2),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
        enable_external_shared_experts=True,
    )

    with pytest.raises(AssertionError, match="unrelated model initialization failure"):
        capture_artifact_for_profile(profile)


def test_deepseek_routed_expert_config_rejects_illegal_moe_layout_before_model_build(monkeypatch) -> None:
    from tools.model_diagnostics.errors import SourceLoadError
    from tools.model_diagnostics.sources.runtime_capture import _model_is_moe, capture_artifact_for_profile
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    model_runner_called = False

    class _Runner:
        def __init__(self, _user_input):
            nonlocal model_runner_called
            model_runner_called = True

    monkeypatch.setattr("tensor_cast.core.model_runner.ModelRunner", _Runner)
    monkeypatch.setattr(
        "tensor_cast.transformers.utils.AutoModelConfigLoader.load_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            model_type="deepseek_v32",
            n_routed_experts=256,
            num_experts_per_tok=8,
        ),
    )
    _model_is_moe.cache_clear()
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="test/deepseek-v32-illegal-layout",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=2,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(tensor_parallel_size=2),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )

    try:
        with pytest.raises(SourceLoadError, match="must equal pipeline stage world_size"):
            capture_artifact_for_profile(profile)
    finally:
        _model_is_moe.cache_clear()
    assert model_runner_called is False
