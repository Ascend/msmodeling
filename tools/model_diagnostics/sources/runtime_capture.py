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
"""Diagnostics-owned orchestration over an existing msmodeling ModelRunner."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from tools.model_diagnostics.domain import (
    INPUT,
    OUTPUT,
    ModelRunContext,
    OperatorCallRecord,
    ParallelContext,
    ProducerInfo,
    SimulationExecutionArtifact,
    TensorInfo,
    validate_expert_parallel_features,
)
from tools.model_diagnostics.errors import SourceLoadError
from tensor_cast.runtime import Runtime

_CAPTURE_BACKEND = "tensor_cast.runtime_observer"
_SCHEMA_VERSION = "1"


def _is_moe_config(config: object) -> bool:
    """Return whether the loaded HF config exposes routed-MoE fields."""

    has_routed_experts = any(
        getattr(config, key, None) is not None for key in ("num_experts", "n_routed_experts")
    )
    return has_routed_experts and getattr(config, "num_experts_per_tok", None) is not None


@lru_cache(maxsize=32)
def _model_is_moe(model_name: str) -> bool:
    """Classify a model once without retaining its mutable HF config object.

    Assumes the config loaded here matches what ModelRunner loads for the same
    model id. If a runner applies config patches before building the model,
    keep this classification and the runner's MoE execution flags in sync.
    """

    from tensor_cast.transformers.utils import AutoModelConfigLoader

    loaded_config = AutoModelConfigLoader().load_config(model_name)
    if loaded_config is None:
        raise SourceLoadError("failed to load model config before Runtime capture")
    return _is_moe_config(loaded_config)


def _validate_moe_capture_parallel(parallel: ParallelContext) -> None:
    """Validate the diagnostics MoE layout with MoE tensor parallel fixed at 1."""

    tp = parallel.tensor_parallel_size
    pp = parallel.pipeline_parallel_size
    dp = parallel.data_parallel_size
    ep = parallel.expert_parallel_size
    mdp = parallel.moe_data_parallel_size
    world_size = tp * dp * pp
    moe_world_size = world_size // pp
    if mdp * ep != moe_world_size:
        raise SourceLoadError(
            f"moe_data_parallel_size ({mdp}) * expert_parallel_size ({ep}) must equal "
            f"pipeline stage world_size ({moe_world_size}) "
            f"derived from world_size ({world_size}) / pipeline_parallel_size ({pp}). "
            f"Fix parallel.data_parallel_size / expert_parallel_size / moe_data_parallel_size "
            f"in the diagnostics profile (diagnostics does not auto-adjust). "
            f"MoE tensor parallel is fixed at 1 by this module."
        )


@dataclass
class RuntimeArtifactCapture:
    """One-shot capture boundary around one Runtime instance."""

    _runtime: object
    _start_invocation_index: int
    _run_context: ModelRunContext
    _finished: bool = False

    @classmethod
    def begin(cls, runtime: object, *, run_context: ModelRunContext) -> RuntimeArtifactCapture:
        invocations = getattr(runtime, "op_invoke_infos", None)
        if invocations is None:
            raise SourceLoadError("runtime does not expose op_invoke_infos")
        return cls(runtime, len(invocations), run_context)

    @staticmethod
    def snapshot(
        runtime: object,
        *,
        run_context: ModelRunContext,
        producer: ProducerInfo,
        start_invocation_index: int | None = None,
    ) -> SimulationExecutionArtifact:
        """Capture neutral evidence from a completed Runtime execution."""

        return _build_runtime_artifact(
            runtime,
            start_invocation_index=start_invocation_index,
            run_context=run_context,
            producer=producer,
        )

    def finish(self, runtime: object, *, producer: ProducerInfo) -> SimulationExecutionArtifact:
        if runtime is not self._runtime:
            raise SourceLoadError("capture must finish with the same runtime used at begin")
        if self._finished:
            raise RuntimeError("capture session has already finished")
        artifact = RuntimeArtifactCapture.snapshot(
            runtime,
            run_context=self._run_context,
            producer=producer,
            start_invocation_index=self._start_invocation_index,
        )
        self._finished = True
        return artifact


def _iter_tensor_leaves(value: Any) -> Iterable[Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensor_leaves(item)


def _normalize_dtype(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_info(tensor: object, *, is_output: bool, index: int) -> TensorInfo:
    try:
        shape = tuple(int(dimension) for dimension in tensor.shape)
    except (TypeError, ValueError) as error:
        raise SourceLoadError("captured tensor has an invalid shape") from error
    slot = OUTPUT[index] if is_output else INPUT[index]
    return TensorInfo(slot=slot, shape=shape, dtype=_normalize_dtype(tensor.dtype))


def _capture_tensors(invocation: object) -> tuple[TensorInfo, ...]:
    inputs = tuple(_iter_tensor_leaves((invocation.args, invocation.kwargs)))
    outputs = tuple(_iter_tensor_leaves(invocation.out))
    return tuple(_tensor_info(tensor, is_output=False, index=index) for index, tensor in enumerate(inputs)) + tuple(
        _tensor_info(tensor, is_output=True, index=index) for index, tensor in enumerate(outputs)
    )


def _build_runtime_artifact(
    runtime: object,
    *,
    start_invocation_index: int | None,
    run_context: ModelRunContext,
    producer: ProducerInfo,
) -> SimulationExecutionArtifact:
    events = getattr(runtime, "event_list", None)
    if events is None:
        raise SourceLoadError("runtime does not expose event_list")
    included_invocation_ids: set[int] | None = None
    if start_invocation_index is not None:
        invocations = getattr(runtime, "op_invoke_infos", None)
        if invocations is None:
            raise SourceLoadError("runtime does not expose op_invoke_infos")
        if isinstance(start_invocation_index, bool) or not isinstance(start_invocation_index, int):
            raise TypeError("start_invocation_index must be an integer")
        if start_invocation_index < 0:
            raise SourceLoadError("start_invocation_index must be non-negative")
        if start_invocation_index > len(invocations):
            raise SourceLoadError("start_invocation_index must not exceed the runtime invocation count")
        included_invocation_ids = {id(invocation) for invocation in invocations[start_invocation_index:]}

    calls: list[OperatorCallRecord] = []
    for source_event_index, event in enumerate(events):
        invocation = getattr(event, "op_invoke_info", None)
        if included_invocation_ids is not None and id(invocation) not in included_invocation_ids:
            continue
        func = getattr(invocation, "func", None)
        if func is None:
            raise SourceLoadError(f"runtime event {source_event_index} does not expose func")
        operator_name = str(func)
        if not operator_name.strip():
            raise SourceLoadError(f"runtime event {source_event_index} has an empty func")
        calls.append(
            OperatorCallRecord(
                call_index=len(calls),
                operator_name=operator_name,
                original_operator_name=None,
                tensors=_capture_tensors(invocation),
                source_reference=f"runtime.event_list[{source_event_index}]",
            )
        )
    return SimulationExecutionArtifact(
        schema_version=_SCHEMA_VERSION,
        producer=producer,
        run_context=run_context,
        operator_calls=tuple(calls),
    )


def capture_model_runner_artifact(
    model_runner: object,
    *,
    generate_inputs_func: Callable[..., dict[str, Any]],
    run_context: ModelRunContext,
    producer: ProducerInfo,
    requests: object | None = None,
) -> SimulationExecutionArtifact:
    """Run one capture-only forward without changing ModelRunner or Runtime behavior."""

    model = getattr(model_runner, "model", None)
    if model is None:
        raise SourceLoadError("model_runner does not expose model")
    user_input = getattr(model_runner, "user_input", None)
    if user_input is None:
        raise SourceLoadError("model_runner does not expose user_input")
    selected_requests = getattr(model_runner, "request_info_default", None) if requests is None else requests
    input_kwargs = generate_inputs_func(
        model,
        selected_requests,
        block_size=user_input.block_size,
    )

    with (
        Runtime(
            getattr(model_runner, "perf_models"),
            getattr(model_runner, "device_profile"),
        ) as runtime,
        torch.no_grad(),
    ):
        capture = RuntimeArtifactCapture.begin(runtime, run_context=run_context)
        model.forward(**input_kwargs)

    # `Runtime.event_list` is populated only inside `Runtime.__exit__` (it calls
    # repeat_op_invoke_infos()/replay_op_invoke_infos(), which is where events are
    # first appended), so finish() cannot run inside the `with` block: at that point
    # event_list is always empty. This is safe because replay reuses the exact same
    # OpInvokeInfo objects recorded in op_invoke_infos during the forward pass (it
    # only regroups them; it never copies or reconstructs them), so the id()-based
    # invocation filtering in _build_runtime_artifact keys off identities that are
    # stable across replay. See test_replay_preserves_op_invoke_info_identity for a
    # regression test that pins this invariant.
    return capture.finish(runtime, producer=producer)


def capture_artifact_for_profile(profile: object) -> SimulationExecutionArtifact:
    """Build ModelRunner from a run profile and capture one Runtime Artifact."""

    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
    from tensor_cast.core.user_config import UserInputConfig

    from tools.model_diagnostics.domain import ExecutionPhase
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    if not isinstance(profile, DiagnosticsRunProfile):
        raise TypeError("profile must be a DiagnosticsRunProfile")
    try:
        quant_action = QuantizeLinearAction(profile.quantize_linear_action)
    except ValueError as error:
        raise SourceLoadError(f"unsupported quantize_linear_action {profile.quantize_linear_action!r}") from error
    parallel = profile.parallel
    is_moe = _model_is_moe(profile.model_name)
    if is_moe:
        _validate_moe_capture_parallel(parallel)
    user_input = UserInputConfig(
        device=profile.device,
        model_id=profile.model_name,
        num_queries=profile.batch_size,
        query_len=profile.query_length,
        context_length=0 if profile.context_length is None else profile.context_length,
        do_compile=profile.do_compile,
        decode=profile.phase is ExecutionPhase.DECODE,
        num_mtp_tokens=profile.num_mtp_tokens,
        num_hidden_layers_override=profile.num_hidden_layers_override,
        world_size=(
            parallel.tensor_parallel_size * parallel.pipeline_parallel_size * parallel.data_parallel_size
        ),
        tp_size=parallel.tensor_parallel_size,
        pp_size=parallel.pipeline_parallel_size,
        dp_size=parallel.data_parallel_size,
        ep_size=parallel.expert_parallel_size,
        # Dense keeps the core-derived default; only a confirmed MoE model uses
        # diagnostics' fixed MTPt=1 contract.
        moe_tp_size=1 if is_moe else None,
        moe_dp_size=parallel.moe_data_parallel_size,
        quantize_linear_action=quant_action,
        word_embedding_tp=profile.word_embedding_tp,
        enable_redundant_experts=profile.enable_redundant_experts,
        enable_external_shared_experts=profile.enable_external_shared_experts,
        performance_model=["analytic"],
    )
    try:
        validate_expert_parallel_features(
            profile.parallel.expert_parallel_size,
            enable_external_shared_experts=profile.enable_external_shared_experts,
            enable_redundant_experts=profile.enable_redundant_experts,
        )
    except ValueError as error:
        raise SourceLoadError(str(error)) from error
    model_runner = ModelRunner(user_input)
    run_context = _run_context_after_model_load(profile, model_runner)
    return capture_model_runner_artifact(
        model_runner,
        generate_inputs_func=generate_inputs,
        run_context=run_context,
        producer=ProducerInfo(
            package_version="local",
            git_revision=None,
            capture_backend=_CAPTURE_BACKEND,
        ),
    )


def _run_context_after_model_load(profile: object, model_runner: object) -> ModelRunContext:
    from tools.model_diagnostics.specification.run_profile import DiagnosticsRunProfile

    assert isinstance(profile, DiagnosticsRunProfile)
    model_config: dict[str, object] = {}
    model = getattr(model_runner, "model", None)
    hf_config = getattr(model, "text_config", None)
    if hf_config is None:
        root_config = getattr(model, "hf_config", None)
        get_text_config = getattr(root_config, "get_text_config", None)
        hf_config = get_text_config() if callable(get_text_config) else root_config
    if hf_config is None:
        hf_config = getattr(model, "config", None)
    if hf_config is None:
        raise SourceLoadError("loaded model does not expose config for ModelRunContext")
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "vocab_size",
        "head_dim",
        "model_type",
        # MoE (category 2): routed expert count / top-k / MoE FFN width.
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "first_k_dense_replace",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_topk",
    ):
        value = getattr(hf_config, key, None)
        if value is not None:
            model_config[key] = value
    if "index_topk" not in model_config:
        topk_limit = getattr(hf_config, "topk_limit", None)
        if topk_limit is not None:
            model_config["index_topk"] = topk_limit
    if _is_moe_config(hf_config):
        # Keep the captured Context consistent with UserInputConfig: these are
        # diagnostics MoE execution settings, not generic model metadata.
        model_config["moe_tp_size"] = 1  # MoE TP is fixed at 1 (MTPt>1 unsupported).
        model_config["moe_dp_size"] = profile.parallel.moe_data_parallel_size
        model_config["enable_redundant_experts"] = profile.enable_redundant_experts
        model_config["enable_external_shared_experts"] = profile.enable_external_shared_experts
    torch_dtype = getattr(hf_config, "torch_dtype", None)
    if torch_dtype is not None:
        declared = str(torch_dtype).removeprefix("torch.")
        # Theory binds to the Runtime-executed dtype. HF card dtype is preserved
        # separately; Artifact tensor evidence is never rewritten.
        if declared in {"bfloat16", "bf16"}:
            model_config["declared_torch_dtype"] = declared
            model_config["torch_dtype"] = "float16"
        else:
            model_config["torch_dtype"] = declared
    features: list[str] = []
    if profile.do_compile:
        features.append("compiled")
    if features:
        model_config["features"] = features
    if profile.word_embedding_tp is not None:
        model_config["word_embedding_tp"] = profile.word_embedding_tp
    if profile.num_hidden_layers_override > 0:
        model_config["effective_num_hidden_layers"] = profile.num_hidden_layers_override
    else:
        layers = model_config.get("num_hidden_layers")
        if isinstance(layers, int) and layers > 0:
            model_config["effective_num_hidden_layers"] = layers
    user_input = getattr(model_runner, "user_input", None)
    num_mtp_tokens = getattr(user_input, "num_mtp_tokens", None)
    if isinstance(num_mtp_tokens, int) and not isinstance(num_mtp_tokens, bool) and num_mtp_tokens >= 0:
        model_config["num_mtp_tokens"] = num_mtp_tokens
    block_size = getattr(user_input, "block_size", None)
    if isinstance(block_size, int) and block_size > 0:
        model_config["block_size"] = block_size
    quantization_config: dict[str, object] = {
        "enabled": profile.quantize_linear_action != "DISABLED",
        "action": profile.quantize_linear_action,
    }
    if profile.quantize_linear_action in {
        "W8A8_DYNAMIC",
        "W8A8_STATIC",
        "W4A8_DYNAMIC",
        "W4A8_STATIC",
    }:
        quantization_config["linear_input_dtype"] = "int8"
    return ModelRunContext(
        model_name=profile.model_name,
        entrypoint=profile.entrypoint,
        phase=profile.phase,
        batch_size=profile.batch_size,
        query_length=profile.query_length,
        context_length=profile.context_length,
        parallel=profile.parallel,
        model_config=model_config,
        quantization_config=quantization_config,
    )
