# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""Hermetic public CLI lifecycle tests for Qwen-Image-Edit Transformer workloads."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.diffusers.cache_agent.dit_block_cache import DiTBlockCache
from tensor_cast.diffusers.diffusers_model import build_diffusers_transformer_model
from tensor_cast.diffusers.model_resolver import DiffusersModelSelection
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.runtime import Runtime

FIXTURE_ROOT = Path(__file__).parents[2] / "assets" / "model_config"
CASES = (
    (
        "Qwen/Qwen-Image-Edit",
        "Qwen-Image-Edit",
        "qwen-image-edit",
        ((512, 2048),),
    ),
    (
        "Qwen/Qwen-Image-Edit-2509",
        "Qwen-Image-Edit-2509",
        "qwen-image-edit-2509",
        ((1024, 1024), (768, 1024), (512, 2048)),
    ),
    (
        "Qwen/Qwen-Image-Edit-2511",
        "Qwen-Image-Edit-2511",
        "qwen-image-edit-2511",
        ((1024, 1024),),
    ),
)


class _ConstantPerformanceModel(PerformanceModel):
    def __init__(self, device_profile: Any) -> None:
        super().__init__("constant", device_profile)

    def process_op(self, op_invoke_info: object) -> PerformanceModel.Result:
        del op_invoke_info
        return PerformanceModel.Result(execution_time_s=1e-6)


class _CheapQwenTransformer(torch.nn.Module):
    def __init__(self, *, zero_cond_t: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(guidance_embeds=False, zero_cond_t=zero_cond_t)
        self.calls: list[dict[str, Any]] = []
        self.internal_timestep_shape: tuple[int, ...] | None = None
        self.modulate_index_shape: tuple[int, int] | None = None
        self.modulate_index_values: list[list[int]] | None = None

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor]:
        self.calls.append(kwargs)
        hidden_states = cast(torch.Tensor, kwargs["hidden_states"])
        timestep = cast(torch.Tensor, kwargs["timestep"])
        img_shapes = cast(list[list[tuple[int, int, int]]], kwargs["img_shapes"])
        if self.config.zero_cond_t:
            internal_timestep = torch.cat((timestep, timestep * 0), dim=0)
            self.internal_timestep_shape = tuple(internal_timestep.shape)
            self.modulate_index_values = [
                [0] * (shape[0][1] * shape[0][2]) + [1] * sum(item[1] * item[2] for item in shape[1:])
                for shape in img_shapes
            ]
            self.modulate_index_shape = (
                len(self.modulate_index_values),
                len(self.modulate_index_values[0]),
            )
        return (hidden_states + hidden_states,)


def _run_inference_kwargs(**overrides: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "device": "TEST_DEVICE",
        "batch_size": 2,
        "output_image_size": (512, 512),
        "text_seq_len": 8,
        "source_image_sizes": (),
        "sample_step": 2,
        "use_cfg": False,
        "dtype": "float16",
        "remote_source": "huggingface",
        "quantize_linear_action": "DISABLED",
        "quantize_attention_action": "DISABLED",
        "mxfp4_group_size": 32,
        "compile_enabled": False,
        "compile_allow_graph_break": False,
        "world_size": 1,
        "ulysses_size": 1,
        "cfg_parallel": False,
        "dit_cache": False,
        "cache_step_range": None,
        "cache_step_interval": 1,
        "cache_block_range": None,
        "chrome_trace": None,
    }
    values.update(overrides)
    return values


def _patched_cli(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from cli.inference import image_generate
    from tensor_cast.diffusers import model_resolver

    selections = {
        model_id: DiffusersModelSelection(
            repository_root=str(FIXTURE_ROOT / fixture_name),
            variant_path=str(FIXTURE_ROOT / fixture_name),
            variant_id=None,
            source=None,
            is_remote=False,
        )
        for model_id, fixture_name, _kind, _sources in CASES
    }
    runtimes: list[Runtime] = []
    built_models: list[tuple[str, Any]] = []
    forward_records: list[dict[str, Any]] = []
    lifecycle: list[tuple[str, str]] = []
    cache_spec_calls: list[str] = []
    compile_calls: list[object] = []
    collectives: list[tuple[str, int]] = []
    backend = object()

    def forbidden_boundary(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden image, payload, weight, scheduler, tokenizer, or network boundary used")

    monkeypatch.setattr(
        image_generate.DeviceProfile,
        "all_device_profiles",
        {"TEST_DEVICE": SimpleNamespace()},
    )
    monkeypatch.setattr(
        image_generate,
        "resolve_diffusers_model_selection",
        lambda model_id, remote_source: (
            selections[model_id]
            if remote_source == "huggingface"
            else (_ for _ in ()).throw(AssertionError("unexpected remote source"))
        ),
    )
    monkeypatch.setattr(image_generate, "AnalyticPerformanceModel", lambda *_args: object())
    monkeypatch.setattr(image_generate, "MemoryTracker", lambda *_args: MagicMock())
    monkeypatch.setattr(
        image_generate,
        "Runtime",
        lambda _performance_model, device_profile, **_kwargs: (
            runtimes.append(Runtime(_ConstantPerformanceModel(device_profile), device_profile)) or runtimes[-1]
        ),
    )
    monkeypatch.setattr(image_generate, "set_sp_group", lambda *_args: None)
    monkeypatch.setattr(image_generate, "use_custom_sdpa", lambda *_args: contextlib.nullcontext())
    monkeypatch.setattr(image_generate.time, "perf_counter", lambda: 1.0)
    monkeypatch.setattr(image_generate, "get_backend", MagicMock(return_value=backend))

    def compile_model(model: Any, **kwargs: Any) -> Any:
        assert kwargs == {"backend": backend, "dynamic": False, "fullgraph": True}
        lifecycle.append(("compile", getattr(model, "model_id", "unknown")))
        compile_calls.append(model)
        return model

    monkeypatch.setattr(image_generate.torch, "compile", compile_model)

    def build(
        model_id: str,
        parallel_config: Any,
        quant_config: Any,
        dtype: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        lifecycle.append(("build", model_id))
        model, config = build_diffusers_transformer_model(model_id, parallel_config, quant_config, dtype, **kwargs)
        built_models.append((model_id, model))
        return model, config

    monkeypatch.setattr(image_generate, "build_diffusers_transformer_model", build)

    real_prepare_model = image_generate.prepare_image_model

    def prepare_model(kind: str, model: Any, config: Any) -> Any:
        lifecycle.append(("prepare", getattr(model, "model_id", kind)))
        return real_prepare_model(kind, model, config)

    monkeypatch.setattr(image_generate, "prepare_image_model", prepare_model)

    real_cache_spec = image_generate.image_cache_spec

    def cache_spec(kind: str, config: Any) -> Any:
        cache_spec_calls.append(kind)
        return real_cache_spec(kind, config)

    monkeypatch.setattr(image_generate, "image_cache_spec", cache_spec)

    from tensor_cast.diffusers.diffusers_model import DiffusersTransformerModel

    real_enable_cache = DiffusersTransformerModel.enable_dit_block_cache

    def enable_cache(model: Any, *args: Any, **kwargs: Any) -> Any:
        lifecycle.append(("replace", getattr(model, "model_id", "unknown")))
        return real_enable_cache(model, *args, **kwargs)

    monkeypatch.setattr(DiffusersTransformerModel, "enable_dit_block_cache", enable_cache)

    real_forward = image_generate.forward_image_model

    def forward(kind: str, model: Any, inputs: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        hidden_states = cast(torch.Tensor, inputs["hidden_states"])
        model_id = cast(str, model.model_id)
        inner = model._inner
        blocks = list(getattr(inner, "transformer_blocks", ()))
        cache_blocks = [block for block in blocks if isinstance(block, DiTBlockCache)]
        state = cache_blocks[0]._state if cache_blocks else None
        before_shapes = cast(list[list[tuple[int, int, int]]], inputs["img_shapes"])
        before_shapes = [[tuple(descriptor) for descriptor in descriptors] for descriptors in before_shapes]
        zero_cond_t = bool(model.model_config.model_config.get("zero_cond_t", False))
        cheap_inner = _CheapQwenTransformer(zero_cond_t=zero_cond_t)
        model._inner = cheap_inner
        try:
            output = real_forward(kind, model, inputs, **kwargs)
        finally:
            model._inner = inner
        assert len(cheap_inner.calls) == 1
        assert inputs["img_shapes"] == before_shapes
        assert output.shape == (
            hidden_states.shape[0],
            kwargs["generated_token_count"],
            hidden_states.shape[2],
        )
        forward_records.append(
            {
                "model_id": model_id,
                "kind": kind,
                "inputs": inputs,
                "call": cheap_inner.calls[0],
                "output_shape": tuple(output.shape),
                "generated_token_count": kwargs["generated_token_count"],
                "cache": state is not None,
                "cache_reuse": None if state is None else state.reuse,
                "internal_timestep_shape": cheap_inner.internal_timestep_shape,
                "modulate_index_shape": cheap_inner.modulate_index_shape,
                "modulate_index_values": cheap_inner.modulate_index_values,
            }
        )
        lifecycle.append(("forward", model_id))
        return output

    monkeypatch.setattr(image_generate, "forward_image_model", forward)

    def cfg_group(*_args: object, **_kwargs: object) -> MagicMock:
        group = MagicMock()
        group.all_gather.side_effect = lambda output, dim: collectives.append(("cfg", dim)) or output
        return group

    monkeypatch.setattr(image_generate, "ParallelGroup", cfg_group)

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", forbidden_boundary)
    monkeypatch.setattr("tensor_cast.model_hub.snapshot_huggingface_config_only", forbidden_boundary)
    monkeypatch.setattr(torch, "load", forbidden_boundary)

    for module_name in ("diffusers", "transformers"):
        module = __import__(module_name)
        for class_name in (
            "FlowMatchEulerDiscreteScheduler",
            "AutoencoderKLQwenImage",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2Tokenizer",
            "Qwen2VLProcessor",
        ):
            component = getattr(module, class_name, None)
            if component is None:
                continue
            for method_name in ("from_pretrained", "from_config"):
                monkeypatch.setattr(component, method_name, forbidden_boundary, raising=False)

    return SimpleNamespace(
        module=image_generate,
        runtimes=runtimes,
        built_models=built_models,
        forward_records=forward_records,
        lifecycle=lifecycle,
        cache_spec_calls=cache_spec_calls,
        compile_calls=compile_calls,
        collectives=collectives,
    )


def _records_for(harness: SimpleNamespace, model_id: str) -> list[dict[str, Any]]:
    return [record for record in harness.forward_records if record["model_id"] == model_id]


def test_public_cli_runs_all_canonical_variants_hermetically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _patched_cli(monkeypatch)

    for model_id, _fixture_name, expected_kind, source_sizes in CASES:
        harness.module.run_inference(
            model_id,
            **_run_inference_kwargs(source_image_sizes=source_sizes, sample_step=2),
        )
        records = _records_for(harness, model_id)
        assert len([item for item in harness.built_models if item[0] == model_id]) == 1
        assert len(records) == 2
        assert all(record["kind"] == expected_kind for record in records)
        assert all(record["cache"] is False for record in records)
        assert all(record["output_shape"][1] == record["generated_token_count"] for record in records)
        assert all(record["inputs"]["hidden_states"].device.type == "meta" for record in records)
        assert all(record["inputs"]["encoder_hidden_states_mask"].dtype is torch.bool for record in records)
        assert all(
            set(record["call"])
            == {
                "hidden_states",
                "encoder_hidden_states",
                "encoder_hidden_states_mask",
                "timestep",
                "guidance",
                "img_shapes",
                "attention_kwargs",
                "return_dict",
            }
            and record["call"]["return_dict"] is False
            for record in records
        )

    assert harness.cache_spec_calls == []
    assert len(harness.runtimes) == 3
    assert all(runtime.event_list for runtime in harness.runtimes)
    assert all(runtime.total_execution_time_s()["constant"] > 0 for runtime in harness.runtimes)
    for _model_id, model in harness.built_models:
        assert all(tensor.device.type == "meta" for tensor in model.parameters())


def test_public_cli_compile_matrix_covers_each_variant_without_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _patched_cli(monkeypatch)

    for compile_enabled in (False, True):
        for model_id, _fixture_name, expected_kind, source_sizes in CASES:
            harness.module.run_inference(
                model_id,
                **_run_inference_kwargs(
                    source_image_sizes=source_sizes,
                    sample_step=1,
                    compile_enabled=compile_enabled,
                ),
            )
            records = _records_for(harness, model_id)
            assert records[-1]["kind"] == expected_kind
            assert records[-1]["output_shape"][1] == records[-1]["generated_token_count"]

    assert len(harness.compile_calls) == 3
    assert [model.model_id for model in harness.compile_calls] == [case[0] for case in CASES]


def test_public_original_cache_compile_lifecycle_without_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = CASES[0][0]
    harness = _patched_cli(monkeypatch)

    harness.module.run_inference(
        model_id,
        **_run_inference_kwargs(
            source_image_sizes=CASES[0][3],
            use_cfg=False,
            sample_step=2,
            compile_enabled=True,
            dit_cache=True,
            cache_step_range="0,1",
            cache_step_interval=2,
            cache_block_range="0,2",
        ),
    )

    records = _records_for(harness, model_id)
    assert len(records) == 2
    assert all(record["cache"] for record in records)
    assert [record["cache_reuse"] for record in records] == [False, True]
    assert all(record["inputs"]["hidden_states"].shape[0] == 2 for record in records)
    assert all(len(record["inputs"]["img_shapes"]) == 2 for record in records)
    assert all(len(record["inputs"]["img_shapes"][0]) == 2 for record in records)
    assert harness.cache_spec_calls == ["qwen-image-edit"]
    assert len(harness.built_models) == 2
    assert len(harness.compile_calls) == 2
    assert [name for name, _ in harness.lifecycle][:7] == [
        "build",
        "prepare",
        "build",
        "prepare",
        "replace",
        "compile",
        "compile",
    ]


def test_public_2511_ordinary_cfg_cache_preserves_metadata_and_zero_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = CASES[2][0]
    harness = _patched_cli(monkeypatch)

    harness.module.run_inference(
        model_id,
        **_run_inference_kwargs(
            source_image_sizes=CASES[2][3],
            batch_size=2,
            use_cfg=True,
            sample_step=2,
            dit_cache=True,
            cache_step_range="0,1",
            cache_step_interval=2,
            cache_block_range="0,2",
        ),
    )

    records = _records_for(harness, model_id)
    assert len(records) == 2
    assert all(record["cache"] for record in records)
    assert [record["cache_reuse"] for record in records] == [False, True]
    for record in records:
        inputs = record["inputs"]
        assert inputs["hidden_states"].shape[0] == 4
        assert inputs["encoder_hidden_states"].shape[0] == 4
        assert inputs["encoder_hidden_states_mask"].shape == (4, 8)
        assert len(inputs["img_shapes"]) == 4
        assert all(len(descriptors) == 2 for descriptors in inputs["img_shapes"])
        assert inputs["condition_image_sizes"] == ((384, 384),)
        assert inputs["img_shapes"][0] is not inputs["img_shapes"][2]
        assert record["internal_timestep_shape"] == (8,)
        assert record["modulate_index_shape"][0] == 4
        modulate_index_values = record["modulate_index_values"]
        assert all(values == modulate_index_values[0] for values in modulate_index_values)
        assert modulate_index_values[0][0] == 0
        assert modulate_index_values[0][-1] == 1


def test_public_2509_cfg_cache_update_reuse_preserves_prefix_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = CASES[1][0]
    source_sizes = CASES[1][3]
    harness = _patched_cli(monkeypatch)
    common = {
        "source_image_sizes": source_sizes,
        "batch_size": 2,
        "use_cfg": True,
        "sample_step": 3,
    }

    harness.module.run_inference(model_id, **_run_inference_kwargs(**common))
    baseline_records = _records_for(harness, model_id)
    harness.module.run_inference(
        model_id,
        **_run_inference_kwargs(
            **common,
            dit_cache=True,
            cache_step_range="0,1",
            cache_step_interval=2,
            cache_block_range="10,13",
        ),
    )
    cached_records = _records_for(harness, model_id)[3:]

    assert len(baseline_records) == len(cached_records) == 3
    assert [record["output_shape"] for record in baseline_records] == [
        record["output_shape"] for record in cached_records
    ]
    assert [record["cache_reuse"] for record in cached_records] == [False, True, None]
    assert all(record["cache"] for record in cached_records[:2])
    assert cached_records[2]["cache"] is False
    for record in cached_records:
        inputs = record["inputs"]
        assert inputs["hidden_states"].shape[0] == 4
        assert inputs["encoder_hidden_states"].shape[0] == 4
        assert inputs["encoder_hidden_states_mask"].shape == (4, 8)
        assert len(inputs["img_shapes"]) == 4
        assert inputs["condition_image_sizes"] == ((384, 384), (320, 448), (192, 768))
        assert inputs["img_shapes"][0] is not inputs["img_shapes"][1]
    assert harness.cache_spec_calls == ["qwen-image-edit-2509"]
    assert len(harness.built_models) == 3
    assert [name for name, _ in harness.lifecycle if name in {"build", "prepare", "replace"}][-5:] == [
        "build",
        "prepare",
        "build",
        "prepare",
        "replace",
    ]


def test_public_2511_cfg_parallel_cache_compile_runtime_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_id = CASES[2][0]
    trace_path = tmp_path / "qwen-2511-runtime.json"
    harness = _patched_cli(monkeypatch)

    harness.module.run_inference(
        model_id,
        **_run_inference_kwargs(
            source_image_sizes=CASES[2][3],
            use_cfg=True,
            cfg_parallel=True,
            world_size=2,
            ulysses_size=1,
            sample_step=2,
            compile_enabled=True,
            dit_cache=True,
            cache_step_range="0,1",
            cache_step_interval=2,
            cache_block_range="0,2",
            chrome_trace=str(trace_path),
        ),
    )

    records = _records_for(harness, model_id)
    assert len(records) == 2
    assert all(record["inputs"]["hidden_states"].shape[0] == 2 for record in records)
    assert [record["cache_reuse"] for record in records] == [False, True]
    assert all(record["internal_timestep_shape"] == (4,) for record in records)
    assert all(record["modulate_index_shape"][0] == 2 for record in records)
    assert all(record["modulate_index_values"][0] == record["modulate_index_values"][1] for record in records)
    assert harness.collectives == [("cfg", 0), ("cfg", 0)]
    assert len(harness.compile_calls) == 2
    assert [name for name, _ in harness.lifecycle][:5] == [
        "build",
        "prepare",
        "build",
        "prepare",
        "replace",
    ]
    assert [name for name, _ in harness.lifecycle][5:7] == ["compile", "compile"]

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert isinstance(trace["traceEvents"], list)
    assert [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert harness.runtimes[0].total_execution_time_s()["constant"] > 0


def test_public_qwen_validation_failures_happen_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _patched_cli(monkeypatch)
    model_id = CASES[0][0]

    with pytest.raises(ValueError, match="expected exactly 1 source image"):
        harness.module.run_inference(model_id, **_run_inference_kwargs(source_image_sizes=()))
    assert harness.runtimes == []

    with pytest.raises(ValueError, match="Qwen.*Ulysses.*U=1"):
        harness.module.run_inference(
            model_id,
            **_run_inference_kwargs(source_image_sizes=CASES[0][3], world_size=2, ulysses_size=2),
        )
    assert harness.runtimes == []


def test_public_qwen_runtime_failure_does_not_export_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _patched_cli(monkeypatch)
    trace_path = tmp_path / "qwen-failed-runtime.json"
    monkeypatch.setattr(
        harness.module,
        "forward_image_model",
        MagicMock(side_effect=RuntimeError("forward failed")),
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        harness.module.run_inference(
            CASES[0][0],
            **_run_inference_kwargs(source_image_sizes=CASES[0][3], chrome_trace=str(trace_path)),
        )

    assert not trace_path.exists()
