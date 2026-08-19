"""Hermetic canonical CLI lifecycle tests for FLUX.1-dev."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.diffusers import diffusers_model as _dm
from tensor_cast.diffusers.cache_agent.dit_block_cache import DiTBlockCache
from tensor_cast.layers.minimax_m3_attention import RMSNormFusedWrapper
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.runtime import Runtime

FIXTURE_ROOT = Path(__file__).parents[2] / "assets" / "model_config" / "FLUX.1-dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"


def _kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "device": "TEST_DEVICE",
        "batch_size": 1,
        "output_image_size": (64, 64),
        "text_seq_len": 8,
        "source_image_sizes": (),
        "sample_step": 3,
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


def _patched_cli(monkeypatch: pytest.MonkeyPatch, *, split_dim: int | None = None):
    from cli.inference import image_generate
    from tensor_cast.diffusers.model_resolver import DiffusersModelSelection
    from tensor_cast.model_config import RemoteSource

    selection = DiffusersModelSelection(
        repository_root=str(FIXTURE_ROOT),
        variant_path=str(FIXTURE_ROOT),
        variant_id=None,
        source=RemoteSource.huggingface,
        is_remote=False,
    )
    runtime = MagicMock()
    runtime.__enter__.return_value = runtime
    runtime.__exit__.return_value = False
    runtime.table_averages.return_value = {"steps": 3}
    built_models = []
    forwards: list[dict[str, object]] = []
    forward_inputs: list[dict[str, object]] = []
    sp_group = MagicMock()
    slices: list[tuple[tuple[int, ...], int]] = []

    def slice_input(value: torch.Tensor, dim: int) -> torch.Tensor:
        slices.append((tuple(value.shape), dim))
        return torch.narrow(value, dim=dim, start=0, length=value.shape[dim] // 2)

    sp_group.slice.side_effect = slice_input

    monkeypatch.setattr(
        image_generate.DeviceProfile,
        "all_device_profiles",
        {"TEST_DEVICE": SimpleNamespace()},
    )

    def resolve_selection(model_id: str, remote_source: str) -> DiffusersModelSelection:
        assert (remote_source, model_id) == ("huggingface", MODEL_ID)
        return selection

    monkeypatch.setattr(image_generate, "resolve_diffusers_model_selection", resolve_selection)
    monkeypatch.setattr(image_generate, "Runtime", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(image_generate, "AnalyticPerformanceModel", lambda *_args: object())
    monkeypatch.setattr(image_generate, "MemoryTracker", lambda *_args: object())
    monkeypatch.setattr(image_generate, "set_sp_group", lambda *_args: None)
    monkeypatch.setattr(image_generate, "use_custom_sdpa", lambda *_args: contextlib.nullcontext())
    monkeypatch.setattr(image_generate.time, "perf_counter", MagicMock(side_effect=lambda: 1.0))
    monkeypatch.setattr("builtins.print", MagicMock())

    def build(
        model_id: str,
        parallel_config: object,
        quant_config: object,
        dtype: object,
        *,
        remote_source: str,
        model_selection: object,
    ):
        model, model_config = _dm.build_diffusers_transformer_model(
            model_id,
            parallel_config,
            quant_config,
            dtype,
            remote_source=remote_source,
            model_selection=model_selection,
        )
        built_models.append(model)
        return model, model_config

    monkeypatch.setattr(image_generate, "build_diffusers_transformer_model", build)

    real_forward_image_model = image_generate.forward_image_model

    def forward(kind: str, model: object, inputs: dict[str, object], **kwargs: object):
        inner = model._inner
        norms = [
            getattr(block.attn, name)
            for block in (*inner.transformer_blocks, *inner.single_transformer_blocks)
            for name in (
                ("norm_q", "norm_k", "norm_added_q", "norm_added_k")
                if block in inner.transformer_blocks
                else ("norm_q", "norm_k")
            )
        ]
        assert len(norms) == 152
        assert all(isinstance(norm, RMSNormFusedWrapper) for norm in norms)
        forwards.append(kwargs)
        forward_inputs.append(inputs)

        original_inner = model._inner
        hidden_states = inputs["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        cheap_inner = _CheapFluxTransformer(hidden_states)
        model._inner = cheap_inner
        try:
            output = real_forward_image_model(kind, model, inputs, **kwargs)
        finally:
            model._inner = original_inner
        assert len(cheap_inner.kwargs) == 1
        assert set(cheap_inner.kwargs[0]) == {
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
            "return_dict",
        }
        assert cheap_inner.kwargs[0]["return_dict"] is False
        assert output.shape == hidden_states.shape
        assert output.dtype is torch.float16
        assert output.device.type == "meta"
        return output

    monkeypatch.setattr(image_generate, "forward_image_model", forward)
    if split_dim is not None:
        from tensor_cast.diffusers import diffusers_model as _dm_split

        monkeypatch.setattr(_dm_split, "get_sp_group", lambda *_args, **_kwargs: sp_group)

    return SimpleNamespace(
        module=image_generate,
        runtime=runtime,
        sp_group=sp_group,
        built_models=built_models,
        forwards=forwards,
        forward_inputs=forward_inputs,
        slices=slices,
    )


def _flux_norms(model: object) -> list[torch.nn.Module]:
    inner = model._inner
    norms = [
        norm
        for block in inner.transformer_blocks
        for norm in (
            block.attn.norm_q,
            block.attn.norm_k,
            block.attn.norm_added_q,
            block.attn.norm_added_k,
        )
    ]
    norms.extend(norm for block in inner.single_transformer_blocks for norm in (block.attn.norm_q, block.attn.norm_k))
    return norms


class _CheapFluxTransformer(torch.nn.Module):
    def __init__(self, hidden_states: torch.Tensor):
        super().__init__()
        self.hidden_states = hidden_states
        self.kwargs: list[dict[str, object]] = []

    def forward(self, **kwargs: object) -> tuple[torch.Tensor]:
        self.kwargs.append(kwargs)
        return (self.hidden_states + self.hidden_states,)


class _ConstantPerformanceModel(PerformanceModel):
    def __init__(self, device_profile: object):
        super().__init__("constant", device_profile)

    def process_op(self, op_invoke_info: object) -> PerformanceModel.Result:
        del op_invoke_info
        return PerformanceModel.Result(execution_time_s=1e-6)


def test_canonical_cli_uses_real_flux_dispatch_config_build_and_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _patched_cli(monkeypatch)

    harness.module.run_inference(MODEL_ID, **_kwargs())

    assert len(harness.built_models) == 1
    assert len(harness.forwards) == 3
    assert all(call["generated_token_count"] == 16 for call in harness.forwards)
    harness.runtime.table_averages.assert_called_once_with(group_by_input_shapes=False)


def test_ordinary_cfg_lifecycle_uses_effective_double_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_cast.diffusers import flux_image

    harness = _patched_cli(monkeypatch)
    real_apply_image_cfg = harness.module.apply_image_cfg
    real_cat = torch.cat
    prepared_inputs: list[dict[str, object]] = []
    configured_inputs: list[dict[str, object]] = []
    cat_calls: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]] = []

    def cat(tensors: tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> torch.Tensor:
        result = real_cat(tensors, dim=dim)
        cat_calls.append((tensors[0], tensors[1], dim, result))
        return result

    def apply_cfg(kind: str, inputs: dict[str, object], **kwargs: object) -> dict[str, object]:
        prepared_inputs.append(inputs)
        with monkeypatch.context() as cfg_patch:
            cfg_patch.setattr(flux_image.torch, "cat", cat)
            result = real_apply_image_cfg(kind, inputs, **kwargs)
        configured_inputs.append(result)
        return result

    monkeypatch.setattr(harness.module, "apply_image_cfg", apply_cfg)
    harness.module.run_inference(MODEL_ID, **_kwargs(use_cfg=True))

    assert len(prepared_inputs) == len(configured_inputs) == 1
    prepared = prepared_inputs[0]
    configured = configured_inputs[0]
    batch_first_names = (
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
        "guidance",
    )
    assert len(cat_calls) == len(batch_first_names)
    for name, (first, second, dim, result) in zip(batch_first_names, cat_calls, strict=True):
        assert first is prepared[name]
        assert second is prepared[name]
        assert dim == 0
        assert result is configured[name]
    assert configured["img_ids"] is prepared["img_ids"]
    assert configured["txt_ids"] is prepared["txt_ids"]

    assert len(harness.forwards) == 3
    assert len(harness.forward_inputs) == 3
    for inputs in harness.forward_inputs:
        assert inputs["hidden_states"].shape == (2, 16, 64)
        assert inputs["encoder_hidden_states"].shape == (2, 8, 4096)
        assert inputs["pooled_projections"].shape == (2, 768)
        assert inputs["timestep"].shape == inputs["guidance"].shape == (2,)
        assert inputs["img_ids"].shape == (16, 3)
        assert inputs["txt_ids"].shape == (8, 3)
        assert all(inputs[name] is configured[name] for name in batch_first_names)
        assert inputs["img_ids"] is prepared["img_ids"]
        assert inputs["txt_ids"] is prepared["txt_ids"]


def test_cfg_parallel_lifecycle_gathers_ulysses_before_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _patched_cli(monkeypatch, split_dim=1)
    gathers: list[tuple[str, int]] = []
    cfg_group = MagicMock()
    cfg_group.all_gather.side_effect = lambda output, dim: (gathers.append(("cfg", dim)) or output)
    harness.sp_group.all_gather.side_effect = lambda output, dim: (gathers.append(("ulysses", dim)) or output)
    monkeypatch.setattr(harness.module, "ParallelGroup", MagicMock(return_value=cfg_group))

    harness.module.run_inference(
        MODEL_ID,
        **_kwargs(use_cfg=True, cfg_parallel=True, world_size=4, ulysses_size=2),
    )

    assert len(harness.forwards) == 3
    assert len(harness.forward_inputs) == 3
    for inputs in harness.forward_inputs:
        assert inputs["hidden_states"].shape == (1, 8, 64)
        assert inputs["encoder_hidden_states"].shape == (1, 4, 4096)
        assert inputs["pooled_projections"].shape == (1, 768)
        assert inputs["timestep"].shape == inputs["guidance"].shape == (1,)
        assert inputs["img_ids"].shape == (8, 3)
        assert inputs["txt_ids"].shape == (4, 3)
    assert harness.slices == [
        ((1, 16, 64), 1),
        ((1, 8, 4096), 1),
        ((16, 3), 0),
        ((8, 3), 0),
    ]
    assert gathers == [("ulysses", 1), ("cfg", 0)] * 3


def test_canonical_cli_cache_compile_runtime_and_trace_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cli.inference import image_generate
    from tensor_cast.diffusers.model_resolver import DiffusersModelSelection
    from tensor_cast.model_config import RemoteSource

    selection = DiffusersModelSelection(
        repository_root=str(FIXTURE_ROOT),
        variant_path=str(FIXTURE_ROOT),
        variant_id=None,
        source=RemoteSource.huggingface,
        is_remote=False,
    )
    built_models: list[object] = []
    compile_calls: list[object] = []
    forward_models: list[object] = []
    reuse_schedule: list[bool | None] = []
    forward_kwargs: list[dict[str, object]] = []
    runtimes: list[Runtime] = []

    def resolve_selection(model_id: str, remote_source: str) -> DiffusersModelSelection:
        assert (remote_source, model_id) == ("huggingface", MODEL_ID)
        return selection

    monkeypatch.setattr(image_generate, "resolve_diffusers_model_selection", resolve_selection)
    monkeypatch.setattr(image_generate, "use_custom_sdpa", lambda *_args: contextlib.nullcontext())
    monkeypatch.setattr(image_generate, "set_sp_group", lambda *_args: None)
    backend = object()
    monkeypatch.setattr(image_generate, "get_backend", MagicMock(return_value=backend))
    monkeypatch.setattr(image_generate.time, "perf_counter", MagicMock(side_effect=(10.0, 12.5)))

    def build(
        model_id: str,
        parallel_config: object,
        quant_config: object,
        dtype: object,
        *,
        remote_source: str,
        model_selection: object,
    ):
        model, model_config = _dm.build_diffusers_transformer_model(
            model_id,
            parallel_config,
            quant_config,
            dtype,
            remote_source=remote_source,
            model_selection=model_selection,
        )
        built_models.append(model)
        return model, model_config

    monkeypatch.setattr(image_generate, "build_diffusers_transformer_model", build)

    def compile_model(model: object, **kwargs: object):
        assert kwargs == {"backend": backend, "dynamic": False, "fullgraph": True}
        norms = _flux_norms(model)
        assert len(norms) == 152
        assert all(isinstance(norm, RMSNormFusedWrapper) for norm in norms)
        if len(compile_calls) == 1:
            blocks = [
                *model._inner.transformer_blocks,
                *model._inner.single_transformer_blocks,
            ]
            assert len(blocks) == 57
            assert all(isinstance(block, DiTBlockCache) for block in blocks)
        compile_calls.append(model)
        return model

    monkeypatch.setattr(image_generate.torch, "compile", compile_model)

    real_runtime = Runtime

    def make_runtime(_performance_model: object, device_profile: object, **_kwargs: object) -> Runtime:
        runtime = real_runtime(_ConstantPerformanceModel(device_profile), device_profile)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(image_generate, "Runtime", make_runtime)

    real_forward_image_model = image_generate.forward_image_model

    def forward(
        kind: str,
        model: object,
        inputs: dict[str, object],
        *,
        generated_token_count: int,
    ) -> torch.Tensor:
        assert kind == "flux1-dev"
        assert generated_token_count == 16
        hidden_states = inputs["hidden_states"]
        encoder_hidden_states = inputs["encoder_hidden_states"]
        pooled_projections = inputs["pooled_projections"]
        img_ids = inputs["img_ids"]
        txt_ids = inputs["txt_ids"]
        timestep = inputs["timestep"]
        guidance = inputs["guidance"]
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(encoder_hidden_states, torch.Tensor)
        assert isinstance(pooled_projections, torch.Tensor)
        assert isinstance(img_ids, torch.Tensor)
        assert isinstance(txt_ids, torch.Tensor)
        assert isinstance(timestep, torch.Tensor)
        assert isinstance(guidance, torch.Tensor)
        assert hidden_states.shape == (1, 16, 64)
        assert encoder_hidden_states.shape == (1, 8, 4096)
        assert pooled_projections.shape == (1, 768)
        assert img_ids.shape == (16, 3)
        assert txt_ids.shape == (8, 3)
        assert timestep.shape == guidance.shape == (1,)
        assert hidden_states.dtype == encoder_hidden_states.dtype == pooled_projections.dtype == torch.float16
        assert img_ids.dtype == txt_ids.dtype == timestep.dtype == guidance.dtype == torch.float32
        assert all(value.device.type == "meta" for value in inputs.values() if isinstance(value, torch.Tensor))
        forward_models.append(model)
        inner = model._inner
        blocks = [*inner.transformer_blocks, *inner.single_transformer_blocks]
        if isinstance(blocks[0], DiTBlockCache):
            reuse_schedule.append(blocks[0]._state.reuse)
        else:
            reuse_schedule.append(None)

        cheap_inner = _CheapFluxTransformer(hidden_states)
        model._inner = cheap_inner
        try:
            output = real_forward_image_model(kind, model, inputs, generated_token_count=generated_token_count)
        finally:
            model._inner = inner
        assert len(cheap_inner.kwargs) == 1
        forward_kwargs.append(cheap_inner.kwargs[0])
        return output

    monkeypatch.setattr(image_generate, "forward_image_model", forward)

    trace_path = tmp_path / "flux-runtime.json"
    image_generate.run_inference(
        MODEL_ID,
        **_kwargs(
            compile_enabled=True,
            dit_cache=True,
            cache_step_range="0,1",
            cache_step_interval=2,
            cache_block_range="0,57",
            chrome_trace=str(trace_path),
        ),
    )

    assert len(built_models) == 2
    assert built_models[0] is not built_models[1]
    for model in built_models:
        tensors = [*model.parameters(), *model.buffers()]
        assert tensors
        assert all(tensor.device.type == "meta" for tensor in tensors)
    assert compile_calls == built_models
    assert forward_models == [built_models[1], built_models[1], built_models[0]]
    assert reuse_schedule == [False, True, None]
    assert len(forward_kwargs) == 3
    assert all(
        set(call)
        == {
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
            "return_dict",
        }
        and call["return_dict"] is False
        for call in forward_kwargs
    )
    assert len(runtimes) == 1
    assert len(runtimes[0].event_list) == 3

    runtime_table = runtimes[0].table_averages(group_by_input_shapes=False)
    assert capsys.readouterr().out == (
        f"Runtime execution time: 2.5s\n{runtime_table}\nChrome trace written to: {trace_path}\n"
    )

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert isinstance(trace["traceEvents"], list)
    complete_events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert len(complete_events) == 3
    assert all(event["name"] == "aten.add.Tensor" for event in complete_events)


def test_canonical_cli_does_not_export_trace_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _patched_cli(monkeypatch)
    trace_path = tmp_path / "failed-runtime.json"
    monkeypatch.setattr(
        harness.module,
        "forward_image_model",
        MagicMock(side_effect=RuntimeError("forward failed")),
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        harness.module.run_inference(
            MODEL_ID,
            **_kwargs(chrome_trace=str(trace_path)),
        )

    harness.runtime.export_chrome_trace.assert_not_called()
    assert not trace_path.exists()
