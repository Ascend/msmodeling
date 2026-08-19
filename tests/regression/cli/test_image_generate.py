"""Regression tests for the Core image-generation CLI contract."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensor_cast.diffusers.cache_agent import CacheState
from tensor_cast.model_config import DiffusersConfig
from tests.helpers.cli_runner import run_module_main


def minimum_args() -> list[str]:
    return [
        "org/fake",
        "--batch-size",
        "1",
        "--output-image-size",
        "512",
        "512",
        "--text-seq-len",
        "32",
    ]


def test_help_exposes_public_contract_without_legacy_flags() -> None:
    result = run_module_main("cli.inference.image_generate", ["--help"])
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "Description:" in result.stdout
    assert "Prompt encoding, VAE, scheduler, and image I/O are excluded." in help_text
    for option in (
        "--batch-size",
        "--output-image-size",
        "--text-seq-len",
        "--sample-step",
        "--mxfp4-group-size",
        "--use-cfg",
        "--source-image-size",
        "--num-devices",
        "--ulysses-size",
        "--cfg-parallel",
        "--dit-cache",
        "--cache-step-range",
        "--cache-step-interval",
        "--cache-block-range",
        "--chrome-trace-file",
        "--model-id",
        "--log-level",
    ):
        assert option in result.stdout
    for forbidden in (
        "--negative-text-seq-len",
        "--prompt",
        "--seed",
        "--frame-num",
        "--world-size",
        "--model_id",
        "--debug",
        "--log-file",
    ):
        assert forbidden not in result.stdout
    assert "--chrome-trace" not in result.stdout.replace("--chrome-trace-file", "")


def test_startup_prints_mindstudio_logo() -> None:
    with patch("cli.inference.image_generate.run_inference") as runner:
        result = run_module_main("cli.inference.image_generate", minimum_args())

    assert result.returncode == 0
    assert "MindStudio" in result.stderr
    assert "THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE" in result.stderr
    assert result.stdout == ""
    runner.assert_called_once()


@pytest.mark.parametrize(
    ("extra_args", "error"),
    [
        (
            ["--output-image-size", "256", "256"],
            "--output-image-size must be provided exactly once",
        ),
        (["--cfg-parallel"], "cfg_parallel requires use_cfg"),
        (["--world-size", "2"], "world_size must equal 1"),
        (
            ["--use-cfg", "--cfg-parallel", "--world-size", "4", "--ulysses-size", "3"],
            "world_size must equal 6",
        ),
        (
            ["--dit-cache", "--cache-step-interval", "2"],
            "--cache-step-range is required",
        ),
        (["--cache-step-interval", "2"], "requires --dit-cache"),
    ],
)
def test_argument_validation_precedes_runner(extra_args: list[str], error: str) -> None:
    with patch("cli.inference.image_generate.run_inference") as runner:
        result = run_module_main("cli.inference.image_generate", minimum_args() + extra_args)
    assert result.returncode == 2
    assert error in result.stderr
    assert result.stdout == ""
    runner.assert_not_called()


def test_unsupported_core_model_fails_without_success_lifecycle() -> None:
    with patch(
        "cli.inference.image_generate.run_inference",
        side_effect=ValueError("unsupported"),
    ) as runner:
        result = run_module_main("cli.inference.image_generate", minimum_args())
    assert result.returncode == 1
    assert "unsupported" in result.stderr
    assert result.stdout == ""
    runner.assert_called_once()


def _run_inference_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "device": "TEST_DEVICE",
        "batch_size": 1,
        "output_image_size": (512, 512),
        "text_seq_len": 32,
        "source_image_sizes": (),
        "sample_step": 1,
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
    kwargs.update(overrides)
    return kwargs


def _patch_successful_core_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: list[MagicMock] | None = None,
    split_dim: int | None = None,
    events: list[str] | None = None,
) -> SimpleNamespace:
    from cli.inference import image_generate

    selection = SimpleNamespace(variant_path="/reviewed/model", is_remote=False)
    config = DiffusersConfig(image_dispatch_validated=True)
    config.transformer_config = SimpleNamespace()
    quant_config = SimpleNamespace(attention_configs={-1: "attention-config"})
    runtime = MagicMock()
    runtime.__enter__.return_value = runtime
    runtime.__exit__.return_value = False
    runtime.table_averages.return_value = {"ok": True}
    lifecycle_events = events if events is not None else []
    model_list = models or [MagicMock()]
    model_labels = {id(model): str(model._mock_name or "model") for model in model_list}

    monkeypatch.setattr(
        image_generate.DeviceProfile,
        "all_device_profiles",
        {"TEST_DEVICE": SimpleNamespace()},
    )
    monkeypatch.setattr(image_generate, "resolve_diffusers_model_selection", lambda *_args: selection)
    monkeypatch.setattr(
        image_generate,
        "resolve_diffusers_pipeline_manifest",
        lambda _selection: SimpleNamespace(),
    )
    build_model = MagicMock(side_effect=[(model, config) for model in model_list])
    monkeypatch.setattr(image_generate, "build_diffusers_transformer_model", build_model)
    monkeypatch.setattr(image_generate, "resolve_image_model_kind", lambda *_args: "test-kind")
    monkeypatch.setattr(image_generate, "validate_image_config", lambda *_args: None)
    create_quant_config = MagicMock(return_value=quant_config)
    monkeypatch.setattr(image_generate, "create_quant_config", create_quant_config)
    monkeypatch.setattr(image_generate, "prepare_image_model", lambda _kind, model, _config: model)
    monkeypatch.setattr(
        image_generate,
        "prepare_image_inputs",
        lambda *_args, **_kwargs: ({"hidden_states": object()}, 7),
    )
    monkeypatch.setattr(image_generate, "apply_image_cfg", lambda _kind, inputs, **_kwargs: inputs)
    monkeypatch.setattr(
        image_generate,
        "shard_image_inputs",
        lambda *_args, **_kwargs: ({"hidden_states": object()}, split_dim),
    )
    monkeypatch.setattr(image_generate, "AnalyticPerformanceModel", MagicMock())
    monkeypatch.setattr(image_generate, "MemoryTracker", MagicMock())
    monkeypatch.setattr(image_generate, "Runtime", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(image_generate.torch, "no_grad", contextlib.nullcontext)
    monkeypatch.setattr(image_generate.time, "perf_counter", MagicMock(side_effect=lambda: 1.0))
    monkeypatch.setattr("builtins.print", MagicMock())

    def forward(_kind: str, model: MagicMock, _inputs: dict[str, object], **_kwargs: object) -> torch.Tensor:
        lifecycle_events.append(f"forward:{model_labels[id(model)]}")
        return torch.zeros(1, device="meta")

    monkeypatch.setattr(image_generate, "forward_image_model", MagicMock(side_effect=forward))
    return SimpleNamespace(
        module=image_generate,
        config=config,
        build_diffusers_transformer_model=build_model,
        quant_config=quant_config,
        create_quant_config=create_quant_config,
        runtime=runtime,
        models=model_list,
        events=lifecycle_events,
    )


def test_genuine_cache_rejects_empty_window_before_second_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline])
    cache_spec = MagicMock()
    monkeypatch.setattr(harness.module, "image_cache_spec", MagicMock(return_value=cache_spec))

    with pytest.raises(ValueError, match="cache step range is empty after clamp"):
        harness.module.run_inference(
            "org/fake",
            **_run_inference_kwargs(
                sample_step=2,
                dit_cache=True,
                cache_step_range="5,10",
                cache_step_interval=2,
            ),
        )

    harness.module.build_diffusers_transformer_model.assert_called_once()
    harness.module.image_cache_spec.assert_not_called()


def test_config_only_resolution_failure_recommends_authorized_local_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.inference import image_generate

    monkeypatch.setattr(
        image_generate.DeviceProfile,
        "all_device_profiles",
        {"TEST_DEVICE": SimpleNamespace()},
    )
    monkeypatch.setattr(
        image_generate,
        "resolve_diffusers_model_selection",
        MagicMock(side_effect=RuntimeError("401 gated repository")),
    )

    with pytest.raises(RuntimeError, match="authorized local Diffusers config directory"):
        image_generate.run_inference("org/gated", **_run_inference_kwargs())


def test_manifest_resolution_failure_recommends_authorized_local_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.inference import image_generate

    selection = SimpleNamespace(variant_path="/config-only/model", is_remote=True)
    monkeypatch.setattr(
        image_generate.DeviceProfile,
        "all_device_profiles",
        {"TEST_DEVICE": SimpleNamespace()},
    )
    monkeypatch.setattr(
        image_generate,
        "resolve_diffusers_model_selection",
        MagicMock(return_value=selection),
    )
    monkeypatch.setattr(
        image_generate,
        "resolve_diffusers_pipeline_manifest",
        MagicMock(side_effect=ValueError("missing model_index.json")),
    )

    with pytest.raises(RuntimeError, match="authorized local Diffusers config directory"):
        image_generate.run_inference("org/missing-manifest", **_run_inference_kwargs())


def test_mxfp4_quantization_wires_group_size_and_tensorcast_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_cast.quantize_utils import QuantGranularity

    baseline = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline])

    harness.module.run_inference(
        "org/fake",
        **_run_inference_kwargs(
            quantize_linear_action="MXFP4",
            mxfp4_group_size=64,
        ),
    )

    harness.create_quant_config.assert_called_once_with(
        "MXFP4",
        quantize_attention_action="DISABLED",
        weight_group_size=64,
        weight_quant_granularity=QuantGranularity.PER_GROUP,
    )
    parallel_config = harness.build_diffusers_transformer_model.call_args.args[1]
    assert parallel_config.world_size == 1 and parallel_config.ulysses_size == 1


def test_remote_config_snapshot_skips_local_path_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline])
    harness.module.resolve_diffusers_model_selection = MagicMock(
        return_value=SimpleNamespace(
            variant_path="/cache/models--org--fake/snapshots/revision",
            is_remote=True,
        )
    )

    harness.module.run_inference("org/fake", **_run_inference_kwargs())

    assert harness.build_diffusers_transformer_model.call_args.kwargs["model_selection"].is_remote is True


def test_builds_model_before_preparing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline])
    events: list[str] = []
    harness.module.prepare_image_inputs = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("prepare") or ({"hidden_states": object()}, 7)
    )
    harness.module.build_diffusers_transformer_model.side_effect = lambda *_args, **_kwargs: (
        events.append("build") or (baseline, harness.config)
    )

    harness.module.run_inference("org/fake", **_run_inference_kwargs())

    assert events[:2] == ["build", "prepare"]


def test_cache_interval_one_skips_ranges_spec_and_second_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline])
    monkeypatch.setattr(
        harness.module,
        "parse_int_range",
        MagicMock(side_effect=AssertionError("range parsed")),
    )
    monkeypatch.setattr(
        harness.module,
        "image_cache_spec",
        MagicMock(side_effect=AssertionError("spec requested")),
    )

    harness.module.run_inference(
        "org/fake",
        **_run_inference_kwargs(
            dit_cache=True,
            cache_step_range="not-a-range",
            cache_step_interval=1,
            cache_block_range="not-a-range",
        ),
    )

    harness.module.build_diffusers_transformer_model.assert_called_once()
    harness.module.parse_int_range.assert_not_called()
    harness.module.image_cache_spec.assert_not_called()


def test_each_step_has_one_forward_and_collectives_run_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MagicMock(name="baseline")
    model.sp_group.all_gather.side_effect = lambda output, dim: harness.events.append("ulysses-gather") or output
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[model], split_dim=1)
    cfg_group = MagicMock()
    cfg_group.all_gather.side_effect = lambda output, dim: harness.events.append("cfg-gather") or output
    monkeypatch.setattr(harness.module, "ParallelGroup", MagicMock(return_value=cfg_group))
    monkeypatch.setattr(
        harness.module,
        "set_sp_group",
        lambda group: harness.events.append("clear-sp-group" if group is None else "set-sp-group"),
        raising=False,
    )
    sdpa = MagicMock()
    sdpa.__enter__.side_effect = lambda: harness.events.append("sdpa-enter")
    sdpa.__exit__.return_value = False
    monkeypatch.setattr(harness.module, "use_custom_sdpa", MagicMock(return_value=sdpa), raising=False)

    harness.module.run_inference(
        "org/fake",
        **_run_inference_kwargs(
            sample_step=2,
            use_cfg=True,
            world_size=4,
            ulysses_size=2,
            cfg_parallel=True,
        ),
    )

    assert harness.events == [
        "clear-sp-group",
        "sdpa-enter",
        "set-sp-group",
        "forward:baseline",
        "ulysses-gather",
        "cfg-gather",
        "set-sp-group",
        "forward:baseline",
        "ulysses-gather",
        "cfg-gather",
        "clear-sp-group",
    ]
    assert harness.module.forward_image_model.call_count == 2
    harness.module.ParallelGroup.assert_called_once_with(0, [[0, 2], [1, 3]], 4)
    harness.module.use_custom_sdpa.assert_called_once_with("attention-config")


def test_attention_group_is_cleared_after_forward_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[model])
    sp_groups: list[object | None] = []
    monkeypatch.setattr(harness.module, "set_sp_group", sp_groups.append, raising=False)
    harness.module.forward_image_model.side_effect = RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        harness.module.run_inference(
            "org/fake",
            **_run_inference_kwargs(ulysses_size=2, world_size=2),
        )

    assert sp_groups == [None, model.sp_group, None]


def test_singleton_run_clears_stale_attention_group_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MagicMock(name="baseline")
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[model])
    events: list[str] = []
    monkeypatch.setattr(
        harness.module,
        "set_sp_group",
        lambda group: events.append("clear-sp-group" if group is None else "set-sp-group"),
        raising=False,
    )
    harness.module.forward_image_model.side_effect = lambda *_args, **_kwargs: (
        events.append("forward") or torch.zeros(1, device="meta")
    )

    harness.module.run_inference("org/fake", **_run_inference_kwargs())

    assert events == ["clear-sp-group", "forward", "clear-sp-group"]


def test_cache_replacement_precedes_separate_model_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    cache_model = MagicMock(name="cache")
    cache_model.enable_dit_block_cache.side_effect = lambda *_args: (
        harness.events.append("replace-cache") or CacheState()
    )
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline, cache_model])
    monkeypatch.setattr(harness.module, "image_cache_spec", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        harness.module,
        "prepare_image_model",
        MagicMock(
            side_effect=lambda _kind, model, _config: harness.events.append(f"prepare:{model._mock_name}") or model
        ),
    )

    compiled_baseline = MagicMock(name="compiled-baseline")
    compiled_cache = MagicMock(name="compiled-cache")

    def compile_model(model: MagicMock, **_kwargs: object) -> MagicMock:
        harness.events.append(f"compile:{model._mock_name}")
        return compiled_baseline if model is baseline else compiled_cache

    monkeypatch.setattr(harness.module, "get_backend", MagicMock(return_value=object()))
    monkeypatch.setattr(harness.module.torch, "compile", MagicMock(side_effect=compile_model))
    harness.module.forward_image_model.side_effect = lambda _kind, model, _inputs, **_kwargs: (
        harness.events.append(f"forward:{model._mock_name}") or torch.zeros(1, device="meta")
    )

    harness.module.run_inference(
        "org/fake",
        **_run_inference_kwargs(
            sample_step=1,
            dit_cache=True,
            cache_step_range="0,0",
            cache_step_interval=2,
            compile_enabled=True,
        ),
    )

    assert harness.events[:5] == [
        "prepare:baseline",
        "prepare:cache",
        "replace-cache",
        "compile:baseline",
        "compile:cache",
    ]
    assert harness.events[5:] == ["forward:compiled-cache"]
    assert harness.module.torch.compile.call_count == 2


def test_chrome_trace_exports_only_after_successful_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    events: list[str] = []
    harness = _patch_successful_core_lifecycle(monkeypatch, models=[baseline], events=events)
    harness.runtime.__exit__.side_effect = lambda *_args: events.append("runtime-exit") or False
    harness.runtime.export_chrome_trace.side_effect = lambda _path: events.append("trace")

    harness.module.run_inference(
        "org/fake",
        **_run_inference_kwargs(chrome_trace="/tmp/image-trace.json"),
    )

    assert events == ["forward:baseline", "runtime-exit", "trace"]

    failing = _patch_successful_core_lifecycle(monkeypatch, models=[MagicMock(name="failing")])
    failing.module.forward_image_model.side_effect = RuntimeError("runtime failed")

    with pytest.raises(RuntimeError, match="runtime failed"):
        failing.module.run_inference(
            "org/fake",
            **_run_inference_kwargs(chrome_trace="/tmp/image-trace.json"),
        )

    failing.runtime.export_chrome_trace.assert_not_called()


def test_genuine_cache_uses_inclusive_window_and_per_run_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = MagicMock(name="baseline")
    cache_model = MagicMock(name="cache")
    first_state = CacheState()
    second_state = CacheState()
    cache_model.enable_dit_block_cache.side_effect = [first_state, second_state]
    harness = _patch_successful_core_lifecycle(
        monkeypatch,
        models=[baseline, cache_model, baseline, cache_model],
    )
    monkeypatch.setattr(harness.module, "image_cache_spec", MagicMock(return_value=MagicMock()))

    for _ in range(2):
        harness.module.run_inference(
            "org/fake",
            **_run_inference_kwargs(
                sample_step=3,
                dit_cache=True,
                cache_step_range="1,2",
                cache_step_interval=2,
            ),
        )

    assert harness.events == [
        "forward:baseline",
        "forward:cache",
        "forward:cache",
        "forward:baseline",
        "forward:cache",
        "forward:cache",
    ]
    assert first_state is not second_state
    assert second_state.reuse is True
