import contextlib
import importlib
import sys
import types

import pytest
import torch

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.diffusers import model_resolver


@pytest.mark.parametrize(
    "module_name",
    [
        "cli.inference.video_generate",
    ],
)
def test_video_generate_help_includes_remote_source(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", ["video_generate", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    normalized_output = output.lower()
    assert "--remote-source" in output
    assert "huggingface" in normalized_output
    assert "modelscope" in normalized_output
    assert "remote repo id" in normalized_output
    assert "subfolder" in normalized_output


def test_cli_video_generate_main_passes_remote_source_to_run_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.inference import video_generate

    captured: dict[str, object] = {}

    def fake_run_inference(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(video_generate, "print_logo", lambda: None)
    monkeypatch.setattr(video_generate, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "video_generate",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--batch-size",
            "1",
            "--seq-len",
            "128",
            "--remote-source",
            "modelscope",
        ],
    )

    video_generate.main()

    assert captured["model_id"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert captured["remote_source"] == "modelscope"


def test_cli_video_generate_run_inference_passes_remote_source_to_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.inference import video_generate

    captured_builds: list[dict[str, object]] = []
    resolver_calls: list[tuple[str, str]] = []

    class DummyRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "DummyRuntime":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def table_averages(self, *args: object, **kwargs: object) -> str:
            return "runtime table"

    class DummyModel:
        sp_group = None

        def forward(self, **kwargs: object) -> torch.Tensor:
            return torch.zeros([1], device="meta")

    model_config = types.SimpleNamespace(
        transformer_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(ulysses_size=1),
            model_config={"_class_name": "WanTransformer3DModel"},
            dtype=torch.float16,
        )
    )

    def fake_build_diffusers_transformer_model(
        model_id: str,
        parallel_config: object,
        quant_config: object,
        dtype: torch.dtype,
        remote_source: str,
        resolved_model_path: str | None = None,
    ) -> tuple[DummyModel, object]:
        captured_builds.append(
            {
                "model_id": model_id,
                "parallel_config": parallel_config,
                "quant_config": quant_config,
                "dtype": dtype,
                "remote_source": remote_source,
                "resolved_model_path": resolved_model_path,
            }
        )
        return DummyModel(), model_config

    monkeypatch.setitem(
        sys.modules,
        "tensor_cast.diffusers.diffusers_attention",
        types.SimpleNamespace(
            set_sp_group=lambda group: None,
            use_custom_sdpa=contextlib.nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensor_cast.diffusers.diffusers_model",
        types.SimpleNamespace(build_diffusers_transformer_model=fake_build_diffusers_transformer_model),
    )

    def fake_resolve_diffusers_model_path(model_id: str, remote_source: str) -> str:
        resolver_calls.append((model_id, remote_source))
        return "/cache/modelscope/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    monkeypatch.setitem(
        sys.modules,
        "tensor_cast.diffusers.model_resolver",
        types.SimpleNamespace(resolve_diffusers_model_path=fake_resolve_diffusers_model_path),
    )
    monkeypatch.setattr(video_generate, "AnalyticPerformanceModel", lambda device_profile: object())
    monkeypatch.setattr(video_generate, "MemoryTracker", lambda device_profile: object())
    monkeypatch.setattr(video_generate, "Runtime", DummyRuntime)
    monkeypatch.setattr(
        video_generate,
        "generate_diffusers_inputs",
        lambda *args, **kwargs: {"hidden_states": torch.zeros([1], device="meta")},
    )
    monkeypatch.setattr(
        video_generate,
        "process_input",
        lambda input_kwargs, model_config: (input_kwargs, None),
    )

    video_generate.run_inference(
        device="TEST_DEVICE",
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        batch_size=1,
        seq_len=128,
        sample_step=0,
        remote_source="modelscope",
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )

    assert captured_builds == [
        {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "parallel_config": captured_builds[0]["parallel_config"],
            "quant_config": captured_builds[0]["quant_config"],
            "dtype": torch.float16,
            "remote_source": "modelscope",
            "resolved_model_path": "/cache/modelscope/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        }
    ]
    assert resolver_calls == [("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "modelscope")]


def test_cli_video_generate_remote_resolution_failure_includes_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_huggingface_config_only(repo_id: str) -> str:
        raise TimeoutError("network timeout")

    monkeypatch.setattr(
        model_resolver,
        "snapshot_huggingface_config_only",
        fake_snapshot_huggingface_config_only,
    )

    with pytest.raises(RuntimeError, match=r"TimeoutError: network timeout"):
        model_resolver.resolve_diffusers_model_path("Wan-AI/Wan2.2-T2V-A14B-Diffusers")


def test_cli_video_generate_run_inference_cfg_batch_concat_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.inference import video_generate

    class DummyRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "DummyRuntime":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def table_averages(self, *args: object, **kwargs: object) -> str:
            return "runtime table"

    class DummyModel:
        sp_group = None

        def forward(self, **kwargs: object) -> torch.Tensor:
            assert kwargs["hidden_states"].shape[0] == 2
            return torch.zeros([1], device="meta")

    model_config = types.SimpleNamespace(
        transformer_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(ulysses_size=1),
            model_config={"_class_name": "WanTransformer3DModel"},
            dtype=torch.float16,
        )
    )

    monkeypatch.setitem(
        sys.modules,
        "tensor_cast.diffusers.diffusers_attention",
        types.SimpleNamespace(
            set_sp_group=lambda group: None,
            use_custom_sdpa=contextlib.nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensor_cast.diffusers.diffusers_model",
        types.SimpleNamespace(
            build_diffusers_transformer_model=lambda *args, **kwargs: (
                DummyModel(),
                model_config,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensor_cast.diffusers.model_resolver",
        types.SimpleNamespace(resolve_diffusers_model_path=lambda model_id, remote_source: model_id),
    )
    monkeypatch.setattr(video_generate, "AnalyticPerformanceModel", lambda device_profile: object())
    monkeypatch.setattr(video_generate, "MemoryTracker", lambda device_profile: object())
    monkeypatch.setattr(video_generate, "Runtime", DummyRuntime)
    monkeypatch.setattr(
        video_generate,
        "generate_diffusers_inputs",
        lambda *args, **kwargs: {"hidden_states": torch.zeros([1, 3], device="meta")},
    )
    monkeypatch.setattr(
        video_generate,
        "process_input",
        lambda input_kwargs, model_config: (input_kwargs, None),
    )

    video_generate.run_inference(
        device="TEST_DEVICE",
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        batch_size=1,
        seq_len=128,
        sample_step=1,
        use_cfg=True,
        cfg_parallel=False,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
