import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.diffusers import diffusers_model


def test_build_diffusers_transformer_model_passes_remote_source_to_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_transformer_config = object()
    fake_model_config = SimpleNamespace(transformer_config=fake_transformer_config)

    def fake_resolver(model_id: str, remote_source: str) -> str:
        calls["resolver"] = (model_id, remote_source)
        return "/cache/modelscope/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    def fake_load_config_from_file(**kwargs: object) -> object:
        calls["load"] = kwargs["model_path"]
        calls["dtype"] = kwargs["dtype"]
        return fake_model_config

    class FakeDiffusersTransformerModel:
        def __init__(self, model_id: str, transformer_config: object) -> None:
            calls["model"] = (model_id, transformer_config)

    monkeypatch.setattr(diffusers_model, "resolve_diffusers_model_path", fake_resolver)
    monkeypatch.setattr(diffusers_model, "load_config_from_file", fake_load_config_from_file)
    monkeypatch.setattr(diffusers_model, "DiffusersTransformerModel", FakeDiffusersTransformerModel)

    model, model_config = diffusers_model.build_diffusers_transformer_model(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        parallel_config=None,
        quant_config=None,
        dtype=torch.float16,
        remote_source="modelscope",
    )

    assert isinstance(model, FakeDiffusersTransformerModel)
    assert model_config is fake_model_config
    assert calls["resolver"] == ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "modelscope")
    assert calls["load"] == "/cache/modelscope/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert calls["dtype"] is torch.float16
    assert calls["model"] == ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", fake_transformer_config)


def test_load_config_from_file_accepts_single_variant_transformer_config(tmp_path: Path) -> None:
    variant_dir = tmp_path / "transformer" / "720p_i2v_distilled_sparse"
    variant_dir.mkdir(parents=True)
    config_path = variant_dir / "config.json"
    transformer_config = {
        "_class_name": "HunyuanVideo_1_5_DiffusionTransformer",
        "in_channels": 16,
        "text_embed_dim": 4096,
    }
    config_path.write_text(json.dumps(transformer_config), encoding="utf-8")

    model_config = diffusers_model.load_config_from_file(
        model_path=str(variant_dir),
        parallel_config=None,
        quant_config=None,
        quant_linear_cls=None,
        attention_cls=None,
        dtype=torch.float16,
    )

    assert model_config.model_path == str(variant_dir.resolve())
    assert model_config.transformer_config.config_json == str(config_path.resolve())
    assert model_config.transformer_config.model_config == transformer_config


def test_build_diffusers_transformer_model_surfaces_unsupported_snapshot_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_snapshot = tmp_path / "snapshot"
    empty_snapshot.mkdir()

    monkeypatch.setattr(
        diffusers_model,
        "resolve_diffusers_model_path",
        lambda _model_id, _remote_source: str(empty_snapshot),
    )

    with pytest.raises(ValueError, match="Diffusers-style model directory"):
        diffusers_model.build_diffusers_transformer_model(
            "repo/without-transformer-config",
            parallel_config=None,
            quant_config=None,
            dtype=torch.float16,
            remote_source="huggingface",
        )
