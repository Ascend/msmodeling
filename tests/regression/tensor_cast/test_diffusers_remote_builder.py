import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.diffusers import diffusers_model, diffusers_utils
from tensor_cast.model_config import DiffusersPipelineMetadata


def test_build_diffusers_transformer_model_passes_remote_source_to_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_transformer_config = object()
    fake_model_config = SimpleNamespace(transformer_config=fake_transformer_config)

    selection = diffusers_model.DiffusersModelSelection(
        repository_root="/cache/modelscope/tencent/HunyuanVideo-1.5",
        variant_path="/cache/modelscope/tencent/HunyuanVideo-1.5/transformer/480p_t2v_distilled",
        variant_id="transformer/480p_t2v_distilled",
        source=None,
        is_remote=True,
    )

    def fake_resolver(model_id: str, remote_source: str) -> object:
        calls["resolver"] = (model_id, remote_source)
        return selection

    def fake_load_config_from_file(**kwargs: object) -> object:
        calls["load"] = kwargs["model_path"]
        calls["selection"] = kwargs["model_selection"]
        calls["validate_local_path"] = kwargs["validate_local_path"]
        calls["dtype"] = kwargs["dtype"]
        return fake_model_config

    class FakeDiffusersTransformerModel:
        def __init__(self, model_id: str, transformer_config: object) -> None:
            calls["model"] = (model_id, transformer_config)

    monkeypatch.setattr(diffusers_model, "resolve_diffusers_model_selection", fake_resolver)
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
    assert calls["load"] == "/cache/modelscope/tencent/HunyuanVideo-1.5/transformer/480p_t2v_distilled"
    assert calls["selection"] is selection
    assert calls["validate_local_path"] is False
    assert calls["dtype"] is torch.float16
    assert calls["model"] == ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", fake_transformer_config)


def test_build_diffusers_transformer_model_validates_supplied_local_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    selection = diffusers_model.DiffusersModelSelection(
        repository_root=str(tmp_path),
        variant_path=str(tmp_path),
        variant_id=None,
        source=None,
        is_remote=False,
    )

    monkeypatch.setattr(
        diffusers_model,
        "load_config_from_file",
        lambda **kwargs: calls.update(kwargs) or SimpleNamespace(transformer_config=object()),
    )
    monkeypatch.setattr(diffusers_model, "DiffusersTransformerModel", lambda *args: object())

    diffusers_model.build_diffusers_transformer_model(
        "remote-looking/model-id",
        parallel_config=None,
        quant_config=None,
        dtype=torch.float16,
        model_selection=selection,
    )

    assert calls["validate_local_path"] is True
    assert calls["model_selection"] is selection


def test_build_diffusers_transformer_model_accepts_huggingface_snapshot_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_cache = tmp_path / "models--Wan-AI--Wan2.2-T2V-A14B"
    snapshot = repo_cache / "snapshots" / "revision"
    high_noise_dir = snapshot / "high_noise_model"
    high_noise_dir.mkdir(parents=True)
    blob = repo_cache / "blobs" / "config-blob"
    blob.parent.mkdir()
    blob.write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}),
        encoding="utf-8",
    )
    (high_noise_dir / "config.json").symlink_to("../../../blobs/config-blob")

    monkeypatch.setattr(
        diffusers_model,
        "resolve_diffusers_model_selection",
        lambda _model_id, _remote_source: diffusers_model.DiffusersModelSelection(
            repository_root=str(snapshot),
            variant_path=str(snapshot),
            variant_id=None,
            source=None,
            is_remote=_model_id != str(snapshot),
        ),
    )
    monkeypatch.setattr(
        diffusers_model,
        "DiffusersTransformerModel",
        lambda model_id, transformer_config: (model_id, transformer_config),
    )

    model, model_config = diffusers_model.build_diffusers_transformer_model(
        "Wan-AI/Wan2.2-T2V-A14B",
        parallel_config=None,
        quant_config=None,
        dtype=torch.bfloat16,
    )

    assert model[0] == "Wan-AI/Wan2.2-T2V-A14B"
    assert model_config.model_path == str(snapshot.resolve())
    assert model_config.transformer_config.model_config["_class_name"] == "WanTransformer3DModel"

    with pytest.raises(ValueError, match="must not contain symlinks"):
        diffusers_model.build_diffusers_transformer_model(
            str(snapshot),
            parallel_config=None,
            quant_config=None,
            dtype=torch.bfloat16,
        )


def test_load_config_from_file_rejects_legacy_tencent_hunyuanvideo15_transformer_config(
    tmp_path: Path,
) -> None:
    variant_dir = tmp_path / "transformer" / "480p_t2v_distilled"
    variant_dir.mkdir(parents=True)
    (variant_dir / "config.json").write_text(
        json.dumps({"_class_name": "HunyuanVideo_1_5_DiffusionTransformer"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical Diffusers checkpoint"):
        diffusers_model.load_config_from_file(
            model_path=str(variant_dir),
            parallel_config=None,
            quant_config=None,
            quant_linear_cls=None,
            attention_cls=None,
            dtype=torch.float16,
            model_selection=diffusers_model.DiffusersModelSelection(
                repository_root=str(tmp_path),
                variant_path=str(variant_dir),
                variant_id="transformer/480p_t2v_distilled",
                source=None,
                is_remote=False,
            ),
        )


def test_load_config_from_file_rejects_hunyuan_pipeline_with_non_hunyuan_transformer(tmp_path: Path) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}),
        encoding="utf-8",
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Pipeline",
                "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
            }
        ),
        encoding="utf-8",
    )
    selection = diffusers_model.DiffusersModelSelection(
        repository_root=str(tmp_path),
        variant_path=str(tmp_path),
        variant_id=None,
        source=None,
        is_remote=False,
    )

    with pytest.raises(ValueError, match="requires HunyuanVideo15Transformer3DModel"):
        diffusers_model.load_config_from_file(
            model_path=str(tmp_path),
            parallel_config=None,
            quant_config=None,
            quant_linear_cls=None,
            attention_cls=None,
            dtype=torch.float16,
            model_selection=selection,
        )


def test_generate_hunyuanvideo15_input_uses_pipeline_vision_contract() -> None:
    pipeline_metadata = DiffusersPipelineMetadata(
        pipeline_class="HunyuanVideo_1_5_Pipeline",
        contract_version="tencent-hunyuanvideo15-v1",
        vision_num_semantic_tokens=729,
        vision_states_dim=1152,
    )

    inputs = diffusers_utils.generate_hunyuanvideo15_input(
        batch_size=2,
        seq_lens=128,
        dtype=torch.float16,
        pipeline_metadata=pipeline_metadata,
    )

    assert isinstance(inputs["image_embeds"], diffusers_utils.SafeMetaTensor)
    assert inputs["image_embeds"].shape == (2, 729, 1152)
    assert inputs["image_embeds"].dtype is torch.float16
    assert inputs["image_embeds"].device.type == "meta"


def test_safe_meta_tensor_boolean_indexing_returns_base_meta_tensor() -> None:
    image_embeds = diffusers_utils.SafeMetaTensor((2, 729, 32), dtype=torch.float16)
    projected_image = image_embeds + torch.zeros((), device="meta", dtype=torch.float16)
    image_mask = torch.zeros((2, 729), device="meta", dtype=torch.bool)

    assert isinstance(projected_image, diffusers_utils.SafeMetaTensor)
    selected = projected_image[image_mask]
    unselected = projected_image[~image_mask]

    assert selected.device.type == "meta"
    assert unselected.device.type == "meta"
    assert type(selected) is torch.Tensor
    assert type(unselected) is torch.Tensor
    assert selected.shape == (2, 729, 32)
    assert unselected.shape == (2, 729, 32)


def test_build_diffusers_transformer_model_surfaces_unsupported_snapshot_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_snapshot = tmp_path / "snapshot"
    empty_snapshot.mkdir()

    monkeypatch.setattr(
        diffusers_model,
        "resolve_diffusers_model_selection",
        lambda _model_id, _remote_source: diffusers_model.DiffusersModelSelection(
            repository_root=str(empty_snapshot),
            variant_path=str(empty_snapshot),
            variant_id=None,
            source=None,
            is_remote=True,
        ),
    )

    with pytest.raises(ValueError, match="Diffusers-style model directory"):
        diffusers_model.build_diffusers_transformer_model(
            "repo/without-transformer-config",
            parallel_config=None,
            quant_config=None,
            dtype=torch.float16,
            remote_source="huggingface",
        )
