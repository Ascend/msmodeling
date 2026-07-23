import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from tensor_cast.diffusers import model_resolver, pipeline_metadata
from tensor_cast.model_config import RemoteSource


def test_resolve_diffusers_model_path_returns_local_directory_without_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_model = tmp_path / "local_model"
    (local_model / "transformer").mkdir(parents=True)

    def fail_hf(_model_id: str) -> str:
        raise AssertionError("Hugging Face should not be called for local paths")

    def fail_ms(_model_id: str) -> str:
        raise AssertionError("ModelScope should not be called for local paths")

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", fail_hf)
    monkeypatch.setattr(model_resolver, "snapshot_modelscope_config_only", fail_ms)

    result = model_resolver.resolve_diffusers_model_path(str(local_model), "unknown")

    assert result == str(local_model)


def test_resolve_diffusers_model_path_rejects_missing_explicit_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_model = tmp_path / "missing_model"

    def fail_hf(_model_id: str) -> str:
        raise AssertionError("Hugging Face should not be called for explicit local paths")

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", fail_hf)

    with pytest.raises(ValueError, match="local path"):
        model_resolver.resolve_diffusers_model_path(str(missing_model), RemoteSource.huggingface)


def test_resolve_diffusers_model_path_downloads_huggingface_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_hf(model_id: str) -> str:
        calls.append(model_id)
        return "/cache/hf/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", fake_hf)

    result = model_resolver.resolve_diffusers_model_path(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "huggingface",
    )

    assert result == "/cache/hf/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert calls == ["Wan-AI/Wan2.2-T2V-A14B-Diffusers"]


def test_resolve_diffusers_model_path_warns_for_remote_model_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_id = f"SecurityTest/DiffusersRemoteWarning-{uuid4()}"
    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", lambda _model_id: "/cache/hf/model")

    result = model_resolver.resolve_diffusers_model_path(model_id, "huggingface")

    warning = capsys.readouterr().err
    assert result == "/cache/hf/model"
    assert model_id in warning
    assert "trust_remote_code=True" in warning


def test_resolve_diffusers_model_path_does_not_emit_info_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        model_resolver,
        "snapshot_huggingface_config_only",
        lambda _model_id: "/cache/hf/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    caplog.set_level(logging.INFO, logger="tensor_cast.diffusers.model_resolver")

    result = model_resolver.resolve_diffusers_model_path(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "huggingface",
    )

    assert result == "/cache/hf/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert "resolved to config-only snapshot path" not in caplog.text


def test_resolve_diffusers_model_selection_preserves_snapshot_root_and_selected_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    target_subfolder = snapshot_root / "transformer" / "480p_t2v_distilled"
    target_subfolder.mkdir(parents=True)

    monkeypatch.setattr(
        model_resolver,
        "snapshot_huggingface_config_only",
        lambda _model_id: str(snapshot_root),
    )

    selection = model_resolver.resolve_diffusers_model_selection(
        "tencent/HunyuanVideo-1.5/transformer/480p_t2v_distilled",
        "huggingface",
    )

    assert selection.repository_root == str(snapshot_root)
    assert selection.variant_path == str(target_subfolder)
    assert selection.variant_id == "transformer/480p_t2v_distilled"
    assert selection.is_remote


@pytest.mark.parametrize("model_id", ("tencent/HunyuanVideo-1.5", "Tencent-Hunyuan/HunyuanVideo-1.5"))
def test_resolve_diffusers_model_selection_rejects_raw_tencent_root(
    model_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"_class_name": "HunyuanVideo_1_5_Pipeline"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", lambda _model_id: str(tmp_path))

    with pytest.raises(ValueError, match="canonical Diffusers checkpoint"):
        model_resolver.resolve_diffusers_model_selection(model_id, "huggingface")


def test_resolve_diffusers_model_selection_rejects_raw_local_tencent_root(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"_class_name": "HunyuanVideo_1_5_Pipeline"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical Diffusers checkpoint"):
        model_resolver.resolve_diffusers_model_selection(str(tmp_path), "huggingface")


def test_resolve_diffusers_model_selection_rejects_direct_local_hunyuan_variant(tmp_path: Path) -> None:
    variant = tmp_path / "transformer" / "480p_t2v_distilled"
    variant.mkdir(parents=True)
    (variant / "config.json").write_text(
        json.dumps({"_class_name": "HunyuanVideo15Transformer3DModel"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical checkpoint root.*model_index.json"):
        model_resolver.resolve_diffusers_model_selection(str(variant), "huggingface")


def test_resolve_hunyuanvideo15_pipeline_metadata_uses_diffusers_profile() -> None:
    manifest = model_resolver.DiffusersPipelineManifest(
        config_path="/snapshot/model_index.json",
        config={
            "_class_name": "HunyuanVideo15Pipeline",
            "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
        },
        format="diffusers",
    )
    transformer_config = {
        "_class_name": "HunyuanVideo15Transformer3DModel",
        "image_embed_dim": 1152,
        "task_type": "t2v",
    }

    metadata = pipeline_metadata.resolve_hunyuanvideo15_pipeline_metadata(manifest, transformer_config)

    assert metadata.contract_version == "diffusers-hunyuanvideo15-v1"
    assert metadata.vision_num_semantic_tokens == 729
    assert metadata.vision_states_dim == 1152


def test_resolve_hunyuanvideo15_pipeline_metadata_uses_transformer_image_embed_dim() -> None:
    manifest = model_resolver.DiffusersPipelineManifest(
        config_path="/snapshot/model_index.json",
        config={
            "_class_name": "HunyuanVideo15Pipeline",
            "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
        },
        format="diffusers",
    )

    metadata = pipeline_metadata.resolve_hunyuanvideo15_pipeline_metadata(
        manifest,
        {
            "_class_name": "HunyuanVideo15Transformer3DModel",
            "image_embed_dim": 1024,
            "task_type": "t2v",
        },
    )

    assert metadata.vision_num_semantic_tokens == 729
    assert metadata.vision_states_dim == 1024


@pytest.mark.parametrize(
    "transformer_component",
    (None, "HunyuanVideo15Transformer3DModel", ["diffusers", "WanTransformer3DModel"]),
)
def test_resolve_hunyuanvideo15_pipeline_metadata_rejects_invalid_transformer_component(
    transformer_component: object,
) -> None:
    manifest = model_resolver.DiffusersPipelineManifest(
        config_path="/snapshot/model_index.json",
        config={"_class_name": "HunyuanVideo15Pipeline", "transformer": transformer_component},
        format="diffusers",
    )

    with pytest.raises(ValueError, match="transformer component"):
        pipeline_metadata.resolve_hunyuanvideo15_pipeline_metadata(
            manifest,
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "image_embed_dim": 1152,
                "task_type": "t2v",
            },
        )


def test_resolve_diffusers_pipeline_manifest_rejects_missing_root_manifest(tmp_path: Path) -> None:
    selection = model_resolver.DiffusersModelSelection(
        repository_root=str(tmp_path),
        variant_path=str(tmp_path / "transformer" / "480p_t2v_distilled"),
        variant_id="transformer/480p_t2v_distilled",
        source=RemoteSource.huggingface,
        is_remote=True,
    )

    with pytest.raises(ValueError, match="expected model_index.json"):
        model_resolver.resolve_diffusers_pipeline_manifest(selection)


@pytest.mark.parametrize(
    ("manifest", "transformer_config", "match"),
    [
        (
            model_resolver.DiffusersPipelineManifest(
                config_path="/snapshot/model_index.json",
                config={"_class_name": "HunyuanVideo15Pipeline"},
                format="diffusers",
            ),
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "image_embed_dim": 1152,
                "task_type": "i2v",
            },
            "I2V variants are not supported",
        ),
        (
            model_resolver.DiffusersPipelineManifest(
                config_path="/snapshot/model_index.json",
                config={
                    "_class_name": "HunyuanVideo15Pipeline",
                    "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
                },
                format="diffusers",
            ),
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "image_embed_dim": "invalid",
                "task_type": "t2v",
            },
            "image_embed_dim",
        ),
        (
            model_resolver.DiffusersPipelineManifest(
                config_path="/snapshot/model_index.json",
                config={"_class_name": "HunyuanVideo15PipelineV2"},
                format="diffusers",
            ),
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "image_embed_dim": 1152,
                "task_type": "t2v",
            },
            "Unsupported HunyuanVideo1.5 pipeline contract",
        ),
    ],
)
def test_resolve_hunyuanvideo15_pipeline_metadata_rejects_invalid_contracts(
    manifest: model_resolver.DiffusersPipelineManifest,
    transformer_config: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        pipeline_metadata.resolve_hunyuanvideo15_pipeline_metadata(manifest, transformer_config)


def test_resolve_diffusers_model_path_allows_huggingface_repo_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    target_subfolder = snapshot_root / "transformer" / "720p_i2v_distilled_sparse"
    target_subfolder.mkdir(parents=True)
    calls: list[str] = []

    def fake_hf(model_id: str) -> str:
        calls.append(model_id)
        return str(snapshot_root)

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", fake_hf)

    result = model_resolver.resolve_diffusers_model_path(
        "tencent/HunyuanVideo-1.5/transformer/720p_i2v_distilled_sparse",
        "huggingface",
    )

    assert result == str(target_subfolder)
    assert calls == ["tencent/HunyuanVideo-1.5"]


def test_resolve_diffusers_model_path_downloads_modelscope_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_modelscope(model_id: str) -> str:
        calls.append(model_id)
        return "/cache/modelscope/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    monkeypatch.setattr(model_resolver, "snapshot_modelscope_config_only", fake_modelscope)

    result = model_resolver.resolve_diffusers_model_path(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "modelscope",
    )

    assert result == "/cache/modelscope/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert calls == ["Wan-AI/Wan2.2-T2V-A14B-Diffusers"]


def test_resolve_diffusers_model_path_allows_modelscope_repo_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    target_subfolder = snapshot_root / "transformer" / "720p_t2v"
    target_subfolder.mkdir(parents=True)
    calls: list[str] = []

    def fake_modelscope(model_id: str) -> str:
        calls.append(model_id)
        return str(snapshot_root)

    monkeypatch.setattr(model_resolver, "snapshot_modelscope_config_only", fake_modelscope)

    result = model_resolver.resolve_diffusers_model_path(
        "Tencent-Hunyuan/HunyuanVideo-1.5/transformer/720p_t2v",
        "modelscope",
    )

    assert result == str(target_subfolder)
    assert calls == ["Tencent-Hunyuan/HunyuanVideo-1.5"]


def test_resolve_diffusers_model_path_rejects_missing_remote_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", lambda _model_id: str(snapshot_root))

    with pytest.raises(ValueError, match="does not exist in downloaded config snapshot"):
        model_resolver.resolve_diffusers_model_path(
            "tencent/HunyuanVideo-1.5/transformer/missing_variant",
            "huggingface",
        )


def test_resolve_diffusers_model_path_rejects_unsafe_remote_subfolder() -> None:
    with pytest.raises(ValueError, match="'<namespace>/<repo>/<subfolder>'"):
        model_resolver.resolve_diffusers_model_path("repo/model/../escape", "huggingface")


def test_resolve_diffusers_model_path_rejects_unknown_remote_source() -> None:
    with pytest.raises(ValueError, match="remote_source must be one of"):
        model_resolver.resolve_diffusers_model_path("repo/model", "unknown")


def test_resolve_diffusers_model_path_wraps_remote_download_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_hf(model_id: str) -> str:
        raise RuntimeError(f"network denied for {model_id}")

    monkeypatch.setattr(model_resolver, "snapshot_huggingface_config_only", fail_hf)

    with pytest.raises(RuntimeError) as exc_info:
        model_resolver.resolve_diffusers_model_path(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            RemoteSource.huggingface,
        )

    message = str(exc_info.value)
    assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in message
    assert "huggingface" in message
    assert "not a local directory" in message
    assert "automatic" in message
    assert "Download the Diffusers model directory manually" in message
