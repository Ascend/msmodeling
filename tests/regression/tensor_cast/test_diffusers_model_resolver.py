import logging
from pathlib import Path

import pytest

from tensor_cast.diffusers import model_resolver
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
