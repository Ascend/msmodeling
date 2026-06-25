import logging
import sys
import types

import pytest

from tensor_cast import model_hub

EXPECTED_CONFIG_JSON_ALLOW_PATTERNS = [
    "config.json",
    "**/config.json",
    "preprocessor_config.json",
]


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    monkeypatch.setitem(sys.modules, name, module)


def test_config_json_allow_patterns_cover_root_and_nested_configs() -> None:
    assert model_hub.CONFIG_JSON_ALLOW_PATTERNS == EXPECTED_CONFIG_JSON_ALLOW_PATTERNS


def test_huggingface_filter_patterns_match_root_and_nested_configs() -> None:
    from huggingface_hub.utils import filter_repo_objects

    assert list(
        filter_repo_objects(
            [
                "config.json",
                "sub/config.json",
                "preprocessor_config.json",
                "weights.safetensors",
            ],
            allow_patterns=model_hub.CONFIG_JSON_ALLOW_PATTERNS,
        )
    ) == ["config.json", "sub/config.json", "preprocessor_config.json"]


def test_huggingface_config_only_snapshot_uses_allow_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return "/cache/hf/Wan-AI/Wan2.2"

    _install_module(monkeypatch, "huggingface_hub", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_huggingface_config_only("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert result == "/cache/hf/Wan-AI/Wan2.2"
    assert calls == [
        (
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            {"allow_patterns": EXPECTED_CONFIG_JSON_ALLOW_PATTERNS},
        )
    ]


def test_huggingface_config_only_snapshot_suppresses_snapshot_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        print("Fetching 15 files")
        print("snapshot progress", file=sys.stderr)
        logging.getLogger("huggingface_hub").warning("snapshot warning")
        return f"/cache/hf/{repo_id}"

    _install_module(monkeypatch, "huggingface_hub", snapshot_download=fake_snapshot_download)
    caplog.set_level(logging.WARNING, logger="huggingface_hub")

    result = model_hub.snapshot_huggingface_config_only("tencent/HunyuanVideo-1.5")
    captured = capsys.readouterr()

    assert result == "/cache/hf/tencent/HunyuanVideo-1.5"
    assert captured.out == ""
    assert captured.err == ""
    assert "snapshot warning" not in caplog.text


def test_modelscope_config_only_snapshot_prefers_allow_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(model_id: str, allow_patterns: object = None) -> str:
        calls.append((model_id, {"allow_patterns": allow_patterns}))
        return "/cache/modelscope/wan"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_modelscope_config_only("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert result == "/cache/modelscope/wan"
    assert calls == [
        (
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            {"allow_patterns": EXPECTED_CONFIG_JSON_ALLOW_PATTERNS},
        )
    ]


def test_modelscope_config_only_snapshot_supports_allow_file_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(model_id: str, allow_file_pattern: object = None) -> str:
        calls.append((model_id, {"allow_file_pattern": allow_file_pattern}))
        return "/cache/modelscope/hunyuan"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_modelscope_config_only("Tencent-Hunyuan/HunyuanVideo-1.5")

    assert result == "/cache/modelscope/hunyuan"
    assert calls == [
        (
            "Tencent-Hunyuan/HunyuanVideo-1.5",
            {"allow_file_pattern": EXPECTED_CONFIG_JSON_ALLOW_PATTERNS},
        )
    ]


def test_modelscope_config_only_snapshot_suppresses_snapshot_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_snapshot_download(model_id: str, allow_patterns: object = None) -> str:
        print("Downloading model files")
        print("modelscope progress", file=sys.stderr)
        return "/cache/modelscope/hunyuan"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_modelscope_config_only("Tencent-Hunyuan/HunyuanVideo-1.5")
    captured = capsys.readouterr()

    assert result == "/cache/modelscope/hunyuan"
    assert captured.out == ""
    assert captured.err == ""


def test_modelscope_config_only_snapshot_prefers_explicit_allow_file_pattern_over_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(model_id: str, allow_file_pattern: object = None, **kwargs: object) -> str:
        calls.append((model_id, {"allow_file_pattern": allow_file_pattern, **kwargs}))
        return "/cache/modelscope/hunyuan"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_modelscope_config_only("Tencent-Hunyuan/HunyuanVideo-1.5")

    assert result == "/cache/modelscope/hunyuan"
    assert calls == [
        (
            "Tencent-Hunyuan/HunyuanVideo-1.5",
            {"allow_file_pattern": EXPECTED_CONFIG_JSON_ALLOW_PATTERNS},
        )
    ]


def test_modelscope_config_only_snapshot_rejects_var_keyword_only_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(model_id: str, **kwargs: object) -> str:
        calls.append((model_id, kwargs))
        return "/cache/modelscope/hunyuan"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    with pytest.raises(RuntimeError, match="config-only allowlist"):
        model_hub.snapshot_modelscope_config_only("Tencent-Hunyuan/HunyuanVideo-1.5")

    assert calls == []


def test_modelscope_config_only_snapshot_rejects_versions_without_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(model_id: str) -> str:
        return f"/cache/{model_id}"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    with pytest.raises(RuntimeError, match="config-only allowlist"):
        model_hub.snapshot_modelscope_config_only("Wan-AI/Wan2.2-T2V-A14B-Diffusers")


def test_modelscope_without_weights_reuses_ignore_pattern_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(model_id: str, ignore_file_pattern: object = None) -> str:
        calls.append((model_id, {"ignore_file_pattern": ignore_file_pattern}))
        return "/cache/modelscope/text"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_modelscope_without_weights("Qwen/Qwen3-32B")

    assert result == "/cache/modelscope/text"
    assert calls == [
        (
            "Qwen/Qwen3-32B",
            {"ignore_file_pattern": model_hub.MODELSCOPE_WEIGHT_IGNORE_PATTERNS},
        )
    ]


def test_modelscope_without_weights_prefers_explicit_ignore_file_pattern_over_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(model_id: str, ignore_file_pattern: object = None, **kwargs: object) -> str:
        calls.append((model_id, {"ignore_file_pattern": ignore_file_pattern, **kwargs}))
        return "/cache/modelscope/text"

    _install_module(monkeypatch, "modelscope", snapshot_download=fake_snapshot_download)

    result = model_hub.snapshot_modelscope_without_weights("Qwen/Qwen3-32B")

    assert result == "/cache/modelscope/text"
    assert calls == [
        (
            "Qwen/Qwen3-32B",
            {"ignore_file_pattern": model_hub.MODELSCOPE_WEIGHT_IGNORE_PATTERNS},
        )
    ]
