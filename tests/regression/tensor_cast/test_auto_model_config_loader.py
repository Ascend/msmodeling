"""Unit tests for ``AutoModelConfigLoader`` internals.

Exercises remote-code model-type conversion and the ModelScope config-only
allowlist using monkeypatched ``transformers``/``modelscope`` — fully offline,
no Hub access. The live-Hub integration counterpart lives in
``test_auto_model_config.py``'s network-marked ``AutoModelAndConfigRemoteTestCase``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from tensor_cast.transformers import utils
from tensor_cast.transformers.utils import AutoModelConfigLoader


def test_load_config_remote_code_converts_real_model_type(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeConfig:
        model_type = "kimi_k2"

        def to_dict(self):
            return {"model_type": "deepseek_v3"}

    class FakeNativeConfig:
        model_type = "deepseek_v3"

    class FakeAutoConfig:
        @staticmethod
        def get_config_dict(model_id):
            assert model_id == "moonshotai/Kimi-K2-Base"
            return (
                {
                    "model_type": "kimi_k2",
                    "auto_map": {"AutoConfig": "configuration_kimi.KimiConfig"},
                },
                {},
            )

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == "moonshotai/Kimi-K2-Base"
            if not kwargs.get("trust_remote_code"):
                raise ValueError("requires trust_remote_code")
            calls.append(kwargs)
            return FakeConfig()

        @staticmethod
        def for_model(model_type):
            assert model_type == "deepseek_v3"
            return SimpleNamespace(from_dict=lambda _: FakeNativeConfig())

    monkeypatch.setattr("transformers.AutoConfig", FakeAutoConfig)

    loader = AutoModelConfigLoader()
    config = loader.load_config("moonshotai/Kimi-K2-Base")

    assert calls == [{"trust_remote_code": True}]
    assert isinstance(config, FakeNativeConfig)
    assert loader.is_transformers_natively_supported is True


def test_load_config_probe_passes_trust_remote_code_false(monkeypatch) -> None:
    """The native-support probe must pass trust_remote_code=False explicitly.

    With ``False`` (rather than ``None``), transformers never enters the
    interactive y/N prompt branch when the model id resolves to a config
    that does not require remote code.
    """

    seen: list[dict] = []

    class FakeNativeConfig:
        model_type = "llama"

        def to_dict(self):
            return {"model_type": "llama"}

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.append(kwargs)
            return FakeNativeConfig()

    monkeypatch.setattr("transformers.AutoConfig", FakeAutoConfig)

    loader = AutoModelConfigLoader()
    config = loader.load_config("meta-llama/Llama-3-8B")

    assert seen == [{"trust_remote_code": False}]
    assert isinstance(config, FakeNativeConfig)
    assert loader.is_transformers_natively_supported is True


def test_load_config_converts_native_deepseek_v4_to_builtin_config(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeNativeDeepseekV4Config:
        model_type = "deepseek_v4"

        def to_dict(self):
            return {"model_type": "deepseek_v4", "hidden_size": 128}

    class FakeBuiltinDeepseekV4Config:
        model_type = "deepseek_v4"

        @classmethod
        def from_dict(cls, config_dict):
            calls.append(config_dict)
            return cls()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == "deepseek-ai/DeepSeek-V4-Flash"
            assert kwargs == {"trust_remote_code": False}
            return FakeNativeDeepseekV4Config()

    monkeypatch.setattr("transformers.AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(
        "tensor_cast.transformers.builtin_model.deepseek_v4.DeepseekV4Config",
        FakeBuiltinDeepseekV4Config,
    )

    loader = AutoModelConfigLoader()
    config = loader.load_config("deepseek-ai/DeepSeek-V4-Flash")

    assert isinstance(config, FakeBuiltinDeepseekV4Config)
    assert calls == [{"model_type": "deepseek_v4", "hidden_size": 128}]
    assert loader.is_transformers_natively_supported is True


def test_load_config_deepseek_v4_import_failure_is_not_silent(monkeypatch, caplog) -> None:
    class FakeNativeDeepseekV4Config:
        model_type = "deepseek_v4"

        def to_dict(self):
            return {"model_type": "deepseek_v4", "hidden_size": 128}

    class FakeAutoConfig:
        calls: list[dict] = []

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "deepseek-ai/DeepSeek-V4-Flash"
            cls.calls.append(kwargs)
            if kwargs == {"trust_remote_code": True}:
                raise AssertionError("deepseek_v4 builtin import failure must not silently fall back")
            return FakeNativeDeepseekV4Config()

    original_import_module = utils.importlib.import_module

    def fake_import_module(name, package=None):
        if name == ".builtin_model.deepseek_v4" and package == "tensor_cast.transformers":
            raise ImportError("synthetic missing deepseek_v4 builtin")
        return original_import_module(name, package=package)

    monkeypatch.setattr("transformers.AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(utils.importlib, "import_module", fake_import_module)

    loader = AutoModelConfigLoader()
    with caplog.at_level(logging.ERROR, logger="tensor_cast.transformers.utils"):
        with pytest.raises(ImportError, match="DeepSeek V4 native config was detected"):
            loader.load_config("deepseek-ai/DeepSeek-V4-Flash")

    assert FakeAutoConfig.calls == [{"trust_remote_code": False}]
    assert "refusing to silently fall back to remote code" in caplog.text


def test_modelscope_snapshot_config_only_uses_allowlist(monkeypatch) -> None:
    call: dict = {}

    def fake_snapshot_download(model_id, ignore_patterns=None):
        call["model_id"] = model_id
        call["kwargs"] = {"ignore_patterns": ignore_patterns}
        return "/tmp/snapshot"

    monkeypatch.setattr("modelscope.snapshot_download", fake_snapshot_download)

    result = utils._modelscope_snapshot_config_only("ZhipuAI/GLM-4.7")

    assert result == "/tmp/snapshot"
    assert call["model_id"] == "ZhipuAI/GLM-4.7"
    assert call["kwargs"]["ignore_patterns"] == utils._MODELSCOPE_WEIGHT_IGNORE_PATTERNS
