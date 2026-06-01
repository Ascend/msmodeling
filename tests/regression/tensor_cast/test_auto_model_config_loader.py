from __future__ import annotations

from types import SimpleNamespace

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


def test_modelscope_snapshot_config_only_uses_allowlist(monkeypatch) -> None:
    call: dict = {}

    def fake_snapshot_download(model_id, **kwargs):
        call["model_id"] = model_id
        call["kwargs"] = kwargs
        return "/tmp/snapshot"

    monkeypatch.setattr("modelscope.snapshot_download", fake_snapshot_download)

    result = utils._modelscope_snapshot_config_only("ZhipuAI/GLM-4.7")

    assert result == "/tmp/snapshot"
    assert call["model_id"] == "ZhipuAI/GLM-4.7"
    assert call["kwargs"]["ignore_patterns"] == utils._MODELSCOPE_WEIGHT_IGNORE_PATTERNS
