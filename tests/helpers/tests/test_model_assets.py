from pathlib import Path

import pytest

from tests.helpers.model_assets import (
    resolve_offline_model_id,
    vendored_model_config_path,
    vendored_preprocessor_config_path,
)


@pytest.mark.parametrize(
    "model_id",
    (
        "MiniMaxAI/MiniMax-M2",
        "MiniMaxAI/MiniMax-M2.7",
        "inclusionAI/Ling-flash-2.0",
        "moonshotai/Kimi-K2-Thinking",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-397B-A17B",
        "XiaomiMiMo/MiMo-V2-Flash",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V4-Flash",
        "zai-org/GLM-4.1V-9B-Thinking",
        "zai-org/GLM-4.5V",
        "zai-org/GLM-4.7",
        "zai-org/GLM-5.1",
    ),
)
def test_vendored_model_config_path_returns_complete_local_fixture(model_id: str) -> None:
    config_dir = vendored_model_config_path(model_id)
    assert (config_dir_path := Path(config_dir)).is_dir()
    assert (config_dir_path / "config.json").is_file()


def test_vendored_model_config_path_rejects_unknown_model() -> None:
    with pytest.raises(KeyError, match="No vendored model config registered"):
        vendored_model_config_path("unknown/Model")


def test_vendored_preprocessor_config_path_for_qwen3_vl_8b() -> None:
    path = vendored_preprocessor_config_path("Qwen/Qwen3-VL-8B-Instruct")
    assert path is not None
    assert path.name == "preprocessor_config.json"
    assert path.is_file()


def test_vendored_preprocessor_config_path_for_qwen3_vl_235b() -> None:
    path = vendored_preprocessor_config_path("Qwen/Qwen3-VL-235B-A22B-Instruct")
    assert path is not None
    assert path.name == "preprocessor_config.json"
    assert path.is_file()


def test_vendored_preprocessor_config_path_unknown_model_returns_none() -> None:
    assert vendored_preprocessor_config_path("unknown/Model") is None


def test_resolve_offline_model_id_rewrites_registered_hub_id() -> None:
    resolved = resolve_offline_model_id("Qwen/Qwen3-32B")
    assert resolved == vendored_model_config_path("Qwen/Qwen3-32B")
    assert resolve_offline_model_id("unknown/Model") == "unknown/Model"
