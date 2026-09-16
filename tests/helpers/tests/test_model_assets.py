from pathlib import Path

import pytest

from tests.helpers.model_assets import vendored_model_config_path, vendored_preprocessor_config_path


@pytest.mark.parametrize(
    "model_id",
    (
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3.5-397B-A17B",
        "XiaomiMiMo/MiMo-V2-Flash",
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


def test_vendored_preprocessor_config_path_unknown_model_returns_none() -> None:
    assert vendored_preprocessor_config_path("unknown/Model") is None
