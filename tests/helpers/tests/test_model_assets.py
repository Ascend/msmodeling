from tests.helpers.model_assets import vendored_preprocessor_config_path


def test_vendored_preprocessor_config_path_for_qwen3_vl_8b() -> None:
    path = vendored_preprocessor_config_path("Qwen/Qwen3-VL-8B-Instruct")
    assert path is not None
    assert path.name == "preprocessor_config.json"
    assert path.is_file()


def test_vendored_preprocessor_config_path_unknown_model_returns_none() -> None:
    assert vendored_preprocessor_config_path("unknown/Model") is None
