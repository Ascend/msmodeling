from tests.helpers.model_builder import build_or_get_cached_model, make_user_input_config


def test_make_user_input_config_sets_defaults():
    user_config = make_user_input_config(model_id="Qwen/Qwen3-32B")
    assert user_config.model_id == "Qwen/Qwen3-32B"
    assert user_config.device == "TEST_DEVICE"
    assert user_config.query_len == 32


def test_build_or_get_cached_model_reuses_cache(monkeypatch):
    calls = []

    def _fake_build_model(user_config):
        calls.append(user_config.model_id)
        return {"model_id": user_config.model_id}

    monkeypatch.setattr("tests.helpers.model_builder.build_model", _fake_build_model)
    cache = {}
    user_config = make_user_input_config(model_id="Qwen/Qwen3-32B")

    model_a = build_or_get_cached_model(user_config, cache)
    model_b = build_or_get_cached_model(user_config, cache)

    assert model_a == model_b
    assert calls == ["Qwen/Qwen3-32B"]
