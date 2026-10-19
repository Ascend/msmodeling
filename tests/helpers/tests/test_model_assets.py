# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from pathlib import Path

import pytest

from tests.helpers.model_assets import (
    resolve_offline_model_id,
    set_offline_hub_remap,
    vendored_model_config_path,
    vendored_preprocessor_config_path,
    wrap_load_config,
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


def test_wrap_load_config_rewrites_registered_hub_id_before_hub_access() -> None:
    seen: dict[str, str] = {}

    def _load_config(_self, model_id, remote_source="huggingface"):
        seen["model_id"] = model_id
        seen["remote_source"] = remote_source
        return "config"

    wrapped = wrap_load_config(_load_config)
    assert wrapped(object(), "Qwen/Qwen3-32B", remote_source="huggingface") == "config"
    assert seen["model_id"] == vendored_model_config_path("Qwen/Qwen3-32B")
    assert seen["remote_source"] == "huggingface"

    set_offline_hub_remap(False)
    try:
        wrapped(object(), "Qwen/Qwen3-32B")
    finally:
        set_offline_hub_remap(True)
    assert seen["model_id"] == "Qwen/Qwen3-32B"


def test_get_hf_config_loads_vendored_directory_for_registered_hub_id(monkeypatch) -> None:
    loaded: dict[str, str] = {}

    class _Loader:
        def load_config(self, model_id: str):
            loaded["model_id"] = model_id
            return {"model_type": "qwen3"}

    monkeypatch.setattr("tests.helpers.model_cache.AutoModelConfigLoader", lambda: _Loader())
    monkeypatch.setattr("tests.helpers.model_cache._HF_CONFIG_CACHE", {})
    from tests.helpers.model_cache import get_hf_config

    get_hf_config("Qwen/Qwen3-32B")
    assert loaded["model_id"] == vendored_model_config_path("Qwen/Qwen3-32B")
    assert "hf-mirror.com" not in loaded["model_id"]
    assert not loaded["model_id"].startswith("Qwen/")
