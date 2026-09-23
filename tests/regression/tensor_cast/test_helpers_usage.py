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

from tests.helpers.model_assets import vendored_model_config_path
from tests.helpers.model_builder import build_or_get_cached_model, make_user_input_config

_QWEN3_32B = "Qwen/Qwen3-32B"


def test_make_user_input_config_sets_defaults():
    user_config = make_user_input_config(model_id=_QWEN3_32B)
    assert user_config.model_id == vendored_model_config_path(_QWEN3_32B)
    assert user_config.device == "TEST_DEVICE"
    assert user_config.query_len == 32


def test_build_or_get_cached_model_reuses_cache(monkeypatch):
    calls = []

    def _fake_build_model(user_config):
        calls.append(user_config.model_id)
        return {"model_id": user_config.model_id}

    monkeypatch.setattr("tests.helpers.model_builder.build_model", _fake_build_model)
    cache = {}
    user_config = make_user_input_config(model_id=_QWEN3_32B)
    expected_model_id = vendored_model_config_path(_QWEN3_32B)

    model_a = build_or_get_cached_model(user_config, cache)
    model_b = build_or_get_cached_model(user_config, cache)

    assert model_a == model_b
    assert calls == [expected_model_id]
