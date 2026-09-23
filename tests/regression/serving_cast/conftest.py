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

import pytest
from tests.helpers.op_registry import build_op_registry


@pytest.fixture(scope="session")
def cast_model(cfg_registry, op_registry, model_zoo):
    """Consume shared tensor_cast artifacts without rebuilding base fixtures."""
    selected_alias = "deepseek_v32"
    if selected_alias not in model_zoo:
        aliases = ", ".join(sorted(model_zoo))
        raise pytest.UsageError(f"Unknown model alias '{selected_alias}'. Available aliases: {aliases}")
    model_id = model_zoo[selected_alias]
    hf_config = cfg_registry.get(model_id)
    op_meta = op_registry.get(model_id)
    if op_meta is None:
        op_registry.update(build_op_registry(cfg_registry))
        op_meta = op_registry.get(model_id)
    if hf_config is None:
        raise pytest.UsageError(f"hf_config missing for model_id='{model_id}'")
    if not op_meta:
        raise pytest.UsageError(f"op_meta missing for model_id='{model_id}'")
    return {
        "alias": selected_alias,
        "model_id": model_id,
        "hf_config": hf_config,
        "op_meta": op_meta,
    }


@pytest.fixture(scope="module")
def ttft_ctx(cast_model):
    """Context object used by serving TTFT/TPOT tests."""
    return {
        "cast_model": cast_model,
        "ttft_ms": None,
        "tpot_ms": None,
    }
