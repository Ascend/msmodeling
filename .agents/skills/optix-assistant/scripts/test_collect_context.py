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

"""Ownership: pso (test). Unit tests for collect_context.py."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("collect_context.py")


def load_module():
    spec = importlib.util.spec_from_file_location("collect_context", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_parallel_search_space_position_follows_engine():
    # 默认并行参数的位置必须跟引擎走：vLLM 用 run（字段模型渲染），
    # 其他引擎（mindie）保持 env，否则候选被接受却无渲染方，寻优静默失效。
    module = load_module()

    for p in module.default_parallel_search_space("vllm", 8)["parameters"]:
        assert p["config_position"] == "run", f"{p['name']} should be run for vllm"

    for p in module.default_parallel_search_space("mindie", 8)["parameters"]:
        assert p["config_position"] == "env", f"{p['name']} should be env for mindie"


def test_model_derived_params_position_follows_engine():
    # 模型派生注入（MoE/MTP/深模型）与默认并行参数同规则：位置跟引擎走。
    module = load_module()
    model_info = {"is_moe": True, "use_mtp": True, "num_hidden_layers": 48}

    vllm_params = module._inject_model_derived_params("vllm", {"parameters": [], "constants": []}, model_info)[
        "parameters"
    ]
    for p in vllm_params:
        assert p["config_position"] == "run", f"{p['name']} should be run for vllm"

    mindie_params = module._inject_model_derived_params("mindie", {"parameters": [], "constants": []}, model_info)[
        "parameters"
    ]
    for p in mindie_params:
        assert p["config_position"] == "env", f"{p['name']} should be env for mindie"
