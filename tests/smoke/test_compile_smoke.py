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

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig

DATA_DIR = "tests/assets/model_config"


def test_compile_no_graph_break_deepseek():
    """Validate that DeepSeek compile path has no graph break using a tiny local config."""
    user_config = UserInputConfig(
        model_id=f"{DATA_DIR}/deepseek_new",
        do_compile=True,
        num_hidden_layers_override=1,
    )
    model = build_model(user_config)
    assert model is not None


def test_compile_no_graph_break_minimax():
    """Validate that Minimax compile path has no graph break."""
    user_config = UserInputConfig(
        model_id=f"{DATA_DIR}/minimax_m2",
        do_compile=True,
        num_hidden_layers_override=1,
    )
    model = build_model(user_config)
    assert model is not None
