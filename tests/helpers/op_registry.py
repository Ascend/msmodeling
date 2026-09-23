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

"""Shared op metadata builder for regression fixtures."""


def build_op_registry(cfg_registry: dict) -> dict:
    """Build a lightweight op registry from shared hf config cache."""
    per_model_ops = {}
    for model_id, hf_config in cfg_registry.items():
        per_model_ops[model_id] = {
            "model_type": getattr(hf_config, "model_type", None),
            "num_hidden_layers": getattr(hf_config, "num_hidden_layers", None),
        }
    return per_model_ops
