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

import json
from pathlib import Path

from loguru import logger

# Characteristic fields in a MoE model's config.json; presence of any one identifies it as MoE
_MOE_INDICATOR_KEYS = {
    "num_experts_per_tok",
    "n_routed_experts",
    "num_local_experts",
    "moe_intermediate_size",
}


def detect_is_moe(model_path: str) -> bool:
    """Read config.json under the model path and determine whether it is a MoE model.

    Args:
        model_path: Path to the model root directory (the directory containing config.json)

    Returns:
        True if the model's config.json contains MoE characteristic fields, otherwise False.
        Returns False when the path is empty, the file does not exist, or reading fails.
    """
    if not model_path:
        return False
    config_file = Path(model_path) / "config.json"
    if not config_file.exists():
        logger.debug(f"Model config not found: {config_file}, assuming non-MoE")
        return False
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        return any(k in model_config for k in _MOE_INDICATOR_KEYS)
    except Exception as e:
        logger.warning(f"Failed to read model config for MoE detection: {config_file}: {e}")
        return False
