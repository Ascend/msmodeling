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

"""Model features API for benchmark mode.

Provides model-specific default overrides for benchmark form.
"""

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

_CONFIG_PATH = Path(__file__).parents[2] / "var" / "config" / "benchmark_model_features.json"


@router.get("/api/model-features/{model_id:path}")
def get_model_features(model_id: str) -> dict:
    """Get model-specific overrides for benchmark form.

    Uses substring matching: first match wins. Returns empty overrides if no match.

    Args:
        model_id: Model identifier (e.g. "zai-org/GLM-5").

    Returns:
        Dict with "overrides" key containing field overrides.
    """
    if not _CONFIG_PATH.exists():
        return {"overrides": {}}

    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"overrides": {}}

    models = config.get("models", {})
    # Substring matching: first match wins
    for pattern, data in models.items():
        if pattern in model_id:
            return {"overrides": data.get("overrides", {})}

    return {"overrides": {}}
