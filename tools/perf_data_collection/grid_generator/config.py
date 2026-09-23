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

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_shape_grid_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_op_mapping_metadata(data_dir: Path) -> dict[str, dict]:
    op_mapping_path = data_dir / "op_mapping.yaml"
    if not op_mapping_path.exists():
        return {}

    with op_mapping_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    meta: dict[str, dict] = {}
    mappings = data.get("operator_mappings", {})
    for _op_name, info in mappings.items():
        if not isinstance(info, dict):
            continue
        kt = info.get("kernel_type")
        if not kt:
            continue

        is_zero = bool(info.get("zero_cost", False))
        is_composite = bool(info.get("composite", False))
        is_comm = info.get("category") == "communication"
        qm = info.get("query_mode")

        if kt not in meta:
            meta[kt] = {
                "zero_cost": is_zero,
                "composite": is_composite,
                "communication": is_comm,
                "query_mode": qm,
            }

        for alt in info.get("alternate_kernel_types", []):
            if alt not in meta:
                meta[alt] = {
                    "zero_cost": is_zero,
                    "composite": is_composite,
                    "communication": is_comm,
                    "query_mode": qm,
                    "alternates_of": kt,
                }
    return meta
