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

"""Shared PD phase identifiers and result serialization without runner dependencies."""

import json
from pathlib import Path
from typing import Any

from ..io_utils import open_file

PREFILL_PHASE = "prefill"
DECODE_PHASE = "decode"
PHASES = (PREFILL_PHASE, DECODE_PHASE)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write PD artifacts, preserving paths and scalar metric values."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with open_file(path, "w") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
