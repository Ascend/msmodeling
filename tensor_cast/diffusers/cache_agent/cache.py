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

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheConfig:
    """Block replacement range for DiT cache wrappers."""

    block_start: int = 0
    block_end: int = 10000


@dataclass
class CacheState:
    """Shared runtime state for DiT cache simulation wrappers."""

    reuse: bool = False
    range_hidden: Optional[Any] = None
    range_encoder: Optional[Any] = None
    delta_hidden: Optional[Any] = None
    delta_encoder: Optional[Any] = None
