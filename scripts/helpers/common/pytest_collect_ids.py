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

"""Pytest plugin that records collected node ids, one per line."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class _Node(Protocol):
    nodeid: str


class _Session(Protocol):
    items: list[_Node]


def pytest_collection_finish(session: _Session) -> None:
    """Record each collected node id.

    ``-qq`` only prints ``path: count``. Node ids go to ``MSMODELING_COLLECT_NODE_IDS``
    when that path is set, and also to stdout for callers that parse it.
    """
    node_ids = [item.nodeid for item in session.items]
    path = os.environ.get("MSMODELING_COLLECT_NODE_IDS", "").strip()
    if path:
        Path(path).write_text("".join(f"{node_id}\n" for node_id in node_ids), encoding="utf-8")
    for node_id in node_ids:
        print(node_id, flush=True)
