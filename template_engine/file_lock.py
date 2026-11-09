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

"""Destination-scoped transactions using the project's filelock dependency."""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

_LOCKS = {}
_GUARD = threading.Lock()


@contextmanager
def destination_lock(destination, timeout=60):
    """Reuse one reentrant lock object per path, including nested registration."""
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(path))
    with _GUARD:
        lock = _LOCKS.setdefault(key, FileLock(str(path) + ".lock"))
    with lock.acquire(timeout=max(0, timeout)):
        yield
