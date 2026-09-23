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

"""Runner adapters + factory.

Every adapter module imports its heavy deps (torch / tensor_cast /
serving_cast) INSIDE ``run``, so this package imports cleanly without the
simulation stack (the FastAPI app stays bootable).
"""

from .registry import (
    UnknownRunnerError,
    create_runner,
    get_runner_class,
    runner_class_for_module,
)

__all__ = [
    "UnknownRunnerError",
    "create_runner",
    "get_runner_class",
    "runner_class_for_module",
]
