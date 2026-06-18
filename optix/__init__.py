# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""OptiX service parameter optimizer for LLM inference tuning."""

import functools
import os
import sys

from loguru import logger


@functools.cache
def configure_logger() -> None:
    """Configure optix loguru handler without clearing unrelated handlers."""
    log_level = os.getenv("MODELEVALSTATE_LEVEL", "INFO").upper()
    logger.remove(0)
    logger.add(sys.stderr, level=log_level, enqueue=True)


def main() -> None:
    """CLI entry for ``python -m optix`` and coverage tests."""
    configure_logger()
    from optix.optimizer.optimizer import main as optimizer_main

    optimizer_main()
