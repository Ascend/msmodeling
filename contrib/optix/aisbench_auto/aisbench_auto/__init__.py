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
"""OptiX plugin for an AISBench-compatible benchmark command."""


def register():
    """Register the AISBench Auto benchmark and settings factory with OptiX.

    Called by the plugin loader during initialization. Registers ``AisBenchAuto``
    under ``aisbench_auto`` and installs the ``AisBenchAutoSettings`` factory.
    An existing benchmark under the same key is overwritten with a warning;
    the current settings factory is replaced on every call.
    """
    from optix.config.config import register_settings
    from optix.optimizer.register import register_benchmarks

    from .benchmark import AisBenchAuto
    from .config import AisBenchAutoSettings

    register_benchmarks("aisbench_auto", AisBenchAuto)
    register_settings(lambda: AisBenchAutoSettings())
