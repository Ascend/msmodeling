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

"""CLI module specifications for the parameter registry.

Each module file (e.g., text_generate.py) exports a SPEC object of type ModuleSpec
that defines all CLI parameters for that module.
"""

from cli.registry.modules.image_generate import SPEC as IMAGE_GENERATE_SPEC
from cli.registry.modules.text_generate import SPEC as TEXT_GENERATE_SPEC
from cli.registry.modules.throughput_optimizer import SPEC as THROUGHPUT_OPTIMIZER_SPEC
from cli.registry.modules.video_generate import SPEC as VIDEO_GENERATE_SPEC

# Registry of all module specs
MODULE_SPECS = {
    "text_generate": TEXT_GENERATE_SPEC,
    "throughput_optimizer": THROUGHPUT_OPTIMIZER_SPEC,
    "video_generate": VIDEO_GENERATE_SPEC,
    "image_generate": IMAGE_GENERATE_SPEC,
}


def get_spec(module_id: str):
    """Get the ModuleSpec for a given module ID.

    Args:
        module_id: The module identifier (e.g., "text_generate")

    Returns:
        The ModuleSpec for the module

    Raises:
        KeyError: If the module_id is not registered
    """
    if module_id not in MODULE_SPECS:
        raise KeyError(f"Unknown module: {module_id}. Available: {list(MODULE_SPECS.keys())}")
    return MODULE_SPECS[module_id]


__all__ = ["MODULE_SPECS", "TEXT_GENERATE_SPEC", "get_spec"]
