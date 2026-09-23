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

"""CLI/WebUI parameter registry.

This package is the single source of truth for CLI-side parameter definitions:
- ``datatypes``: Param, ValidatorRef, ModuleSpec
- ``shared``: cross-module shared Param definitions (MODEL_ID, DEVICE, etc.)
- ``argparse_adapter``: build CLI ArgumentParser from ModuleSpec
- ``validators``: cross-field Python validator functions (L2)
- ``modules``: per-module ModuleSpec definitions (one per CLI command)

UI types (I18nText, UIFieldProps) live in web_ui/backend/services/ui_props/.
"""

from .datatypes import (
    ModuleSpec,
    Param,
    ValidatorRef,
)

__all__ = [
    "ModuleSpec",
    "Param",
    "ValidatorRef",
]
