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

"""Specification loading and resolution errors."""

from __future__ import annotations

from tools.model_diagnostics.errors import ModelDiagnosticsError, SourceLoadError


class SpecificationError(ModelDiagnosticsError):
    """Base error for diagnostics specification failures."""


class SpecificationLoadError(SpecificationError):
    """YAML, schema, formula, or strategy-parameter validation failed."""


class UnsupportedModelSpec(SpecificationError):
    """No Spec matched the request context exactly."""


class AmbiguousModelSpec(SpecificationError):
    """More than one Spec matched the request context."""


__all__ = [
    "AmbiguousModelSpec",
    "SourceLoadError",
    "SpecificationError",
    "SpecificationLoadError",
    "UnsupportedModelSpec",
]
