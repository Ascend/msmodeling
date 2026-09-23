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

"""Stage comparison contracts and strategy registry."""

from .builtin import (
    BoundaryEqualStrategy,
    ConcatShapeStrategy,
    OneToOneEqualStrategy,
)
from .models import StageComparisonRequest, StageComparisonStrategy
from .operator_policy import (
    DEFAULT_OPERATOR_ALIASES,
    resolve_operator_aliases,
)
from .parsers import (
    BoundaryEqualOptionParser,
    ComparisonOptionParseError,
    ConcatOptionParser,
    OneToOneOptionParser,
)
from .registry import (
    OptionParser,
    StageComparisonRegistry,
    StrategyRegistrationError,
    StrategyResolutionError,
)

__all__ = [
    "DEFAULT_OPERATOR_ALIASES",
    "OptionParser",
    "BoundaryEqualStrategy",
    "BoundaryEqualOptionParser",
    "ComparisonOptionParseError",
    "ConcatShapeStrategy",
    "ConcatOptionParser",
    "OneToOneEqualStrategy",
    "OneToOneOptionParser",
    "resolve_operator_aliases",
    "StageComparisonRegistry",
    "StageComparisonRequest",
    "StageComparisonStrategy",
    "StrategyRegistrationError",
    "StrategyResolutionError",
]
