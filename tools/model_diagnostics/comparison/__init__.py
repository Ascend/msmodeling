# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
