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
"""Typed registry for stage comparison strategies and option parsers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from tools.model_diagnostics.domain import ComparisonOptions

from .models import StageComparisonStrategy


class StrategyRegistrationError(ValueError):
    """A strategy registry entry is internally inconsistent."""


class StrategyResolutionError(LookupError):
    """A validated strategy id cannot be resolved at runtime."""


class OptionParser(Protocol):
    def parse(self, raw: Mapping[str, object]) -> ComparisonOptions: ...


@dataclass(frozen=True)
class _Entry:
    option_parser: OptionParser
    strategy: StageComparisonStrategy


class StageComparisonRegistry:
    """Register strategy and parser as one inseparable entry."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        strategy_id: str,
        *,
        option_parser: OptionParser,
        strategy: StageComparisonStrategy,
    ) -> None:
        if not strategy_id.strip():
            raise StrategyRegistrationError("strategy_id must not be empty")
        if strategy_id in self._entries:
            raise StrategyRegistrationError(f"duplicate strategy_id {strategy_id!r}")
        if strategy.strategy_id != strategy_id:
            raise StrategyRegistrationError(
                f"strategy id {strategy.strategy_id!r} does not match registration {strategy_id!r}"
            )
        self._entries[strategy_id] = _Entry(option_parser=option_parser, strategy=strategy)

    def resolve(self, strategy_id: str) -> StageComparisonStrategy:
        try:
            return self._entries[strategy_id].strategy
        except KeyError as error:
            raise StrategyResolutionError(f"unregistered strategy_id {strategy_id!r}") from error

    def parse_options(
        self,
        strategy_id: str,
        raw: Mapping[str, object],
    ) -> ComparisonOptions:
        try:
            parser = self._entries[strategy_id].option_parser
        except KeyError as error:
            raise StrategyResolutionError(f"unregistered strategy_id {strategy_id!r}") from error
        return parser.parse(raw)

    def registered_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)
