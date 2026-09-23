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

"""Typed registry for stage comparison strategies and option parsers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from tools.model_diagnostics.domain import ComparisonOptions
from tools.model_diagnostics.errors import ModelDiagnosticsError

from .models import StageComparisonStrategy


class StrategyRegistrationError(ValueError):
    """A strategy registry entry is internally inconsistent."""


class StrategyResolutionError(ModelDiagnosticsError):
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
