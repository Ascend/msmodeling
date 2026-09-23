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

from dataclasses import dataclass

import pytest

from tools.model_diagnostics.comparison import (
    StageComparisonRegistry,
    StrategyRegistrationError,
    StrategyResolutionError,
)
from tools.model_diagnostics.domain import (
    OneToOneOptions,
    TensorMapping,
    TensorMappingMode,
)


@dataclass(frozen=True)
class _Parser:
    result: OneToOneOptions

    def parse(self, raw):
        assert raw == {"mapping": "positional"}
        return self.result


@dataclass(frozen=True)
class _Strategy:
    strategy_id: str

    def execute(self, request):
        return ()


def _options() -> OneToOneOptions:
    return OneToOneOptions(mapping=TensorMapping(mode=TensorMappingMode.POSITIONAL))


def test_registry_resolves_strategy_and_its_option_parser_together() -> None:
    registry = StageComparisonRegistry()
    strategy = _Strategy("one_to_one")
    options = _options()

    registry.register("one_to_one", option_parser=_Parser(options), strategy=strategy)

    assert registry.resolve("one_to_one") is strategy
    assert registry.parse_options("one_to_one", {"mapping": "positional"}) is options
    assert registry.registered_ids() == ("one_to_one",)


def test_registry_rejects_duplicate_and_mismatched_ids() -> None:
    registry = StageComparisonRegistry()
    registry.register("one_to_one", option_parser=_Parser(_options()), strategy=_Strategy("one_to_one"))

    with pytest.raises(StrategyRegistrationError, match="duplicate"):
        registry.register("one_to_one", option_parser=_Parser(_options()), strategy=_Strategy("one_to_one"))
    with pytest.raises(StrategyRegistrationError, match="does not match"):
        StageComparisonRegistry().register(
            "one_to_one",
            option_parser=_Parser(_options()),
            strategy=_Strategy("boundary"),
        )


def test_registry_fails_fast_for_unregistered_strategy() -> None:
    registry = StageComparisonRegistry()

    with pytest.raises(StrategyResolutionError, match="missing"):
        registry.resolve("missing")
    with pytest.raises(StrategyResolutionError, match="missing"):
        registry.parse_options("missing", {})
