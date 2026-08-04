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
