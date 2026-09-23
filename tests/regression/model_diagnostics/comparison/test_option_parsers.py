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

import pytest

from tools.model_diagnostics.builtin import create_stage_comparison_registry
from tools.model_diagnostics.comparison import ComparisonOptionParseError
from tools.model_diagnostics.domain import (
    ConcatOptions,
    OneToOneOptions,
    TensorDirection,
    TensorMappingMode,
)


def _slot(direction="output", index=0, name=None):
    return {"direction": direction, "index": index, "name": name}


def test_default_registry_parses_positional_and_concat_options() -> None:
    registry = create_stage_comparison_registry()

    positional = registry.parse_options(
        "one_to_one",
        {"mapping": {"mode": "positional", "pairs": [], "relations": []}},
    )
    concat = registry.parse_options(
        "concat_shape",
        {
            "axis": 1,
            "mapping": {
                "mode": "composite",
                "pairs": [],
                "relations": [
                    {
                        "left": [{"call_index": 0, "slot": _slot()}],
                        "right": [{"call_index": 1, "slot": _slot("input")}],
                        "operation": "concat",
                        "axis": None,
                    }
                ],
            },
        },
    )

    assert isinstance(positional, OneToOneOptions)
    assert positional.mapping.mode is TensorMappingMode.POSITIONAL
    assert isinstance(concat, ConcatOptions)
    assert concat.axis == 1
    assert concat.mapping.relations[0].right[0].slot.direction is TensorDirection.INPUT
    default_concat = registry.parse_options("concat_shape", {})
    assert isinstance(default_concat, ConcatOptions)
    assert default_concat.axis == -1
    assert default_concat.mapping.mode is TensorMappingMode.COMPOSITE
    assert default_concat.mapping.relations == ()
    axis_only_concat = registry.parse_options("concat_shape", {"axis": 0})
    assert axis_only_concat.axis == 0
    assert axis_only_concat.mapping.mode is TensorMappingMode.COMPOSITE
    assert axis_only_concat.mapping.relations == ()
    assert registry.registered_ids() == (
        "one_to_one",
        "concat_shape",
        "boundary_equal",
    )


def test_one_to_one_parser_rejects_unknown_options() -> None:
    with pytest.raises(ComparisonOptionParseError):
        create_stage_comparison_registry().parse_options(
            "one_to_one",
            {"selected_slots": ["INPUT[0]"]},
        )


@pytest.mark.parametrize(
    "raw",
    (
        {"mapping": {"mode": "positional", "pairs": [], "relations": [], "extra": 1}},
        {
            "mapping": {
                "mode": "explicit",
                "pairs": [],
                "relations": [],
            }
        },
        {
            "mapping": {
                "mode": "composite",
                "pairs": [],
                "relations": [],
            }
        },
    ),
)
def test_option_parsers_reject_structurally_invalid_mapping(raw) -> None:
    with pytest.raises(ComparisonOptionParseError):
        create_stage_comparison_registry().parse_options("one_to_one", raw)
