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

"""Tests for generic theory fallback after query collection."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from tools.perf_data_collection.grid_generator.generators.base import TheoryShapeRow
from tools.perf_data_collection.grid_generator.theory_fallback import (
    build_theory_fallback_rows,
    theory_generation_is_skipped,
)


def _source_row(tokens: int) -> dict[str, str]:
    return {
        "OP State": "dynamic",
        "Input Shapes": f"{tokens},4;{tokens},4",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": f"{tokens},4",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
        "Runtime source_profile": "profile_a",
    }


def _theory_row(tokens: int) -> TheoryShapeRow:
    return TheoryShapeRow(
        input_shapes=[(tokens, 4), (tokens, 4)],
        output_shapes=[(tokens, 4)],
        extra_values={"Runtime source_profile": "theory"},
    )


def test_theory_fallback_budget_counts_only_new_unique_rows() -> None:
    headers = list(_source_row(1))
    source_rows = [_source_row(1)]
    generator = iter([_theory_row(1), _theory_row(2), _theory_row(3)])

    with mock.patch(
        "tools.perf_data_collection.grid_generator.theory_fallback.get_default_theory_generator",
        return_value=generator,
    ):
        generated, summary = build_theory_fallback_rows(
            kernel_type="Add",
            model_names=["org/model"],
            config={},
            op_meta={},
            csv_path=Path("Add.csv"),
            headers=headers,
            source_rows=source_rows,
            row_limit=2,
        )

    assert [row["Input Shapes"].strip('"') for row in generated] == ["2,4;2,4", "3,4;3,4"]
    assert {row["Average Duration(us)"] for row in generated} == {"0"}
    assert {row["Runtime source_profile"] for row in generated} == {""}
    assert summary == {"attempted": 3, "duplicates": 1, "appended": 2}


def test_theory_skip_accepts_explicit_config_and_mapping_boundaries() -> None:
    assert theory_generation_is_skipped(
        "BatchMatMulV2",
        {"assignments": {"BatchMatMulV2": "skip"}},
        {},
    )
    assert theory_generation_is_skipped(
        "CompositeKernel",
        {"assignments": {}},
        {"CompositeKernel": {"composite": True}},
    )
    assert not theory_generation_is_skipped(
        "Add",
        {"assignments": {"Add": "elementwise_binary"}},
        {"Add": {}},
    )
