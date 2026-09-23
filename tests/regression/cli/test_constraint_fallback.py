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

"""Tests for constraint-driven Shape fallback rows."""

from tools.perf_data_collection.grid_generator.constraint_fallback import (
    _FallbackShape,
    _fallback_to_row,
)


def test_fallback_row_clears_all_inherited_performance_results() -> None:
    template = {
        "OP State": "dynamic",
        "Accelerator Core": "MIX_AIC",
        "Input Shapes": "1,32,128",
        "Output Shapes": "1,32,128",
        "Average Duration(us)": "10",
        "Profiling Average Duration(us)": "11",
        "Profiling Median Duration(us)": "12",
        "MicroBench aicore_time(us)": "13",
        "Profiling Average aicore_time(us)": "14",
        "Profiling Average aiv_vec_ratio": "15",
        "Profiling Average cube_utilization(%)": "16",
        "Runtime source_profile": "measured",
    }
    shape = _FallbackShape(
        kernel_type="LightningIndexer",
        input_shapes=[[(2, 32, 128)]],
        input_dtypes=["DT_BF16"],
        input_formats=["ND"],
        output_shapes=[(2, 1, 2048)],
        output_dtypes=["DT_BF16"],
        output_formats=["ND"],
    )

    row = _fallback_to_row(shape, list(template), template)

    performance_columns = [
        "Average Duration(us)",
        "Profiling Average Duration(us)",
        "Profiling Median Duration(us)",
        "MicroBench aicore_time(us)",
        "Profiling Average aicore_time(us)",
        "Profiling Average aiv_vec_ratio",
        "Profiling Average cube_utilization(%)",
    ]
    assert all(row[column] == "" for column in performance_columns)
    assert row["Runtime source_profile"] == "constraint_fallback"
