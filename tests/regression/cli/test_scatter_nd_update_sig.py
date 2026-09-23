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

"""Tests for ScatterNdUpdate signature normalization in signature_utils."""

from tools.perf_data_collection.signature_utils import canonicalize_profile_signature


def _row(input_shapes: str, dtypes: str = "DT_BF16;INT32;DT_BF16", formats: str = "ND;ND;ND") -> dict[str, str]:
    return {
        "Input Shapes": input_shapes,
        "Input Data Types": dtypes,
        "Input Formats": formats,
        "Output Shapes": input_shapes.split(";")[0],
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
    }


def test_scatter_nd_update_2d_data_shape_is_unchanged() -> None:
    shapes, dtypes, formats = canonicalize_profile_signature(
        _row("226048,128;102,1;102,128"),
        op_name="ScatterNdUpdate",
    )
    assert shapes == "226048,128;102,1;102,128"


def test_scatter_nd_update_3d_paged_data_shape_is_flattened() -> None:
    shapes, dtypes, formats = canonicalize_profile_signature(
        _row("161424,128,128;1536,1;1536,128"),
        op_name="ScatterNdUpdate",
    )
    assert shapes == "20662272,128;1536,1;1536,128"


def test_scatter_nd_update_ai_core_alias_also_flattens() -> None:
    shapes, _, _ = canonicalize_profile_signature(
        _row("316,128,128;3,1;3,128"),
        op_name="ScatterNdUpdateAiCore",
    )
    # 316 * 128 = 40448
    assert shapes == "40448,128;3,1;3,128"
