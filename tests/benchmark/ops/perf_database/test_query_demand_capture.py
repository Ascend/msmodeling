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

"""Tests that profiling HIT and MISS paths emit backend query demands."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.profiling_database.profiling_data_source import ProfilingDataSource
from tensor_cast.performance_model.profiling_database.query_demand import (
    QUERY_TRACE_DIR_ENV,
    load_query_demand_traces,
)


def _write_elementwise_database(path: Path) -> None:
    (path / "op_mapping.yaml").write_text(
        """
version: test
operator_mappings:
  aten.add.Tensor:
    kernel_type: Add
    query_mode: elementwise
""".strip()
        + "\n",
        encoding="utf-8",
    )
    headers = [
        "OP State",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "Average Duration(us)",
    ]
    with (path / "Add.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "OP State": "dynamic",
                "Input Shapes": "2,4;2,4",
                "Input Data Types": "DT_BF16;DT_BF16",
                "Input Formats": "ND;ND",
                "Output Shapes": "2,4",
                "Output Data Types": "DT_BF16",
                "Output Formats": "ND",
                "Average Duration(us)": "1.0",
            }
        )


def _add_info(tokens: int) -> OpInvokeInfo:
    left = torch.empty((tokens, 4), dtype=torch.bfloat16)
    right = torch.empty((tokens, 4), dtype=torch.bfloat16)
    output = torch.empty((tokens, 4), dtype=torch.bfloat16)
    return OpInvokeInfo(torch.ops.aten.add.Tensor, (left, right), {}, output)


def test_elementwise_hit_and_miss_are_both_captured(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "database"
    trace_dir = tmp_path / "trace"
    database.mkdir()
    _write_elementwise_database(database)
    monkeypatch.setenv(QUERY_TRACE_DIR_ENV, str(trace_dir))
    source = ProfilingDataSource(database)

    assert source.lookup(_add_info(2)) is not None
    assert source.lookup(_add_info(3)) is None

    demands = load_query_demand_traces(trace_dir)
    assert {demand.output_shapes for demand in demands} == {((2, 4),), ((3, 4),)}
    assert {demand.kernel_type for demand in demands} == {"Add"}
    assert {demand.query_mode for demand in demands} == {"elementwise"}


def test_grouped_list_inputs_are_collapsed_to_kernel_visible_shapes() -> None:
    activations = [
        torch.empty((2, 4), dtype=torch.bfloat16),
        torch.empty((3, 4), dtype=torch.bfloat16),
    ]
    weights = [
        torch.empty((4, 8), dtype=torch.int8),
        torch.empty((4, 8), dtype=torch.int8),
    ]
    output = torch.empty((5, 8), dtype=torch.bfloat16)
    info = OpInvokeInfo(
        torch.ops.tensor_cast.grouped_matmul.default,
        (activations, weights, [None, None]),
        {},
        output,
    )

    projected = ProfilingDataSource._extract_grouped_query_inputs(info)

    assert projected == [
        ((5, 4), torch.bfloat16),
        ((2, 4, 8), torch.int8),
    ]


def test_compute_scale_trace_preserves_physical_dtype_scalar_and_regime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "database"
    trace_dir = tmp_path / "trace"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        """
version: test
operator_mappings:
  tensor_cast.dynamic_quantize_symmetric.default:
    kernel_type: DynamicQuant
    compute_subcategory: compute_scale
    tc_input_count: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (database / "DynamicQuant.csv").write_text(
        """Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"64,32",DT_BF16,ND,"64,32;64",INT8;FLOAT,ND;ND,1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(QUERY_TRACE_DIR_ENV, str(trace_dir))
    source = ProfilingDataSource(database)
    input_tensor = torch.empty((17, 32), dtype=torch.float16)
    output = (
        torch.empty((17, 32), dtype=torch.int8),
        torch.empty((), dtype=torch.float32),
    )
    info = OpInvokeInfo(
        torch.ops.tensor_cast.dynamic_quantize_symmetric.default,
        (input_tensor, []),
        {},
        output,
    )

    assert source.lookup(info) is None

    demands = load_query_demand_traces(trace_dir)
    assert len(demands) == 1
    demand = demands[0]
    assert demand.query_mode == "compute_scale"
    assert demand.input_dtypes == ("DT_FLOAT16",)
    assert demand.output_shapes == ((17, 32), ())
    assert demand.output_dtypes == ("INT8", "FLOAT")
    assert demand.attributes["scale_mode"] == "per_tensor"
    assert demand.attributes["output_count"] == 2
