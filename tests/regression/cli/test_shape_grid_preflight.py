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

"""Tests for the pure-Python replay preflight."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

from tools.perf_data_collection.grid_generator.preflight import (
    RowPreflightResult,
    _StubTensor,
    preflight_generated_rows,
)


OP_REPLAY_DIR = Path(__file__).resolve().parents[3] / "tools" / "perf_data_collection" / "op_replay"


def test_stub_tensor_numel_is_derived_from_shape() -> None:
    tensor = _StubTensor((2, 3, 4), "DT_BF16", "test")

    assert tensor.numel() == 24
    assert tensor.nelement() == 24


def test_preflight_restores_shared_replay_runtime() -> None:
    replay_path = str(OP_REPLAY_DIR)
    if replay_path not in sys.path:
        sys.path.insert(0, replay_path)
    replay_common = importlib.reload(importlib.import_module("common"))
    original_attributes = {
        name: getattr(replay_common, name)
        for name in (
            "torch",
            "torch_npu",
            "DTYPE_MAP",
            "init_runtime",
            "get_runtime_modules",
            "build_input_tensor",
            "build_host_tensor",
            "maybe_cast_internal_format",
            "ensure_npu_available",
        )
    }

    results = preflight_generated_rows("Add", [_add_row()], OP_REPLAY_DIR)

    assert results == [RowPreflightResult(0, True)]
    for name, original in original_attributes.items():
        assert getattr(replay_common, name) is original


def _add_row(**overrides: str) -> dict[str, str]:
    row = {
        "Input Shapes": '"1,4;1,4"',
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": '"1,4"',
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
    }
    row.update(overrides)
    return row


def test_valid_add_row_passes_build_case_contract() -> None:
    results = preflight_generated_rows("Add", [_add_row()], OP_REPLAY_DIR)

    assert len(results) == 1
    assert results[0].passed, results[0].reason


def test_unparseable_shape_fails_with_reason() -> None:
    results = preflight_generated_rows("Add", [_add_row(**{"Input Shapes": '"1,x;1,4"'})], OP_REPLAY_DIR)

    assert not results[0].passed
    assert results[0].reason
    assert "Error" in results[0].reason or "error" in results[0].reason or "invalid" in results[0].reason.lower()


def test_unsupported_dtype_fails_closed() -> None:
    results = preflight_generated_rows("Add", [_add_row(**{"Input Data Types": "DT_BF16;DT_FLOAT128"})], OP_REPLAY_DIR)

    assert not results[0].passed
    assert "Unsupported dtype" in results[0].reason or "Unsupported" in results[0].reason


def test_kernel_without_run_module_fails_all_rows() -> None:
    results = preflight_generated_rows("NoSuchKernel", [_add_row(), _add_row()], OP_REPLAY_DIR)

    assert len(results) == 2
    assert all(not result.passed for result in results)
    assert all("no op_replay entry point" in result.reason for result in results)


def test_missing_required_row_column_is_reported_not_raised() -> None:
    broken = _add_row()
    del broken["Input Data Types"]

    results = preflight_generated_rows("Add", [broken], OP_REPLAY_DIR)

    assert not results[0].passed
    assert results[0].reason


def test_runtime_rich_attention_row_preflight_uses_real_constraints() -> None:
    # FusedInferAttentionScore's build_case validates the runtime metadata
    # contract; a row without the required runtime columns must be rejected
    # before it is written to the CSV.
    row = {
        "Input Shapes": '"256,4,128"',
        "Input Data Types": "DT_BF16",
        "Input Formats": "ND",
        "Output Shapes": '"256,4,128"',
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
    }

    results = preflight_generated_rows("FusedInferAttentionScore", [row], OP_REPLAY_DIR)

    assert not results[0].passed
    assert results[0].reason
