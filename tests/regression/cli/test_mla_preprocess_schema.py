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

"""Pure contract tests for the architecture-neutral MLA preprocess adapter."""

import pytest

from tools.perf_data_collection.op_replay import mla_preprocess_0_mix_aic_run as adapter
from tools.perf_data_collection.op_replay.mla_preprocess_schema import (
    MLA_PREPROCESS_INPUT_DTYPES,
    MLA_PREPROCESS_INPUT_FORMATS,
    MLA_PREPROCESS_OUTPUT_DTYPES,
    MLA_PREPROCESS_OUTPUT_FORMATS,
    MlaPreprocessRuntime,
    RUNTIME_BLOCK_SIZE,
    RUNTIME_CACHE_MODE,
    RUNTIME_CASE_ID,
    RUNTIME_ENABLE_INNER_OUT,
    RUNTIME_HIDDEN_SIZE,
    RUNTIME_KV_LORA_RANK,
    RUNTIME_LOCAL_NUM_HEADS,
    RUNTIME_METADATA_COMPLETENESS,
    RUNTIME_NUM_TOKENS,
    RUNTIME_QK_NOPE_HEAD_DIM,
    RUNTIME_QK_ROPE_HEAD_DIM,
    RUNTIME_QUANT_MODE,
    RUNTIME_Q_LORA_RANK,
    RUNTIME_SOURCE_PROFILE,
    RUNTIME_WEIGHT_FORMAT,
    RUNTIME_WEIGHT_QUANTIZED,
)


def _runtime_row() -> dict[str, str]:
    return {
        RUNTIME_CASE_ID: "synthetic_mla_case",
        RUNTIME_NUM_TOKENS: "65",
        RUNTIME_HIDDEN_SIZE: "4096",
        RUNTIME_LOCAL_NUM_HEADS: "16",
        RUNTIME_Q_LORA_RANK: "1024",
        RUNTIME_KV_LORA_RANK: "256",
        RUNTIME_QK_NOPE_HEAD_DIM: "128",
        RUNTIME_QK_ROPE_HEAD_DIM: "32",
        RUNTIME_BLOCK_SIZE: "64",
        RUNTIME_CACHE_MODE: "krope_ctkv",
        RUNTIME_QUANT_MODE: "per_tensor_quant_asymm",
        RUNTIME_ENABLE_INNER_OUT: "true",
        RUNTIME_WEIGHT_QUANTIZED: "true",
        RUNTIME_WEIGHT_FORMAT: "FRACTAL_NZ",
        RUNTIME_SOURCE_PROFILE: "synthetic-test",
        RUNTIME_METADATA_COMPLETENESS: "generated",
    }


def _shape_cell(shapes: list[tuple[int, ...]]) -> str:
    return ";".join(",".join(str(dimension) for dimension in shape) for shape in shapes)


def test_runtime_metadata_derives_shapes_without_model_config():
    runtime = MlaPreprocessRuntime.from_row(_runtime_row())
    inputs, outputs = runtime.shapes()

    assert runtime.case_id == "synthetic_mla_case"
    assert inputs[1] == (1, 128, 1312, 32)
    assert inputs[5] == (1, 32, 2560, 32)
    assert inputs[11] == (2, 64, 1, 256)
    assert outputs[0] == (65, 16, 256)
    assert outputs[2] == (65, 16, 32)


def test_adapter_accepts_arbitrary_legal_runtime_case_id():
    row = _runtime_row()
    inputs, outputs = MlaPreprocessRuntime.from_row(row).shapes()
    row.update(
        {
            "Input Shapes": _shape_cell(inputs),
            "Output Shapes": _shape_cell(outputs),
            "Input Data Types": ";".join(MLA_PREPROCESS_INPUT_DTYPES),
            "Input Formats": ";".join(MLA_PREPROCESS_INPUT_FORMATS),
            "Output Data Types": ";".join(MLA_PREPROCESS_OUTPUT_DTYPES),
            "Output Formats": ";".join(MLA_PREPROCESS_OUTPUT_FORMATS),
        }
    )

    metadata = adapter.resolve_case_metadata(row)

    assert metadata["case_id"] == "synthetic_mla_case"
    assert metadata["num_tokens"] == 65


def test_runtime_schema_rejects_legacy_metadata():
    row = {**_runtime_row(), RUNTIME_METADATA_COMPLETENESS: "legacy"}

    with pytest.raises(ValueError, match="does not accept legacy"):
        MlaPreprocessRuntime.from_row(row)
