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

"""Hardware-free tests for generated replay shape validation."""

from tools.perf_data_collection.op_replay.shape_validation import validate_replay_row


def _row(inputs: str, outputs: str, input_dtypes: str, input_formats: str) -> dict[str, str]:
    output_slots = len(outputs.split(";")) if outputs else 0
    return {
        "Input Shapes": inputs,
        "Input Data Types": input_dtypes,
        "Input Formats": input_formats,
        "Output Shapes": outputs,
        "Output Data Types": ";".join(["DT_BF16"] * output_slots),
        "Output Formats": ";".join(["ND"] * output_slots),
    }


def test_mul_validation_accepts_broadcast_and_scalar_rows():
    broadcast = _row(
        "1,24,6144;1,24,1",
        "1,24,6144",
        "DT_BF16;DT_BF16",
        "ND;ND",
    )
    scalar = _row(
        "1,24,64;",
        "1,24,64",
        "DT_BF16;DT_BF16",
        "NCL;ND",
    )

    assert validate_replay_row("Mul", broadcast).valid
    assert validate_replay_row("Mul", scalar).valid


def test_mul_validation_rejects_wrong_broadcast_output():
    result = validate_replay_row(
        "Mul",
        _row("1,6,6144;6144", "1,5,6144", "DT_BF16;DT_BF16", "ND;ND"),
    )

    assert not result.valid
    assert any("Mul output shape" in reason for reason in result.reasons)


def test_mul_validation_accepts_degenerate_bmm_rows():
    row = _row(
        "1,32,1;1,1,27",
        "1,32,27",
        "DT_BF16;DT_BF16",
        "ND;ND",
    )

    assert validate_replay_row("Mul", row).valid


def test_fill_validation_requires_rank_metadata_shape():
    valid = _row("2;", "27,55296", "INT64;BOOL", "ND;ND")
    valid["Output Data Types"] = "BOOL"
    invalid = _row("27,55296;", "27,55296", "INT64;BOOL", "ND;ND")
    invalid["Output Data Types"] = "BOOL"

    assert validate_replay_row("Fill", valid).valid
    result = validate_replay_row("Fill", invalid)
    assert not result.valid
    assert any("shape-metadata length" in reason for reason in result.reasons)


def test_transpose_batch_matmul_validation_enforces_alignment_and_output_layout():
    valid = _row(
        "4,48,256;4,256,256",
        "48,4,256",
        "DT_BF16;DT_BF16",
        "ND;ND",
    )
    invalid = _row(
        "64,16384,64;64,64,1024",
        "16384,64,1024",
        "DT_BF16;DT_BF16",
        "ND;ND",
    )

    assert validate_replay_row("TransposeBatchMatMul", valid).valid
    result = validate_replay_row("TransposeBatchMatMul", invalid)
    assert not result.valid
    assert any("divisible by 128" in reason for reason in result.reasons)


def test_quant_batch_matmul_v3_rejects_cann_x2_dimension_overflow():
    valid = _row(
        "1,2304;1210,144,16,32;38720;38720",
        "1,38720",
        "INT8;INT8;FLOAT;INT32",
        "ND;FRACTAL_NZ;ND;ND",
    )
    invalid = _row(
        "1,2304;4840,144,16,32;154880;154880",
        "1,154880",
        "INT8;INT8;FLOAT;INT32",
        "ND;FRACTAL_NZ;ND;ND",
    )

    assert validate_replay_row("QuantBatchMatmulV3", valid).valid
    result = validate_replay_row("QuantBatchMatmulV3", invalid)
    assert not result.valid
    assert any("must not exceed 65535" in reason for reason in result.reasons)


def test_kv_cache_validation_accepts_paged_decode_shape():
    row = _row(
        "6,1,1,576;512;6,1,1,64;6,1,1,64;6;48,128,1,64;48,128,1,512;;;;;",
        "48,128,1,64;48,128,1,512;6,1,1,64;6,1,1,512",
        (
            "DT_BF16;DT_BF16;DT_BF16;DT_BF16;INT64;DT_BF16;DT_BF16;"
            "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"
        ),
        "ND;ND;ND;ND;ND;ND;ND;NULL;NULL;NULL;NULL;NULL",
    )

    assert validate_replay_row("KvRmsNormRopeCache", row).valid


def test_kv_cache_validation_rejects_cache_width_mismatch():
    row = _row(
        "24,1,1,576;512;24,1,1,64;24,1,1,64;24;192,128,1,32;192,128,1,512;;;;;",
        "192,128,1,32;192,128,1,512;24,1,1,64;24,1,1,512",
        (
            "DT_BF16;DT_BF16;DT_BF16;DT_BF16;INT64;DT_BF16;DT_BF16;"
            "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"
        ),
        "ND;ND;ND;ND;ND;ND;ND;NULL;NULL;NULL;NULL;NULL",
    )

    result = validate_replay_row("KvRmsNormRopeCache", row)
    assert not result.valid
    assert any("cache widths" in reason for reason in result.reasons)


def test_validation_can_require_a_replay_script(tmp_path):
    result = validate_replay_row(
        "MissingKernel",
        _row("1,64", "1,64", "DT_BF16", "ND"),
        require_replay_script=True,
        replay_dir=tmp_path,
    )

    assert not result.valid
    assert "missing replay script" in result.reasons[0]


def test_cast_validation_rejects_double_tensors_on_ascend():
    row = _row("16", "16", "FLOAT", "ND")
    row["Output Data Types"] = "DOUBLE"

    result = validate_replay_row("Cast", row)

    assert not result.valid
    assert any("DOUBLE tensors are not replayable" in reason for reason in result.reasons)
