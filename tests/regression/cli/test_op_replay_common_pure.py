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

"""Tests for op_replay/common.py — pure functions (no NPU needed)."""

import argparse
import csv
import unittest

from tools.perf_data_collection.op_replay import merge_shard_results
from tools.perf_data_collection.signature_utils import get_sig
from tools.perf_data_collection.op_replay.common import (
    SUPPORTED_DEVICES,
    DEFAULT_DEVICE,
    DEFAULT_REPLAY_REPEAT_COUNT,
    SUPPORTED_UPDATE_MODES,
    MICROBENCH_DURATION,
    check_version,
    normalize_device_name,
    normalize_vllm_ascend_version,
    parse_list_field,
    split_metadata_field,
    parse_shape,
    parse_shape_or_none,
    normalize_dtype_name,
    normalize_op_name,
    expand_fractal_nz_shape,
    normalize_shape,
    build_version_dir_name,
    is_version_dir_name,
    _normalize_stack_component,
    INVALID_REPLAY_ROWS,
    get_runtime_replay_cases,
    has_real_duration,
    is_positive_finite,
    process_replay_csvs,
    record_runtime_replay_case,
    reset_runtime_replay_cases,
)


class TestConstants(unittest.TestCase):
    def test_supported_devices(self):
        self.assertIn(DEFAULT_DEVICE, SUPPORTED_DEVICES)
        self.assertGreater(len(SUPPORTED_DEVICES), 3)

    def test_default_replay_repeat_count(self):
        self.assertGreater(DEFAULT_REPLAY_REPEAT_COUNT, 0)

    def test_supported_update_modes(self):
        self.assertIn("all", SUPPORTED_UPDATE_MODES)
        self.assertIn("missing-only", SUPPORTED_UPDATE_MODES)

    def test_microbench_duration_column(self):
        self.assertEqual(MICROBENCH_DURATION, "Average Duration(us)")

    def test_invalid_replay_rows_is_list(self):
        self.assertIsInstance(INVALID_REPLAY_ROWS, list)

    def test_duration_must_be_positive_and_finite(self):
        self.assertTrue(is_positive_finite(0.1))
        for value in (0.0, -1.0, float("inf"), float("-inf"), float("nan")):
            self.assertFalse(is_positive_finite(value))

    def test_csv_duration_rejects_non_finite_values(self):
        self.assertTrue(has_real_duration({MICROBENCH_DURATION: "0.1"}, MICROBENCH_DURATION))
        for value in ("", "0", "-1", "inf", "-inf", "nan"):
            self.assertFalse(has_real_duration({MICROBENCH_DURATION: value}, MICROBENCH_DURATION))

    def test_shard_merge_ignores_non_finite_duration(self):
        row = {
            "Average Duration(us)": "inf",
            "Profiling Average Duration(us)": "2.5",
            "Profiling Median Duration(us)": "nan",
        }
        self.assertEqual(merge_shard_results._row_best_duration(row), 2.5)


class TestCheckVersion(unittest.TestCase):
    def test_valid_simple(self):
        self.assertEqual(check_version("0.9.2"), "0.9.2")

    def test_valid_with_v(self):
        self.assertIsNotNone(check_version("v0.13.0"))

    def test_valid_with_underscore(self):
        self.assertIsNotNone(check_version("vllm0.18.0_torch2.9.0_cann8.5"))

    def test_invalid_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_version("bad version with spaces")


class TestNormalizeDeviceName(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(normalize_device_name("  ATLAS_800  "), "ATLAS_800")


class TestNormalizeVllmAscendVersion(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(normalize_vllm_ascend_version("  0.13.0  "), "0.13.0")


class TestNormalizeStackComponent(unittest.TestCase):
    def test_vllm_prefix(self):
        self.assertEqual(_normalize_stack_component("vllm", "vllm0.18.0"), "0.18.0")

    def test_v_prefix(self):
        self.assertEqual(_normalize_stack_component("vllm", "v0.18.0"), "0.18.0")

    def test_torch_prefix(self):
        result = _normalize_stack_component("torch", "torch2.9.0+cpu")
        self.assertIn("2.9.0", result)

    def test_cann_prefix(self):
        self.assertEqual(_normalize_stack_component("cann", "cann8.5"), "8.5")


class TestBuildVersionDirName(unittest.TestCase):
    def test_standard(self):
        result = build_version_dir_name(
            vllm_ascend_version="0.18.0",
            torch_version="2.9.0",
            cann_version="8.5",
        )
        self.assertEqual(result, "vllm0.18.0_torch2.9.0_cann8.5")

    def test_with_prefixes(self):
        result = build_version_dir_name(
            vllm_ascend_version="v0.18.0",
            torch_version="torch2.9.0",
            cann_version="cann8.5",
        )
        self.assertEqual(result, "vllm0.18.0_torch2.9.0_cann8.5")


class TestIsVersionDirName(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_version_dir_name("vllm0.18.0_torch2.9.0_cann8.5"))

    def test_invalid(self):
        self.assertFalse(is_version_dir_name("not_a_version"))


class TestParseListField(unittest.TestCase):
    def test_semicolon(self):
        self.assertEqual(parse_list_field("a;b;c"), ["a", "b", "c"])

    def test_quoted(self):
        self.assertEqual(parse_list_field('"a;b;c"'), ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(parse_list_field(""), [])


class TestSplitMetadataField(unittest.TestCase):
    def test_semicolon(self):
        self.assertEqual(split_metadata_field("a;b"), ["a", "b"])

    def test_quoted(self):
        self.assertEqual(split_metadata_field('"a;b"'), ["a", "b"])

    def test_empty(self):
        self.assertEqual(split_metadata_field(""), [""])


class TestParseShape(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_shape("128,5120"), (128, 5120))

    def test_single_dim(self):
        self.assertEqual(parse_shape("4096"), (4096,))

    def test_scalar_marker(self):
        self.assertEqual(parse_shape("[]"), ())


class TestParseShapeOrNone(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_shape_or_none("128,5120"), (128, 5120))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_shape_or_none("  "))


class TestNormalizeDtypeName(unittest.TestCase):
    def test_with_prefix(self):
        self.assertEqual(normalize_dtype_name("DT_BF16"), "DT_BF16")

    def test_without_prefix(self):
        self.assertEqual(normalize_dtype_name("BF16"), "DT_BF16")

    def test_empty_returns_undefined(self):
        self.assertEqual(normalize_dtype_name(""), "DT_UNDEFINED")


class TestNormalizeOpName(unittest.TestCase):
    def test_removes_run_py(self):
        self.assertEqual(normalize_op_name("MatMulV2_run.py"), "MatMulV2")

    def test_removes_run(self):
        self.assertEqual(normalize_op_name("PadV3_run"), "PadV3")

    def test_removes_csv(self):
        self.assertEqual(normalize_op_name("SoftmaxV2.csv"), "SoftmaxV2")

    def test_passthrough(self):
        self.assertEqual(normalize_op_name("MatMulV2"), "MatMulV2")


class TestExpandFractalNzShape(unittest.TestCase):
    def test_valid(self):
        result = expand_fractal_nz_shape((2, 3, 4, 5))
        self.assertEqual(result, (12, 10))

    def test_invalid_dims_raises(self):
        with self.assertRaises(ValueError):
            expand_fractal_nz_shape((2, 3))


class TestNormalizeShape(unittest.TestCase):
    def test_regular_passthrough(self):
        self.assertEqual(normalize_shape((128, 5120), "ND"), (128, 5120))

    def test_fractal_nz_expands(self):
        result = normalize_shape((2, 3, 4, 5), "FRACTAL_NZ")
        self.assertEqual(result, (12, 10))


def _write_replay_csv(path):
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Input Shapes", "Average Duration(us)"])
        writer.writeheader()
        writer.writerow({"Input Shapes": "1,1", "Average Duration(us)": ""})


def test_runtime_failure_retains_shape_for_retry(tmp_path):
    csv_path = tmp_path / "Example.csv"
    _write_replay_csv(csv_path)

    def fail_at_runtime(*_args):
        raise RuntimeError("HCCL_BUFFSIZE is too small")

    _, failures, _, _ = process_replay_csvs(
        kernel_type="Example",
        csv_paths=[csv_path],
        repeat_count=1,
        run_row_fn=fail_at_runtime,
    )

    with csv_path.open("r", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert failures[0]["action"] == "retained"


def test_contract_failure_deletes_malformed_shape(tmp_path):
    csv_path = tmp_path / "Example.csv"
    _write_replay_csv(csv_path)

    def fail_contract(*_args):
        raise ValueError("invalid shape contract")

    _, failures, _, _ = process_replay_csvs(
        kernel_type="Example",
        csv_paths=[csv_path],
        repeat_count=1,
        run_row_fn=fail_contract,
    )

    with csv_path.open("r", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows == []
    assert failures[0]["action"] == "deleted"


def test_mid_repeat_failure_rolls_back_runtime_case(tmp_path):
    csv_path = tmp_path / "Example.csv"
    _write_replay_csv(csv_path)
    reset_runtime_replay_cases()
    calls = 0

    def fail_second_repeat(path, row_index, _row):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second repeat failed")
        record_runtime_replay_case(
            kernel_type="Example",
            case_id="case-1",
            csv_path=path,
            row_index=row_index,
        )

    process_replay_csvs(
        kernel_type="Example",
        csv_paths=[csv_path],
        repeat_count=2,
        run_row_fn=fail_second_repeat,
    )

    assert get_runtime_replay_cases() == []


def test_repeats_use_fresh_rows_and_copy_back_only_declared_fields(tmp_path):
    csv_path = tmp_path / "Example.csv"
    _write_replay_csv(csv_path)
    received_rows = []

    def mutate_repeat(_path, _row_index, row):
        received_rows.append(dict(row))
        row["temporary state"] = "must not leak"
        row[MICROBENCH_DURATION] = str(len(received_rows))

    process_replay_csvs(
        kernel_type="Example",
        csv_paths=[csv_path],
        repeat_count=2,
        run_row_fn=mutate_repeat,
        copy_back_fields=(MICROBENCH_DURATION,),
    )

    assert received_rows == [
        {"Input Shapes": "1,1", MICROBENCH_DURATION: ""},
        {"Input Shapes": "1,1", MICROBENCH_DURATION: ""},
    ]
    with csv_path.open("r", encoding="utf-8-sig") as csv_file:
        written = list(csv.DictReader(csv_file))
    assert written == [{"Input Shapes": '"1,1"', MICROBENCH_DURATION: "2"}]


if __name__ == "__main__":
    unittest.main()


class TestStringSignatureDtypeDistinction(unittest.TestCase):
    """as_str signatures must keep dtype-distinct rows from colliding.

    FLOAT and BF16 variants of the same MatMul shape are distinct perf cases.
    merge_shard_results dedups by the as_str signature, so the string form must
    carry dtype/format identity; otherwise one variant is silently dropped.
    """

    def _matmul_row(self, dtype: str, duration: str = "1.0") -> dict[str, str]:
        return {
            "OP State": "static",
            "Input Shapes": '"1,6144;38720,6144"',
            "Input Data Types": dtype,
            "Input Formats": "ND;ND",
            "Output Shapes": '"1,38720"',
            "Output Data Types": dtype,
            "Average Duration(us)": duration,
        }

    def test_matmul_float_and_bf16_get_distinct_string_sigs(self):
        float_row = self._matmul_row("FLOAT;FLOAT")
        bf16_row = self._matmul_row("DT_BF16;DT_BF16")
        float_sig = get_sig(float_row, as_str=True, op_name="MatMulV2")
        bf16_sig = get_sig(bf16_row, as_str=True, op_name="MatMulV2")
        self.assertNotEqual(float_sig, bf16_sig)

    def test_matmul_same_dtype_collapses_to_same_string_sig(self):
        a = self._matmul_row("FLOAT;FLOAT", "1.0")
        b = self._matmul_row("FLOAT;FLOAT", "2.0")
        self.assertEqual(
            get_sig(a, as_str=True, op_name="MatMulV2"),
            get_sig(b, as_str=True, op_name="MatMulV2"),
        )

    def test_general_op_dtype_distinction(self):
        base = {
            "OP State": "static",
            "Input Shapes": '"1024"',
            "Input Data Types": "DT_BF16",
            "Input Formats": "ND",
            "Output Shapes": '"1024"',
            "Output Data Types": "DT_BF16",
        }
        fp16 = dict(base, **{"Input Data Types": "DT_FLOAT16", "Output Data Types": "DT_FLOAT16"})
        self.assertNotEqual(
            get_sig(base, as_str=True, op_name="DynamicQuant"),
            get_sig(fp16, as_str=True, op_name="DynamicQuant"),
        )
