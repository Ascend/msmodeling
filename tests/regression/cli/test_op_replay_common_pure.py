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
from tools.perf_data_collection.signature_utils import canonicalize_shape_columns, get_sig
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
    scalar_fill_value_for_dtype,
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


MERGE_SAMPLE_FIELDNAMES = [
    "OP State",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Average Duration(us)",
    "Profiling Average Duration(us)",
    "Profiling Median Duration(us)",
    "Profiling Std Duration(us)",
    "Profiling Average aiv_vec_time(us)",
]

# (mean, median, std, aiv_vec_time) of one aggregated sampling summary row.
SLOW_SAMPLE = ("100", "100", "10", "100")
FAST_SAMPLE = ("20", "20", "2", "20")


def _write_sample_shard(shard_dir, samples):
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (shard_dir / "Add.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGE_SAMPLE_FIELDNAMES)
        writer.writeheader()
        for mean, median, std, vec_time in samples:
            writer.writerow(
                {
                    "OP State": "static",
                    "Input Shapes": "4,8;4,8",
                    "Input Data Types": "DT_FLOAT16;DT_FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "4,8",
                    "Output Data Types": "DT_FLOAT16",
                    "Average Duration(us)": "",
                    "Profiling Average Duration(us)": mean,
                    "Profiling Median Duration(us)": median,
                    "Profiling Std Duration(us)": std,
                    "Profiling Average aiv_vec_time(us)": vec_time,
                }
            )


def _merge_sample_shards(tmp_path, slow_samples, fast_samples, slow_first=True):
    slow_dir = tmp_path / "slow"
    fast_dir = tmp_path / "fast"
    _write_sample_shard(slow_dir, slow_samples)
    _write_sample_shard(fast_dir, fast_samples)
    out_dir = tmp_path / "merged"
    shard_dirs = [slow_dir, fast_dir] if slow_first else [fast_dir, slow_dir]
    merge_shard_results.merge_shard_directories(shard_dirs, out_dir)
    with (out_dir / "Add.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class TestShardMergeKeepsWinningSampleWhole:
    """min-of-N merge must keep the winning sample's full measurement row.

    Regression for #500: when a faster shard row won, only the three
    duration columns were replaced and the remaining columns were only
    backfilled when empty, so the merged record mixed the fast shard's
    mean/median with the slow shard's std and hardware counters.
    """

    def test_faster_shard_replaces_whole_record(self, tmp_path):
        merged_rows = _merge_sample_shards(tmp_path, [SLOW_SAMPLE], [FAST_SAMPLE])
        assert len(merged_rows) == 1
        row = merged_rows[0]
        assert row["Profiling Average Duration(us)"] == "20"
        assert row["Profiling Median Duration(us)"] == "20"
        assert row["Profiling Std Duration(us)"] == "2"
        assert row["Profiling Average aiv_vec_time(us)"] == "20"

    def test_first_seen_fast_sample_is_not_polluted_by_later_slow_sample(self, tmp_path):
        merged_rows = _merge_sample_shards(tmp_path, [SLOW_SAMPLE], [FAST_SAMPLE], slow_first=False)
        # The winning sample must also be intact when the fast shard is the
        # first-seen one and the slow shard never wins the replacement.
        row = merged_rows[0]
        assert row["Profiling Average Duration(us)"] == "20"
        assert row["Profiling Std Duration(us)"] == "2"
        assert row["Profiling Average aiv_vec_time(us)"] == "20"

    def test_merge_is_symmetric_in_shard_order(self, tmp_path):
        forward = _merge_sample_shards(tmp_path, [SLOW_SAMPLE], [FAST_SAMPLE])
        reversed_rows = _merge_sample_shards(tmp_path, [SLOW_SAMPLE], [FAST_SAMPLE], slow_first=False)
        assert len(forward) == 1
        assert len(reversed_rows) == 1
        for row in (forward[0], reversed_rows[0]):
            assert row["Profiling Average Duration(us)"] == "20"
            assert row["Profiling Median Duration(us)"] == "20"
            assert row["Profiling Std Duration(us)"] == "2"
            assert row["Profiling Average aiv_vec_time(us)"] == "20"

    def test_multi_row_slow_shard_every_row_loses_to_fast_sample(self, tmp_path):
        # slow shard carries two aggregated summaries (as when the first row
        # is an older measurement); each must lose to the fast sample whole.
        merged_rows = _merge_sample_shards(
            tmp_path,
            [SLOW_SAMPLE, ("80", "80", "8", "80")],
            [FAST_SAMPLE],
        )
        assert len(merged_rows) == 1
        row = merged_rows[0]
        assert row["Profiling Average Duration(us)"] == "20"
        assert row["Profiling Std Duration(us)"] == "2"
        assert row["Profiling Average aiv_vec_time(us)"] == "20"

    def test_empty_winning_measurement_not_backfilled_from_loser(self, tmp_path):
        # Fast sample is first-seen (no replacement happens) but has empty
        # std/vec columns. The losing slower sample's non-empty values must
        # not fill them: "exists but empty" is the winning sample's own
        # measurement fact, and backfilling it would re-create the exact
        # cross-sample hybrid record that #500 forbids.
        merged_rows = _merge_sample_shards(tmp_path, [SLOW_SAMPLE], [("20", "20", "", "")])
        assert len(merged_rows) == 1
        row = merged_rows[0]
        assert row["Profiling Average Duration(us)"] == "20"
        assert row["Profiling Median Duration(us)"] == "20"
        assert row["Profiling Std Duration(us)"] == ""
        assert row["Profiling Average aiv_vec_time(us)"] == ""

    def test_missing_column_from_narrow_header_shard_still_backfilled(self, tmp_path):
        # Column-drift fallback must survive the tightened backfill: a shard
        # CSV with a narrower header yields rows that lack the column key
        # entirely, and those are still filled from later shards.
        slow_dir = tmp_path / "slow"
        fast_dir = tmp_path / "fast"
        fast_dir.mkdir(parents=True, exist_ok=True)
        narrow_fields = [f for f in MERGE_SAMPLE_FIELDNAMES if f != "Profiling Average aiv_vec_time(us)"]
        mean, median, std, _ = FAST_SAMPLE
        with (fast_dir / "Add.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=narrow_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "OP State": "static",
                    "Input Shapes": "4,8;4,8",
                    "Input Data Types": "DT_FLOAT16;DT_FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "4,8",
                    "Output Data Types": "DT_FLOAT16",
                    "Average Duration(us)": "",
                    "Profiling Average Duration(us)": mean,
                    "Profiling Median Duration(us)": median,
                    "Profiling Std Duration(us)": std,
                }
            )
        _write_sample_shard(slow_dir, [SLOW_SAMPLE])
        out_dir = tmp_path / "merged"
        merge_shard_results.merge_shard_directories([fast_dir, slow_dir], out_dir)
        with (out_dir / "Add.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["Profiling Average Duration(us)"] == "20"
        assert row["Profiling Std Duration(us)"] == "2"
        # Missing-key column is backfilled from the slower shard's row.
        assert row["Profiling Average aiv_vec_time(us)"] == "100"


class TestCanonicalizeShapeStorage(unittest.TestCase):
    """issue #510: DB write paths must emit canonical shape storage.

    Legacy "()" / "[]" scalar slots are rewritten to the empty-slot contract
    so the query-layer parser stays reachable, while real slot structure
    (trailing empty slots) is preserved verbatim.
    """

    def test_legacy_empty_tuple_slot_normalized_to_empty_slot(self):
        row = canonicalize_shape_columns({"Output Shapes": "1,1;()"})
        self.assertEqual(row["Output Shapes"], '"1,1;"')

    def test_legacy_empty_list_slot_normalized_to_empty_slot(self):
        row = canonicalize_shape_columns({"Output Shapes": "1,1;[]"})
        self.assertEqual(row["Output Shapes"], '"1,1;"')

    def test_trailing_empty_slot_is_preserved(self):
        # MaskedFill-style scalar value slots (issue #510) are stored as a
        # trailing empty slot; canonicalization must not shorten the schema.
        row = canonicalize_shape_columns({"Input Shapes": "144,6144;144,1;"})
        self.assertEqual(row["Input Shapes"], '"144,6144;144,1;"')

    def test_over_quoted_cell_collapses_to_one_quote_pair(self):
        row = canonicalize_shape_columns({"Input Shapes": '"1,2"'})
        self.assertEqual(row["Input Shapes"], '"1,2"')

    def test_bare_cell_gains_quote_pair(self):
        row = canonicalize_shape_columns({"Input Shapes": "1,2", "Output Shapes": "1,1;1"})
        self.assertEqual(row["Input Shapes"], '"1,2"')
        self.assertEqual(row["Output Shapes"], '"1,1;1"')

    def test_dimensioned_paren_shape_untouched(self):
        # A "(1,2,3)"-style slot has no defined scalar contract; canonicalize
        # must not guess a rewrite for it (TransData rank-5 semantics etc.).
        row = canonicalize_shape_columns({"Input Shapes": "(1,2,3)"})
        self.assertEqual(row["Input Shapes"], '"(1,2,3)"')

    def test_missing_shape_columns_left_untouched(self):
        row = {"OP State": "static", "Average Duration(us)": "1.0"}
        self.assertEqual(canonicalize_shape_columns(row), row)


class TestShardMergeCanonicalizesShapeStorage:
    """issue #510: the shard merge write path must emit canonical storage.

    Before this fix, merge_shard_results echoed raw shard rows verbatim, so
    bare (unquoted) shape cells and legacy "()" slots survived back into the
    database as quote variants of canonical rows.
    """

    def test_merge_wraps_bare_shape_cells_in_canonical_quotes(self, tmp_path):
        merged_rows = _merge_sample_shards(tmp_path, [SLOW_SAMPLE], [FAST_SAMPLE])
        assert len(merged_rows) == 1
        assert merged_rows[0]["Input Shapes"] == '"4,8;4,8"'
        assert merged_rows[0]["Output Shapes"] == '"4,8"'

    def test_merge_rewrites_legacy_empty_tuple_output_slot(self, tmp_path):
        shard_dir = tmp_path / "shard"
        shard_dir.mkdir(parents=True)
        with (shard_dir / "DynamicQuant.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MERGE_SAMPLE_FIELDNAMES)
            writer.writeheader()
            writer.writerow(
                {
                    "OP State": "static",
                    "Input Shapes": '"1,512"',
                    "Input Data Types": "DT_BF16",
                    "Input Formats": "ND",
                    "Output Shapes": '"1,512;()"',
                    "Output Data Types": "INT8;FLOAT",
                    "Average Duration(us)": "1.5",
                }
            )
        out_dir = tmp_path / "merged"
        merge_shard_results.merge_shard_directories([shard_dir], out_dir)
        with (out_dir / "DynamicQuant.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["Output Shapes"] == '"1,512;"'


class TestScalarFillValueForDtype(unittest.TestCase):
    """issue #510: MaskedFill rows record the scalar value slot by dtype only.

    The scalar value operand has an empty shape slot with an active dtype, so
    replay rebuilds it as a literal matching the recorded dtype family.
    """

    def test_integer_family_gets_int_zero(self):
        for dtype in (
            "INT32",
            "DT_INT32",
            "INT8",
            "DT_INT8",
            "DT_INT64",
            "UINT4",
            "DT_UINT4",
            "DT_UINT8",
            "DT_UINT16",
            "DT_UINT32",
            "DT_UINT64",
            "DT_BOOL",
        ):
            value = scalar_fill_value_for_dtype(dtype)
            self.assertIsInstance(value, int)
            self.assertEqual(value, 0)

    def test_float_family_gets_float_zero(self):
        for dtype in ("FLOAT", "DT_FLOAT", "DT_BF16", "DT_FLOAT16", "BF16"):
            value = scalar_fill_value_for_dtype(dtype)
            self.assertIsInstance(value, float)
            self.assertEqual(value, 0.0)
