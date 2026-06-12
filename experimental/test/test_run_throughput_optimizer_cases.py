#!/usr/bin/env python
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for optix.run_throughput_optimizer_cases."""

import csv
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure experimental directory is on sys.path so that optix package is importable
experimental_dir = str(Path(__file__).resolve().parents[1])
if experimental_dir not in sys.path:
    sys.path.insert(0, experimental_dir)

# Ensure project root is on sys.path so that tensor_cast package is importable
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optix.run_throughput_optimizer_cases import (  # noqa: E402
    DEFAULT_TPOT_LIMIT_MS,
    FLUSH_BATCH_SIZE,
    LOG_LEVELS,
    BenchmarkCase,
    BenchmarkResult,
    CSV_CONFIG_HEADER,
    _build_optimizer_args,
    _configure_logging,
    _csv_header_and_ref_row,
    _filter_best_row,
    _parse_args,
    _parse_bool,
    _parse_list_float,
    _parse_list_int,
    _parse_mode,
    _parse_optional_bool,
    _parse_parallel,
    _result_row,
    _safe_float,
    _single_limit,
    load_cases_from_csv,
    save_results_to_csv,
    write_template_csv,
)
from tensor_cast.core.quantization.datatypes import (  # noqa: E402
    QuantizeLinearAction,
    QuantizeAttentionAction,
)


class TestParseListFloat(unittest.TestCase):
    def test_normal_semicolon_separated(self):
        # Parse a semicolon-separated list of floats into a Python list
        self.assertEqual(_parse_list_float("1.0;2.0;3.0"), [1.0, 2.0, 3.0])

    def test_single_value(self):
        # A single value without separators returns a one-element list
        self.assertEqual(_parse_list_float("50.0"), [50.0])

    def test_empty_string(self):
        # Empty string returns an empty list
        self.assertEqual(_parse_list_float(""), [])

    def test_none(self):
        # None input returns an empty list
        self.assertEqual(_parse_list_float(None), [])

    def test_whitespace_only(self):
        # Whitespace-only string returns an empty list
        self.assertEqual(_parse_list_float("   "), [])

    def test_values_with_spaces(self):
        # Leading/trailing spaces around values are trimmed
        self.assertEqual(_parse_list_float(" 1.0 ; 2.0 "), [1.0, 2.0])


class TestParseListInt(unittest.TestCase):
    def test_normal(self):
        # Parse a semicolon-separated list of integers into a Python list
        self.assertEqual(_parse_list_int("1;2;4"), [1, 2, 4])

    def test_single_value(self):
        # A single integer value returns a one-element list
        self.assertEqual(_parse_list_int("8"), [8])

    def test_empty_returns_none(self):
        # Empty string returns None (distinct from _parse_list_float which returns [])
        self.assertIsNone(_parse_list_int(""))

    def test_none(self):
        # None input returns None
        self.assertIsNone(_parse_list_int(None))


class TestParseBool(unittest.TestCase):
    def test_true_variants(self):
        # Recognizes "true", "True", "1", "yes", "YES" as True
        for v in ("true", "True", "1", "yes", "YES"):
            self.assertTrue(_parse_bool(v), f"Expected True for '{v}'")

    def test_false_variants(self):
        # "false", "0", "no", and any unrecognized string are treated as False
        for v in ("false", "0", "no", "random"):
            self.assertFalse(_parse_bool(v), f"Expected False for '{v}'")

    def test_none(self):
        # None input returns False
        self.assertFalse(_parse_bool(None))

    def test_empty(self):
        # Empty string returns False
        self.assertFalse(_parse_bool(""))


class TestParseOptionalBool(unittest.TestCase):
    def test_true(self):
        # "true" is parsed as True
        self.assertTrue(_parse_optional_bool("true"))

    def test_false(self):
        # "false" is parsed as False
        self.assertFalse(_parse_optional_bool("false"))

    def test_empty_returns_none(self):
        # Empty string returns None (no value provided)
        self.assertIsNone(_parse_optional_bool(""))

    def test_none_returns_none(self):
        # None input returns None
        self.assertIsNone(_parse_optional_bool(None))

    def test_invalid_returns_none(self):
        # Unrecognized string returns None instead of raising an error
        self.assertIsNone(_parse_optional_bool("maybe"))


class TestParseMode(unittest.TestCase):
    def test_agg(self):
        # "agg" is returned as-is for aggregation mode
        self.assertEqual(_parse_mode("agg"), "agg")

    def test_disagg(self):
        # "disagg" is returned as-is for disaggregation mode
        self.assertEqual(_parse_mode("disagg"), "disagg")

    def test_default_empty(self):
        # Empty string defaults to "agg"
        self.assertEqual(_parse_mode(""), "agg")

    def test_default_none(self):
        # None defaults to "agg"
        self.assertEqual(_parse_mode(None), "agg")

    def test_invalid_falls_back_to_agg(self):
        # Unrecognized value falls back to "agg" instead of raising
        self.assertEqual(_parse_mode("invalid"), "agg")


class TestParseParallel(unittest.TestCase):
    def test_valid(self):
        # Parse compact format "tp1pp1dp1" into (1, 1, 1)
        self.assertEqual(_parse_parallel("tp1pp1dp1"), (1, 1, 1))

    def test_multi_digit(self):
        # Parse compact format with multi-digit values
        self.assertEqual(_parse_parallel("tp2pp3dp4"), (2, 3, 4))

    def test_empty(self):
        # Empty string returns (None, None, None)
        self.assertEqual(_parse_parallel(""), (None, None, None))

    def test_invalid_format(self):
        # Unrecognized string returns (None, None, None)
        self.assertEqual(_parse_parallel("abc"), (None, None, None))

    def test_none(self):
        # None input returns (None, None, None)
        self.assertEqual(_parse_parallel(None), (None, None, None))

    def test_verbose_format(self):
        # Parse verbose format "TP=4 | PP=1 | DP=1" from disagg optimizer output
        self.assertEqual(_parse_parallel("TP=4 | PP=1 | DP=1"), (4, 1, 1))

    def test_verbose_format_lower(self):
        # Parse verbose format in lowercase
        self.assertEqual(_parse_parallel("tp=2 | pp=3 | dp=4"), (2, 3, 4))

    def test_verbose_partial(self):
        # Parse verbose format with missing component (PP absent)
        self.assertEqual(_parse_parallel("TP=4 | DP=1"), (4, None, 1))


class TestSingleLimit(unittest.TestCase):
    def test_empty_returns_none(self):
        # Empty list means no limit was provided; return None
        self.assertIsNone(_single_limit([], "test_field"))

    def test_single_value_returns_value(self):
        # A single-element list returns that value directly
        self.assertEqual(_single_limit([50.0], "test_field"), 50.0)

    def test_multiple_raises_value_error(self):
        # Multiple values are ambiguous for a single-limit field; raise ValueError
        with self.assertRaises(ValueError) as ctx:
            _single_limit([1.0, 2.0], "ttft_limits")
        self.assertIn("ttft_limits", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    def test_error_message_contains_name_and_count(self):
        # Error message includes the field name and the count of values provided
        with self.assertRaises(ValueError) as ctx:
            _single_limit([1.0, 2.0, 3.0], "my_limits")
        msg = str(ctx.exception)
        self.assertIn("my_limits", msg)
        self.assertIn("3", msg)


class TestLoadCasesFromCsv(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, rows, header=None):
        path = os.path.join(self.tmpdir, "cases.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header is None:
                header = CSV_CONFIG_HEADER
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        return path

    def test_basic_load(self):
        # A well-formed CSV row is parsed into a BenchmarkCase with correct field values
        path = self._write_csv(
            [
                [
                    "test_case",
                    "TEST_DEVICE",
                    "8",
                    "Qwen/Qwen3-32B",
                    "3500",
                    "1500",
                    "2000",
                    "50",
                    "",
                    "W8A8_DYNAMIC",
                    "DISABLED",
                    "",
                    "0",
                    "",
                    "true",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            ]
        )
        cases = load_cases_from_csv(path)
        self.assertEqual(len(cases), 1)
        c = cases[0]
        self.assertEqual(c.case_name, "test_case")
        self.assertEqual(c.device, "TEST_DEVICE")
        self.assertEqual(c.num_devices, 8)
        self.assertEqual(c.model_id, "Qwen/Qwen3-32B")
        self.assertEqual(c.input_length, 3500)
        self.assertEqual(c.output_length, 1500)
        self.assertEqual(c.ttft_limits, [2000.0])
        self.assertEqual(c.tpot_limits, [50.0])
        self.assertEqual(c.quantize_linear_action, QuantizeLinearAction.W8A8_DYNAMIC)
        self.assertEqual(c.quantize_attention_action, QuantizeAttentionAction.DISABLED)
        self.assertTrue(c.do_compile)
        self.assertEqual(c.mode, "agg")

    def test_empty_tpot_uses_default_ms(self):
        # When tpot_limits is empty in CSV, the default 50.0 ms is applied
        path = self._write_csv(
            [
                [
                    "test_case",
                    "TEST_DEVICE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "100",
                    "50",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            ]
        )
        cases = load_cases_from_csv(path)
        self.assertEqual(cases[0].tpot_limits, [DEFAULT_TPOT_LIMIT_MS])

    def test_invalid_quantize_linear_raises(self):
        # An invalid quantize_linear_action value raises ValueError with the invalid value listed
        path = self._write_csv(
            [
                [
                    "test_case",
                    "TEST_DEVICE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "INVALID_QUANT",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            load_cases_from_csv(path)
        msg = str(ctx.exception)
        self.assertIn("quantize_linear_action", msg)
        self.assertIn("INVALID_QUANT", msg)
        self.assertIn("Valid options:", msg)

    def test_invalid_quantize_attention_raises(self):
        # An invalid quantize_attention_action value raises ValueError with the invalid value listed
        path = self._write_csv(
            [
                [
                    "test_case",
                    "TEST_DEVICE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "",
                    "BAD_ATTN",
                    "",
                    "0",
                    "",
                    "false",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            load_cases_from_csv(path)
        msg = str(ctx.exception)
        self.assertIn("quantize_attention_action", msg)
        self.assertIn("BAD_ATTN", msg)
        self.assertIn("Valid options:", msg)

    def test_error_message_lists_valid_quantize_options(self):
        # Error message for invalid quantize_linear_action includes all valid enum values
        path = self._write_csv(
            [
                [
                    "test_case",
                    "TEST_DEVICE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "NOPE",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            load_cases_from_csv(path)
        msg = str(ctx.exception)
        for action in QuantizeLinearAction:
            self.assertIn(action.value, msg)

    def test_no_header_raises(self):
        # A CSV file with no content at all raises ValueError about missing header
        path = os.path.join(self.tmpdir, "empty.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        with self.assertRaises(ValueError) as ctx:
            load_cases_from_csv(path)
        self.assertIn("no header", str(ctx.exception))

    def test_empty_rows_skipped(self):
        # A row with all empty cells is skipped and produces no cases
        path = self._write_csv([[""] * len(CSV_CONFIG_HEADER)])
        cases = load_cases_from_csv(path)
        self.assertEqual(len(cases), 0)

    def test_missing_case_name_gets_row_n(self):
        # When case_name is empty, it is auto-generated as "row_N"
        path = self._write_csv(
            [
                [
                    "",
                    "TEST_DEVICE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            ]
        )
        cases = load_cases_from_csv(path)
        self.assertEqual(cases[0].case_name, "row_1")


class TestWriteTemplateCsv(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_template_has_correct_header(self):
        # Template CSV header matches CSV_CONFIG_HEADER exactly
        path = os.path.join(self.tmpdir, "template.csv")
        write_template_csv(path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, CSV_CONFIG_HEADER)

    def test_template_has_example_rows(self):
        # Template contains multiple example rows with known case_names and model_ids
        path = os.path.join(self.tmpdir, "template.csv")
        write_template_csv(path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], "1card_agg_w8a8")
        self.assertEqual(rows[1][0], "8card_agg_w8a8")
        self.assertEqual(rows[2][0], "4card_disagg_mtp")
        for row in rows:
            self.assertEqual(row[3], "Qwen/Qwen3-32B")

    def test_template_example_tpot_is_50ms(self):
        # All template example rows use the default TPOT limit of 50 ms
        path = os.path.join(self.tmpdir, "template.csv")
        write_template_csv(path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.assertEqual(row[7], str(int(DEFAULT_TPOT_LIMIT_MS)))


class TestBuildOptimizerArgs(unittest.TestCase):
    def _make_case(self, **overrides):
        defaults = dict(
            case_name="test",
            device="TEST_DEVICE",
            num_devices=1,
            model_id="Qwen/Qwen3-32B",
            input_length=100,
            output_length=50,
            ttft_limits=[2000.0],
            tpot_limits=[50.0],
        )
        defaults.update(overrides)
        return BenchmarkCase(**defaults)

    def test_agg_mode_no_disagg(self):
        # agg mode sets disagg=False on the resulting Namespace
        case = self._make_case(mode="agg")
        args = _build_optimizer_args(case)
        self.assertFalse(args.disagg)

    def test_disagg_mode_sets_disagg_true(self):
        # disagg mode sets disagg=True on the resulting Namespace
        case = self._make_case(mode="disagg")
        args = _build_optimizer_args(case)
        self.assertTrue(args.disagg)

    def test_single_limit_applied_to_ttft(self):
        # A single ttft_limits value is extracted and set as args.ttft_limits
        case = self._make_case(ttft_limits=[2000.0])
        args = _build_optimizer_args(case)
        self.assertEqual(args.ttft_limits, 2000.0)

    def test_single_limit_applied_to_tpot(self):
        # A single tpot_limits value is extracted and set as args.tpot_limits
        case = self._make_case(tpot_limits=[50.0])
        args = _build_optimizer_args(case)
        self.assertEqual(args.tpot_limits, 50.0)

    def test_empty_ttft_gives_none(self):
        # Empty ttft_limits list results in args.ttft_limits=None (no constraint)
        case = self._make_case(ttft_limits=[])
        args = _build_optimizer_args(case)
        self.assertIsNone(args.ttft_limits)

    def test_multiple_ttft_raises(self):
        # Multiple ttft_limits values raise ValueError (only single value allowed)
        case = self._make_case(ttft_limits=[1.0, 2.0])
        with self.assertRaises(ValueError) as ctx:
            _build_optimizer_args(case)
        self.assertIn("ttft_limits", str(ctx.exception))

    def test_multiple_tpot_raises(self):
        # Multiple tpot_limits values raise ValueError (only single value allowed)
        case = self._make_case(tpot_limits=[50.0, 100.0])
        with self.assertRaises(ValueError) as ctx:
            _build_optimizer_args(case)
        self.assertIn("tpot_limits", str(ctx.exception))


class TestBenchmarkResult(unittest.TestCase):
    def test_csv_header_matches_result_row_length(self):
        # Output CSV header and _result_row have the same number of columns
        header, _ = _csv_header_and_ref_row()
        result = BenchmarkResult(
            case_name="test",
            device="DEV",
            num_devices=1,
            model_id="model",
            input_length=100,
            output_length=50,
        )
        row = _result_row(result)
        self.assertEqual(len(header), len(row))

    def test_csv_roundtrip(self):
        # A BenchmarkResult written to CSV and read back preserves key fields
        tmpdir = tempfile.mkdtemp()
        try:
            result = BenchmarkResult(
                case_name="test",
                device="DEV",
                num_devices=2,
                model_id="Qwen/Qwen3-32B",
                input_length=100,
                output_length=50,
                best_decode_tpot_ms=40.5,
                best_decode_total_tps=100.0,
                best_decode_tps_per_device=50.0,
                best_decode_tp_size=2,
                best_decode_pp_size=1,
                best_decode_dp_size=1,
                best_decode_concurrency=10,
            )
            path = os.path.join(tmpdir, "results.csv")
            save_results_to_csv([result], path)
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # ref header
                next(reader)  # ref row
                row = next(reader)
            self.assertEqual(row[0], "test")
            self.assertEqual(row[1], "DEV")
            self.assertIn("40.50", row[12])  # Decode_TPOT(ms)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_no_error_fields_in_result(self):
        # BenchmarkResult dataclass has no best_decode_error or best_prefill_error attributes
        result = BenchmarkResult(
            case_name="test",
            device="DEV",
            num_devices=1,
            model_id="model",
            input_length=100,
            output_length=50,
        )
        self.assertFalse(hasattr(result, "best_decode_error"))
        self.assertFalse(hasattr(result, "best_prefill_error"))


class TestParseArgs(unittest.TestCase):
    def test_input_csv(self):
        # --input-csv flag is parsed and returned as the first tuple element
        with patch.object(sys, "argv", ["prog", "--input-csv", "cases.csv"]):
            input_csv, _, _, _, _ = _parse_args()
        self.assertEqual(input_csv, "cases.csv")

    def test_write_template(self):
        # --write-template flag is parsed and returned as the second tuple element
        with patch.object(sys, "argv", ["prog", "--write-template", "tmpl.csv"]):
            _, write_template, _, _, _ = _parse_args()
        self.assertEqual(write_template, "tmpl.csv")

    def test_output_csv(self):
        # --output-csv flag is parsed and returned as the third tuple element
        with patch.object(sys, "argv", ["prog", "--output-csv", "out.csv"]):
            _, _, output_csv, _, _ = _parse_args()
        self.assertEqual(output_csv, "out.csv")

    def test_test_conversion_flag(self):
        # --test-conversion flag sets the fourth tuple element to True
        with patch.object(sys, "argv", ["prog", "--test-conversion"]):
            _, _, _, test_conv, _ = _parse_args()
        self.assertTrue(test_conv)

    def test_defaults(self):
        # When no flags are provided, all values are None or False
        with patch.object(sys, "argv", ["prog"]):
            input_csv, write_template, output_csv, test_conv, validate_csv = _parse_args()
        self.assertIsNone(input_csv)
        self.assertIsNone(write_template)
        self.assertIsNone(output_csv)
        self.assertFalse(test_conv)
        self.assertIsNone(validate_csv)

    def test_help_exits(self):
        # --help triggers SystemExit with exit code 0
        with patch.object(sys, "argv", ["prog", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                _parse_args()
            self.assertEqual(ctx.exception.code, 0)


class TestSaveResultsToCsv(unittest.TestCase):
    def test_results_written_correctly(self):
        # Multiple BenchmarkResult objects are written to CSV with case names present
        tmpdir = tempfile.mkdtemp()
        try:
            results = [
                BenchmarkResult(
                    case_name="case1",
                    device="DEV1",
                    num_devices=1,
                    model_id="model1",
                    input_length=100,
                    output_length=50,
                ),
                BenchmarkResult(
                    case_name="case2",
                    device="DEV2",
                    num_devices=2,
                    model_id="model2",
                    input_length=200,
                    output_length=100,
                ),
            ]
            path = os.path.join(tmpdir, "results.csv")
            save_results_to_csv(results, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("case1", content)
            self.assertIn("case2", content)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_batch_flush_constant(self):
        # FLUSH_BATCH_SIZE is 10 (flush CSV every 10 cases for crash safety)
        self.assertEqual(FLUSH_BATCH_SIZE, 10)


class TestDefaultTpotLimitMs(unittest.TestCase):
    def test_default_is_50_ms(self):
        # Default TPOT limit constant is 50.0 milliseconds
        self.assertEqual(DEFAULT_TPOT_LIMIT_MS, 50.0)


class TestSafeFloat(unittest.TestCase):
    def test_normal_float(self):
        # Normal float value is returned as-is
        self.assertEqual(_safe_float(3.14), 3.14)

    def test_none_returns_none(self):
        # None input returns None
        self.assertIsNone(_safe_float(None))

    def test_nan_returns_none(self):
        # float('nan') returns None (not a number is treated as missing)
        self.assertIsNone(_safe_float(float("nan")))

    def test_inf_returns_none(self):
        # float('inf') returns None (infinity is treated as missing)
        self.assertIsNone(_safe_float(float("inf")))

    def test_negative_inf_returns_none(self):
        # float('-inf') returns None
        self.assertIsNone(_safe_float(float("-inf")))

    def test_string_float(self):
        # String "3.14" is converted to float
        self.assertEqual(_safe_float("3.14"), 3.14)

    def test_invalid_string_returns_none(self):
        # Non-numeric string returns None
        self.assertIsNone(_safe_float("abc"))

    def test_int_returns_float(self):
        # Integer input is converted to float
        self.assertEqual(_safe_float(42), 42.0)


class TestRequiredColumns(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_required_columns_raises(self):
        # CSV missing required columns (e.g., 'device') raises ValueError with column name
        path = os.path.join(self.tmpdir, "bad.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case_name", "model_id", "input_length", "output_length"])
            writer.writerow(["test", "Qwen/Qwen3-32B", "100", "50"])
        with self.assertRaises(ValueError) as ctx:
            load_cases_from_csv(path)
        msg = str(ctx.exception)
        self.assertIn("missing required columns", msg)
        self.assertIn("device", msg)

    def test_all_required_columns_present(self):
        # CSV with all required columns does not raise
        path = os.path.join(self.tmpdir, "good.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_CONFIG_HEADER)
            writer.writerow(
                [
                    "test",
                    "TEST_DEVICE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "agg",
                    "8192",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            )
        cases = load_cases_from_csv(path)
        self.assertEqual(len(cases), 1)


class TestValidateCsvArg(unittest.TestCase):
    def test_validate_csv_flag(self):
        # --validate-csv flag is parsed and returned as the fifth tuple element
        with patch.object(sys, "argv", ["prog", "--validate-csv", "cases.csv"]):
            _, _, _, _, validate_csv = _parse_args()
        self.assertEqual(validate_csv, "cases.csv")


class TestFilterBestRow(unittest.TestCase):
    """Tests for _filter_best_row using only public OptimizerSummary API."""

    def _make_summary(self, df, tpot_limits=None, ttft_limits=None):
        class MockDataConfig:
            pass

        dc = MockDataConfig()
        dc.tpot_limits = tpot_limits
        dc.ttft_limits = ttft_limits

        class MockSummary:
            data_config = dc

            def get_summary_df(self):
                return df

        return MockSummary()

    def test_returns_best_row_when_within_slo(self):
        # When a row meets the SLO, _filter_best_row returns it
        import pandas as pd

        df = pd.DataFrame(
            [
                {"parallel": "tp1pp1dp1", "tpot": 40.0, "ttft": None, "token/s": 100.0},
                {"parallel": "tp2pp1dp1", "tpot": 30.0, "ttft": None, "token/s": 200.0},
            ]
        )
        summary = self._make_summary(df, tpot_limits=50.0, ttft_limits=None)
        best = _filter_best_row(summary)
        self.assertIsNotNone(best)
        self.assertEqual(best["parallel"], "tp2pp1dp1")  # highest token/s under SLO
        self.assertEqual(best["token/s"], 200.0)

    def test_returns_none_when_no_row_meets_slo(self):
        # When all rows exceed the SLO, returns None
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "parallel": "tp1pp1dp1",
                    "tpot": 100.0,
                    "ttft": None,
                    "token/s": 100.0,
                },
            ]
        )
        summary = self._make_summary(df, tpot_limits=50.0, ttft_limits=None)
        self.assertIsNone(_filter_best_row(summary))

    def test_returns_none_for_empty_df(self):
        # Empty DataFrame returns None
        import pandas as pd

        df = pd.DataFrame(columns=["parallel", "tpot", "ttft", "token/s"])
        summary = self._make_summary(df, tpot_limits=50.0, ttft_limits=None)
        self.assertIsNone(_filter_best_row(summary))

    def test_prefill_phase_isolates_ttft_filter(self):
        # Prefill summary has tpot_limits=None; only TTFT is filtered
        # A row with tpot=None and ttft=1500 should pass when ttft_limit=2000
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "parallel": "tp1pp1dp1",
                    "tpot": None,
                    "ttft": 1500.0,
                    "token/s": 100.0,
                },
            ]
        )
        summary = self._make_summary(df, tpot_limits=None, ttft_limits=2000.0)
        best = _filter_best_row(summary)
        self.assertIsNotNone(best)
        self.assertEqual(best["ttft"], 1500.0)

    def test_decode_phase_isolates_tpot_filter(self):
        # Decode summary has ttft_limits=None; only TPOT is filtered
        # A row with ttft=None and tpot=40 should pass when tpot_limit=50
        import pandas as pd

        df = pd.DataFrame(
            [
                {"parallel": "tp1pp1dp1", "tpot": 40.0, "ttft": None, "token/s": 100.0},
            ]
        )
        summary = self._make_summary(df, tpot_limits=50.0, ttft_limits=None)
        best = _filter_best_row(summary)
        self.assertIsNotNone(best)
        self.assertEqual(best["tpot"], 40.0)


class TestConfigureLogging(unittest.TestCase):
    """Tests for _configure_logging helper."""

    def test_known_level(self):
        # _configure_logging accepts a valid log level and sets the root logger
        import logging

        _configure_logging("debug")
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_unknown_level_falls_back_to_info(self):
        # Unknown log level falls back to INFO
        import logging

        _configure_logging("unknown_level")
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_log_levels_constant_present(self):
        # LOG_LEVELS dict exposes all standard levels
        for k in ("debug", "info", "warning", "error", "fatal", "critical"):
            self.assertIn(k, LOG_LEVELS)


class TestIntegrationExampleCase(unittest.TestCase):
    """Integration test based on example_cases.csv input and benchmark_cases_results.csv output.

    Input CSV row (from example_cases.csv):
        deepseek-ai/DeepSeek-V3,ATLAS_800_A3_752T_128G_DIE,64,
        deepseek-ai/DeepSeek-V3,3500,1000,,20,1,W8A8_DYNAMIC,DISABLED,,
        3,0.9;0.6;0.4,TRUE,disagg,16000,,0,8,critical,32,0,FALSE

    Expected output CSV row (from result.csv):
        deepseek-ai/DeepSeek-V3,ATLAS_800_A3_752T_128G_DIE,64,3500,1000,
        deepseek-ai/DeepSeek-V3,W8A8_DYNAMIC,DISABLED,,3,20.00,512,18.81,
        27216.8,425.3,68.50,13.86,17.12,0.52,1,1,64,
        (prefill fields empty)
    """

    # -- Input side: CSV row values --
    INPUT_CSV_ROW = [
        "deepseek-ai/DeepSeek-V3",
        "ATLAS_800_A3_752T_128G_DIE",
        "64",
        "deepseek-ai/DeepSeek-V3",
        "3500",
        "1000",
        "",
        "20",
        "1",
        "W8A8_DYNAMIC",
        "DISABLED",
        "",
        "3",
        "0.9;0.6;0.4",
        "TRUE",
        "disagg",
        "16000",
        "",
        "0",
        "8",
        "critical",
        "32",
        "0",
        "FALSE",
    ]

    # -- Expected BenchmarkCase field values after CSV load --
    EXPECTED_CASE = dict(
        case_name="deepseek-ai/DeepSeek-V3",
        device="ATLAS_800_A3_752T_128G_DIE",
        num_devices=64,
        model_id="deepseek-ai/DeepSeek-V3",
        input_length=3500,
        output_length=1000,
        ttft_limits=[],
        tpot_limits=[20.0],
        tp_sizes=[1],
        quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
        quantize_attention_action=QuantizeAttentionAction.DISABLED,
        ep_sizes=None,
        num_mtp_tokens=3,
        mtp_acceptance_rate=[0.9, 0.6, 0.4],
        do_compile=True,
        mode="disagg",
        max_prefill_tokens=16000,
        batch_range=None,
        serving_cost=0.0,
        jobs=8,
        log_level="critical",
        mxfp4_group_size=32,
        reserved_memory_gb=0.0,
        compile_allow_graph_break=False,
    )

    # -- Expected BenchmarkResult based on actual optimizer output --
    EXPECTED_RESULT = dict(
        case_name="deepseek-ai/DeepSeek-V3",
        device="ATLAS_800_A3_752T_128G_DIE",
        num_devices=64,
        model_id="deepseek-ai/DeepSeek-V3",
        input_length=3500,
        output_length=1000,
        # Decode fields
        best_decode_linear_quant_type="W8A8_DYNAMIC",
        best_decode_attn_quant_type="DISABLED",
        best_decode_tp_size=1,
        best_decode_pp_size=1,
        best_decode_dp_size=64,
        best_decode_use_ep="",
        best_decode_mtp_tokens=3,
        best_decode_slo_target_ms=20.0,
        best_decode_concurrency=512,
        best_decode_tpot_ms=18.81,
        best_decode_total_tps=27216.8,
        best_decode_tps_per_device=425.3,
        best_decode_mem_pct="68.50",
        best_decode_comm_pct="13.86",
        best_decode_cube_pct="17.12",
        best_decode_vec_pct="0.52",
    )

    # -- Expected _result_row output (mix of raw ints and _csv_val-formatted strings) --
    EXPECTED_OUTPUT_ROW = [
        "deepseek-ai/DeepSeek-V3",
        "ATLAS_800_A3_752T_128G_DIE",
        64,
        3500,
        1000,
        "deepseek-ai/DeepSeek-V3",
        "W8A8_DYNAMIC",
        "DISABLED",
        "",
        "3",
        "20.00",
        "512",
        "18.81",
        "27216.8",
        "425.3",
        "68.50",
        "13.86",
        "17.12",
        "0.52",
        "1",
        "1",
        "64",
    ]

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, rows, header=None):
        path = os.path.join(self.tmpdir, "cases.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header is None:
                header = CSV_CONFIG_HEADER
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        return path

    # --- CSV load → BenchmarkCase verification ---

    def test_csv_load_parses_example_case(self):
        # Verify that the DeepSeek-V3 CSV row is loaded into BenchmarkCase with all fields correct,
        # including tp_sizes=[1], mtp_acceptance_rate=[0.9,0.6,0.4], mode="disagg", do_compile=True
        path = self._write_csv([self.INPUT_CSV_ROW])
        cases = load_cases_from_csv(path)
        self.assertEqual(len(cases), 1)
        c = cases[0]
        for field, expected in self.EXPECTED_CASE.items():
            actual = getattr(c, field)
            self.assertEqual(
                actual,
                expected,
                f"BenchmarkCase.{field}: expected {expected!r}, got {actual!r}",
            )

    # --- BenchmarkCase → _build_optimizer_args verification ---

    def test_build_optimizer_args_from_example_case(self):
        # Verify _build_optimizer_args produces correct Namespace for the DeepSeek-V3 case,
        # including disagg=True, tp_sizes=[1], and ParallelRunner-required defaults
        path = self._write_csv([self.INPUT_CSV_ROW])
        cases = load_cases_from_csv(path)
        args = _build_optimizer_args(cases[0])
        self.assertEqual(args.device, "ATLAS_800_A3_752T_128G_DIE")
        self.assertEqual(args.num_devices, 64)
        self.assertEqual(args.model_id, "deepseek-ai/DeepSeek-V3")
        self.assertEqual(args.input_length, 3500)
        self.assertEqual(args.output_length, 1000)
        self.assertEqual(args.tpot_limits, 20.0)
        self.assertIsNone(args.ttft_limits)
        self.assertTrue(args.disagg)
        self.assertTrue(args.compile)
        self.assertFalse(args.compile_allow_graph_break)
        self.assertEqual(args.num_mtp_tokens, 3)
        self.assertEqual(args.mtp_acceptance_rate, [0.9, 0.6, 0.4])
        self.assertEqual(args.quantize_linear_action, QuantizeLinearAction.W8A8_DYNAMIC)
        self.assertEqual(args.quantize_attention_action, QuantizeAttentionAction.DISABLED)
        self.assertEqual(args.tp_sizes, [1])
        self.assertEqual(args.max_prefill_tokens, 16000)
        self.assertEqual(args.log_level, "critical")
        # Default-filled attributes required by ParallelRunner
        self.assertIsNone(args.image_batch_size)
        self.assertEqual(args.prefix_cache_hit_rate, 0.0)
        self.assertFalse(args.enable_optimize_prefill_decode_ratio)

    # --- BenchmarkResult → _result_row → CSV output verification ---

    def test_result_row_from_example_output(self):
        # Verify _result_row output matches the actual result.csv data for decode fields,
        # and that prefill fields are empty (disagg decode-only case)
        result = BenchmarkResult(
            case_name=self.EXPECTED_RESULT["case_name"],
            device=self.EXPECTED_RESULT["device"],
            num_devices=self.EXPECTED_RESULT["num_devices"],
            model_id=self.EXPECTED_RESULT["model_id"],
            input_length=self.EXPECTED_RESULT["input_length"],
            output_length=self.EXPECTED_RESULT["output_length"],
            best_decode_linear_quant_type=self.EXPECTED_RESULT["best_decode_linear_quant_type"],
            best_decode_attn_quant_type=self.EXPECTED_RESULT["best_decode_attn_quant_type"],
            best_decode_tp_size=self.EXPECTED_RESULT["best_decode_tp_size"],
            best_decode_pp_size=self.EXPECTED_RESULT["best_decode_pp_size"],
            best_decode_dp_size=self.EXPECTED_RESULT["best_decode_dp_size"],
            best_decode_use_ep=self.EXPECTED_RESULT["best_decode_use_ep"],
            best_decode_mtp_tokens=self.EXPECTED_RESULT["best_decode_mtp_tokens"],
            best_decode_slo_target_ms=self.EXPECTED_RESULT["best_decode_slo_target_ms"],
            best_decode_concurrency=self.EXPECTED_RESULT["best_decode_concurrency"],
            best_decode_tpot_ms=self.EXPECTED_RESULT["best_decode_tpot_ms"],
            best_decode_total_tps=self.EXPECTED_RESULT["best_decode_total_tps"],
            best_decode_tps_per_device=self.EXPECTED_RESULT["best_decode_tps_per_device"],
            best_decode_mem_pct=self.EXPECTED_RESULT["best_decode_mem_pct"],
            best_decode_comm_pct=self.EXPECTED_RESULT["best_decode_comm_pct"],
            best_decode_cube_pct=self.EXPECTED_RESULT["best_decode_cube_pct"],
            best_decode_vec_pct=self.EXPECTED_RESULT["best_decode_vec_pct"],
        )
        row = _result_row(result)
        # Verify decode fields match expected output (first 22 columns)
        for i, expected in enumerate(self.EXPECTED_OUTPUT_ROW):
            self.assertEqual(row[i], expected, f"Column {i}: expected {expected!r}, got {row[i]!r}")
        # Prefill fields should be empty (no prefill result in disagg decode-only)
        for i in range(22, 38):
            self.assertEqual(row[i], "", f"Prefill column {i} should be empty, got {row[i]!r}")

    # --- Full CSV roundtrip: write CSV → load → build result → save → read back ---

    def test_full_csv_roundtrip(self):
        # End-to-end: write input CSV, load it, construct BenchmarkResult from expected
        # optimizer output, save to result CSV, read back and verify all fields
        path = self._write_csv([self.INPUT_CSV_ROW])
        cases = load_cases_from_csv(path)
        self.assertEqual(len(cases), 1)

        # Simulate result from optimizer
        result = BenchmarkResult(
            case_name=self.EXPECTED_RESULT["case_name"],
            device=self.EXPECTED_RESULT["device"],
            num_devices=self.EXPECTED_RESULT["num_devices"],
            model_id=self.EXPECTED_RESULT["model_id"],
            input_length=self.EXPECTED_RESULT["input_length"],
            output_length=self.EXPECTED_RESULT["output_length"],
            best_decode_linear_quant_type=self.EXPECTED_RESULT["best_decode_linear_quant_type"],
            best_decode_attn_quant_type=self.EXPECTED_RESULT["best_decode_attn_quant_type"],
            best_decode_tp_size=self.EXPECTED_RESULT["best_decode_tp_size"],
            best_decode_pp_size=self.EXPECTED_RESULT["best_decode_pp_size"],
            best_decode_dp_size=self.EXPECTED_RESULT["best_decode_dp_size"],
            best_decode_use_ep=self.EXPECTED_RESULT["best_decode_use_ep"],
            best_decode_mtp_tokens=self.EXPECTED_RESULT["best_decode_mtp_tokens"],
            best_decode_slo_target_ms=self.EXPECTED_RESULT["best_decode_slo_target_ms"],
            best_decode_concurrency=self.EXPECTED_RESULT["best_decode_concurrency"],
            best_decode_tpot_ms=self.EXPECTED_RESULT["best_decode_tpot_ms"],
            best_decode_total_tps=self.EXPECTED_RESULT["best_decode_total_tps"],
            best_decode_tps_per_device=self.EXPECTED_RESULT["best_decode_tps_per_device"],
            best_decode_mem_pct=self.EXPECTED_RESULT["best_decode_mem_pct"],
            best_decode_comm_pct=self.EXPECTED_RESULT["best_decode_comm_pct"],
            best_decode_cube_pct=self.EXPECTED_RESULT["best_decode_cube_pct"],
            best_decode_vec_pct=self.EXPECTED_RESULT["best_decode_vec_pct"],
        )

        # Save to CSV and read back
        out_path = os.path.join(self.tmpdir, "results.csv")
        save_results_to_csv([result], out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            ref_row = next(reader)
            data_row = next(reader)

        # Header matches expected
        expected_header, _ = _csv_header_and_ref_row()
        self.assertEqual(header, expected_header)

        # Ref row has quantize options
        self.assertTrue(len(ref_row[-2]) > 0)  # QuantizeLinearAction_options
        self.assertTrue(len(ref_row[-1]) > 0)  # QuantizeAttentionAction_options

        # Data row decode fields match expected output
        # csv.reader returns all strings, so compare as strings
        for i, expected in enumerate(self.EXPECTED_OUTPUT_ROW):
            self.assertEqual(
                data_row[i],
                str(expected),
                f"Column {i}: expected {expected!r}, got {data_row[i]!r}",
            )

    # --- Verify _parse_parallel handles disagg output format ---

    def test_parse_parallel_disagg_output(self):
        # The disagg optimizer outputs parallel as 'TP=1 | PP=1 | DP=64';
        # _parse_parallel must extract (1, 1, 64) from this format
        tp, pp, dp = _parse_parallel("TP=1 | PP=1 | DP=64")
        self.assertEqual(tp, 1)
        self.assertEqual(pp, 1)
        self.assertEqual(dp, 64)


if __name__ == "__main__":
    unittest.main()
