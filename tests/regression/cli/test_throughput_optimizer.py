# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import csv
import os
import re
import shutil
import sys
import tempfile
from argparse import Namespace
from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
from serving_cast.service.optimizer_summary import SHOW_COLUMNS
from tests.helpers.cli_runner import run_module_main

from cli.inference._batch_cases import (
    CSV_CONFIG_HEADER,
    DEFAULT_TPOT_LIMIT_MS,
    _build_optimizer_args,
    _csv_header_and_ref_row,
    _export_op_profile_csv,
    _filter_best_row,
    _summary_to_csv_row,
    _write_op_csv,
    load_cases_from_csv,
    run_cases_and_save,
)

THROUGHPUT_OPTIMIZER_MODULE = "cli.inference.throughput_optimizer"

# Match current PD titles and legacy Aggregation / Disaggregation (Prefill|Decode) titles across branches.
AGG_TABLE_TITLE_RE = r"Top\s+\d+\s+(?:PD\s+Aggregated|Aggregation)\s+Configurations\s*:?"
DISAGG_PREFILL_TITLE_RE = (
    r"Top\s+\d+\s+(?:PD\s+Disaggregated\s+Prefill|Disaggregation\s+\(Prefill\))\s+Configurations\s*:?"
)
DISAGG_DECODE_TITLE_RE = (
    r"Top\s+\d+\s+(?:PD\s+Disaggregated\s+Decode|Disaggregation\s+\(Decode\))\s+Configurations\s*:?"
)


def parse_args_with_input_csv(argv):
    """Parse throughput_optimizer args with patched sys.argv (batch mode entry)."""
    from cli.inference import throughput_optimizer as throughput_optimizer_module

    full_argv = ["throughput_optimizer"] + argv
    with patch.object(sys, "argv", full_argv):
        return throughput_optimizer_module.arg_parse()


class TestThroughputOptimizer(TestCase):
    """Performance analysis script system test class"""

    def test_arg_parse_reserved_memory_default_is_ten(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertEqual(args.reserved_memory_gb, 10.0)

    def _run_throughput_optimizer(self, args, check=True):
        """Run throughput_optimizer's main() in-process so coverage sees the core path."""
        result = run_module_main(THROUGHPUT_OPTIMIZER_MODULE, args)
        if check and result.returncode != 0:
            raise RuntimeError(f"throughput_optimizer failed (rc={result.returncode}): {result.stderr}")
        return result

    def _validate_table_structure(self, output_text, required_columns, table_start_pattern):
        """Validate the overall table structure and format"""
        # Check for required sections
        required_sections = [
            "Input Configuration:",
            "Overall Best Configuration:",
        ]

        for section in required_sections:
            self.assertIsNotNone(
                re.search(section, output_text),
                f"Required section '{section}' not found in output",
            )

        # Check for table header columns
        header_line = None

        for line in output_text.split("\n"):
            if all(col in line for col in required_columns):
                header_line = line
                break

        self.assertIsNotNone(header_line, "Table header with required columns not found")

        # Check for table borders (prettytable format)
        border_pattern = r"\+-+\+"
        borders = re.findall(border_pattern, output_text)
        self.assertGreaterEqual(len(borders), 2, "Table borders not found or incomplete")

        # Check for data rows in table format
        data_row_pattern = r"\|\s*\d+\s*\|.*\|"
        data_rows = re.findall(data_row_pattern, output_text)
        self.assertGreaterEqual(len(data_rows), 1, "Table data rows not found")

        # Check for the specific table format
        self.assertIsNotNone(
            re.search(table_start_pattern, output_text),
            "Configurations table title not found",
        )

        # Throughput column may embed ANSI escape codes around the numeric cell.
        throughput_pattern = r"\|\s*\d+\s*\|[^\|\n]*\d+(?:\.\d+)?[^\|\n]*\|"
        throughput_matches = re.findall(throughput_pattern, output_text)
        self.assertGreaterEqual(len(throughput_matches), 1, "Throughput values not found in table")

    def test_aggregation_functionality_with_output_validation(self):
        """Test aggregation functionality with comprehensive output validation"""
        args = [
            "--input-length=3500",
            "--output-length=1500",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--tpot-limits=50",
            "--compile",
        ]

        # Execute command
        result = self._run_throughput_optimizer(args, check=False)

        # Basic execution check
        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr

        # Validate table structure
        required_columns = SHOW_COLUMNS
        table_start_pattern = AGG_TABLE_TITLE_RE
        self._validate_table_structure(full_output, required_columns, table_start_pattern)

    def test_disaggregation_prefill_only_with_output_validation(self):
        """Test disaggregation prefill only functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--ttft-limits=1000",
            "--compile",
            "--disagg",
        ]

        # Execute command
        result = self._run_throughput_optimizer(args, check=False)

        # Basic execution check
        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        # Validate table structure
        local_columns = SHOW_COLUMNS.copy()
        local_columns.remove("TPOT (ms)")
        table_start_pattern = DISAGG_PREFILL_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)

    def test_disaggregation_decode_only_with_output_validation(self):
        """Test disaggregation decode only functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--tpot-limits=50",
            "--compile",
            "--disagg",
            "--tp-sizes",
            "2",
            "4",
            "--batch-range",
            "1",
            "8",
        ]

        # Execute command
        result = self._run_throughput_optimizer(args, check=False)

        # Basic execution check
        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        # Validate table structure
        local_columns = SHOW_COLUMNS.copy()
        local_columns.remove("TTFT (ms)")
        table_start_pattern = DISAGG_DECODE_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)

    def test_prefix_cache_hit_rate_rejects_invalid_value(self):
        args = [
            "--input-length=20",
            "--output-length=128",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--prefix-cache-hit-rate=1.0",
        ]

        result = self._run_throughput_optimizer(args, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("valid range [0, 1)", result.stderr)

    def test_prefix_cache_hit_rate_aggregation_valid(self):
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--tpot-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_prefix_cache_hit_rate_disaggregation_prefill_valid(self):
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--ttft-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
            "--disagg",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_prefix_cache_hit_rate_disaggregation_decode_valid(self):
        args = [
            "--input-length=64",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--tpot-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
            "--disagg",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_prefix_cache_hit_rate_allows_chunked_prefill_when_effective_input_exceeds_max_batched_tokens(
        self,
    ):
        args = [
            "--input-length=200",
            "--output-length=16",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--jobs=1",
            "--tpot-limits=1000",
            "--batch-range",
            "1",
            "2",
            "--prefix-cache-hit-rate=0.5",
            "--max-batched-tokens=99",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_deepseek_model_pd_ratio_with_output_validation(self):
        """Test deepseek model PD ratio with comprehensive output validation"""
        args = [
            "--input-length=3500",
            "--output-length=1500",
            "deepseek-ai/DeepSeek-V3.1",
            "--enable-optimize-prefill-decode-ratio",
            "--prefill-devices-per-instance=32",
            "--decode-devices-per-instance=32",
            "--compile",
            "--quantize-linear-action=W8A8_DYNAMIC",
            "--quantize-attention-action=INT8",
            "--device=TEST_DEVICE",
            "--jobs=10",
            "--ttft-limits=7000",
            "--tpot-limits=200",
        ]

        result = self._run_throughput_optimizer(args)

        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        full_output = result.stdout + result.stderr
        local_columns = [
            "Top",
            "PD Ratio",
            "P QPS (req/s)",
            "D QPS (req/s)",
            "TTFT (ms)",
            "TPOT (ms)",
            "P Parallel",
            "D Parallel",
            "P Devices/Instance",
            "D Devices/Instance",
            "P Batch Size",
            "D Batch Size",
            "P Concurrency",
            "D Concurrency",
        ]
        table_start_pattern = r"\s*Top\s+\d+\s+PD Ratio Configurations:"
        self._validate_table_structure(full_output, local_columns, table_start_pattern)


@pytest.mark.nightly
class TestThroughputOptimizerNightly(TestCase):
    def _run_throughput_optimizer(self, args, check=True):
        return TestThroughputOptimizer._run_throughput_optimizer(self, args, check)

    def _validate_table_structure(self, output_text, required_columns, table_start_pattern):
        return TestThroughputOptimizer._validate_table_structure(
            self, output_text, required_columns, table_start_pattern
        )

    def test_vl_model_aggregation_with_output_validation(self):
        """Test VL model aggregation functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "--device=TEST_DEVICE",
            "--num-devices=4",
            "--tpot-limits=100",
            "--image-height=512",
            "--image-width=512",
        ]

        result = self._run_throughput_optimizer(args)

        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        full_output = result.stdout + result.stderr
        local_columns = SHOW_COLUMNS.copy()
        table_start_pattern = AGG_TABLE_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)

    def test_vl_model_disaggregation_prefill_with_output_validation(self):
        """Test VL model disaggregation prefill only functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--ttft-limits=2000",
            "--image-height=512",
            "--image-width=512",
            "--disagg",
            "--batch-range",
            "1",
            "8",
        ]

        result = self._run_throughput_optimizer(args)

        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        full_output = result.stdout + result.stderr
        local_columns = SHOW_COLUMNS.copy()
        local_columns.remove("TPOT (ms)")
        table_start_pattern = DISAGG_PREFILL_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)

    def test_vl_model_disaggregation_decode_with_output_validation(self):
        """Test VL model disaggregation decode only functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "zai-org/GLM-4.5V",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--tpot-limits=100",
            "--image-height=512",
            "--image-width=512",
            "--disagg",
        ]

        result = self._run_throughput_optimizer(args)

        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        full_output = result.stdout + result.stderr
        local_columns = SHOW_COLUMNS.copy()
        local_columns.remove("TTFT (ms)")
        table_start_pattern = DISAGG_DECODE_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)

    def test_VL_MOE_model_aggregation_with_output_validation(self):
        """Test VL MOE model aggregation functionality with comprehensive output validation"""
        args = [
            "--input-length=20",
            "--output-length=128",
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--image-height=1080",
            "--image-width=1920",
            "--compile",
            "--quantize-linear-action=W8A8_DYNAMIC",
            "--quantize-attention-action=INT8",
            "--batch-range",
            "1",
            "4",
            "--max-batched-tokens=100",
        ]

        result = self._run_throughput_optimizer(args)

        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        full_output = result.stdout + result.stderr
        local_columns = SHOW_COLUMNS.copy()
        table_start_pattern = AGG_TABLE_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)


class TestBatchCasesMode(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cases_csv = os.path.join(self.tmp_dir, "cases.csv")
        self.results_csv = os.path.join(self.tmp_dir, "results.csv")

        with open(self.cases_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_CONFIG_HEADER)
            writer.writerow(
                [
                    "8card_agg_w8a8",
                    "ATLAS_800_A3_752T_128G_DIE",
                    "8",
                    "Qwen/Qwen3-32B",
                    "3500",
                    "1500",
                    "",
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
            )
            writer.writerow(
                [
                    "4card_disagg_ep",
                    "ATLAS_800_A3_752T_128G_DIE",
                    "4",
                    "Qwen/Qwen3-32B",
                    "2000",
                    "500",
                    "",
                    "50",
                    "",
                    "W8A8_DYNAMIC",
                    "INT8",
                    "2",
                    "0",
                    "",
                    "true",
                    "disagg",
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
            writer.writerow(
                [
                    "1card_disagg_mtp",
                    "ATLAS_800_A3_752T_128G_DIE",
                    "1",
                    "Qwen/Qwen3-32B",
                    "16000",
                    "1000",
                    "",
                    "50",
                    "",
                    "W8A8_DYNAMIC",
                    "DISABLED",
                    "",
                    "3",
                    "0.9;0.6;0.4",
                    "true",
                    "disagg",
                    "16000",
                    "",
                    "0",
                    "8",
                    "info",
                    "32",
                    "0",
                    "false",
                ]
            )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run_throughput_optimizer(self, args, check=True):
        result = run_module_main(THROUGHPUT_OPTIMIZER_MODULE, args)
        if check and result.returncode != 0:
            raise RuntimeError(f"throughput_optimizer failed (rc={result.returncode}): {result.stderr}")
        return result

    def test_batch_mode_generates_results_csv(self):
        args = [
            "--input-csv",
            self.cases_csv,
            "--output-csv",
            self.results_csv,
        ]
        result = self._run_throughput_optimizer(args, check=False)

        if result.returncode != 0:
            self.fail(f"批处理执行失败 rc={result.returncode}: {result.stderr}")

        self.assertTrue(os.path.exists(self.results_csv), "results.csv 未生成")

        with open(self.results_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]
        expected_columns = [
            "Case_Name",
            "Device Type",
            "Decode_TPOT(ms)",
            "Prefill_TTFT(ms)",
            "QuantizeLinearAction_options",
            "QuantizeAttentionAction_options",
            "Decode_Total TPS",
        ]
        for col in expected_columns:
            self.assertIn(col, header, f"表头缺少列: {col}")
        self.assertEqual(len(header), 40, f"表头列数应为 40，实际 {len(header)}")

        ref_row = rows[1]
        self.assertEqual(len(ref_row), 40)

        self.assertTrue(ref_row[-2], "ref_row 倒数第二列（QuantizeLinearAction_options）应为非空")
        self.assertTrue(ref_row[-1], "ref_row 最后一列（QuantizeAttentionAction_options）应为非空")

        decode_tps_idx = header.index("Decode_Total TPS")
        case_rows = rows[2:5]
        self.assertEqual(len(case_rows), 3, f"应有 3 个 case 行，实际 {len(case_rows)}")

        first_case_row = case_rows[0]
        self.assertEqual(first_case_row[0], "8card_agg_w8a8")
        self.assertTrue(
            first_case_row[decode_tps_idx],
            f"8card_agg_w8a8 的 Decode_Total TPS 应非空，实际为 {first_case_row[decode_tps_idx]!r}",
        )

    def test_load_cases_from_csv_basic(self):
        cases = load_cases_from_csv(self.cases_csv)
        self.assertEqual(len(cases), 3)

        c0 = cases[0]
        self.assertEqual(c0["case_name"], "8card_agg_w8a8")
        self.assertEqual(c0["device"], "ATLAS_800_A3_752T_128G_DIE")
        self.assertEqual(c0["num_devices"], 8)
        self.assertEqual(c0["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(c0["input_length"], 3500)
        self.assertEqual(c0["output_length"], 1500)
        self.assertEqual(c0["mode"], "agg")
        self.assertTrue(c0["do_compile"])

        c1 = cases[1]
        self.assertEqual(c1["case_name"], "4card_disagg_ep")
        self.assertEqual(c1["mode"], "disagg")
        self.assertEqual(c1["ep_sizes"], [2])

        c2 = cases[2]
        self.assertEqual(c2["case_name"], "1card_disagg_mtp")
        self.assertEqual(c2["num_mtp_tokens"], 3)
        self.assertEqual(c2["mtp_acceptance_rate"], [0.9, 0.6, 0.4])

    def test_load_cases_from_csv_missing_required_column(self):
        bad_header = ["case_name", "num_devices", "model_id", "input_length", "output_length"]
        bad_csv = os.path.join(self.tmp_dir, "bad.csv")
        with open(bad_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(bad_header)
            writer.writerow(["test", "1", "model", "100", "50"])

        with self.assertRaises(ValueError) as ctx:
            load_cases_from_csv(bad_csv)
        self.assertIn("missing required columns", str(ctx.exception))
        self.assertIn("device", str(ctx.exception))

    def test_load_cases_from_csv_empty_row_skipped(self):
        with open(self.cases_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] * len(CSV_CONFIG_HEADER))

        cases = load_cases_from_csv(self.cases_csv)
        self.assertEqual(len(cases), 3, "空行应被跳过")

    def test_load_cases_from_csv_auto_case_name(self):
        empty_name_csv = os.path.join(self.tmp_dir, "empty_name.csv")
        with open(empty_name_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(CSV_CONFIG_HEADER)

            writer.writerow(
                [
                    "",
                    "DEV",
                    "1",
                    "model",
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

        cases = load_cases_from_csv(empty_name_csv)

        self.assertEqual(len(cases), 1)

        self.assertEqual(cases[0]["case_name"], "row_2")

    def test_load_cases_from_csv_invalid_quantize_linear_isolated(self):
        invalid_csv = os.path.join(self.tmp_dir, "invalid.csv")
        with open(invalid_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(CSV_CONFIG_HEADER)

            writer.writerow(
                [
                    "test",
                    "DEV",
                    "1",
                    "model",
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
            )

        cases = load_cases_from_csv(invalid_csv)

        self.assertEqual(len(cases), 1)

        self.assertIn("parse_error", cases[0])

        self.assertIn("quantize_linear_action", cases[0]["parse_error"])

        self.assertIn("INVALID_QUANT", cases[0]["parse_error"])

        self.assertIn("W8A8_DYNAMIC", cases[0]["parse_error"])

        self.assertEqual(cases[0]["case_name"], "test")

    def test_load_cases_from_csv_empty_tpot_uses_default(self):
        empty_tpot_csv = os.path.join(self.tmp_dir, "empty_tpot.csv")
        with open(empty_tpot_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_CONFIG_HEADER)
            writer.writerow(
                [
                    "test",
                    "DEV",
                    "1",
                    "model",
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
            )

        cases = load_cases_from_csv(empty_tpot_csv)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["tpot_limits"], [DEFAULT_TPOT_LIMIT_MS])
        self.assertEqual(cases[0]["tpot_limits"][0], 50.0)

    def _make_case_dict(self, **overrides):
        defaults = {
            "case_name": "test",
            "device": "DEV",
            "num_devices": 8,
            "model_id": "model",
            "input_length": 100,
            "output_length": 50,
            "ttft_limits": [2000.0],
            "tpot_limits": [50.0],
            "tp_sizes": None,
            "quantize_linear_action": None,
            "quantize_attention_action": None,
            "ep_sizes": None,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": None,
            "do_compile": False,
            "mode": "agg",
            "max_batched_tokens": 8192,
            "batch_range": None,
            "serving_cost": 0.0,
            "jobs": 8,
            "log_level": "info",
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "compile_allow_graph_break": False,
        }
        defaults.update(overrides)
        return defaults

    def _make_base_args(self, **overrides):
        defaults = {
            "dump_original_results": False,
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
            "prefix_cache_hit_rate": 0.0,
            "enable_multistream": True,
            "enable_optimize_prefill_decode_ratio": False,
            "moe_dp_sizes": None,
            "concurrency_search_strategy": "exponential",
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_build_optimizer_args_agg_mode(self):
        case_dict = self._make_case_dict(mode="agg")
        args = _build_optimizer_args(case_dict, self._make_base_args())
        self.assertFalse(args.disagg)

    def test_build_optimizer_args_disagg_mode(self):
        case_dict = self._make_case_dict(mode="disagg")
        args = _build_optimizer_args(case_dict, self._make_base_args())
        self.assertTrue(args.disagg)

    def test_build_optimizer_args_single_limit_extraction(self):
        case_dict = self._make_case_dict(ttft_limits=[2000.0], tpot_limits=[50.0])
        args = _build_optimizer_args(case_dict, self._make_base_args())
        self.assertEqual(args.ttft_limits, 2000.0)
        self.assertEqual(args.tpot_limits, 50.0)

    def test_build_optimizer_args_concurrency_strategy_inherited(self):
        base_args = self._make_base_args(concurrency_search_strategy="linear")
        case_dict = self._make_case_dict()
        args = _build_optimizer_args(case_dict, base_args)
        self.assertEqual(args.concurrency_search_strategy, "linear")

    def test_parse_bool_invalid_value_raises(self):
        from cli.inference._batch_cases import _parse_bool

        for invalid in ["ture", "flase", "yes_please", "2", "maybe"]:
            with self.assertRaises(ValueError):
                _parse_bool(invalid)

    def test_parse_bool_empty_returns_false(self):
        from cli.inference._batch_cases import _parse_bool

        self.assertFalse(_parse_bool(None))

        self.assertFalse(_parse_bool(""))

    def test_parse_bool_valid_values(self):
        from cli.inference._batch_cases import _parse_bool

        for true_val in ["true", "1", "yes", "TRUE", "True"]:
            self.assertTrue(_parse_bool(true_val))

        for false_val in ["false", "0", "no", "FALSE", "False"]:
            self.assertFalse(_parse_bool(false_val))

    def test_parse_mode_invalid_value_raises(self):
        from cli.inference._batch_cases import _parse_mode

        for invalid in ["disag", "aggr", "disaggregation", "aggregation", "unknown"]:
            with self.assertRaises(ValueError):
                _parse_mode(invalid)

    def test_parse_mode_empty_defaults_to_agg(self):
        from cli.inference._batch_cases import _parse_mode

        self.assertEqual(_parse_mode(None), "agg")

        self.assertEqual(_parse_mode(""), "agg")

    def test_parse_mode_case_insensitive(self):
        from cli.inference._batch_cases import _parse_mode

        self.assertEqual(_parse_mode("AGG"), "agg")

        self.assertEqual(_parse_mode("Disagg"), "disagg")

    def test_load_cases_from_csv_invalid_row_isolated(self):
        mixed_csv = os.path.join(self.tmp_dir, "mixed.csv")
        with open(mixed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(CSV_CONFIG_HEADER)

            writer.writerow(
                [
                    "valid_case",
                    "DEV",
                    "1",
                    "model",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "W8A8_DYNAMIC",
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

            writer.writerow(
                [
                    "bad_mode",
                    "DEV",
                    "1",
                    "model",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "W8A8_DYNAMIC",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "disag",
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

            writer.writerow(
                [
                    "another_valid",
                    "DEV",
                    "1",
                    "model",
                    "100",
                    "50",
                    "",
                    "50",
                    "",
                    "W8A8_DYNAMIC",
                    "",
                    "",
                    "0",
                    "",
                    "false",
                    "disagg",
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

        cases = load_cases_from_csv(mixed_csv)

        self.assertEqual(len(cases), 3)

        self.assertEqual(cases[0]["case_name"], "valid_case")
        self.assertNotIn("parse_error", cases[0])

        self.assertIn("parse_error", cases[1])

        self.assertIn("Invalid mode", cases[1]["parse_error"])

        self.assertEqual(cases[1]["case_name"], "bad_mode")

        self.assertNotIn("parse_error", cases[2])
        self.assertEqual(cases[2]["case_name"], "another_valid")

    def test_run_cases_and_save_skips_parse_error_case(self):
        from unittest.mock import patch

        error_csv = os.path.join(self.tmp_dir, "with_error.csv")
        output_csv = os.path.join(self.tmp_dir, "out.csv")
        with open(error_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(CSV_CONFIG_HEADER)

            writer.writerow(
                [
                    "bad_compile",
                    "DEV",
                    "1",
                    "model",
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
                    "ture",
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

        with patch("serving_cast.parallel_runner.ParallelRunner"):
            run_cases_and_save(error_csv, output_csv, self._make_base_args())

        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        self.assertEqual(len(rows), 3)

        self.assertEqual(rows[2][0], "bad_compile")

        decode_tps_idx = rows[0].index("Decode_Total TPS")

        self.assertEqual(rows[2][decode_tps_idx], "")

    def test_filter_best_row_empty_df_returns_none(self):
        import pandas as pd

        empty_df = pd.DataFrame(columns=["parallel", "tpot", "ttft", "token/s"])
        mock_summary = Mock()
        mock_summary.get_summary_df.return_value = empty_df
        mock_data_config = Mock()
        mock_data_config.tpot_limits = 50.0
        mock_data_config.ttft_limits = None
        mock_summary.data_config = mock_data_config

        result = _filter_best_row(mock_summary)
        self.assertIsNone(result)

    def test_summary_to_csv_row_length_matches_header(self):
        case_dict = self._make_case_dict()
        row = _summary_to_csv_row(case_dict, [])
        header, _ = _csv_header_and_ref_row()
        self.assertEqual(len(row), len(header))
        self.assertEqual(len(row), 40)

    def test_csv_header_and_ref_row_structure(self):
        header, ref_row = _csv_header_and_ref_row()
        self.assertEqual(len(header), 40)
        self.assertEqual(len(ref_row), 40)

        self.assertTrue(ref_row[-2], "ref_row 倒数第二列（QuantizeLinearAction_options）应为非空")
        self.assertTrue(ref_row[-1], "ref_row 最后一列（QuantizeAttentionAction_options）应为非空")

    def test_export_op_profile_flag_default_false(self):
        args = parse_args_with_input_csv(["--input-csv", "cases.csv"])
        assert args.export_op_profile is False

    def test_export_op_profile_flag_set_true(self):
        args = parse_args_with_input_csv(["--input-csv", "cases.csv", "--export-op-profile"])
        assert args.export_op_profile is True

    def test_write_op_csv_content(self):
        from tensor_cast.core.model_runner import OpProfileSummary

        ops = OpProfileSummary(
            rows=[
                {
                    "name": "aten.mm.default",
                    "perf_model": "analytic",
                    "perf_total": 0.5,
                    "perf_avg": 0.1,
                    "call_times": 5,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "test_op.csv")
            _write_op_csv(filepath, [("prefill", ops)])

            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows[0] == ["Phase", "Op_Name", "Perf_Model", "Perf_Total_s", "Perf_Avg_s", "Call_Times"]
            assert rows[1] == ["prefill", "aten.mm.default", "analytic", "0.5", "0.1", "5"]

    def test_export_op_profile_agg_mode(self):
        import pandas as pd
        from tensor_cast.core.model_runner import OpProfileSummary

        ops = OpProfileSummary(
            rows=[
                {
                    "name": "aten.mm.default",
                    "perf_model": "analytic",
                    "perf_total": 0.5,
                    "perf_avg": 0.1,
                    "call_times": 5,
                },
            ]
        )
        df = pd.DataFrame(
            [
                {
                    "parallel": "tp4pp1dp1",
                    "batch_size": 8,
                    "token/s": 1000.0,
                    "tpot": 30.0,
                    "ttft": 500.0,
                }
            ]
        )
        mock_dc = Mock()
        mock_dc.ttft_limits = 2000.0
        mock_dc.tpot_limits = 50.0
        mock_summary = Mock()
        mock_summary.get_summary_df.return_value = df
        mock_summary.data_config = mock_dc
        mock_summary.get_op_profile_for.return_value = {"prefill": ops, "decode": ops}

        case_dict = self._make_case_dict(case_name="agg_case")
        with tempfile.TemporaryDirectory() as out_dir:
            _export_op_profile_csv(case_dict, [mock_summary], out_dir)

            op_file = os.path.join(out_dir, "op_profiles", "agg_case.csv")
            assert os.path.exists(op_file)
            with open(op_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows[0] == ["Phase", "Op_Name", "Perf_Model", "Perf_Total_s", "Perf_Avg_s", "Call_Times"]
            phases = [r[0] for r in rows[1:]]
            assert "prefill" in phases
            assert "decode" in phases

    def test_export_op_profile_disagg_mode(self):
        import pandas as pd
        from tensor_cast.core.model_runner import OpProfileSummary

        ops = OpProfileSummary(
            rows=[
                {
                    "name": "aten.mm.default",
                    "perf_model": "analytic",
                    "perf_total": 0.5,
                    "perf_avg": 0.1,
                    "call_times": 5,
                },
            ]
        )
        df = pd.DataFrame(
            [
                {
                    "parallel": "tp4pp1dp1",
                    "batch_size": 8,
                    "token/s": 1000.0,
                    "tpot": 30.0,
                    "ttft": 500.0,
                }
            ]
        )

        prefill_dc = Mock()
        prefill_dc.ttft_limits = 2000.0
        prefill_dc.tpot_limits = None
        prefill_summary = Mock()
        prefill_summary.get_summary_df.return_value = df
        prefill_summary.data_config = prefill_dc
        prefill_summary.get_op_profile_for.return_value = {"prefill": ops}

        decode_dc = Mock()
        decode_dc.ttft_limits = None
        decode_dc.tpot_limits = 50.0
        decode_summary = Mock()
        decode_summary.get_summary_df.return_value = df
        decode_summary.data_config = decode_dc
        decode_summary.get_op_profile_for.return_value = {"decode": ops}

        case_dict = self._make_case_dict(case_name="disagg_case")
        with tempfile.TemporaryDirectory() as out_dir:
            _export_op_profile_csv(case_dict, [prefill_summary, decode_summary], out_dir)

            prefill_file = os.path.join(out_dir, "op_profiles", "disagg_case_prefill.csv")
            decode_file = os.path.join(out_dir, "op_profiles", "disagg_case_decode.csv")
            assert os.path.exists(prefill_file)
            assert os.path.exists(decode_file)

    def test_export_op_profile_disabled_no_directory(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp_dir:
            cases_csv = os.path.join(tmp_dir, "cases.csv")
            with open(cases_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_CONFIG_HEADER)
                writer.writerow(
                    [
                        "test",
                        "DEV",
                        "1",
                        "model",
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
            output_csv = os.path.join(tmp_dir, "results.csv")

            empty_df = pd.DataFrame(columns=["parallel", "tpot", "ttft", "token/s", "batch_size"])
            mock_dc = Mock()
            mock_dc.tpot_limits = 50.0
            mock_dc.ttft_limits = None
            mock_summary = Mock()
            mock_summary.get_summary_df.return_value = empty_df
            mock_summary.data_config = mock_dc

            with patch("serving_cast.parallel_runner.ParallelRunner") as mock_runner_cls:
                mock_runner = Mock()
                mock_runner.run_agg.return_value = [mock_summary]
                mock_runner_cls.return_value = mock_runner
                run_cases_and_save(
                    cases_csv,
                    output_csv,
                    self._make_base_args(),
                    export_op_profile=False,
                )

            op_dir = os.path.join(tmp_dir, "op_profiles")
            assert not os.path.exists(op_dir)

    def test_export_op_profile_skips_missing_op_profile(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "parallel": "tp4pp1dp1",
                    "batch_size": 8,
                    "token/s": 1000.0,
                    "tpot": 30.0,
                    "ttft": 500.0,
                }
            ]
        )
        mock_dc = Mock()
        mock_dc.ttft_limits = 2000.0
        mock_dc.tpot_limits = 50.0
        mock_summary = Mock()
        mock_summary.get_summary_df.return_value = df
        mock_summary.data_config = mock_dc
        mock_summary.get_op_profile_for.return_value = None  # 无 op profile

        case_dict = self._make_case_dict(case_name="skip_case")
        with tempfile.TemporaryDirectory() as out_dir:
            _export_op_profile_csv(case_dict, [mock_summary], out_dir)

            op_dir = os.path.join(out_dir, "op_profiles")
            assert not os.path.exists(op_dir)

    def test_single_case_mode_warning(self):
        import logging

        log_records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = _CaptureHandler(logging.WARNING)

        target_logger = logging.getLogger("cli.inference.throughput_optimizer")
        target_logger.addHandler(handler)
        try:
            args = ["--export-op-profile", "TEST_MODEL", "--input-length", "100", "--output-length", "100"]
            result = self._run_throughput_optimizer(args, check=False)
        finally:
            target_logger.removeHandler(handler)

        assert any("only effective with --input-csv" in record.getMessage() for record in log_records)

        result = self._run_throughput_optimizer(["--export-op-profile"], check=False)

        assert result.returncode == 2

        assert "model_id is required" in result.stderr
