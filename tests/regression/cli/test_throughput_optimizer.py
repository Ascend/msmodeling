# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
import sys
from unittest import TestCase
from unittest.mock import patch

import pytest
from serving_cast.service.optimizer_summary import SHOW_COLUMNS
from tests.helpers.cli_runner import run_module_main

THROUGHPUT_OPTIMIZER_MODULE = "cli.inference.throughput_optimizer"

# Match current PD titles and legacy Aggregation / Disaggregation (Prefill|Decode) titles across branches.
AGG_TABLE_TITLE_RE = r"Top\s+\d+\s+(?:PD\s+Aggregated|Aggregation)\s+Configurations\s*:?"
DISAGG_PREFILL_TITLE_RE = (
    r"Top\s+\d+\s+(?:PD\s+Disaggregated\s+Prefill|Disaggregation\s+\(Prefill\))\s+Configurations\s*:?"
)
DISAGG_DECODE_TITLE_RE = (
    r"Top\s+\d+\s+(?:PD\s+Disaggregated\s+Decode|Disaggregation\s+\(Decode\))\s+Configurations\s*:?"
)


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

    def test_prefix_cache_hit_rate_respects_effective_input_length_for_max_prefill_tokens(
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
            "--max-prefill-tokens=99",
        ]

        result = self._run_throughput_optimizer(args, check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_main_uses_optimizer_data_effective_input_length_for_prefill_check(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        class DummyArgs:
            log_level = "error"
            input_length = 200
            output_length = 16
            prefix_cache_hit_rate = 0.5
            max_prefill_tokens = 99
            num_mtp_tokens = 0
            mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]
            disagg = False
            enable_optimize_prefill_decode_ratio = False
            device = ["TEST_DEVICE"]
            num_devices = 1

        with (
            patch.object(throughput_optimizer_module, "arg_parse", return_value=DummyArgs()),
            patch.object(
                throughput_optimizer_module,
                "check_device_targets",
                return_value=["TEST_DEVICE"],
            ),
            patch(
                "cli.inference.throughput_optimizer.OptimizerData.get_effective_input_length",
                return_value=100,
            ) as mock_get_effective_input_length,
        ):
            self.assertEqual(throughput_optimizer_module.main(), 1)
            mock_get_effective_input_length.assert_called_once_with()

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
            "--max-prefill-tokens=100",
        ]

        result = self._run_throughput_optimizer(args)

        if result.returncode != 0:
            self.fail(f"Script execution failed with return code {result.returncode}: {result.stderr}")

        full_output = result.stdout + result.stderr
        local_columns = SHOW_COLUMNS.copy()
        table_start_pattern = AGG_TABLE_TITLE_RE
        self._validate_table_structure(full_output, local_columns, table_start_pattern)
