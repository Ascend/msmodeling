# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
import sys
from unittest import TestCase
from unittest.mock import Mock, patch


import pytest

from serving_cast.service.optimizer_summary import SHOW_COLUMNS
from serving_cast.service.utils import build_pp_search_candidates
from tests.helpers.cli_runner import run_module_main
from tests.helpers.model_assets import vendored_model_config_path

THROUGHPUT_OPTIMIZER_MODULE = "cli.inference.throughput_optimizer"
LENGTH_DISTRIBUTION_PATH = "serving_cast/example/length_distribution.yaml"

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

    def test_arg_parse_max_batched_tokens_defaults_to_none(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertIsNone(args.max_batched_tokens)

    def test_arg_parse_num_mtp_token_sizes_defaults_to_empty_list(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertEqual(args.num_mtp_tokens, 0)
        self.assertEqual(args.num_mtp_token_sizes, [])

    def test_num_mtp_token_candidates_validate_acceptance_rate_length(self):
        # RFC §3.7 (gap #2.1): spec.validators now fires in parse_module_args()
        # before main() runs. The error goes to stderr via parser.error()
        # (argparse-style), not via logger.error(). The inline check in main()
        # is preserved as fallback but never reached.
        args = [
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--num-mtp-tokens",
            "0",
            "6",
        ]

        result = self._run_throughput_optimizer(args, check=False)

        self.assertNotEqual(result.returncode, 0)
        # Registry validator mtp_tokens_vs_acceptance_rate fires via
        # parser.error() with a message of the form:
        #   "num_mtp_tokens(6) must be ≤ len(mtp_acceptance_rate)+1 (5)"
        # Assert on the concrete candidate value (6), the limit (5 = len(default rates)+1),
        # and the relational operator — not just the field name.
        self.assertIn("num_mtp_tokens(6)", result.stderr)
        self.assertIn("len(mtp_acceptance_rate)+1 (5)", result.stderr)

    def test_pd_instance_device_arguments_require_pd_ratio_mode(self):
        args = [
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=1",
            "--disagg",
            "--prefill-devices-per-instance=1",
            "--decode-devices-per-instance=1",
        ]

        with self.assertLogs("cli.inference.throughput_optimizer", "ERROR") as logs:
            result = self._run_throughput_optimizer(args, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require --enable-optimize-prefill-decode-ratio", "\n".join(logs.output))

    def test_arg_parse_performance_model_defaults_to_analytic(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertEqual(args.performance_model, ["analytic"])

    def test_arg_parse_accepts_analytic_calibration_options(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "--analytic-calibration-profile=profile.sqlite",
            "--analytic-calibration-stack=cann8.5",
            "--performance-model=calibrated",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertEqual(args.analytic_calibration_profile, "profile.sqlite")
        self.assertEqual(args.analytic_calibration_stack, "cann8.5")
        self.assertEqual(args.performance_model, ["calibrated"])

    def test_arg_parse_profiling_without_database_errors(self):
        argv = [
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
            "--performance-model=profiling",
        ]

        result = self._run_throughput_optimizer(argv, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("--profiling-database", result.stderr)

    def test_arg_parse_performance_model_is_list(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "--performance-model=profiling",
            "--profiling-database=/tmp/fake_db",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertEqual(args.performance_model, ["profiling"])

    def test_arg_parse_performance_model_appends_when_repeated(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "--performance-model=analytic",
            "--performance-model=profiling",
            "--profiling-database=/tmp/fake_db",
            "Qwen/Qwen3-32B",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        self.assertEqual(args.performance_model, ["analytic", "profiling"])

    def test_arg_parse_performance_model_rejects_invalid_choice(self):
        """--performance-model should reject unsupported values."""
        argv = [
            "--input-length=1",
            "--output-length=1",
            "--performance-model=invalid",
            "Qwen/Qwen3-32B",
        ]

        result = self._run_throughput_optimizer(argv, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

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

    @pytest.mark.nightly
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
        # Registry-generated range message (argparse_adapter _float_type).
        self.assertIn("out of range; must be < 1", result.stderr)

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

    @pytest.mark.nightly
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
            "Balanced QPS (req/s)",
            "P QPS/Instance",
            "D QPS/Instance",
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

    def test_arg_parse_accepts_input_length_distribution_file(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            f"--input-length={LENGTH_DISTRIBUTION_PATH}",
            "--output-length=1",
            "test-model",
        ]

        with patch.object(sys, "argv", argv):
            args = throughput_optimizer_module.arg_parse()

        # Compare with forward slashes: resolve() normalizes to OS-native
        # separators (backslash on Windows).
        self.assertEqual(
            str(args.input_length).replace("\\", "/"),
            LENGTH_DISTRIBUTION_PATH.replace("\\", "/"),
        )
        self.assertFalse(hasattr(args, "length_distribution"))

    def test_arg_parse_rejects_invalid_input_length(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--input-length=not-a-length-or-file",
            "--output-length=1",
            "test-model",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                throughput_optimizer_module.arg_parse()

    def test_arg_parse_rejects_missing_input_length(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        argv = [
            "throughput_optimizer",
            "--output-length=1",
            "test-model",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                throughput_optimizer_module.arg_parse()

    def test_main_length_distribution_mode_rejects_decode_only_disagg(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        class DummyArgs:
            log_level = "error"
            model_id = "test-model"
            device = ["TEST_DEVICE"]
            num_devices = 1
            input_length = LENGTH_DISTRIBUTION_PATH
            word_embedding_tp = None
            output_length = 16
            prefix_cache_hit_rate = 0.5
            max_batched_tokens = 8192
            num_mtp_tokens = 0
            num_mtp_token_sizes = []
            mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]
            ttft_limits = None
            tpot_limits = 50
            disagg = True
            enable_optimize_prefill_decode_ratio = False
            compilation_config = None

        mock_tasks = Mock()
        mock_tasks.run_disagg.return_value = []

        with (
            patch.object(throughput_optimizer_module, "arg_parse", return_value=DummyArgs()),
            patch.object(
                throughput_optimizer_module,
                "check_device_targets",
                return_value=DummyArgs.device,
            ),
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_tasks),
        ):
            self.assertEqual(throughput_optimizer_module.main(), 1)
            mock_tasks.run_disagg.assert_not_called()

    def test_main_loads_length_distribution_into_runtime_args(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module
        from serving_cast.service.utils import LengthBin, LengthDistribution

        class DummyArgs:
            log_level = "error"
            model_id = "test-model"
            device = ["TEST_DEVICE"]
            num_devices = 1
            input_length = LENGTH_DISTRIBUTION_PATH
            word_embedding_tp = None
            output_length = 16
            prefix_cache_hit_rate = 0.0
            max_batched_tokens = 8192
            num_mtp_tokens = 0
            num_mtp_token_sizes = []
            mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]
            ttft_limits = 1000
            tpot_limits = None
            disagg = True
            enable_optimize_prefill_decode_ratio = False
            compilation_config = None

        loaded_distribution = LengthDistribution(
            bins=[
                LengthBin(min_tokens=0, max_tokens=500, weight=0.6),
                LengthBin(min_tokens=500, max_tokens=1500, weight=0.4),
            ]
        )
        mock_tasks = Mock()
        mock_tasks.run_disagg.return_value = []

        with (
            patch.object(throughput_optimizer_module, "arg_parse", return_value=DummyArgs()),
            patch.object(
                throughput_optimizer_module,
                "check_device_targets",
                return_value=DummyArgs.device,
            ),
            patch.object(
                throughput_optimizer_module,
                "load_length_distribution",
                return_value=loaded_distribution,
            ) as mock_load_distribution,
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_tasks) as mock_parallel_runner,
        ):
            self.assertEqual(throughput_optimizer_module.main(), None)
            mock_load_distribution.assert_called_once_with(LENGTH_DISTRIBUTION_PATH)
            self.assertEqual(mock_parallel_runner.call_args.args[0].input_length, LENGTH_DISTRIBUTION_PATH)
            self.assertFalse(hasattr(mock_parallel_runner.call_args.args[0], "length_distribution"))
            mock_tasks.run_disagg.assert_called_once_with()

    def test_main_length_distribution_auto_max_batched_tokens_preserves_runtime_none(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module
        from serving_cast.service.utils import LengthBin, LengthDistribution

        class DummyArgs:
            log_level = "error"
            model_id = "test-model"
            device = ["TEST_DEVICE"]
            num_devices = 1
            input_length = LENGTH_DISTRIBUTION_PATH
            word_embedding_tp = None
            output_length = 16
            prefix_cache_hit_rate = 0.0
            max_batched_tokens = None
            num_mtp_tokens = 0
            num_mtp_token_sizes = []
            mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]
            ttft_limits = 1000
            tpot_limits = None
            disagg = True
            enable_optimize_prefill_decode_ratio = False
            compilation_config = None

        loaded_distribution = LengthDistribution(
            bins=[
                LengthBin(min_tokens=0, max_tokens=500, weight=0.6),
                LengthBin(min_tokens=500, max_tokens=1500, weight=0.4),
            ]
        )
        mock_tasks = Mock()
        mock_tasks.run_disagg.return_value = []

        with (
            patch.object(throughput_optimizer_module, "arg_parse", return_value=DummyArgs()),
            patch.object(
                throughput_optimizer_module,
                "check_device_targets",
                return_value=DummyArgs.device,
            ),
            patch.object(
                throughput_optimizer_module,
                "load_length_distribution",
                return_value=loaded_distribution,
            ),
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_tasks) as mock_parallel_runner,
        ):
            self.assertEqual(throughput_optimizer_module.main(), None)
            self.assertIsNone(mock_parallel_runner.call_args.args[0].max_batched_tokens)
            mock_tasks.run_disagg.assert_called_once_with()

    def test_main_rejects_invalid_length_distribution_input(self):
        from cli.inference import throughput_optimizer as throughput_optimizer_module

        class DummyArgs:
            log_level = "error"
            model_id = "test-model"
            device = ["TEST_DEVICE"]
            num_devices = 1
            input_length = LENGTH_DISTRIBUTION_PATH
            word_embedding_tp = None
            output_length = 16
            prefix_cache_hit_rate = 0.0
            max_batched_tokens = 8192
            num_mtp_tokens = 0
            num_mtp_token_sizes = []
            mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]
            ttft_limits = 1000
            tpot_limits = None
            disagg = True
            enable_optimize_prefill_decode_ratio = False
            compilation_config = None

        mock_tasks = Mock()
        mock_tasks.run_disagg.return_value = []

        with (
            patch.object(throughput_optimizer_module, "arg_parse", return_value=DummyArgs()),
            patch.object(
                throughput_optimizer_module,
                "check_device_targets",
                return_value=DummyArgs.device,
            ),
            patch.object(
                throughput_optimizer_module,
                "load_length_distribution",
                side_effect=ValueError("bad distribution"),
            ),
            patch("serving_cast.parallel_runner.ParallelRunner", return_value=mock_tasks),
        ):
            self.assertEqual(throughput_optimizer_module.main(), 1)
            mock_tasks.run_disagg.assert_not_called()


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
            vendored_model_config_path("Qwen/Qwen3-VL-235B-A22B-Instruct"),
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


class TestThroughputOptimizerDraftSpecCli(TestCase):
    """RFC G2/G3: Dflash / DSpark CLI wiring for throughput_optimizer."""

    def _parse(self, extra: list[str]):
        from cli.inference import throughput_optimizer as mod

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return mod.arg_parse()

    def test_defaults_keep_draft_disabled(self):
        args = self._parse([])
        self.assertIsNone(args.speculative_method)
        self.assertIsNone(args.num_speculative_tokens)
        self.assertEqual(args.num_mtp_tokens, 0)

    def test_dflash_resolves_block_and_clamps_acceptance(self):
        args = self._parse(
            [
                "--speculative-method=dflash",
                "--num-speculative-tokens=15",
                "--acceptance-length=99",
                "--num-draft-layers=2",
            ]
        )
        self.assertEqual(args.speculative_method, "dflash")
        self.assertEqual(args.num_speculative_tokens, 15)
        self.assertEqual(args.draft_block_size, 16)
        self.assertEqual(args.acceptance_length, 15.0)  # clamp to B-1
        self.assertEqual(args.num_draft_layers, 2)
        self.assertEqual(args.num_mtp_tokens, 0)
        self.assertEqual(args.num_mtp_token_sizes, [15])  # N candidates in search slot

    def test_dspark_resolves_block_and_clamps_acceptance_to_n(self):
        args = self._parse(
            [
                "--speculative-method=dspark",
                "--num-speculative-tokens=7",
                "--acceptance-length=99",
                "--dspark-markov-rank=128",
                "--dspark-markov-head=gated",
            ]
        )
        self.assertEqual(args.speculative_method, "dspark")
        self.assertEqual(args.num_speculative_tokens, 7)
        self.assertEqual(args.draft_block_size, 8)
        self.assertEqual(args.acceptance_length, 7.0)  # clamp to n (= B-1)
        self.assertEqual(args.dspark_markov_rank, 128)
        self.assertEqual(args.dspark_markov_head, "gated")
        self.assertEqual(args.num_mtp_tokens, 0)

    def test_g3_dependent_without_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--num-speculative-tokens=15"])

    def test_g3_acceptance_without_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--acceptance-length=3"])

    def test_g3_shared_draft_without_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--num-draft-layers=4"])

    def test_g3_markov_requires_dspark_method(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--dspark-markov-rank=128"])

    def test_g2_dflash_and_mtp_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-mtp-tokens", "2"])

    def test_g2_dspark_and_mtp_candidate_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dspark", "--num-mtp-tokens", "0", "2"])

    def test_dspark_cannot_mix_legacy_mtp_zero(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dspark", "--num-speculative-tokens=7", "--num-mtp-tokens", "0"])

    def test_builtin_num_speculative_tokens_maps_to_block_eight(self):
        args = self._parse(["--speculative-method=dflash"])
        self.assertEqual(args.num_speculative_tokens, 7)
        self.assertEqual(args.draft_block_size, 8)

    def test_mtp_new_entry_single_n(self):
        args = self._parse(["--speculative-method=mtp", "--num-speculative-tokens=2", "--acceptance-length=1.5"])
        self.assertEqual(args.speculative_method, "mtp")
        self.assertEqual(args.num_speculative_tokens, 2)
        self.assertEqual(args.num_speculative_token_sizes, [2])
        self.assertEqual(args.num_mtp_token_sizes, [2])

    def test_mtp_new_entry_multi_n_search(self):
        args = self._parse(
            ["--speculative-method=mtp", "--num-speculative-tokens", "1", "2", "--acceptance-length=1.5"]
        )
        self.assertEqual(args.speculative_method, "mtp")
        self.assertEqual(args.num_speculative_token_sizes, [1, 2])
        self.assertEqual(args.num_mtp_token_sizes, [1, 2])

    def test_dflash_multi_n_search(self):
        args = self._parse(
            ["--speculative-method=dflash", "--num-speculative-tokens", "3", "7", "--acceptance-length=5"]
        )
        self.assertEqual(args.speculative_method, "dflash")
        self.assertEqual(args.num_speculative_token_sizes, [3, 7])
        self.assertEqual(args.num_mtp_token_sizes, [3, 7])

    def test_zero_candidates_with_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-speculative-tokens", "0"])
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-speculative-tokens", "0", "0"])
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-speculative-tokens", "0", "3", "7"])
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp", "--num-speculative-tokens", "0", "1", "2"])

    def test_mtp_with_draft_layers_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp", "--num-speculative-tokens=2", "--num-draft-layers=4"])

    def test_mtp_and_dflash_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-mtp-tokens", "2"])

    def test_mtp_method_requires_num_speculative_tokens(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp"])

    def test_mtp_cannot_mix_legacy_num_mtp_tokens(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp", "--num-speculative-tokens=2", "--num-mtp-tokens", "2"])

    def test_mtp_cannot_mix_legacy_acceptance_rate(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--speculative-method=mtp",
                    "--num-speculative-tokens=2",
                    "--mtp-acceptance-rate",
                    "0.9",
                    "0.6",
                ]
            )

    def test_legacy_mtp_cannot_mix_acceptance_length(self):
        with self.assertRaises(SystemExit):
            self._parse(["--num-mtp-tokens", "2", "--acceptance-length=1.5"])


class TestPPCandidateGeneration:
    """Unit tests for stage-local PP candidate generation (Task 3)."""

    @staticmethod
    def _candidates(**overrides):
        kwargs = {
            "num_devices": 32,
            "tp_sizes": [1],
            "pp_sizes": [2],
            "num_hidden_layers": 80,
            "ep_sizes": [1],
            "moe_dp_sizes": [1],
        }
        kwargs.update(overrides)
        return build_pp_search_candidates(**kwargs)

    def test_pp_search_defaults_to_one(self):
        from cli.inference import throughput_optimizer as module

        argv = [
            "throughput_optimizer",
            "--input-length=1",
            "--output-length=1",
            "Qwen/Qwen3-32B",
        ]
        with patch.object(sys, "argv", argv):
            args = module.arg_parse()
        assert args.pp_sizes is None

    def test_pp_candidates_derive_data_parallel_size_from_tp_and_pp(self):
        configs = self._candidates(
            tp_sizes=[8],
            pp_sizes=[1, 2, 4],
        )
        assert [c.dp_size for c in configs] == [4, 2, 1]

    @pytest.mark.parametrize(
        "overrides",
        (
            {"num_devices": 8, "tp_sizes": [4], "pp_sizes": [3]},
            {
                "num_devices": 8,
                "tp_sizes": [1],
                "pp_sizes": [100],
                "num_hidden_layers": 4,
            },
            {
                "num_devices": 8,
                "tp_sizes": None,
                "pp_sizes": [16],
                "ep_sizes": None,
            },
        ),
    )
    def test_invalid_pp_candidates_are_filtered(self, overrides):
        if max(overrides["pp_sizes"]) > overrides.get("num_devices", 32):
            with pytest.raises(ValueError, match="larger than 'num_devices'"):
                self._candidates(**overrides)
        else:
            assert self._candidates(**overrides) == []

    def test_moe_tp_uses_stage_local_devices(self):
        # Stage-local arithmetic: moe_tp = (num_devices / pp) / (ep * moe_dp).
        # With EP=1 (expert parallelism off) moe_tp > 1 stays meaningful, so
        # moe_dp=2 keeps exercising the stage-local division (issue #456).
        configs = self._candidates(
            num_devices=16,
            tp_sizes=[2],
            pp_sizes=[2],
            ep_sizes=[1],
            moe_dp_sizes=[2],
        )
        assert len(configs) == 1
        assert configs[0].moe_tp_size == 4
        assert configs[0].dp_size == 4

    def test_ep_domain_broken_candidates_filtered(self):
        """Issue #456: EP > 1 requires EP * MOE-DP == TP * DP.

        num_devices=16, pp=2 -> stage=8, tp=2 -> dp=4, so the vLLM-valid EP is
        8. EP=2 previously produced an inconsistent moe_tp=4 candidate
        (EP groups holding only a fraction of the token world) and is now
        filtered at candidate generation instead of failing in torch.compile.
        """
        configs = self._candidates(
            num_devices=16,
            tp_sizes=[2],
            pp_sizes=[2],
            ep_sizes=[2],
            enforce_ep_domain=True,
        )
        assert configs == []

    @pytest.mark.parametrize(
        ("pp_size", "partition"),
        ((4, "20,20,20"), (2, "10,10")),
    )
    def test_invalid_partition_raises(self, pp_size, partition):
        with pytest.raises(ValueError, match="partition"):
            self._candidates(
                pp_sizes=[pp_size],
                pp_layer_partitions=[partition],
            )

    def test_valid_partition_produces_candidate(self):
        configs = self._candidates(
            pp_sizes=[2],
            pp_layer_partitions=["40,40"],
        )
        assert len(configs) == 1
        assert configs[0].layer_partition == (40, 40)

    def test_pp1_preserves_no_partition_field(self):
        configs = build_pp_search_candidates(num_devices=8, tp_sizes=[2], pp_sizes=[1], num_hidden_layers=80)
        assert len(configs) == 1
        assert configs[0].pp_size == 1
        assert configs[0].layer_partition is None

    def test_ep_default_matches_legacy_num_devices(self):
        """PP=1 without --ep-sizes must default EP to num_devices (legacy), not 1."""
        configs = build_pp_search_candidates(num_devices=8, tp_sizes=[1], pp_sizes=None, num_hidden_layers=80)
        ep_sizes_used = {c.ep_size for c in configs}
        assert 8 in ep_sizes_used

    def test_multiple_pp_sizes_with_matching_partitions(self):
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[1],
            pp_sizes=[2, 4],
            pp_layer_partitions=["40,40", "20,20,20,20"],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
        )
        pp2 = [c for c in configs if c.pp_size == 2]
        pp4 = [c for c in configs if c.pp_size == 4]
        assert len(pp2) == 1
        assert pp2[0].layer_partition == (40, 40)
        assert len(pp4) == 1
        assert pp4[0].layer_partition == (20, 20, 20, 20)

    def test_pp1_and_pp2_with_pp2_partition(self):
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[1],
            pp_sizes=[1, 2],
            pp_layer_partitions=["40,40"],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
        )
        pp1 = [c for c in configs if c.pp_size == 1]
        pp2 = [c for c in configs if c.pp_size == 2]
        assert len(pp1) == 1
        assert pp1[0].layer_partition is None
        assert len(pp2) == 1
        assert pp2[0].layer_partition == (40, 40)

    def test_pp_gt1_with_mtp_allowed(self):
        """PP>1 with non-zero MTP produces valid candidates (gate removed)."""
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[1],
            pp_sizes=[2],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
            num_mtp_tokens=4,
        )
        assert len(configs) == 1
        assert configs[0].pp_size == 2
        assert configs[0].num_mtp_tokens == 4

    def test_mixed_pp1_pp2_with_mtp_keeps_both(self):
        """Both PP=1 and PP=2 produce candidates with non-zero MTP."""
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[1],
            pp_sizes=[1, 2],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
            num_mtp_tokens=4,
        )
        assert len(configs) > 0
        pp1 = [c for c in configs if c.pp_size == 1]
        pp2 = [c for c in configs if c.pp_size == 2]
        assert len(pp1) == 1
        assert pp1[0].num_mtp_tokens == 4
        assert len(pp2) == 1
        assert pp2[0].num_mtp_tokens == 4

    def test_pp2_with_mtp_zero_and_nonzero_keeps_both(self):
        """PP=2 retains both MTP=0 and MTP=4 candidates."""
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[1],
            pp_sizes=[2],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
            num_mtp_token_sizes=[0, 4],
        )
        assert len(configs) > 0
        mtp_values = sorted(c.num_mtp_tokens for c in configs)
        assert mtp_values == [0, 4]

    def test_mtp_candidate_survives_when_other_pp_fails_divisibility(self):
        """PP=4 has valid base divisibility and produces an MTP candidate;
        PP=3 fails divisibility and produces none.
        """
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[4],
            pp_sizes=[3, 4],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
            num_mtp_tokens=4,
        )
        assert len(configs) == 1
        assert configs[0].pp_size == 4
        assert configs[0].num_mtp_tokens == 4

    def test_mtp_candidate_survives_when_another_pp_exceeds_layer_count(self):
        """PP=2 stays within num_hidden_layers and produces an MTP candidate;
        PP=128 exceeds num_hidden_layers and produces none.
        """
        configs = build_pp_search_candidates(
            num_devices=256,
            tp_sizes=[1],
            pp_sizes=[2, 128],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
            num_mtp_tokens=4,
        )
        assert len(configs) == 1
        assert configs[0].pp_size == 2
        assert configs[0].num_mtp_tokens == 4

    def test_pp_gt1_with_mtp_zero_allowed(self):
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=[1],
            pp_sizes=[2],
            num_hidden_layers=80,
            ep_sizes=[1],
            moe_dp_sizes=[1],
            num_mtp_tokens=0,
        )
        assert len(configs) == 1
        assert configs[0].num_mtp_tokens == 0

    def test_pp_only_search_uses_stage_local_tp_ep_defaults(self):
        configs = build_pp_search_candidates(
            num_devices=32,
            tp_sizes=None,
            pp_sizes=[2],
            ep_sizes=None,
            num_hidden_layers=80,
            moe_dp_sizes=[1],
        )
        assert len(configs) >= 1
        assert all(c.pp_size == 2 for c in configs)
        assert all(c.dp_size == 1 for c in configs)

    def test_ep_domain_mismatch_candidates_filtered(self):
        """Issue #456: TP=2 x DP=4 x EP=4 (EP != TP x DP) must not be generated.

        num_devices=32, pp=4 -> stage=8 devices, tp=2 -> dp=4. The vLLM-valid
        EP is tp * dp = 8; EP=4 previously passed stage-local divisibility
        (8 % 4 == 0) and crashed torch.compile with a residual broadcast error
        under SP + dispatch_ffn_combine.
        """
        configs = self._candidates(
            num_devices=32,
            tp_sizes=[2],
            pp_sizes=[4],
            ep_sizes=[4, 8, 16, 32],
            moe_dp_sizes=[1],
            enforce_ep_domain=True,
        )
        ep_sizes_used = {c.ep_size for c in configs}
        assert ep_sizes_used == {8}
        assert all(c.moe_tp_size == 1 for c in configs)
        assert all(c.dp_size == 4 for c in configs)

    def test_ep_domain_filter_off_by_default(self):
        """Without SP / dispatch_ffn_combine, moe_tp > 1 with EP stays searchable.

        cli/registry/validators.py accepts moe_tp x moe_dp x ep == num_devices
        regardless of TP x DP, so the EP token-domain filter must be opt-in
        (review note on PR #894 by ChenHuiwen).
        """
        configs = self._candidates(
            num_devices=32,
            tp_sizes=[2],
            pp_sizes=[4],
            ep_sizes=[4, 8],
            moe_dp_sizes=[1],
        )
        ep_sizes_used = {c.ep_size for c in configs}
        assert ep_sizes_used == {4, 8}
        target = next(c for c in configs if c.ep_size == 4)
        assert target.moe_tp_size == 2

    def test_ep_domain_conservation_keeps_moe_dp_replication(self):
        """EP x MOE-DP == TP x DP with MOE-DP > 1 stays a valid candidate."""
        configs = self._candidates(
            num_devices=32,
            tp_sizes=[2],
            pp_sizes=[4],
            ep_sizes=[4],
            moe_dp_sizes=[2],
            enforce_ep_domain=True,
        )
        assert len(configs) == 1
        assert configs[0].ep_size == 4
        assert configs[0].moe_dp_size == 2
        assert configs[0].moe_tp_size == 1
        assert configs[0].dp_size == 4

    def test_ep_domain_check_skipped_without_ep(self):
        """EP=1 candidates are unaffected by the EP-domain constraint."""
        configs = self._candidates(
            num_devices=32,
            tp_sizes=[2],
            pp_sizes=[4],
            ep_sizes=[1],
            moe_dp_sizes=[1],
            enforce_ep_domain=True,
        )
        assert len(configs) == 1
        assert configs[0].ep_size == 1
        assert configs[0].moe_tp_size == 8

    def test_ep_search_all_powers_of_two_yields_only_valid_domain(self):
        """`--ep-sizes` with no values searches powers of two; only EP = TP x DP survives above EP=1."""
        configs = self._candidates(
            num_devices=32,
            tp_sizes=[2],
            pp_sizes=[4],
            ep_sizes=[],
            moe_dp_sizes=[1],
            enforce_ep_domain=True,
        )
        ep_sizes_used = {c.ep_size for c in configs}
        # EP=1 (expert parallelism off) is unconstrained; EP>1 must equal TP x DP.
        assert ep_sizes_used == {1, 8}

    def test_is_valid_ep_domain_unit(self):
        from serving_cast.service.utils import is_valid_ep_domain

        # Issue #456 repro: TP=2, DP=4, EP=4, MOE-DP=1 is domain-broken.
        assert not is_valid_ep_domain(ep=4, moe_dp=1, tp=2, dp=4)
        # vLLM semantics: EP = TP x DP.
        assert is_valid_ep_domain(ep=8, moe_dp=1, tp=2, dp=4)
        # Expert replication across MOE-DP sub-groups conserves the token domain.
        assert is_valid_ep_domain(ep=4, moe_dp=2, tp=2, dp=4)
        # EP disabled: no constraint.
        assert is_valid_ep_domain(ep=1, moe_dp=2, tp=2, dp=4)
        assert is_valid_ep_domain(ep=1, moe_dp=1, tp=8, dp=1)
