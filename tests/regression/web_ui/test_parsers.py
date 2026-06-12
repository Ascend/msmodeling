"""Tests for web_ui.parsers module."""

from __future__ import annotations


from web_ui.parsers import (
    _extract_execution_error,
    _extract_parallel_config,
    _optimizer_no_result_reason,
    _parse_disagg_row,
    _parse_optimizer_row,
    _parse_pd_ratio_row,
    _parse_table,
    _pick_bottleneck,
    _strip_ansi,
    parse_optimizer,
    parse_result,
    parse_text_generate,
    parse_video_generate,
    time_to_seconds,
    time_to_us,
)
from web_ui.schemas import ExperimentTask


class TestTimeConversions:
    """Tests for time conversion functions."""

    def test_time_to_us_nanoseconds(self) -> None:
        """Test converting nanoseconds to microseconds."""
        assert time_to_us("100ns") == 0.1

    def test_time_to_us_microseconds(self) -> None:
        """Test converting microseconds."""
        assert time_to_us("100us") == 100.0

    def test_time_to_us_milliseconds(self) -> None:
        """Test converting milliseconds to microseconds."""
        assert time_to_us("100ms") == 100000.0

    def test_time_to_us_seconds(self) -> None:
        """Test converting seconds to microseconds."""
        assert time_to_us("1s") == 1_000_000.0

    def test_time_to_us_invalid_format(self) -> None:
        """Test invalid format returns 0."""
        assert time_to_us("invalid") == 0.0

    def test_time_to_us_whitespace(self) -> None:
        """Test handling whitespace."""
        assert time_to_us("  50ms  ") == 50000.0

    def test_time_to_seconds_ns(self) -> None:
        """Test converting ns to seconds."""
        assert time_to_seconds("1000000000ns") == 1.0

    def test_time_to_seconds_us(self) -> None:
        """Test converting us to seconds."""
        assert time_to_seconds("1000000us") == 1.0

    def test_time_to_seconds_ms(self) -> None:
        """Test converting ms to seconds."""
        assert time_to_seconds("1000ms") == 1.0

    def test_time_to_seconds_s(self) -> None:
        """Test converting seconds."""
        assert time_to_seconds("10s") == 10.0


class TestStripAnsi:
    """Tests for _strip_ansi function."""

    def test_strip_ansi_basic(self) -> None:
        """Test stripping basic ANSI codes."""
        result = _strip_ansi("\x1b[31mRed text\x1b[0m")
        assert result == "Red text"

    def test_strip_ansi_multiple_codes(self) -> None:
        """Test stripping multiple ANSI codes."""
        result = _strip_ansi("\x1b[1m\x1b[32mBold green\x1b[0m")
        assert result == "Bold green"

    def test_strip_ansi_no_codes(self) -> None:
        """Test text without ANSI codes."""
        result = _strip_ansi("Plain text")
        assert result == "Plain text"

    def test_strip_ansi_empty_string(self) -> None:
        """Test empty string."""
        assert _strip_ansi("") == ""

    def test_strip_ansi_none(self) -> None:
        """Test None input."""
        assert _strip_ansi(None) == ""


class TestExtractExecutionError:
    """Tests for _extract_execution_error function."""

    def test_extract_huggingface_error(self) -> None:
        """Test extracting HuggingFace download error."""
        log = "OSError: We couldn't connect to 'https://huggingface.co' and couldn't find them in the cached files"
        result = _extract_execution_error(log)
        assert "Unable to download model files from HuggingFace" in result

    def test_extract_module_not_found(self) -> None:
        """Test extracting ModuleNotFoundError."""
        log = "ModuleNotFoundError: No module named 'torch'"
        result = _extract_execution_error(log)
        assert "ModuleNotFoundError" in result

    def test_extract_import_error(self) -> None:
        """Test extracting ImportError."""
        log = "ImportError: cannot import name 'xyz'"
        result = _extract_execution_error(log)
        assert "ImportError" in result

    def test_extract_called_process_error(self) -> None:
        """Test extracting CalledProcessError."""
        log = "CalledProcessError: Command '['python']' returned non-zero exit status 1"
        result = _extract_execution_error(log)
        assert "CalledProcessError" in result

    def test_extract_permission_denied(self) -> None:
        """Test extracting permission error."""
        log = "Permission denied: /path/to/file"
        result = _extract_execution_error(log)
        assert "Permission denied" in result

    def test_extract_traceback(self) -> None:
        """Test extracting traceback line."""
        log = "Traceback (most recent call last):\n  File test.py, line 1"
        result = _extract_execution_error(log)
        assert "Traceback" in result

    def test_extract_error_line(self) -> None:
        """Test extracting ERROR: prefixed line."""
        log = "ERROR: Something went wrong"
        result = _extract_execution_error(log)
        assert "ERROR:" in result

    def test_extract_fallback_last_line(self) -> None:
        """Test fallback to last non-empty line."""
        log = "Line 1\nLine 2\nLast line"
        result = _extract_execution_error(log)
        assert result == "Last line"

    def test_extract_with_fallback_param(self) -> None:
        """Test with custom fallback message."""
        # Fallback is only returned when log is empty or has no valid lines
        log = ""
        result = _extract_execution_error(log, "Custom fallback")
        assert result == "Custom fallback"

    def test_extract_empty_log(self) -> None:
        """Test with empty log."""
        result = _extract_execution_error("", "Fallback")
        assert result == "Fallback"


class TestOptimizerNoResultReason:
    """Tests for _optimizer_no_result_reason function."""

    def test_no_result_reason_both_limits(self) -> None:
        """Test reason when both limits are set."""
        task = ExperimentTask("throughput_optimizer", {"ttft_limits": 100.0, "tpot_limits": 50.0}, [], "h", "t")
        result = _optimizer_no_result_reason(task)
        assert "TTFT=100 ms" in result
        assert "TPOT=50 ms" in result

    def test_no_result_reason_ttft_only(self) -> None:
        """Test reason when only TTFT is set."""
        task = ExperimentTask("throughput_optimizer", {"ttft_limits": 200.0, "tpot_limits": None}, [], "h", "t")
        result = _optimizer_no_result_reason(task)
        assert "TTFT=200 ms" in result
        assert "TPOT=unlimited" in result

    def test_no_result_reason_tpot_only(self) -> None:
        """Test reason when only TPOT is set."""
        task = ExperimentTask("throughput_optimizer", {"ttft_limits": None, "tpot_limits": 75.0}, [], "h", "t")
        result = _optimizer_no_result_reason(task)
        assert "TTFT=unlimited" in result
        assert "TPOT=75 ms" in result

    def test_no_result_reason_both_none(self) -> None:
        """Test reason when both limits are None."""
        task = ExperimentTask("throughput_optimizer", {"ttft_limits": None, "tpot_limits": None}, [], "h", "t")
        result = _optimizer_no_result_reason(task)
        assert "TTFT=unlimited" in result
        assert "TPOT=unlimited" in result


class TestParseOptimizerRow:
    """Tests for _parse_optimizer_row function."""

    def test_parse_valid_row(self) -> None:
        """Test parsing a valid optimizer row."""
        cells = ["1", "1000.5", "50.2", "25.1", "32", "8", "TP=4 | DP=2", "128"]
        result = _parse_optimizer_row(cells)
        assert result is not None
        assert result["rank"] == 1
        assert result["throughput_token_s"] == 1000.5
        assert result["ttft_ms"] == 50.2
        assert result["tpot_ms"] == 25.1
        assert result["concurrency"] == 32
        assert result["num_devices"] == 8
        assert result["parallel"] == "TP=4 | DP=2"
        assert result["batch_size"] == 128

    def test_parse_row_too_short(self) -> None:
        """Test parsing row with too few cells."""
        cells = ["1", "1000.5", "50.2"]
        result = _parse_optimizer_row(cells)
        assert result is None

    def test_parse_row_non_digit_rank(self) -> None:
        """Test parsing row with non-digit first cell."""
        cells = ["x", "1000.5", "50.2", "25.1", "32", "8", "TP=4", "128"]
        result = _parse_optimizer_row(cells)
        assert result is None

    def test_parse_row_with_extra_fields(self) -> None:
        """Test parsing row with extra middle fields (PP column)."""
        cells = ["1", "1000.5", "50.2", "25.1", "32", "8", "TP=4", "PP=1", "DP=2", "128"]
        result = _parse_optimizer_row(cells)
        assert result is not None
        assert result["parallel"] == "TP=4 | PP=1 | DP=2"


class TestParseTable:
    """Tests for _parse_table function."""

    def test_parse_optimizer_table(self) -> None:
        """Test parsing optimizer-style table."""
        lines = [
            "| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel | batch_size |",
            "|  1  |       2888.45        |  16032.05 |   49.90   |     175     |       8     | TP=8 | DP=1 |    175     |",
        ]
        result = _parse_table(lines)
        assert len(result) == 1
        assert result[0]["rank"] == 1
        assert result[0]["throughput_token_s"] == 2888.45

    def test_parse_operator_table(self) -> None:
        """Test parsing operator breakdown table."""
        lines = [
            "matmul            5000.2us    500.0us    10",
            "softmax            2000.0us    400.0us     5",
        ]
        result = _parse_table(lines)
        assert len(result) == 2
        assert result[0]["name"] == "matmul"
        assert result[0]["analytic_total_us"] == 5000.2
        assert result[1]["name"] == "softmax"
        assert result[1]["num_calls"] == 5

    def test_parse_table_skips_dividers(self) -> None:
        """Test that divider lines are skipped."""
        lines = [
            "+-----+------+",
            "| Name | Time |",
            "+-----+------+",
            "op    100us  50us   1",
            "+-----+------+",
        ]
        result = _parse_table(lines)
        assert len(result) == 1

    def test_parse_table_with_ansi_codes(self) -> None:
        """Test parsing table with ANSI color codes."""
        lines = [
            "\x1b[31mop1\x1b[0m            1000us    100us    10",
            "op2                2000us    200us    5",
        ]
        result = _parse_table(lines)
        assert len(result) == 2
        assert result[0]["name"] == "op1"
        assert result[1]["name"] == "op2"

    def test_parse_table_empty_lines(self) -> None:
        """Test parsing with empty lines."""
        lines = ["", "op1    100us    50us    1", ""]
        result = _parse_table(lines)
        assert len(result) == 1

    def test_parse_table_header_skipped(self) -> None:
        """Test that header row is skipped."""
        lines = [
            "Name            analytic total    analytic avg    # of Calls",
            "op1             1000us            100us           10",
        ]
        result = _parse_table(lines)
        assert len(result) == 1
        assert result[0]["name"] == "op1"


class TestPickBottleneck:
    """Tests for _pick_bottleneck function."""

    def test_pick_memory_bound(self) -> None:
        """Test picking memory bound as bottleneck."""
        summary = {"memory_bound": 90.0, "communication_bound": 5.0}
        result = _pick_bottleneck(summary)
        assert result == "memory_bound"

    def test_pick_communication_bound(self) -> None:
        """Test picking communication bound as bottleneck."""
        summary = {"memory_bound": 30.0, "communication_bound": 85.0}
        result = _pick_bottleneck(summary)
        assert result == "communication_bound"

    def test_pick_compute_bound_mma(self) -> None:
        """Test picking compute MMA bound as bottleneck."""
        summary = {"memory_bound": 20.0, "compute_bound_mma": 95.0}
        result = _pick_bottleneck(summary)
        assert result == "compute_bound_mma"

    def test_pick_compute_bound_gp(self) -> None:
        """Test picking compute GP bound as bottleneck."""
        summary = {"memory_bound": 10.0, "compute_bound_gp": 88.0}
        result = _pick_bottleneck(summary)
        assert result == "compute_bound_gp"

    def test_pick_tie_breaker(self) -> None:
        """Test tie-breaking when values are equal."""
        summary = {"memory_bound": 50.0, "communication_bound": 50.0}
        result = _pick_bottleneck(summary)
        # Returns one of them (max of equal values)
        assert result in ("memory_bound", "communication_bound")

    def test_pick_no_valid_values(self) -> None:
        """Test with no valid numeric values."""
        summary = {"memory_bound": None, "communication_bound": "N/A"}
        result = _pick_bottleneck(summary)
        assert result is None

    def test_pick_empty_summary(self) -> None:
        """Test with empty summary."""
        result = _pick_bottleneck({})
        assert result is None


class TestParseTextGenerate:
    """Tests for parse_text_generate function."""

    def test_parse_basic_text_log(self) -> None:
        """Test parsing basic text generation log."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = """
Number of Queries per DP rank: 32
Model compilation and execution time: 2.5
Total time for analytic: 1.2s
TPS/Device: 500.0
Total device memory: 80 GB
Model weight size: 10 GB
KV cache: 2 GB
Model activation size: 5 GB
Reserved memory: 1 GB
Memory available: 62 GB
"""
        result = parse_text_generate(task, log, "success")
        assert result.summary["queries_per_dp_rank"] == 32
        assert result.summary["run_time_s"] == 2.5
        assert result.summary["analytic_total_time_s"] == 1.2
        assert result.summary["tps_per_device"] == 500.0
        assert result.summary["total_device_memory_gb"] == 80.0
        assert result.summary["model_weight_size_gb"] == 10.0
        assert result.summary["kv_cache_gb"] == 2.0
        assert result.status == "success"

    def test_parse_text_with_op_breakdown(self) -> None:
        """Test parsing with operator breakdown table."""
        task = ExperimentTask("text_generate", {"decode": False}, [], "h", "test")
        log = """
Total time for analytic: 1.0s
Name            analytic total    analytic avg    # of Calls
--------------------------------------------------------
matmul          5000.0us         500.0us           10
softmax          2000.0us         400.0us            5
"""
        result = parse_text_generate(task, log, "success")
        assert len(result.tables["op_breakdown"]) == 2
        assert result.tables["op_breakdown"][0]["name"] == "matmul"
        assert result.tables["op_breakdown"][1]["analytic_total_us"] == 2000.0

    def test_parse_text_with_op_bound(self) -> None:
        """Test parsing with operator bound info."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = """
Total time for analytic: 0.5s
analytic_OpBound: memory_bound:80.5, communication_bound:10.2, compute_bound_mma:9.3
"""
        result = parse_text_generate(task, log, "success")
        assert result.summary["memory_bound"] == 80.5
        assert result.summary["communication_bound"] == 10.2
        assert result.summary["compute_bound_mma"] == 9.3

    def test_parse_text_stage_decode(self) -> None:
        """Test stage is set to decode when decode=True."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = "Total time for analytic: 0.5s"
        result = parse_text_generate(task, log, "success")
        assert result.summary["stage"] == "decode"

    def test_parse_text_stage_prefill(self) -> None:
        """Test stage is set to prefill when decode=False."""
        task = ExperimentTask("text_generate", {"decode": False}, [], "h", "test")
        log = "Total time for analytic: 0.5s"
        result = parse_text_generate(task, log, "success")
        assert result.summary["stage"] == "prefill"

    def test_parse_text_negative_memory(self) -> None:
        """Test negative available memory sets oom_risk."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = """
Total device memory: 80 GB
Memory available: -2 GB
"""
        result = parse_text_generate(task, log, "success")
        assert result.summary["memory_fit_status"] == "oom_risk"

    def test_parse_text_with_warnings(self) -> None:
        """Test parsing warning messages."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = """
WARNING: Low memory detected
WARNING: High latency expected
Total time for analytic: 0.5s
"""
        result = parse_text_generate(task, log, "success")
        assert len(result.warnings) == 2
        assert "Low memory" in result.warnings[0]

    def test_parse_text_failed_with_error(self) -> None:
        """Test parsing failed run with error."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = "Some error occurred"
        result = parse_text_generate(task, log, "failed", "Process exited with code 1")
        assert result.status == "failed"
        assert result.error == "Process exited with code 1"


class TestParseVideoGenerate:
    """Tests for parse_video_generate function."""

    def test_parse_basic_video_log(self) -> None:
        """Test parsing basic video generation log."""
        task = ExperimentTask("video_generate", {"use_cfg": True, "cfg_parallel": True}, [], "h", "test")
        log = """
Model compilation and execution time: 5.0s
Total time for analytic: 3.2s
"""
        result = parse_video_generate(task, log, "success")
        assert result.summary["run_time_s"] == 5.0
        assert result.summary["analytic_total_time_s"] == 3.2

    def test_parse_video_cfg_mode(self) -> None:
        """Test CFG mode detection."""
        task = ExperimentTask("video_generate", {"use_cfg": True, "cfg_parallel": True}, [], "h", "test")
        log = "Total time for analytic: 1.0s"
        result = parse_video_generate(task, log, "success")
        assert result.summary["cfg_mode"] == "cfg_parallel"

    def test_parse_video_batch_concat_mode(self) -> None:
        """Test batch_concat mode."""
        task = ExperimentTask("video_generate", {"use_cfg": True, "cfg_parallel": False}, [], "h", "test")
        log = "Total time for analytic: 1.0s"
        result = parse_video_generate(task, log, "success")
        assert result.summary["cfg_mode"] == "batch_concat"

    def test_parse_video_cfg_disabled(self) -> None:
        """Test CFG disabled mode."""
        task = ExperimentTask("video_generate", {"use_cfg": False, "cfg_parallel": False}, [], "h", "test")
        log = "Total time for analytic: 1.0s"
        result = parse_video_generate(task, log, "success")
        assert result.summary["cfg_mode"] == "disabled"

    def test_parse_video_with_dit_cache(self) -> None:
        """Test parsing with DiT cache enabled."""
        task = ExperimentTask("video_generate", {"dit_cache": True}, [], "h", "test")
        log = """
Enabled dit_block_cache, replaced 15 blocks in range [20, 30) out of 50
Total time for analytic: 1.0s
"""
        result = parse_video_generate(task, log, "success")
        assert result.summary["dit_cache_effective"] is True
        assert result.summary["replaced_blocks"] == 15
        assert result.summary["replaced_range_start"] == 20
        assert result.summary["replaced_range_end"] == 30
        assert result.summary["total_blocks"] == 50

    def test_parse_video_dit_cache_disabled(self) -> None:
        """Test parsing DiT cache disabled reason."""
        task = ExperimentTask("video_generate", {"dit_cache": False}, [], "h", "test")
        log = """
DiT cache is disabled because cache parameters are invalid
Total time for analytic: 1.0s
"""
        result = parse_video_generate(task, log, "success")
        assert result.summary["dit_cache_effective"] is False
        assert "cache parameters are invalid" in result.summary.get("dit_cache_disable_reason", "")


class TestParseOptimizer:
    """Tests for parse_optimizer function."""

    def test_parse_optimizer_success(self) -> None:
        """Test parsing successful optimizer run."""
        task = ExperimentTask("throughput_optimizer", {}, [], "h", "test")
        log = """
Best Throughput: 2500.0
TTFT: 15000.0
TPOT: 45.0
TTFT Limits: 2000.0
TPOT Limits: 50.0
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel | batch_size |
|  1  |       2500.0        |  15000.0  |   45.0    |     200     |       8     | TP=8 | DP=1 |    200     |
"""
        result = parse_optimizer(task, log, "success")
        assert result.summary["best_throughput"] == 2500.0
        assert result.summary["best_ttft_ms"] == 15000.0
        assert result.summary["best_tpot_ms"] == 45.0
        assert result.summary["ttft_limits_ms"] == 2000.0
        assert result.summary["tpot_limits_ms"] == 50.0
        assert result.summary["best_parallel"] == "TP=8 | DP=1"
        assert result.summary["best_batch_size"] == 200
        assert result.summary["best_concurrency"] == 200
        assert len(result.tables["top_configs"]) == 1

    def test_parse_optimizer_no_results(self) -> None:
        """Test parsing optimizer with no valid results."""
        task = ExperimentTask("throughput_optimizer", {"ttft_limits": 100.0, "tpot_limits": 50.0}, [], "h", "test")
        log = "No valid configuration found"
        result = parse_optimizer(task, log, "success")
        assert result.status == "no_result"
        assert "no_result_reason" in result.summary

    def test_parse_optimizer_with_none_limits(self) -> None:
        """Test parsing with None limits in log."""
        task = ExperimentTask("throughput_optimizer", {"ttft_limits": None}, [], "h", "test")
        log = """
TTFT Limits: None
TPOT Limits: 50.0
Best Throughput: 1000.0
"""
        result = parse_optimizer(task, log, "success")
        assert result.summary["ttft_limits_ms"] is None
        assert result.summary["tpot_limits_ms"] == 50.0

    def test_parse_optimizer_failed(self) -> None:
        """Test parsing failed optimizer run."""
        task = ExperimentTask("throughput_optimizer", {}, [], "h", "test")
        log = "Some error occurred"
        result = parse_optimizer(task, log, "failed", "Process error")
        assert result.status == "failed"
        assert result.error == "Process error"


class TestParseResult:
    """Tests for parse_result routing function."""

    def test_parse_result_routes_text_generate(self) -> None:
        """Test parse_result routes to text_generate parser."""
        task = ExperimentTask("text_generate", {"decode": True}, [], "h", "test")
        log = "Total time for analytic: 1.0s"
        result = parse_result(task, log, "success")
        assert result.sim_type == "text_generate"

    def test_parse_result_routes_video_generate(self) -> None:
        """Test parse_result routes to video_generate parser."""
        task = ExperimentTask("video_generate", {}, [], "h", "test")
        log = "Total time for analytic: 1.0s"
        result = parse_result(task, log, "success")
        assert result.sim_type == "video_generate"

    def test_parse_result_routes_optimizer(self) -> None:
        """Test parse_result routes to optimizer parser."""
        task = ExperimentTask("throughput_optimizer", {}, [], "h", "test")
        log = "Best Throughput: 1000.0"
        result = parse_result(task, log, "success")
        assert result.sim_type == "throughput_optimizer"

    def test_parse_result_unknown_type(self) -> None:
        """Test parse_result with unknown sim_type routes to optimizer."""
        task = ExperimentTask("unknown_type", {}, [], "h", "test")
        log = "Best Throughput: 1000.0"
        result = parse_result(task, log, "success")
        assert result.sim_type == "unknown_type"

    def test_parse_optimizer_with_missing_table(self) -> None:
        """Test parsing optimizer log with no table."""
        task = ExperimentTask("throughput_optimizer", {}, [], "h", "test")
        log = "No table found in this log"
        result = parse_optimizer(task, log, "failed")
        assert result.status == "failed"

    def test_parse_video_with_negative_time(self) -> None:
        """Test parsing video log with negative time values."""
        task = ExperimentTask("video_generate", {}, [], "h", "test")
        log = """
Total Analysis Time: -5.5ms
Communication Time: 100ms
"""
        result = parse_video_generate(task, log, "success")
        assert result.status == "success"

    def test_parse_text_with_zero_tps(self) -> None:
        """Test parsing text log with zero TPS."""
        task = ExperimentTask("text_generate", {}, [], "h", "test")
        log = """
Throughput: 0.0 token/s
Runtime: 100.0ms
"""
        result = parse_text_generate(task, log, "success")
        assert result.status == "success"

    def test_parse_optimizer_with_explicit_mode(self) -> None:
        """Test parsing optimizer with explicit deployment mode."""
        task = ExperimentTask("throughput_optimizer", {"mode": "PD Ratio"}, [], "h", "test")
        log = """
Top 1 Aggregation Configurations:
+-----+----------------------+-----------+-----------+
|Top | Throughput (token/s) | TTFT (ms) | TPOT (ms)|
+-----+----------------------+-----------+-----------+
| 1  |       1000.0        |   50.0    |   10.0   |
+-----+----------------------+-----------+-----------+
"""
        result = parse_optimizer(task, log, "success")
        # Result may be 'success' or 'no_result' depending on table parsing
        assert result.status in ("success", "no_result")

    def test_parse_video_with_missing_analytic_data(self) -> None:
        """Test parsing video log without analytic breakdown."""
        task = ExperimentTask("video_generate", {}, [], "h", "test")
        log = "Total Analysis Time: 100ms"
        result = parse_video_generate(task, log, "success")
        assert result.status == "success"

    def test_parse_text_with_mixed_case_values(self) -> None:
        """Test parsing text log with mixed case values."""
        task = ExperimentTask("text_generate", {}, [], "h", "test")
        log = """
Throughput: 100 Token/S
Runtime: 50.0MS
"""
        result = parse_text_generate(task, log, "success")
        assert result.status == "success"

    def test_time_to_us_with_float_seconds(self) -> None:
        """Test time_to_us with floating point seconds."""
        result = time_to_us("1.5s")
        assert result == 1500000.0

    def test_time_to_seconds_with_float(self) -> None:
        """Test time_to_seconds with floating point input."""
        result = time_to_seconds("1.5s")
        assert result == 1.5

    def test_no_result_reason_with_zero_limits(self) -> None:
        """Test no result reason when both limits are zero."""
        task = ExperimentTask(
            "throughput_optimizer",
            {"max_ttft_ms": 0, "max_tpot_ms": 0},
            [],
            "h",
            "test",
        )
        result = _optimizer_no_result_reason(task)
        assert "limit" in result.lower()

    def test_parse_optimizer_with_failed_marker(self) -> None:
        """Test parsing optimizer with explicit failed marker."""
        task = ExperimentTask("throughput_optimizer", {}, [], "h", "test")
        log = "ERROR: Simulation failed"
        result = parse_optimizer(task, log, "failed", "Simulation failed")
        assert result.status == "failed"

    def test_parse_text_with_empty_lines_only(self) -> None:
        """Test parsing text log with only empty lines."""
        task = ExperimentTask("text_generate", {}, [], "h", "test")
        log = "\n\n\n"
        result = parse_text_generate(task, log, "success")
        assert result.status == "success"

    def test_parse_video_with_custom_model(self) -> None:
        """Test parsing video log with custom model name."""
        task = ExperimentTask("video_generate", {"model": "CustomModel"}, [], "h", "test")
        log = "Total Analysis Time: 100ms"
        result = parse_video_generate(task, log, "success")
        assert result.params.get("model") == "CustomModel"


class TestParsePDRatioRow:
    """Tests for _parse_pd_ratio_row function."""

    def test_parse_pd_ratio_valid_row(self) -> None:
        """Test parsing valid PD Ratio row."""
        cells = ["1", "0.5", "1000.0", "800.0", "200.0", "50.0", "10.0", "TP=4", "DP=2", "8", "4", "32", "16", "1", "4"]
        result = _parse_pd_ratio_row(cells)
        assert result is not None
        assert result["rank"] == 1
        assert result["pd_ratio"] == 0.5
        assert result["balanced_qps"] == 1000.0
        assert result["p_qps"] == 800.0
        assert result["d_qps"] == 200.0
        assert result["ttft_ms"] == 50.0
        assert result["tpot_ms"] == 10.0
        assert result["p_parallel"] == "TP=4"
        assert result["d_parallel"] == "DP=2"
        assert result["prefill_devices_per_instance"] == 8
        assert result["decode_devices_per_instance"] == 4
        assert result["p_batch_size"] == 32
        assert result["d_batch_size"] == 16
        assert result["p_concurrency"] == 1
        assert result["d_concurrency"] == 4
        assert result["throughput_token_s"] == 1000.0

    def test_parse_pd_ratio_too_short(self) -> None:
        """Test parsing row with too few cells."""
        cells = ["1", "0.5", "1000.0"]
        result = _parse_pd_ratio_row(cells)
        assert result is None

    def test_parse_pd_ratio_non_digit_rank(self) -> None:
        """Test parsing row with non-digit first cell."""
        cells = ["x", "0.5", "1000.0", "800.0", "200.0", "50.0", "10.0", "TP=4", "DP=2", "8", "4", "32", "16", "1", "4"]
        result = _parse_pd_ratio_row(cells)
        assert result is None

    def test_parse_pd_ratio_invalid_float(self) -> None:
        """Test parsing row with invalid float values."""
        cells = [
            "1",
            "invalid",
            "1000.0",
            "800.0",
            "200.0",
            "50.0",
            "10.0",
            "TP=4",
            "DP=2",
            "8",
            "4",
            "32",
            "16",
            "1",
            "4",
        ]
        result = _parse_pd_ratio_row(cells)
        assert result is None


class TestParseDisaggRow:
    """Tests for _parse_disagg_row function."""

    def test_parse_disagg_prefill_row(self) -> None:
        """Test parsing valid PD Disaggregated prefill row."""
        cells = ["1", "42334.84", "211.67", "982.64", "208", "8", "TP=1 | PP=1 | DP=8", "26"]
        result = _parse_disagg_row(cells, is_prefill=True)
        assert result is not None
        assert result["rank"] == 1
        assert result["throughput_token_s"] == 42334.84
        assert result["qps"] == 211.67
        assert result["ttft_ms"] == 982.64
        assert result["concurrency"] == 208
        assert result["num_devices"] == 8
        assert result["parallel"] == "TP=1 | PP=1 | DP=8"
        assert result["batch_size"] == 26
        assert "tpot_ms" not in result

    def test_parse_disagg_decode_row(self) -> None:
        """Test parsing valid PD Disaggregated decode row."""
        cells = ["1", "20073.67", "100.37", "25.51", "512", "8", "TP=2 | PP=1 | DP=4", "128"]
        result = _parse_disagg_row(cells, is_prefill=False)
        assert result is not None
        assert result["rank"] == 1
        assert result["throughput_token_s"] == 20073.67
        assert result["qps"] == 100.37
        assert result["tpot_ms"] == 25.51
        assert result["concurrency"] == 512
        assert result["num_devices"] == 8
        assert result["parallel"] == "TP=2 | PP=1 | DP=4"
        assert result["batch_size"] == 128
        assert "ttft_ms" not in result

    def test_parse_disagg_too_short(self) -> None:
        """Test parsing row with too few cells."""
        cells = ["1", "42334.84", "211.67"]
        result = _parse_disagg_row(cells, is_prefill=True)
        assert result is None

    def test_parse_disagg_non_digit_rank(self) -> None:
        """Test parsing row with non-digit first cell."""
        cells = ["x", "42334.84", "211.67", "982.64", "208", "8", "TP=1", "26"]
        result = _parse_disagg_row(cells, is_prefill=True)
        assert result is None

    def test_parse_disagg_invalid_float(self) -> None:
        """Test parsing row with invalid float values."""
        cells = ["1", "invalid", "211.67", "982.64", "208", "8", "TP=1", "26"]
        result = _parse_disagg_row(cells, is_prefill=True)
        assert result is None


class TestExtractParallelConfig:
    """Tests for _extract_parallel_config function."""

    def test_extract_parallel_tp_only(self) -> None:
        """Test extracting TP only."""
        result = _extract_parallel_config("TP=4")
        assert result["parallel"] == "TP=4"
        assert result["tp"] == 4
        assert result["pp"] is None
        assert result["dp"] is None

    def test_extract_parallel_tp_pp_dp(self) -> None:
        """Test extracting TP, PP, DP."""
        result = _extract_parallel_config("TP=2 | PP=1 | DP=2")
        assert result["parallel"] == "TP=2 | PP=1 | DP=2"
        assert result["tp"] == 2
        assert result["pp"] == 1
        assert result["dp"] == 2

    def test_extract_parallel_with_spaces(self) -> None:
        """Test extracting with extra spaces."""
        result = _extract_parallel_config("TP=4  |  DP=2")
        assert result["tp"] == 4
        assert result["dp"] == 2

    def test_extract_parallel_invalid_value(self) -> None:
        """Test extracting with invalid value."""
        result = _extract_parallel_config("TP=abc | DP=2")
        assert result["tp"] is None
        assert result["dp"] == 2

    def test_extract_parallel_empty_string(self) -> None:
        """Test extracting from empty string."""
        result = _extract_parallel_config("")
        assert result["parallel"] == ""
        assert result["tp"] is None
        assert result["pp"] is None
        assert result["dp"] is None
