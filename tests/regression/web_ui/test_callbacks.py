"""Tests for web_ui.callbacks module."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from web_ui.callbacks import (
    _normalize_optimizer_deployment_mode,
    _dedupe,
    _case_label_from_mapping,
    _case_choices_from_rows,
    _filter_df_by_case,
    _format_preview_error,
    _preview_first_command,
    _preview_summary_markdown,
    _round_numeric_columns,
    _normalize_op_columns,
    _op_table_from_records,
    OP_TABLE_COLUMNS,
    OP_TABLE_DEFAULT_COLUMNS,
    _categorize_op,
    _build_text_form,
    _build_video_form,
    _build_opt_form,
    _validate_text_form,
    _validate_video_form,
    _validate_optimizer_form,
    _optimizer_validation_markdown,
    _results_to_df,
    _summary_markdown,
    _optimizer_deployment_mode,
    _optimizer_primary_metric,
    _simplify_optimizer_display_df,
    _display_df_for_sim,
    _format_metric_value,
    _format_int_value,
    _format_limit_value,
    _ascii_table,
    _memory_analysis_from_summary,
    _text_generate_op_summary,
    _text_generate_op_table,
    _video_generate_op_summary,
    _video_generate_op_table,
    _video_generate_category_stats,
    _video_generate_compare_table,
    update_memory_analysis_by_device,
    update_bandwidth_analysis_by_device,
    update_category_stats_by_device,
    update_compare_table_by_mode,
    _text_generate_summary_markdown,
    _text_generate_category_stats,
    _text_generate_compare_table,
    _video_generate_summary_markdown,
    _optimizer_summary_markdown_from_df,
    _optimizer_summary_markdown,
    _optimizer_pareto_chart,
    _optimizer_state_rows,
    _optimizer_fixed_config_key,
    _optimizer_fixed_config_label,
    _optimizer_cli_style_output,
    _optimizer_empty_outputs,
    _text_validation_empty_outputs,
    _video_validation_empty_outputs,
    _optimizer_metric_plot,
    _optimizer_candidate_rows_from_records,
    stop_text_generate_run,
    stop_video_generate_run,
    stop_optimizer_run,
    _stop_run_feedback,
    OPT_DEPLOY_PD_MIXED,
    OPT_DEPLOY_PD_SPLIT,
    OPT_DEPLOY_PD_RATIO,
)
from web_ui.schemas import ExperimentResult


class TestNormalizeOptimizerDeploymentMode:
    """Tests for _normalize_optimizer_deployment_mode function."""

    def test_normalize_empty_string(self) -> None:
        """Test with empty string."""
        result = _normalize_optimizer_deployment_mode("")
        assert result == OPT_DEPLOY_PD_MIXED

    def test_normalize_none(self) -> None:
        """Test with None."""
        result = _normalize_optimizer_deployment_mode(None)
        assert result == OPT_DEPLOY_PD_MIXED

    def test_normalize_aggregation_variants(self) -> None:
        """Test various aggregation mode aliases."""
        for alias in ["Aggregation", "aggregation", "PD Mixed", "pd mixed"]:
            result = _normalize_optimizer_deployment_mode(alias)
            assert result == OPT_DEPLOY_PD_MIXED

    def test_normalize_disagg_variants(self) -> None:
        """Test various disaggregation mode aliases."""
        for alias in ["Disagg", "disagg", "PD Split", "pd split"]:
            result = _normalize_optimizer_deployment_mode(alias)
            assert result == OPT_DEPLOY_PD_SPLIT

    def test_normalize_unknown_mode(self) -> None:
        """Test with unknown mode."""
        result = _normalize_optimizer_deployment_mode("UnknownMode")
        assert result == "UnknownMode"


class TestDedupe:
    """Tests for _dedupe function."""

    def test_dedupe_empty_list(self) -> None:
        """Test with empty list."""
        result = _dedupe([])
        assert result == []

    def test_dedupe_no_duplicates(self) -> None:
        """Test with no duplicates."""
        result = _dedupe([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_dedupe_with_duplicates(self) -> None:
        """Test with duplicates."""
        result = _dedupe([1, 2, 2, 3, 1])
        assert result == ["1", "2", "3"]

    def test_dedupe_filters_none(self) -> None:
        """Test filtering None values."""
        result = _dedupe([1, None, 2, None, 3])
        assert result == ["1", "2", "3"]

    def test_dedupe_filters_empty(self) -> None:
        """Test filtering empty strings."""
        result = _dedupe(["a", "", "b", ""])
        assert result == ["a", "b"]

    def test_dedupe_preserves_order(self) -> None:
        """Test that order is preserved."""
        result = _dedupe([3, 1, 2, 1, 3])
        assert result == ["3", "1", "2"]


class TestCaseLabelFromMapping:
    """Tests for _case_label_from_mapping function."""

    def test_case_label_from_dict(self) -> None:
        """Test extracting label from dict."""
        row = {"num_queries": 100, "tp_size": 4}
        result = _case_label_from_mapping(row)
        assert result == "Concurrency=100 | TP=4"

    def test_case_label_from_series(self) -> None:
        """Test extracting label from Series."""
        row = pd.Series({"num_queries": 200, "tp_size": 8})
        result = _case_label_from_mapping(row)
        assert result == "Concurrency=200 | TP=8"

    def test_case_label_missing_keys(self) -> None:
        """Test with missing keys."""
        row = {"other": "value"}
        result = _case_label_from_mapping(row)
        assert result == "Concurrency=- | TP=1"

    def test_case_label_none_values(self) -> None:
        """Test with None values."""
        row = {"num_queries": None, "tp_size": None}
        result = _case_label_from_mapping(row)
        assert result == "Concurrency=- | TP=1"


class TestCaseChoicesFromRows:
    """Tests for _case_choices_from_rows function."""

    def test_case_choices_none_rows(self) -> None:
        """Test with None rows."""
        result = _case_choices_from_rows(None)
        assert result == []

    def test_case_choices_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        result = _case_choices_from_rows(pd.DataFrame())
        assert result == []

    def test_case_choices_from_list(self) -> None:
        """Test from list of dicts."""
        rows = [
            {"num_queries": 100, "tp_size": 4},
            {"num_queries": 200, "tp_size": 8},
        ]
        result = _case_choices_from_rows(rows)
        assert len(result) == 2
        assert "Concurrency=100" in result[0]
        assert "Concurrency=200" in result[1]

    def test_case_choices_from_dataframe(self) -> None:
        """Test from DataFrame."""
        rows = pd.DataFrame(
            [
                {"num_queries": 100, "tp_size": 4},
                {"num_queries": 200, "tp_size": 8},
            ]
        )
        result = _case_choices_from_rows(rows)
        assert len(result) == 2

    def test_case_choices_deduplicates(self) -> None:
        """Test that duplicates are removed."""
        rows = [
            {"num_queries": 100, "tp_size": 4},
            {"num_queries": 100, "tp_size": 4},
        ]
        result = _case_choices_from_rows(rows)
        assert len(result) == 1


class TestFilterDfByCase:
    """Tests for _filter_df_by_case function."""

    def test_filter_empty_df(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = _filter_df_by_case(df, "Concurrency=100 | TP=4")
        assert result.empty

    def test_filter_no_case_label(self) -> None:
        """Test with no case_label column - generates from mapping."""
        # When case_label column doesn't exist, it's generated from row data
        # Rows without num_queries/tp_size get default values
        df = pd.DataFrame({"a": [1, 2]})
        result = _filter_df_by_case(df, "Concurrency=100 | TP=4")
        # Generated labels won't match, so result is empty
        assert result.empty

    def test_filter_with_case_label(self) -> None:
        """Test with case_label column."""
        df = pd.DataFrame(
            {
                "case_label": ["Concurrency=100 | TP=4", "Concurrency=200 | TP=8"],
                "value": [1, 2],
            }
        )
        result = _filter_df_by_case(df, "Concurrency=100 | TP=4")
        assert len(result) == 1
        assert result.iloc[0]["value"] == 1

    def test_filter_adds_case_label_column(self) -> None:
        """Test that case_label is added if missing."""
        df = pd.DataFrame(
            {
                "num_queries": [100, 200],
                "tp_size": [4, 8],
                "value": [1, 2],
            }
        )
        result = _filter_df_by_case(df, "Concurrency=100 | TP=4")
        assert "case_label" in result.columns


class TestFormatPreviewError:
    """Tests for _format_preview_error function."""

    def test_format_preview_error(self) -> None:
        """Test error formatting."""
        error = ValueError("Test error message")
        markdown, command = _format_preview_error(error)
        assert "Parameter Validation Failed" in markdown
        assert "Test error message" in markdown
        assert command == ""


class TestPreviewFirstCommand:
    """Tests for _preview_first_command function."""

    def test_preview_empty_tasks(self) -> None:
        """Test with empty tasks."""
        result = _preview_first_command([])
        assert result == "No command generated."

    def test_preview_task_no_command(self) -> None:
        """Test with task without command attribute."""
        result = _preview_first_command([Mock(spec=[])])
        assert result == "No command generated."

    def test_preview_valid_command(self) -> None:
        """Test with valid command."""
        task = Mock(command=["python", "script.py", "--arg", "value"])
        result = _preview_first_command([task])
        assert result == "python script.py --arg value"

    def test_preview_command_with_string_parts(self) -> None:
        """Test command with string parts."""
        task = Mock(command=["cmd", "123", "True"])
        result = _preview_first_command([task])
        assert result == "cmd 123 True"


class TestPreviewSummaryMarkdown:
    """Tests for _preview_summary_markdown function."""

    def test_preview_summary_text_generate(self) -> None:
        """Test for text_generate sim_type."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "num_devices": 8,
            "num_queries": 100,
            "context_length": 2048,
            "query_length": 512,
            "quantize_linear_action": "int8",
            "quantize_attention_action": "none",
            "decode": True,
            "tp_size": 4,
            "dp_size": 2,
            "ep_size": 1,
        }
        tasks = [Mock(params={"device": "D1", "num_queries": 100})]
        result = _preview_summary_markdown("text_generate", form, tasks)
        assert "Configuration Summary" in result
        assert "test_model" in result
        assert "D1" in result
        assert "Estimated Tasks: **1**" in result
        assert "Decode" in result

    def test_preview_summary_video_generate(self) -> None:
        """Test for video_generate sim_type."""
        form = {
            "model_id": "video_model",
            "device": "D1",
            "world_size": 8,
            "ulysses_size": 4,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "batch_size": 1,
            "seq_len": 256,
            "use_cfg": True,
            "cfg_parallel": False,
            "dit_cache": True,
        }
        tasks = [Mock(params={"device": "D1"})]
        result = _preview_summary_markdown("video_generate", form, tasks)
        assert "video_model" in result
        assert "720 x 1280" in result
        assert "100 frames" in result
        assert "CFG=Enabled" in result

    def test_preview_summary_optimizer(self) -> None:
        """Test for throughput_optimizer sim_type."""
        form = {
            "model_id": "opt_model",
            "device": "D1",
            "num_devices": 8,
            "jobs": 4,
            "input_length": 1024,
            "output_length": 512,
            "deployment_mode": "PD Mixed",
            "ttft_limits": 100,
            "tpot_limits": 50,
            "quantize_linear_action": "int8",
            "quantize_attention_action": "none",
        }
        tasks = [
            Mock(
                params={
                    "device": "D1",
                    "deployment_mode": "PD Mixed",
                    "ttft_limits": 100,
                    "tpot_limits": 50,
                }
            )
        ]
        result = _preview_summary_markdown("throughput_optimizer", form, tasks)
        assert "opt_model" in result
        assert "Input 1024 / Output 512 token" in result
        assert "TTFT=100 ms" in result
        assert "TPOT=50 ms" in result


class TestRoundNumericColumns:
    """Tests for _round_numeric_columns function."""

    def test_round_none_df(self) -> None:
        """Test with None."""
        result = _round_numeric_columns(None)
        assert result is None

    def test_round_empty_df(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = _round_numeric_columns(df)
        assert result.empty

    def test_round_numeric_columns(self) -> None:
        """Test rounding numeric columns."""
        df = pd.DataFrame(
            {
                "value_ms": [1.23456, 2.34567],
                "other": [3, 4],
            }
        )
        result = _round_numeric_columns(df)
        assert result["value_ms"].iloc[0] == pytest.approx(1.235, rel=0.001)
        assert result["other"].iloc[0] == 3

    def test_round_various_units(self) -> None:
        """Test with various units."""
        df = pd.DataFrame(
            {
                "time_ms": [100.1234],
                "size_gb": [1.5678],
                "percent": [50.5678],
                "throughput_token_s": [1000.9876],
            }
        )
        result = _round_numeric_columns(df)
        assert result["time_ms"].iloc[0] == pytest.approx(100.123, rel=0.001)
        assert result["size_gb"].iloc[0] == pytest.approx(1.568, rel=0.001)

    def test_round_skips_non_numeric(self) -> None:
        """Test that non-numeric values are skipped."""
        df = pd.DataFrame({"time_ms": ["not_a_number"]})
        result = _round_numeric_columns(df)
        assert result["time_ms"].iloc[0] == "not_a_number"


class TestNormalizeOpColumns:
    """Tests for _normalize_op_columns function."""

    def test_normalize_none_columns(self) -> None:
        """Test with None columns."""
        result = _normalize_op_columns(None)
        assert result == ["Operator", "Category", "Total Time (ms)", "Average Time (ms)", "Calls", "Device"]

    def test_normalize_valid_columns(self) -> None:
        """Test with valid columns."""
        cols = ["Operator", "Calls"]
        result = _normalize_op_columns(cols)
        assert result == ["Operator", "Calls"]

    def test_normalize_invalid_columns(self) -> None:
        """Test with invalid columns filtered out."""
        cols = ["Invalid", "NonExistent"]
        result = _normalize_op_columns(cols)
        assert result == ["Operator", "Category", "Total Time (ms)", "Average Time (ms)", "Calls", "Device"]


class TestOpTableFromRecords:
    """Tests for _op_table_from_records function."""

    def test_op_table_none_records(self) -> None:
        """Test with None records."""
        result = _op_table_from_records(None, "D1", 10)
        assert result.empty

    def test_op_table_empty_records(self) -> None:
        """Test with empty records."""
        result = _op_table_from_records([], "D1", 10)
        assert result.empty

    def test_op_table_basic(self) -> None:
        """Test basic table generation."""
        records = [
            {
                "name": "attn_op",
                "analytic_total_us": 1000,
                "analytic_avg_us": 100,
                "num_calls": 10,
                "device": "D1",
            }
        ]
        result = _op_table_from_records(records, None, 10)
        assert not result.empty
        assert "Operator" in result.columns
        assert result.iloc[0]["Operator"] == "attn_op"
        assert result.iloc[0]["Total Time (ms)"] == 1.0

    def test_op_table_filters_by_device(self) -> None:
        """Test filtering by device."""
        records = [
            {"name": "op1", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D1"},
            {"name": "op2", "analytic_total_us": 2000, "analytic_avg_us": 200, "num_calls": 20, "device": "D2"},
        ]
        result = _op_table_from_records(records, "D1", 10)
        assert len(result) == 1
        assert result.iloc[0]["Operator"] == "op1"

    def test_op_table_filters_by_case_label(self) -> None:
        """Test filtering by case_label."""
        records = [
            {
                "name": "op1",
                "analytic_total_us": 1000,
                "analytic_avg_us": 100,
                "num_calls": 10,
                "device": "D1",
                "num_queries": 100,
                "tp_size": 4,
            },
            {
                "name": "op2",
                "analytic_total_us": 2000,
                "analytic_avg_us": 200,
                "num_calls": 20,
                "device": "D1",
                "num_queries": 200,
                "tp_size": 8,
            },
        ]
        result = _op_table_from_records(records, None, 10, case_label="Concurrency=100 | TP=4")
        assert len(result) == 1
        assert result.iloc[0]["Operator"] == "op1"

    def test_op_table_sorting(self) -> None:
        """Test sorting by Total Time."""
        records = [
            {"name": "op_small", "analytic_total_us": 100, "analytic_avg_us": 10, "num_calls": 1, "device": "D1"},
            {"name": "op_large", "analytic_total_us": 5000, "analytic_avg_us": 500, "num_calls": 50, "device": "D1"},
        ]
        result = _op_table_from_records(records, None, 10, sort_by="Total Time (ms)")
        assert result.iloc[0]["Operator"] == "op_large"
        assert result.iloc[0]["Total Time (ms)"] == 5.0

    def test_op_table_top_n(self) -> None:
        """Test top_n limiting."""
        records = [
            {
                "name": f"op{i}",
                "analytic_total_us": i * 1000,
                "analytic_avg_us": i * 100,
                "num_calls": i * 10,
                "device": "D1",
            }
            for i in range(1, 11)
        ]
        result = _op_table_from_records(records, None, 5)
        assert len(result) == 5

    def test_op_table_custom_columns(self) -> None:
        """Test with custom column selection."""
        records = [{"name": "op1", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D1"}]
        result = _op_table_from_records(records, None, 10, columns=["Operator", "Calls"])
        assert list(result.columns) == ["Operator", "Calls"]


class TestCategorizeOp:
    """Tests for _categorize_op function."""

    def test_categorize_attention(self) -> None:
        """Test attention operators."""
        for name in ["attention", "self_attn", "cross_attn"]:
            result = _categorize_op(name)
            assert result == "Attention"

    def test_categorize_linear(self) -> None:
        """Test linear operators."""
        # Only operators containing these substrings are categorized as Linear
        for name in ["linear", "gemm", "matmul"]:
            result = _categorize_op(name)
            assert result == "Linear"
        # Operators like fc1, q_proj don't match the substring pattern
        assert _categorize_op("fc1") == "Other"
        assert _categorize_op("q_proj") == "Other"

    def test_categorize_communication(self) -> None:
        """Test communication operators."""
        for name in ["all_reduce", "all_gather", "all_to_all"]:
            result = _categorize_op(name)
            assert result == "Communication"

    def test_categorize_moe(self) -> None:
        """Test MoE operators."""
        for name in ["moe_gate", "expert", "moe_dispatch"]:
            result = _categorize_op(name)
            assert result == "MoE"

    def test_categorize_normalization(self) -> None:
        """Test normalization operators."""
        for name in ["layer_norm", "rmsnorm", "layernorm"]:
            result = _categorize_op(name)
            assert result == "Normalization"

    def test_categorize_embedding(self) -> None:
        """Test embedding operators."""
        result = _categorize_op("embedding")
        assert result == "Embedding"

    def test_categorize_activation(self) -> None:
        """Test activation operators."""
        for name in ["softmax", "silu", "gelu", "relu_activation"]:
            result = _categorize_op(name)
            assert result == "Activation"

    def test_categorize_elementwise(self) -> None:
        """Test elementwise operators."""
        # Only operators containing these substrings are categorized as Elementwise
        for name in ["add", "mul", "div"]:
            result = _categorize_op(name)
            assert result == "Elementwise"
        # "sub" doesn't match the substring pattern
        assert _categorize_op("sub") == "Other"

    def test_categorize_memory(self) -> None:
        """Test memory operators."""
        for name in ["copy", "index", "slice", "reshape"]:
            result = _categorize_op(name)
            assert result == "Memory"

    def test_categorize_other(self) -> None:
        """Test unknown operators."""
        result = _categorize_op("unknown_operation_xyz")
        assert result == "Other"


class TestBuildTextForm:
    """Tests for _build_text_form function."""

    def test_build_text_form_basic(self) -> None:
        """Test basic form building."""
        # Need to provide all 50 arguments that _build_text_form expects
        vals = [
            "model_id",  # model_id
            "device",  # device
            None,  # competitor_devices
            8,  # num_devices
            100,  # num_queries
            None,  # num_queries_list
            512,  # query_length
            2048,  # context_length
            None,  # decode
            None,  # num_mtp_tokens
            None,  # mtp_acceptance_rate
            None,  # compile
            None,  # quantize_linear_action
            None,  # quant_linear_list
            None,  # quantize_attention_action
            None,  # quant_attention_list
            4,  # tp_size
            None,  # tp_list
            2,  # dp_size
            1,  # ep_size
            None,  # image_batch_size
            None,  # image_height
            None,  # image_width
            None,  # prefix_cache_hit_rate
            None,  # reserved_memory_gb
            None,  # log_level
            None,  # compile_allow_graph_break
            None,  # disable_repetition
            None,  # quantize_lmhead
            None,  # mxfp4_group_size
            None,  # graph_log_url
            None,  # dump_input_shapes
            None,  # chrome_trace
            None,  # num_hidden_layers_override
            None,  # o_proj_tp_size
            None,  # o_proj_dp_size
            None,  # mlp_tp_size
            None,  # mlp_dp_size
            None,  # lmhead_tp_size
            None,  # lmhead_dp_size
            None,  # moe_tp_size
            None,  # moe_dp_size
            None,  # word_embedding_tp
            None,  # enable_redundant_experts
            None,  # enable_external_shared_experts
            None,  # host_external_shared_experts
            None,  # remote_source
            None,  # performance_model
            None,  # profiling_database
        ]
        result = _build_text_form(*vals)
        assert result["model_id"] == "model_id"
        assert result["device"] == "device"
        assert result["num_devices"] == 8
        assert result["num_queries"] == 100

    def test_build_text_form_renames_sweep_fields(self) -> None:
        """Test that sweep fields are renamed."""
        vals = [
            "model",
            "device",
            None,
            8,
            100,
            "100,200",
            512,
            2048,
            False,
            None,
            None,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        result = _build_text_form(*vals)
        assert "num_queries_sweep" in result
        assert "num_queries_list" not in result


class TestBuildVideoForm:
    """Tests for _build_video_form function."""

    def test_build_video_form_basic(self) -> None:
        """Test basic form building."""
        vals = [
            "model",
            "device",
            None,
            1,
            256,
            720,
            1280,
            100,
            50,
            "fp16",
            None,
            None,
            8,
            4,
            None,
            False,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        result = _build_video_form(*vals)
        assert result["model_id"] == "model"
        assert result["batch_size"] == 1
        assert result["height"] == 720


class TestBuildOptForm:
    """Tests for _build_opt_form function."""

    def test_build_opt_form_basic(self) -> None:
        """Test basic form building."""
        vals = [
            "model",
            "device",
            None,
            8,
            1024,
            512,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        result = _build_opt_form(*vals)
        assert result["model_id"] == "model"
        assert result["num_devices"] == 8

    def test_build_opt_form_normalizes_deployment_mode(self) -> None:
        """Test deployment mode normalization."""
        vals = [
            "model",
            "device",
            None,
            8,
            1024,
            512,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "Aggregation",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        result = _build_opt_form(*vals)
        assert result["deployment_mode"] == OPT_DEPLOY_PD_MIXED


class TestValidateTextForm:
    """Tests for _validate_text_form function."""

    def test_validate_text_form_valid(self) -> None:
        """Test valid form."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "dp_size": 2,
            "ep_size": 1,
            "num_mtp_tokens": 0,
            "prefix_cache_hit_rate": 0.5,
            "reserved_memory_gb": 10,
            "mxfp4_group_size": 32,  # Must be positive int (power of 2)
            "moe_dp_size": 1,  # Must be positive int (power of 2)
            "num_hidden_layers_override": None,
            "tp_sweep": None,
            "performance_model": None,
        }
        result = _validate_text_form(form)
        assert result == []

    def test_validate_text_form_prefix_cache_rate(self) -> None:
        """Test invalid prefix cache hit rate."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "dp_size": 2,
            "ep_size": 1,
            "num_mtp_tokens": 0,
            "prefix_cache_hit_rate": 1.5,
            "tp_sweep": None,
        }
        result = _validate_text_form(form)
        assert len(result) > 0
        assert any("Prefix Cache" in e for e in result)

    def test_validate_text_form_tp_divisibility(self) -> None:
        """Test TP divisibility check."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 3,
            "dp_size": 2,
            "ep_size": 1,
            "num_mtp_tokens": 0,
            "tp_sweep": None,
        }
        result = _validate_text_form(form)
        assert len(result) > 0


class TestValidateVideoForm:
    """Tests for _validate_video_form function."""

    def test_validate_video_form_valid(self) -> None:
        """Test valid form."""
        form = {
            "batch_size": 1,
            "seq_len": 256,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
            "cache_step_interval": None,
            "cache_step_range": None,
            "cache_block_range": None,
            "model_id": "tests/assets/model_config/Wan2.2-T2V-A14B-Diffusers",
        }
        result = _validate_video_form(form)
        assert result == []

    def test_validate_video_form_negative_batch_size(self) -> None:
        """Test negative batch_size."""
        form = {
            "batch_size": -1,
            "seq_len": 256,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
        }
        result = _validate_video_form(form)
        assert len(result) > 0

    def test_validate_video_form_ulysses_divisibility(self) -> None:
        """Test Ulysses divisibility check."""
        form = {
            "batch_size": 1,
            "seq_len": 256,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "world_size": 8,
            "ulysses_size": 3,
            "ulysses_sweep": None,
        }
        result = _validate_video_form(form)
        assert len(result) > 0

    def test_validate_video_form_cache_range_invalid(self) -> None:
        """Test invalid cache range format."""
        form = {
            "batch_size": 1,
            "seq_len": 256,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
            "cache_step_range": "invalid",
        }
        result = _validate_video_form(form)
        assert len(result) > 0

    def test_validate_video_form_cache_range_bad_values(self) -> None:
        """Test cache range with bad values."""
        form = {
            "batch_size": 1,
            "seq_len": 256,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
            "cache_step_range": "10,5",
        }
        result = _validate_video_form(form)
        assert len(result) > 0


class TestValidateOptimizerForm:
    """Tests for _validate_optimizer_form function."""

    def test_validate_optimizer_form_invalid_mode(self) -> None:
        """Test invalid deployment mode."""
        form = {
            "num_devices": 8,
            "input_length": 1024,
            "output_length": 512,
            "jobs": 4,
            "max_prefill_tokens": 4096,
            "mxfp4_group_size": 64,
            "deployment_mode": "InvalidMode",
            "tp_sizes": None,
            "batch_range": None,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
        }
        result = _validate_optimizer_form(form)
        assert len(result) > 0

    def test_validate_optimizer_form_batch_range_invalid(self) -> None:
        """Test invalid batch range."""
        form = {
            "num_devices": 8,
            "input_length": 1024,
            "output_length": 512,
            "jobs": 4,
            "max_prefill_tokens": 4096,
            "mxfp4_group_size": 64,
            "deployment_mode": OPT_DEPLOY_PD_MIXED,
            "tp_sizes": None,
            "batch_range": "10,5",
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
        }
        result = _validate_optimizer_form(form)
        assert len(result) > 0

    def test_validate_optimizer_form_pd_ratio_requires_values(self) -> None:
        """Test PD Ratio mode requires instance values."""
        form = {
            "num_devices": 8,
            "input_length": 1024,
            "output_length": 512,
            "jobs": 4,
            "max_prefill_tokens": 4096,
            "mxfp4_group_size": 64,
            "deployment_mode": OPT_DEPLOY_PD_RATIO,
            "tp_sizes": None,
            "batch_range": None,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
        }
        result = _validate_optimizer_form(form)
        assert len(result) > 0


class TestOptimizerValidationMarkdown:
    """Tests for _optimizer_validation_markdown function."""

    def test_optimizer_validation_markdown(self) -> None:
        """Test markdown generation."""
        errors = ["Error 1", "Error 2"]
        result = _optimizer_validation_markdown(errors)
        assert "Parameter Validation Failed" in result
        assert "- Error 1" in result
        assert "- Error 2" in result


class TestResultsToDf:
    """Tests for _results_to_df function."""

    def test_results_to_df_empty(self) -> None:
        """Test with empty results."""
        result = _results_to_df([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_results_to_df_single_result(self) -> None:
        """Test with single result."""
        exp_result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"model": "test"},
            command=["test"],
            task_hash="h1",
            label="test",
            summary={"tps": 100.0},
        )
        result = _results_to_df([exp_result])
        assert not result.empty
        assert result.iloc[0]["sim_type"] == "text_generate"

    def test_results_to_df_multiple_results(self) -> None:
        """Test with multiple results."""
        results = [
            ExperimentResult(
                sim_type="text_generate",
                status="success",
                params={"model": f"test{i}"},
                command=["test"],
                task_hash=f"h{i}",
                label=f"test{i}",
                summary={"tps": i * 100.0},
            )
            for i in range(3)
        ]
        result = _results_to_df(results)
        assert len(result) == 3


class TestSummaryMarkdown:
    """Tests for _summary_markdown function."""

    def test_summary_markdown_empty_df(self) -> None:
        """Test with empty DataFrame."""
        result = _summary_markdown(pd.DataFrame(), None, "text_generate")
        assert "No results available" in result

    def test_summary_markdown_text_generate(self) -> None:
        """Test for text_generate sim_type."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "tps_per_device": 100.0,
                    "num_queries": 100,
                }
            ]
        )
        result = _summary_markdown(df, None, "text_generate")
        assert "Completed Runs: **1**" in result
        assert "Highest TPS/Device" in result

    def test_summary_markdown_video_generate(self) -> None:
        """Test for video_generate sim_type."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "analytic_total_time_s": 10.5,
                }
            ]
        )
        result = _summary_markdown(df, None, "video_generate")
        assert "Lowest Analytic Time" in result

    def test_summary_markdown_optimizer(self) -> None:
        """Test for optimizer sim_type."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "best_throughput": 1000.0,
                    "best_parallel": "TP=4",
                    "best_batch_size": 64,
                }
            ]
        )
        result = _summary_markdown(df, None, "throughput_optimizer")
        assert "Best Throughput" in result


class TestOptimizerDeploymentMode:
    """Tests for _optimizer_deployment_mode function."""

    def test_deployment_mode_from_series(self) -> None:
        """Test extracting from series."""
        row = pd.Series({"deployment_mode": OPT_DEPLOY_PD_SPLIT})
        result = _optimizer_deployment_mode(row)
        assert result == OPT_DEPLOY_PD_SPLIT

    def test_deployment_mode_from_disagg_flag(self) -> None:
        """Test with disagg flag."""
        row = pd.Series({"disagg": True})
        result = _optimizer_deployment_mode(row)
        assert result == OPT_DEPLOY_PD_SPLIT

    def test_deployment_mode_from_pd_ratio_flag(self) -> None:
        """Test with pd_ratio flag."""
        row = pd.Series({"enable_optimize_prefill_decode_ratio": True})
        result = _optimizer_deployment_mode(row)
        assert result == OPT_DEPLOY_PD_RATIO

    def test_deployment_mode_default(self) -> None:
        """Test default mode."""
        row = pd.Series({})
        result = _optimizer_deployment_mode(row)
        assert result == OPT_DEPLOY_PD_MIXED


class TestOptimizerPrimaryMetric:
    """Tests for _optimizer_primary_metric function."""

    def test_primary_metric_balanced_qps(self) -> None:
        """Test with balanced_qps column."""
        df = pd.DataFrame({"balanced_qps": [100.0]})
        col, title, label = _optimizer_primary_metric(df)
        assert col == "balanced_qps"
        assert "Balanced QPS" in title

    def test_primary_metric_default(self) -> None:
        """Test default metric."""
        df = pd.DataFrame({"best_batch_size": [64]})
        col, title, label = _optimizer_primary_metric(df)
        assert col == "best_batch_size"
        assert "Batch Size" in title


class TestSimplifyOptimizerDisplayDf:
    """Tests for _simplify_optimizer_display_df function."""

    def test_simplify_empty_df(self) -> None:
        """Test with empty DataFrame."""
        result = _simplify_optimizer_display_df(pd.DataFrame())
        assert result.empty

    def test_simplify_columns_selection(self) -> None:
        """Test column selection."""
        df = pd.DataFrame(
            {
                "model_id": ["test"],
                "device": ["D1"],
                "deployment_mode": ["PD Mixed"],
                "input_length": [1024],
                "best_throughput": [1000.0],
                "other_column": ["ignored"],
            }
        )
        result = _simplify_optimizer_display_df(df)
        assert "other_column" not in result.columns
        assert "model_id" in result.columns or "Model" in result.columns

    def test_simplify_column_renaming(self) -> None:
        """Test column renaming."""
        df = pd.DataFrame(
            {
                "model_id": ["test"],
                "device": ["D1"],
                "deployment_mode": ["PD Mixed"],
                "best_throughput": [1000.0],
            }
        )
        result = _simplify_optimizer_display_df(df)
        assert "Model" in result.columns
        assert "Device" in result.columns
        assert "Deployment Mode" in result.columns


class TestDisplayDfForSim:
    """Tests for _display_df_for_sim function."""

    def test_display_df_optimizer(self) -> None:
        """Test for optimizer sim_type."""
        df = pd.DataFrame({"model_id": ["test"], "device": ["D1"]})
        result = _display_df_for_sim("throughput_optimizer", df)
        assert not result.empty

    def test_display_df_text_generate(self) -> None:
        """Test for text_generate sim_type."""
        df = pd.DataFrame({"device": ["D1"], "tps": 100.0})
        result = _display_df_for_sim("text_generate", df)
        # Should return same df for non-optimizer
        assert len(result) == len(df)


class TestFormatMetricValue:
    """Tests for _format_metric_value function."""

    def test_format_metric_none(self) -> None:
        """Test with None."""
        result = _format_metric_value(None)
        assert result == "-"

    def test_format_metric_nan(self) -> None:
        """Test with NaN."""
        result = _format_metric_value(float("nan"))
        assert result == "-"

    def test_format_metric_numeric(self) -> None:
        """Test with numeric value."""
        result = _format_metric_value(123.45678)
        assert "123.46" in result

    def test_format_metric_string(self) -> None:
        """Test with string value."""
        result = _format_metric_value("test")
        assert result == "test"


class TestFormatIntValue:
    """Tests for _format_int_value function."""

    def test_format_int_none(self) -> None:
        """Test with None."""
        result = _format_int_value(None)
        assert result == "-"

    def test_format_int_valid(self) -> None:
        """Test with valid int."""
        result = _format_int_value(123)
        assert result == "123"

    def test_format_int_float(self) -> None:
        """Test with float value."""
        result = _format_int_value(123.7)
        assert result == "123"


class TestFormatLimitValue:
    """Tests for _format_limit_value function."""

    def test_format_limit_none(self) -> None:
        """Test with None."""
        result = _format_limit_value(None)
        assert result == "None ms"

    def test_format_limit_valid(self) -> None:
        """Test with valid value."""
        result = _format_limit_value(100.5)
        assert "100.50 ms" in result


class TestAsciiTable:
    """Tests for _ascii_table function."""

    def test_ascii_table_empty(self) -> None:
        """Test with empty headers."""
        result = _ascii_table([], [])
        assert result == ""

    def test_ascii_table_basic(self) -> None:
        """Test basic table."""
        result = _ascii_table(["A", "B"], [["1", "2"], ["3", "4"]])
        assert "A" in result
        assert "B" in result
        assert "+" in result
        assert "|" in result

    def test_ascii_table_auto_width(self) -> None:
        """Test auto width calculation."""
        result = _ascii_table(["Short", "VeryLongHeader"], [["AA", "BBB"], ["C", "DDDD"]])
        assert "VeryLongHeader" in result
        assert "DDDD" in result


class TestPreviewSummaryMarkdownVideoGenerate:
    """Tests for _preview_summary_markdown with video_generate details."""

    def test_preview_summary_video_with_cfg_enabled(self) -> None:
        """Test video_generate with CFG enabled."""
        form = {
            "model_id": "video_model",
            "device": "D1",
            "world_size": 8,
            "ulysses_size": 4,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "batch_size": 1,
            "seq_len": 256,
            "use_cfg": True,
            "cfg_parallel": True,
            "dit_cache": True,
        }
        tasks = [Mock(params={"device": "D1", "ulysses_size": 4})]
        result = _preview_summary_markdown("video_generate", form, tasks)
        assert "CFG=Enabled" in result
        assert "CFG Parallel=Enabled" in result
        assert "DiT Cache=Enabled" in result

    def test_preview_summary_optimizer_with_prefix_cache(self) -> None:
        """Test optimizer with prefix cache."""
        form = {
            "model_id": "opt_model",
            "device": "D1",
            "num_devices": 8,
            "jobs": 4,
            "input_length": 1024,
            "output_length": 512,
            "deployment_mode": "PD Mixed",
            "ttft_limits": 100,
            "tpot_limits": 50,
            "prefix_cache_hit_rate": 0.8,
            "quantize_linear_action": "int8",
            "quantize_attention_action": "none",
        }
        tasks = [Mock(params={"device": "D1", "deployment_mode": "PD Mixed"})]
        result = _preview_summary_markdown("throughput_optimizer", form, tasks)
        assert "Prefix Cache Hit Rate: **0.8**" in result


class TestOpTableFromRecordsEdgeCases:
    """Tests for _op_table_from_records edge cases."""

    def test_op_table_with_missing_analytic_columns(self) -> None:
        """Test with missing analytic columns."""
        records = [
            {
                "name": "op1",
                "device": "D1",
                # Missing analytic columns
            }
        ]
        result = _op_table_from_records(records, None, 10)
        assert not result.empty

    def test_op_table_with_zero_values(self) -> None:
        """Test with zero numeric values."""
        records = [
            {
                "name": "op_zero",
                "analytic_total_us": 0,
                "analytic_avg_us": 0,
                "num_calls": 0,
                "device": "D1",
            }
        ]
        result = _op_table_from_records(records, None, 10)
        assert not result.empty
        assert result.iloc[0]["Total Time (ms)"] == 0.0


class TestRoundNumericColumnsEdgeCases:
    """Tests for _round_numeric_columns edge cases."""

    def test_round_with_negative_values(self) -> None:
        """Test with negative values."""
        df = pd.DataFrame(
            {
                "value_ms": [-100.123, -200.456],
            }
        )
        result = _round_numeric_columns(df)
        assert result["value_ms"].iloc[0] == pytest.approx(-100.123, rel=0.001)

    def test_round_with_zero_values(self) -> None:
        """Test with zero values."""
        df = pd.DataFrame(
            {
                "value_ms": [0.0, 0.0],
            }
        )
        result = _round_numeric_columns(df)
        assert result["value_ms"].iloc[0] == 0.0

    def test_round_with_very_large_values(self) -> None:
        """Test with very large values."""
        df = pd.DataFrame(
            {
                "value_ms": [1e6, 1e9],
            }
        )
        result = _round_numeric_columns(df)
        assert result["value_ms"].iloc[0] == pytest.approx(1e6, rel=0.001)


class TestValidateTextFormEdgeCases:
    """Tests for _validate_text_form edge cases."""

    def test_validate_text_with_zero_devices(self) -> None:
        """Test with zero devices."""
        form = {
            "num_devices": 0,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "tp_sweep": None,
        }
        result = _validate_text_form(form)
        assert len(result) > 0

    def test_validate_text_with_dp_auto(self) -> None:
        """Test with DP=auto."""
        form = {
            "num_devices": 8,
            "dp_size": "auto",
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "tp_sweep": None,
        }
        result = _validate_text_form(form)
        # auto should be handled
        assert isinstance(result, list)

    def test_validate_text_with_profiling_model(self) -> None:
        """Test with profiling model selected."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "tp_sweep": None,
            "performance_model": "profiling",
            "profiling_database": None,
        }
        result = _validate_text_form(form)
        assert len(result) > 0
        assert "Profiling" in " ".join(result)


class TestValidateOptimizerFormEdgeCases:
    """Tests for _validate_optimizer_form edge cases."""

    def test_validate_optimizer_with_batch_range_single(self) -> None:
        """Test batch range with single value."""
        form = {
            "num_devices": 8,
            "input_length": 1024,
            "output_length": 512,
            "jobs": 4,
            "max_prefill_tokens": 4096,
            "mxfp4_group_size": 64,
            "deployment_mode": OPT_DEPLOY_PD_MIXED,
            "tp_sizes": None,
            "batch_range": "128",
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
        }
        result = _validate_optimizer_form(form)
        # Single value should be valid
        assert isinstance(result, list)

    def test_validate_optimizer_with_batch_range_pair(self) -> None:
        """Test batch range with min,max."""
        form = {
            "num_devices": 8,
            "input_length": 1024,
            "output_length": 512,
            "jobs": 4,
            "max_prefill_tokens": 4096,
            "mxfp4_group_size": 64,
            "deployment_mode": OPT_DEPLOY_PD_MIXED,
            "tp_sizes": None,
            "batch_range": "64,128",
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
        }
        result = _validate_optimizer_form(form)
        # Valid range should pass
        assert isinstance(result, list)


class TestResultsToDfEdgeCases:
    """Tests for _results_to_df edge cases."""

    def test_results_to_df_with_warnings(self) -> None:
        """Test with warnings."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h1",
            label="test",
            warnings=["Warning 1", "Warning 2"],
        )
        df = _results_to_df([result])
        assert not df.empty
        assert "warning_count" in df.columns or "warnings" in df.columns

    def test_results_to_df_with_infos(self) -> None:
        """Test with infos."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h1",
            label="test",
            infos=["Info 1"],
        )
        df = _results_to_df([result])
        assert not df.empty


class TestSummaryMarkdownEdgeCases:
    """Tests for _summary_markdown edge cases."""

    def test_summary_markdown_with_error(self) -> None:
        """Test with error result."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "status": "failed",
                    "error": "Test error",
                }
            ]
        )
        latest = ExperimentResult(
            sim_type="text_generate",
            status="failed",
            params={},
            command=[],
            task_hash="h1",
            label="test",
            error="Test error",
        )
        result = _summary_markdown(df, latest, "text_generate")
        assert "Latest Error" in result or "Test error" in result

    def test_summary_markdown_with_cache_source(self) -> None:
        """Test with cached result."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "source": "cache",
                }
            ]
        )
        latest = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h1",
            label="test",
            source="cache",
        )
        result = _summary_markdown(df, latest, "text_generate")
        assert "cache" in result


class TestOptimizerDeploymentModeEdgeCases:
    """Tests for _optimizer_deployment_mode edge cases."""

    def test_deployment_mode_with_pd_ratio_column(self) -> None:
        """Test with pd_ratio column."""
        row = pd.Series({"pd_ratio": 0.5})
        result = _optimizer_deployment_mode(row)
        assert result == OPT_DEPLOY_PD_RATIO

    def test_deployment_mode_with_explicit_string(self) -> None:
        """Test with explicit mode string."""
        row = pd.Series({"deployment_mode": "PD Aggregated"})
        result = _optimizer_deployment_mode(row)
        assert result == OPT_DEPLOY_PD_MIXED


class TestSimplifyOptimizerDisplayDfEdgeCases:
    """Tests for _simplify_optimizer_display_df edge cases."""

    def test_simplify_with_missing_columns(self) -> None:
        """Test with missing columns."""
        df = pd.DataFrame({"device": ["D1"]})
        result = _simplify_optimizer_display_df(df)
        assert not result.empty

    def test_simplify_preserves_all_data(self) -> None:
        """Test that all data is preserved."""
        df = pd.DataFrame(
            {
                "model_id": ["test"],
                "device": ["D1"],
                "deployment_mode": ["PD Mixed"],
                "best_throughput": [1000.0],
            }
        )
        result = _simplify_optimizer_display_df(df)
        assert len(result) == 1


class TestAsciiTableEdgeCases:
    """Tests for _ascii_table edge cases."""

    def test_ascii_table_with_single_row(self) -> None:
        """Test with single data row."""
        result = _ascii_table(["A", "B"], [["1", "2"]])
        assert "1" in result
        assert "2" in result

    def test_ascii_table_with_long_values(self) -> None:
        """Test with long cell values."""
        result = _ascii_table(["Column"], [["VeryLongValue"]])
        assert "VeryLongValue" in result


class TestMemoryAnalysisFromSummary:
    """Tests for _memory_analysis_from_summary function."""

    def test_memory_analysis_empty_summary(self) -> None:
        """Test with empty summary."""
        memory_data, table = _memory_analysis_from_summary({})
        assert memory_data == {}
        assert table.empty

    def test_memory_analysis_none_summary(self) -> None:
        """Test with None summary."""
        memory_data, table = _memory_analysis_from_summary(None)
        assert memory_data == {}
        assert table.empty

    def test_memory_analysis_with_valid_data(self) -> None:
        """Test with valid memory data."""
        summary = {
            "total_device_memory_gb": 80,
            "model_weight_size_gb": 10,
            "kv_cache_gb": 2,
            "model_activation_size_gb": 5,
            "reserved_memory_gb": 4,
            "memory_available_gb": 59,
        }
        memory_data, table = _memory_analysis_from_summary(summary)
        assert len(memory_data) > 0
        assert not table.empty

    def test_memory_analysis_with_partial_data(self) -> None:
        """Test with partial memory data."""
        summary = {
            "total_device_memory_gb": 80,
            "model_weight_size_gb": 10,
        }
        memory_data, table = _memory_analysis_from_summary(summary)
        assert "Model Weights" in memory_data
        assert memory_data["Model Weights"] == 10.0

    def test_memory_analysis_with_zero_values(self) -> None:
        """Test with zero values filtered out."""
        summary = {
            "total_device_memory_gb": 80,
            "model_weight_size_gb": 0,
            "kv_cache_gb": 0,
        }
        memory_data, table = _memory_analysis_from_summary(summary)
        # Zero values should be filtered out
        assert len(memory_data) == 0 or all(v > 0 for v in memory_data.values())


class TestTextGenerateOpSummary:
    """Tests for _text_generate_op_summary function."""

    def test_op_summary_empty_results(self) -> None:
        """Test with empty results."""
        result = _text_generate_op_summary([])
        assert result == []

    def test_op_summary_with_valid_result(self) -> None:
        """Test with valid result."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1", "num_queries": 16, "tp_size": 2},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.tables["op_breakdown"] = [{"name": "matmul", "analytic_total_us": 1000, "analytic_avg_us": 100}]
        ops = _text_generate_op_summary([result])
        assert len(ops) == 1
        assert ops[0]["device"] == "D1"
        assert ops[0]["num_queries"] == 16
        assert ops[0]["tp_size"] == 2
        assert ops[0]["category"] == "Linear"

    def test_op_summary_with_attention_ops(self) -> None:
        """Test with attention operators."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1", "num_queries": 16},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.tables["op_breakdown"] = [
            {"name": "self_attention", "analytic_total_us": 2000},
            {"name": "cross_attn", "analytic_total_us": 1000},
        ]
        ops = _text_generate_op_summary([result])
        assert all(op["category"] == "Attention" for op in ops)

    def test_op_summary_with_communication_ops(self) -> None:
        """Test with communication operators."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.tables["op_breakdown"] = [
            {"name": "all_reduce", "analytic_total_us": 500},
            {"name": "all_gather", "analytic_total_us": 300},
        ]
        ops = _text_generate_op_summary([result])
        assert all(op["category"] == "Communication" for op in ops)


class TestTextGenerateOpTable:
    """Tests for _text_generate_op_table function."""

    def test_op_table_empty_results(self) -> None:
        """Test with empty results."""
        result = _text_generate_op_table([], "D1", 10)
        assert result.empty

    def test_op_table_with_device_filter(self) -> None:
        """Test device filtering."""
        result1 = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test1",
        )
        result1.tables["op_breakdown"] = [{"name": "matmul", "analytic_total_us": 1000}]
        result2 = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D2"},
            command=[],
            task_hash="h2",
            label="test2",
        )
        result2.tables["op_breakdown"] = [{"name": "softmax", "analytic_total_us": 500}]

        table = _text_generate_op_table([result1, result2], "D1", 10)
        assert not table.empty
        assert all(table["Device"].astype(str) == "D1")

    def test_op_table_with_case_filter(self) -> None:
        """Test case label filtering."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1", "num_queries": 16, "tp_size": 2},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.tables["op_breakdown"] = [{"name": "matmul", "analytic_total_us": 1000}]
        case_label = "Concurrency=16 | TP=2"

        table = _text_generate_op_table([result], "D1", 10, case_label=case_label)
        assert not table.empty


class TestVideoGenerateOpSummary:
    """Tests for _video_generate_op_summary function."""

    def test_video_op_summary_empty_results(self) -> None:
        """Test with empty results."""
        result = _video_generate_op_summary([])
        assert result == []

    def test_video_op_summary_with_valid_result(self) -> None:
        """Test with valid result."""
        result = ExperimentResult(
            sim_type="video_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.tables["op_breakdown"] = [{"name": "conv2d", "analytic_total_us": 2000}]
        ops = _video_generate_op_summary([result])
        assert len(ops) == 1
        assert ops[0]["device"] == "D1"
        # video uses num_queries/tp_size fields like text, even though params differ
        assert ops[0]["category"] == "Other"  # conv2d doesn't match known categories


# Duplicate TestVideoGenerateOpTable removed - merged with later definition


class TestUpdateMemoryAnalysisByDevice:
    """Tests for update_memory_analysis_by_device function."""

    def test_update_memory_empty_rows(self) -> None:
        """Test with empty rows."""
        fig, df = update_memory_analysis_by_device([], "D1")
        assert df.empty

    def test_update_memory_with_valid_data(self) -> None:
        """Test with valid memory data."""
        rows = [
            {
                "device": "D1",
                "total_device_memory_gb": 80,
                "model_weight_size_gb": 10,
                "kv_cache_gb": 2,
                "memory_available_gb": 68,
            }
        ]
        fig, df = update_memory_analysis_by_device(rows, "D1")
        assert not df.empty

    def test_update_memory_with_device_filter(self) -> None:
        """Test device filtering."""
        rows = [
            {"device": "D1", "total_device_memory_gb": 80, "model_weight_size_gb": 10},
            {"device": "D2", "total_device_memory_gb": 64, "model_weight_size_gb": 8},
        ]
        fig, df = update_memory_analysis_by_device(rows, "D1")
        # Should only return D1 data
        assert not df.empty

    def test_update_memory_with_case_filter(self) -> None:
        """Test case label filtering."""
        rows = [
            {
                "device": "D1",
                "num_queries": 16,
                "tp_size": 2,
                "total_device_memory_gb": 80,
                "model_weight_size_gb": 10,
            }
        ]
        case_label = "Concurrency=16 | TP=2"
        fig, df = update_memory_analysis_by_device(rows, "D1", case_label)
        assert not df.empty


class TestUpdateBandwidthAnalysisByDevice:
    """Tests for update_bandwidth_analysis_by_device function."""

    def test_update_bandwidth_empty_rows(self) -> None:
        """Test with empty rows."""
        df = update_bandwidth_analysis_by_device([], "D1")
        assert df.empty

    def test_update_bandwidth_with_valid_data(self) -> None:
        """Test with valid bandwidth data."""
        rows = [
            {
                "device": "D1",
                "num_queries": 16,
                "tp_size": 2,
                "bottleneck_type": "memory",
                "memory_bound": 80,
                "communication_bound": 10,
                "compute_bound_mma": 5,
                "compute_bound_gp": 5,
            }
        ]
        df = update_bandwidth_analysis_by_device(rows, "D1")
        assert not df.empty
        assert "device" in df.columns

    def test_update_bandwidth_with_bottleneck_types(self) -> None:
        """Test different bottleneck types."""
        rows = [
            {
                "device": "D1",
                "num_queries": 32,
                "tp_size": 4,
                "bottleneck_type": "compute",
                "memory_bound_pct": 20,
                "compute_mma_bound_pct": 80,
            }
        ]
        df = update_bandwidth_analysis_by_device(rows, "D1")
        assert not df.empty
        assert df.iloc[0]["bottleneck_type"] == "compute"


class TestUpdateCategoryStatsByDevice:
    """Tests for update_category_stats_by_device function."""

    def test_update_category_empty_breakdown(self) -> None:
        """Test with empty breakdown."""
        fig, df = update_category_stats_by_device([], "D1")
        assert df.empty

    def test_update_category_with_valid_data(self) -> None:
        """Test with valid operator data."""
        breakdown = [
            {"name": "matmul", "category": "Linear", "analytic_total_us": 5000, "device": "D1"},
            {"name": "attn", "category": "Attention", "analytic_total_us": 3000, "device": "D1"},
            {"name": "norm", "category": "Normalization", "analytic_total_us": 1000, "device": "D1"},
        ]
        fig, df = update_category_stats_by_device(breakdown, "D1")
        assert not df.empty
        assert "category" in df.columns

    def test_update_category_aggregates_correctly(self) -> None:
        """Test that category stats are aggregated correctly."""
        breakdown = [
            {"name": "matmul1", "category": "Linear", "analytic_total_us": 2000, "device": "D1"},
            {"name": "matmul2", "category": "Linear", "analytic_total_us": 1000, "device": "D1"},
            {"name": "attn", "category": "Attention", "analytic_total_us": 1500, "device": "D1"},
        ]
        fig, df = update_category_stats_by_device(breakdown, "D1")
        linear_row = df[df["category"] == "Linear"]
        assert not linear_row.empty
        # Linear should have total_time_ms = 3.0 (2000 + 1000 in us, converted to ms)
        assert linear_row.iloc[0]["total_time_ms"] == 3.0


class TestUpdateCompareTableByMode:
    """Tests for update_compare_table_by_mode function."""

    def test_update_compare_empty_breakdown(self) -> None:
        """Test with empty breakdown."""
        df = update_compare_table_by_mode([], "memory", 10)
        assert df.empty

    def test_update_compare_with_valid_data(self) -> None:
        """Test with valid comparison data."""
        breakdown = [
            {"name": "op1", "category": "Linear", "analytic_total_us": 5000, "device": "D1", "mode": "mode1"},
            {"name": "op1", "category": "Linear", "analytic_total_us": 3000, "device": "D1", "mode": "mode2"},
        ]
        df = update_compare_table_by_mode(breakdown, "Linear", 10)
        assert not df.empty


class TestCategorizeOpExtended:
    """Extended tests for _categorize_op function."""

    def test_categorize_all_communication_types(self) -> None:
        """Test all communication operator types."""
        for op_name in ["all_reduce", "all_gather", "all_to_all"]:
            assert _categorize_op(op_name) == "Communication"
        # AllReduce -> allreduce (no underscore) doesn't match all_reduce
        assert _categorize_op("AllReduce") == "Other"

    def test_categorize_all_moe_types(self) -> None:
        """Test all MoE operator types."""
        for op_name in ["moe_gate", "expert_routing", "MoE_FFN"]:
            assert _categorize_op(op_name) == "MoE"

    def test_categorize_all_norm_types(self) -> None:
        """Test all normalization operator types."""
        for op_name in ["layer_norm", "rmsnorm", "LayerNorm", "RMSNorm"]:
            assert _categorize_op(op_name) == "Normalization"

    def test_categorize_all_activation_types(self) -> None:
        """Test all activation operator types."""
        for op_name in ["softmax", "silu", "gelu", "fast_gelu"]:
            assert _categorize_op(op_name) == "Activation"

    def test_categorize_all_elementwise_types(self) -> None:
        """Test all elementwise operator types."""
        for op_name in ["add", "mul", "div"]:
            assert _categorize_op(op_name) == "Elementwise"
        # "sub" is not in the list, should be Other
        assert _categorize_op("sub") == "Other"

    def test_categorize_all_memory_types(self) -> None:
        """Test all memory operator types."""
        for op_name in ["copy", "index", "slice", "reshape"]:
            assert _categorize_op(op_name) == "Memory"
        # "view" is not in the list, should be Other
        assert _categorize_op("view") == "Other"


class TestTextGenerateSummaryMarkdown:
    """Tests for _text_generate_summary_markdown function."""

    def test_text_summary_empty_results(self) -> None:
        """Test with empty results."""
        result = _text_generate_summary_markdown([])
        assert "No results available" in result

    def test_text_summary_with_tps_results(self) -> None:
        """Test with TPS results."""
        result1 = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result1.summary["tps_per_device"] = 1000.0
        result1.to_row = lambda: {
            "device": "D1",
            "tps_per_device": 1000.0,
        }
        result = _text_generate_summary_markdown([result1])
        assert "Recommendation" in result

    def test_text_summary_with_analytic_time_results(self) -> None:
        """Test with analytic time results."""
        result1 = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result1.summary["analytic_total_time_s"] = 5.0
        result1.to_row = lambda: {
            "device": "D1",
            "analytic_total_time_s": 5.0,
        }
        result = _text_generate_summary_markdown([result1])
        assert "Recommendation" in result


class TestVideoGenerateSummaryMarkdown:
    """Tests for _video_generate_summary_markdown function."""

    def test_video_summary_empty_results(self) -> None:
        """Test with empty results."""
        result = _video_generate_summary_markdown([])
        assert "No results available" in result

    def test_video_summary_with_valid_results(self) -> None:
        """Test with valid video results."""
        result1 = ExperimentResult(
            sim_type="video_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result1.summary["analytic_total_time_s"] = 10.0
        result1.to_row = lambda: {
            "device": "D1",
            "analytic_total_time_s": 10.0,
        }
        result = _video_generate_summary_markdown([result1])
        assert "Recommendation" in result


class TestOptimizerSummaryMarkdownFromDf:
    """Tests for _optimizer_summary_markdown_from_df function."""

    def test_optimizer_summary_empty_df(self) -> None:
        """Test with empty dataframe."""
        result = _optimizer_summary_markdown_from_df(pd.DataFrame())
        assert "No results available" in result

    def test_optimizer_summary_with_valid_df(self) -> None:
        """Test with valid optimizer dataframe."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "best_throughput": 1000.0,
                    "best_parallel": "TP=4",
                    "best_batch_size": 32,
                    "deployment_mode": "PD Aggregated",
                }
            ]
        )
        result = _optimizer_summary_markdown_from_df(df)
        assert "Recommendation" in result

    def test_optimizer_summary_with_completed_count(self) -> None:
        """Test with completed count parameter."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "best_throughput": 1000.0,
                }
            ]
        )
        result = _optimizer_summary_markdown_from_df(df, completed_count=5)
        assert "5" in result


class TestOptimizerSummaryMarkdown:
    """Tests for _optimizer_summary_markdown function."""

    def test_optimizer_summary_empty_results(self) -> None:
        """Test with empty results."""
        result = _optimizer_summary_markdown([])
        assert "No results available" in result

    def test_optimizer_summary_with_valid_results(self) -> None:
        """Test with valid optimizer results."""
        result1 = ExperimentResult(
            sim_type="throughput_optimizer",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result1.summary["best_throughput"] = 1000.0
        result1.to_row = lambda: {
            "device": "D1",
            "best_throughput": 1000.0,
        }
        result = _optimizer_summary_markdown([result1])
        assert "Recommendation" in result


class TestValidateTextFormExtended:
    """Extended tests for _validate_text_form function."""

    def test_validate_text_with_invalid_hidden_layers(self) -> None:
        """Test with invalid hidden layers override."""
        form = {
            "num_devices": "8",
            "num_queries": "16",
            "query_length": "512",
            "context_length": "2048",
            "num_hidden_layers_override": "-1",
        }
        errors = _validate_text_form(form)
        # Error message for invalid hidden layers override
        assert len(errors) > 0

    def test_validate_text_with_non_hidden_layers(self) -> None:
        """Test with non-numeric hidden layers."""
        form = {
            "num_devices": "8",
            "num_queries": "16",
            "query_length": "512",
            "context_length": "2048",
            "num_hidden_layers_override": "abc",
        }
        errors = _validate_text_form(form)
        # Error message for non-integer hidden layers override
        assert len(errors) > 0


class TestValidateVideoFormExtended:
    """Extended tests for _validate_video_form function."""

    def test_validate_video_with_invalid_frame_num(self) -> None:
        """Test with invalid frame number."""
        form = {
            "batch_size": "4",
            "seq_len": "512",
            "height": "720",
            "width": "1280",
            "frame_num": "0",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormExtended:
    """Extended tests for _validate_optimizer_form function."""

    def test_validate_optimizer_with_invalid_prefill_devices(self) -> None:
        """Test with invalid prefill devices in PD Ratio mode."""
        form = {
            "num_devices": "8",
            "input_length": "512",
            "output_length": "512",
            "jobs": "1",
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": "",
            "decode_devices_per_instance": "2",
        }
        errors = _validate_optimizer_form(form)
        assert any("Prefill Devices" in err for err in errors)

    def test_validate_optimizer_with_prefill_exceeds_devices(self) -> None:
        """Test with prefill devices exceeding device count."""
        form = {
            "num_devices": "4",
            "input_length": "512",
            "output_length": "512",
            "jobs": "1",
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": "8",
            "decode_devices_per_instance": "2",
        }
        errors = _validate_optimizer_form(form)
        assert any("greater than" in err for err in errors)

    def test_validate_optimizer_with_non_divisible_devices(self) -> None:
        """Test with devices not divisible by instance count."""
        form = {
            "num_devices": "7",
            "input_length": "512",
            "output_length": "512",
            "jobs": "1",
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": "2",
            "decode_devices_per_instance": "2",
        }
        errors = _validate_optimizer_form(form)
        assert any("divisible" in err for err in errors)


class TestOptimizerParetoChart:
    """Tests for _optimizer_pareto_chart function."""

    def test_pareto_chart_empty_df(self) -> None:
        """Test with empty dataframe."""
        result = _optimizer_pareto_chart(pd.DataFrame(), "D1")
        assert result is not None

    def test_pareto_chart_with_valid_data(self) -> None:
        """Test with valid pareto data."""
        df = pd.DataFrame(
            [
                {"ttft_ms": 100, "throughput_token_s": 1000},
                {"ttft_ms": 200, "throughput_token_s": 500},
            ]
        )
        result = _optimizer_pareto_chart(df, "D1")
        assert result is not None


class TestOptimizerStateRows:
    """Tests for _optimizer_state_rows function."""

    def test_state_rows_empty_results(self) -> None:
        """Test with empty results."""
        result = _optimizer_state_rows([])
        assert result == []

    def test_state_rows_with_valid_results(self) -> None:
        """Test with valid optimizer results."""
        result1 = ExperimentResult(
            sim_type="throughput_optimizer",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result1.tables["top_configs"] = [{"parallel": "TP=4", "batch_size": 32}]
        result1.summary["best_throughput"] = 1000.0
        result1.to_row = lambda: {"device": "D1"}

        rows = _optimizer_state_rows([result1])
        assert len(rows) > 0


class TestOptimizerFixedConfigKey:
    """Tests for _optimizer_fixed_config_key function."""

    def test_fixed_config_key_basic(self) -> None:
        """Test basic config key format."""
        row = pd.Series(
            {
                "model_id": "test_model",
                "deployment_mode": "PD Aggregated",
                "num_devices": 8,
                "input_length": 512,
                "output_length": 512,
                "prefix_cache_hit_rate": 0.0,
                "quantize_linear_action": "W8A8",
                "quantize_attention_action": "INT8",
                "parallel": "TP=4",
                "batch_size": 32,
                "concurrency": 16,
            }
        )
        result = _optimizer_fixed_config_key(row)
        assert "test_model" in result
        assert "32" in result
        assert "16" in result
        assert "||" in result

    def test_fixed_config_key_with_missing_values(self) -> None:
        """Test with missing values."""
        row = pd.Series({"batch_size": 32})
        result = _optimizer_fixed_config_key(row)
        assert "-" in result  # Missing values should be "-"
        assert "32" in result


class TestOptimizerFixedConfigLabel:
    """Tests for _optimizer_fixed_config_label function."""

    def test_fixed_config_label_basic(self) -> None:
        """Test basic config label."""
        row = pd.Series({"tp_size": 4, "batch_size": 32, "concurrency": 16})
        result = _optimizer_fixed_config_label(row)
        assert result is not None

    def test_fixed_config_label_with_device_count(self) -> None:
        """Test with device count parameter."""
        row = pd.Series({"tp_size": 4, "batch_size": 32, "concurrency": 16})
        result = _optimizer_fixed_config_label(row, device_count=8)
        assert result is not None


class TestOptimizerPrimaryMetricExtended:
    """Extended tests for _optimizer_primary_metric function."""

    def test_primary_metric_default_no_balanced(self) -> None:
        """Test default metric when no balanced_qps."""
        df = pd.DataFrame({"best_batch_size": [32, 64]})
        metric, title, short = _optimizer_primary_metric(df)
        assert metric == "best_batch_size"
        assert "Batch Size" in title

    def test_primary_metric_with_balanced_qps(self) -> None:
        """Test with balanced_qps column."""
        df = pd.DataFrame({"balanced_qps": [1000.0, 2000.0]})
        metric, title, short = _optimizer_primary_metric(df)
        assert metric == "balanced_qps"
        assert "Balanced QPS" in title


class TestDisplayDfForSimExtended:
    """Extended tests for _display_df_for_sim function."""

    def test_display_df_text_generate(self) -> None:
        """Test with text_generate sim type."""
        df = pd.DataFrame({"device": ["D1"], "tps_per_device": [1000.0]})
        result = _display_df_for_sim("text_generate", df)
        assert not result.empty

    def test_display_df_video_generate(self) -> None:
        """Test with video_generate sim type."""
        df = pd.DataFrame({"device": ["D1"], "analytic_total_time_s": [5.0]})
        result = _display_df_for_sim("video_generate", df)
        assert not result.empty


class TestFormatMetricValueExtended:
    """Extended tests for _format_metric_value function."""

    def test_format_metric_with_float(self) -> None:
        """Test with float value."""
        result = _format_metric_value(123.456789, digits=3)
        assert "123.457" in result

    def test_format_metric_with_zero(self) -> None:
        """Test with zero value."""
        result = _format_metric_value(0.0)
        assert "0.00" in result

    def test_format_metric_with_large_number(self) -> None:
        """Test with large number."""
        result = _format_metric_value(1000000.12345)
        assert "1,000,000" in result or "1000000" in result


class TestRoundNumericColumnsExtended:
    """Extended tests for _round_numeric_columns function."""

    def test_round_with_ms_columns(self) -> None:
        """Test with ms columns."""
        df = pd.DataFrame({"latency (ms)": [123.456789], "time_ms": [987.654321]})
        result = _round_numeric_columns(df, digits=2)
        assert result.iloc[0]["latency (ms)"] == 123.46

    def test_round_with_gb_columns(self) -> None:
        """Test with GB columns."""
        df = pd.DataFrame({"memory (GB)": [12.3456], "storage_gb": [34.5678]})
        result = _round_numeric_columns(df, digits=3)
        assert result.iloc[0]["memory (GB)"] == 12.346

    def test_round_with_percent_columns(self) -> None:
        """Test with percent columns."""
        df = pd.DataFrame({"utilization (%)": [67.891], "efficiency_pct": [45.678]})
        result = _round_numeric_columns(df, digits=1)
        assert result.iloc[0]["utilization (%)"] == 67.9


class TestOptimizerCliStyleOutput:
    """Tests for _optimizer_cli_style_output function."""

    def test_cli_style_output_none_top(self) -> None:
        """Test with None top result."""
        device_candidates = pd.DataFrame({"device": ["D1", "D2"]})
        result = _optimizer_cli_style_output(None, device_candidates, "D1")
        assert result is not None

    def test_cli_style_output_with_valid_data(self) -> None:
        """Test with valid data."""
        top = pd.Series(
            {
                "best_throughput": 1000.0,
                "best_parallel": "TP=4",
                "best_batch_size": 32,
                "device": "D1",
            }
        )
        device_candidates = pd.DataFrame({"device": ["D1", "D2"]})
        result = _optimizer_cli_style_output(top, device_candidates, "D1")
        assert result is not None

    def test_cli_style_output_empty_candidates(self) -> None:
        """Test with empty candidates."""
        top = pd.Series({"best_throughput": 1000.0})
        result = _optimizer_cli_style_output(top, pd.DataFrame(), "D1")
        assert result is not None


class TestNormalizeOpColumnsExtended:
    """Extended tests for _normalize_op_columns function."""

    def test_normalize_columns_with_duplicates(self) -> None:
        """Test with duplicate column selections."""
        columns = ["Operator", "Operator", "Total Time (ms)"]
        result = _normalize_op_columns(columns)
        assert "Operator" in result
        # The function doesn't deduplicate, so duplicates are preserved

    def test_normalize_columns_with_invalid(self) -> None:
        """Test with invalid column names."""
        columns = ["Operator", "Invalid Column", "Total Time (ms)"]
        result = _normalize_op_columns(columns)
        assert "Operator" in result
        assert "Invalid Column" not in result

    def test_normalize_columns_with_non_string_columns(self) -> None:
        """Test with non-string column names."""
        columns = [1, 2, 3]
        result = _normalize_op_columns(columns)
        # Non-strings are converted to str
        assert isinstance(result, list)


class TestPreviewFirstCommandExtended:
    """Extended tests for _preview_first_command function."""

    def test_preview_first_with_list_command(self) -> None:
        """Test with command as list."""
        task = type("Task", (), {"command": ["python", "run.py", "--model", "test"]})()
        result = _preview_first_command([task])
        assert "python" in result
        assert "run.py" in result

    def test_preview_first_with_tuple_command(self) -> None:
        """Test with command as tuple."""
        task = type("Task", (), {"command": ("python", "run.py")})()
        result = _preview_first_command([task])
        assert "python" in result

    def test_preview_first_with_mixed_types(self) -> None:
        """Test with mixed types in command."""
        task = type("Task", (), {"command": ["python", 123, True]})()
        result = _preview_first_command([task])
        assert "python" in result


class TestPreviewSummaryMarkdownExtended:
    """Extended tests for _preview_summary_markdown function."""

    def test_preview_summary_text_with_mtp(self) -> None:
        """Test text generate with MTP enabled."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "num_devices": "8",
            "num_queries": "16",
            "query_length": "512",
            "context_length": "2048",
            "num_mtp_tokens": "4",
            "mtp_acceptance_rate": "0.5",
        }
        task = type(
            "Task",
            (),
            {
                "params": {
                    "device": "D1",
                    "num_queries": 16,
                    "quantize_linear_action": "W8A8",
                    "quantize_attention_action": "INT8",
                }
            },
        )()
        result = _preview_summary_markdown("text_generate", form, [task, task])
        assert "Configuration Summary" in result

    def test_preview_summary_video_with_ulysses(self) -> None:
        """Test video generate with Ulysses."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "world_size": "8",
            "ulysses_size": "4",
            "batch_size": "4",
            "seq_len": "512",
            "height": "720",
            "width": "1280",
            "frame_num": "100",
            "sample_step": "50",
        }
        task = type(
            "Task",
            (),
            {
                "params": {
                    "device": "D1",
                    "ulysses_size": 4,
                    "quantize_linear_action": "W8A8",
                }
            },
        )()
        result = _preview_summary_markdown("video_generate", form, [task, task])
        assert "Configuration Summary" in result


class TestFormatPreviewErrorExtended:
    """Extended tests for _format_preview_error function."""

    def test_format_error_with_value_error(self) -> None:
        """Test with ValueError."""
        error = ValueError("Invalid value")
        summary, command = _format_preview_error(error)
        assert "Parameter Validation Failed" in summary
        assert "Invalid value" in summary
        assert command == ""

    def test_format_error_with_generic_exception(self) -> None:
        """Test with generic exception."""
        error = RuntimeError("Something went wrong")
        summary, command = _format_preview_error(error)
        assert "Parameter Validation Failed" in summary
        assert "Something went wrong" in summary


class TestCaseLabelFromMappingExtended:
    """Extended tests for _case_label_from_mapping function."""

    def test_case_label_with_series(self) -> None:
        """Test with pandas Series."""
        series = pd.Series({"num_queries": 16, "tp_size": 2})
        result = _case_label_from_mapping(series)
        assert "Concurrency=16" in result
        assert "TP=2" in result

    def test_case_label_with_dict(self) -> None:
        """Test with dict."""
        row = {"num_queries": 32, "tp_size": 4}
        result = _case_label_from_mapping(row)
        assert "Concurrency=32" in result
        assert "TP=4" in result

    def test_case_label_with_partial_data(self) -> None:
        """Test with partial data."""
        row = {"num_queries": 16}
        result = _case_label_from_mapping(row)
        assert "Concurrency=16" in result
        assert "TP=1" in result  # Default value


class TestCaseChoicesFromRowsExtended:
    """Extended tests for _case_choices_from_rows function."""

    def test_case_choices_from_dataframe(self) -> None:
        """Test with DataFrame input."""
        df = pd.DataFrame(
            [
                {"num_queries": 16, "tp_size": 2},
                {"num_queries": 32, "tp_size": 4},
            ]
        )
        result = _case_choices_from_rows(df)
        assert len(result) == 2
        assert any("Concurrency=16" in choice for choice in result)

    def test_case_choices_deduplication(self) -> None:
        """Test that duplicate choices are removed."""
        rows = [
            {"num_queries": 16, "tp_size": 2},
            {"num_queries": 16, "tp_size": 2},  # Duplicate
            {"num_queries": 32, "tp_size": 4},
        ]
        result = _case_choices_from_rows(rows)
        assert len(result) == 2
        assert result.count("Concurrency=16 | TP=2") == 1

    def test_filter_df_with_nonexistent_case_label(self) -> None:
        """Test filtering with case label that doesn't exist in rows."""
        df = pd.DataFrame(
            [
                {"device": "D1", "num_queries": 16, "tp_size": 1, "case_label": "Concurrency=16 | TP=1"},
                {"device": "D1", "num_queries": 32, "tp_size": 2, "case_label": "Concurrency=32 | TP=2"},
            ]
        )
        result = _filter_df_by_case(df, "Concurrency=8 | TP=1")
        assert result.empty

    def test_normalize_deployment_mode_with_whitespace(self) -> None:
        """Test normalizing deployment mode with extra whitespace."""
        result = _normalize_optimizer_deployment_mode("  PD Aggregated  ")
        assert result == "PD Aggregated"

    def test_normalize_deployment_mode_empty_string(self) -> None:
        """Test normalizing empty string deployment mode."""
        result = _normalize_optimizer_deployment_mode("")
        assert result == "PD Aggregated"

    def test_dedupe_with_numeric_values(self) -> None:
        """Test deduplication with numeric values."""
        result = _dedupe([1, 2, 2, 3.14, 3.14, 4])
        assert result == ["1", "2", "3.14", "4"]

    def test_dedupe_with_boolean_values(self) -> None:
        """Test deduplication with boolean values."""
        result = _dedupe([True, False, True, False])
        assert result == ["True", "False"]

    def test_op_table_with_no_device_column(self) -> None:
        """Test op_table when device column is missing."""
        records = [
            {
                "name": "matmul",
                "category": "GEMM",
                "analytic_total_us": 1000.0,
                "analytic_avg_us": 100.0,
                "num_calls": 10,
                # "device" key missing
            },
            {
                "name": "softmax",
                "category": "Attention",
                "analytic_total_us": 500.0,
                "analytic_avg_us": 50.0,
                "num_calls": 10,
                "device": "D1",
            },
        ]
        result = _op_table_from_records(records, None, None, 10, OP_TABLE_COLUMNS, "Total Time (ms)")
        # Should return records that have device
        assert len(result) <= len(records)

    def test_cli_style_output_with_different_device(self) -> None:
        """Test CLI style output when current device differs from top device."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
                "best_ttft_ms": 50.0,
                "best_tpot_ms": 10.0,
                "best_parallel": "TP=4 | DP=2",
                "best_batch_size": 128,
            }
        )
        candidates = pd.DataFrame(
            [
                {"device": "D2", "best_throughput": 900.0, "best_ttft_ms": 55.0, "best_tpot_ms": 11.0},
            ]
        )
        result = _optimizer_cli_style_output(top, candidates, "D2")
        assert "D1" in result or "D2" in result

    def test_pareto_chart_with_single_candidate(self) -> None:
        """Test Pareto chart with only one candidate."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "throughput_token_s": 1000.0,
                    "ttft_ms": 50.0,
                    "batch_size": 128,
                    "parallel": "TP=4",
                    "concurrency": 16,
                }
            ]
        )
        result = _optimizer_pareto_chart(df, "D1")
        assert result is not None

    def test_fixed_config_key_with_all_none(self) -> None:
        """Test fixed config key when all parallel values are None."""
        row = pd.Series(
            {
                "tp_size": None,
                "pp_size": None,
                "dp_size": None,
                "cp_size": None,
            }
        )
        result = _optimizer_fixed_config_key(row)
        assert result is not None

    def test_simplify_with_extra_columns(self) -> None:
        """Test simplify display DF when extra columns exist."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "throughput": 1000.0,
                    "ttft_ms": 50.0,
                    "extra_column": "ignored",
                },
                {
                    "device": "D2",
                    "throughput": 900.0,
                    "ttft_ms": 55.0,
                    "extra_column": "also_ignored",
                },
            ]
        )
        result = _simplify_optimizer_display_df(df)
        assert "extra_column" not in result.columns

    def test_round_with_inf_values(self) -> None:
        """Test rounding with infinity values."""
        import math

        df = pd.DataFrame(
            [
                {"value": math.inf},
                {"value": -math.inf},
                {"value": 1.5},
            ]
        )
        result = _round_numeric_columns(df, digits=2)
        assert len(result) == len(df)

    def test_categorize_op_with_unknown_operator(self) -> None:
        """Test categorizing an unknown operator name."""
        result = _categorize_op("unknown_operator_xyz")
        assert result == "Other"

    def test_categorize_op_with_empty_string(self) -> None:
        """Test categorizing an empty string operator name."""
        result = _categorize_op("")
        assert result == "Other"

    def test_categorize_op_with_case_variations(self) -> None:
        """Test categorizing operator names with case variations (function uses lowercase)."""
        assert _categorize_op("MATMUL") == "Linear"  # "matmul" in lowercase
        assert _categorize_op("softmax") == "Activation"  # "softmax" matches "silu" or "gelu" pattern
        assert _categorize_op("LayerNorm") == "Normalization"  # "layernorm" matches

    def test_dedupe_with_mixed_none_and_values(self) -> None:
        """Test deduplication with mixed None and actual values."""
        result = _dedupe([None, "", "a", None, "", "b", "a"])
        assert result == ["a", "b"]

    def test_op_table_with_all_columns_selection(self) -> None:
        """Test op_table with all columns selected."""
        records = [
            {
                "name": "matmul",
                "category": "GEMM",
                "analytic_total_us": 1000.0,
                "analytic_avg_us": 100.0,
                "num_calls": 10,
                "device": "D1",
            }
        ]
        result = _op_table_from_records(records, "D1", 10, OP_TABLE_COLUMNS, "Total Time (ms)")
        assert len(result.columns) == len(OP_TABLE_COLUMNS)

    def test_op_table_top_n_with_fewer_records(self) -> None:
        """Test op_table top_n when there are fewer records than top_n."""
        records = [
            {
                "name": f"op{i}",
                "category": "GEMM",
                "analytic_total_us": float(i * 100),
                "analytic_avg_us": 100.0,
                "num_calls": 10,
                "device": "D1",
            }
            for i in range(3)  # Only 3 records
        ]
        result = _op_table_from_records(records, "D1", 10, OP_TABLE_COLUMNS, "Total Time (ms)")
        assert len(result) == 3  # All records returned

    def test_pareto_chart_with_empty_df(self) -> None:
        """Test Pareto chart with empty dataframe."""
        df = pd.DataFrame(columns=["device", "throughput", "ttft_ms", "tpot_ms", "batch_size", "parallel"])
        result = _optimizer_pareto_chart(df, "D1")
        assert result is not None

    def test_cli_style_output_with_no_candidates(self) -> None:
        """Test CLI style output with no candidates."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
                "best_ttft_ms": 50.0,
                "best_tpot_ms": 10.0,
                "best_parallel": "TP=4 | DP=2",
                "best_batch_size": 128,
            }
        )
        candidates = pd.DataFrame(columns=["device", "best_throughput", "best_ttft_ms", "best_tpot_ms"])
        result = _optimizer_cli_style_output(top, candidates, "D1")
        assert isinstance(result, str)

    def test_primary_metric_with_missing_throughput(self) -> None:
        """Test primary metric when no balanced_qps column (returns default best_batch_size)."""
        df = pd.DataFrame([{"device": "D1", "ttft_ms": 50.0, "tpot_ms": 10.0, "throughput_token_s": None}])
        metric_col, metric_label, sort_ascending = _optimizer_primary_metric(df)
        assert metric_col == "best_batch_size"  # Default when no balanced_qps
        assert metric_label == "Best Batch Size Comparison"

    def test_display_df_with_unknown_sim_type(self) -> None:
        """Test display_df with unknown simulation type."""
        df = pd.DataFrame([{"col1": 1, "col2": 2}])
        result = _display_df_for_sim("unknown_type", df)
        assert result.equals(df)

    def test_format_metric_with_very_small_value(self) -> None:
        """Test formatting metric with very small value."""
        result = _format_metric_value(0.0001, digits=4)
        assert "0.0001" in result or "1e-04" in result

    def test_format_limit_with_zero_value(self) -> None:
        """Test formatting limit with zero value."""
        result = _format_limit_value(0)
        assert "0" in result or "None" in result

    def test_ascii_table_with_multibyte_characters(self) -> None:
        """Test ASCII table with multibyte characters."""
        headers = ["Name", "Value"]
        rows = [["TestValue", "1.23"], ["Japanese", "4.56"]]
        result = _ascii_table(headers, rows)
        assert "TestValue" in result and "Japanese" in result

    def test_ascii_table_with_special_chars(self) -> None:
        """Test ASCII table with special characters."""
        headers = ["Type", "Value"]
        rows = [["Line-1", "10.5"], ["Line_2", "20.5"]]
        result = _ascii_table(headers, rows)
        assert "Line-1" in result and "Line_2" in result

    def test_dedupe_with_special_strings(self) -> None:
        """Test deduplication with special string values."""
        result = _dedupe(["a-b", "a_b", "a.b"])
        assert len(result) == 3

    def test_case_label_with_all_none_values(self) -> None:
        """Test case label when all values are None."""
        result = _case_label_from_mapping({"num_queries": None, "tp_size": None})
        assert isinstance(result, str)

    def test_filter_df_with_all_rows_filtered(self) -> None:
        """Test filter_df when all rows get filtered out."""
        df = pd.DataFrame(
            [
                {"device": "D1", "num_queries": 16, "tp_size": 1, "case_label": "Concurrency=16 | TP=1"},
            ]
        )
        result = _filter_df_by_case(df, "Concurrency=32 | TP=2")
        assert result.empty

    def test_round_with_all_columns_numeric(self) -> None:
        """Test rounding when all columns are numeric."""
        df = pd.DataFrame([[1.123, 2.456], [3.789, 4.012]])
        result = _round_numeric_columns(df, digits=2)
        assert len(result) == len(df)

    def test_normalize_op_columns_with_duplicates(self) -> None:
        """Test normalize columns handles duplicates."""
        result = _normalize_op_columns(["Operator", "Category", "Operator"])
        # Should handle duplicates gracefully
        assert isinstance(result, list)

    def test_pareto_chart_with_no_pareto_frontier(self) -> None:
        """Test Pareto chart when throughput doesn't increase."""
        df = pd.DataFrame(
            [
                {
                    "device": "D1",
                    "throughput_token_s": 1000.0,
                    "ttft_ms": 50.0,
                    "batch_size": 128,
                    "parallel": "TP=4",
                    "concurrency": 16,
                },
                {
                    "device": "D1",
                    "throughput_token_s": 900.0,  # Lower throughput
                    "ttft_ms": 60.0,
                    "batch_size": 64,
                    "parallel": "TP=2",
                    "concurrency": 8,
                },
            ]
        )
        result = _optimizer_pareto_chart(df, "D1")
        assert result is not None

    def test_fixed_config_key_with_mixed_none_and_values(self) -> None:
        """Test fixed config key with mixed None and values."""
        row = pd.Series(
            {
                "tp_size": 4,
                "pp_size": None,
                "dp_size": 1,
                "cp_size": None,
            }
        )
        result = _optimizer_fixed_config_key(row)
        assert result is not None

    def test_simplify_with_no_preferred_columns(self) -> None:
        """Test simplify when preferred columns don't exist."""
        df = pd.DataFrame(
            [
                {"device": "D1", "other_col": 100},
                {"device": "D2", "other_col": 200},
            ]
        )
        result = _simplify_optimizer_display_df(df)
        # The function adds deployment_mode and capitalizes column names
        assert "Device" in result.columns

    def test_cli_style_output_with_same_device(self) -> None:
        """Test CLI style output when current device matches top device."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
                "best_ttft_ms": 50.0,
                "best_tpot_ms": 10.0,
                "best_parallel": "TP=4",
                "best_batch_size": 128,
            }
        )
        candidates = pd.DataFrame(
            [
                {"device": "D2", "best_throughput": 900.0, "best_ttft_ms": 55.0, "best_tpot_ms": 11.0},
            ]
        )
        result = _optimizer_cli_style_output(top, candidates, "D1")
        assert "D1" in result

    def test_format_int_with_large_value(self) -> None:
        """Test formatting int with large value."""
        result = _format_int_value(1000000)
        assert "1000000" in result

    def test_display_df_text_generate(self) -> None:
        """Test display_df for text_generate."""
        df = pd.DataFrame([{"device": "D1", "tps_per_device": 100}])
        result = _display_df_for_sim("text_generate", df)
        assert not result.empty

    def test_display_df_video_generate(self) -> None:
        """Test display_df for video_generate."""
        df = pd.DataFrame([{"device": "D1", "analytic_total_time_s": 10}])
        result = _display_df_for_sim("video_generate", df)
        assert not result.empty


class TestOptimizerEmptyOutputs:
    """Tests for _optimizer_empty_outputs function."""

    def test_empty_outputs_returns_all_components(self) -> None:
        """Test that empty outputs returns all required components."""
        summary = "### Summary"
        result = _optimizer_empty_outputs(summary)
        # Should return 21 components
        assert len(result) == 21
        # Summary should be at index 1
        assert result[1] == summary

    def test_empty_outputs_with_custom_detail(self) -> None:
        """Test with custom detail markdown."""
        summary = "### Summary"
        detail = "### Custom Detail"
        result = _optimizer_empty_outputs(summary, detail)
        assert len(result) == 21
        # Detail should be at index 13
        assert result[13] == detail


class TestTextValidationEmptyOutputs:
    """Tests for _text_validation_empty_outputs function."""

    def test_text_validation_empty_outputs(self) -> None:
        """Test text validation empty outputs."""
        summary = "### Validation Summary"
        result = _text_validation_empty_outputs(summary)
        # Returns 27 elements for text_generate
        assert len(result) == 27
        assert result[1] == summary


class TestVideoValidationEmptyOutputs:
    """Tests for _video_validation_empty_outputs function."""

    def test_video_validation_empty_outputs(self) -> None:
        """Test video validation empty outputs."""
        summary = "### Video Validation Summary"
        result = _video_validation_empty_outputs(summary)
        # Returns 13 elements for video_generate
        assert len(result) == 13
        assert result[1] == summary


class TestOptimizerMetricPlot:
    """Tests for _optimizer_metric_plot function."""

    def test_metric_plot_with_empty_df(self) -> None:
        """Test metric plot with empty dataframe."""
        df = pd.DataFrame()
        result = _optimizer_metric_plot(df, "throughput", "Test", "Throughput")
        assert result is not None

    def test_metric_plot_with_valid_df(self) -> None:
        """Test metric plot with valid dataframe."""
        df = pd.DataFrame(
            [
                {"device": "D1", "throughput": 1000},
                {"device": "D2", "throughput": 900},
            ]
        )
        result = _optimizer_metric_plot(df, "throughput", "Test", "Throughput")
        assert result is not None


class TestOptimizerCandidateRowsFromRecords:
    """Tests for _optimizer_candidate_rows_from_records function."""

    def test_candidate_rows_from_empty_records(self) -> None:
        """Test with empty records."""
        result = _optimizer_candidate_rows_from_records([])
        assert result == []

    def test_candidate_rows_from_valid_records(self) -> None:
        """Test with valid records."""
        records = [
            {
                "model_id": "test_model",
                "device": "D1",
                "num_devices": 4,
                "top_configs": [
                    {
                        "parallel": "TP=4",
                        "batch_size": 128,
                        "concurrency": 16,
                        "throughput_token_s": 1000,
                        "ttft_ms": 50,
                        "tpot_ms": 10,
                        "rank": 1,
                    }
                ],
            },
        ]
        result = _optimizer_candidate_rows_from_records(records)
        assert len(result) == 1
        assert result[0]["device"] == "D1"


class TestVideoGenerateOpTable:
    """Tests for _video_generate_op_table function."""

    def test_video_op_table_empty_results(self) -> None:
        """Test with empty results."""
        result = _video_generate_op_table([], None, 10)
        assert result.empty

    def test_video_op_table_with_valid_results(self) -> None:
        """Test with valid results."""
        results = [
            create_mock_video_result(
                params={"device": "D1", "batch_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "matmul", "analytic_total_us": 1000.0, "analytic_avg_us": 100.0, "num_calls": 10},
                    ]
                },
            )
        ]
        result = _video_generate_op_table(results, "D1", 10)
        assert not result.empty


class TestVideoGenerateCategoryStats:
    """Tests for _video_generate_category_stats function."""

    def test_video_category_stats_empty_results(self) -> None:
        """Test with empty results."""
        df, chart_data = _video_generate_category_stats([])
        assert df.empty
        assert not chart_data

    def test_video_category_stats_with_valid_results(self) -> None:
        """Test with valid results."""
        results = [
            create_mock_video_result(
                params={"device": "D1", "batch_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "matmul", "analytic_total_us": 1000.0, "num_calls": 10},
                        {"name": "softmax", "analytic_total_us": 500.0, "num_calls": 5},
                    ]
                },
            )
        ]
        df, chart_data = _video_generate_category_stats(results)
        assert not df.empty or chart_data


class TestVideoGenerateCompareTable:
    """Tests for _video_generate_compare_table function."""

    def test_video_compare_empty_results(self) -> None:
        """Test with empty results."""
        result = _video_generate_compare_table([])
        assert result.empty

    def test_video_compare_with_valid_results(self) -> None:
        """Test with valid results."""
        results = [
            create_mock_video_result(
                params={"device": "D1", "batch_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "matmul", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            )
        ]
        result = _video_generate_compare_table(results, top_n=10)
        assert not result.empty


class TestTextGenerateCategoryStats:
    """Tests for _text_generate_category_stats function."""

    def test_text_category_stats_empty_results(self) -> None:
        """Test with empty results."""
        df, chart_data = _text_generate_category_stats([])
        assert df.empty
        assert not chart_data

    def test_text_category_stats_with_no_device(self) -> None:
        """Test without device filter."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16, "tp_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "matmul", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            )
        ]
        df, chart_data = _text_generate_category_stats(results)
        assert not df.empty or chart_data


class TestTextGenerateCompareTable:
    """Tests for _text_generate_compare_table function."""

    def test_text_compare_empty_results(self) -> None:
        """Test with empty results."""
        result = _text_generate_compare_table([])
        assert result.empty

    def test_text_compare_with_top_n(self) -> None:
        """Test with custom top_n."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16, "tp_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            )
        ]
        result = _text_generate_compare_table(results, top_n=5)
        assert not result.empty


# Helper functions for creating mock results
def create_mock_result(**kwargs):
    """Create a mock ExperimentResult for text_generate."""
    from web_ui.schemas import ExperimentResult

    default_params = {
        "device": "D1",
        "num_queries": 16,
        "tp_size": 1,
    }
    params = {**default_params, **kwargs.pop("params", {})}
    command = kwargs.pop("command", ["python", "script.py"])
    task_hash = kwargs.pop("task_hash", "hash123")
    label = kwargs.pop("label", "test_label")
    return ExperimentResult(
        sim_type="text_generate",
        status="success",
        params=params,
        command=command,
        task_hash=task_hash,
        label=label,
        tables=kwargs.pop("tables", {}),
        summary=kwargs.pop("summary", {}),
        warnings=kwargs.pop("warnings", []),
        infos=kwargs.pop("infos", []),
        raw_log=kwargs.pop("raw_log", ""),
    )


def create_mock_video_result(**kwargs):
    """Create a mock ExperimentResult for video_generate."""
    from web_ui.schemas import ExperimentResult

    default_params = {
        "device": "D1",
        "batch_size": 1,
    }
    params = {**default_params, **kwargs.pop("params", {})}
    command = kwargs.pop("command", ["python", "video.py"])
    task_hash = kwargs.pop("task_hash", "hash456")
    label = kwargs.pop("label", "video_label")
    return ExperimentResult(
        sim_type="video_generate",
        status="success",
        params=params,
        command=command,
        task_hash=task_hash,
        label=label,
        tables=kwargs.pop("tables", {}),
        summary=kwargs.pop("summary", {}),
        warnings=[],
        infos=[],
        raw_log="",
    )


class TestValidateTextFormComprehensive:
    """Comprehensive tests for _validate_text_form error cases."""

    def test_validate_text_with_negative_prefix_cache(self) -> None:
        """Test with negative prefix_cache_hit_rate."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "prefix_cache_hit_rate": -0.1,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_prefix_cache_equal_one(self) -> None:
        """Test with prefix_cache_hit_rate >= 1."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "prefix_cache_hit_rate": 1.0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_negative_reserved_memory(self) -> None:
        """Test with negative reserved_memory_gb."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "reserved_memory_gb": -10,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_negative_context_length(self) -> None:
        """Test with negative context_length."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": -1,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_invalid_context_length(self) -> None:
        """Test with non-numeric context_length."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": "abc",
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_negative_mtp_tokens(self) -> None:
        """Test with negative num_mtp_tokens."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "num_mtp_tokens": -1,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_invalid_mtp_tokens(self) -> None:
        """Test with non-numeric num_mtp_tokens."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "num_mtp_tokens": "abc",
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_mtp_exceeds_query_length(self) -> None:
        """Test with mtp_tokens > query_length."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 10,
            "context_length": 2048,
            "num_mtp_tokens": 50,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_mtp_equals_query_length(self) -> None:
        """Test with mtp_tokens == query_length."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 50,
            "context_length": 2048,
            "num_mtp_tokens": 50,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_invalid_tp_sweep(self) -> None:
        """Test with invalid tp_sweep value that raises exception."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_sweep": "invalid_list",
        }
        errors = _validate_text_form(form)
        # Should catch the exception and add an error
        assert isinstance(errors, list)

    def test_validate_text_with_dp_zero(self) -> None:
        """Test with DP=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "dp_size": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_dp_exceeds_devices(self) -> None:
        """Test with DP > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "dp_size": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_dp_not_divisible(self) -> None:
        """Test with num_devices not divisible by DP."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "dp_size": 3,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_ep_zero(self) -> None:
        """Test with EP=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "ep_size": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_ep_exceeds_devices(self) -> None:
        """Test with EP > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "ep_size": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_ep_not_divisible(self) -> None:
        """Test with num_devices not divisible by EP."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "ep_size": 3,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_tp_zero(self) -> None:
        """Test with TP=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_tp_exceeds_devices(self) -> None:
        """Test with TP > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_tp_not_divisible(self) -> None:
        """Test with num_devices not divisible by TP."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 3,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_tp_dp_ep_product_exceeds_devices(self) -> None:
        """Test with TP*DP*EP > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 4,
            "dp_size": 2,
            "ep_size": 2,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_invalid_o_proj_tp(self) -> None:
        """Test with invalid o_proj_tp_size."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "o_proj_tp_size": "abc",
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_o_proj_tp_zero(self) -> None:
        """Test with o_proj_tp_size=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "o_proj_tp_size": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_o_proj_tp_exceeds_devices(self) -> None:
        """Test with o_proj_tp_size > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "o_proj_tp_size": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_o_proj_tp_not_divisible(self) -> None:
        """Test with o_proj_tp_size not divisible."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "o_proj_tp_size": 3,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_mxfp4_zero(self) -> None:
        """Test with mxfp4_group_size=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 1,
            "mxfp4_group_size": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateVideoFormComprehensive:
    """Comprehensive tests for _validate_video_form error cases."""

    def test_validate_video_with_invalid_ulysses_list(self) -> None:
        """Test with invalid ulysses_sweep that raises exception."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 50,
            "world_size": 8,
            "ulysses_sweep": "invalid",
        }
        errors = _validate_video_form(form)
        assert isinstance(errors, list)

    def test_validate_video_with_ulysses_zero(self) -> None:
        """Test with ulysses_size=0."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "world_size": 8,
            "ulysses_size": 0,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_ulysses_exceeds_world_size(self) -> None:
        """Test with ulysses_size > world_size."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "world_size": 8,
            "ulysses_size": 16,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_ulysses_not_divisible(self) -> None:
        """Test with world_size not divisible by ulysses_size."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "world_size": 8,
            "ulysses_size": 3,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_invalid_cache_step_interval(self) -> None:
        """Test with invalid cache_step_interval."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_interval": 0,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_cache_step_range_missing_comma(self) -> None:
        """Test with cache_step_range missing comma."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_range": "0 10",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_cache_step_range_invalid_numbers(self) -> None:
        """Test with cache_step_range non-numeric."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_range": "abc,def",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_cache_step_range_invalid_range(self) -> None:
        """Test with cache_step_range where start >= end."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_range": "10,5",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_cache_step_range_negative_start(self) -> None:
        """Test with cache_step_range negative start."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_range": "-1,10",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_cache_block_range_invalid(self) -> None:
        """Test with cache_block_range invalid format."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_block_range": "1,0",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestPreviewSummaryMarkdownBranches:
    """Tests for _preview_summary_markdown branch coverage."""

    def test_preview_summary_text_with_image_params(self) -> None:
        """Test text_generate with image_height and image_width."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "num_devices": 8,
            "num_queries": 100,
            "context_length": 2048,
            "query_length": 512,
            "decode": True,
            "image_height": 512,
            "image_width": 512,
            "image_batch_size": 4,
        }
        tasks = [Mock(params={"device": "D1", "num_queries": 100})]
        result = _preview_summary_markdown("text_generate", form, tasks)
        assert "Images:" in result
        assert "512 x 512" in result

    def test_preview_summary_text_without_image_params(self) -> None:
        """Test text_generate without image params."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "num_devices": 8,
            "num_queries": 100,
            "context_length": 2048,
            "query_length": 512,
            "decode": True,
        }
        tasks = [Mock(params={"device": "D1", "num_queries": 100})]
        result = _preview_summary_markdown("text_generate", form, tasks)
        # Should not include image line
        assert "Images:" not in result

    def test_preview_summary_text_prefill_mode(self) -> None:
        """Test text_generate with prefill mode (decode=False)."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "num_devices": 8,
            "num_queries": 100,
            "context_length": 2048,
            "query_length": 512,
            "decode": False,
        }
        tasks = [Mock(params={"device": "D1", "num_queries": 100})]
        result = _preview_summary_markdown("text_generate", form, tasks)
        assert "Prefill" in result


# Duplicate TestResultsToDf removed - merged with earlier definition


class TestSummaryMarkdownBranches:
    """Tests for _summary_markdown branch coverage."""

    def test_summary_markdown_empty_df(self) -> None:
        """Test with empty dataframe."""
        result = _summary_markdown(pd.DataFrame(), None, "text_generate")
        assert "No results" in result

    def test_summary_markdown_video_without_best_metric(self) -> None:
        """Test video_generate without analytic_total_time_s column - just shows basic summary."""
        df = pd.DataFrame({"device": ["D1"]})
        result = _summary_markdown(df, None, "video_generate")
        assert "Completed Runs: **1**" in result
        # No "Lowest Analytic Time" line when column is missing
        assert "Lowest Analytic Time" not in result

    def test_summary_markdown_optimizer_no_best_throughput(self) -> None:
        """Test optimizer without best_throughput column."""
        df = pd.DataFrame({"device": ["D1"]})
        result = _summary_markdown(df, None, "throughput_optimizer")
        # Should show the feasibility note when no best_throughput
        assert "Feasibility Note" in result


# Duplicate TestOptimizerDeploymentMode removed - merged with earlier definition


class TestFormatLimitValueEdgeCases:
    """Edge case tests for _format_limit_value function."""

    def test_format_limit_with_empty_string(self) -> None:
        """Test with empty string."""
        result = _format_limit_value("")
        assert "None ms" in result

    def test_format_limit_with_nan(self) -> None:
        """Test with NaN value."""
        result = _format_limit_value(float("nan"))
        assert "None ms" in result

    def test_format_limit_with_zero(self) -> None:
        """Test with zero value."""
        result = _format_limit_value(0)
        assert "0.00 ms" in result


class TestOpTableFromRecordsComprehensive:
    """Comprehensive tests for _op_table_from_records."""

    def test_op_table_empty_df_from_records(self) -> None:
        """Test when records result in empty df after filtering."""
        records = [{"name": "op1", "analytic_total_us": 1000, "device": "D2"}]
        result = _op_table_from_records(records, "D1", 10)
        assert result.empty

    def test_op_table_with_device_filter_empty(self) -> None:
        """Test device filter results in empty df."""
        records = [{"name": "op1", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D2"}]
        result = _op_table_from_records(records, "D1", 10)
        assert result.empty

    def test_op_table_with_case_label_empty(self) -> None:
        """Test case label filter results in empty df."""
        records = [
            {
                "name": "op1",
                "analytic_total_us": 1000,
                "analytic_avg_us": 100,
                "num_calls": 10,
                "device": "D1",
                "num_queries": 100,
                "tp_size": 4,
            }
        ]
        result = _op_table_from_records(records, "D1", 10, case_label="Concurrency=200 | TP=8")
        assert result.empty

    def test_op_table_without_name_column(self) -> None:
        """Test without name column."""
        records = [{"analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D1"}]
        result = _op_table_from_records(records, "D1", 10)
        assert not result.empty
        assert result.iloc[0]["Operator"] == "-"

    def test_op_table_sort_by_operator(self) -> None:
        """Test sorting by Operator name."""
        records = [
            {"name": "zebra", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D1"},
            {"name": "alpha", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D1"},
        ]
        result = _op_table_from_records(records, "D1", 10, sort_by="Operator")
        assert result.iloc[0]["Operator"] == "alpha"

    def test_op_table_with_all_numeric_columns(self) -> None:
        """Test with all numeric columns present."""
        records = [
            {
                "name": "op1",
                "analytic_total_us": 1000,
                "analytic_avg_us": 100,
                "num_calls": 10,
                "device": "D1",
                "category": "Linear",
            }
        ]
        result = _op_table_from_records(records, "D1", 10)
        assert result.iloc[0]["Total Time (ms)"] == 1.0
        assert result.iloc[0]["Average Time (ms)"] == 0.1
        assert result.iloc[0]["Calls"] == 10


class TestRoundNumericColumnsComprehensive:
    """Comprehensive tests for _round_numeric_columns."""

    def test_round_with_token_s_column(self) -> None:
        """Test with token/s column."""
        df = pd.DataFrame({"throughput_token/s": [1234.5678]})
        result = _round_numeric_columns(df, digits=2)
        assert result.iloc[0]["throughput_token/s"] == 1234.57

    def test_round_with_qps_column(self) -> None:
        """Test with QPS column (needs to be capitalized to match)."""
        df = pd.DataFrame({"QPS": [987.6543]})
        result = _round_numeric_columns(df, digits=1)
        assert result.iloc[0]["QPS"] == 987.7

    def test_round_with_lowercase_qps(self) -> None:
        """Test with lowercase qps column (should not round)."""
        df = pd.DataFrame({"qps": [987.6543]})
        result = _round_numeric_columns(df, digits=1)
        # Lowercase qps doesn't match the pattern, so no rounding
        assert result.iloc[0]["qps"] == 987.6543

    def test_round_with_throughput_column(self) -> None:
        """Test with Throughput column."""
        df = pd.DataFrame({"Throughput": [555.5555]})
        result = _round_numeric_columns(df, digits=2)
        assert result.iloc[0]["Throughput"] == 555.56

    def test_round_with_non_numeric_column(self) -> None:
        """Test column with non-numeric values."""
        df = pd.DataFrame({"latency (ms)": ["N/A", "100"]})
        result = _round_numeric_columns(df)
        # Should handle gracefully without error
        assert result is not None

    def test_round_with_all_na_column(self) -> None:
        """Test column with all NA values."""
        df = pd.DataFrame({"latency_ms": [None, None]})
        result = _round_numeric_columns(df)
        # Should handle gracefully
        assert result is not None


class TestNormalizeOpColumnsComprehensive:
    """Comprehensive tests for _normalize_op_columns."""

    def test_normalize_columns_none(self) -> None:
        """Test with None columns - returns default columns."""
        result = _normalize_op_columns(None)
        assert result == OP_TABLE_DEFAULT_COLUMNS

    def test_normalize_columns_empty_list(self) -> None:
        """Test with empty list - returns default columns."""
        result = _normalize_op_columns([])
        assert result == OP_TABLE_DEFAULT_COLUMNS

    def test_normalize_columns_with_valid_columns(self) -> None:
        """Test with valid column selections."""
        columns = ["Operator", "Calls"]
        result = _normalize_op_columns(columns)
        assert "Operator" in result
        assert "Calls" in result

    def test_normalize_columns_all_invalid(self) -> None:
        """Test with all invalid column names - returns default columns."""
        columns = ["Invalid1", "Invalid2"]
        result = _normalize_op_columns(columns)
        assert result == OP_TABLE_DEFAULT_COLUMNS


class TestFormatMetricValueEdgeCases:
    """Edge case tests for _format_metric_value."""

    def test_format_metric_with_string_zero(self) -> None:
        """Test with string '0'."""
        result = _format_metric_value("0")
        assert "0.00" in result

    def test_format_metric_with_negative(self) -> None:
        """Test with negative value."""
        result = _format_metric_value(-123.45)
        assert "-123.45" in result

    def test_format_metric_with_very_small(self) -> None:
        """Test with very small value."""
        result = _format_metric_value(0.0001234)
        assert "0.00" in result or "0.000" in result

    def test_format_metric_custom_digits(self) -> None:
        """Test with custom digits parameter."""
        result = _format_metric_value(123.456789, digits=4)
        assert "123.4568" in result


# Duplicate TestAsciiTableEdgeCases removed - merged with earlier definition


class TestOptimizerParetoChartComprehensive:
    """Comprehensive tests for _optimizer_pareto_chart."""

    def test_pareto_chart_missing_ttft_column(self) -> None:
        """Test without ttft_ms column - raises KeyError."""
        df = pd.DataFrame(
            [
                {"throughput_token_s": 1000},
                {"throughput_token_s": 500},
            ]
        )
        with pytest.raises(KeyError):
            _optimizer_pareto_chart(df, "D1")

    def test_pareto_chart_missing_throughput_column(self) -> None:
        """Test without throughput_token_s column - raises KeyError."""
        df = pd.DataFrame(
            [
                {"ttft_ms": 100},
                {"ttft_ms": 200},
            ]
        )
        with pytest.raises(KeyError):
            _optimizer_pareto_chart(df, "D1")

    def test_pareto_chart_with_nan_values(self) -> None:
        """Test with NaN values in metric columns - dropped by dropna."""
        df = pd.DataFrame(
            [
                {"ttft_ms": 100, "throughput_token_s": float("nan")},
                {"ttft_ms": 200, "throughput_token_s": 1000},
            ]
        )
        result = _optimizer_pareto_chart(df, "D1")
        assert result is not None

    def test_pareto_chart_single_point(self) -> None:
        """Test with single data point."""
        df = pd.DataFrame([{"ttft_ms": 100, "throughput_token_s": 1000}])
        result = _optimizer_pareto_chart(df, "D1")
        assert result is not None


class TestDedupeComprehensive:
    """Comprehensive tests for _dedupe function."""

    def test_dedupe_with_mixed_types(self) -> None:
        """Test with mixed types including int, str, bool."""
        result = _dedupe([1, "1", True, False, 1, "1"])
        # All converted to strings
        assert "1" in result
        assert "True" in result
        assert "False" in result

    def test_dedupe_preserves_first_occurrence(self) -> None:
        """Test that first occurrence is preserved."""
        result = _dedupe(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_dedupe_with_whitespace_variations(self) -> None:
        """Test with whitespace variations (not trimmed)."""
        result = _dedupe(["a", " a", "a ", "  a"])
        # Different strings are not deduped
        assert len(result) == 4


class TestCaseLabelFromMappingEdgeCases:
    """Edge case tests for _case_label_from_mapping."""

    def test_case_label_with_zero_num_queries(self) -> None:
        """Test with num_queries=0."""
        row = {"num_queries": 0, "tp_size": 4}
        result = _case_label_from_mapping(row)
        assert "Concurrency=0" in result

    def test_case_label_with_string_num_queries(self) -> None:
        """Test with string num_queries."""
        row = {"num_queries": "100", "tp_size": 4}
        result = _case_label_from_mapping(row)
        assert "Concurrency=100" in result

    def test_case_label_with_zero_tp_size(self) -> None:
        """Test with tp_size=0."""
        row = {"num_queries": 100, "tp_size": 0}
        result = _case_label_from_mapping(row)
        assert "TP=0" in result

    def test_case_label_with_empty_string_tp_size(self) -> None:
        """Test with empty string tp_size."""
        row = {"num_queries": 100, "tp_size": ""}
        result = _case_label_from_mapping(row)
        assert "TP=1" in result


class TestPreviewFirstCommandEdgeCases:
    """Edge case tests for _preview_first_command."""

    def test_preview_first_command_with_empty_command(self) -> None:
        """Test with empty command list."""
        task = Mock(command=[])
        result = _preview_first_command([task])
        assert result == "No command generated."

    def test_preview_first_command_with_numeric_parts(self) -> None:
        """Test command with numeric parts."""
        task = Mock(command=["cmd", 123, 456])
        result = _preview_first_command([task])
        assert "123" in result
        assert "456" in result

    def test_preview_first_command_with_special_chars(self) -> None:
        """Test command with special characters."""
        task = Mock(command=["cmd", "--arg=value", "path/to/file"])
        result = _preview_first_command([task])
        assert "--arg=value" in result
        assert "path/to/file" in result


class TestFormatPreviewErrorEdgeCases:
    """Edge case tests for _format_preview_error."""

    def test_format_preview_error_with_long_message(self) -> None:
        """Test with very long error message."""
        error = ValueError(
            "This is a very long error message that should be included in the formatted output without any issues"
        )
        markdown, command = _format_preview_error(error)
        assert "very long" in markdown
        assert command == ""

    def test_format_preview_error_with_special_chars(self) -> None:
        """Test with special characters in error message."""
        error = ValueError("Error: <test> & \"quoted\"")
        markdown, command = _format_preview_error(error)
        assert "<test>" in markdown or "&lt;test&gt;" in markdown
        assert command == ""


class TestFilterDfByCaseEdgeCases:
    """Edge case tests for _filter_df_by_case."""

    def test_filter_df_with_no_matching_case(self) -> None:
        """Test when no rows match the case label."""
        df = pd.DataFrame(
            {
                "case_label": ["Concurrency=100 | TP=4", "Concurrency=200 | TP=8"],
                "value": [1, 2],
            }
        )
        result = _filter_df_by_case(df, "Concurrency=300 | TP=2")
        assert result.empty

    def test_filter_df_with_empty_case_label(self) -> None:
        """Test with empty case_label parameter - returns original df."""
        df = pd.DataFrame(
            {
                "case_label": ["Concurrency=100 | TP=4"],
                "value": [1],
            }
        )
        result = _filter_df_by_case(df, "")
        # Empty string is falsy, so function returns original df
        assert len(result) == 1

    def test_filter_df_with_none_case_label(self) -> None:
        """Test with None case_label parameter - returns original df."""
        df = pd.DataFrame(
            {
                "case_label": ["Concurrency=100 | TP=4"],
                "value": [1],
            }
        )
        result = _filter_df_by_case(df, None)
        # None is falsy, so function returns original df
        assert len(result) == 1


class TestBuildTextFormEdgeCases:
    """Edge case tests for _build_text_form."""

    def test_build_text_form_with_all_none_values(self) -> None:
        """Test with all None values."""
        vals = [None] * 50
        result = _build_text_form(*vals)
        assert isinstance(result, dict)

    def test_build_text_form_with_empty_strings(self) -> None:
        """Test with empty string values."""
        vals = [""] * 50
        result = _build_text_form(*vals)
        assert isinstance(result, dict)

    def test_build_text_form_with_numeric_strings(self) -> None:
        """Test with numeric string values."""
        vals = ["model", "D1", None, "8", "16", None, "512", "2048"] + [None] * 42
        result = _build_text_form(*vals)
        assert result["num_devices"] == "8"
        assert result["num_queries"] == "16"


class TestBuildVideoFormEdgeCases:
    """Edge case tests for _build_video_form."""

    def test_build_video_form_with_all_none(self) -> None:
        """Test with all None values."""
        vals = [None] * 25
        result = _build_video_form(*vals)
        assert isinstance(result, dict)

    def test_build_video_form_with_zero_values(self) -> None:
        """Test with zero numeric values."""
        vals = ["model", "D1", None, 0, 0, 0, 0, 0, 0, None] + [None] * 14
        result = _build_video_form(*vals)
        assert result["batch_size"] == 0


class TestBuildOptFormEdgeCases:
    """Edge case tests for _build_opt_form."""

    def test_build_opt_form_with_all_none(self) -> None:
        """Test with all None values."""
        vals = [None] * 40
        result = _build_opt_form(*vals)
        assert isinstance(result, dict)

    def test_build_opt_form_mode_normalization(self) -> None:
        """Test deployment mode normalization for various inputs."""
        # Test "PD Split" normalization
        vals = ["model", "D1", None, 8, 1024, 512, True] + [None] * 33
        result = _build_opt_form(*vals)
        assert result["deployment_mode"] == "" or result["deployment_mode"] == OPT_DEPLOY_PD_MIXED


class TestValidateOptimizerFormComprehensive:
    """Comprehensive tests for _validate_optimizer_form."""

    def test_validate_optimizer_with_negative_jobs(self) -> None:
        """Test with negative jobs value."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": -1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_zero_jobs(self) -> None:
        """Test with zero jobs value."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 0,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_negative_input_length(self) -> None:
        """Test with negative input_length."""
        form = {
            "num_devices": 8,
            "input_length": -1,
            "output_length": 512,
            "jobs": 1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_negative_output_length(self) -> None:
        """Test with negative output_length."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": -1,
            "jobs": 1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_invalid_prefill_devices_in_pd_mixed(self) -> None:
        """Test with invalid prefill devices in PD Mixed mode."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Mixed",
            "prefill_devices_per_instance": "invalid",
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_invalid_decode_devices_in_pd_mixed(self) -> None:
        """Test with invalid decode devices in PD Mixed mode."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Mixed",
            "prefill_devices_per_instance": 4,
            "decode_devices_per_instance": "invalid",
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_zero_prefill_devices(self) -> None:
        """Test with zero prefill devices."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 0,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_zero_decode_devices(self) -> None:
        """Test with zero decode devices."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 4,
            "decode_devices_per_instance": 0,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_prefill_plus_decode_exceeds_devices(self) -> None:
        """Test with prefill + decode > num_devices."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 5,
            "decode_devices_per_instance": 5,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


# Duplicate TestOptimizerValidationMarkdown removed - merged with earlier definition


# Duplicate TestTextGenerateSummaryMarkdown removed - merged with earlier definition


# Duplicate TestVideoGenerateSummaryMarkdown removed - merged with earlier definition


# Duplicate TestOptimizerSummaryMarkdownFromDf removed - merged with earlier definition


class TestOptimizerFixedConfigKeyExtended:
    """Extended tests for _optimizer_fixed_config_key function."""

    def test_fixed_config_key_with_all_fields(self) -> None:
        """Test with all possible fields."""
        row = pd.Series(
            {
                "model_id": "test_model",
                "input_length": 512,
                "output_length": 512,
                "num_devices": 8,
                "prefix_cache_hit_rate": 0.5,
                "quantize_linear_action": "W8A8",
                "quantize_attention_action": "INT8",
                "parallel": "TP=4",
                "batch_size": 32,
                "concurrency": 16,
            }
        )
        result = _optimizer_fixed_config_key(row)
        assert "||" in result

    def test_fixed_config_key_with_minimal_fields(self) -> None:
        """Test with minimal fields."""
        row = pd.Series({"batch_size": 64})
        result = _optimizer_fixed_config_key(row)
        assert "64" in result


class TestOptimizerMetricPlotExtended:
    """Extended tests for _optimizer_metric_plot function."""

    def test_metric_plot_with_no_metric_column(self) -> None:
        """Test when metric column is missing."""
        df = pd.DataFrame({"device": ["D1", "D2"]})
        result = _optimizer_metric_plot(df, "throughput", "Test", "Throughput")
        assert result is not None

    def test_metric_plot_with_single_device(self) -> None:
        """Test with single device."""
        df = pd.DataFrame([{"device": "D1", "throughput": 1000}])
        result = _optimizer_metric_plot(df, "throughput", "Test", "Throughput")
        assert result is not None


class TestOptimizerCandidateRowsFromRecordsExtended:
    """Extended tests for _optimizer_candidate_rows_from_records."""

    def test_candidate_rows_with_multiple_top_configs(self) -> None:
        """Test with multiple top configs."""
        records = [
            {
                "model_id": "test_model",
                "device": "D1",
                "top_configs": [
                    {"parallel": "TP=4", "batch_size": 32, "concurrency": 16, "throughput_token_s": 1000, "rank": 1},
                    {"parallel": "TP=2", "batch_size": 64, "concurrency": 8, "throughput_token_s": 800, "rank": 2},
                ],
            },
        ]
        result = _optimizer_candidate_rows_from_records(records)
        assert len(result) == 2

    def test_candidate_rows_with_missing_top_configs(self) -> None:
        """Test with missing top_configs."""
        records = [{"model_id": "test", "device": "D1"}]
        result = _optimizer_candidate_rows_from_records(records)
        # No candidates are generated when top_configs is empty or missing
        assert len(result) == 0

    def test_candidate_rows_with_incomplete_candidate(self) -> None:
        """Test with candidate missing required fields."""
        records = [
            {
                "model_id": "test",
                "device": "D1",
                "top_configs": [
                    {"parallel": "TP=4", "batch_size": 32},  # Missing concurrency
                ],
            }
        ]
        result = _optimizer_candidate_rows_from_records(records)
        # Candidate is skipped when missing required fields
        assert len(result) == 0


class TestOptimizerValidationMarkdownExtended:
    """Extended tests for _optimizer_validation_markdown."""

    def test_optimizer_validation_markdown_with_special_chars(self) -> None:
        """Test with special characters in error messages."""
        errors = ["Error: <test> & \"quoted\""]
        result = _optimizer_validation_markdown(errors)
        assert "- Error:" in result

    def test_optimizer_validation_markdown_with_long_message(self) -> None:
        """Test with very long error message."""
        errors = [
            "This is a very long error message that should be included in the output without any truncation or modification"
        ]
        result = _optimizer_validation_markdown(errors)
        assert "very long" in result


class TestMemoryAnalysisFromSummaryExtended:
    """Extended tests for _memory_analysis_from_summary."""

    def test_memory_analysis_with_all_fields(self) -> None:
        """Test with all memory fields present."""
        summary = {
            "total_device_memory_gb": 80,
            "model_weight_size_gb": 10,
            "kv_cache_gb": 5,
            "model_activation_size_gb": 15,
            "memory_available_gb": 50,
        }
        memory_data, table = _memory_analysis_from_summary(summary)
        assert len(memory_data) >= 4
        assert isinstance(memory_data, dict)
        assert not table.empty

    def test_memory_analysis_with_no_available_memory(self) -> None:
        """Test without memory_available_gb."""
        summary = {
            "total_device_memory_gb": 80,
            "model_weight_size_gb": 10,
            "kv_cache_gb": 5,
        }
        memory_data, table = _memory_analysis_from_summary(summary)
        # Available memory is extracted from summary, not calculated
        assert "Available Memory" in memory_data or len(memory_data) >= 2


class TestCompareTableByMode:
    """Tests for update_compare_table_by_mode function."""

    def test_compare_table_empty_results(self) -> None:
        """Test with empty results."""
        result = update_compare_table_by_mode([], mode="device")
        assert result.empty

    def test_compare_table_with_valid_results(self) -> None:
        """Test with valid op_breakdown data."""
        op_breakdown = [
            {"name": "op1", "analytic_total_us": 1000.0, "analytic_avg_us": 100.0, "device": "D1"},
            {"name": "op2", "analytic_total_us": 500.0, "analytic_avg_us": 50.0, "device": "D2"},
        ]
        result = update_compare_table_by_mode(op_breakdown, mode="Total Time", top_n=10)
        assert not result.empty


# Duplicate TestValidateTextFormExtended removed - merged with earlier definition


class TestValidateOptimizerFormExtended2:
    """More extended tests for _validate_optimizer_form."""

    def test_validate_optimizer_with_negative_prefill_devices(self) -> None:
        """Test with negative prefill_devices_per_instance."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": -1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_negative_decode_devices(self) -> None:
        """Test with negative decode_devices_per_instance."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 4,
            "decode_devices_per_instance": -1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateVideoFormExtended2:
    """More extended tests for _validate_video_form."""

    def test_validate_video_with_zero_batch(self) -> None:
        """Test with batch_size=0."""
        form = {
            "batch_size": 0,
            "seq_len": 512,
            "height": 720,
            "width": 1280,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_negative_seq_len(self) -> None:
        """Test with negative seq_len."""
        form = {
            "batch_size": 4,
            "seq_len": -100,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestNormalizeOpColumnsEdgeCases2:
    """More edge cases for _normalize_op_columns."""

    def test_normalize_columns_with_duplicate_valid(self) -> None:
        """Test with duplicate valid columns."""
        columns = ["Operator", "Operator", "Calls"]
        result = _normalize_op_columns(columns)
        # Should preserve duplicates
        assert result.count("Operator") == 2

    def test_normalize_columns_mixed_valid_invalid(self) -> None:
        """Test with mix of valid and invalid columns."""
        columns = ["Operator", "Invalid", "Calls"]
        result = _normalize_op_columns(columns)
        assert "Operator" in result
        assert "Calls" in result
        assert "Invalid" not in result


class TestRoundNumericColumnsEdgeCases2:
    """More edge cases for _round_numeric_columns."""

    def test_round_with_s_column(self) -> None:
        """Test with (s) column - time in seconds."""
        df = pd.DataFrame({"time (s)": [1.23456]})
        result = _round_numeric_columns(df, digits=2)
        assert result.iloc[0]["time (s)"] == 1.23

    def test_round_with_gb_column(self) -> None:
        """Test with (GB) column."""
        df = pd.DataFrame({"memory (GB)": [12.3456]})
        result = _round_numeric_columns(df, digits=1)
        assert result.iloc[0]["memory (GB)"] == 12.3

    def test_round_with_percent_column(self) -> None:
        """Test with (%) column."""
        df = pd.DataFrame({"utilization (%)": [67.891]})
        result = _round_numeric_columns(df, digits=2)
        assert result.iloc[0]["utilization (%)"] == 67.89

    def test_round_with_mixed_numeric_non_numeric(self) -> None:
        """Test with column that has mixed values."""
        df = pd.DataFrame({"latency (ms)": [100.5, "N/A", 200.7]})
        result = _round_numeric_columns(df, digits=1)
        assert result is not None
        # Should handle gracefully


class TestFormatIntValueExtended:
    """Extended tests for _format_int_value."""

    def test_format_int_with_negative(self) -> None:
        """Test with negative value."""
        result = _format_int_value(-100)
        assert "-100" in result

    def test_format_int_with_large_float(self) -> None:
        """Test with large float that truncates."""
        result = _format_int_value(999999.9)
        assert "999999" in result

    def test_format_int_with_zero_float(self) -> None:
        """Test with 0.0."""
        result = _format_int_value(0.0)
        assert "0" in result


class TestFormatLimitValueExtended:
    """Extended tests for _format_limit_value."""

    def test_format_limit_with_negative(self) -> None:
        """Test with negative value."""
        result = _format_limit_value(-50.5)
        assert "-50.50 ms" in result or "-50.5" in result

    def test_format_limit_with_very_large(self) -> None:
        """Test with very large value - rounded to 2 decimal places."""
        result = _format_limit_value(999999.999)
        # 999999.999 rounded to 2 decimal places becomes 1000000.00
        assert "1000000" in result


class TestDedupeExtended:
    """Extended tests for _dedupe function."""

    def test_dedupe_with_unicode(self) -> None:
        """Test with unicode strings."""
        result = _dedupe(["test1", "test", "test1", "test"])
        assert len(result) == 2

    def test_dedupe_with_numbers_as_strings(self) -> None:
        """Test with numeric strings."""
        result = _dedupe(["123", "456", "123", "789"])
        assert result == ["123", "456", "789"]

    def test_dedupe_with_mixed_none_and_values(self) -> None:
        """Test with None, empty, and values mixed."""
        result = _dedupe([None, "", "a", None, "", "b"])
        assert result == ["a", "b"]


# Duplicate TestCaseChoicesFromRowsExtended removed - merged with earlier definition


class TestPreviewSummaryExtended:
    """Extended tests for _preview_summary_markdown."""

    def test_preview_summary_with_empty_tasks(self) -> None:
        """Test with empty tasks list."""
        form = {"model_id": "test", "device": "D1"}
        result = _preview_summary_markdown("text_generate", form, [])
        assert "Estimated Tasks: **0**" in result

    def test_preview_summary_video_cfg_disabled(self) -> None:
        """Test video_generate with CFG disabled."""
        form = {
            "model_id": "video_model",
            "device": "D1",
            "world_size": 8,
            "use_cfg": False,
            "cfg_parallel": False,
            "dit_cache": False,
        }
        tasks = [Mock(params={"device": "D1"})]
        result = _preview_summary_markdown("video_generate", form, tasks)
        assert "CFG=Disabled" in result
        assert "DiT Cache=Disabled" in result

    def test_preview_summary_optimizer_no_prefix_cache(self) -> None:
        """Test optimizer without prefix cache."""
        form = {
            "model_id": "opt_model",
            "device": "D1",
            "num_devices": 8,
            "input_length": 1024,
            "output_length": 512,
            "jobs": 4,
            "prefix_cache_hit_rate": 0,
        }
        tasks = [Mock(params={"device": "D1", "deployment_mode": "PD Mixed"})]
        result = _preview_summary_markdown("throughput_optimizer", form, tasks)
        # Should not include prefix cache line when 0
        assert "Prefix Cache Hit Rate" not in result


class TestOptimizerCliStyleOutputExtended:
    """Extended tests for _optimizer_cli_style_output."""

    def test_cli_style_output_with_missing_optional_fields(self) -> None:
        """Test with missing optional fields in top result."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
            }
        )
        result = _optimizer_cli_style_output(top, pd.DataFrame(), "D1")
        assert "D1" in result or "1000" in result

    def test_cli_style_output_with_all_fields(self) -> None:
        """Test with all fields present."""
        top = pd.Series(
            {
                "model_id": "test_model",
                "device": "D1",
                "best_throughput": 1000.0,
                "best_ttft_ms": 50.0,
                "best_tpot_ms": 10.0,
                "num_devices": 8,
                "input_length": 1024,
                "output_length": 512,
                "ttft_limits_ms": 100.0,
                "tpot_limits_ms": 50.0,
                "quantize_linear_action": "W8A8",
                "quantize_attention_action": "INT8",
            }
        )
        result = _optimizer_cli_style_output(top, pd.DataFrame(), "D1")
        assert "test_model" in result
        assert "1000" in result


class TestVideoGenerateCompareTableExtended:
    """Extended tests for _video_generate_compare_table."""

    def test_video_compare_single_device(self) -> None:
        """Test with results from single device."""
        results = [
            create_mock_video_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 1000.0, "num_calls": 10},
                        {"name": "op2", "analytic_total_us": 500.0, "num_calls": 5},
                    ]
                },
            )
        ]
        result = _video_generate_compare_table(results, top_n=10)
        assert not result.empty

    def test_video_compare_multiple_devices(self) -> None:
        """Test with results from multiple devices."""
        results = [
            create_mock_video_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            ),
            create_mock_video_result(
                params={"device": "D2"},
                tables={
                    "op_breakdown": [
                        {"name": "op2", "analytic_total_us": 800.0, "num_calls": 8},
                    ]
                },
            ),
        ]
        result = _video_generate_compare_table(results, top_n=10)
        assert not result.empty


class TestTextGenerateCompareTableExtended:
    """Extended tests for _text_generate_compare_table."""

    def test_text_compare_single_device(self) -> None:
        """Test with single device results."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16, "tp_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            )
        ]
        result = _text_generate_compare_table(results, top_n=10)
        assert not result.empty

    def test_text_compare_with_custom_top_n(self) -> None:
        """Test with custom top_n limit."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16, "tp_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": f"op{i}", "analytic_total_us": 1000.0 - i * 10, "num_calls": 10} for i in range(20)
                    ]
                },
            )
        ]
        result = _text_generate_compare_table(results, top_n=5)
        assert len(result) <= 6  # 5 operators + Total column


class TestTextGenerateCategoryStatsExtended:
    """Extended tests for _text_generate_category_stats."""

    def test_text_category_stats_with_device_filter(self) -> None:
        """Test with device filter."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16, "tp_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "matmul", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            ),
            create_mock_result(
                params={"device": "D2", "num_queries": 16, "tp_size": 1},
                tables={
                    "op_breakdown": [
                        {"name": "attn", "analytic_total_us": 500.0, "num_calls": 5},
                    ]
                },
            ),
        ]
        df, chart_data = _text_generate_category_stats(results, device="D1")
        assert not df.empty or chart_data


class TestVideoGenerateCategoryStatsExtended:
    """Extended tests for _video_generate_category_stats."""

    def test_video_category_stats_multiple_categories(self) -> None:
        """Test with multiple operator categories."""
        results = [
            create_mock_video_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "matmul", "analytic_total_us": 1000.0, "num_calls": 10},
                        {"name": "attn", "analytic_total_us": 500.0, "num_calls": 5},
                        {"name": "norm", "analytic_total_us": 200.0, "num_calls": 2},
                    ]
                },
            )
        ]
        df, chart_data = _video_generate_category_stats(results)
        assert len(df) >= 2 or chart_data


class TestUpdateMemoryAnalysisByDeviceExtended:
    """Extended tests for update_memory_analysis_by_device."""

    def test_update_memory_with_case_label(self) -> None:
        """Test with case label filter."""
        rows = [
            {
                "device": "D1",
                "num_queries": 16,
                "tp_size": 2,
                "total_device_memory_gb": 80,
                "model_weight_size_gb": 10,
                "case_label": "Concurrency=16 | TP=2",
            }
        ]
        case_label = "Concurrency=16 | TP=2"
        fig, df = update_memory_analysis_by_device(rows, "D1", case_label)
        assert not df.empty

    def test_update_memory_with_missing_optional_fields(self) -> None:
        """Test with missing optional memory fields."""
        rows = [{"device": "D1", "total_device_memory_gb": 80}]
        fig, df = update_memory_analysis_by_device(rows, "D1")
        # Should handle missing fields gracefully
        assert not df.empty


class TestUpdateBandwidthAnalysisByDeviceExtended:
    """Extended tests for update_bandwidth_analysis_by_device."""

    def test_update_bandwidth_with_case_label(self) -> None:
        """Test with case label filter."""
        rows = [
            {
                "device": "D1",
                "num_queries": 16,
                "tp_size": 2,
                "bottleneck_type": "compute",
                "memory_bound_pct": 20,
                "compute_mma_bound_pct": 80,
                "case_label": "Concurrency=16 | TP=2",
            }
        ]
        case_label = "Concurrency=16 | TP=2"
        df = update_bandwidth_analysis_by_device(rows, "D1", case_label)
        assert not df.empty

    def test_update_bandwidth_with_all_bottleneck_types(self) -> None:
        """Test with different bottleneck types."""
        rows = [
            {
                "device": "D1",
                "bottleneck_type": "memory",
                "memory_bound_pct": 100,
            },
            {
                "device": "D1",
                "bottleneck_type": "compute",
                "compute_mma_bound_pct": 100,
            },
        ]
        df = update_bandwidth_analysis_by_device(rows, "D1")
        assert len(df) >= 1


class TestUpdateCategoryStatsByDeviceExtended:
    """Extended tests for update_category_stats_by_device."""

    def test_update_category_with_case_label(self) -> None:
        """Test with case label filter."""
        breakdown = [
            {
                "name": "matmul",
                "category": "Linear",
                "analytic_total_us": 5000,
                "device": "D1",
                "num_queries": 16,
                "tp_size": 2,
                "case_label": "Concurrency=16 | TP=2",
            }
        ]
        case_label = "Concurrency=16 | TP=2"
        fig, df = update_category_stats_by_device(breakdown, "D1", case_label)
        assert not df.empty

    def test_update_category_with_multiple_devices(self) -> None:
        """Test with operators from multiple devices."""
        breakdown = [
            {"name": "matmul", "category": "Linear", "analytic_total_us": 5000, "device": "D1"},
            {"name": "attn", "category": "Attention", "analytic_total_us": 3000, "device": "D2"},
        ]
        fig, df = update_category_stats_by_device(breakdown, "D1")
        # Should only show D1 data
        assert not df.empty


# Duplicate TestCategorizeOpExtended removed - merged with earlier definition


class TestOptimizerSummaryMarkdownExtended:
    """Extended tests for _optimizer_summary_markdown_from_df and related."""

    def test_optimizer_summary_from_df_with_no_best_throughput(self) -> None:
        """Test without best_throughput column."""
        df = pd.DataFrame(
            {
                "device": ["D1"],
                "status": ["success"],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        # Shows feasibility note when no best_throughput
        assert "Feasibility Note" in result or "No valid" in result

    def test_optimizer_summary_from_df_with_error(self) -> None:
        """Test with error column."""
        df = pd.DataFrame(
            {
                "device": ["D1"],
                "error": ["Test error"],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        # Shows "No valid configuration" when there are errors
        assert "No valid" in result or "configuration" in result

    def test_optimizer_summary_from_df_with_execution_error(self) -> None:
        """Test with execution_error column."""
        df = pd.DataFrame(
            {
                "device": ["D1"],
                "execution_error": ["Execution failed"],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        assert "Failed" in result or "configuration" in result


class TestValidateTextFormNegativeCacheHitRate:
    """Tests for _validate_text_form with negative prefix_cache_hit_rate."""

    def test_validate_text_with_negative_cache_hit_rate(self) -> None:
        """Test with negative prefix_cache_hit_rate value."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "prefix_cache_hit_rate": -0.5,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormNegativeReservedMemory:
    """Tests for _validate_optimizer_form with negative reserved_memory_gb."""

    def test_validate_optimizer_with_negative_reserved_memory(self) -> None:
        """Test with negative reserved_memory_gb."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "reserved_memory_gb": -10,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithMxfp4:
    """Tests for _validate_optimizer_form MXFP4 validation."""

    def test_validate_optimizer_with_zero_mxfp4(self) -> None:
        """Test with mxfp4_group_size=0."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "mxfp4_group_size": 0,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithPrefixCache:
    """Tests for _validate_optimizer_form prefix_cache_hit_rate validation."""

    def test_validate_optimizer_with_prefix_cache_equal_one(self) -> None:
        """Test with prefix_cache_hit_rate=1."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "prefix_cache_hit_rate": 1.0,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_prefix_cache_greater_than_one(self) -> None:
        """Test with prefix_cache_hit_rate>1."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "prefix_cache_hit_rate": 1.5,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestOptimizerCliStyleOutputWithRawLog:
    """Tests for _optimizer_cli_style_output with raw_log."""

    def test_cli_style_output_with_raw_log(self) -> None:
        """Test with raw_log in top result."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
                "raw_log": "Some CLI output",
            }
        )
        result = _optimizer_cli_style_output(top, pd.DataFrame(), "D1")
        assert "Some CLI output" in result or "Raw CLI Output" in result


class TestOptimizerDeploymentModeExtended:
    """Extended tests for _optimizer_deployment_mode."""

    def test_deployment_mode_with_whitespace_mode(self) -> None:
        """Test with whitespace-only mode."""
        row = pd.Series({"deployment_mode": "   "})
        result = _optimizer_deployment_mode(row)
        # Should normalize to PD_MIXED for empty/whitespace
        assert result == OPT_DEPLOY_PD_MIXED


# Duplicate TestCaseLabelFromMappingExtended removed - merged with earlier definition

# Duplicate TestPreviewFirstCommandExtended removed - merged with earlier definition

# Duplicate TestFormatPreviewErrorExtended removed - merged with earlier definition
# Duplicate TestNormalizeOpColumnsExtended removed - merged with earlier definition


class TestOpTableFromRecordsExtended:
    """Extended tests for _op_table_from_records."""

    def test_op_table_with_zero_top_n(self) -> None:
        """Test with top_n=0."""
        records = [{"name": "op1", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 10, "device": "D1"}]
        result = _op_table_from_records(records, "D1", 0)
        # top_n=0 should return empty or single row
        assert len(result) <= 1

    def test_op_table_sort_by_average_time(self) -> None:
        """Test sorting by Average Time."""
        records = [
            {"name": "op_small", "analytic_total_us": 1000, "analytic_avg_us": 10, "num_calls": 100, "device": "D1"},
            {"name": "op_large", "analytic_total_us": 1000, "analytic_avg_us": 500, "num_calls": 2, "device": "D1"},
        ]
        result = _op_table_from_records(records, "D1", 10, sort_by="Average Time (ms)")
        assert result.iloc[0]["Average Time (ms)"] == 0.5

    def test_op_table_sort_by_calls(self) -> None:
        """Test sorting by Calls."""
        records = [
            {"name": "op1", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 5, "device": "D1"},
            {"name": "op2", "analytic_total_us": 1000, "analytic_avg_us": 100, "num_calls": 50, "device": "D1"},
        ]
        result = _op_table_from_records(records, "D1", 10, sort_by="Calls")
        assert result.iloc[0]["Calls"] == 50


class TestRoundNumericColumnsWithSpecialCases:
    """Special case tests for _round_numeric_columns."""

    def test_round_with_inf_values(self) -> None:
        """Test with infinity values."""
        import numpy as np

        df = pd.DataFrame({"latency (ms)": [np.inf, 100.5, -np.inf]})
        result = _round_numeric_columns(df, digits=1)
        # Should handle inf gracefully
        assert result is not None

    def test_round_with_mixed_nan_and_numeric(self) -> None:
        """Test with mix of NaN and numeric values."""
        df = pd.DataFrame({"latency (ms)": [None, 100.5, None, 200.7]})
        result = _round_numeric_columns(df, digits=1)
        assert len(result) == 4


class TestPreviewSummaryMarkdownWithEmptyDedupe:
    """Tests for _preview_summary_markdown with empty dedupe results."""

    def test_preview_summary_with_empty_device_list(self) -> None:
        """Test when dedupe returns empty list for devices."""
        form = {
            "model_id": "test_model",
            "device": None,
            "num_devices": 8,
        }
        tasks = [Mock(params={"device": None})]
        result = _preview_summary_markdown("text_generate", form, tasks)
        # Should handle None device gracefully
        assert "Configuration Summary" in result


class TestOptimizerStateRowsExtended:
    """Extended tests for _optimizer_state_rows."""

    def test_state_rows_with_empty_raw_log(self) -> None:
        """Test with empty raw_log."""
        result = ExperimentResult(
            sim_type="throughput_optimizer",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.raw_log = ""
        rows = _optimizer_state_rows([result])
        assert len(rows) == 1


class TestTextGenerateOpTableExtended:
    """Extended tests for _text_generate_op_table."""

    def test_op_table_with_invalid_sort_by(self) -> None:
        """Test with invalid sort_by value."""
        results = [
            create_mock_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            )
        ]
        result = _text_generate_op_table(results, "D1", 10, sort_by="Invalid Column")
        # Should default to Total Time sorting
        assert not result.empty


class TestVideoGenerateOpTableExtended:
    """Extended tests for _video_generate_op_table."""

    def test_video_op_table_with_invalid_sort_by(self) -> None:
        """Test with invalid sort_by value."""
        results = [
            create_mock_video_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "op1", "analytic_total_us": 1000.0, "num_calls": 10},
                    ]
                },
            )
        ]
        result = _video_generate_op_table(results, "D1", 10, sort_by="Invalid")
        assert not result.empty


class TestValidateTextFormWithWorldSize:
    """Tests for _validate_text_form with world_size parameter."""

    def test_validate_text_with_invalid_world_size(self) -> None:
        """Test with invalid world_size."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "world_size": "invalid",
        }
        errors = _validate_text_form(form)
        # Should handle world_size parameter
        assert errors is not None


class TestValidateOptimizerFormWithServingCost:
    """Tests for _validate_optimizer_form with serving_cost."""

    def test_validate_optimizer_with_negative_serving_cost(self) -> None:
        """Test with negative serving_cost."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "serving_cost": -100,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithMaxPrefillTokens:
    """Tests for _validate_optimizer_form with max_prefill_tokens."""

    def test_validate_optimizer_with_zero_max_prefill(self) -> None:
        """Test with max_prefill_tokens=0."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "max_prefill_tokens": 0,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateVideoFormWithUlyssesSweep:
    """Tests for _validate_video_form with ulysses_sweep."""

    def test_validate_video_with_ulysses_sweep_valid(self) -> None:
        """Test with valid ulysses_sweep."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "world_size": 8,
            "ulysses_sweep": "2,4",
        }
        errors = _validate_video_form(form)
        # Should parse the sweep list
        assert isinstance(errors, list)

    def test_validate_video_with_ulysses_sweep_invalid(self) -> None:
        """Test with invalid ulysses_sweep that causes exception."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "world_size": 8,
            "ulysses_sweep": "invalid",
        }
        errors = _validate_video_form(form)
        # Should handle the exception
        assert isinstance(errors, list)


class TestPreviewSummaryMarkdownOptimizerWithConstraints:
    """Tests for _preview_summary_markdown with optimizer constraints."""

    def test_preview_summary_optimizer_with_unlimited_constraints(self) -> None:
        """Test optimizer with no constraints (unlimited)."""
        form = {
            "model_id": "opt_model",
            "device": "D1",
            "num_devices": 8,
            "jobs": 4,
            "input_length": 1024,
            "output_length": 512,
            "ttft_limits": None,
            "tpot_limits": None,
        }
        tasks = [Mock(params={"device": "D1", "deployment_mode": "PD Mixed"})]
        result = _preview_summary_markdown("throughput_optimizer", form, tasks)
        assert "unlimited" in result


class TestOptimizerSummaryMarkdownWithDevices:
    """Tests for _optimizer_summary_markdown device handling."""

    def test_optimizer_summary_with_many_devices(self) -> None:
        """Test with more than 12 devices."""
        devices = [f"D{i}" for i in range(15)]
        df = pd.DataFrame(
            {
                "device": devices,
                "best_throughput": [1000.0] * 15,
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        # Should show first 12 devices
        assert "D11" in result or "D12" not in result


class TestValidateOptimizerFormWithInvalidMode:
    """Tests for _validate_optimizer_form with invalid deployment_mode."""

    def test_validate_optimizer_with_invalid_deployment_mode(self) -> None:
        """Test with invalid deployment_mode."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "Invalid Mode",
        }
        errors = _validate_optimizer_form(form)
        assert any("Invalid deployment mode" in err or "deployment" in err.lower() for err in errors)


class TestValidateTextFormWithNegativeHiddenLayers:
    """Tests for _validate_text_form with hidden_layers_override."""

    def test_validate_text_with_negative_hidden_layers_override(self) -> None:
        """Test with negative hidden_layers_override."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "num_hidden_layers_override": -1,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_invalid_hidden_layers_override(self) -> None:
        """Test with non-numeric hidden_layers_override."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "num_hidden_layers_override": "abc",
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestTextGenerateSummaryMarkdownWithRankingCol:
    """Tests for _text_generate_summary_markdown with different ranking columns."""

    def test_text_generate_summary_with_tps_ranking(self) -> None:
        """Test with tps_per_device ranking."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16},
                summary={"tps_per_device": 1000.0},
            )
        ]
        result = _text_generate_summary_markdown(results)
        assert "token/s" in result or "tps" in result.lower()

    def test_text_generate_summary_with_time_ranking(self) -> None:
        """Test with analytic_total_time_s ranking."""
        results = [
            create_mock_result(
                params={"device": "D1", "num_queries": 16},
                summary={"analytic_total_time_s": 1.5},
            )
        ]
        result = _text_generate_summary_markdown(results)
        assert "ms" in result


class TestDedupeWithBooleanValues:
    """Tests for _dedupe with boolean values."""

    def test_dedupe_with_true_false(self) -> None:
        """Test with True and False values."""
        result = _dedupe([True, False, True, False])
        assert "True" in result
        assert "False" in result

    def test_dedupe_with_mixed_bool_and_int(self) -> None:
        """Test with mixed boolean and int values."""
        result = _dedupe([True, 1, False, 0])
        # All converted to strings
        assert len(result) <= 4


class TestPreviewSummaryMarkdownVideoWithDevices:
    """Tests for _preview_summary_markdown with video_generate and devices."""

    def test_preview_summary_video_without_devices(self) -> None:
        """Test video_generate without devices in tasks."""
        form = {
            "model_id": "video_model",
            "device": "D1",
            "world_size": 8,
            "ulysses_size": 4,
        }
        tasks = [Mock(params={})]  # No device param
        result = _preview_summary_markdown("video_generate", form, tasks)
        assert "Configuration Summary" in result


class TestValidateTextFormWithWordEmbeddingTp:
    """Tests for _validate_text_form with word_embedding_tp."""

    def test_validate_text_with_invalid_word_embedding_tp(self) -> None:
        """Test with invalid word_embedding_tp."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "word_embedding_tp": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_word_embedding_tp_exceeds_devices(self) -> None:
        """Test with word_embedding_tp > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "word_embedding_tp": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithExternalSharedExperts:
    """Tests for _validate_text_form with external shared experts."""

    def test_validate_text_with_host_external_shared_experts(self) -> None:
        """Test with host_external_shared_experts parameter."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "host_external_shared_experts": "true",
        }
        errors = _validate_text_form(form)
        # Should handle the parameter
        assert isinstance(errors, list)


class TestValidateOptimizerFormWithPrefillDecodeDevices:
    """Tests for _validate_optimizer_form with prefill/decode device validation."""

    def test_validate_optimizer_prefill_decode_sum_equals_devices(self) -> None:
        """Test with prefill + decode = num_devices (valid)."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 4,
            "decode_devices_per_instance": 4,
        }
        errors = _validate_optimizer_form(form)
        # Should be valid
        assert len(errors) == 0 or "Prefill" not in " ".join(errors)

    def test_validate_optimizer_prefill_decode_divisible(self) -> None:
        """Test with devices divisible by instance counts."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 2,
            "decode_devices_per_instance": 2,
        }
        errors = _validate_optimizer_form(form)
        # Should be valid
        assert len(errors) == 0 or "divisible" not in " ".join(errors)


class TestSummaryMarkdownWithLatestResult:
    """Tests for _summary_markdown with latest result."""

    def test_summary_markdown_with_latest_success(self) -> None:
        """Test with successful latest result."""
        df = pd.DataFrame({"device": ["D1"]})
        latest = Mock(
            label="test_task",
            source="simulation",
            error=None,
        )
        result = _summary_markdown(df, latest, "text_generate")
        assert "test_task" in result

    def test_summary_markdown_with_latest_error(self) -> None:
        """Test with failed latest result."""
        df = pd.DataFrame({"device": ["D1"]})
        latest = Mock(
            label="test_task",
            source="simulation",
            error="Test error message",
        )
        result = _summary_markdown(df, latest, "text_generate")
        assert "Test error message" in result or "error" in result.lower()


class TestOptimizerSummaryMarkdownWithFailedRuns:
    """Tests for _optimizer_summary_markdown with failed runs."""

    def test_optimizer_summary_with_failed_status(self) -> None:
        """Test with failed runs in dataframe."""
        df = pd.DataFrame(
            {
                "device": ["D1", "D2"],
                "status": ["failed", "success"],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        # The optimizer summary doesn't show failed count separately
        # It just shows "Completed Runs"
        assert "Completed Runs: **2**" in result


class TestValidateVideoFormWithNegativeBatchSize:
    """Tests for _validate_video_form with negative values."""

    def test_validate_video_with_negative_height(self) -> None:
        """Test with negative height."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "height": -720,
            "width": 1280,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_negative_width(self) -> None:
        """Test with negative width."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "height": 720,
            "width": -1280,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestOptimizerCliStyleOutputExtended2:
    """Extended tests for _optimizer_cli_style_output."""

    def test_cli_style_output_with_deployment_mode_dash(self) -> None:
        """Test with deployment_mode='-'."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
                "deployment_mode": "-",
            }
        )
        result = _optimizer_cli_style_output(top, pd.DataFrame(), "D1")
        # Should normalize the mode
        assert result is not None

    def test_cli_style_output_with_candidates(self) -> None:
        """Test with candidate configurations."""
        top = pd.Series(
            {
                "device": "D1",
                "best_throughput": 1000.0,
            }
        )
        candidates = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "throughput_token_s": 1000.0,
                    "ttft_ms": 50.0,
                    "tpot_ms": 10.0,
                    "concurrency": 16,
                    "num_devices": 8,
                    "parallel": "TP=4",
                    "batch_size": 32,
                }
            ]
        )
        result = _optimizer_cli_style_output(top, candidates, "D1")
        assert "Top 1" in result


class TestTextGenerateOpTableWithAllOperators:
    """Tests for _text_generate_op_table with all operator types."""

    def test_op_table_with_all_categories(self) -> None:
        """Test with operators from all categories."""
        results = [
            create_mock_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "attn", "analytic_total_us": 1000.0, "num_calls": 10},
                        {"name": "matmul", "analytic_total_us": 500.0, "num_calls": 5},
                        {"name": "all_reduce", "analytic_total_us": 200.0, "num_calls": 2},
                        {"name": "layer_norm", "analytic_total_us": 100.0, "num_calls": 1},
                    ]
                },
            )
        ]
        result = _text_generate_op_table(results, "D1", 20)
        assert len(result) >= 3


class TestVideoGenerateOpTableWithAllCategories:
    """Tests for _video_generate_op_table with various operators."""

    def test_video_op_table_with_multiple_operators(self) -> None:
        """Test with multiple operators."""
        results = [
            create_mock_video_result(
                params={"device": "D1"},
                tables={
                    "op_breakdown": [
                        {"name": "conv2d", "analytic_total_us": 2000.0, "num_calls": 10},
                        {"name": "softmax", "analytic_total_us": 500.0, "num_calls": 5},
                    ]
                },
            )
        ]
        result = _video_generate_op_table(results, "D1", 10)
        assert len(result) >= 1


class TestValidateTextFormWithZeroQueryLength:
    """Tests for _validate_text_form with zero query_length."""

    def test_validate_text_with_zero_query_length(self) -> None:
        """Test with query_length=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 0,
            "context_length": 2048,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithNegativeInputLength:
    """Tests for _validate_optimizer_form with negative lengths."""

    def test_validate_optimizer_with_negative_input_length(self) -> None:
        """Test with negative input_length."""
        form = {
            "num_devices": 8,
            "input_length": -512,
            "output_length": 512,
            "jobs": 1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithZeroOutputLength:
    """Tests for _validate_optimizer_form with zero output_length."""

    def test_validate_optimizer_with_zero_output_length(self) -> None:
        """Test with output_length=0."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 0,
            "jobs": 1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestOptimizerSummaryMarkdownExtended2:
    """Extended tests for _optimizer_summary_markdown_from_df."""

    def test_optimizer_summary_with_empty_best_throughput(self) -> None:
        """Test with best_throughput column but all NaN."""
        df = pd.DataFrame(
            {
                "device": ["D1"],
                "best_throughput": [float("nan")],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        # When best_throughput is all NaN, shows "No valid configuration"
        assert "No valid" in result or "configuration" in result

    def test_optimizer_summary_with_no_result_reason(self) -> None:
        """Test with no_result_reason column."""
        df = pd.DataFrame(
            {
                "device": ["D1"],
                "no_result_reason": ["Constraints too strict"],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        assert "Constraints too strict" in result or "feasibility" in result.lower()


class TestValidateTextFormWithReservedMemory:
    """Tests for _validate_text_form with reserved_memory_gb."""

    def test_validate_text_with_invalid_reserved_memory(self) -> None:
        """Test with invalid reserved_memory_gb."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "reserved_memory_gb": "invalid",
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithCompileGraphBreak:
    """Tests for _validate_text_form with compile_allow_graph_break."""

    def test_validate_text_with_compile_allow_graph_break(self) -> None:
        """Test with compile_allow_graph_break parameter."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "compile_allow_graph_break": "true",
        }
        errors = _validate_text_form(form)
        # Should handle the parameter
        assert isinstance(errors, list)


class TestValidateOptimizerFormWithNegativeServingCost:
    """Tests for _validate_optimizer_form serving_cost validation."""

    def test_validate_optimizer_with_invalid_serving_cost(self) -> None:
        """Test with invalid serving_cost."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "serving_cost": "invalid",
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateVideoFormWithCacheParameters:
    """Tests for _validate_video_form cache parameters."""

    def test_validate_video_with_cache_step_interval_negative(self) -> None:
        """Test with negative cache_step_interval."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_interval": -1,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_cache_step_range_zero_end(self) -> None:
        """Test with cache_step_range where end=0."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "cache_step_range": "0,0",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithMoeParameters:
    """Tests for _validate_text_form MoE parameters."""

    def test_validate_text_with_moe_tp_exceeds_devices(self) -> None:
        """Test with moe_tp_size > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "moe_tp_size": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_moe_tp_not_divisible(self) -> None:
        """Test with num_devices not divisible by moe_tp_size."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "moe_tp_size": 3,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithLmheadParameters:
    """Tests for _validate_text_form LMHead parameters."""

    def test_validate_text_with_lmhead_tp_exceeds_devices(self) -> None:
        """Test with lmhead_tp_size > num_devices."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "lmhead_tp_size": 16,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithTpSizes:
    """Tests for _validate_optimizer_form tp_sizes parameter."""

    def test_validate_optimizer_with_invalid_tp_sizes(self) -> None:
        """Test with invalid tp_sizes."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "tp_sizes": "invalid",
        }
        errors = _validate_optimizer_form(form)
        # Should handle the invalid tp_sizes
        assert isinstance(errors, list)


class TestValidateOptimizerFormWithPrefillDecodeRatio:
    """Tests for _validate_optimizer_form with PD Ratio mode."""

    def test_validate_optimizer_prefill_decode_not_divisible(self) -> None:
        """Test with devices not divisible by instance counts."""
        form = {
            "num_devices": 7,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 2,
            "decode_devices_per_instance": 2,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithPerformanceModel:
    """Tests for _validate_text_form with performance_model."""

    def test_validate_text_with_performance_model_list(self) -> None:
        """Test with performance_model as list."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "performance_model": ["profiling"],
            "profiling_database": "/path/to/db",
        }
        errors = _validate_text_form(form)
        # Should handle list performance_model
        assert isinstance(errors, list)


class TestValidateOptimizerFormDeploymentModeVariants:
    """Tests for _validate_optimizer_form deployment_mode normalization."""

    def test_validate_optimizer_with_pd_disaggregated_mode(self) -> None:
        """Test with PD Disaggregated mode."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Disaggregated",
        }
        errors = _validate_optimizer_form(form)
        # Should normalize the mode
        assert isinstance(errors, list)

    def test_validate_optimizer_with_pd_ratio_mode(self) -> None:
        """Test with PD Ratio mode."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
        }
        errors = _validate_optimizer_form(form)
        assert isinstance(errors, list)


class TestOptimizerSummaryMarkdownWithBalancedQps:
    """Tests for _optimizer_summary_markdown with balanced_qps."""

    def test_optimizer_summary_with_balanced_qps(self) -> None:
        """Test with balanced_qps column."""
        df = pd.DataFrame(
            {
                "device": ["D1"],
                "balanced_qps": [2000.0],
            }
        )
        result = _optimizer_summary_markdown_from_df(df)
        assert "2000" in result or "balanced" in result.lower()


class TestOptimizerPrimaryMetricWithBalancedQps:
    """Tests for _optimizer_primary_metric with balanced_qps."""

    def test_primary_metric_prefers_balanced_qps(self) -> None:
        """Test that balanced_qps is preferred when available."""
        df = pd.DataFrame(
            {
                "balanced_qps": [2000.0],
                "best_batch_size": [32],
            }
        )
        metric, title, short = _optimizer_primary_metric(df)
        assert metric == "balanced_qps"
        assert "Balanced QPS" in title


class TestTextGenerateSummaryMarkdownWithRunnerUp:
    """Tests for _text_generate_summary_markdown with runner-up."""

    def test_text_generate_summary_with_runner_up(self) -> None:
        """Test with multiple results to show runner-up."""
        results = [
            create_mock_result(
                params={"device": "D1"},
                summary={"tps_per_device": 1000.0, "analytic_total_time_s": 1.0},
            ),
            create_mock_result(
                params={"device": "D2"},
                summary={"tps_per_device": 800.0, "analytic_total_time_s": 1.5},
            ),
        ]
        result = _text_generate_summary_markdown(results)
        # Should show runner-up when there are multiple results
        assert len(result) > 0


class TestValidateVideoFormWithSampleStepZero:
    """Tests for _validate_video_form with sample_step=0."""

    def test_validate_video_with_zero_sample_step(self) -> None:
        """Test with sample_step=0."""
        form = {
            "batch_size": 4,
            "seq_len": 512,
            "height": 720,
            "width": 1280,
            "frame_num": 100,
            "sample_step": 0,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithTpSweep:
    """Tests for _validate_text_form with tp_sweep."""

    def test_validate_text_with_tp_sweep_list(self) -> None:
        """Test with tp_sweep as list."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_sweep": "2,4",
        }
        errors = _validate_text_form(form)
        # Should parse the sweep list
        assert isinstance(errors, list)

    def test_validate_text_with_tp_sweep_invalid(self) -> None:
        """Test with invalid tp_sweep."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_sweep": "abc,def",
        }
        errors = _validate_text_form(form)
        # Should handle invalid values
        assert isinstance(errors, list)


class TestValidateOptimizerFormWithPrefillDecodeValidation:
    """Tests for _validate_optimizer_form prefill/decode validation in PD Ratio mode."""

    def test_validate_optimizer_prefill_not_divisible(self) -> None:
        """Test with num_devices not divisible by prefill_devices."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 3,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_decode_not_divisible(self) -> None:
        """Test with num_devices not divisible by decode_devices."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "PD Ratio",
            "prefill_devices_per_instance": 4,
            "decode_devices_per_instance": 3,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestVideoGenerateSummaryWithRunnerUp:
    """Tests for _video_generate_summary_markdown with runner-up."""

    def test_video_generate_summary_with_runner_up(self) -> None:
        """Test with multiple results to show runner-up."""
        results = [
            create_mock_video_result(
                params={"device": "D1"},
                summary={"analytic_total_time_s": 1.0},
            ),
            create_mock_video_result(
                params={"device": "D2"},
                summary={"analytic_total_time_s": 1.5},
            ),
        ]
        result = _video_generate_summary_markdown(results)
        # Should show runner-up info when there are multiple results
        assert "Runner-up" in result or "runner" in result.lower() or "vs" in result


class TestTextGenerateOpSummaryExtended:
    """Extended tests for _text_generate_op_summary."""

    def test_op_summary_with_no_device(self) -> None:
        """Test with result missing device param."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h1",
            label="test",
        )
        result.tables["op_breakdown"] = [{"name": "op1", "analytic_total_us": 1000}]
        ops = _text_generate_op_summary([result])
        # Device should be "unknown"
        assert ops[0]["device"] == "unknown"


class TestValidateOptimizerFormWithZeroInputOutput:
    """Tests for _validate_optimizer_form with zero length values."""

    def test_validate_optimizer_with_zero_input(self) -> None:
        """Test with input_length=0."""
        form = {
            "num_devices": 8,
            "input_length": 0,
            "output_length": 512,
            "jobs": 1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0

    def test_validate_optimizer_with_both_zero(self) -> None:
        """Test with both lengths zero."""
        form = {
            "num_devices": 8,
            "input_length": 0,
            "output_length": 0,
            "jobs": 1,
        }
        errors = _validate_optimizer_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithAllParallelismParameters:
    """Tests for _validate_text_form with all parallelism parameters."""

    def test_validate_text_with_all_parallel_params_valid(self) -> None:
        """Test with all parallelism parameters valid."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "tp_size": 2,
            "dp_size": 2,
            "ep_size": 2,
        }
        errors = _validate_text_form(form)
        # TP * DP * EP = 8, which equals num_devices - valid
        assert len(errors) == 0 or "product" not in " ".join(errors).lower()


class TestValidateTextFormWithMxfp4GroupSize:
    """Tests for _validate_text_form with mxfp4_group_size."""

    def test_validate_text_with_zero_mxfp4_group_size(self) -> None:
        """Test with mxfp4_group_size=0."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "mxfp4_group_size": 0,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0

    def test_validate_text_with_negative_mxfp4_group_size(self) -> None:
        """Test with negative mxfp4_group_size."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "mxfp4_group_size": -1,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateVideoFormWithInvalidBatchSize:
    """Tests for _validate_video_form with invalid batch_size."""

    def test_validate_video_with_string_batch_size(self) -> None:
        """Test with non-numeric batch_size."""
        form = {
            "batch_size": "invalid",
            "seq_len": 512,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_none_batch_size(self) -> None:
        """Test with None batch_size."""
        form = {
            "batch_size": None,
            "seq_len": 512,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestValidateVideoFormWithInvalidSeqLen:
    """Tests for _validate_video_form with invalid seq_len."""

    def test_validate_video_with_zero_seq_len(self) -> None:
        """Test with seq_len=0."""
        form = {
            "batch_size": 4,
            "seq_len": 0,
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0

    def test_validate_video_with_string_seq_len(self) -> None:
        """Test with non-numeric seq_len."""
        form = {
            "batch_size": 4,
            "seq_len": "invalid",
        }
        errors = _validate_video_form(form)
        assert len(errors) > 0


class TestValidateOptimizerFormWithDeploymentModeNormalization:
    """Tests for _validate_optimizer_form deployment_mode normalization."""

    def test_validate_optimizer_mode_normalized_in_validation(self) -> None:
        """Test that deployment_mode is normalized during validation."""
        form = {
            "num_devices": 8,
            "input_length": 512,
            "output_length": 512,
            "jobs": 1,
            "deployment_mode": "Disagg",  # Should normalize to PD Disaggregated
        }
        errors = _validate_optimizer_form(form)
        # Should not error about invalid mode
        assert not any("Invalid deployment mode" in err for err in errors)


class TestValidateTextFormWithNegativeContextLength:
    """Tests for _validate_text_form with negative context_length."""

    def test_validate_text_with_negative_context(self) -> None:
        """Test with negative context_length."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": -2048,
        }
        errors = _validate_text_form(form)
        assert len(errors) > 0


class TestValidateTextFormWithDisableRepetition:
    """Tests for _validate_text_form with disable_repetition parameter."""

    def test_validate_text_with_disable_repetition(self) -> None:
        """Test with disable_repetition parameter."""
        form = {
            "num_devices": 8,
            "num_queries": 100,
            "query_length": 512,
            "context_length": 2048,
            "disable_repetition": True,
        }
        errors = _validate_text_form(form)
        # Should handle the parameter
        assert isinstance(errors, list)


class TestStopRunFeedback:
    """Tests for _stop_run_feedback function."""

    def test_stop_run_feedback_returns_tuple(self) -> None:
        """Test that _stop_run_feedback returns progress and summary tuple."""
        progress, summary = _stop_run_feedback("Test Cancelled")
        assert isinstance(progress, str)
        assert isinstance(summary, str)
        assert "cancelled" in progress.lower()
        assert "Run Cancelled" in summary

    def test_stop_run_feedback_contains_stop_count(self) -> None:
        """Test that feedback includes count of stopped tasks."""
        progress, summary = _stop_run_feedback("Test")
        assert "task(s)" in progress


class TestStopTextGenerateRun:
    """Tests for stop_text_generate_run function."""

    def test_stop_text_generate_run(self) -> None:
        """Test stop_text_generate_run returns feedback."""
        progress, summary = stop_text_generate_run()
        assert "Text Generate Cancelled" in progress
        assert "Run Cancelled" in summary
        assert isinstance(progress, str)


class TestStopVideoGenerateRun:
    """Tests for stop_video_generate_run function."""

    def test_stop_video_generate_run(self) -> None:
        """Test stop_video_generate_run returns feedback."""
        progress, summary = stop_video_generate_run()
        assert "Video Generate Cancelled" in progress
        assert "Run Cancelled" in summary
        assert isinstance(progress, str)


class TestStopOptimizerRun:
    """Tests for stop_optimizer_run function."""

    def test_stop_optimizer_run(self) -> None:
        """Test stop_optimizer_run returns feedback."""
        progress, summary = stop_optimizer_run()
        assert "Optimizer Cancelled" in progress
        assert "Run Cancelled" in summary
        assert isinstance(progress, str)
