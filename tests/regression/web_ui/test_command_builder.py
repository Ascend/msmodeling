"""Tests for web_ui.command_builder module."""

from __future__ import annotations


from web_ui.command_builder import (
    _as_bool,
    _base_cmd,
    _device_matrix,
    _mode_name,
    _normalize_optimizer_deployment_mode,
    _optional_str,
    _performance_models,
    build_optimizer_tasks,
    build_text_generate_tasks,
    build_video_generate_tasks,
)


class TestNormalizeOptimizerDeploymentMode:
    """Tests for _normalize_optimizer_deployment_mode function."""

    def test_normalize_empty_string(self) -> None:
        """Test normalizing empty string."""
        result = _normalize_optimizer_deployment_mode("")
        assert result == "PD Aggregated"

    def test_normalize_pd_aggregated(self) -> None:
        """Test normalizing PD Aggregated."""
        result = _normalize_optimizer_deployment_mode("PD Aggregated")
        assert result == "PD Aggregated"

    def test_normalize_pd_disaggregated(self) -> None:
        """Test normalizing PD Disaggregated."""
        result = _normalize_optimizer_deployment_mode("PD Disaggregated")
        assert result == "PD Disaggregated"

    def test_normalize_pd_ratio(self) -> None:
        """Test normalizing PD Ratio."""
        result = _normalize_optimizer_deployment_mode("PD Ratio")
        assert result == "PD Ratio"

    def test_normalize_aggregation_alias(self) -> None:
        """Test normalizing Aggregation alias."""
        result = _normalize_optimizer_deployment_mode("Aggregation")
        assert result == "PD Aggregated"

    def test_normalize_pd_mixed_alias(self) -> None:
        """Test normalizing PD Mixed alias."""
        result = _normalize_optimizer_deployment_mode("PD Mixed")
        assert result == "PD Aggregated"

    def test_normalize_disagg_alias(self) -> None:
        """Test normalizing Disagg alias."""
        result = _normalize_optimizer_deployment_mode("Disagg")
        assert result == "PD Disaggregated"

    def test_normalize_unknown_mode(self) -> None:
        """Test normalizing unknown mode returns as-is."""
        result = _normalize_optimizer_deployment_mode("Unknown Mode")
        assert result == "Unknown Mode"

    def test_normalize_none(self) -> None:
        """Test normalizing None returns default deployment mode."""
        result = _normalize_optimizer_deployment_mode(None)
        # Empty string maps to PD Aggregated (default deployment mode)
        assert result == "PD Aggregated"


class TestDeviceMatrix:
    """Tests for _device_matrix function."""

    def test_device_matrix_with_primary_only(self) -> None:
        """Test device matrix with primary device only."""
        result = _device_matrix("D1", [])
        assert result == ["D1"]

    def test_device_matrix_with_competitors(self) -> None:
        """Test device matrix with competitors."""
        result = _device_matrix("D1", ["D2", "D3"])
        assert result == ["D1", "D2", "D3"]

    def test_device_matrix_deduplicates(self) -> None:
        """Test device matrix removes duplicates."""
        result = _device_matrix("D1", ["D1", "D2", "D1"])
        assert result == ["D1", "D2"]

    def test_device_matrix_filters_empty(self) -> None:
        """Test device matrix filters empty strings."""
        result = _device_matrix("D1", ["", "D2", None])
        assert result == ["D1", "D2"]

    def test_device_matrix_returns_primary_when_empty(self) -> None:
        """Test device matrix returns primary when all competitors empty."""
        result = _device_matrix("D1", ["", None, ""])
        assert result == ["D1"]

    def test_device_matrix_preserves_order(self) -> None:
        """Test device matrix preserves order."""
        result = _device_matrix("D1", ["D3", "D2"])
        assert result == ["D1", "D3", "D2"]


class TestBaseCmd:
    """Tests for _base_cmd function."""

    def test_base_cmd_text_generate(self) -> None:
        """Test base command for text generate module."""
        result = _base_cmd("cli.inference.text_generate")
        assert "python" in result[0].lower()
        assert result[1] == "-m"
        assert result[2] == "cli.inference.text_generate"

    def test_base_cmd_video_generate(self) -> None:
        """Test base command for video generate module."""
        result = _base_cmd("cli.inference.video_generate")
        assert "python" in result[0].lower()
        assert result[2] == "cli.inference.video_generate"

    def test_base_cmd_optimizer(self) -> None:
        """Test base command for optimizer module."""
        result = _base_cmd("cli.inference.throughput_optimizer")
        assert "python" in result[0].lower()
        assert result[2] == "cli.inference.throughput_optimizer"


class TestAsBool:
    """Tests for _as_bool function."""

    def test_as_bool_true(self) -> None:
        """Test _as_bool with True."""
        assert _as_bool(True) is True

    def test_as_bool_false(self) -> None:
        """Test _as_bool with False."""
        assert _as_bool(False) is False

    def test_as_bool_truthy_value(self) -> None:
        """Test _as_bool with truthy value."""
        assert _as_bool(1) is True

    def test_as_bool_falsy_value(self) -> None:
        """Test _as_bool with falsy value."""
        assert _as_bool(0) is False

    def test_as_bool_none(self) -> None:
        """Test _as_bool with None."""
        assert _as_bool(None) is False


class TestOptionalStr:
    """Tests for _optional_str function."""

    def test_optional_str_with_value(self) -> None:
        """Test _optional_str with valid string."""
        result = _optional_str("test_value")
        assert result == "test_value"

    def test_optional_str_with_none(self) -> None:
        """Test _optional_str with None."""
        result = _optional_str(None)
        assert result is None

    def test_optional_str_with_empty_string(self) -> None:
        """Test _optional_str with empty string."""
        result = _optional_str("")
        assert result is None

    def test_optional_str_with_number(self) -> None:
        """Test _optional_str with number."""
        result = _optional_str(123)
        assert result == "123"


class TestPerformanceModels:
    """Tests for _performance_models function."""

    def test_performance_models_none(self) -> None:
        """Test with None returns analytic default."""
        result = _performance_models(None)
        assert result == ["analytic"]

    def test_performance_models_empty(self) -> None:
        """Test with empty returns analytic default."""
        result = _performance_models("")
        assert result == ["analytic"]

    def test_performance_models_single_string(self) -> None:
        """Test with single string value."""
        result = _performance_models("profiling")
        assert result == ["profiling"]

    def test_performance_models_comma_string(self) -> None:
        """Test with comma-separated string."""
        result = _performance_models("analytic,profiling")
        assert result == ["analytic", "profiling"]

    def test_performance_models_list(self) -> None:
        """Test with list value."""
        result = _performance_models(["analytic", "profiling"])
        assert result == ["analytic", "profiling"]

    def test_performance_models_tuple(self) -> None:
        """Test with tuple value."""
        result = _performance_models(("analytic", "profiling"))
        assert result == ["analytic", "profiling"]


class TestModeName:
    """Tests for _mode_name function."""

    def test_mode_name_offline(self) -> None:
        """Test mode name for offline (no limits)."""
        result = _mode_name(None, None)
        assert result == "offline"

    def test_mode_name_ttft_constrained(self) -> None:
        """Test mode name for TTFT only constraint."""
        result = _mode_name(100.0, None)
        assert result == "ttft_constrained"

    def test_mode_name_tpot_constrained(self) -> None:
        """Test mode name for TPOT only constraint."""
        result = _mode_name(None, 50.0)
        assert result == "tpot_constrained"

    def test_mode_name_both_constrained(self) -> None:
        """Test mode name for both TTFT and TPOT constraints."""
        result = _mode_name(100.0, 50.0)
        assert result == "ttft_tpot_constrained"


class TestBuildTextGenerateTasks:
    """Tests for build_text_generate_tasks function."""

    def test_build_basic_text_task(self) -> None:
        """Test building a basic text generation task."""
        form = {
            "model_id": "Qwen/Qwen3-32B",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 4,
            "num_queries": 32,
            "num_queries_sweep": None,
            "query_length": 8,
            "context_length": 4500,
            "decode": True,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "",
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "DISABLED",
            "quant_attention_sweep": None,
            "tp_size": 2,
            "tp_sweep": None,
            "dp_size": "auto",
            "ep_size": 1,
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "prefix_cache_hit_rate": 0.0,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "compile_allow_graph_break": False,
            "disable_repetition": False,
            "quantize_lmhead": False,
            "mxfp4_group_size": 32,
            "graph_log_url": None,
            "dump_input_shapes": False,
            "chrome_trace": None,
            "num_hidden_layers_override": 0,
            "o_proj_tp_size": None,
            "o_proj_dp_size": None,
            "mlp_tp_size": None,
            "mlp_dp_size": None,
            "lmhead_tp_size": None,
            "lmhead_dp_size": None,
            "moe_tp_size": None,
            "moe_dp_size": "1",
            "word_embedding_tp": None,
            "enable_redundant_experts": False,
            "enable_external_shared_experts": False,
            "host_external_shared_experts": False,
            "remote_source": "huggingface",
            "performance_model": ["analytic"],
            "profiling_database": None,
        }
        tasks = build_text_generate_tasks(form)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.sim_type == "text_generate"
        assert task.params["model_id"] == "Qwen/Qwen3-32B"
        assert task.params["device"] == "D1"
        assert task.params["num_queries"] == 32
        assert "--num-queries" in task.command
        assert "--compile" in task.command

    def test_build_text_task_with_sweep(self) -> None:
        """Test building tasks with num_queries sweep."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 4,
            "num_queries": "32",
            "num_queries_sweep": "[16,32,64]",
            "query_length": 8,
            "context_length": 4500,
            "decode": True,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "",
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "DISABLED",
            "quant_attention_sweep": None,
            "tp_size": 1,
            "tp_sweep": None,
            "dp_size": "auto",
            "ep_size": 1,
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "prefix_cache_hit_rate": 0.0,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "compile_allow_graph_break": False,
            "disable_repetition": False,
            "quantize_lmhead": False,
            "mxfp4_group_size": 32,
            "graph_log_url": None,
            "dump_input_shapes": False,
            "chrome_trace": None,
            "num_hidden_layers_override": 0,
            "o_proj_tp_size": None,
            "o_proj_dp_size": None,
            "mlp_tp_size": None,
            "mlp_dp_size": None,
            "lmhead_tp_size": None,
            "lmhead_dp_size": None,
            "moe_tp_size": None,
            "moe_dp_size": "1",
            "word_embedding_tp": None,
            "enable_redundant_experts": False,
            "enable_external_shared_experts": False,
            "host_external_shared_experts": False,
            "remote_source": "huggingface",
            "performance_model": ["analytic"],
            "profiling_database": None,
        }
        tasks = build_text_generate_tasks(form)
        assert len(tasks) == 3
        assert {t.params["num_queries"] for t in tasks} == {16, 32, 64}

    def test_build_text_task_with_multiple_devices(self) -> None:
        """Test building tasks with competitor devices."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": ["D2", "D3"],
            "num_devices": 4,
            "num_queries": 32,
            "num_queries_sweep": None,
            "query_length": 8,
            "context_length": 4500,
            "decode": True,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "",
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "DISABLED",
            "quant_attention_sweep": None,
            "tp_size": 1,
            "tp_sweep": None,
            "dp_size": "auto",
            "ep_size": 1,
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "prefix_cache_hit_rate": 0.0,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "compile_allow_graph_break": False,
            "disable_repetition": False,
            "quantize_lmhead": False,
            "mxfp4_group_size": 32,
            "graph_log_url": None,
            "dump_input_shapes": False,
            "chrome_trace": None,
            "num_hidden_layers_override": 0,
            "o_proj_tp_size": None,
            "o_proj_dp_size": None,
            "mlp_tp_size": None,
            "mlp_dp_size": None,
            "lmhead_tp_size": None,
            "lmhead_dp_size": None,
            "moe_tp_size": None,
            "moe_dp_size": "1",
            "word_embedding_tp": None,
            "enable_redundant_experts": False,
            "enable_external_shared_experts": False,
            "host_external_shared_experts": False,
            "remote_source": "huggingface",
            "performance_model": ["analytic"],
            "profiling_database": None,
        }
        tasks = build_text_generate_tasks(form)
        assert len(tasks) == 3
        devices = {t.params["device"] for t in tasks}
        assert devices == {"D1", "D2", "D3"}

    def test_build_text_task_with_mtp(self) -> None:
        """Test building task with MTP tokens."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 4,
            "num_queries": 32,
            "num_queries_sweep": None,
            "query_length": 8,
            "context_length": 4500,
            "decode": True,
            "num_mtp_tokens": 5,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "DISABLED",
            "quant_attention_sweep": None,
            "tp_size": 1,
            "tp_sweep": None,
            "dp_size": "auto",
            "ep_size": 1,
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "prefix_cache_hit_rate": 0.0,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "compile_allow_graph_break": False,
            "disable_repetition": False,
            "quantize_lmhead": False,
            "mxfp4_group_size": 32,
            "graph_log_url": None,
            "dump_input_shapes": False,
            "chrome_trace": None,
            "num_hidden_layers_override": 0,
            "o_proj_tp_size": None,
            "o_proj_dp_size": None,
            "mlp_tp_size": None,
            "mlp_dp_size": None,
            "lmhead_tp_size": None,
            "lmhead_dp_size": None,
            "moe_tp_size": None,
            "moe_dp_size": "1",
            "word_embedding_tp": None,
            "enable_redundant_experts": False,
            "enable_external_shared_experts": False,
            "host_external_shared_experts": False,
            "remote_source": "huggingface",
            "performance_model": ["analytic"],
            "profiling_database": None,
        }
        tasks = build_text_generate_tasks(form)
        assert len(tasks) == 1
        assert "--num-mtp-tokens" in tasks[0].command
        assert "5" in tasks[0].command
        assert "--mtp-acceptance-rate" in tasks[0].command

    def test_build_text_task_with_vl_params(self) -> None:
        """Test building task with VL image parameters."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 4,
            "num_queries": 32,
            "num_queries_sweep": None,
            "query_length": 8,
            "context_length": 4500,
            "decode": True,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "",
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "DISABLED",
            "quant_attention_sweep": None,
            "tp_size": 1,
            "tp_sweep": None,
            "dp_size": "auto",
            "ep_size": 1,
            "image_batch_size": 2,
            "image_height": 1024,
            "image_width": 1024,
            "prefix_cache_hit_rate": 0.0,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "compile_allow_graph_break": False,
            "disable_repetition": False,
            "quantize_lmhead": False,
            "mxfp4_group_size": 32,
            "graph_log_url": None,
            "dump_input_shapes": False,
            "chrome_trace": None,
            "num_hidden_layers_override": 0,
            "o_proj_tp_size": None,
            "o_proj_dp_size": None,
            "mlp_tp_size": None,
            "mlp_dp_size": None,
            "lmhead_tp_size": None,
            "lmhead_dp_size": None,
            "moe_tp_size": None,
            "moe_dp_size": "1",
            "word_embedding_tp": None,
            "enable_redundant_experts": False,
            "enable_external_shared_experts": False,
            "host_external_shared_experts": False,
            "remote_source": "huggingface",
            "performance_model": ["analytic"],
            "profiling_database": None,
        }
        tasks = build_text_generate_tasks(form)
        assert len(tasks) == 1
        assert "--image-batch-size" in tasks[0].command
        assert "--image-height" in tasks[0].command
        assert "--image-width" in tasks[0].command

    def test_build_text_task_task_hash_format(self) -> None:
        """Test that task hash is a 16-character string."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 4,
            "num_queries": 32,
            "num_queries_sweep": None,
            "query_length": 8,
            "context_length": 4500,
            "decode": True,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "",
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "DISABLED",
            "quant_attention_sweep": None,
            "tp_size": 1,
            "tp_sweep": None,
            "dp_size": "auto",
            "ep_size": 1,
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "prefix_cache_hit_rate": 0.0,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "compile_allow_graph_break": False,
            "disable_repetition": False,
            "quantize_lmhead": False,
            "mxfp4_group_size": 32,
            "graph_log_url": None,
            "dump_input_shapes": False,
            "chrome_trace": None,
            "num_hidden_layers_override": 0,
            "o_proj_tp_size": None,
            "o_proj_dp_size": None,
            "mlp_tp_size": None,
            "mlp_dp_size": None,
            "lmhead_tp_size": None,
            "lmhead_dp_size": None,
            "moe_tp_size": None,
            "moe_dp_size": "1",
            "word_embedding_tp": None,
            "enable_redundant_experts": False,
            "enable_external_shared_experts": False,
            "host_external_shared_experts": False,
            "remote_source": "huggingface",
            "performance_model": ["analytic"],
            "profiling_database": None,
        }
        tasks = build_text_generate_tasks(form)
        assert len(tasks[0].task_hash) == 16


class TestBuildVideoGenerateTasks:
    """Tests for build_video_generate_tasks function."""

    def _basic_video_form(self, remote_source: str | None = None) -> dict[str, object]:
        form: dict[str, object] = {
            "model_id": "Wan2.2-T2V-A14B-Diffusers",
            "device": "D1",
            "competitor_devices": [],
            "batch_size": 1,
            "seq_len": 128,
            "height": 1280,
            "width": 720,
            "frame_num": 129,
            "sample_step": 50,
            "dtype": "float16",
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
            "use_cfg": True,
            "cfg_parallel": True,
            "dit_cache": False,
            "cache_step_range": None,
            "cache_step_interval": None,
            "cache_block_range": None,
            "chrome_trace": None,
            "log_level": "info",
        }
        if remote_source is not None:
            form["remote_source"] = remote_source
        return form

    def test_build_video_task_defaults_to_huggingface_without_flag(self) -> None:
        tasks = build_video_generate_tasks(self._basic_video_form())

        assert len(tasks) == 1
        task = tasks[0]
        assert task.params["remote_source"] == "huggingface"
        assert "--remote-source" not in task.command

    def test_build_video_task_adds_modelscope_remote_source_flag(self) -> None:
        tasks = build_video_generate_tasks(self._basic_video_form(remote_source="modelscope"))

        assert len(tasks) == 1
        task = tasks[0]
        assert task.params["remote_source"] == "modelscope"
        assert "--remote-source" in task.command
        idx = task.command.index("--remote-source")
        assert task.command[idx + 1] == "modelscope"

    def test_build_basic_video_task(self) -> None:
        """Test building a basic video generation task."""
        form = {
            "model_id": "Wan2.2-T2V-A14B-Diffusers",
            "device": "D1",
            "competitor_devices": [],
            "batch_size": 1,
            "seq_len": 128,
            "height": 1280,
            "width": 720,
            "frame_num": 129,
            "sample_step": 50,
            "dtype": "float16",
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
            "use_cfg": True,
            "cfg_parallel": True,
            "dit_cache": False,
            "cache_step_range": None,
            "cache_step_interval": None,
            "cache_block_range": None,
            "chrome_trace": None,
            "log_level": "info",
        }
        tasks = build_video_generate_tasks(form)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.sim_type == "video_generate"
        assert task.params["model_id"] == "Wan2.2-T2V-A14B-Diffusers"
        assert task.params["device"] == "D1"
        assert "--batch-size" in task.command
        assert "--use-cfg" in task.command
        assert "--cfg-parallel" in task.command

    def test_build_video_task_with_dit_cache(self) -> None:
        """Test building video task with DiT cache enabled."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "batch_size": 1,
            "seq_len": 128,
            "height": 720,
            "width": 480,
            "frame_num": 64,
            "sample_step": 30,
            "dtype": "float16",
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": None,
            "use_cfg": False,
            "cfg_parallel": False,
            "dit_cache": True,
            "cache_step_range": "20,30",
            "cache_step_interval": 5,
            "cache_block_range": "0,50",
            "chrome_trace": None,
            "log_level": "info",
        }
        tasks = build_video_generate_tasks(form)
        assert len(tasks) == 1
        assert "--dit-cache" in tasks[0].command
        assert "--cache-step-range" in tasks[0].command
        assert "--cache-step-interval" in tasks[0].command
        assert "--cache-block-range" in tasks[0].command

    def test_build_video_task_with_ulysses_sweep(self) -> None:
        """Test building video tasks with Ulysses sweep."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "batch_size": 1,
            "seq_len": 128,
            "height": 720,
            "width": 480,
            "frame_num": 64,
            "sample_step": 30,
            "dtype": "float16",
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "world_size": 8,
            "ulysses_size": 4,
            "ulysses_sweep": "[2,4,8]",
            "use_cfg": False,
            "cfg_parallel": False,
            "dit_cache": False,
            "cache_step_range": None,
            "cache_step_interval": None,
            "cache_block_range": None,
            "chrome_trace": None,
            "log_level": "info",
        }
        tasks = build_video_generate_tasks(form)
        assert len(tasks) == 3
        assert {t.params["ulysses_size"] for t in tasks} == {2, 4, 8}


class TestBuildOptimizerTasks:
    """Tests for build_optimizer_tasks function."""

    def test_build_basic_optimizer_task(self) -> None:
        """Test building a basic optimizer task."""
        form = {
            "model_id": "Qwen/Qwen3-32B",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 8,
            "input_length": 3500,
            "output_length": 1500,
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "INT8",
            "quant_attention_sweep": None,
            "tpot_limits": None,
            "tpot_sweep": None,
            "ttft_limits": None,
            "ttft_sweep": None,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "max_prefill_tokens": 8192,
            "image_height": None,
            "image_width": None,
            "tp_sizes": "",
            "batch_range": "[1,256]",
            "jobs": 8,
            "deployment_mode": "PD Aggregated",
            "prefix_cache_hit_rate": 0.0,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
            "compile_allow_graph_break": False,
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "serving_cost": 0.0,
            "dump_original_results": False,
        }
        tasks = build_optimizer_tasks(form)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.sim_type == "throughput_optimizer"
        assert task.params["deployment_mode"] == "PD Aggregated"
        assert "--input-length" in task.command
        assert "--output-length" in task.command
        assert "--batch-range" in task.command

    def test_build_optimizer_task_with_limits(self) -> None:
        """Test building optimizer task with latency limits."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 8,
            "input_length": 1000,
            "output_length": 500,
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "INT8",
            "quant_attention_sweep": None,
            "tpot_limits": 50.0,
            "tpot_sweep": None,
            "ttft_limits": 2000.0,
            "ttft_sweep": None,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "max_prefill_tokens": 8192,
            "image_height": None,
            "image_width": None,
            "tp_sizes": "",
            "batch_range": "[1,128]",
            "jobs": 8,
            "deployment_mode": "PD Aggregated",
            "prefix_cache_hit_rate": 0.0,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
            "compile_allow_graph_break": False,
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "serving_cost": 0.0,
            "dump_original_results": False,
        }
        tasks = build_optimizer_tasks(form)
        assert len(tasks) == 1
        assert "--tpot-limits" in tasks[0].command
        assert "--ttft-limits" in tasks[0].command

    def test_build_optimizer_task_pd_ratio_mode(self) -> None:
        """Test building optimizer task in PD Ratio mode."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 8,
            "input_length": 1000,
            "output_length": 500,
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "INT8",
            "quant_attention_sweep": None,
            "tpot_limits": None,
            "tpot_sweep": None,
            "ttft_limits": None,
            "ttft_sweep": None,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "max_prefill_tokens": 8192,
            "image_height": None,
            "image_width": None,
            "tp_sizes": "",
            "batch_range": "[1,64]",
            "jobs": 8,
            "deployment_mode": "PD Ratio",
            "prefix_cache_hit_rate": 0.0,
            "prefill_devices_per_instance": 2,
            "decode_devices_per_instance": 6,
            "compile_allow_graph_break": False,
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "serving_cost": 0.0,
            "dump_original_results": False,
        }
        tasks = build_optimizer_tasks(form)
        assert len(tasks) == 1
        assert "--enable-optimize-prefill-decode-ratio" in tasks[0].command
        assert "--prefill-devices-per-instance" in tasks[0].command
        assert "--decode-devices-per-instance" in tasks[0].command

    def test_build_optimizer_with_mtp(self) -> None:
        """Test building optimizer task with MTP tokens."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 8,
            "input_length": 500,
            "output_length": 500,
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "INT8",
            "quant_attention_sweep": None,
            "tpot_limits": None,
            "tpot_sweep": None,
            "ttft_limits": None,
            "ttft_sweep": None,
            "num_mtp_tokens": 5,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "max_prefill_tokens": 8192,
            "image_height": None,
            "image_width": None,
            "tp_sizes": "",
            "batch_range": "[1,128]",
            "jobs": 8,
            "deployment_mode": "PD Aggregated",
            "prefix_cache_hit_rate": 0.0,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
            "compile_allow_graph_break": False,
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "serving_cost": 0.0,
            "dump_original_results": False,
        }
        tasks = build_optimizer_tasks(form)
        assert len(tasks) == 1
        assert "--num-mtp-tokens" in tasks[0].command
        assert "--mtp-acceptance-rate" in tasks[0].command

    def test_build_optimizer_optimization_mode_offline(self) -> None:
        """Test optimization mode is offline when no limits."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 8,
            "input_length": 1000,
            "output_length": 500,
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "INT8",
            "quant_attention_sweep": None,
            "tpot_limits": None,
            "tpot_sweep": None,
            "ttft_limits": None,
            "ttft_sweep": None,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "max_prefill_tokens": 8192,
            "image_height": None,
            "image_width": None,
            "tp_sizes": "",
            "batch_range": "[1,256]",
            "jobs": 8,
            "deployment_mode": "PD Aggregated",
            "prefix_cache_hit_rate": 0.0,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
            "compile_allow_graph_break": False,
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "serving_cost": 0.0,
            "dump_original_results": False,
        }
        tasks = build_optimizer_tasks(form)
        assert tasks[0].params["optimization_mode"] == "offline"

    def test_build_optimizer_optimization_mode_ttft_only(self) -> None:
        """Test optimization mode with TTFT only."""
        form = {
            "model_id": "test_model",
            "device": "D1",
            "competitor_devices": [],
            "num_devices": 8,
            "input_length": 1000,
            "output_length": 500,
            "compile": True,
            "quantize_linear_action": "W8A8_DYNAMIC",
            "quant_linear_sweep": None,
            "quantize_attention_action": "INT8",
            "quant_attention_sweep": None,
            "tpot_limits": None,
            "tpot_sweep": None,
            "ttft_limits": 2000.0,
            "ttft_sweep": None,
            "num_mtp_tokens": 0,
            "mtp_acceptance_rate": "0.9,0.6,0.4,0.2",
            "max_prefill_tokens": 8192,
            "image_height": None,
            "image_width": None,
            "tp_sizes": "",
            "batch_range": "[1,256]",
            "jobs": 8,
            "deployment_mode": "PD Aggregated",
            "prefix_cache_hit_rate": 0.0,
            "prefill_devices_per_instance": None,
            "decode_devices_per_instance": None,
            "compile_allow_graph_break": False,
            "mxfp4_group_size": 32,
            "reserved_memory_gb": 0.0,
            "log_level": "error",
            "serving_cost": 0.0,
            "dump_original_results": False,
        }
        tasks = build_optimizer_tasks(form)
        assert tasks[0].params["optimization_mode"] == "ttft_constrained"
