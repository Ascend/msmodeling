"""Ownership: pso (test). Unit tests for recommend_params.py."""

import importlib.util
import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path


SCRIPT = Path(__file__).with_name("recommend_params.py")
#: 本测试位于 <root>/.agents/skills/optix-assistant/scripts/，parents[4] 才是仓库根
REPO_ROOT = Path(__file__).parents[4]
CONFIG_SKILL_SCRIPT_PATH = ".agents/skills/optix-assistant/scripts/config_writer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("recommend_params", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_model_config(tmp_path):
    model_config = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768,
        "vocab_size": 151936,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config), encoding="utf-8")
    return config_path


def base_context(tmp_path, engine):
    return {
        "engine": engine,
        "hardware": {
            "single_card_mem_gb": 64,
            "world_size": 8,
            "num_per_nodes": 8,
            "num_nodes": 1,
        },
        "model": {
            "config_path": str(write_model_config(tmp_path)),
        },
        "workload": {
            "input_len_avg": 1024,
            "input_len_max": 4096,
            "output_len_avg": 256,
            "output_len_max": 512,
        },
        "target": "balanced",
    }


def write_minimal_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        """
[vllm.command]
others = ""

[mindie]
""".lstrip(),
        encoding="utf-8",
    )
    return config


def write_nested_model_config(tmp_path, model_type="qwen3_5"):
    """Write a config with nested text_config structure like Qwen3.5, Qwen3-VL."""
    model_config = {
        "architectures": [f"{model_type}_for_conditional_generation"],
        "model_type": model_type,
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "dtype": "bfloat16",
            "max_position_embeddings": 262144,
            "vocab_size": 248320,
        },
        "vision_config": {
            "hidden_size": 1152,
            "depth": 27,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config), encoding="utf-8")
    return config_path


def assert_handoff_commands_parse(handoff, tmp_path):
    """Verify handoff commands run through config_writer (skip on encoding issues)."""
    config_script = REPO_ROOT / CONFIG_SKILL_SCRIPT_PATH
    if not config_script.exists():
        return
    try:
        config_path = write_minimal_config(tmp_path)
        for command in handoff["apply_commands"]:
            args = shlex.split(command)
            # 生成命令以可移植的 "python" 开头，本机可能只有 python3：换当前解释器执行
            if args and args[0] == "python":
                args[0] = sys.executable
            args += ["--dry-run", "--config", str(config_path)]
            result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, encoding="utf-8")
            # Skip if subprocess fails due to encoding (Windows GBK limitation)
            if result.returncode != 0 and "UnicodeEncodeError" not in result.stderr:
                assert result.returncode == 0, result.stderr + result.stdout
    except UnicodeEncodeError:
        pass  # Skip on Windows encoding limitations


def test_missing_required_fields_returns_need_more_info():
    module = load_module()

    result = module.recommend({"engine": "vllm"})

    assert result["status"] == "need_more_info"
    assert "hardware.single_card_mem_gb" in result["missing_fields"]
    assert result["next_question"]


def test_vllm_recommendation_defaults_to_ais_bench_and_parallel_constraint(tmp_path):
    module = load_module()

    result = module.recommend(base_context(tmp_path, "vllm"))

    assert result["status"] == "ok"
    assert result["benchmark_policy"] == "ais_bench"
    assert "DP * TP * PP == world_size" in result["constraints"][0]["expression"]
    items = {item["name"]: item for item in result["recommendations"]}
    names = set(items)
    assert {"MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS", "TENSOR_PARALLEL_SIZE", "PIPELINE_PARALLEL_SIZE"} <= names
    assert {"ENABLE_PREFIX_CACHING", "ENABLE_CHUNKED_PREFILL", "COMPILATION_CONFIG", "BLOCK_SIZE"} <= names
    assert items["ENABLE_PREFIX_CACHING"]["value"] == "vllm_default"
    assert items["ENABLE_PREFIX_CACHING"]["search"] is False
    assert items["ENABLE_CHUNKED_PREFILL"]["value"] == "vllm_default"
    assert items["ENABLE_CHUNKED_PREFILL"]["search"] is False
    assert "[ais_bench.command]" in result["toml_snippet"]
    assert "[[ais_bench.target_field]]" not in result["toml_snippet"]
    parsed = tomllib.loads(result["toml_snippet"])
    vllm_fields = {item["name"]: item for item in parsed["vllm"]["target_field"]}
    assert "ENABLE_PREFIX_CACHING" not in vllm_fields
    assert "ENABLE_CHUNKED_PREFILL" not in vllm_fields
    handoff = result["config_skill_handoff"]
    assert handoff["consumer_skill"] == "optix-assistant"
    assert handoff["handoff_type"] == "target_fields_and_commands"
    assert not any(field["name"] == "ENABLE_PREFIX_CACHING" for field in handoff["target_fields"])
    assert any("--target-field" in command for command in handoff["apply_commands"])
    # --cli-arg 冗余已删除：写命令只 upsert target_field 块
    assert not any("--cli-arg" in command for command in handoff["apply_commands"])
    assert not any("$TENSOR_PARALLEL_SIZE" in command for command in handoff["apply_commands"])
    # ais_bench 段 scheduler env（CONCURRENCY/REQUESTRATE）不生成写命令，保留在 target_fields JSON
    assert not any("engine=ais_bench" in command for command in handoff["apply_commands"])
    assert any(field["name"] == "CONCURRENCY" for field in handoff["target_fields"])
    # #825 起 vLLM 服务参数是 run 字段：others 生成 `--flag $NAME` 占位符兜底，
    # 真值由字段模型渲染成同名 --flag，经渲染层 last-wins 去重压制占位符
    # （$NAME↔target_field 一致性由 config_writer 的协议校验兜底）。
    others = handoff["vllm_command_others"]
    assert "--max-model-len $MAX_MODEL_LEN" in others
    assert "--tensor-parallel-size $TENSOR_PARALLEL_SIZE" in others
    assert "--gpu-memory-utilization $GPU_MEMORY_UTILIZATION" in others
    assert "$COMPILATION_CONFIG" in others
    # 容量字段走 BUILTIN_COMMAND_FIELDS 例外，不生成占位符（由字段模型渲染）
    assert "$MAX_NUM_SEQS" not in others
    assert "$MAX_NUM_BATCHED_TOKENS" not in others
    # benchmark 负载参数属 env 位置，绝不进 serve others
    assert "$CONCURRENCY" not in others
    assert "$REQUESTRATE" not in others
    assert all(field["config_position"] == "run" for field in handoff["target_fields"] if field["section"] == "vllm")
    assert_handoff_commands_parse(handoff, tmp_path)


def test_mindie_recommendation_includes_batch_fields_and_ais_bench(tmp_path):
    module = load_module()

    result = module.recommend(base_context(tmp_path, "mindie"))

    assert result["status"] == "ok"
    assert result["benchmark_policy"] == "ais_bench"
    names = {item["name"] for item in result["recommendations"]}
    assert {"max_batch_size", "max_prefill_batch_size", "CONCURRENCY", "REQUESTRATE"} <= names
    assert {"max_preempt_count", "prefill_policy_type", "decode_policy_type"} <= names
    assert "[[mindie.target_field]]" in result["toml_snippet"]
    assert (
        "config_position=BackendConfig.ScheduleConfig.maxBatchSize"
        in result["config_skill_handoff"]["apply_commands"][0]
    )
    assert_handoff_commands_parse(result["config_skill_handoff"], tmp_path)
    tomllib.loads(result["toml_snippet"])


def test_mindie_moe_model_includes_moe_parallel_fields(tmp_path):
    module = load_module()
    context = base_context(tmp_path, "mindie")
    context["model"]["is_moe"] = True

    result = module.recommend(context)

    assert result["status"] == "ok"
    names = {item["name"] for item in result["recommendations"]}
    assert {"moe_ep", "moe_tp"} <= names
    tomllib.loads(result["toml_snippet"])


def test_vllm_help_discovery_adds_relevant_optional_parameters(tmp_path):
    module = load_module()
    context = base_context(tmp_path, "vllm")
    context["model"]["is_multimodal"] = True
    context["model"]["is_moe"] = True
    context["workload"]["input_len_max"] = 8192
    context["discovery"] = {
        "enabled": True,
        "vllm_help_text": """
        --max-num-partial-prefills INTEGER
        --long-prefill-token-threshold INTEGER
        --disable-chunked-mm-input
        --enable-expert-parallel
        """,
    }

    result = module.recommend(context)

    assert result["status"] == "ok"
    items = {item["name"]: item for item in result["recommendations"]}
    assert {"MAX_NUM_PARTIAL_PREFILLS", "LONG_PREFILL_TOKEN_THRESHOLD"} <= set(items)
    assert {"DISABLE_CHUNKED_MM_INPUT", "ENABLE_EXPERT_PARALLEL"} <= set(items)
    assert items["DISABLE_CHUNKED_MM_INPUT"]["search"] is False
    assert items["ENABLE_EXPERT_PARALLEL"]["search"] is False
    assert result["discovery"]["enabled"] is True
    assert len(result["discovery"]["added_parameters"]) == 4
    assert "--enable-expert-parallel" in result["config_skill_handoff"]["vllm_command_others"]
    assert "$ENABLE_EXPERT_PARALLEL" not in result["config_skill_handoff"]["vllm_command_others"]
    assert_handoff_commands_parse(result["config_skill_handoff"], tmp_path)
    tomllib.loads(result["toml_snippet"])


def test_nested_text_config_loaded_correctly(tmp_path):
    """Test that configs with nested text_config (Qwen3.5, Qwen3-VL, Kimi) are parsed correctly."""
    module = load_module()
    config_path = write_nested_model_config(tmp_path, "qwen3_5")
    context = {
        "engine": "vllm",
        "hardware": {
            "single_card_mem_gb": 64,
            "world_size": 8,
            "num_per_nodes": 8,
            "num_nodes": 1,
        },
        "model": {
            "config_path": str(config_path),
        },
        "workload": {
            "input_len_avg": 1024,
            "input_len_max": 4096,
            "output_len_avg": 256,
            "output_len_max": 512,
        },
        "target": "balanced",
    }

    result = module.recommend(context)

    assert result["status"] == "ok"
    items = {item["name"]: item for item in result["recommendations"]}
    # Verify nested fields were loaded (24 heads divisible by TP candidates)
    assert items["TENSOR_PARALLEL_SIZE"]["dtype"] == "enum"
    assert items["MAX_MODEL_LEN"]["value"] == 4608  # input_len_max(4096) + output_len_max(512)
