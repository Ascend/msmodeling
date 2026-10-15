"""Ownership: agent (test). Unit tests for config_writer.py."""

import importlib.util
import sys
from pathlib import Path

import tomlkit


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "config_writer.py"


def load_module():
    # config_writer 以 `from common import ...` 引用同目录模块，需先把 scripts 目录挂上 sys.path
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("config_writer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, *set_args):
    module = load_module()
    config = tmp_path / "config.toml"
    config.write_text("[vllm]\n")
    argv = ["--config", str(config)]
    for kv in set_args:
        argv += ["--set", kv]
    assert module.main(argv) == 0
    return tomlkit.parse(config.read_text())


def test_env_numeric_value_stays_string(tmp_path):
    # 回归现场：env 段 schema 是 dict[str, str]，--set vllm.env.HCCL_BUFFSIZE=512
    # 曾被 coerce 数字推断写成 int，pydantic 校验拒绝；env 叶子必须 force_str。
    doc = _run(tmp_path, "vllm.env.HCCL_BUFFSIZE=512")
    assert doc["vllm"]["env"]["HCCL_BUFFSIZE"] == "512"
    assert isinstance(doc["vllm"]["env"]["HCCL_BUFFSIZE"], str)


def test_env_embedded_quotes_are_stripped(tmp_path):
    # 回归现场：为对抗数字推断手工加引号（'"512"'）曾把引号嵌入值，HCCL
    # StrToULong 解析崩溃；force_str 路径须剥掉成对包围引号。
    doc = _run(tmp_path, 'vllm.env.HCCL_BUFFSIZE="512"')
    assert doc["vllm"]["env"]["HCCL_BUFFSIZE"] == "512"


def test_non_env_numeric_inference_unchanged(tmp_path):
    # 非 env 路径的数字推断行为不变：command 普通字段仍转 int；剥引号是先于
    # 推断的输入归一化，带引号的数字在推断路径上仍应转成数字。
    doc = _run(tmp_path, "vllm.command.max_num_seqs=32", 'vllm.command.gpu_memory_utilization="0.9"')
    assert doc["vllm"]["command"]["max_num_seqs"] == 32
    assert doc["vllm"]["command"]["gpu_memory_utilization"] == 0.9


def test_string_command_field_unchanged(tmp_path):
    # _STRING_COMMAND_FIELDS（port）保持字符串语义不变。
    doc = _run(tmp_path, "vllm.command.port=8089")
    assert doc["vllm"]["command"]["port"] == "8089"
    assert isinstance(doc["vllm"]["command"]["port"], str)
