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

"""Ownership: shared-precheck (test). Unit tests for collect_context.py.

验证 Round 0 的环境事实与确认负载采集契约，不生成搜索空间。
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("collect_context.py")


def load_module():
    spec = importlib.util.spec_from_file_location("collect_context", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", ["vllm", "mindie"])
def test_collect_facts_without_planning_or_recommendation_artifacts(tmp_path, engine):
    config = tmp_path / "config.toml"
    config.write_text(
        f'optimizer_strategy = "agent"\n[{engine}]\n'
        f'[[{engine}.target_field]]\nname="tp"\ndtype="int"\nmin=1\nmax=8\nvalue=8\nconfig_position="run"\n'
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "qwen3_moe",
                    "num_experts": 64,
                    "num_hidden_layers": 48,
                    "num_nextn_predict_layers": 1,
                }
            }
        )
    )
    run = tmp_path / "run"
    before = config.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run),
            "--config-toml",
            str(config),
            "--engine",
            engine,
            "--world-size",
            "8",
            "--model-path",
            str(model),
            "--input-len",
            "1024",
            "--output-len",
            "256",
            "--num-requests",
            "100",
            "--concurrency-min",
            "1",
            "--concurrency-max",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    context = json.loads((run / "context.json").read_text())
    assert not {"search_space", "knowledge", "planning", "config_handoff"} & context.keys()
    assert context["source"]["config_toml"] == str(config.resolve())
    assert context["model_info"]["is_moe"] is True
    assert context["model_info"]["mtp_num_hidden_layers"] == 1
    assert context["benchmark_spec_confirmed"] is True
    assert context["benchmark_spec"]["concurrency_range"] == [1, 8]
    assert config.read_bytes() == before
    assert {p.name for p in run.iterdir()} == {"context.json"}
    from check_search_space import check_parameters

    assert not [issue for issue in check_parameters(context) if issue["level"] == "error"]
    assert "search_space" not in context


def test_vendor_preset_written_when_model_matches(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'optimizer_strategy = "pso"\n[vllm]\n'
        '[[vllm.target_field]]\nname="tp"\ndtype="int"\nmin=1\nmax=8\nvalue=8\nconfig_position="run"\n'
    )
    run = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run),
            "--config-toml",
            str(config),
            "--engine",
            "vllm",
            "--world-size",
            "8",
            "--model-name",
            "MiniMax-M2",
            "--input-len",
            "3500",
            "--output-len",
            "1500",
            "--num-requests",
            "100",
            "--concurrency",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    context = json.loads((run / "context.json").read_text())
    vendor_preset = context["knowledge"]["vendor_preset"]
    assert vendor_preset["model"] == "MiniMax-M2"
    assert vendor_preset["scenario_id"]
    for key in ("params", "env", "additional_config"):
        assert isinstance(vendor_preset[key], dict) and vendor_preset[key]
    assert vendor_preset["hardware"]["actual_world_size"] == 8
    assert "search_space" not in context


def test_missing_config_fails_without_writing_context(tmp_path):
    run = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run),
            "--engine",
            "vllm",
            "--world-size",
            "8",
            "--config-toml",
            str(tmp_path / "missing.toml"),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert not (run / "context.json").exists()


def test_draft_path_is_not_reclassified_as_eagle(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_config": {"speculative_model": "draft"}}))
    info = load_module()._read_model_info(tmp_path)
    assert info["speculative_model_path"] == "draft"
    assert not info["eagle_head_path"]
