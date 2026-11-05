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


import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]


def collect(tmp_path, model="model"):
    config = tmp_path / "base.toml"
    if not config.exists():
        config.write_text(
            '''[vllm.command]
model="/models/model"
served_model_name="model"
[[vllm_pd.nodes]]
role="prefill"
ssh_ip="p-host"
gpu_ids=[0,1]
[[vllm_pd.nodes]]
role="decode"
ssh_ip="d-host"
gpu_ids=[0,1]
''',
            encoding="utf-8",
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "contrib/optix/vllm_pd_simulator") + ":" + str(ROOT)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / ".agents/skills/optix-assistant/scripts/collect_context.py"),
            "--run-dir",
            str(tmp_path / "run"),
            "--config-toml",
            str(config),
            "--engine",
            "vllm",
            "--deployment",
            "pd",
            "--model-name",
            model,
            "--input-len",
            "128",
            "--output-len",
            "64",
            "--num-requests",
            "100",
            "--concurrency",
            "4",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_recollection_preserves_agent_space_and_fixed_target(tmp_path):
    first = collect(tmp_path)
    assert first.returncode == 0, first.stderr
    path = tmp_path / "run/context.json"
    context = json.loads(path.read_text(encoding="utf-8"))
    context["search_space"] = {"parameters": [], "constants": []}
    context["target"] = "throughput"
    path.write_text(json.dumps(context), encoding="utf-8")
    second = collect(tmp_path)
    assert second.returncode == 0, second.stderr
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["search_space"] == context["search_space"]
    assert after["target"] == "throughput"


def test_model_change_rejected_without_overwriting_context(tmp_path):
    assert collect(tmp_path).returncode == 0
    path = tmp_path / "run/context.json"
    before = path.read_bytes()
    rejected = collect(tmp_path, "other-model")
    assert rejected.returncode != 0
    assert "model differs" in rejected.stderr
    assert path.read_bytes() == before
