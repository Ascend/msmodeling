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


import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
import tomlkit
from optix.optimizer.agentic.pd import configuration_fingerprint, load_plugin_config, device_inventory, resource_nodes


ROOT = Path(__file__).resolve().parents[4]

spec = importlib.util.spec_from_file_location(
    "generate_pd", ROOT / ".agents/skills/optix-assistant/scripts/generate_pd_config.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def experiment(tmp_path):
    document = {
        "vllm": {"command": {"model": "/models/model", "served_model_name": "model"}},
        "vllm_pd": {
            "nodes": [
                {"role": "prefill", "ssh_ip": "a", "gpu_ids": [0, 1]},
                {"role": "decode", "ssh_ip": "b", "gpu_ids": [0, 1]},
            ]
        },
    }
    base = tmp_path / "base.toml"
    base.write_text(tomlkit.dumps(document), encoding="utf-8")
    config = load_plugin_config(base)
    context = {
        "engine": "vllm",
        "source": {"config_toml": str(base)},
        "deployment": {
            "type": "pd",
            "config_sha256": configuration_fingerprint(base),
            "plugin_config": config.model_dump(mode="json"),
            "device_inventory": [list(device) for device in sorted(device_inventory(resource_nodes(config)))],
        },
        "constraints": {"world_size": 4, "expressions": []},
        "search_space": {
            "parameters": [
                {
                    "name": "TENSOR_PARALLEL_SIZE_prefill",
                    "dtype": "int",
                    "min": 1,
                    "max": 2,
                    "default": 1,
                    "config_position": "run",
                },
                {
                    "name": "TENSOR_PARALLEL_SIZE_decode",
                    "dtype": "int",
                    "min": 1,
                    "max": 2,
                    "default": 1,
                    "config_position": "run",
                },
            ]
        },
    }
    return context


def test_same_inventory_can_change_pd_ratio(tmp_path):
    original = experiment(tmp_path)
    proposal = {
        "candidate_id": "r1",
        "params": {},
        "deployment": {
            "nodes": [
                {"source_node": 0, "role": "prefill", "gpu_ids": [0, 1]},
                {"source_node": 1, "role": "prefill", "gpu_ids": [0]},
                {"source_node": 1, "role": "decode", "gpu_ids": [1]},
            ]
        },
    }
    result = generator.generate_config(original, original, proposal, tmp_path / "round-1.toml")
    assert result["layout"]["roles"]["prefill"]["devices"] == 3
    assert result["layout"]["roles"]["decode"]["devices"] == 1
    generated = load_plugin_config(tmp_path / "round-1.toml")
    assert len(device_inventory(resource_nodes(generated))) == 4
    assert configuration_fingerprint(original["source"]["config_toml"]) == original["deployment"]["config_sha256"]


def test_out_of_inventory_device_rejected_without_writing(tmp_path):
    original = experiment(tmp_path)
    proposal = {"params": {}, "deployment": {"nodes": [{"source_node": 0, "role": "prefill", "gpu_ids": [7]}]}}
    with pytest.raises(ValueError, match="inventory"):
        generator.generate_config(original, original, proposal, tmp_path / "bad.toml")
    assert not (tmp_path / "bad.toml").exists()


def test_changed_base_rejected(tmp_path):
    original = experiment(tmp_path)
    Path(original["source"]["config_toml"]).write_text("changed = true", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        generator.generate_config(original, original, {}, tmp_path / "bad.toml")


def test_next_round_does_not_inherit_previous_candidate_binding(tmp_path):
    original = experiment(tmp_path)
    current = deepcopy(original)
    current["deployment"]["round_manifest"] = {"candidate_id": "previous", "params": {}}
    proposal = {
        "candidate_id": "next",
        "params": {},
        "deployment": {
            "nodes": [
                {"source_node": 0, "role": "prefill", "gpu_ids": [0, 1]},
                {"source_node": 1, "role": "decode", "gpu_ids": [0, 1]},
            ]
        },
    }
    result = generator.generate_config(original, current, proposal, tmp_path / "next.toml")
    assert result["layout"]["roles"]["prefill"]["assignments"] == [{"host": "a", "gpu_ids": [0, 1]}]


def test_existing_hardlink_cannot_overwrite_base(tmp_path):
    import os

    context = experiment(tmp_path)
    base = Path(context["source"]["config_toml"])
    original = base.read_bytes()
    alias = tmp_path / "hardlink.toml"
    os.link(base, alias)
    with pytest.raises(ValueError, match="new file"):
        generator.generate_config(context, context, {}, alias)
    assert base.read_bytes() == original


def test_agent_rejects_unknown_host_reference_with_strict_plugin_loader(tmp_path):
    base = tmp_path / "invalid-host.toml"
    base.write_text(
        '[vllm_pd]\n'
        '[[vllm_pd.nodes]]\nrole="prefill"\nhosts="missing"\ngpu_ids=[0]\n'
        '[[vllm_pd.nodes]]\nrole="decode"\nssh_ip="node-b"\ngpu_ids=[0]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown deployment host reference"):
        load_plugin_config(base)
