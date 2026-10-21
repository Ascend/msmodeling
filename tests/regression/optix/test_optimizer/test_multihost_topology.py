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


"""Cluster snapshots and baseline ordering fail before launching invalid trials."""

import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from optix.optimizer.agentic import topology
from optix.optimizer.agentic.orchestrator import AgentOptimizer


def cluster(tmp_path, monkeypatch):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[vllm_mix]\nchips_per_node=8\n[[vllm_mix.node]]\nhost="master"\n[[vllm_mix.workers]]\nhost="worker"\n'
    )
    context = {
        "deployment": {"type": "multihost", "cluster_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "source": {"cluster_config": str(path)},
        "hardware": {"num_nodes": 2, "num_per_node": 8, "world_size": 16},
        "constraints": {"world_size": 16},
    }
    simulator = NS(
        remote_executor=NS(config=NS(node=NS(host="master"), workers=[NS(host="worker")], chips_per_node=8, env={}))
    )
    monkeypatch.setenv("CLUSTER_CONFIG_PATH", str(path))
    return path, context, simulator


def test_topology_reads_once_and_rechecks_changes(tmp_path, monkeypatch):
    path, context, simulator = cluster(tmp_path, monkeypatch)
    read = Path.read_bytes
    calls = []

    def counted(p):
        calls.append(p)
        return read(p)

    monkeypatch.setattr(Path, "read_bytes", counted)
    monkeypatch.setattr(Path, "read_text", Mock(side_effect=AssertionError("duplicate parse")))
    topology.validate_simulator_topology(context, simulator)
    assert calls == [path]
    path.write_bytes(read(path) + b"\n# changed")
    with pytest.raises(ValueError, match="Cluster config changed"):
        topology.validate_simulator_topology(context, simulator)


def test_topology_still_checks_hardware(tmp_path, monkeypatch):
    _, context, simulator = cluster(tmp_path, monkeypatch)
    context["hardware"]["world_size"] = 8
    with pytest.raises(ValueError, match="hardware conflicts"):
        topology.validate_simulator_topology(context, simulator)


@pytest.mark.parametrize(
    "engine,expected",
    [
        ("vllm", ["baseline", "pin", "context", "rounds"]),
        ("multihost_infer", ["context", "validate", "baseline", "pin", "rounds"]),
    ],
)
def test_orchestrator_preserves_order(tmp_path, monkeypatch, engine, expected):
    events = []
    opt = AgentOptimizer(pso=NS(scheduler=NS(engine=engine, simulator=object())), agent_config=NS(), run_dir=tmp_path)
    opt._scan_completed_rounds = lambda: (set(), float("inf"))
    opt._ensure_baseline = lambda: events.append("baseline")
    opt._pin_concurrency = lambda: events.append("pin")
    opt._load_search_space_context = lambda: events.append("context") or {}
    opt._run_rounds = lambda *args: events.append("rounds")
    monkeypatch.setattr(topology, "validate_simulator_topology", lambda *args: events.append("validate"))
    opt.run()
    assert events == expected


def test_bad_topology_prevents_baseline(tmp_path, monkeypatch):
    opt = AgentOptimizer(
        pso=NS(scheduler=NS(engine="multihost_infer", simulator=object())), agent_config=NS(), run_dir=tmp_path
    )
    opt._scan_completed_rounds = lambda: (set(), float("inf"))
    opt._ensure_baseline = Mock()
    opt._load_search_space_context = lambda: {}
    monkeypatch.setattr(topology, "validate_simulator_topology", Mock(side_effect=ValueError("bad topology")))
    with pytest.raises(ValueError, match="bad topology"):
        opt.run()
    opt._ensure_baseline.assert_not_called()


@pytest.mark.parametrize(
    "sizes,message",
    [
        ((2, 3, 1, 1, 2, 8), "divisible"),
        ((2, 2, 1, 8, 2, 8), "exceeds"),
        ((True, 2, 1, 1, 2, 8), "positive integers"),
        ((2, 2, 1, 1, 2, 8), "must equal"),
    ],
)
def test_invalid_parallel_placement_is_rejected(sizes, message):
    with pytest.raises(ValueError, match=message):
        topology.validate_parallel_placement(*sizes)
