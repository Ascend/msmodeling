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


"""Validate the unchanged multihost plugin's homogeneous DP placement contract."""

import hashlib
import os
from pathlib import Path

from ...serve_plan import TOPOLOGY_ONLY_FIELDS

# Preserve the existing validation/skill import name without duplicating the set.
TOPOLOGY_FACT_NAMES = TOPOLOGY_ONLY_FIELDS


def is_multihost_context(context: dict | None) -> bool:
    deployment = (context or {}).get("deployment")
    if deployment is None:
        return False
    if not isinstance(deployment, dict):
        raise ValueError("context.deployment must be an object")
    return deployment.get("type") == "multihost"


def validate_topology_snapshot(context: dict) -> dict | None:
    if not is_multihost_context(context):
        return None
    deployment = context["deployment"]
    path = (context.get("source") or {}).get("cluster_config")
    digest = deployment.get("cluster_config_sha256")
    if not path or not digest:
        raise ValueError("Cluster config changed or topology snapshot is missing; collect context again")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Cluster config changed or topology snapshot is missing; collect context again")

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    section = tomllib.loads(raw.decode("utf-8"))["vllm_mix"]
    nodes = section.get("node") or []
    workers = section.get("workers") or []
    if len(nodes) != 1:
        raise ValueError("Multihost configuration requires exactly one master")
    count = 1 + len(workers)
    chips = section.get("chips_per_node", 8)
    hardware = context.get("hardware") or {}
    expected = {"num_nodes": count, "num_per_node": chips, "world_size": count * chips}
    if any(hardware.get(key) != value for key, value in expected.items()):
        raise ValueError("context.hardware conflicts with cluster config; collect context again")
    if (context.get("constraints") or {}).get("world_size") != expected["world_size"]:
        raise ValueError("context.constraints.world_size conflicts with cluster config")
    return section


def validate_parallel_placement(tp, dp, pp, pcp, node_count, chips_per_node) -> None:
    values = (tp, dp, pp, pcp, node_count, chips_per_node)
    if any(isinstance(v, bool) or float(v) != int(v) or int(v) <= 0 for v in values):
        raise ValueError("Parallel sizes and topology facts must be positive integers")
    tp, dp, pp, pcp, node_count, chips_per_node = map(int, values)
    if dp % node_count:
        raise ValueError("Multihost DP must be divisible by the configured node count")
    if dp // node_count * tp * pp * pcp > chips_per_node:
        raise ValueError("Multihost local DP * TP * PP * PCP exceeds chips_per_node")
    if tp * dp * pp * pcp != node_count * chips_per_node:
        raise ValueError("Multihost TP * DP * PP * PCP must equal the configured total device count")


def validate_simulator_topology(context: dict, simulator) -> None:
    """Ensure the loaded plugin configuration is the round's recorded topology."""
    if not is_multihost_context(context):
        raise ValueError("Multihost agent execution requires a cluster-config context")
    section = validate_topology_snapshot(context)
    selected_path = os.environ.get("CLUSTER_CONFIG_PATH")
    if not selected_path or Path(selected_path).resolve() != Path(context["source"]["cluster_config"]).resolve():
        raise ValueError("Set CLUSTER_CONFIG_PATH to the context cluster config before starting multihost optix")
    config = simulator.remote_executor.config
    expected_hosts = [section["node"][0]["host"], *[node["host"] for node in section.get("workers", [])]]
    actual_hosts = [config.node.host, *[node.host for node in config.workers]]
    if actual_hosts != expected_hosts or config.chips_per_node != section.get("chips_per_node", 8):
        raise ValueError("Loaded multihost plugin topology differs from context; set CLUSTER_CONFIG_PATH and restart")
    expected_env = {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in section.get("env", {}).items()
    }
    if config.env != expected_env:
        raise ValueError("Loaded multihost plugin environment differs from cluster config; restart optix")
