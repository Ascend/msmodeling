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


"""Bounded legal-layout enumeration and per-host port planning; no remote I/O."""

from copy import deepcopy
from itertools import product
from math import isfinite

from .pd import (
    ROLES,
    device_inventory,
    positive_integer,
    resource_nodes,
    validate_resources,
)


def plan_node_ports(config):
    """Allocate non-overlapping blocks before the original flat-pool splitter runs."""
    if not config.nodes:
        raise ValueError("Port planning requires flat PD nodes")
    occupied = {}
    proxy = next((n for n in config.nodes if n.role == "proxy"), config.proxy)
    occupied.setdefault(proxy.ssh_ip, set()).add(proxy.service_port)
    # The plugin assigns group DP coordinator ports in these two ranges.
    # The exact number of groups cannot exceed the number of declared devices.
    total = sum(len(n.gpu_ids) for n in config.nodes if n.role in ROLES)
    for node in config.nodes:
        if node.role in ROLES:
            occupied.setdefault(node.ssh_ip, set()).update(range(12345, 12345 + total))
            occupied[node.ssh_ip].update(range(13345, 13345 + total))
    for node in config.nodes:
        if node.role not in ROLES:
            continue
        used = occupied[node.ssh_ip]
        width = len(node.gpu_ids)
        if not width:
            raise ValueError("PD port planning requires nonempty device assignments")
        for name in ("service_port", "kv_port", "rpc_port", "ascend_base_port"):
            base = getattr(node, name)
            if base is None:
                base = config.ascend_base_port or config.ASCEND_DEFAULT_BASE_PORT
            while base > 0 and base + width <= 65536 and any(p in used for p in range(base, base + width)):
                base += width
            if base < 1 or base + width > 65536:
                raise ValueError(f"No valid {name} range on {node.ssh_ip}")
            setattr(node, name, base)
            used.update(range(base, base + width))
    return config


def enumerate_layouts(
    config,
    params,
    *,
    max_candidates=32,
    max_combinations=10000,
    use_full_device=True,
    prefill_qps=None,
    decode_qps=None,
):
    """Enumerate deterministic contiguous partitions; explicitly report bounded coverage.

    Source entries remain distinct for config generation. This covers partition
    counts in source order, not all permutations of physical cards/NUMA placement.
    """
    positive_integer(max_candidates, "max candidates")
    positive_integer(max_combinations, "max combinations")
    nodes = resource_nodes(config)
    device_inventory(nodes)
    sources = [(i, n) for i, n in enumerate(nodes) if n.get("role") in ROLES]
    tp = {r: positive_integer(params.get("TENSOR_PARALLEL_SIZE_" + r, 1), r + " TP") for r in ROLES}
    dp = {r: positive_integer(params.get("DATA_PARALLEL_SIZE_" + r, 1), r + " DP") for r in ROLES}
    choices = []
    for _, node in sources:
        count = len(node["gpu_ids"])
        choices.append(
            [
                (p, d)
                for p in range(0, count + 1, tp["prefill"])
                for d in range(0, count - p + 1, tp["decode"])
                if not use_full_device or p + d == count
            ]
        )
    layouts = []
    visited = 0
    truncated = False
    for partition in product(*choices):
        if visited >= max_combinations:
            truncated = True
            break
        visited += 1
        assignments, proposed = [], []
        counts = dict.fromkeys(ROLES, 0)
        for (index, node), (p, d) in zip(sources, partition):
            for role, cards in (
                ("prefill", node["gpu_ids"][:p]),
                ("decode", node["gpu_ids"][p : p + d]),
            ):
                if cards:
                    assignments.append({"source_node": index, "role": role, "gpu_ids": cards})
                    proposed.append({**deepcopy(node), "role": role, "gpu_ids": cards})
                    counts[role] += len(cards)
        if any(not counts[r] or counts[r] % (tp[r] * dp[r]) for r in ROLES):
            continue
        data = config.model_dump(mode="json") if hasattr(config, "model_dump") else deepcopy(config)
        data.update(nodes=proposed, prefill_groups=[], decode_groups=[])
        if any(
            data.get(r + "_instances") is not None and data[r + "_instances"] != counts[r] // (tp[r] * dp[r])
            for r in ROLES
        ):
            continue
        layout = validate_resources(data, params)
        item = {"deployment": {"nodes": assignments}, "layout": layout}
        if prefill_qps is not None and decode_qps is not None:
            if not isfinite(prefill_qps) or not isfinite(decode_qps) or prefill_qps <= 0 or decode_qps <= 0:
                raise ValueError("Phase QPS must be positive")
            p_count = layout["roles"]["prefill"]["instances"]
            d_count = layout["roles"]["decode"]["instances"]
            item["estimated_balanced_qps"] = min(p_count * prefill_qps, d_count * decode_qps)
            item["ratio_error"] = abs(p_count / d_count - decode_qps / prefill_qps)
        layouts.append(item)
    layouts.sort(key=lambda x: (x.get("ratio_error", 0), -x.get("estimated_balanced_qps", 0)))
    return {
        "layouts": layouts[:max_candidates],
        "examined_partitions": visited,
        "truncated": truncated or len(layouts) > max_candidates,
        "coverage": "contiguous source partitions; physical-card permutations are not enumerated",
    }
