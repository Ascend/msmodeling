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

"""Pure planning for the PSO search-to-deployment handoff."""

import shlex
from copy import deepcopy
from ipaddress import ip_address
from socket import inet_aton
from string import Template

from optix.serve_plan import render_serve_flags


def _address_key(value):
    """Canonicalize addresses for comparison without DNS or network access."""
    value = value.strip()
    try:
        address = ip_address(value)
    except ValueError:
        try:
            address = ip_address(inet_aton(value))
        except OSError:
            return value.casefold()
    address = getattr(address, "ipv4_mapped", None) or address
    return "*" if address.is_unspecified else str(address)


def validate_resource_pool(config, total_devices):
    """Validate explicit resources without opening SSH connections."""
    if not config.nodes:
        raise ValueError("full_service requires a flat vllm_pd.nodes resource pool")
    devices = set()
    for node in config.nodes:
        if node.role == "proxy":
            continue
        if node.role not in {"", "auto", "prefill", "decode"}:
            raise ValueError(f"Unsupported resource role: {node.role}")
        for card in node.gpu_ids:
            identity = (node.ssh_ip, node.ssh_port, card)
            if card < 0 or identity in devices:
                raise ValueError(f"Invalid or overlapping device: {identity}")
            devices.add(identity)
    if len(devices) < total_devices:
        raise ValueError(f"Resource pool has {len(devices)} devices, total_devices={total_devices}")
    proxies = [node for node in config.nodes if node.role == "proxy"]
    if len(proxies) > 1:
        raise ValueError("full_service supports one Proxy endpoint")
    proxy = proxies[0] if proxies else config.proxy
    if _address_key(proxy.bind_ip) in {"", "*"}:
        raise ValueError("full_service requires an explicit reachable Proxy bind_ip/service_ip")


def plan_role(command, fields, devices_per_instance, common_env=None):
    """Preserve serving flags while extracting topology owned by PD templates."""
    env = {field.name: str(field.value) for field in fields if field.config_position == "env"}
    others = Template(command.get("others", "")).substitute({**(common_env or {}), **env})
    argv = render_serve_flags([field for field in fields if field.config_position == "run"], others)
    topology = {"tp": 1, "dp": 1, "pp": 1}
    aliases = {
        "--tensor-parallel-size": "tp",
        "-tp": "tp",
        "--data-parallel-size": "dp",
        "-dp": "dp",
        "--pipeline-parallel-size": "pp",
        "-pp": "pp",
    }
    retained = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag, separator, value = token.partition("=")
        if flag in aliases:
            if not separator:
                index += 1
                if index == len(argv):
                    raise ValueError(f"Missing value for {flag}")
                value = argv[index]
            topology[aliases[flag]] = int(value)
        elif flag in {"--host", "--port", "--served-model-name", "--kv-transfer-config"}:
            raise ValueError(f"PD plugin owns {flag}; remove it from phase others")
        else:
            retained.append(token)
        index += 1
    if topology["pp"] != 1 or min(topology.values()) <= 0:
        raise ValueError("PD deployment requires positive TP/DP and PP=1")
    if topology["tp"] * topology["dp"] != devices_per_instance:
        raise ValueError("Phase TP * DP must equal devices_per_instance for PD deployment")
    if not command.get("model") or not command.get("served_model_name"):
        raise ValueError("PD deployment requires model and served_model_name")
    return {**command, "others": shlex.join(retained), **topology, "env": env}


def allocate_resource_pool(config, allocation, plans=None):
    """Assign explicit cards to P/D, respecting pinned roles and unused capacity."""
    result = deepcopy(config)
    workers = [node for node in result.nodes if node.role != "proxy"]
    remaining = [list(node.gpu_ids) for node in workers]
    assigned = []
    for role in ("prefill", "decode"):
        needed = allocation[f"{role}_devices"]
        tp = plans[role]["tp"] if plans else 1
        # Consume pinned cards first, leaving flexible cards for the other role.
        order = sorted(range(len(workers)), key=lambda i: workers[i].role != role)
        for index in order:
            node = workers[index]
            if node.role not in {role, "", "auto"} or not needed:
                continue
            count = min(needed, len(remaining[index])) // tp * tp
            if not count:
                continue
            offset = len(node.gpu_ids) - len(remaining[index])
            assigned.append(
                node.model_copy(
                    update={
                        "role": role,
                        "gpu_ids": remaining[index][:count],
                        "service_port": node.service_port + offset,
                        "kv_port": node.kv_port + offset,
                        "rpc_port": node.rpc_port + offset,
                        "ascend_base_port": (node.ascend_base_port or result.ascend_base_port or 20000) + offset,
                    }
                )
            )
            remaining[index] = remaining[index][count:]
            needed -= count
        if needed:
            raise ValueError(f"Not enough permitted devices for {role}: missing {needed}")
    proxies = [node for node in result.nodes if node.role == "proxy"]
    if proxies:
        result.proxy = proxies[0]
    result.nodes = assigned
    result.prefill_groups = []
    result.decode_groups = []
    result.prefill_instances = allocation["prefill"]
    result.decode_instances = allocation["decode"]
    result.target_field = []
    return result


def validate_deployment(config, plans):
    """Reject layouts the launcher cannot realize and conflicting service ports."""
    used = {}

    def reserve(node, port, *, wildcard=False):
        address = _address_key(node.bind_ip or node.ssh_ip)
        context = (_address_key(node.ssh_ip), node.ssh_port, node.docker_container_id or "")
        identity = (context, address, port)
        if not 0 < port < 65536:
            raise ValueError(f"Invalid or conflicting PD endpoint: {identity}")
        wildcard = wildcard or address == "*"
        for other_context, other_address, other_wildcard in used.get(port, []):
            same_endpoint = address != "*" and address == other_address
            same_listener = context == other_context and (wildcard or other_wildcard)
            if same_endpoint or same_listener:
                raise ValueError(f"Invalid or conflicting PD endpoint: {identity}")
        used.setdefault(port, []).append((context, address, wildcard))

    reserve(config.proxy, config.proxy.service_port)
    for role in ("prefill", "decode"):
        tp = plans[role]["tp"]
        for group in getattr(config, f"{role}_groups"):
            for node in group.nodes:
                if len(node.gpu_ids) % tp:
                    raise ValueError("PD plugin requires each TP replica to fit on one node")
                for rank in range(len(node.gpu_ids) // tp):
                    # P/D HTTP templates listen on all IPv4 interfaces, not bind_ip.
                    reserve(node, node.service_port + rank, wildcard=True)
                    reserve(node, node.rpc_port + rank)
                    for offset in range(tp):
                        reserve(node, node.kv_port + rank * tp + offset)
