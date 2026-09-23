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

"""Cluster config loading: parses the node information in config.toml and serves as the
single source of truth for this plugin's configuration.

Structure of config.toml:
    [[vllm_mix.node]]      Master node (same machine as the optimizer, launched locally, no SSH info needed)
    [[vllm_mix.workers]]   Worker nodes (launched remotely over SSH)
    [vllm_mix.env]         Static environment variables exported into every generated startup script
    docker_use_sudo        Whether commands are prefixed with non-interactive sudo -n

This is a pure configuration module: it only depends on the standard library and does
not depend back on executor / simulator, so it can be imported both from inside the
plugin package and by build_shell_scripts.py running as a standalone subprocess.
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass
class NodeConfig:
    """Configuration for a single node; the attributes match the fields SshRemote.from_node reads."""

    host: str
    ssh_port: int = 22
    ssh_user: str = "root"
    docker_container_id: Optional[str] = None
    nic_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeConfig":
        # Reject credentials instead of silently accepting an unsafe legacy config.
        auth_fields = {"password", "ssh_password", "private_key", "key_filename", "pkey", "passphrase"}
        invalid_fields = auth_fields.intersection(data)
        if invalid_fields:
            names = ", ".join(sorted(invalid_fields))
            raise ValueError(
                "SSH authentication fields are not supported in node configuration. "
                f"Remove {names}; "
                "configure passwordless SSH in the optimizer's runtime environment."
            )
        # Keep only known fields, ignoring other extra keys in config.toml
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


def _normalize_env_value(value: Any) -> str:
    """Normalize a TOML env value to the string rendered into ``export KEY=VALUE``.

    TOML booleans (``OMP_PROC_BIND = false``) parse as Python ``bool``; stringify them
    as lowercase ``true``/``false`` so the generated shell keeps the exact value the
    OpenMP/HCCL runtime expects, rather than Python's capitalized ``True``/``False``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


#: Default regex matched against the node's startup log to decide the DP cluster is
#: fully ready. This is uvicorn's line printed after the FastAPI lifespan completes
#: (which, in vLLM data-parallel mode, includes the rendezvous with all headless workers).
DEFAULT_READY_LOG_PATTERN = "Application startup complete"


@dataclass
class Config:
    """Cluster configuration, loaded from config.toml."""

    node: Optional[NodeConfig] = None
    workers: List[NodeConfig] = field(default_factory=list)
    docker_use_sudo: bool = False
    chips_per_node: int = 8
    # DP handshake RPC port (--data-parallel-rpc-port). None means "not fixed":
    # build_shell_scripts picks a free port each time it generates the scripts.
    data_parallel_rpc_port: Optional[int] = None
    # Static environment variables ([vllm_mix.env]) exported into every generated
    # startup script. Values are pre-normalized to strings (booleans become lowercase).
    env: Dict[str, str] = field(default_factory=dict)
    # Regex matched against the node's startup log to decide the DP cluster is fully
    # ready. Defaults to uvicorn's "Application startup complete" line.
    ready_log_pattern: str = DEFAULT_READY_LOG_PATTERN

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "Config":
        if path is None:
            path = os.environ.get("CLUSTER_CONFIG_PATH") or Path(__file__).resolve().parent / "config.toml"
        with open(path, "rb") as f:
            data = tomllib.load(f)
        vllm_mix = data.get("vllm_mix", {})
        nodes = vllm_mix.get("node", [])
        node = NodeConfig.from_dict(nodes[0]) if nodes else None
        workers = [NodeConfig.from_dict(w) for w in vllm_mix.get("workers", [])]
        chips_per_node = int(vllm_mix.get("chips_per_node", 8))
        # Allow 0 to mean "not fixed", equivalent to leaving it unset
        rpc_port_raw = vllm_mix.get("data_parallel_rpc_port")
        rpc_port = int(rpc_port_raw) if rpc_port_raw else None
        env = {str(k): _normalize_env_value(v) for k, v in vllm_mix.get("env", {}).items()}
        ready_log_pattern = vllm_mix.get("ready_log_pattern")
        if ready_log_pattern is None:
            ready_log_pattern = DEFAULT_READY_LOG_PATTERN

        return cls(
            node=node,
            workers=workers,
            docker_use_sudo=bool(data.get("docker_use_sudo", False)),
            chips_per_node=chips_per_node,
            data_parallel_rpc_port=rpc_port,
            env=env,
            ready_log_pattern=ready_log_pattern,
        )


def load_cluster_config(
    path: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]], int, Optional[int], Dict[str, str]]:
    """For use by build_shell_scripts: returns (node_ip, all_nodes, chips_per_node, rpc_port, env).

    - node_ip: host of the first [[vllm_mix.node]] entry, used as
      --data-parallel-address for all nodes (NODE_IP in the template).
    - all_nodes: list[dict] ordered by rank:
        - rank 0 is [[vllm_mix.node]] (the master node, generates start_node.sh);
        - ranks 1..N are the [[vllm_mix.workers]] entries in order (generating
          start_work_0.sh ...).
      Each item holds host (the node's LOCAL_IP) and nic_name (NIC_NAME).
    - chips_per_node: chips per node (A3=16, A2=8), used to compute dp_size_local.
    - rpc_port: the DP handshake RPC port, or None when unset (the caller then picks a
      free port automatically).
    - env: static environment variables from [vllm_mix.env], exported into every
      generated script.

    Raises ValueError when a required field is missing, leaving presentation to the caller.
    """
    cfg = Config.from_file(path)
    if cfg.node is None or not cfg.node.host:
        raise ValueError("missing [[vllm_mix.node]] configuration (host required) in config.toml")

    all_nodes = [{"host": cfg.node.host, "nic_name": cfg.node.nic_name or ""}]
    for w in cfg.workers:
        if not w.host:
            raise ValueError("an item in [[vllm_mix.workers]] is missing the host field")
        all_nodes.append({"host": w.host, "nic_name": w.nic_name or ""})

    return cfg.node.host, all_nodes, cfg.chips_per_node, cfg.data_parallel_rpc_port, cfg.env
