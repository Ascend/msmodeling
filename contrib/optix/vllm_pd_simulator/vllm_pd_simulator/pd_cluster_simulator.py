# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""
vLLM DP Sep plugin.

Starts P-Deployer, D-Deployer and Proxy remotely via SSH.

Model parameters (model, tp, max_model_len, etc.) are passed in by the optimizer
via update_config.
"""

import os
import shlex
import shutil
import time
from types import SimpleNamespace
from collections import defaultdict
from typing import ClassVar, Dict, List, Optional, Tuple, Any
from pathlib import Path

from loguru import logger

from optix.optimizer.interfaces.simulator import SimulatorInterface
from optix.optimizer.utils import close_file_fp, remove_file
from optix.config.constant import ProcessState, Stage
from optix.config.config import get_settings, OptimizerConfigField, range_to_enum
from .tools import render_template, shell_quote, SshRemote, detect_is_moe
from .config import VLLMPDDisaggConfig, load_pd_config, PDGroup

# proxy 脚本文件名（缺失时提示用户手动下载，不自动下载）
_PROXY_SCRIPT_FILES = [
    "load_balance_proxy_server_example.py",
    "load_balance_proxy_layerwise_server_example.py",
]

# 对应上游 URL（提示信息用）
_PROXY_SCRIPT_URLS = {
    "load_balance_proxy_server_example.py": "https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
    "load_balance_proxy_layerwise_server_example.py": "https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py",
}


def _check_proxy_scripts(scripts_dir) -> None:
    """Check proxy load-balancing scripts exist; warn + hint if missing.

    Called from __init__ to validate deployment prerequisites early.
    Missing files do NOT auto-download — user must run
    `bash download_proxy_scripts.sh` manually.
    """
    scripts_dir = Path(scripts_dir)
    missing = []
    for name in _PROXY_SCRIPT_FILES:
        target = scripts_dir / name
        if not target.exists():
            missing.append(name)
    if missing:
        logger.error(
            f"[vllm_pd] Missing proxy scripts: {missing}\n"
            f"  Run: bash {scripts_dir / 'download_proxy_scripts.sh'}\n"
            f"  Or manually download from:"
        )
        for name in missing:
            logger.error(f"    {_PROXY_SCRIPT_URLS[name]} -> {scripts_dir / name}")
        logger.error("  PD deployment will fail at proxy startup without these files.")


class _RemoteProcessPlaceholder:
    """Placeholder for remote processes with no local subprocess.Popen.

    str() returns the process name for readable scheduler logs.
    poll() always returns None (mimics a running local process).
    returncode is always None (never exited locally).
    Implements enough of the subprocess.Popen interface (pid, kill, wait,
    send_signal) so that CustomProcess.stop() can execute without
    AttributeError when self.process is this placeholder instead of a
    real local subprocess.
    """

    def __init__(self, name: str):
        self._name = name
        self.returncode = None
        # pid must exist: CustomProcess.stop() accesses self.process.pid
        # via psutil.Process(self.process.pid). Returning None signals
        # "no local PID" — callers that need the real PID should use
        # self._remote_pids instead.
        self.pid = None

    def poll(self):
        return None

    def kill(self):
        """No-op: remote processes are killed via SSH in PdClusterSimulator.stop()."""
        self.returncode = 0

    def wait(self, timeout=None):
        """No-op: nothing to wait for locally. Mimics successful termination."""
        self.returncode = 0
        return 0

    def send_signal(self, sig):
        """No-op: signals cannot be sent to remote processes from here."""
        pass

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"_RemoteProcessPlaceholder({self._name!r})"


def _clean_rendered_shell(content: str) -> str:
    """Remove blank lines and lines with only spaces/backslash after rendering."""
    lines = content.splitlines()
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in ("\\",):
            continue
        clean.append(line)
    return "\n".join(clean)


class PdClusterSimulator(SimulatorInterface):
    """vLLM PD cluster simulator (started remotely via SSH)."""

    required_executable: ClassVar[Optional[str]] = None

    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, process_name="vllm_pd_simulator", **kwargs)

        settings = get_settings()
        self.model_path = settings.vllm.command.model
        self.served_model_name = settings.vllm.command.served_model_name
        self.vllm_others = settings.vllm.command.others

        if config:
            self.config = config
        else:
            # Always load via load_pd_config(), which reads the [[hosts]]
            # section from the config file specified by -c and resolves references
            self.config = load_pd_config() or VLLMPDDisaggConfig()

        # Merge the main framework's vllm.target_field into the plugin target_field.
        # Plugin-defined parameters take precedence: same-name parameters in the
        # plugin config override those from the main framework.
        if settings.vllm.target_field:
            plugin_names = {f.name for f in self.config.target_field} if self.config.target_field else set()
            for field in settings.vllm.target_field:
                if field.name not in plugin_names:
                    self.config.target_field.append(field)

        # Convert tuning params with dtype="range" to dtype="enum", consistent with the main framework's handling
        if self.config.target_field:
            range_to_enum(self.config.target_field)

        # Auto-detect MoE model: read config.json from the main framework's vllm.command.model path
        self._is_moe = detect_is_moe(self.model_path)
        if self._is_moe:
            logger.info(f"Detected MoE model at {self.model_path}")

        # Dynamic parameters for P/D nodes
        self._prefill_env_vars: dict = {}
        self._decode_env_vars: dict = {}
        self._prefill_run_vars: dict = {}
        self._decode_run_vars: dict = {}

        # TP/DP sizes injected by tuning parameters (default 1, not config.toml fields)
        self._prefill_tp_size: int = 1
        self._prefill_dp_size: int = 1
        self._decode_tp_size: int = 1
        self._decode_dp_size: int = 1

        # Override the parent class process: vLLM DP is deployed remotely and has
        # no local vllm subprocess. Use _RemoteProcessPlaceholder to keep the
        # scheduler's monitoring_status working with process.poll(): it always
        # returns None (meaning alive), and str() yields a meaningful name.
        self.process = _RemoteProcessPlaceholder(self.process_name)
        self.run_log = None

        # Remote executor cache (closed during stop)
        self._executors: Dict[str, SshRemote] = {}

        self._ssh_cmd_timeout = getattr(self.config, 'ssh_command_timeout', 30) or 30

        self._remote_tmp_dir = getattr(self.config, 'remote_tmp_dir', '/tmp') or '/tmp'  # nosec B108

        # Remote process PID tracking (for targeted kill during stop + runtime status monitoring)
        self._remote_pids: Dict[str, int] = {}
        self._remote_pid_nodes: Dict[str, Any] = {}
        self._remote_logs: Dict[str, str] = {}
        self._cluster_id: int = 0
        # 已跑过的不同粒子（按 (name,value) 参数键去重）。编号 = len(this list)。
        # retry 传同一参数键 -> in 命中 -> 不新增，故 retry 不再消耗编号。
        self._particle_list: list = []
        self._session_timestamp: Optional[str] = None  # Deprecated, kept only for compatibility with existing mocks
        self._round_tmp_dir: Optional[Path] = None  # Cache for this round's temp dir (tempfile, deleted after use)
        self._node_infos: Optional[list] = None

        self._process_stage = ProcessState(stage=Stage.stop)
        self.command = None

        if self.config.nodes:
            # New format: flat node pool, auto-split at runtime based on ep_size
            roles = [n.role for n in self.config.nodes]
            logger.info(
                f"[OPT] __init__: flat node pool with {len(self.config.nodes)} node(s) "
                f"(prefill={roles.count('prefill')}, decode={roles.count('decode')}, proxy={roles.count('proxy')}), "
                f"groups will be auto-split in update_config"
            )
            rows = [
                (
                    n.role,
                    f"{n.ssh_ip}:{n.ssh_port}",
                    str(len(n.gpu_ids)),
                    str(n.gpu_ids),
                    n.bind_ip or n.ssh_ip,
                    n.network_interface,
                )
                for n in self.config.nodes
            ]
        elif self.config.prefill_groups or self.config.decode_groups:
            # Legacy format: prefill_groups/decode_groups/proxy written directly in config.toml
            prefill_cnt = sum(len(g.nodes) for g in self.config.prefill_groups)
            decode_cnt = sum(len(g.nodes) for g in self.config.decode_groups)
            logger.info(
                f"[OPT] __init__: legacy format with prefill_groups({len(self.config.prefill_groups)} group(s), "
                f"{prefill_cnt} node(s)) and decode_groups({len(self.config.decode_groups)} group(s), "
                f"{decode_cnt} node(s)), proxy={self.config.proxy.ssh_ip}:{self.config.proxy.service_port}, "
                f"auto-split skipped"
            )
            rows = []
            for g in self.config.prefill_groups:
                for n in g.nodes:
                    rows.append(
                        (
                            "prefill",
                            f"{n.ssh_ip}:{n.ssh_port}",
                            str(len(n.gpu_ids)),
                            str(n.gpu_ids),
                            n.bind_ip or n.ssh_ip,
                            n.network_interface,
                        )
                    )
            for g in self.config.decode_groups:
                for n in g.nodes:
                    rows.append(
                        (
                            "decode",
                            f"{n.ssh_ip}:{n.ssh_port}",
                            str(len(n.gpu_ids)),
                            str(n.gpu_ids),
                            n.bind_ip or n.ssh_ip,
                            n.network_interface,
                        )
                    )
            p = self.config.proxy
            rows.append(
                (
                    "proxy",
                    f"{p.ssh_ip}:{p.ssh_port}",
                    "0",
                    "[]",
                    p.bind_ip or p.ssh_ip,
                    "-",
                )
            )
        else:
            rows = []

        if rows:
            headers = ("Role", "Host", "Cards", "GPU_IDs", "ServiceIP", "NIC")
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, val in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(val))
            fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
            sep_len = len(fmt.format(*headers))
            lines = ["=" * sep_len, fmt.format(*headers), "-" * sep_len]
            for row in rows:
                lines.append(fmt.format(*row))
            lines.append("=" * sep_len)
            logger.info("\n".join(lines))

        # Check proxy scripts exist (deployment prerequisite — no auto-download)
        scripts_dir = self._get_scripts_dir()
        _check_proxy_scripts(scripts_dir)

    @property
    def base_url(self) -> str:
        proxy = self.config.proxy
        url = f"http://{proxy.bind_ip}:{proxy.service_port}/v1/chat/completions"
        logger.debug(f"[OPT] base_url -> {url}")
        return url

    def update_command(self) -> None:
        logger.debug("[OPT] update_command called (no-op)")

    @property
    def _particle_count(self) -> int:
        """已跑过的不同粒子数 = 列表长度（1-based 编号即新加入粒子在列表中的位次）。"""
        return len(self._particle_list)

    def _particle_key(self, run_params) -> tuple:
        """从 run_params 派生粒子身份键。retry 复用同一份参数，键相同 -> 去重不新增。"""
        if not run_params:
            return ()
        try:
            return tuple((getattr(f, "name", None), getattr(f, "value", None)) for f in run_params)
        except TypeError:
            # run_params 不是可迭代的字段序列时的兜底（理论上不会走到）
            return (run_params,)

    def _get_scripts_dir(self) -> str:
        """Return the scripts directory under the plugin install directory."""
        plugin_dir = Path(__file__).resolve().parent
        scripts_dir = plugin_dir / "scripts" / "pd"
        return str(scripts_dir)

    def update_config(self, params: Optional[Tuple[OptimizerConfigField]] = None) -> bool:
        """Receive model parameters from the optimizer and route them to P/D nodes dynamically.

        Parameters fall into two categories:
        - Parameters with an _prefill/_decode suffix: routed to the corresponding node by suffix (plugin-defined parameters)
        - Parameters without a suffix: applied to both P and D sides (common parameters from the main framework)

        When a parameter with the same name exists in both the common and suffixed
        versions, the suffixed one takes precedence.
        """
        if not params:
            # 空参时仍需构建拓扑（flat-node 模式 _apply_ep_split 从 nodes 池拆分 P/D groups）；
            # TP/DP 保持实例属性默认值（或上轮寻优值），不从 params 覆盖。
            self._apply_ep_split()
            return True

        # Clear parameters from the previous round
        self._prefill_env_vars = {}
        self._decode_env_vars = {}
        self._prefill_run_vars = {}
        self._decode_run_vars = {}

        logger.debug(f"update_config called with {len(params)} param(s)")

        # Pass 1: collect common parameters without a suffix (applied to both P and D sides)
        common_env_prefill = {}
        common_env_decode = {}
        common_run_prefill = {}
        common_run_decode = {}

        for param in params:
            name = param.name
            value = self._resolve_enum_value(param)
            config_position = param.config_position if hasattr(param, 'config_position') else ""

            # TP/DP parameters control cluster topology and are not injected via env/run; handled separately by _apply_tp_dp_from_params
            if name in (
                "TENSOR_PARALLEL_SIZE_prefill",
                "DATA_PARALLEL_SIZE_prefill",
                "TENSOR_PARALLEL_SIZE_decode",
                "DATA_PARALLEL_SIZE_decode",
            ):
                continue

            # Parse the name suffix to determine the node type
            # Format: [param_name]_prefill or [param_name]_decode
            if name.endswith("_prefill"):
                node_type = "prefill"
                param_name = name[:-8]
            elif name.endswith("_decode"):
                node_type = "decode"
                param_name = name[:-7]
            else:
                # Common parameter without a suffix, applied to both P and D
                if config_position not in ("env", "run"):
                    logger.warning(f"Parameter '{name}' has invalid config_position '{config_position}', skipping")
                    continue
                logger.debug(f"  common param: name={name}, value={value}, config_position={config_position}")
                if config_position == "run":
                    common_run_prefill[name] = str(value)
                    common_run_decode[name] = str(value)
                else:
                    common_env_prefill[name] = str(value)
                    common_env_decode[name] = str(value)
                continue

            if config_position not in ("env", "run"):
                logger.warning(f"Parameter '{name}' has invalid config_position '{config_position}', skipping")
                continue

            logger.debug(
                f"  param: name={name}, value={value}, "
                f"node_type={node_type}, param_name={param_name}, "
                f"config_position={config_position}"
            )

            if config_position == "run":
                target = self._prefill_run_vars if node_type == "prefill" else self._decode_run_vars
                target[param_name] = str(value)
            else:
                target = self._prefill_env_vars if node_type == "prefill" else self._decode_env_vars
                self._add_param(target, param_name, value)

        # Pass 2: merge common parameters; suffixed parameters override same-name common ones
        # env vars: common parameters form the base, suffixed parameters override by param_name key
        for k, v in common_env_prefill.items():
            self._prefill_env_vars.setdefault(k, v)
        for k, v in common_env_decode.items():
            self._decode_env_vars.setdefault(k, v)
        for k, v in common_run_prefill.items():
            self._prefill_run_vars.setdefault(k, v)
        for k, v in common_run_decode.items():
            self._decode_run_vars.setdefault(k, v)

        # Pass 3: extract TP/DP values from tuning parameters into instance attributes
        self._apply_tp_dp_from_params(params)

        # Pass 4: split the flat node pool into prefill_groups/decode_groups based on ep_size
        self._apply_ep_split()

        logger.info("vllm_pd_simulator updated:")
        logger.info(f"  P tp={self._prefill_tp_size}, dp={self._prefill_dp_size}")
        logger.info(f"  D tp={self._decode_tp_size}, dp={self._decode_dp_size}")
        logger.info(f"  P env_vars: {self._prefill_env_vars}")
        logger.info(f"  P run_vars: {self._prefill_run_vars}")
        logger.info(f"  D env_vars: {self._decode_env_vars}")
        logger.info(f"  D run_vars: {self._decode_run_vars}")
        return True

    def _apply_tp_dp_from_params(self, params):
        """Extract TENSOR_PARALLEL_SIZE / DATA_PARALLEL_SIZE from tuning parameters,
        setting instance attributes _prefill/_decode_tp_size and dp_size.
        When not configured, the default value of 1 is kept.
        """
        tp_dp_map = {
            "TENSOR_PARALLEL_SIZE_prefill": "_prefill_tp_size",
            "DATA_PARALLEL_SIZE_prefill": "_prefill_dp_size",
            "TENSOR_PARALLEL_SIZE_decode": "_decode_tp_size",
            "DATA_PARALLEL_SIZE_decode": "_decode_dp_size",
        }
        for param in params:
            if param.name in tp_dp_map:
                attr = tp_dp_map[param.name]
                value = int(self._resolve_enum_value(param))
                setattr(self, attr, value)
                logger.info(f"  {attr} set by param {param.name}={value}")

    def _apply_ep_split(self):
        """Split the flat node pool into prefill_groups / decode_groups based on the current ep_size, and extract the proxy."""

        # Legacy config.toml configures prefill_groups/decode_groups/proxy directly;
        # node topology is already fixed, so no auto-split is needed. But the
        # ascend_base_port still must be offset by the cross-group cumulative GPU
        # count, otherwise multiple same-host groups would all fall on the default
        # 20000 ASCEND Direct Transport port and their ranges would overlap.
        if not self.config.nodes and (self.config.prefill_groups or self.config.decode_groups):
            logger.info("[OPT] _apply_ep_split: legacy format detected, apply ascend_base_port offset")
            self._apply_legacy_ascend_offset()
            self._validate_ascend_ports()
            return

        prefill_ep = self._prefill_tp_size * self._prefill_dp_size
        decode_ep = self._decode_tp_size * self._decode_dp_size

        self.config.prefill_groups = self._split_pool(
            [n for n in self.config.nodes if n.role == "prefill"], prefill_ep, "P", self.config.prefill_instances
        )
        self.config.decode_groups = self._split_pool(
            [n for n in self.config.nodes if n.role == "decode"], decode_ep, "D", self.config.decode_instances
        )

        proxy_nodes = [n for n in self.config.nodes if n.role == "proxy"]
        if proxy_nodes:
            self.config.proxy = proxy_nodes[0]
            logger.info(
                f"[OPT] proxy extracted from node pool: {self.config.proxy.ssh_ip}:{self.config.proxy.service_port}"
            )
        self._validate_ascend_ports()

    def _apply_legacy_ascend_offset(self):
        """For the legacy direct-config format, offset ascend_base_port for the
        configured prefill_groups/decode_groups nodes by the cross-group (same-host)
        cumulative GPU count.

        This stays consistent with the offset semantics in _split_pool: base
        resolution priority is node-level > instance-level > default 20000, and the
        offset equals the number of GPUs already allocated to that machine in earlier
        groups. Traverse in prefill→decode order, accumulating by machine IP
        (bind_ip or ssh_ip) as the key; cross-role same-host allocations are also
        accumulated, so every vLLM process on the same machine gets a non-overlapping base.

        Note: under legacy, service_port/kv_port/rpc_port/gpu_ids are configured
        explicitly per node by the user in config.toml; this method does not modify
        them, it only offsets ascend_base_port.
        """
        inst_base = getattr(self.config, "ascend_base_port", None)
        # Machine IP -> cumulative number of GPUs already allocated to that machine in earlier groups
        machine_gpu_offset: Dict[str, int] = {}

        for groups_attr in ("prefill_groups", "decode_groups"):
            groups = getattr(self.config, groups_attr)
            new_groups = []
            for grp in groups:
                new_nodes = []
                for node in grp.nodes:
                    ip = node.bind_ip or node.ssh_ip
                    offset = machine_gpu_offset.get(ip, 0)
                    # Base resolution priority: node-level > instance-level > default 20000
                    raw_base = node.ascend_base_port if node.ascend_base_port is not None else inst_base
                    resolved_base = raw_base if raw_base is not None else self.config.ASCEND_DEFAULT_BASE_PORT
                    ascend_base = resolved_base + offset
                    new_nodes.append(node.model_copy(update={"ascend_base_port": ascend_base}))
                    machine_gpu_offset[ip] = offset + len(node.gpu_ids)
                new_groups.append(grp.model_copy(update={"nodes": new_nodes}))
            setattr(self.config, groups_attr, new_groups)

        logger.info(
            "[OPT] legacy ascend_base_port offset applied: "
            f"{len(self.config.prefill_groups)} prefill group(s), "
            f"{len(self.config.decode_groups)} decode group(s)"
        )

    def _validate_ascend_ports(self):
        """Detect whether ASCEND Direct Transport port ranges overlap within this config.

        Multiple vLLM processes on the same host (same bind_ip/ssh_ip) trigger a warn
        if their base ranges overlap. The same base across different hosts is legal
        (each server has its own independent port space) and is not flagged. Only the
        multi-group case within this config is covered; same-host multi-instance
        setups across independent processes cannot be validated here and rely on
        instance-level fields plus documentation constraints.
        """
        spans = []  # (ip, base, rank_count, role, group_idx)
        for role, groups in (("P", self.config.prefill_groups), ("D", self.config.decode_groups)):
            for gi, grp in enumerate(groups):
                for n in grp.nodes:
                    ip = n.bind_ip or n.ssh_ip
                    ranks = len(n.gpu_ids)
                    base = (
                        n.ascend_base_port if n.ascend_base_port is not None else self.config.ASCEND_DEFAULT_BASE_PORT
                    )
                    spans.append((ip, base, ranks, role, gi))
        by_ip = defaultdict(list)
        for ip, base, ranks, role, gi in spans:
            by_ip[ip].append((base, ranks, role, gi))
        for ip, items in by_ip.items():
            items.sort()
            for i in range(len(items) - 1):
                b1, w1, r1, g1 = items[i]
                b2, w2, r2, g2 = items[i + 1]
                if b1 + w1 > b2:  # overlap
                    logger.warning(
                        "[OPT] ASCEND port overlap on %s: %s group%d "
                        "[base=%d,+%d) vs %s group%d [base=%d,+%d). "
                        "调大 ascend_base_port 间距.",
                        ip,
                        r1,
                        g1,
                        b1,
                        w1,
                        r2,
                        g2,
                        b2,
                        w2,
                    )

    def _split_pool(self, pool: list, ep_size: int, role: str, instances: Optional[int] = None) -> List[PDGroup]:
        """Split the node pool into groups of ep_size cards each, returning a list of PDGroups.

        Cross-node teaming is supported: when ep_size exceeds the number of cards on
        a single node, multiple nodes are merged into one group.

        instances specifies the desired number of groups (instances). When None it
        is derived automatically as total_gpus // ep_size; when specified explicitly
        it must satisfy instances * ep_size == total_gpus, otherwise an error is raised.
        """
        if not pool:
            return []
        if ep_size <= 0:
            raise ValueError(f"{role} ep_size must be positive, got {ep_size}")

        total_gpus = sum(len(n.gpu_ids) for n in pool)
        if total_gpus < ep_size:
            raise ValueError(f"{role} pool has {total_gpus} GPUs total but ep_size={ep_size}, cannot form a group")
        if total_gpus % ep_size != 0:
            raise ValueError(f"{role} pool has {total_gpus} GPUs total which is not a multiple of ep_size={ep_size}")

        groups: List[PDGroup] = []
        global_group_idx = 0
        role_offset = 0 if role == "P" else 1000

        if instances is not None:
            if instances <= 0:
                raise ValueError(f"{role} instances must be positive, got {instances}")
            if instances * ep_size != total_gpus:
                raise ValueError(
                    f"{role} instances={instances} * ep_size={ep_size} = "
                    f"{instances * ep_size} != total_gpus={total_gpus}, "
                    f"无法整除分配"
                )
            num_groups = instances
        else:
            num_groups = total_gpus // ep_size

        # Flatten all nodes' GPUs, keeping node references for cross-node teaming
        gpu_queue = []
        for node in pool:
            for gpu_id in node.gpu_ids:
                gpu_queue.append((node, gpu_id))

        # Track each node's cumulative allocated GPU count across groups, used to compute port/kv_port offsets
        node_gpu_offset: Dict[int, int] = {}
        # ascend_base_port offset parameter: inst_base is the instance-level base
        inst_base = getattr(self.config, "ascend_base_port", None)

        for g in range(num_groups):
            start = g * ep_size
            end = start + ep_size
            slice_pairs = gpu_queue[start:end]

            # Aggregate by node; each node keeps the gpu_ids allocated to it
            node_map = {}  # id(node) -> (node, [gpu_ids])
            for node, gpu_id in slice_pairs:
                key = id(node)
                if key not in node_map:
                    node_map[key] = (node, [])
                node_map[key][1].append(gpu_id)

            group_nodes = []
            for node, gpus in node_map.values():
                key = id(node)
                offset = node_gpu_offset.get(key, 0)
                # Base resolution priority: node-level > instance-level > default 20000
                raw_base = node.ascend_base_port if node.ascend_base_port is not None else inst_base
                resolved_base = raw_base if raw_base is not None else self.config.ASCEND_DEFAULT_BASE_PORT
                ascend_base = resolved_base + offset
                group_node = node.model_copy(
                    update={
                        "gpu_ids": gpus,
                        "service_port": node.service_port + offset,
                        "kv_port": node.kv_port + offset,
                        "rpc_port": node.rpc_port + offset,
                        "ascend_base_port": ascend_base,
                    }
                )
                group_nodes.append(group_node)
                node_gpu_offset[key] = offset + len(gpus)

            groups.append(
                PDGroup(
                    dp_rpc_port=12345 + role_offset + global_group_idx,
                    nodes=group_nodes,
                    env=dict(group_nodes[0].env) if group_nodes else {},
                )
            )
            global_group_idx += 1

        logger.info(f"[OPT] {role}: split {len(pool)} node(s) into {len(groups)} group(s) (ep_size={ep_size})")
        return groups

    def _add_param(self, env_vars: dict, param_name: str, value):
        env_vars[param_name] = str(value)

    @staticmethod
    def _resolve_enum_value(param) -> object:
        """The real value of enum / le_enum fields is already selected by map_param_with_value
        during the forward mapping stage, and range_to_enum aligns value to a legal candidate
        during conversion, so the consumer returns param.value directly without any reverse
        mapping (reverse mapping is handled uniformly by the core field_to_param).
        """
        return param.value

    def _get_round_dir(self) -> Path:
        """Return this round's script staging directory (fixed remote_dir path).

        Pinned to the per-cluster remote_dir so that, on a node that also runs the
        optimizer (local service node), Stage 0/Phase 0 script generation prepares
        the very directory the service reads from, letting the upload step self-skip.
        The directory is (re)created idempotently on every call (surviving a
        mid-round _clear_remote_dirs) and is cleared each round by
        _reset_round_dir(); stop() deletes it via _cleanup_round_dir().
        """
        if self._round_tmp_dir is None:
            self._round_tmp_dir = Path(self._remote_dir)
            logger.info(f"[OPT] round dir: {self._round_tmp_dir}")
        self._round_tmp_dir.mkdir(parents=True, exist_ok=True)
        return self._round_tmp_dir

    def _clear_remote_dirs(self):
        """Pre-Stage-0 sweep: delete each service node's remote_dir if it exists.

        Runs over SSH (incl. ssh-to-self for a local service node) so the subsequent
        upload sees a missing dir and re-uploads fresh scripts; a local service
        node's dir is regenerated by Stage 0 and self-skipped at upload. Must run
        before Stage 0 (which repopulates the local round_dir) and must NOT run
        inside Stage 2/3, which depend on remote_dir being present.
        """
        seen: set = set()
        for node in self._all_nodes():
            key = self._container_key(node)
            if key in seen:
                continue
            seen.add(key)
            try:
                executor = self._exec_for_node(node)
                executor.run(
                    f"if [ -d {shlex.quote(self._remote_dir)} ]; then rm -rf {shlex.quote(self._remote_dir)}; fi",
                    hide=True,
                    warn=True,
                    timeout=self._ssh_cmd_timeout,
                )
            except Exception as e:
                logger.warning(f"[{key}] clear remote dir failed: {e}")

    def _reset_round_dir(self) -> None:
        """Called before a new round starts: unconditionally purge the previous round's
        local round dir (bypassing the bak_path guard) so Stage 0 generates into a clean
        directory, then clear the cache.

        与 stop()->_cleanup_round_dir()（bak_path 空时保留失败日志）不同，此处的 round-start
        清理是无条件的：上一轮被守卫保留的失败 round dir 会被这里删掉。需跨轮保留的失败日志
        应通过配置 bak_path 由 backup() 归档。
        """
        self._purge_round_dir()
        self._round_tmp_dir = None
        self._round_log_dir = None

    def _cleanup_round_dir(self) -> None:
        """Delete this round's temp directory (scripts + logs). Idempotent; safe when the directory does not exist or is None.

        bak_path 守卫：bak_path 为空或不存在时保留 round dir 作为日志唯一副本，不误删
        （与 backup() 文档语义一致：bak_path 非空时才由 stop 删除，否则保留）。
        """
        d = getattr(self, '_round_tmp_dir', None)
        if not d:
            return
        bak = getattr(self, 'bak_path', None)
        if not bak or not Path(bak).exists():
            logger.info(f"[OPT] keep round tmp dir {d} (bak_path empty/missing; sole log copy)")
            return
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            logger.warning(f"[OPT] failed to clean round tmp dir {d}: {e}")
        finally:
            self._round_tmp_dir = None

    def _purge_round_dir(self) -> None:
        """Unconditionally delete this round's local temp dir (scripts + logs), ignoring
        the bak_path guard. Called at round start (before Stage 0) to guarantee a clean
        slate regardless of whether stop() retained the dir.

        清理前打印 round dir 路径，清理后打印结果。_round_tmp_dir 为 None（已被 stop 删除）时
        为 no-op（仅记日志）。
        """
        d = getattr(self, '_round_tmp_dir', None)
        if not d:
            logger.info("[OPT] no round dir cached, skip purge")
            return
        logger.info(f"[OPT] purge round dir before generate scripts: {d}")
        try:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"[OPT] round dir purged: {d}")
            else:
                logger.info(f"[OPT] round dir not exists, nothing to purge: {d}")
        except Exception as e:
            logger.warning(f"[OPT] failed to purge round dir {d}: {e}")

    @property
    def _remote_dir(self) -> str:
        """Per-cluster remote working dir, isolating concurrent clusters."""
        return f"{self._remote_tmp_dir}/vllm_pd_{self._cluster_id}"

    def _exec_for_node(self, node) -> SshRemote:
        """Get or create the node's SshRemote (cached by container_key)."""
        key = SshRemote.from_node(node).container_key
        if key not in self._executors:
            self._executors[key] = SshRemote.from_node(
                node, docker_use_sudo=getattr(node, 'docker_use_sudo', False), ssh_command_timeout=self._ssh_cmd_timeout
            )
        return self._executors[key]

    def _conn_for_node(self, node):
        """Get or create the node's SshRemote (cached) and return its Connection."""
        return self._exec_for_node(node).conn

    def _build_context(
        self, role: str, node, dp_rank: int, gpu_ids: list, port: int, kv_port: int, rpc_port: int, group_idx: int = 0
    ) -> dict:
        groups = self.config.prefill_groups if role == "P" else self.config.decode_groups
        group = groups[group_idx] if group_idx < len(groups) else groups[0]

        node_ext = {
            "dp_rank": dp_rank,
            "port": port,
            "service_port": port,
            "kv_port": kv_port,
            "rpc_port": rpc_port,
            "gpu_ids": ",".join(str(g) for g in gpu_ids),
            "engine_id": (self._engine_id_for(role, group_idx, dp_rank)),
        }
        current_node = SimpleNamespace(
            **{
                **{
                    k: getattr(node, k)
                    for k in (
                        "ssh_ip",
                        "ssh_port",
                        "ssh_user",
                        "bind_ip",
                        "port",
                        "gpu_ids",
                        "kv_port",
                        "rpc_port",
                        "network_interface",
                        "hccl_if_ip",
                        "docker_container_id",
                        "dp_rpc_port",
                        "ascend_base_port",
                    )
                    if hasattr(node, k)
                },
                **node_ext,
            }
        )

        # Aggregate all top-level vllm_pd fields
        vllm_pd_dict = {}
        for f in self.config.model_fields:
            vllm_pd_dict[f] = getattr(self.config, f)

        # The four TP/DP sizes are instance attributes (injected by tuning params);
        # inject them manually for template rendering
        vllm_pd_dict["prefill_tp_size"] = self._prefill_tp_size
        vllm_pd_dict["prefill_dp_size"] = self._prefill_dp_size
        vllm_pd_dict["decode_tp_size"] = self._decode_tp_size
        vllm_pd_dict["decode_dp_size"] = self._decode_dp_size

        # is_moe_model is now an instance attribute; inject it manually for template rendering
        vllm_pd_dict["is_moe_model"] = self._is_moe

        if not self._is_moe:
            vllm_pd_dict["prefill_dp_size"] = 1
            vllm_pd_dict["decode_dp_size"] = 1

        dp_size = vllm_pd_dict["prefill_dp_size"] if role == "P" else vllm_pd_dict["decode_dp_size"]
        vllm_pd_dict["enable_dp_rank"] = self._moe_dp_enabled(dp_size)

        # atb_llm_hccl/lcoc: all nodes share the same service_ip -> hccl=0/lcoc=1; different -> hccl=1/lcoc=0
        all_ips = {
            (n.bind_ip or n.ssh_ip)
            for grp in (self.config.prefill_groups + self.config.decode_groups)
            for n in grp.nodes
        }
        if len(all_ips) <= 1:
            vllm_pd_dict["atb_llm_hccl_enable"] = 0
            vllm_pd_dict["atb_llm_lcoc_enable"] = 1
        else:
            vllm_pd_dict["atb_llm_hccl_enable"] = 1
            vllm_pd_dict["atb_llm_lcoc_enable"] = 0

        extra_env = self._prefill_env_vars if role == "P" else self._decode_env_vars
        # Three-layer env merge: role env (lowest) < group env (per-node) < PSO env (highest)
        role_env = self.config.env_prefill if role == "P" else self.config.env_decode
        merged_static = {**role_env, **group.env}
        static_env = {k: v for k, v in merged_static.items() if k not in extra_env}
        env_targets = "\n".join(f"export {k}={shell_quote(v)}" for k, v in extra_env.items())
        env_lines = [f"export {k}={shell_quote(v)}" for k, v in static_env.items()]
        env_targets = "\n".join(env_lines + [env_targets]) if env_targets else "\n".join(env_lines)

        extra_run = self._prefill_run_vars if role == "P" else self._decode_run_vars
        run_lines = []
        for param_name, value in extra_run.items():
            sval = str(value)
            if sval == "":
                # 1) 空值：忽略该参数，不拼接（flag 请改用 value="--xxx" 形式）
                continue
            if sval.startswith("--"):
                # 2) flag 型：value 本身即完整 CLI 片段（如 --enforce-eager），直接拼接
                run_lines.append(f"    {sval}")
            else:
                # 3) 普通 key-value：告警（若本意是 flag，请用 -- 前缀），仍按 --key value 拼接保持兼容
                cli_arg = "--" + param_name.lower().replace("_", "-")
                run_lines.append(f"    {cli_arg} {shell_quote(value)}")
        run_targets = " \\\n".join(run_lines)

        run_envs_lines = []
        for env_key, env_val in os.environ.items():
            if env_key.startswith("PD_") and env_key not in extra_env:
                run_envs_lines.append(f"export {env_key}={shell_quote(env_val)}")
        vllm_pd_dict["run_envs"] = "\n".join(run_envs_lines)

        partial_context = {
            "vllm_pd": SimpleNamespace(**vllm_pd_dict),
            "vllm.pso": SimpleNamespace(env_targets=env_targets, run_targets=run_targets),
            "current_group": SimpleNamespace(
                dp_address=group.nodes[0].bind_ip or group.nodes[0].ssh_ip if group.nodes else "127.0.0.1",
                dp_rpc_port=group.dp_rpc_port,
            ),
            "current_node": current_node,
        }
        rendered_others = render_template(self.vllm_others, partial_context)

        vllm_command_ns = SimpleNamespace(
            model=self.model_path,
            served_model_name=self.served_model_name,
            others=rendered_others,
        )
        partial_context["vllm.command"] = vllm_command_ns
        return partial_context

    def _engine_id_for(self, role: str, group_idx: int = 0, dp_rank: int = 0) -> str:
        if self._is_moe:
            return f"{'prefill' if role == 'P' else 'decode'}_instance_{group_idx}"
        return f"{role}-{dp_rank}"

    def _moe_dp_enabled(self, dp_size: int) -> bool:
        return self._is_moe and dp_size > 1

    def _build_run_shell(
        self, role: str, node, dp_rank: int, gpu_ids: list, port: int, kv_port: int, rpc_port: int, group_idx: int = 0
    ) -> str:
        """Generate the full shell script from a template (the rendered run script)."""
        scripts_dir = self._get_scripts_dir()

        template_name = "run_pd_prefill.sh" if role == "P" else "run_pd_decode.sh"
        template_content = (Path(scripts_dir) / template_name).read_text()

        context = self._build_context(role, node, dp_rank, gpu_ids, port, kv_port, rpc_port, group_idx)
        rendered = render_template(template_content, context)
        rendered = _clean_rendered_shell(rendered)

        parts = [
            "#!/bin/bash",
            f"# Auto-generated for {role}-R{dp_rank}",
            "",
            rendered,
        ]
        return "\n".join(parts)

    def _build_proxy_shell(self, p_instances, d_instances) -> str:
        """Build the Proxy startup script."""
        proxy = self.config.proxy
        scripts_dir = self._get_scripts_dir()

        extra_env = {}
        env_targets = "\n".join(f"export {k}={shell_quote(v)}" for k, v in extra_env.items())
        run_envs_lines = []
        for env_key, env_val in os.environ.items():
            if env_key.startswith("PD_") and env_key not in extra_env:
                run_envs_lines.append(f"export {env_key}={shell_quote(env_val)}")
        run_envs = "\n".join(run_envs_lines)

        context = {
            "vllm_pd.proxy": proxy,
            "vllm.pso.env_targets": env_targets,
            "vllm_pd.run_envs": run_envs,
            "p_hosts": " ".join([ip for ip, _ in p_instances]),
            "p_ports": " ".join([str(port) for _, port in p_instances]),
            "d_hosts": " ".join([ip for ip, _ in d_instances]),
            "d_ports": " ".join([str(port) for _, port in d_instances]),
        }
        template_content = (Path(scripts_dir) / "run_pd_proxy.sh").read_text()
        rendered = render_template(template_content, context)
        rendered = _clean_rendered_shell(rendered)

        return "\n".join(
            [
                "#!/bin/bash",
                "# Auto-generated proxy",
                "",
                rendered,
            ]
        )

    def _exec_remote(self, node_config, node_label: str, port: int = None, log_file: str = None, append: bool = False):
        """Used in stage 3: execute the already-uploaded main script remotely and return (pid, log_file).

        log_file/append are used by restart() to reuse the existing log file (append
        mode) instead of creating a new random one.
        """
        executor = self._exec_for_node(node_config)
        remote_script = f"{self._remote_dir}/{node_label}.sh"

        inner_cmd = f"bash -l {shlex.quote(remote_script)}"

        return executor.background(
            inner_cmd,
            node_label,
            self._remote_pids,
            self._remote_pid_nodes,
            node_config,
            log_file=log_file,
            append=append,
        )

    def _container_key(self, node) -> str:
        return self._exec_for_node(node).container_key

    @staticmethod
    def _exec_suffix(node) -> str:
        cid = getattr(node, 'docker_container_id', None) or "none"
        return f"{node.ssh_ip}_{node.ssh_port}_{cid}"

    @staticmethod
    def _node_identity(node) -> str:
        return SshRemote.from_node(node).container_key

    def _generate_all_scripts(self):
        round_dir = self._get_round_dir()
        round_dir.mkdir(parents=True, exist_ok=True)

        scripts_dir = Path(self._get_scripts_dir())
        # proxy scripts validated at __init__ (no auto-download here)
        copy_files = [
            "check_pd_process.sh",
            "load_balance_proxy_server_example.py",
            "load_balance_proxy_layerwise_server_example.py",
            "stop_pd_process.sh",
            "pd-result-reproduction.md",
            "net_traffic.sh",
        ]
        for name in copy_files:
            src = scripts_dir / name
            if src.exists():
                dst = round_dir / name
                if not dst.exists():
                    dst.write_text(src.read_text())

        p_instances: List[Tuple[str, int]] = []
        d_instances: List[Tuple[str, int]] = []
        node_infos: list = []

        node_groups = [
            ("P", self.config.prefill_groups, self._prefill_tp_size, self._prefill_dp_size, p_instances),
            ("D", self.config.decode_groups, self._decode_tp_size, self._decode_dp_size, d_instances),
        ]
        for role, groups, tp_size, dp_size, instances_list in node_groups:
            for group_idx, group in enumerate(groups):
                dp_rank_counter = 0
                for node in group.nodes:
                    if dp_rank_counter >= dp_size:
                        logger.info(f"[{role}-I{group_idx}] dp_size={dp_size} reached, skip node {node.ssh_ip}")
                        continue
                    node_gpu_count = len(node.gpu_ids)
                    max_slots = max(dp_size - dp_rank_counter, 0)
                    per_node_processes = min(node_gpu_count // tp_size, max_slots)
                    if per_node_processes == 0:
                        logger.warning(
                            f"[{role}-I{group_idx}] node {node.ssh_ip} has {node_gpu_count} GPUs "
                            f"but tp_size={tp_size}, skipping"
                        )
                        continue
                    for local_rank in range(per_node_processes):
                        res = self._calc_process_resources(role, node, local_rank)
                        global_dp_rank = dp_rank_counter + local_rank
                        bind_ip = node.bind_ip or node.ssh_ip
                        ep_size = tp_size * dp_size
                        label = f"{role}-I{group_idx}-R{global_dp_rank}-T{tp_size}-D{dp_size}-EP{ep_size}_{self._exec_suffix(node)}"
                        shell_content = self._build_run_shell(
                            role, node, dp_rank=global_dp_rank, group_idx=group_idx, **res
                        )
                        local_script = round_dir / f"{label}.sh"
                        local_script.write_text(shell_content)
                        instances_list.append((bind_ip, res['port']))
                        node_infos.append(
                            {
                                "label": label,
                                "pid": None,
                                "conn": None,
                                "log_file": None,
                                "bind_ip": bind_ip,
                                "port": res['port'],
                                "host": f"{node.ssh_ip}:{node.ssh_port}",
                                "_node_config": node,
                                "_dp_rank": global_dp_rank,
                                "_local_dp_rank": local_rank,
                            }
                        )
                    dp_rank_counter += per_node_processes

        proxy = self.config.proxy
        proxy_bind_ip = proxy.bind_ip
        proxy_content = self._build_proxy_shell(p_instances, d_instances)
        proxy_label = f"proxy_{self._exec_suffix(proxy)}"
        local_proxy_script = round_dir / f"{proxy_label}.sh"
        local_proxy_script.write_text(proxy_content)
        node_infos.append(
            {
                "label": proxy_label,
                "pid": None,
                "conn": None,
                "log_file": None,
                "bind_ip": proxy_bind_ip,
                "port": proxy.service_port,
                "host": f"{proxy.ssh_ip}:{proxy.ssh_port}",
                "_node_config": proxy,
                "_dp_rank": 0,
            }
        )

        return node_infos

    def _upload_all_scripts(self):
        round_dir = self._get_round_dir()
        remote_dir = self._remote_dir

        seen: set = set()
        for node in self._all_nodes():
            key = self._container_key(node)
            if key in seen:
                continue
            seen.add(key)

            try:
                executor = self._exec_for_node(node)
                r = executor.run(
                    f"[ -d {shlex.quote(remote_dir)} ]", hide=True, warn=True, timeout=self._ssh_cmd_timeout
                )
                if r.ok:
                    logger.info(f"[{key}] remote dir exists, skip upload")
                    continue
                executor.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True, warn=True, timeout=self._ssh_cmd_timeout)
                for f in round_dir.iterdir():
                    if f.is_file():
                        executor.put(str(f), f"{remote_dir}/{f.name}")
                logger.info(f"[{key}] scp done")
            except Exception as e:
                logger.warning(f"[{key}] upload error: {e}")

    def _stop_node(self, node, *, label: str = "") -> object:
        """Stop residual processes on a single node via stop_pd_process.sh.
        Shared by _cleanup_all_nodes (full cleanup) and _kill_all_processes (restart kill).
        """
        executor = self._exec_for_node(node)
        gpu_ids = ",".join(str(g) for g in getattr(node, "gpu_ids", []))
        proxy_port = str(getattr(self.config.proxy, "service_port", "")) if getattr(self.config, "proxy", None) else ""
        extra_args = ""
        if gpu_ids:
            extra_args += f" --gpus {gpu_ids}"
        if proxy_port:
            extra_args += f" --port {proxy_port}"
        stop_grace = getattr(self.config, "stop_grace_timeout", 30)
        stop_kill = getattr(self.config, "stop_kill_timeout", 10)
        stop_timeout = self._ssh_cmd_timeout + stop_grace + stop_kill
        return executor.run(
            f'STOP_GRACE_TIMEOUT={stop_grace} STOP_KILL_TIMEOUT={stop_kill} '
            f'REMOTE_DIR={shlex.quote(self._remote_dir)} bash {shlex.quote(self._remote_dir + "/stop_pd_process.sh")}{extra_args}',
            hide=True,
            warn=True,
            timeout=stop_timeout,
        )

    def _cleanup_all_nodes(self):
        """Stage 2: check for residual processes on all remote servers and force-stop them.

        Only the GPUs and proxy port that this node is about to use are checked, to avoid
        killing unrelated processes. Deduplication is independent per container.
        """
        seen: set = set()
        for node in self._all_nodes():
            key = self._container_key(node)
            if key in seen:
                continue
            seen.add(key)
            try:
                executor = self._exec_for_node(node)
                gpu_ids = ",".join(str(g) for g in getattr(node, "gpu_ids", []))
                proxy_port = (
                    str(getattr(self.config.proxy, "service_port", "")) if getattr(self.config, "proxy", None) else ""
                )
                extra_args = ""
                if gpu_ids:
                    extra_args += f" --gpus {gpu_ids}"
                if proxy_port:
                    extra_args += f" --port {proxy_port}"
                check = executor.run(
                    f'REMOTE_DIR={shlex.quote(self._remote_dir)} bash {shlex.quote(self._remote_dir + "/check_pd_process.sh")}{extra_args}',
                    hide=True,
                    warn=True,
                    timeout=self._ssh_cmd_timeout,
                )
                if check.stdout.strip():
                    logger.warning(f"[{key}] residual processes:\n{check.stdout.strip()}")
                # stop（复用 _stop_node helper，返回 stop 结果供返回码检查）
                stop = self._stop_node(node, label=key)
                logger.info(f"[{key}] cleanup:\n{stop.stdout.strip()}")
                if getattr(stop, "failed", False) or getattr(stop, "returncode", 0) != 0:
                    logger.error(
                        f"[{key}] stop did NOT fully release ports/processes "
                        f"(rc={getattr(stop, 'returncode', '?')}); may cause false-ready. "
                        f"stdout={stop.stdout.strip()[-300:]}"
                    )
                # 宿主侧端口释放复核（docker 防御）：docker 场景 stop 在容器内执行，
                # 容器端口释放后宿主映射端口（docker userland proxy）有亚秒级拆除延迟，
                # 而 wait_simulate 探活在宿主侧 curl 127.0.0.1:{mapped_port}。补一次宿主侧复核。
                if proxy_port:
                    host_check = executor.run(
                        f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 2 http://127.0.0.1:{proxy_port}/health || true",
                        container_exec=False,
                        hide=True,
                        warn=True,
                        timeout=10,
                    )
                    code = (host_check.stdout or "").strip().strip("'")
                    if code == "200":
                        logger.error(
                            f"[{key}] proxy port {proxy_port} still serves 200 after stop "
                            f"(docker proxy lag or residual process); will likely false-ready"
                        )
            except Exception as e:
                logger.warning(f"[{key}] cleanup failed: {e}")

    def _all_nodes(self):
        """Return the list of all remote node configs (P + D + Proxy)."""
        nodes = []
        for g in self.config.prefill_groups:
            nodes.extend(g.nodes)
        for g in self.config.decode_groups:
            nodes.extend(g.nodes)
        nodes.append(self.config.proxy)
        return nodes

    def _run_remote(self, node_config, shell_content: str, node_label: str, log_file: str = None, append: bool = False):
        """Generate script -> save locally -> scp to remote -> run in background; return (pid, log_file).

        log_file/append forward to executor.background(): when set, the relaunch
        reuses the existing log file (append mode) instead of a new random one.
        """
        executor = self._exec_for_node(node_config)

        round_dir = self._get_round_dir()
        round_dir.mkdir(parents=True, exist_ok=True)
        local_script = round_dir / f"{node_label}.sh"
        local_script.write_text(shell_content)
        logger.info(f"[{node_label}] script saved to {local_script}")

        remote_script = f"{self._remote_dir}/{node_label}.sh"
        try:
            executor.put(str(local_script), remote_script)
        except Exception as e:
            logger.error(f"[{node_label}] failed to upload script: {e}")
            return None, ""

        inner_cmd = f"bash -l {shlex.quote(remote_script)}"

        return executor.background(
            inner_cmd,
            node_label,
            self._remote_pids,
            self._remote_pid_nodes,
            node_config,
            log_file=log_file,
            append=append,
        )

    def _calc_process_resources(self, role: str, node, dp_rank: int):
        """Compute a single process's GPU/port/KV-port resources based on role and dp_rank."""
        tp_size = self._prefill_tp_size if role == "P" else self._decode_tp_size
        start_gpu = dp_rank * tp_size
        return {
            "gpu_ids": node.gpu_ids[start_gpu : start_gpu + tp_size],
            "port": node.service_port + dp_rank,
            "kv_port": node.kv_port + dp_rank * tp_size,
            "rpc_port": node.rpc_port + dp_rank,
        }

    def _cleanup_node(self, node, port: int | None = None, *, label: str = "") -> None:
        """Single-node cleanup before restart: stop residual processes on this node's
        GPUs/ports, wait for port release. Only touches this node's resources, does not
        affect other nodes.

        Called by _restart_node before relaunching: the old main process has EXITED but
        child workers (VLLM::Worker) may still hold GPU memory / ports. stop_pd_process.sh
        with --gpus + --port targets only this node's resources.
        """
        try:
            executor = self._exec_for_node(node)
            gpu_ids = ",".join(str(g) for g in getattr(node, "gpu_ids", []))
            proxy_port = str(port) if port else ""
            extra_args = ""
            if gpu_ids:
                extra_args += f" --gpus {gpu_ids}"
            if proxy_port:
                extra_args += f" --port {proxy_port}"
            stop_grace = getattr(self.config, "stop_grace_timeout", 30)
            stop_kill = getattr(self.config, "stop_kill_timeout", 10)
            stop_timeout = self._ssh_cmd_timeout + stop_grace + stop_kill
            logger.info(f"[{getattr(node, 'ssh_ip', '?')}] cleanup before restart: gpus={gpu_ids} port={proxy_port}")
            stop = executor.run(
                f'STOP_GRACE_TIMEOUT={stop_grace} STOP_KILL_TIMEOUT={stop_kill} '
                f'REMOTE_DIR={shlex.quote(self._remote_dir)} bash {shlex.quote(self._remote_dir + "/stop_pd_process.sh")}{extra_args}',
                hide=True,
                warn=True,
                timeout=stop_timeout,
            )
            logger.info(f"[{getattr(node, 'ssh_ip', '?')}] cleanup done: {stop.stdout.strip()[-200:]}")
            if getattr(stop, "failed", False) or getattr(stop, "returncode", 0) != 0:
                logger.error(
                    f"[{getattr(node, 'ssh_ip', '?')}] cleanup did NOT fully release "
                    f"ports/processes (rc={getattr(stop, 'returncode', '?')}); "
                    f"restart may hit port conflict. stdout={stop.stdout.strip()[-300:]}"
                )
            # docker 场景宿主侧复核：仅 proxy 端口映射到宿主（P/D 的 /health 在容器内，
            # 宿主侧 curl 不通属正常），与 _cleanup_all_nodes 的宿主侧复核保持一致。
            if proxy_port and label.startswith("proxy_"):
                host_check = executor.run(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 2 http://127.0.0.1:{proxy_port}/health || true",
                    container_exec=False,
                    hide=True,
                    warn=True,
                    timeout=10,
                )
                code = (host_check.stdout or "").strip().strip("'")
                if code == "200":
                    logger.error(
                        f"[{getattr(node, 'ssh_ip', '?')}] port {proxy_port} still serves "
                        f"200 after cleanup; restart may fail"
                    )
        except Exception as e:  # noqa: BLE001 - cleanup failure must not block restart; warn only
            logger.warning(f"[{getattr(node, 'ssh_ip', '?')}] cleanup failed: {e}")

    def _restart_node(self, info: dict, node_infos: list = None) -> Optional[int]:
        label = info["label"]
        node = info.get("_node_config")
        if node is None:
            logger.error(f"[{label}] no node config for restart")
            return None

        # Reuse the existing log file (append mode) instead of creating a new random one.
        # Append a retry separator so each relaunch's output is delimited in the same file.
        old_log = info.get("log_file")
        retry_n = info.get("_restart_count", 0) + 1
        if old_log:
            sep = f"---------------------retry {retry_n}------------------------"
            executor = self._exec_for_node(node)
            try:
                executor.run(
                    f"echo {shlex.quote(sep)} >> {shlex.quote(old_log)}",
                    hide=True,
                    warn=True,
                    timeout=self._ssh_cmd_timeout,
                )
            except Exception:  # nosec B110
                pass

        dp_rank = info.get("_dp_rank", 0)
        label_parts = label.split("-")
        group_idx = int(label_parts[1][1:]) if len(label_parts) > 1 and label_parts[1].startswith("I") else 0

        # 重试前清场：旧主进程已 EXITED，但子进程（VLLM::Worker）可能仍持有显存/端口；
        # 先对目标节点执行 stop_pd_process.sh（--gpus + --port 精准清场，不误杀其他节点），
        # 等端口释放后再拉新进程，避免 Address already in use / OOM。
        self._cleanup_node(node, port=info.get("port"), label=info.get("label", ""))

        if label.startswith("P-") or label.startswith("D-"):
            role = label[0]
            local_dp_rank = info.get("_local_dp_rank", dp_rank)
            res = self._calc_process_resources(role, node, local_dp_rank)
            shell_content = self._build_run_shell(role, node, dp_rank=dp_rank, group_idx=group_idx, **res)
            new_pid, _ = self._run_remote(node, shell_content, label, log_file=old_log, append=True)
        elif label.startswith("proxy_"):
            p_instances, d_instances = [], []
            all_infos = node_infos or getattr(self, '_node_infos', []) or []
            for ni in all_infos:
                if ni.get("bind_ip") and ni.get("port") and not ni["label"].startswith("proxy_"):
                    if ni["label"].startswith("P-"):
                        p_instances.append((ni["bind_ip"], ni["port"]))
                    elif ni["label"].startswith("D-"):
                        d_instances.append((ni["bind_ip"], ni["port"]))
            new_pid, _ = self._start_proxy(p_instances, d_instances, log_file=old_log, append=True)
        else:
            logger.error(f"[{label}] unknown node type, cannot restart")
            return None

        if new_pid is not None:
            info["pid"] = new_pid
            # log_file intentionally NOT updated: the relaunch appended to the
            # existing log file (old_log), so the path is unchanged.
            info["conn"] = self._conn_for_node(node) if hasattr(node, 'ssh_ip') else info.get("conn")
            info["_proc"] = "ALIVE"
            info["_health"] = "-"
            info["_restart_count"] = info.get("_restart_count", 0) + 1
            logger.info(f"[{label}] restarted successfully (new pid={new_pid}, restart #{info['_restart_count']})")
        else:
            logger.error(f"[{label}] restart failed")

        return new_pid

    def _wait_for_all_services(
        self, node_infos: list, timeout: int = 7200, check_interval: int = 10, max_restarts: int = 2
    ) -> bool:
        """Monitor process liveness and health status of all nodes; return True when all are ready.

        Args:
            node_infos: list of node info dicts
            timeout: total timeout in seconds
            check_interval: check interval in seconds
            max_restarts: max restarts per node (applies to ALL exited nodes,
                          including those never healthy before — a node may exit
                          during initial startup before its first health check)
        """
        deadline = time.time() + timeout
        start_time = time.time()
        healthy_nodes: set = set()
        last_table_print = time.time()

        while time.time() < deadline:
            elapsed = int(time.time() - start_time)

            for info in node_infos:
                label = info["label"]

                probe = self._probe_node(info)

                if label in healthy_nodes:
                    if probe["alive"]:
                        continue
                    logger.error(
                        f"[{label}] process exited after being healthy (pid={info['pid']}), log={info.get('log_file', '-')}"
                    )
                    healthy_nodes.discard(label)

                if not probe["alive"] and probe["proc_status"] == "EXITED":
                    log_file = info.get('log_file', '')
                    logger.error(f"[{label}] process exited (pid={info['pid']}), log={log_file}")

                    executor = self._exec_for_node(info.get("_node_config")) if info.get("_node_config") else None
                    if log_file and executor is not None:
                        tail = executor.read_file_tail(log_file, 100)
                        if tail.strip():
                            logger.error(f"[{label}] last 100 lines of log:\n{tail.strip()}")

                    restart_count = info.get("_restart_count", 0)
                    if restart_count < max_restarts and info.get("_node_config") is not None:
                        logger.warning(f"[{label}] attempting restart ({restart_count + 1}/{max_restarts})...")
                        new_pid = self._restart_node(info, node_infos)
                        if new_pid is not None:
                            continue
                        else:
                            logger.error(f"[{label}] restart failed, giving up")
                            return False
                    else:
                        logger.error(f"[{label}] no more restarts allowed (restarted {restart_count} times)")
                        return False

                if probe["healthy"]:
                    healthy_nodes.add(label)
                    bind_ip = info.get("bind_ip", "")
                    logger.info(f"[{label}] {bind_ip}:{info['port']} is ready (elapsed={elapsed}s)")
                    continue

            if len(healthy_nodes) == len(node_infos):
                logger.info(f"All {len(node_infos)} nodes healthy (elapsed={elapsed}s)")
                self._print_nodes_table(
                    node_infos,
                    healthy_nodes,
                    header=f"[{self._cluster_id}_{self._particle_count:03d}] All Nodes Status in Cluster {self._cluster_id} (elapsed={elapsed}s)",
                )
                return True

            # Print the status table once every 30s
            if time.time() - last_table_print >= 30:
                self._print_nodes_table(
                    node_infos,
                    healthy_nodes,
                    header=f"[{self._cluster_id}_{self._particle_count:03d}] All Nodes Status in Cluster {self._cluster_id} (elapsed={elapsed}s)",
                )
                last_table_print = time.time()

            time.sleep(check_interval)

        # Timeout
        unhealthy = [info["label"] for info in node_infos if info["label"] not in healthy_nodes]
        logger.error(f"Timeout waiting for nodes: {', '.join(unhealthy)}")
        return False

    def _print_nodes_table(self, node_infos: list, healthy_nodes: set = None, header: str = ""):
        """Print all node statuses in a unified table format (a single output ensures log atomicity)."""
        if healthy_nodes is None:
            healthy_nodes = set()

        # Collect each column's content and compute widths dynamically
        rows = []
        particle_str = f"{self._cluster_id}_{self._particle_count:03d}"
        for info in node_infos:
            label = info["label"]
            bind_addr = f"{info.get('bind_ip', '127.0.0.1')}:{info['port']}"
            log_file = info.get("log_file", "-")
            if label in healthy_nodes:
                proc_status = "ALIVE"
                health_status = "OK"
            else:
                proc_status = info.get("_proc", "ALIVE")
                health_status = info.get("_health", "-")
            rows.append((particle_str, label, bind_addr, proc_status, health_status, log_file))

        # Sort by (particle_id, label) for deterministic table output
        rows.sort(key=lambda r: (r[0], r[1]))

        # Column width = max(header, row contents) + 1 spacing
        headers = ("Particle", "Label", "Bind", "Alive", "Status", "LogFile")
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))
        # At least 2 spaces between columns
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

        sep_len = len(fmt.format(*headers))
        lines = [
            header,
            "=" * sep_len,
            fmt.format(*headers),
            "-" * sep_len,
        ]
        for row in rows:
            lines.append(fmt.format(*row))
        lines.append("=" * sep_len)
        logger.info("\n".join(lines))

    def _log_cluster_status(self, header: str):
        """Print the current status of all P/D/Proxy nodes (PID liveness + HTTP health)."""
        if not self._node_infos:
            logger.info(f"[OPT] {header}: no node info available")
            return
        for info in self._node_infos:
            self._probe_node(info)
        self._print_nodes_table(self._node_infos, header=header)

    def _verify_all_pids(self, node_infos: list):
        """Run a kill -0 liveness check and an HTTP health probe on all started nodes, and print a summary log."""
        self._probe_all_nodes(node_infos, raise_on_dead=True)

    def before_run(self, run_params=None):
        """Receive optimizer parameters and apply them to the P/D node config."""
        self.update_config(run_params)

    def run(self, run_params=None, **kwargs):
        logger.info(f"[OPT] run called: run_params={run_params}, kwargs={kwargs}")
        key = self._particle_key(run_params)
        if key not in self._particle_list:
            self._particle_list.append(key)
        self.before_run(run_params)
        scripts_dir = self._get_scripts_dir()
        logger.info(f"Scripts directory: {scripts_dir}")

        self._remote_pids = {}
        self._remote_pid_nodes = {}
        self._remote_logs = {}
        self._node_infos = None
        try:
            failures: List[str] = []

            # ============================================================
            # Stage 1: clean round dir (unconditional purge) + clear remote dirs
            # ============================================================
            logger.info(f"[{self._cluster_id}_{self._particle_count:03d}] Stage 1/5: cleaning round dir...")
            self._reset_round_dir()
            self._clear_remote_dirs()
            self._round_log_dir = self._get_round_dir() / "log"

            # ============================================================
            # Stage 2: generate all scripts and save them locally
            # ============================================================
            logger.info(f"[{self._cluster_id}_{self._particle_count:03d}] Stage 2/5: generating all scripts...")
            node_infos = self._generate_all_scripts()

            # ============================================================
            # Stage 3: upload all scripts to the remote servers
            # ============================================================
            logger.info(
                f"[{self._cluster_id}_{self._particle_count:03d}] Stage 3/5: uploading all scripts to remote servers..."
            )
            self._upload_all_scripts()

            # ============================================================
            # Save node_infos early so that stop() can still collect logs if an exception occurs later
            self._node_infos = node_infos
            # Stage 4: check for residual processes and force-stop them
            # ============================================================
            logger.info(
                f"[{self._cluster_id}_{self._particle_count:03d}] Stage 4/5: checking and cleaning residual processes..."
            )
            self._cleanup_all_nodes()

            # ============================================================
            # Stage 5: start all nodes + health check
            # ============================================================
            logger.info(f"[{self._cluster_id}_{self._particle_count:03d}] Stage 5/5: starting all nodes...")

            proxy = self.config.proxy
            for info in node_infos:
                node = info["_node_config"]
                conn = self._conn_for_node(node)
                info["conn"] = conn
                label = info["label"]

                logger.info(
                    f"[{label}] {getattr(node, 'ssh_user', 'root')}@{node.ssh_ip}:{node.ssh_port} "
                    f"bind={info['bind_ip']}:{info['port']}"
                )

                pid, log_file = self._exec_remote(node, label, port=info.get("port"))
                logger.info(f"[{label}] started, pid={pid}, log_file={log_file}")

                info["log_file"] = log_file
                info["pid"] = pid

                if pid is None:
                    failures.append(f"{label} ({node.ssh_ip}:{node.ssh_port}) failed to start")
                    continue

            # Check for accumulated startup failures; raise a single combined error if any
            if failures:
                raise RuntimeError(f"{len(failures)} node(s) failed to start: {'; '.join(failures)}")

            # ============================================================
            # ① Confirm all PIDs are alive + print a summary log
            # ============================================================
            self._verify_all_pids(node_infos)

            # ============================================================
            # ② Monitor all nodes' process status + health; return only when all are ready
            # ============================================================
            timeout = max(
                getattr(self.config.prefill_groups[0].nodes[0], 'timeout_seconds', 7200)
                if self.config.prefill_groups
                else 7200,
                getattr(self.config.decode_groups[0].nodes[0], 'timeout_seconds', 7200)
                if self.config.decode_groups
                else 7200,
                getattr(proxy, 'timeout_seconds', 7200),
            )
            interval = (
                getattr(self.config.prefill_groups[0].nodes[0], 'check_interval_seconds', 10)
                if self.config.prefill_groups
                else 10
            )
            logger.info(f"Waiting for all {len(node_infos)} nodes to become healthy (timeout={timeout}s)...")
            if not self._wait_for_all_services(node_infos, timeout=timeout, check_interval=interval):
                raise RuntimeError("Some nodes failed to become healthy")

            self._process_stage = ProcessState(stage=Stage.running)
            logger.info(f"[{self._cluster_id}_{self._particle_count:03d}] run completed -> stage=running")

        except Exception as e:
            logger.error(f"Failed to start vLLM PD Sep: {e}")
            # 失败路径先 backup() 归档日志/脚本（依赖 _node_infos，须在 stop() 置空前），
            # 再 stop() 清理进程；否则 stop() 清空 _node_infos 并删 round dir，日志丢失。
            try:
                self.backup()
            except Exception as be:
                logger.warning(f"[OPT] backup on failure failed: {be}")
            self.stop(del_log=False)
            raise
        finally:
            pass

    def _start_proxy(
        self,
        p_instances: List[Tuple[str, int]],
        d_instances: List[Tuple[str, int]],
        log_file: str = None,
        append: bool = False,
    ):
        """Start the Proxy and return (pid, log_file).

        log_file/append forward to executor.background(): when set, the relaunch
        reuses the existing log file (append mode) instead of a new random one.
        """
        proxy = self.config.proxy
        executor = self._exec_for_node(proxy)

        proxy_content = self._build_proxy_shell(p_instances, d_instances)

        suffix = self._exec_suffix(proxy)
        label = f"proxy_{suffix}"
        round_dir = self._get_round_dir()
        round_dir.mkdir(parents=True, exist_ok=True)
        local_script = round_dir / f"{label}.sh"
        local_script.write_text(proxy_content)
        logger.info(f"[{label}] script saved to {local_script}")

        remote_script = f"{self._remote_dir}/{label}.sh"
        scripts_dir = Path(self._get_scripts_dir())
        try:
            executor.put(str(local_script), remote_script)
            executor.put(
                str(scripts_dir / "load_balance_proxy_server_example.py"),
                f"{self._remote_dir}/load_balance_proxy_server_example.py",
            )
            executor.put(
                str(scripts_dir / "load_balance_proxy_layerwise_server_example.py"),
                f"{self._remote_dir}/load_balance_proxy_layerwise_server_example.py",
            )
        except Exception as e:
            logger.error(f"[proxy] failed to upload script: {e}")
            return None, ""

        inner_cmd = f"bash -l {shlex.quote(remote_script)}"
        return executor.background(
            inner_cmd, label, self._remote_pids, self._remote_pid_nodes, proxy, log_file=log_file, append=append
        )

    def _health_path(self, label: str) -> str:
        """Return the node's health check path."""
        return "/healthcheck" if label.startswith("proxy_") else "/health"

    def _check_remote_health(self, info: dict, label: str) -> tuple:
        """Check remote health, returning (ok, status_str). SSH mode: curl the
        health endpoint via executor (container_exec=False so curl runs on the
        host / container consistently with the SSH path).
        """
        node_cfg = info.get("_node_config")
        executor = self._exec_for_node(node_cfg) if node_cfg else None
        port = info["port"]
        health_path = self._health_path(label)
        if label.startswith("proxy_"):
            bind_ip = info.get("bind_ip", "0.0.0.0")
            check_ip = "127.0.0.1" if bind_ip == "0.0.0.0" else bind_ip
        else:
            check_ip = "127.0.0.1"
        url = f"http://{check_ip}:{port}{health_path}"
        cmd = f"curl -s -o /dev/null -w '%{{http_code}}' --noproxy '*' --max-time 5 {url}"

        conn = info.get("conn")
        try:
            if executor:
                result = executor.run(cmd, container_exec=False, hide=True, warn=True, timeout=self._ssh_cmd_timeout)
            elif conn:
                result = conn.run(cmd, hide=True, warn=True, timeout=self._ssh_cmd_timeout)
            else:
                return False, "NO_CONN"
        except Exception as e:
            return False, f"ERROR({type(e).__name__})"

        status_code = result.stdout.strip().strip("'")
        if status_code == "200":
            return True, "OK"
        elif status_code == "000":
            return False, "NOK"
        else:
            return False, status_code

    def _probe_node(self, info: dict) -> dict:
        """Run a process-liveness + HTTP-health probe on a single node, update info,
        and return a result summary.

        SSH mode: kill -0 {pid} with container_exec=False (the pid returned by
        background() is the host-side nohup wrapper, so it must be checked on the
        host, NOT inside docker exec -- see S1 lesson). proc_status is ALIVE/EXITED
        when a pid exists, UNKNOWN when pid is None. When alive, an HTTP health
        request is sent via _check_remote_health.

        Returns:
            {
                "alive": bool,          # whether the process is alive
                "healthy": bool,        # whether the HTTP health check passed
                "proc_status": str,     # "ALIVE" / "EXITED" / "UNKNOWN"
                "health_status": str,   # "OK" / "NOK" / "503" / "-"
            }
        """
        label = info["label"]
        pid = info["pid"]
        node_cfg = info.get("_node_config")
        executor = self._exec_for_node(node_cfg) if node_cfg else None

        conn = info.get("conn")

        alive = False
        if pid is not None:
            if node_cfg is not None:
                conn = self._conn_for_node(node_cfg)
                info["conn"] = conn
            if conn is not None:
                try:
                    if executor:
                        check = executor.run(
                            f"kill -0 {pid} 2>/dev/null",
                            container_exec=False,
                            hide=True,
                            warn=True,
                            timeout=self._ssh_cmd_timeout,
                        )
                    else:
                        check = conn.run(
                            f"kill -0 {pid} 2>/dev/null", hide=True, warn=True, timeout=self._ssh_cmd_timeout
                        )
                    alive = check.ok
                except Exception:
                    alive = False

        proc_status = "ALIVE" if alive else ("UNKNOWN" if pid is None else "EXITED")
        info["alive"] = alive
        info["_proc"] = proc_status

        if alive:
            ok, status = self._check_remote_health(info, label)
            info["_health"] = status
        else:
            ok, status = False, "-"
            info["_health"] = status

        return {"alive": alive, "healthy": ok, "proc_status": proc_status, "health_status": status}

    def _probe_all_nodes(self, node_infos: list, raise_on_dead: bool = False) -> list:
        """Run the probe on all nodes and return a list of probe results. If raise_on_dead=True, raise an exception when any process has exited."""
        results = []
        for info in node_infos:
            result = self._probe_node(info)
            results.append(result)

        self._print_nodes_table(
            node_infos,
            header=f"[{self._cluster_id}_{self._particle_count:03d}] All Nodes Status in Cluster {self._cluster_id}",
        )

        if raise_on_dead:
            dead_labels = [info["label"] for info in node_infos if not info.get("alive")]
            if dead_labels:
                raise RuntimeError(f"Nodes exited after startup: {', '.join(dead_labels)}")

        return results

    def _collect_remote_logs(self, log_dir: Path):
        """Collect remote run logs into the specified log_dir directory.
        The filename is suffixed per pre_kill_status to mark the exit state:
        _OK = killed actively (normal exit), _ERR = process already gone before kill (abnormal exit).
        """
        if not self._node_infos:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        for info in self._node_infos:
            label = info.get("label")
            log_file = info.get("log_file")
            node_cfg = info.get("_node_config")
            if not label or not log_file or not node_cfg:
                continue
            # Status probed before killing the process: alive=OK (killed actively), not alive=ERR (abnormal exit)
            pre_alive = info.get("_pre_kill_alive")
            suffix = "_OK" if pre_alive else "_ERR"
            try:
                executor = self._exec_for_node(node_cfg)
                local_path = str(log_dir / f"{label}{suffix}.log")
                executor.get(log_file, local_path)
                logger.info(f"[{label}] collected remote log to {local_path}")
            except Exception as e:
                logger.warning(f"[{label}] failed to collect remote log {log_file}: {e}")

    def backup(self):
        """Back up this round's rendered scripts and remote logs to bak_path.

        Done in one step: probe liveness -> pull remote logs -> back up scripts + logs.
        Log filename suffix: _OK = process alive (normal), _ERR = process exited (abnormal).

        Both scripts and logs live in the tempfile temp directory returned by
        _get_round_dir(); backup only reads them out for archiving, and deletion of
        the temp directory is handled uniformly by stop() (deleted only when bak_path
        is non-empty, otherwise kept as the sole log copy).
        """
        if not self.bak_path:
            return
        # 失败路径可能双重 backup()：run() except 内一次 + 框架成功分支一次。
        # 若 stop() 已清理 round dir（_round_tmp_dir 置 None），此处早退，
        # 避免 _get_round_dir() 重建空目录覆盖第一次归档的脚本/日志。
        if self._round_tmp_dir is None:
            return
        round_dir = self._get_round_dir()
        if not round_dir.exists():
            return

        # 1. Probe process liveness (processes have not yet been killed by stop() at this point)
        if self._node_infos:
            for info in self._node_infos:
                self._probe_node(info)
                info["_pre_kill_alive"] = info.get("alive", False)

        # 2. Pull remote logs (_collect_remote_logs decides the _OK/_ERR suffix based on _pre_kill_alive)
        log_dir = getattr(self, '_round_log_dir', None) or round_dir / "log"
        self._collect_remote_logs(log_dir)

        # 3. Back up scripts to bak_path/PdClusterSimulator/scripts/
        dest = Path(self.bak_path) / self.__class__.__name__ / "scripts"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True, mode=0o750)
        for item in round_dir.iterdir():
            if item.is_file():
                shutil.copy2(str(item), str(dest / item.name))
        logger.info(f"[backup] scripts copied to {dest}")

        # 4. Back up logs to bak_path/PdClusterSimulator/log/
        if log_dir.exists():
            log_dest = Path(self.bak_path) / self.__class__.__name__ / "log"
            if log_dest.exists():
                shutil.rmtree(log_dest)
            shutil.copytree(log_dir, log_dest)
            logger.info(f"[backup] logs copied to {log_dest}")

    def _kill_all_processes(self):
        """Kill all remote vLLM processes via the stop script (without clearing state).

        Used by restart() to stop old processes while keeping _node_infos / _executors
        intact for relaunch. stop() calls this then clears state and closes executors.
        """
        seen: set = set()
        for node in self._all_nodes():
            key = self._container_key(node)
            if key in seen:
                continue
            seen.add(key)
            try:
                self._stop_node(node, label=key)
            except Exception as e:
                logger.warning(f"[{key}] stop error: {e}")

    def stop(self, del_log: bool = True):
        logger.info(f"[OPT] stop called: del_log={del_log}")

        # ---- Replaces super().stop(del_log) ----
        # The parent's stop() would call psutil.Process(self.process.pid).children(),
        # but self.process is a _RemoteProcessPlaceholder with no local PID to kill.
        # Only the meaningful log-cleanup logic from the parent is kept here.
        self.run_log_offset = 0
        close_file_fp(self.run_log_fp)
        if del_log and self.run_log:
            remove_file(Path(self.run_log))
        # ---- End of replacement ----

        self._kill_all_processes()

        for executor in self._executors.values():
            executor.close()
        self._executors.clear()
        self._remote_pids = {}
        self._remote_pid_nodes = {}
        self._remote_logs = {}
        self._node_infos = None
        self._process_stage = ProcessState(stage=Stage.stop)
        # _cleanup_round_dir() is idempotent; unconditionally cleans this round's temp directory (scripts + logs).
        self._cleanup_round_dir()
        logger.info("[OPT] stop completed")

    def __del__(self):
        # Fallback to close SSH executors: if stop() is not reached on an exception
        # path or when the object is GC'd, this avoids leaking fabric Connections.
        # getattr guards __new__-based test objects that bypass __init__ and have no _executors.
        executors = getattr(self, "_executors", None)
        if not executors:
            return
        for ex in executors.values():
            try:
                ex.close()
            except Exception:  # nosec B110
                pass
        executors.clear()

    def health(self) -> ProcessState:
        """Check the cluster's health status. Performs a kill -0 liveness check per node,
        then an HTTP health request to confirm readiness.

        Implementation: for each node, first do a kill -0 liveness check; if alive,
        send one HTTP health request (_probe_node -> _check_remote_health,
        curl --max-time 5) to confirm the service is ready. The previous docstring
        claim of "no HTTP health polling" did not match the implementation and has
        been corrected.
        """
        if self._process_stage.stage == Stage.stop:
            logger.debug("[OPT] health -> stop")
            return ProcessState(stage=Stage.stop)

        if not self._node_infos:
            return ProcessState(stage=Stage.stop)

        dead_labels = []
        for info in self._node_infos:
            probe = self._probe_node(info)
            if not probe["alive"]:
                dead_labels.append(f"{info['label']}({probe['proc_status']})")

        if dead_labels:
            result = ProcessState(stage=Stage.error, info="; ".join(dead_labels))
            logger.info(f"[OPT] health -> {result}")
            return result

        result = ProcessState(stage=Stage.running)
        return result

    def get_last_log(self, number: int = 5, *, retry: bool = True):
        """The PD plugin has no local logs; return an empty string to satisfy the
        health check hook contract.

        retry is accepted only to align with the base class CustomProcess.get_last_log
        signature (health_check calls it with retry=False); there are no logs to retry
        here, so this parameter is ignored.
        """
        return ""
