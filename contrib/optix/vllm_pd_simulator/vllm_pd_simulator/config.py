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

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional
from pathlib import Path
from loguru import logger

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None
from pydantic import BaseModel, Field, ConfigDict


# Import OptimizerConfigField from the core module to ensure compatibility
from optix.config.config import OptimizerConfigField


class SshConnectable(BaseModel):
    """SSH connection fields shared by HostConfig / ClusterNodeConfig."""

    model_config = ConfigDict(extra='allow')
    ssh_ip: str = Field(default="localhost")
    ssh_port: int = Field(default=22)
    ssh_user: str = Field(default="root")
    password: Optional[str] = Field(default=None, description="SSH 密码（支持 base64 编码）")
    docker_container_id: Optional[str] = Field(default=None, description="Docker 容器 ID（走 docker exec）")
    docker_use_sudo: bool = Field(default=False, description="docker 命令是否需要 sudo")


class HostConfig(SshConnectable):
    """Host configuration (deployment info unrelated to business logic)."""

    id: str = Field(default="", description="主机唯一标识")
    service_ip: str = Field(default="", description="服务绑定 IP（空则使用 ssh_ip）")
    network_interface: str = Field(default="lo")


def _resolve_host_ref(node_data: dict, hosts_map: Dict[str, HostConfig]):
    """Resolve the hosts reference in node/proxy into host fields, filling node_data.

    New format: node/proxy contains hosts = "id"; host fields are taken from [[cluster.hosts]].
    Old format: no hosts field; host info is inline and left unchanged.
    """
    host_id = node_data.pop("hosts", None) or node_data.pop("vllm_hosts", None)
    if host_id is None:
        return
    host = hosts_map.get(host_id)
    if host is None:
        logger.warning(f"Host id '{host_id}' not found in [[cluster.hosts]]")
        return
    node_data["ssh_ip"] = host.ssh_ip
    node_data["ssh_port"] = host.ssh_port
    node_data["ssh_user"] = host.ssh_user
    node_data["password"] = host.password
    node_data["docker_container_id"] = host.docker_container_id
    node_data["docker_use_sudo"] = host.docker_use_sudo
    # gpu_ids is determined by the node config itself, not inherited from host
    node_data.setdefault("gpu_ids", [])
    node_data["network_interface"] = host.network_interface
    node_data["bind_ip"] = host.service_ip
    node_data["hccl_if_ip"] = host.service_ip or host.ssh_ip


def _normalize_node_fields(node_data: dict):
    """Map legacy node field names to the new ssh_ip/ssh_user/service_port fields.

    Covers both PD legacy (host/port) and cluster legacy (user_name/bind_port) naming.
    Values are taken from legacy fields only when the target field is missing, to avoid
    overriding explicit new-format configuration.
    """
    if not isinstance(node_data, dict):
        return
    # PD legacy: host -> ssh_ip, port -> service_port
    if "ssh_ip" not in node_data and "host" in node_data:
        node_data["ssh_ip"] = node_data["host"]
    if "service_port" not in node_data and "port" in node_data:
        node_data["service_port"] = node_data["port"]
    # cluster legacy: user_name -> ssh_user, bind_port -> service_port
    if "ssh_user" not in node_data and "user_name" in node_data:
        logger.warning("[DEPRECATION] Node 'user_name' is deprecated; use 'ssh_user' instead.")
        node_data["ssh_user"] = node_data.pop("user_name")
    if "service_port" not in node_data and "bind_port" in node_data:
        logger.warning("[DEPRECATION] Node 'bind_port' is deprecated; use 'service_port' instead.")
        node_data["service_port"] = node_data.pop("bind_port")


class ClusterNodeConfig(SshConnectable):
    """Unified node config for cluster/PD/proxy roles."""

    bind_ip: str = Field(default="", description="vLLM 服务绑定 IP（空则使用 ssh_ip）")
    service_port: int = Field(default=18080, description="vLLM 服务端口基址，实际端口 = 基址 + dp_rank")
    gpu_ids: List[int] = Field(default_factory=list)
    kv_port: int = Field(default=30100, description="KV 传输端口基址，实际端口 = 基址 + dp_rank * tp_size")
    rpc_port: int = Field(default=29500, description="RPC 端口基址，实际端口 = 基址 + dp_rank")
    engine_id: str = Field(
        default="", description="[自动生成] Mooncake engine_id，格式为 P-{dp_rank} / D-{dp_rank}，无需手动配置"
    )
    dp_rpc_port: int = Field(default=12345)
    timeout_seconds: int = Field(default=7200, description="启动超时（秒）")
    check_interval_seconds: int = Field(default=10, description="检查间隔（秒）")
    network_interface: str = Field(default="lo")
    hccl_if_ip: str = Field(default="127.0.0.1", description="HCCL 通信 IP，默认 127.0.0.1")
    ascend_base_port: Optional[int] = Field(
        default=None, description="Ascend Direct Transport 基础端口（节点级，不配置则回落到实例级再回落到 20000）"
    )
    role: str = Field(default="", description="节点角色: prefill/decode/proxy")
    env: Dict[str, str] = Field(default_factory=dict, description="节点级环境变量，切分时继承到对应 group")


class PDGroup(BaseModel):
    model_config = ConfigDict(extra='allow')
    """PD group configuration, including DP coordination address/port and node list."""
    dp_address: str = Field(default='127.0.0.1', description='DP 协调地址')
    dp_rpc_port: int = Field(default=12345, description="DP Coordinator ZMQ 端口")
    env: Dict[str, str] = Field(default_factory=dict, description="组环境变量")
    nodes: List[ClusterNodeConfig] = Field(default_factory=list)


class VLLMPDDisaggConfig(BaseModel):
    ASCEND_DEFAULT_BASE_PORT: ClassVar[int] = (
        20000  # 类常量（ClassVar 语义，pydantic BaseModel 类常量不进 model_fields）
    )
    model_config = ConfigDict(extra='allow')
    """vLLM PD disaggregated configuration."""

    model_path: str = Field(default="/path/to/model")
    served_model_name: str = Field(default="default", description="vLLM --served-model-name")
    vllm_others: str = Field(default="", description="vLLM 额外启动参数")
    prefill_instances: Optional[int] = Field(
        default=None, description="P 侧实例（组）数量，None 时自动按 total_gpus // ep_size 推导"
    )
    decode_instances: Optional[int] = Field(
        default=None, description="D 侧实例（组）数量，None 时自动按 total_gpus // ep_size 推导"
    )
    ssh_command_timeout: int = Field(default=30, description="SSH 命令超时（秒）")
    remote_tmp_dir: str = Field(
        default="/tmp",  # nosec B108
        description="ssh 模式远端工作临时目录前缀；_remote_dir={remote_tmp_dir}/vllm_{pd|cluster}_{cluster_id}",
    )
    stop_grace_timeout: int = Field(
        default=30,
        description="stop 脚本 SIGTERM 后等优雅退出+端口释放的宽限（秒）；可由环境变量 STOP_GRACE_TIMEOUT 覆盖",
    )
    stop_kill_timeout: int = Field(
        default=10, description="stop 脚本 SIGKILL 后等端口释放的超时（秒）；可由环境变量 STOP_KILL_TIMEOUT 覆盖"
    )
    prefill_groups: List[PDGroup] = Field(default_factory=list)
    decode_groups: List[PDGroup] = Field(default_factory=list)
    proxy: ClusterNodeConfig = Field(default_factory=lambda: ClusterNodeConfig(role="proxy", service_port=8000))
    target_field: List[OptimizerConfigField] = Field(default_factory=list)
    nodes: List[ClusterNodeConfig] = Field(
        default_factory=list, description="扁平节点池,用 role 标记 prefill/decode/proxy"
    )
    env_prefill: Dict[str, str] = Field(default_factory=dict, description="P 侧环境变量")
    env_decode: Dict[str, str] = Field(default_factory=dict, description="D 侧环境变量")
    ascend_base_port: Optional[int] = Field(
        default=None,
        description="实例级 Ascend Direct Transport 基础端口（节点未配置时回落到此值）",
    )


def _normalize_proxy_fields(proxy_data: dict):
    """Map legacy proxy field names to ClusterNodeConfig field names.

    Legacy config.toml proxy uses host for the host IP and proxy_port for the service port;
    the new ClusterNodeConfig uses ssh_ip / service_port.
    """
    if not isinstance(proxy_data, dict):
        return
    if "ssh_ip" not in proxy_data and "host" in proxy_data:
        proxy_data["ssh_ip"] = proxy_data["host"]
    if "service_port" not in proxy_data and "proxy_port" in proxy_data:
        proxy_data["service_port"] = proxy_data["proxy_port"]


def _find_config_file() -> Path:
    """Resolve the config file: prefer -c (last toml_file), else plugin config.toml."""
    from optix.config.config import get_settings

    toml_files = get_settings().model_config.get('toml_file', [])
    if toml_files:
        # The last one is specified via -c (highest priority)
        candidate = Path(toml_files[-1])
        if candidate.exists() and candidate != Path(__file__).parent / "config.toml":
            return candidate
    return Path(__file__).parent / "config.toml"


def load_pd_config() -> Optional[VLLMPDDisaggConfig]:
    """Load the plugin config.toml, supporting both legacy format (inline host info) and new format ([[hosts]] reference)."""
    if tomllib is None:
        return None

    config_file = _find_config_file()

    if not config_file.exists():
        return None

    try:
        with open(config_file, "rb") as f:
            toml_data = tomllib.load(f)

        plugin_data = toml_data.get("vllm_pd", {})
        if not plugin_data:
            return None

        # Check whether a [[hosts]] section exists (new format)
        # The [[hosts]] section may be referenced under both the flat nodes style
        # and the legacy prefill_groups/decode_groups/proxy style; resolve hosts
        # references uniformly.
        hosts_list = (
            (toml_data.get("cluster") or {}).get("hosts", [])
            or toml_data.get("vllm_hosts", [])
            or toml_data.get("hosts", [])
        )
        if hosts_list:
            hosts_map: Dict[str, HostConfig] = {}
            for h in hosts_list:
                hc = HostConfig(**h)
                if hc.id:
                    hosts_map[hc.id] = hc

            for node_data in plugin_data.get("nodes", []):
                _resolve_host_ref(node_data, hosts_map)

            for group in plugin_data.get("prefill_groups", []):
                for node_data in group.get("nodes", []):
                    _resolve_host_ref(node_data, hosts_map)

            for group in plugin_data.get("decode_groups", []):
                for node_data in group.get("nodes", []):
                    _resolve_host_ref(node_data, hosts_map)

            proxy_data = plugin_data.get("proxy", {})
            if proxy_data:
                _resolve_host_ref(proxy_data, hosts_map)

        # Map environment variables to env_prefill/env_decode, supporting two formats:
        # 1. [[vllm_pd.env]] array format (each entry may carry a role)
        # 2. [vllm_pd.env] table format (common variables written directly in the table,
        #    with prefill/decode as sub-tables)
        env_data = plugin_data.pop("env", [])
        if isinstance(env_data, list):
            # Array format
            for entry in env_data:
                role = entry.pop("role", "")
                if role == "prefill":
                    plugin_data.setdefault("env_prefill", {}).update(entry)
                elif role == "decode":
                    plugin_data.setdefault("env_decode", {}).update(entry)
                else:
                    # Common env var without a role, merged into both P and D
                    plugin_data.setdefault("env_prefill", {}).update(entry)
                    plugin_data.setdefault("env_decode", {}).update(entry)
        elif isinstance(env_data, dict):
            # Table format: common variables directly in the env table, with prefill/decode as sub-tables
            prefill_env = env_data.pop("prefill", None)
            decode_env = env_data.pop("decode", None)
            # The remaining are common variables, merged into both P and D
            plugin_data.setdefault("env_prefill", {}).update(env_data)
            plugin_data.setdefault("env_decode", {}).update(env_data)
            if prefill_env:
                prefill_env.pop("role", None)
                plugin_data.setdefault("env_prefill", {}).update(prefill_env)
            if decode_env:
                decode_env.pop("role", None)
                plugin_data.setdefault("env_decode", {}).update(decode_env)

        # Legacy config.toml compatibility: nodes in prefill_groups/decode_groups and
        # proxy may use host/port (node) or host/proxy_port (proxy) field names; map
        # them uniformly to ClusterNodeConfig.ssh_ip/service_port.
        for group in plugin_data.get("prefill_groups", []):
            for node_data in group.get("nodes", []):
                _normalize_node_fields(node_data)
        for group in plugin_data.get("decode_groups", []):
            for node_data in group.get("nodes", []):
                _normalize_node_fields(node_data)
        proxy_data = plugin_data.get("proxy", {})
        if proxy_data:
            _normalize_proxy_fields(proxy_data)

        return VLLMPDDisaggConfig(**plugin_data)

    except Exception as e:
        logger.warning(
            f"Failed to load plugin config from {config_file}: {e}. "
            f"Falling back to default VLLMPDDisaggConfig (all hosts = localhost). "
            f"Check that the config.toml TOML syntax is valid and all required fields are present."
        )
        return None
