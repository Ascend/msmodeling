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


"""PD role and resource contracts at the OptiX boundary, without remote I/O."""

from contextlib import contextmanager

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

ROLES = ("prefill", "decode")
TOPOLOGY_FIELDS = {"TENSOR_PARALLEL_SIZE", "DATA_PARALLEL_SIZE"}
FIXED_PARALLEL_FIELDS = {
    "PP",
    "PCP",
    "PIPELINE_PARALLEL_SIZE",
    "PREFILL_CONTEXT_PARALLEL_SIZE",
}
OWNED_ENV = {
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_BASE_PORT",
    "HCCL_IF_IP",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "HCCL_SOCKET_IFNAME",
    "ATB_LLM_HCCL_ENABLE",
    "ATB_LLM_LCOC_ENABLE",
}


def split_role(name):
    """Retain the plugin's case-sensitive suffix and return the real parameter name."""
    for role in ROLES:
        suffix = "_" + role
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return name, None


def merge_role_params(params, role, *, excluded=frozenset()):
    """Merge common and role-specific values, letting the role take precedence."""
    selected = {}
    for owner in (None, role):
        for name, value in params.items():
            base, assigned = split_role(name)
            if assigned == owner and base not in excluded:
                selected[base] = value
    return selected


def role_service_fields(fields, role):
    """Project common and role-specific service fields; explicit role values win."""
    from types import SimpleNamespace

    eligible = {
        field.name: field.value
        for field in fields
        if field.config_position == "run"
        and split_role(field.name)[0] not in TOPOLOGY_FIELDS
        and split_role(field.name)[0].upper() not in FIXED_PARALLEL_FIELDS
    }
    return tuple(
        SimpleNamespace(name=name, value=value, config_position="run")
        for name, value in merge_role_params(eligible, role).items()
    )


PD_OWNED_FLAGS = {
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
    "--data-parallel-size",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--data-parallel-rank",
    "--kv-transfer-config",
}


def render_pd_role_flags(fields, role, others):
    """Apply the same PD-owned option checks in preflight and at launch."""
    import shlex
    from ...serve_plan import render_serve_flags

    if "<%" in others:
        raise ValueError("PD command others must use literal options; node-owned templates belong to the plugin")
    if any(token.split("=", 1)[0] in PD_OWNED_FLAGS for token in shlex.split(others)):
        raise ValueError("PD command others conflicts with plugin-owned deployment options")
    flags = render_serve_flags(role_service_fields(fields, role), others)
    if any(token.split("=", 1)[0] in PD_OWNED_FLAGS for token in flags):
        raise ValueError("PD service parameter conflicts with plugin-owned deployment options")
    for index, token in enumerate(flags):
        option, separator, value = token.partition("=")
        if option in {"--pipeline-parallel-size", "--prefill-context-parallel-size"}:
            size = value if separator else flags[index + 1] if index + 1 < len(flags) else None
            if str(size) != "1":
                raise ValueError("PD effective PP/PCP must remain 1")
    return flags


def positive_integer(value, description):
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a positive integer")
    try:
        number = int(value)
        if float(value) != number or number <= 0:
            raise ValueError
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{description} must be a positive integer") from error
    return number


def configuration_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_snapshot(context):
    deployment = context.get("deployment", {})
    if deployment.get("type") != "pd":
        return
    path = context.get("source", {}).get("config_toml")
    if not path or deployment.get("config_sha256") != configuration_fingerprint(path):
        raise ValueError("PD configuration changed; collect and validate context again")
    from .pd_execution import validate_execution

    validate_execution(context)


def resource_nodes(config):
    """Consume resolved plugin model data, including its legacy grouped format."""
    data = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    if data.get("nodes"):
        return deepcopy(data["nodes"])
    nodes = []
    for role in ROLES:
        for group in data.get(role + "_groups", []):
            nodes.extend({**deepcopy(node), "role": role} for node in group.get("nodes", []))
    return nodes


def device_inventory(nodes):
    """Physical device identity is host plus device ID, regardless of container."""
    seen = set()
    for node in nodes:
        if node.get("role") not in ROLES:
            continue
        host = node.get("ssh_ip")
        if not host:
            raise ValueError("PD nodes must have a resolved ssh_ip")
        if not node.get("gpu_ids"):
            raise ValueError(f"PD node {host} has no assigned devices")
        for device in node["gpu_ids"]:
            if isinstance(device, bool) or not isinstance(device, int) or device < 0:
                raise ValueError("PD device IDs must be nonnegative integers")
            identity = (host, device)
            if identity in seen:
                raise ValueError(f"PD device is assigned more than once: {identity}")
            seen.add(identity)
    return seen


def validate_resources(config, params):
    """Validate exactly the grouping and per-host TP slots the original plugin uses.

    In flat mode the plugin consumes the entire role pool in TP*DP slices.
    A TP slot cannot straddle nodes: the renderer would silently skip fragments.
    Legacy groups must each contain a complete instance, rather than an aggregate
    pool that happens to divide evenly.
    """
    data = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    nodes = resource_nodes(data)
    inventory = device_inventory(nodes)
    layouts = {}
    for role in ROLES:
        tp = positive_integer(params.get("TENSOR_PARALLEL_SIZE_" + role, 1), role + " TP")
        dp = positive_integer(params.get("DATA_PARALLEL_SIZE_" + role, 1), role + " DP")
        pool = [node for node in nodes if node.get("role") == role]
        count = sum(len(node["gpu_ids"]) for node in pool)
        if not count or count % (tp * dp):
            raise ValueError(f"{role} devices must form complete TP*DP instances")
        groups = data.get(role + "_groups", []) if not data.get("nodes") else []
        if groups and any(sum(len(n.get("gpu_ids", [])) for n in g["nodes"]) != tp * dp for g in groups):
            raise ValueError(f"Each legacy {role} group must contain exactly TP*DP devices")
        if any(len(node["gpu_ids"]) % tp for node in pool):
            raise ValueError(f"{role} TP cannot span partial node device slices")
        instances = count // (tp * dp)
        requested = data.get(role + "_instances")
        if requested is not None and positive_integer(requested, role + " instances") != instances:
            raise ValueError(f"{role} instance count conflicts with the device pool and TP*DP")
        layouts[role] = {
            "devices": count,
            "tp": tp,
            "dp": dp,
            "instances": instances,
            "assignments": [{"host": node["ssh_ip"], "gpu_ids": list(node["gpu_ids"])} for node in pool],
        }
    return {"roles": layouts, "devices": [list(item) for item in sorted(inventory)]}


def validate_fields(fields):
    """Reject values the plugin skips or cannot represent; reject unsupported PP/PCP."""
    for field in fields:
        name, _ = split_role(field.name)
        if field.config_position not in {"run", "env"}:
            raise ValueError(f"PD parameter {field.name} has an unsupported config_position")
        if field.value is None:
            raise ValueError(f"PD parameter {field.name} cannot express environment deletion")
        if field.config_position == "env":
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid PD environment variable name: {name}")
            if name.upper() in OWNED_ENV:
                raise ValueError(f"PD environment {name} is owned by the deployment configuration")
        if name.upper() in FIXED_PARALLEL_FIELDS:
            if positive_integer(field.value, name) != 1:
                raise ValueError("PD PP/PCP must remain 1 until plugin resource mapping supports them")
        if name.upper() in {"TP", "DP"}:
            raise ValueError("PD topology uses TENSOR_PARALLEL_SIZE / DATA_PARALLEL_SIZE with role suffixes")
        if name in TOPOLOGY_FIELDS and field.name == name:
            raise ValueError(f"Use role-specific {name}_prefill / {name}_decode")


def layout_fingerprint(layout):
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def validate_simulator(context, simulator):
    """Prove the plugin loaded the collected config before the baseline can start."""
    validate_snapshot(context)
    deployment = context.get("deployment", {})
    if deployment.get("type") != "pd":
        raise ValueError("PD agent requires a PD configuration context")
    from .pd_execution import validate_loaded_config

    validate_loaded_config(context, simulator)
    expected = deployment.get("plugin_config")
    actual = simulator.config.model_dump(mode="json")
    if plugin_signature(expected) != plugin_signature(actual):
        raise ValueError("Loaded PD plugin configuration differs from context; recollect with the same -c file")


def load_plugin_config(path):
    """Use the original loader with an explicit path, restoring process settings."""
    from types import SimpleNamespace
    from ...config import config as settings_module
    from vllm_pd_simulator.config import load_pd_config

    previous = settings_module.settings
    try:
        settings_module.settings = SimpleNamespace(model_config={"toml_file": [str(Path(path).resolve())]})
        config = load_pd_config(strict=True)
    finally:
        settings_module.settings = previous
    if config is None:
        raise ValueError("PD plugin configuration could not be loaded; defaults are not a valid deployment")
    device_inventory(resource_nodes(config))
    return config


def plugin_signature(config):
    data = config.model_dump(mode="json") if hasattr(config, "model_dump") else deepcopy(config)
    data = deepcopy(data)
    data.pop("target_field", None)
    if data.get("nodes"):
        data.pop("prefill_groups", None)
        data.pop("decode_groups", None)
        if any(node.get("role") == "proxy" for node in data["nodes"]):
            data.pop("proxy", None)
    return data


@contextmanager
def controller_session(run_dir):
    """Prevent configuration generation while this local PD controller is active."""
    import os
    import socket

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "pd_controller.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("host") not in {None, socket.gethostname()}:
            raise ValueError("PD controller lock belongs to another host; verify cleanup on that host")
        try:
            os.kill(positive_integer(previous["pid"], "controller PID"), 0)
        except ProcessLookupError:
            marker.unlink()
        else:
            raise ValueError("A PD controller is already running in this run directory")
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)
