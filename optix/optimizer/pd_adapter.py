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


"""Translate validated PD fields into the unchanged plugin's role protocol."""

import shlex
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from ..serve_plan import validate_service_positions
from ..config.base_config import CONCURRENCYS, REQUESTRATES
from .agentic.pd import (
    render_pd_role_flags,
    TOPOLOGY_FIELDS,
    split_role,
    validate_fields,
    validate_resources,
)


def prepare_pd_fields(simulator, fields):
    """Render each role once, including JSON containers, booleans and static options."""
    fields = tuple(field for field in fields if field.name not in {*CONCURRENCYS, *REQUESTRATES})
    validate_service_positions(fields)
    validate_fields(fields)
    params = {field.name: field.value for field in fields}
    layout = validate_resources(simulator.config, params)
    if getattr(simulator, "_is_moe", None) is False and any(role["dp"] != 1 for role in layout["roles"].values()):
        raise ValueError("Original PD plugin renders dense-model DP as 1; use independent instances instead")
    others = simulator.vllm_others or ""
    transported = []
    for field in fields:
        base, _ = split_role(field.name)
        if base in TOPOLOGY_FIELDS or field.config_position == "env":
            transported.append(deepcopy(field))
    for role in ("prefill", "decode"):
        flags = render_pd_role_flags(fields, role, others)
        transported.append(
            SimpleNamespace(
                name="OPTIX_SERVE_ARGS_" + role,
                value=shlex.join(flags),
                config_position="run",
            )
        )
    return tuple(transported), layout


def _snapshot(simulator, layout):
    directory = getattr(simulator, "_round_tmp_dir", None)
    if not directory or not Path(directory).is_dir():
        backup = getattr(simulator, "bak_path", None)
        directory = Path(backup) / type(simulator).__name__ / "scripts" if backup else None
    files = {}
    if directory and Path(directory).is_dir():
        files = {
            path.name: path.read_text(encoding="utf-8") for path in sorted(Path(directory).iterdir()) if path.is_file()
        }
    nodes = []
    for info in getattr(simulator, "_node_infos", None) or []:
        node = info["_node_config"]
        nodes.append(
            {
                "label": info["label"],
                "script": info["label"] + ".sh",
                "ssh_ip": node.ssh_ip,
                "ssh_port": node.ssh_port,
                "ssh_user": node.ssh_user,
                "docker_container_id": node.docker_container_id,
                "pid": info.get("pid"),
                "node_config": node.model_dump(mode="json"),
                "port": info["port"],
                "role": "proxy"
                if info["label"].startswith("proxy_")
                else ("prefill" if info["label"].startswith("P-") else "decode"),
            }
        )
    return {
        "files": files,
        "nodes": nodes,
        "layout": layout,
        "remote_dir": simulator._remote_dir,
        "cleanup_confirmed": False,
    }


def run_pd_trial(simulator, fields, archive_dir, context, purpose="search", identity=None):
    """Use normal plugin run/backup lifecycle; retain scripts before scheduler cleanup."""
    from .agentic.pd import validate_snapshot

    simulator._optix_pd_launch_plan = {}
    if identity and Path(archive_dir).name != identity["archive_id"]:
        raise ValueError("PD archive directory does not match the execution identity")
    validate_snapshot(context)
    from .agentic.pd_execution import validate_loaded_config, validate_execution, parameter_plan, trial_identity

    validate_loaded_config(context, simulator)
    config_text = Path(context["source"]["config_toml"]).read_bytes().decode("utf-8")
    import hashlib

    if hashlib.sha256(config_text.encode("utf-8")).hexdigest() != context["deployment"]["config_sha256"]:
        raise ValueError("PD configuration changed while preparing the launch snapshot")
    transported, layout = prepare_pd_fields(simulator, fields)
    execution = validate_execution(context)
    if execution is not None:
        if identity is None:
            raise ValueError("Bound PD launch requires an execution identity")
        expected_identity = trial_identity(
            context,
            purpose,
            identity["case_id"],
            {key: identity[key] for key in ("round", "attempt", "candidate_id")},
            launch_attempt=identity["launch_attempt"],
        )
        if identity != expected_identity:
            raise ValueError("PD launch identity differs from preflight")
    if execution is not None and parameter_plan(transported) != execution["parameter_plan"]:
        raise ValueError("PD effective launch parameters differ from the preflight candidate")
    from .agentic.pd_state import validate_workflow_candidate, require_phase_cleanup

    require_phase_cleanup(Path(archive_dir).parent.parent)
    validate_workflow_candidate(
        Path(archive_dir).parent.parent,
        {f.name: f.value for f in fields},
        layout,
        bootstrap=purpose == "bootstrap",
    )
    simulator.update_config(transported)
    planned_nodes = validate_runtime_placement(simulator)
    archive_dir = Path(archive_dir)
    from .agentic.pd_state import (
        check_search_budget,
        check_verification_budget,
        check_workflow_budget,
    )

    check_workflow_budget(archive_dir.parent.parent, purpose)
    if purpose == "search":
        check_search_budget(archive_dir.parent.parent, context["pd_budget"])
    elif purpose in {"verification_baseline", "verification_winner"}:
        check_verification_budget(archive_dir.parent.parent)
    elif purpose not in {"baseline", "bootstrap"}:
        raise ValueError("Unknown PD trial purpose")
    archive_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        **(identity or {}),
        "execution_record": execution,
        "purpose": purpose,
        "started_at": time.time(),
        "outcome": "started",
        "layout": layout,
        "nodes": planned_nodes,
        "config_toml": config_text,
        "config_sha256": context["deployment"]["config_sha256"],
        "cleanup_confirmed": False,
    }
    (archive_dir / "deployment.json").write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    previous_backup = simulator.bak_path
    simulator.bak_path = archive_dir
    previous = simulator.vllm_others
    simulator.vllm_others = ""
    simulator._node_infos = None
    try:
        from .agentic.pd_state import workflow_time_limit

        with workflow_time_limit(archive_dir.parent.parent, purpose):
            simulator.run(transported)
    finally:
        try:
            snapshot = _snapshot(simulator, layout)
            snapshot["process_ids_recorded"] = bool(snapshot["nodes"])
            if not snapshot["nodes"]:
                snapshot["nodes"] = planned_nodes
            simulator._optix_pd_launch_plan = {**attempt, **snapshot}
            simulator._optix_pd_launch_plan["config_toml"] = config_text
            simulator._optix_pd_launch_plan["config_sha256"] = context["deployment"]["config_sha256"]
            (archive_dir / "deployment.json").write_text(
                json.dumps(simulator._optix_pd_launch_plan, indent=2), encoding="utf-8"
            )
            simulator._optix_pd_archive = archive_dir
        finally:
            simulator.vllm_others = previous
            simulator.bak_path = previous_backup
    return simulator._optix_pd_launch_plan


def validate_runtime_placement(simulator):
    """Check plugin-derived process placement and port ownership before remote I/O."""
    occupied = {}
    nodes = []

    def reserve(host, port, description):
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"PD {description} port is out of range")
        key = (host, int(port))
        if key in occupied:
            raise ValueError(f"PD port collision: {key}, {occupied[key]} / {description}")
        occupied[key] = description

    # Reserve whole transport ranges, not only the first port of each process.
    for role in ("prefill", "decode"):
        for group_index, group in enumerate(getattr(simulator.config, role + "_groups")):
            if group.nodes:
                reserve(
                    group.nodes[0].ssh_ip,
                    group.dp_rpc_port,
                    f"{role}-{group_index}-dp-rpc",
                )
            for node in group.nodes:
                base = node.ascend_base_port
                if base is None:
                    base = simulator.config.ASCEND_DEFAULT_BASE_PORT
                for rank in range(len(node.gpu_ids)):
                    reserve(node.ssh_ip, base + rank, f"{role}-{group_index}-ascend-{rank}")

    for role, prefix in (("prefill", "P"), ("decode", "D")):
        tp = getattr(simulator, "_" + role + "_tp_size")
        dp = getattr(simulator, "_" + role + "_dp_size")
        for group_index, group in enumerate(getattr(simulator.config, role + "_groups")):
            ranks = 0
            for node in group.nodes:
                slots = min(len(node.gpu_ids) // tp, dp - ranks)
                for local_rank in range(slots):
                    resources = simulator._calc_process_resources(prefix, node, local_rank)
                    if len(resources["gpu_ids"]) != tp:
                        raise ValueError("PD plugin produced an incomplete TP device assignment")
                    nodes.append(
                        {
                            "label": f"{role}-{group_index}-{ranks + local_rank}",
                            "node_config": node.model_dump(mode="json"),
                            "port": resources["port"],
                            "role": role,
                            "pid": None,
                            "devices": resources["gpu_ids"],
                        }
                    )
                    for name in ("port", "kv_port", "rpc_port"):
                        reserve(
                            node.ssh_ip,
                            resources[name],
                            f"{role}-{group_index}-{ranks + local_rank}-{name}",
                        )
                ranks += slots
            if ranks != dp:
                raise ValueError("PD plugin did not place all DP ranks")
    proxy = simulator.config.proxy
    reserve(proxy.ssh_ip, proxy.service_port, "proxy")
    nodes.append(
        {
            "label": "proxy",
            "node_config": proxy.model_dump(mode="json"),
            "port": proxy.service_port,
            "role": "proxy",
            "pid": None,
        }
    )
    return nodes


def verify_shutdown(plan, executor_factory):
    """Require observed absence of controller PIDs and listening service endpoints."""
    plan["cleanup_confirmed"] = False
    if not plan.get("nodes"):
        raise ValueError("PD shutdown cannot be confirmed without the started-node record")
    for node in plan["nodes"]:
        executor = executor_factory(node["node_config"])
        try:
            pid = node.get("pid")
            if pid is not None:
                if not isinstance(pid, int) or pid <= 0:
                    raise ValueError("Invalid recorded PD process ID")
                result = executor.run(f"kill -0 {pid}", container_exec=False, hide=True, warn=True)
                if result.ok:
                    raise ValueError(f"PD process is still alive: {node['label']}")
                if "permitted" in str(getattr(result, "stderr", "")).lower():
                    raise ValueError(f"PD process absence cannot be verified: {node['label']}")
            port = int(node["port"])
            if not 1 <= port <= 65535:
                raise ValueError("Invalid recorded PD endpoint")
            result = executor.run(
                f"command -v ss >/dev/null && ss -H -ltn 'sport = :{port}'",
                container_exec=False,
                hide=True,
                warn=True,
            )
            if not result.ok or result.stdout.strip():
                raise ValueError(f"PD endpoint shutdown is unconfirmed: {node['label']}")
        finally:
            executor.close()
    plan["cleanup_confirmed"] = True


def verify_saved_shutdown(plan):
    from vllm_pd_simulator.tools.ssh_remote_tools import SshRemote

    def create(node):
        return SshRemote(
            host=node["ssh_ip"],
            ssh_port=node["ssh_port"],
            ssh_user=node["ssh_user"],
            docker_container_id=node.get("docker_container_id"),
            docker_use_sudo=node.get("docker_use_sudo", False),
        )

    verify_shutdown(plan, create)
