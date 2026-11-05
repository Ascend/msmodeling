#!/usr/bin/env python3

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

"""Generate a round-local PD config from a structured candidate and fixed inventory."""

import argparse
import json
import os
import socket
import sys
from copy import deepcopy
from pathlib import Path

import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from optix.optimizer.agentic.pd import (
    configuration_fingerprint,
    device_inventory,
    load_plugin_config,
    resource_nodes,
    validate_resources,
)
from optix.optimizer.agentic.validation import validate_candidate
from optix.optimizer.pd_adapter import verify_saved_shutdown


def generate_config(experiment, context, candidate, output):
    base = Path(experiment["source"]["config_toml"])
    output = Path(output)
    if output.exists() or output.resolve() == base.resolve():
        raise ValueError("Round config must be a new file, separate from the experiment base")
    if configuration_fingerprint(base) != experiment["deployment"]["config_sha256"]:
        raise ValueError("Experiment base configuration changed")
    config = load_plugin_config(base)
    templates = resource_nodes(config)
    proposal = candidate.get("deployment")
    if not isinstance(proposal, dict) or set(proposal) != {"nodes"}:
        raise ValueError("PD candidate deployment must declare only nodes")
    allowed = {tuple(device) for device in experiment["deployment"]["device_inventory"]}
    proposed = []
    for assignment in proposal["nodes"]:
        if set(assignment) != {"source_node", "role", "gpu_ids"}:
            raise ValueError("Each assignment requires source_node, role, gpu_ids")
        index = assignment["source_node"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(templates):
            raise ValueError("source_node must index the experiment's resolved resource nodes")
        if assignment["role"] not in {"prefill", "decode"}:
            raise ValueError("Only P/D roles may be reassigned")
        node = deepcopy(templates[index])
        if not set(assignment["gpu_ids"]) <= set(node["gpu_ids"]):
            raise ValueError("Assigned devices exceed the source node's declared inventory")
        node.update(role=assignment["role"], gpu_ids=assignment["gpu_ids"])
        proposed.append(node)
    inventory = device_inventory(proposed)
    if not inventory <= allowed:
        raise ValueError("PD deployment exceeds the experiment inventory")
    proxy_nodes = [node for node in config.nodes if node.role == "proxy"]
    if proxy_nodes:
        config.proxy = proxy_nodes[0]
    config.nodes = [type(config.proxy)(**node) for node in proposed]
    config.prefill_groups = []
    config.decode_groups = []
    from optix.optimizer.agentic.pd_layouts import plan_node_ports

    plan_node_ports(config)
    validation_context = deepcopy(context)
    # The previous round binding cannot constrain a newly proposed deployment.
    validation_context["deployment"].pop("round_manifest", None)
    validation_context["deployment"].pop("execution_record", None)
    validation_context["deployment"]["plugin_config"] = config.model_dump(mode="json")
    validation = validate_candidate(validation_context, candidate)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    params = validation["params"]
    layout = validate_resources(config, params)
    from optix.optimizer.agentic.pd_state import validate_workflow_candidate, require_phase_cleanup

    require_phase_cleanup(Path(output).parent)
    validate_workflow_candidate(Path(output).parent, params, layout)
    document = tomlkit.parse(base.read_text(encoding="utf-8"))
    for section_name, section in document.items():
        if not hasattr(section, "get"):
            continue
        for field in section.get("target_field", []):
            name = field.get("name")
            if name in params:
                field["value"] = params[name]
    from optix.optimizer.agentic.parameters import resolve_execution_fields
    from optix.config.base_config import CONCURRENCYS, REQUESTRATES

    service_params = {name: value for name, value in params.items() if name not in {*CONCURRENCYS, *REQUESTRATES}}
    config.target_field = resolve_execution_fields(service_params, config.target_field, context)
    document["vllm_pd"] = tomlkit.item(config.model_dump(mode="json", exclude_none=True))
    document.setdefault("agent_optimizer", {})["run_dir"] = str(Path(output).parent.resolve())
    document["optimizer_strategy"] = "agent"
    document["use_request_rate_calibration"] = False
    # Materialization is followed by context recollection; never mutate the base.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(tomlkit.dumps(document))
    return {
        "config_toml": str(output.resolve()),
        "config_sha256": configuration_fingerprint(output),
        "layout": layout,
        "deployment": proposal,
        "params": params,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--purpose",
        choices=["bootstrap", "search", "verification_baseline", "verification_winner"],
        default="search",
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    try:
        dispatch = run_dir / ".pd_dispatch" / "pd_controller.json"
        if dispatch.exists() and json.loads(dispatch.read_text(encoding="utf-8")) != {
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }:
            raise ValueError("PD round dispatcher owns this run; use run_pd_round.py")
        controller = run_dir / "pd_controller.json"
        if controller.exists():
            owner = json.loads(controller.read_text(encoding="utf-8"))
            if owner.get("host") not in {None, socket.gethostname()}:
                raise ValueError("Previous PD controller belongs to another host; verify cleanup there")
            pid = owner["pid"]
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("Previous OptiX PD controller is still running")
        # A failed controller may never have written latest.json. Inspect every
        # attempted deployment so an unconfirmed launch cannot be skipped.
        for archive in sorted((run_dir / "pd_deployments").glob("*/deployment.json")):
            record = json.loads(archive.read_text(encoding="utf-8"))
            if not record.get("cleanup_confirmed"):
                verify_saved_shutdown(record)
            if "finished_at" not in record:
                from optix.optimizer.agentic.pd_state import finish_interrupted_attempt

                finish_interrupted_attempt(record)
            if record.get("cleanup_confirmed"):
                archive.write_text(json.dumps(record, indent=2), encoding="utf-8")
        latest = run_dir / "pd_deployments/latest.json"
        if latest.exists():
            verify_saved_shutdown(json.loads(latest.read_text(encoding="utf-8")))
        proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
        if proposal.get("round") != args.round or len(proposal.get("candidates", [])) != 1 or args.round < 1:
            raise ValueError("PD requires exactly one deployment candidate for the requested positive round")
        if args.attempt < 1:
            raise ValueError("attempt must be positive")
        config_name = f"config.round-{args.round}" + (f".attempt-{args.attempt}" if args.attempt > 1 else "") + ".toml"
        manifest_name = (
            f"deployment.round-{args.round}" + (f".attempt-{args.attempt}" if args.attempt > 1 else "") + ".json"
        )
        manifest_path = run_dir / manifest_name
        if manifest_path.exists():
            raise ValueError("Deployment manifest already exists; select a new attempt")
        if args.purpose in {"bootstrap", "search"}:
            result = generate_config(
                json.loads((run_dir / "pd_experiment.json").read_text(encoding="utf-8")),
                json.loads((run_dir / "context.json").read_text(encoding="utf-8")),
                proposal["candidates"][0],
                run_dir / config_name,
            )
        else:
            from optix.optimizer.agentic.pd_evidence import generate_replay

            result = generate_replay(
                run_dir,
                args.purpose.removeprefix("verification_"),
                proposal["candidates"][0],
                run_dir / config_name,
            )
        result.update(
            round=args.round,
            attempt=args.attempt,
            candidate_id=proposal["candidates"][0].get("candidate_id"),
            purpose=args.purpose,
        )
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"PD config generation failed: {error}\n")


if __name__ == "__main__":
    main()
