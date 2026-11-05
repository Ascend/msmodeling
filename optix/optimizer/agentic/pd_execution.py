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


"""Bind PD preflight, launch and evidence to one execution record."""

import hashlib
import json
import re
import shlex
from pathlib import Path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_execution(context):
    """Reject changes to any prepared input, including the complete context."""
    record_path = context.get("deployment", {}).get("execution_record")
    if not record_path:
        return None  # Legacy entry points still validate their deployment snapshot.
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    for name in ("config", "context", "candidate"):
        item = record[name]
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"PD execution {name} changed after preflight; prepare a new attempt")
    if Path(context["source"]["config_toml"]).resolve() != Path(record["config"]["path"]).resolve():
        raise ValueError("PD execution configuration path does not match preflight")
    archived = json.loads(Path(record["context"]["path"]).read_text(encoding="utf-8"))
    for key in (
        "source",
        "deployment",
        "search_space",
        "benchmark_spec",
        "model",
        "model_info",
        "benchmark",
        "constraints",
    ):
        if context.get(key) != archived.get(key):
            raise ValueError(f"PD execution context.{key} differs from preflight")
    manifest = context["deployment"]["round_manifest"]
    for key in ("round", "attempt", "candidate_id", "purpose"):
        if manifest.get(key, 1 if key == "attempt" else None) != record[key]:
            raise ValueError(f"PD execution {key} differs from preflight")
    proposal = json.loads(Path(record["candidate"]["path"]).read_text(encoding="utf-8"))
    if (
        proposal.get("round") != record["round"]
        or len(proposal.get("candidates", [])) != 1
        or proposal["candidates"][0].get("candidate_id") != record["candidate_id"]
    ):
        raise ValueError("PD execution candidate identity does not match preflight")
    return record


def execution_settings(settings, benchmark):
    """Capture effective settings after normal TOML/default/environment resolution."""
    data = settings.model_dump(mode="json")
    result = {"vllm": {key: data["vllm"][key] for key in ("command", "env")}, "benchmark": data[benchmark]["command"]}
    for key in (
        "optimizer_strategy",
        "use_request_rate_calibration",
        "manage_simulator_lifecycle",
        "agent_optimizer",
        "ttft_slo",
        "tpot_slo",
        "success_rate_slo",
        "ttft_penalty",
        "tpot_penalty",
        "success_rate_penalty",
        "generate_speed_target",
    ):
        if key in data:
            result[key] = data[key]
    return result


def load_execution_settings(path):
    from ...config.config import Settings

    class RoundSettings(Settings):
        model_config = {
            **Settings.model_config,
            "toml_file": [*Settings.model_config.get("toml_file", []), Path(path).resolve()],
            "extra": "allow",
        }

    return RoundSettings()


def prepare_settings(path, benchmark):
    return execution_settings(load_execution_settings(path), benchmark)


def validate_loaded_config(context, simulator):
    """Check public service settings as well as the separately loaded PD plugin."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = Path(context["source"]["config_toml"]).resolve()
    record = validate_execution(context)
    if record is None and context.get("deployment", {}).get("round_manifest"):
        raise ValueError("PD round has not passed bound preflight; use run_pd_round.py")
    if record is None:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        command = config.get("vllm", {}).get("command")
    else:
        command = record["effective_settings"]["vllm"]["command"]
    if command is not None:
        for attribute, key in (
            ("model_path", "model"),
            ("served_model_name", "served_model_name"),
            ("vllm_others", "others"),
        ):
            if getattr(simulator, attribute, None) != command.get(key, ""):
                raise ValueError(f"Loaded PD service {key} differs from the validated configuration")
    if record is None and command is None:
        return
    from ...config.config import get_settings

    settings = get_settings()
    sources = settings.model_config.get("toml_file", [])
    if not sources or Path(sources[-1]).resolve() != path:
        raise ValueError("PD startup -c must select the exact preflight configuration")
    if record is not None and execution_settings(settings, context["benchmark"]) != record["effective_settings"]:
        raise ValueError("Loaded service/benchmark/optimizer settings differ from preflight")


def parameter_plan(fields):
    """Canonical transport plan: role argv, topology and candidate environment."""
    from ...serve_plan import _split_container_flags

    result = {}
    for field in fields:
        if field.name.startswith("OPTIX_SERVE_ARGS_"):
            containers, plain = _split_container_flags(shlex.split(field.value))
            options = {}
            index = 0
            while index < len(plain):
                flag = plain[index]
                if flag.startswith("--") and "=" in flag:
                    flag, value = flag.split("=", 1)
                    index += 1
                else:
                    value = (
                        plain[index + 1] if index + 1 < len(plain) and not plain[index + 1].startswith("--") else None
                    )
                    index += 1 if value is None else 2
                options[flag] = value
            result[field.name] = {"containers": containers, "options": options}
        else:
            result[field.name] = {"position": field.config_position, "value": field.value}
    return result


def trial_identity(context, purpose, case_id, candidate=None, launch_attempt=1):
    """Allocate identity before launch; directory names never depend on the next round."""
    manifest = context.get("deployment", {}).get("round_manifest", {})
    identity = {
        "round": manifest.get("round", 0),
        "attempt": manifest.get("attempt", 1),
        "candidate_id": manifest.get("candidate_id", "baseline"),
    }
    identity.update(candidate or {})
    if manifest:
        if purpose != "baseline" and purpose != manifest.get("purpose", "search"):
            raise ValueError("PD launch purpose differs from preflight")
        for key in ("round", "attempt") if purpose == "baseline" else ("round", "attempt", "candidate_id"):
            if identity[key] != manifest.get(key, 1 if key == "attempt" else None):
                raise ValueError(f"PD launch {key} does not match its prepared deployment")
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("Invalid PD launch case ID")
    if isinstance(launch_attempt, bool) or not isinstance(launch_attempt, int) or launch_attempt < 1:
        raise ValueError("Invalid PD launch attempt")
    identity.update(purpose=purpose, case_id=case_id, launch_attempt=launch_attempt)
    token = hashlib.sha256(str(identity["candidate_id"]).encode()).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(identity["candidate_id"]))[:64]
    identity["archive_id"] = (
        f"round-{identity['round']}.attempt-{identity['attempt']}.{purpose}.{label}-{token}.{case_id}.launch-{launch_attempt}"
    )
    return identity


def validate_result_identity(plan, case_id, candidate_id=None, archive=None):
    """Never relabel a previous attempt when a new trial failed before launch."""
    if plan.get("case_id") is not None and plan["case_id"] != case_id:
        raise ValueError("PD result belongs to a different launch case")
    if candidate_id is not None and plan.get("candidate_id") not in (None, candidate_id):
        raise ValueError("PD result candidate does not match the launched candidate")
    if archive is not None and plan.get("archive_id") is not None and Path(archive).name != plan["archive_id"]:
        raise ValueError("PD archive directory was relabeled after launch")
