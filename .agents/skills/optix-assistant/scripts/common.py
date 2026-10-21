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

"""Ownership: lib. Shared helpers for optix-assistant scripts."""

import json
import os
import hashlib
import re
import shlex
import tomlkit
from pathlib import Path
from math import isclose
from typing import Any, Dict, List

__all__ = [
    "read_json",
    "write_json",
    "load_context",
    "convert_numeric",
    "values_equal",
    "infer_model_family",
    # others / JSON-container helpers
    "COMMAND_SKELETON_KEYS",
    "CONTAINER_FLAGS",
    "build_others",
    "merge_others",
    # tomlkit table navigation + others-JSON validation (shared with
    # config_writer / config_preflight, single implementation)
    "table_or_create",
    "bad_json_tokens",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_context(run_dir: Path) -> Dict[str, Any]:
    return read_json(run_dir / "context.json")


def convert_numeric(value: float, dtype: str):
    if dtype == "int":
        return int(value)
    if dtype == "bool":
        return bool(value)
    return value


def values_equal(left: Any, right: Any) -> bool:
    try:
        return isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-8)
    except (TypeError, ValueError):
        return str(left) == str(right)


# ---------------------------------------------------------------------------
# Model family inference for recommendation matching
# ---------------------------------------------------------------------------

_CORE_FAMILIES = [
    "deepseek",
    "llama",
    "qwen",
    "yi",
    "mistral",
    "baichuan",
    "glm",
    "internlm",
    "gemma",
    "phi",
    "falcon",
    "gpt",
    "dbrx",
]


def infer_model_family(context: Dict[str, Any]) -> str:
    """Extract model family from context.model or model_info.

    Priority:
      1. context.model.family (if explicitly set in context)
      2. context.model.name string matching
      3. context.model.path string matching
      4. architectures field from config.json
    """
    model = context.get("model") or context.get("model_info") or {}

    explicit = model.get("family")
    if explicit and str(explicit).strip():
        return str(explicit).strip().lower()

    combined = (
        (model.get("name") or "") + " " + (model.get("path") or "") + " " + (model.get("model_type") or "")
    ).lower()
    for family in _CORE_FAMILIES:
        if family in combined:
            return family

    architectures = model.get("architectures") or []
    if architectures:
        text = " ".join(architectures).lower()
        for family in _CORE_FAMILIES:
            if family in text:
                return family

    return "unknown"


# ---------------------------------------------------------------------------
# others / JSON-container helpers
#
# Shared between apply_launch_config.py (preset additional_config → others)
# and config_writer.py (manual/structured container writes). A JSON-container
# flag (--speculative-config / --compilation-config / --additional-config) is a
# vLLM flag whose value is a JSON dict. The dict must be written single-quoted
# (`--flag '{"k": "v"}'`) so shlex.split() treats it as ONE token; a bare {…}
# would be split on inner spaces and break both config validation and vLLM
# parsing. Two containers for the same flag are merged key-wise (new wins).
# ---------------------------------------------------------------------------

#: keys in additional_config that are command-skeleton fields (config.toml
#: command has dedicated host/port/model/served_model_name) — not `others` flags
COMMAND_SKELETON_KEYS = {"host", "port", "model", "served_model_name", "served-model-name"}

#: JSON-container flags (vLLM): their value is a JSON dict, and a flag may
#: legally appear once with a *merged* dict. Unlike plain flags, an existing
#: container must NOT be treated as immutable — the new dict's keys must be
#: merged over the old dict (e.g. speculative-config may already hold `method`
#: without `num_speculative_tokens`; re-applying must add the missing key, not
#: skip the whole flag because it "already exists").
CONTAINER_FLAGS = ("--speculative-config", "--compilation-config", "--additional-config")


def build_others(additional_config: dict) -> str:
    """Turn a dict of flags into a vLLM command `others` string.

    JSON dict/list values become single-quoted container flags
    (``--flag '{"k": "v"}'``); bool True → bare ``--flag``; other scalars →
    ``--flag value``. Skeleton command fields (host/port/model/…) are skipped.

    Every value token is emitted via :func:`shlex.quote`, so a value containing
    spaces or quotes (paths, names) survives any later shlex.split round-trip
    (merge_others re-parses this string) — the output is always canonical shell
    text, never a bare value that would re-split on inner whitespace.
    """
    flags: List[str] = []
    for key, value in additional_config.items():
        if key in COMMAND_SKELETON_KEYS:
            continue
        if isinstance(value, bool):
            if value:
                flags.append(f"--{key}")
        elif value not in (None, ""):
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
            flags.append(f"--{key} {shlex.quote(text)}")
    return " ".join(flags)


def _split_flag_groups(tokens: List[str]) -> List[List[str]]:
    """Group flat CLI tokens into (--flag, [value...]) groups, preserving order."""
    groups: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    if current:
        groups.append(current)
    return groups


def _flag_value_dict(group: List[str]) -> Dict[str, Any] | None:
    """Parse a container group's single JSON value into a dict; None if not a dict."""
    if len(group) < 2:
        return None
    raw = " ".join(group[1:])
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def merge_others(existing: str, new: str) -> str:
    """Merge ``new`` flag groups into ``existing``.

    - Plain flags: idempotent — groups already present in ``existing`` are kept
      as-is; new groups are appended.
    - JSON-container flags: the flag appears once; the existing dict is the base
      and ``new``'s keys are merged OVER it (new wins). This lets a re-apply
      repair an incomplete container (e.g. add num_speculative_tokens next to an
      existing method) instead of skipping the whole flag because it exists.
    """
    if not new.strip():
        return existing
    if not existing.strip():
        return new.strip()

    existing_groups = _split_flag_groups(shlex.split(existing))
    new_groups = _split_flag_groups(shlex.split(new))

    existing_container_by_flag: Dict[str, List[str]] = {g[0]: g for g in existing_groups if g[0] in CONTAINER_FLAGS}
    existing_plain_flags = {g[0] for g in existing_groups if g[0] not in CONTAINER_FLAGS}

    merged: List[List[str]] = []
    used_container: set = set()

    # 1. Rebuild existing groups in place. Every container group is re-serialized
    #    as [flag, <json text>] (argv tokens — quoting happens once, in the
    #    serializer below): shlex.split() stripped the single quotes, so appending
    #    the group verbatim would emit a bare {…} that re-splits on inner spaces.
    #    When the new side extends the same flag, its keys are merged OVER the
    #    existing dict (new wins).
    for group in existing_groups:
        flag = group[0]
        if flag in CONTAINER_FLAGS:
            old_dict = _flag_value_dict(group)
            new_cont = next((g for g in new_groups if g[0] == flag), None)
            new_dict = _flag_value_dict(new_cont) if new_cont is not None else None
            if old_dict is not None:
                base = dict(old_dict)
                if new_dict is not None:
                    base.update(new_dict)  # new keys win
                merged.append([flag, json.dumps(base, ensure_ascii=False)])
                used_container.add(flag)
                continue
            merged.append(group)  # unparseable container — keep verbatim
            continue
        merged.append(group)

    # 2. Append new plain groups not already present.
    for group in new_groups:
        flag = group[0]
        if flag in CONTAINER_FLAGS:
            if flag not in used_container and flag not in existing_container_by_flag:
                merged.append(group)  # genuinely new container (no JSON to merge)
            continue
        if flag not in existing_plain_flags:
            merged.append(group)

    # 3. Serialize argv tokens back to a shell string. Values are re-quoted with
    #    shlex.quote so a token containing spaces survives the round-trip: the
    #    split above stripped its quotes (e.g. `--log-file '/tmp/a b.log'` →
    #    token `/tmp/a b.log`), and quoting it again emits `'/tmp/a b.log'`.
    #    Container JSON gets its canonical single-quoted wrapper from the same
    #    call (shlex.quote quotes any string with whitespace/quotes); safe scalar
    #    tokens (32768, 1.5, FULL_DECODE_ONLY) come back unchanged.
    parts: List[str] = []
    for group in merged:
        parts.append(group[0])
        parts.extend(shlex.quote(tok) for tok in group[1:])
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# tomlkit table navigation + others-JSON validation.
#
# config_writer.py and config_preflight.py each used to carry their own copy
# of these helpers; they are shared here so table walking and the
# "$VAR-masked json.loads" rule live in ONE implementation.
# ---------------------------------------------------------------------------
def table_or_create(parent: Any, key: str) -> Any:
    """Return the child table at parent[key], creating it as a tomlkit
    Table when absent (a plain dict would serialize into malformed TOML).
    """
    child = parent.get(key)
    if not isinstance(child, tomlkit.items.Table):
        child = tomlkit.table()
        parent[key] = child
    return child


def bad_json_tokens(others: str) -> List[str]:
    """Return container tokens that look like JSON but fail to parse.

    A token starting with { is a vLLM container value (single-quoted or
    bare). $VAR references are placeholders, not JSON - mask them (each
    $NAME -> 1) before parsing. Shared by config_writer.check_protocol
    and config_preflight's completeness gate so the mask/parse rule lives in
    one place.
    """
    bad: List[str] = []
    for token in shlex.split(others or ""):
        if not token.startswith("{"):
            continue
        masked = re.sub(r"\$[A-Z0-9_]+", "1", token)
        try:
            json.loads(masked)
        except ValueError:
            bad.append(token)
    return bad


CLUSTER_CONFIG_ENV = "CLUSTER_CONFIG_PATH"


def resolve_cluster_config_path(explicit: str | None = None) -> Path | None:
    """Resolve the cluster config path: explicit arg > CLUSTER_CONFIG_PATH env."""
    raw = explicit or os.environ.get(CLUSTER_CONFIG_ENV, "")
    return Path(raw).resolve() if raw else None


def _load_toml_file(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11 only
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _cluster_host(entry: Dict[str, Any]) -> str:
    return str(entry.get("host") or "").strip()


def load_cluster_topology(path: str | None = None) -> Dict[str, Any]:
    """Parse the multihost plugin's cluster config into a normalized dict.

    Always returns a dict. ``ok`` is False — with ``configured`` telling apart
    "no cluster config given" from "given but broken" — when the file is
    absent/unparsable or lacks the ``[vllm_mix]`` section, so callers can turn
    that into an ERROR without exception handling.

    On success:
        {ok, configured, path, chips_per_node, master_host, nodes, workers,
         num_nodes, data_parallel_rpc_port, ready_log_pattern, docker_use_sudo}
    where ``nodes``/``workers`` are the raw node dicts and ``num_nodes`` is
    master + workers (``all_nodes`` order in the plugin: master first).
    """
    resolved = resolve_cluster_config_path(path)
    if resolved is None:
        return {
            "ok": False,
            "configured": False,
            "path": "",
            "error": f"未配置集群配置（--cluster-config 与 {CLUSTER_CONFIG_ENV} 均为空）",
        }
    if not resolved.exists():
        return {"ok": False, "configured": True, "path": str(resolved), "error": f"集群配置不存在: {resolved}"}
    try:
        data = _load_toml_file(resolved)
    except (OSError, ValueError) as exc:
        return {"ok": False, "configured": True, "path": str(resolved), "error": f"集群配置解析失败: {exc}"}

    section = data.get("vllm_mix")
    if not isinstance(section, dict):
        return {
            "ok": False,
            "configured": True,
            "path": str(resolved),
            "error": "集群配置缺少 [vllm_mix] 段（multihost_infer 插件专用格式）",
        }

    nodes = [n for n in (section.get("node") or []) if isinstance(n, dict)]
    workers = [w for w in (section.get("workers") or []) if isinstance(w, dict)]
    if len(nodes) != 1 or not all(_cluster_host(node) for node in nodes + workers):
        return {
            "ok": False,
            "configured": True,
            "path": str(resolved),
            "error": "Exactly one master and a host for every worker are required",
        }
    if len({_cluster_host(node) for node in nodes + workers}) != len(nodes) + len(workers):
        return {"ok": False, "configured": True, "path": str(resolved), "error": "Cluster node hosts must be unique"}
    chips_per_node = section.get("chips_per_node", 8)
    try:
        if isinstance(chips_per_node, bool) or float(chips_per_node) != int(chips_per_node):
            raise ValueError("chips_per_node must be a positive integer")
        chips_per_node = int(chips_per_node)
    except (TypeError, ValueError):
        chips_per_node = 0

    if chips_per_node <= 0:
        return {"ok": False, "configured": True, "path": str(resolved), "error": "chips_per_node must be positive"}
    return {
        "ok": True,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "configured": True,
        "path": str(resolved),
        "chips_per_node": chips_per_node,
        "master_host": _cluster_host(nodes[0]) if nodes else "",
        "nodes": nodes,
        "workers": workers,
        "num_nodes": len(nodes) + len(workers),
        "data_parallel_rpc_port": section.get("data_parallel_rpc_port", 0),
        "ready_log_pattern": section.get("ready_log_pattern", ""),
        "docker_use_sudo": bool(data.get("docker_use_sudo", False)),
        # [vllm_mix.env] is the static base exported into every node script
        "env": {
            str(key): str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in (section.get("env") or {}).items()
        },
    }
