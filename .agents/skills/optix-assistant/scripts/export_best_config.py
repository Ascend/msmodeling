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

"""Ownership: agent. Export best config, serve command and handoff files."""

import argparse
import re
import shlex
from math import isfinite
from pathlib import Path

from common import read_json, write_json


def _parse_others_flags(others: str):
    """Parse config.toml ``others`` into (flag, value) pairs (value may be None)."""
    tokens = shlex.split(others or "")
    flags = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags.append({"flag": tokens[i], "value": tokens[i + 1]})
                i += 2
            else:
                flags.append({"flag": tokens[i], "value": None})
                i += 1
        else:
            i += 1
    return flags


def _flag_covered_by_search(flag: str, search_names: set) -> bool:
    """Whether a config.toml ``others`` flag is already covered by the search space.

    JSON-container flags (``_CONTAINER_FLAGS``: --speculative-config /
    --compilation-config / --additional-config) are NEVER dropped here: their
    dict may carry static keys (e.g. speculative method) that the search space
    cannot express, and render_serve_command merges them with the search-side
    sub-keys (aligns with custom_command merge).
    """
    name = flag.lstrip("-").replace("-", "_").lower()
    return name in {n.lower() for n in search_names}


#: 搜索参数名 -> vLLM 真实 flag 名。本脚本须能在脱离 msmodeling 仓库的环境运行，
#: 不能 import optix.config.custom_command；此表与其 FLAG_NAME_MAP 保持一致靠
#: test_export_best_config.py 的一致性断言兜底（改任一侧都会触发测试失败）。
_FLAG_NAME_MAP = {
    "tp": "tensor-parallel-size",
    "dp": "data-parallel-size",
    "pp": "pipeline-parallel-size",
    "pcp": "prefill-context-parallel-size",
    # 主模型 eager/图模式开关 -> 顶层 --enforce-eager（一义一名）
    "enforce_eager": "enforce-eager",
}

#: 作为 JSON 容器子键渲染的搜索参数 -> (容器 flag, json key)
#: custom_command.JSON_SUBKEY_MAP 的副本（防漂移见上）。投机容器内 eager 用独立名
#: speculative_enforce_eager（容器原子性，须与 method/model 底座成对）
_JSON_SUBKEY_MAP = {
    "cudagraph_mode": ("compilation-config", "cudagraph_mode"),
    "cudagraph_capture_sizes": ("compilation-config", "cudagraph_capture_sizes"),
    "num_speculative_tokens": ("speculative-config", "num_speculative_tokens"),
    "speculative_enforce_eager": ("speculative-config", "enforce_eager"),
    # vLLM Ascend MoE 开关：官方只支持经 --additional-config 下发
    "enable_cpu_binding": ("additional-config", "enable_cpu_binding"),
    "recompute_scheduler_enable": ("additional-config", "recompute_scheduler_enable"),
    "enable_shared_expert_dp": ("additional-config", "enable_shared_expert_dp"),
    "multistream_overlap_shared_expert": ("additional-config", "multistream_overlap_shared_expert"),
}

#: _JSON_SUBKEY_MAP 用到的全部容器 flag：fixed_flags（config.toml others）里的
#: 同名容器必须并入对应 dict，否则搜索侧容器与固定底座会各出一条重复 flag
_CONTAINER_FLAGS = {flag for flag, _ in _JSON_SUBKEY_MAP.values()}

#: benchmark-only 参数，不进入 vLLM serve 命令
_BENCHMARK_ONLY = {"concurrency", "requestrate"}


def _param_to_flag(name: str) -> str:
    """Map a search-param name to the real vLLM CLI flag name (no '--')."""
    if name in _FLAG_NAME_MAP:
        return _FLAG_NAME_MAP[name]
    if name in _JSON_SUBKEY_MAP:
        return _JSON_SUBKEY_MAP[name][0]
    return name.lower().replace("_", "-")


def _round_parameter_metadata(context: dict, best: dict) -> dict:
    """Read field positions from the winning round's archived context."""
    if not isinstance(context, dict):
        raise ValueError("Archived context must be an object")
    space = context.get("search_space")
    if not isinstance(space, dict):
        raise ValueError("Missing or invalid search_space in archived context")
    metadata = {}
    for group in ("parameters", "constants"):
        entries = space.get(group)
        if not isinstance(entries, list):
            raise ValueError(f"Archived search_space.{group} must be an array")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Archived definitions must be objects")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip() or name in metadata:
                raise ValueError(f"Missing or duplicate archived parameter: {name!r}")
            position = entry.get("config_position")
            if not isinstance(position, str) or not position.strip():
                raise ValueError(f"{name}: archived config_position is required")
            metadata[name] = entry
    missing = set(best.get("params", {})) - metadata.keys()
    if missing:
        raise ValueError(f"Winning context lacks definitions: {sorted(missing)}")
    return metadata


def _split_export_params(params: dict, parameter_metadata: dict | None = None) -> tuple[dict, dict]:
    """Separate serve flags and environment overrides using execution metadata."""
    run_params, env = {}, {}
    for name, value in (params or {}).items():
        if name.lower() in _BENCHMARK_ONLY:
            continue
        position = ((parameter_metadata or {}).get(name) or {}).get("config_position", "run")
        if position == "run":
            run_params[name] = value
        elif position == "env":
            env_name = name.upper().strip()
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", env_name):
                raise ValueError(f"Invalid environment variable name: {name!r}")
            env[env_name] = None if value is None or (isinstance(value, str) and not value.strip()) else str(value)
        else:
            raise ValueError(f"Cannot export {name}: unsupported config_position {position!r}")
    return run_params, env


def render_serve_command(
    model: str,
    served_name: str,
    host: str,
    port: str,
    params: dict,
    fixed_flags: list,
    parameter_metadata: dict | None = None,
) -> list:
    """Render a full ``vllm serve`` command from best params + config.toml fixed flags.

    - ``params``: best values (recommended_params); only run fields become flags
    - ``parameter_metadata``: field definitions from the winning round
    - ``fixed_flags``: config.toml ``others`` flags not covered by search
      (e.g. --seed, --quantization ascend)

    Returns the argv list (first token is ``vllm``). JSON-subkey params
    (num_speculative_tokens / cudagraph_mode / cudagraph_capture_sizes /
    speculative_enforce_eager) merge into their container flag; plain params
    render as their own ``--flag value``. `fixed_flags` container dicts act as
    the static base (e.g. speculative method), search sub-keys override — the
    same merge semantics as optix custom_command.
    """
    params, _ = _split_export_params(params, parameter_metadata)
    cmd = ["vllm", "serve", model, "--served-model-name", served_name, "--host", str(host), "--port", str(port)]

    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from optix.serve_plan import render_serve_flags

    tokens = []
    for entry in fixed_flags or []:
        tokens.append(entry["flag"])
        if entry.get("value") is not None:
            tokens.append(str(entry["value"]))
    fields = []
    for name, value in params.items():
        if isinstance(value, (dict, list)):
            import json

            value = json.dumps(value)
        fields.append(SimpleNamespace(name=name, value=value))
    return cmd + render_serve_flags(fields, shlex.join(tokens))


def _check_measured_speculation(script, params, role=None):
    """A historical ineffective candidate cannot become a valid export by repair."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from optix.serve_plan import validate_measured_speculation

    expected = params.get("num_speculative_tokens")
    container = params.get("speculative_config")
    if role:
        expected = params.get("num_speculative_tokens_" + role, expected)
        container = params.get("speculative_config_" + role, container)
    if container is not None:
        import json

        container = json.loads(container) if isinstance(container, str) else container
        expected = expected if expected is not None else container.get("num_speculative_tokens")
    validate_measured_speculation(script, expected)


def _export_multihost_snapshot(run_dir: Path, context: dict, best: dict, parameter_metadata: dict) -> None:
    """Export measured per-node scripts without needing OptiX or a live cluster."""
    plan = (best.get("metadata") or {}).get("launch_plan")
    nodes = (context.get("hardware") or {}).get("num_nodes", 0)
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1:
        raise ValueError("Winning context must record a positive node count")
    expected = {"start_node.sh", *[f"start_work_{index}.sh" for index in range(nodes - 1)]}
    if (
        not isinstance(plan, dict)
        or set(plan) != expected
        or not all(isinstance(s, str) and s.strip() for s in plan.values())
    ):
        raise ValueError("Winning multihost trial has no complete measured launch scripts; rerun the trial")
    for script in plan.values():
        _check_measured_speculation(script, best.get("params", {}))
    directory = run_dir / "best_deployment"
    directory.mkdir(exist_ok=True)
    for name, script in plan.items():
        target = directory / name
        target.write_text(script, encoding="utf-8")
        target.chmod(0o755)
    manifest = {
        "source_ref": best["source_ref"],
        "deployment": "multihost",
        "scripts": {name: f"best_deployment/{name}" for name in sorted(plan)},
        "instructions": "Run start_node.sh on the configured master and each start_work_N.sh on its corresponding worker.",
    }
    write_json(run_dir / "serve_command_multihost.json", manifest)
    write_json(
        run_dir / "optimizer_config_handoff.json",
        {
            "source": "optix-assistant",
            "recommended_params": best.get("params", {}),
            "parameter_metadata": parameter_metadata,
            "fitness": best["fitness"],
            "performance": best.get("performance"),
            "deployment": manifest,
            "selection": best["selection_policy"],
        },
    )


def _candidate_trial_types(path: Path) -> dict | None:
    """Resolve types from the same round's immutable candidate archive."""
    if not path.exists():
        return None
    data = read_json(path)
    if not isinstance(data, dict) or data.get("round") != int(path.stem.rsplit("-", 1)[1]):
        raise ValueError(f"{path.name}: candidate round does not match filename")
    directions = {}
    for direction in data.get("directions", []):
        did = direction.get("direction_id")
        if not isinstance(did, str) or not did.strip() or did in directions:
            raise ValueError(f"{path.name}: missing or duplicate direction_id {did!r}")
        if direction.get("diagnostic_only") or "diagnostic_only" in (direction.get("constraints") or []):
            raise ValueError(f"{path.name}: replace diagnostic_only with trial_type: diagnosis")
        kind = direction.get("trial_type", "optimization")
        if kind not in ("optimization", "diagnosis"):
            raise ValueError(f"{path.name}: invalid trial_type {kind!r}")
        directions[did] = kind
    classified = any("trial_type" in d for d in data.get("directions", []))
    if classified and any("trial_type" not in d for d in data.get("directions", [])):
        raise ValueError(f"{path.name}: every direction must declare trial_type")
    types = {}
    for index, candidate in enumerate(data.get("candidates", [])):
        # Match the original engine's Candidate.from_dict ID convention.
        cid = str(candidate.get("candidate_id") or candidate.get("id") or f"candidate-{index + 1}")
        if cid in types:
            raise ValueError(f"{path.name}: duplicate candidate_id {cid!r}")
        did = candidate.get("direction_id")
        if did is not None and (not isinstance(did, str) or did not in directions):
            raise ValueError(f"{path.name}: unknown direction_id {did!r}")
        if did is None and (directions or "trial_type" in candidate):
            raise ValueError(f"{path.name}: candidate {cid!r} has no direction_id")
        kind = directions[did] if did is not None else "optimization"
        if candidate.get("trial_type", kind) != kind:
            raise ValueError(f"{path.name}: candidate {cid!r} trial_type conflicts with direction")
        types[cid] = kind
    return types


def _is_optimization_trial(trial: dict, types: dict | None, candidates_path: Path) -> bool:
    """Use candidate provenance, checking any existing result classification."""
    recorded = (trial.get("metadata") or {}).get("trial_type")
    if recorded is not None and recorded not in ("optimization", "diagnosis"):
        raise ValueError(f"Invalid result trial_type {recorded!r}")
    if types is None:
        if recorded is None:
            raise ValueError(f"Missing {candidates_path.name}: cannot determine trial_type from unclassified results")
        return recorded == "optimization"
    cid = trial.get("candidate_id")
    if cid not in types:
        raise ValueError(f"{candidates_path.name}: result candidate_id {cid!r} not found")
    kind = types[cid]
    if recorded is not None and recorded != kind:
        raise ValueError(f"{candidates_path.name}: result trial_type conflicts with candidate {cid!r}")
    return kind == "optimization"


def _export_pd_snapshot(run_dir: Path, context: dict, best: dict, parameter_metadata: dict) -> None:
    """Export the complete winning deployment, without importing the runtime plugin."""
    import hashlib

    plan = (best.get("metadata") or {}).get("launch_plan") or {}
    files = plan.get("files")
    nodes = plan.get("nodes")
    if files is None and plan.get("archive_id"):
        archive_id = plan["archive_id"]
        if not isinstance(archive_id, str) or Path(archive_id).name != archive_id or archive_id in {".", ".."}:
            raise ValueError("Invalid PD archive identity")
        archive_root = (run_dir / "pd_deployments").resolve()
        archive = (archive_root / archive_id / "deployment.json").resolve()
        if not archive.is_relative_to(archive_root):
            raise ValueError("PD archive escapes deployment directory")
        recorded = read_json(archive)
        for key in ("archive_id", "candidate_id", "case_id", "config_sha256", "layout"):
            if recorded.get(key) != plan.get(key):
                raise ValueError(f"PD archive {key} does not match winning trial")
        files = recorded.get("files")
    if not isinstance(files, dict) or not isinstance(nodes, list) or not nodes:
        raise ValueError("Winning PD trial has no measured deployment snapshot")
    if {node.get("role") for node in nodes} != {"prefill", "decode", "proxy"}:
        raise ValueError("Winning PD snapshot must include P, D and proxy nodes")
    dependencies = {"load_balance_proxy_server_example.py", "load_balance_proxy_layerwise_server_example.py"}
    required = dependencies | {node.get("script", "") for node in nodes}
    if not required <= set(files):
        raise ValueError("Winning PD snapshot is missing node scripts or proxy dependencies")
    for name, content in files.items():
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Unsafe PD deployment artifact name")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty PD deployment artifact")
    if any("<%" in files[node["script"]] for node in nodes):
        raise ValueError("PD node scripts contain unresolved template tags")
    config_text = plan.get("config_toml")
    expected = (context.get("deployment") or {}).get("config_sha256")
    if not isinstance(config_text, str) or hashlib.sha256(config_text.encode()).hexdigest() != expected:
        raise ValueError("Winning PD configuration is missing or does not match its context")
    verification = {"status": "pending"}
    summary_path = run_dir / "pd_search_summary.json"
    if summary_path.exists():
        candidate_verification = read_json(summary_path).get("verification", {})
        if (
            candidate_verification.get("status") == "verified"
            and candidate_verification.get("winner_config_sha256") == expected
            and candidate_verification.get("winner_params") == best.get("params")
            and candidate_verification.get("benchmark_spec") == context.get("benchmark_spec")
        ):
            verification = candidate_verification
    for node in nodes:
        if node.get("role") in {"prefill", "decode"}:
            _check_measured_speculation(files[node["script"]], best.get("params", {}), node["role"])
    directory = run_dir / "best_deployment"
    if directory.is_symlink():
        raise ValueError("Deployment output directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    # The plugin's generic guide describes a different output directory and
    # benchmark. Generate instructions for this archived deployment instead.
    export_files = {name: content for name, content in files.items() if name != "pd-result-reproduction.md"}
    for name, content in export_files.items():
        target = directory / name
        if target.is_symlink():
            raise ValueError("Deployment output file must not be a symlink")
        target.write_text(content, encoding="utf-8")
        if name.endswith(".sh"):
            target.chmod(0o755)
    target_config = directory / "config.toml"
    if target_config.is_symlink():
        raise ValueError("Deployment configuration output must not be a symlink")
    target_config.write_text(config_text, encoding="utf-8")
    readme = directory / "README.md"
    if readme.is_symlink():
        raise ValueError("Deployment README must not be a symlink")
    lines = [
        "# PD 部署说明",
        "",
        "以下节点与脚本来自胜出 trial 的部署快照。将所列脚本及依赖复制到各节点的目标目录。",
        "",
        "| 角色 | 主机 | 端口 | 脚本 |",
        "| --- | --- | --- | --- |",
    ]
    for node in nodes:
        lines.append(
            f"| {node.get('role')} | {node.get('ssh_ip', node.get('node_config', {}).get('ssh_ip'))} | {node.get('port')} | {node.get('script')} |"
        )
    lines.extend(
        [
            "",
            f"目标目录：{plan['remote_dir']}",
            "",
            "按节点执行对应 P、D 脚本并确认服务就绪，再启动 proxy 脚本。",
            "停止及健康检查请参照归档的 stop_pd_process.sh、check_pd_process.sh，执行前检查脚本中的节点和路径。",
            "复测使用原实验记录的 benchmark 和 workload；请参照该次实验配置执行，基线和胜出方案均应独立复测。",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "deployment": "pd",
        "source_ref": best["source_ref"],
        "config_sha256": expected,
        "config": "best_deployment/config.toml",
        "layout": plan["layout"],
        "nodes": nodes,
        "files": {name: "best_deployment/" + name for name in sorted(export_files)},
        "remote_dir": plan["remote_dir"],
        "verification": verification,
        "instructions": "请参阅 best_deployment/README.md，按节点复制文件、启动并复测。",
    }
    write_json(run_dir / "serve_command_pd.json", manifest)
    write_json(
        run_dir / "optimizer_config_handoff.json",
        {
            "source": "optix-assistant",
            "recommended_params": best.get("params", {}),
            "parameter_metadata": parameter_metadata,
            "fitness": best["fitness"],
            "performance": best.get("performance"),
            "deployment": manifest,
            "selection": best["selection_policy"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Export the best agent optimizer result.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="覆盖配置路径；未指定时使用获胜轮 context.round-N.json 的 source.config_toml，缺省 <cwd>/optix/config.toml",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trials = []
    try:
        for path in sorted(run_dir.glob("results.round-*.json")):
            candidates_path = path.with_name(path.name.replace("results.round-", "candidates.round-", 1))
            types = _candidate_trial_types(candidates_path)
            for idx, trial in enumerate(read_json(path).get("trials", [])):
                if trial.get("early_exit") or (trial.get("performance") or {}).get("early_exit"):
                    continue
                fitness = trial.get("fitness")
                if trial.get("error") is not None or not isinstance(fitness, (int, float)) or not isfinite(fitness):
                    continue
                if _is_optimization_trial(trial, types, candidates_path):
                    trial["_source_ref"] = f"{path.name}#trials[{idx}]"
                    trials.append(trial)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.exit(1, f"Cannot classify trial results: {exc}\n")
    trials = [
        trial
        for trial in trials
        if ((trial.get("metadata") or {}).get("deployment_proposal") or {}).get("purpose", "search") == "search"
    ]
    if not trials:
        raise SystemExit(
            "No successful optimization trial results found "
            "(all trials failed, exited early, or are diagnosis-only); nothing to export."
        )
    # Hard-constraint preference: prefer the best SLO-compliant trial; only fall
    # back to pure fitness when no trial meets the hard constraints.
    compliant = [t for t in trials if t.get("slo_violated") is False]
    pool = compliant if compliant else trials
    best = min(pool, key=lambda item: item.get("fitness", float("inf")))
    best["slo_compliant"] = bool(compliant)
    selection = "slo_compliant_best" if compliant else "fallback_fitness_best"
    best["selection_policy"] = selection
    best["source_ref"] = best.pop("_source_ref", "")
    from datetime import datetime, timezone

    best["selected_at"] = datetime.now(timezone.utc).isoformat()

    round_numbers = {t["_source_ref"].split("#", 1)[0] for t in trials if "_source_ref" in t}
    round_numbers.add(best["source_ref"].split("#", 1)[0])

    # Read definitions and configuration source from the winning round.
    result_name = best["source_ref"].split("#", 1)[0]
    context_path = run_dir / result_name.replace("results.round-", "context.round-", 1)
    if not context_path.is_file():
        # 按轮归档是后引入的规则，老 run 只有 context.json —— 回退兼容
        fallback = run_dir / "context.json"
        import sys as _sys

        print(
            f"  ⚠ WARNING: 按轮归档 {context_path.name} 缺失，回退读取 {fallback.name}"
            "（老 run 无按轮归档，导出使用的是当前 context，可能与获胜轮定义不一致）。",
            file=_sys.stderr,
        )
        if (read_json(fallback).get("deployment") or {}).get("type") == "pd":
            parser.exit(1, "PD export requires the winning round context archive\n")
        context_path = fallback
    try:
        context = read_json(context_path)
        parameter_metadata = _round_parameter_metadata(context, best)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Cannot load winning context {context_path.name}: {exc}\n")
    best["trial_count"] = len(trials)
    best["round_count"] = len(round_numbers)
    if (context.get("deployment") or {}).get("type") == "pd":
        try:
            _export_pd_snapshot(run_dir, context, best, parameter_metadata)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(1, f"Cannot export PD snapshot: {exc}\n")
        write_json(run_dir / "best_result.json", best)
        return
    if (context.get("deployment") or {}).get("type") == "multihost":
        try:
            _export_multihost_snapshot(run_dir, context, best, parameter_metadata)
        except (OSError, ValueError, TypeError) as exc:
            parser.exit(1, f"Cannot export multihost snapshot: {exc}\n")
        write_json(run_dir / "best_result.json", best)
        return
    write_json(run_dir / "best_result.json", best)
    config_path = args.config or (context.get("source") or {}).get("config_toml")
    config_path = Path(config_path) if config_path else Path.cwd() / "optix" / "config.toml"
    toml_data = {}
    if config_path.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with config_path.open("rb") as stream:
                toml_data = tomllib.load(stream)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Cannot read export configuration: {exc}\n")
    engine = (context.get("engine") or toml_data.get("framework") or "vllm").lower()
    if engine != "vllm":
        parser.exit(1, f"Serve command export supports vllm only, got {engine!r}.\n")
    section = next((v for k, v in toml_data.items() if k.lower() == engine), {})
    try:
        run_params, candidate_env = _split_export_params(best.get("params", {}), parameter_metadata)
        serve_env = {name: str(value) for name, value in (section.get("env") or {}).items()}
        serve_env.update(candidate_env)
        for name in serve_env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid environment variable name: {name!r}")
    except ValueError as exc:
        parser.exit(1, f"Cannot export serve configuration: {exc}\n")
    others = str((section.get("command") or {}).get("others") or "")
    config_fixed_params = [
        entry for entry in _parse_others_flags(others) if not _flag_covered_by_search(entry["flag"], set(run_params))
    ]
    for entry in config_fixed_params:
        entry["source"] = f"config.toml [{engine}.command].others"

    # 渲染完整 vllm serve 命令：best 参数 + config.toml 固定 flag
    serve_command = []
    # command 骨架从 config.toml [<engine>.command] 读取
    cmd_toml = section.get("command") or {}
    model = cmd_toml.get("model", "") or (context.get("model") or {}).get("path", "")
    served_name = cmd_toml.get("served_model_name", "") or (context.get("model") or {}).get("name", "")
    host = cmd_toml.get("host", "127.0.0.1")
    port = cmd_toml.get("port", "8000")
    # Guard against placeholder/default host/port silently leaking into the
    # exported serve command. Placeholder "port"/"host" would
    # otherwise render `--port port` and fail at launch.
    if not port or str(port).strip().lower() in {"port", "8000"}:
        import sys as _sys

        print(
            f"  ⚠ WARNING: config.toml [{engine}.command].port 缺失或为占位符/默认值 "
            f"({port!r})，serve_command 将使用该值；若与实际服务端口不符请核对 config.toml。",
            file=_sys.stderr,
        )
    if not host or str(host).strip().lower() in {"host", "127.0.0.1"}:
        import sys as _sys

        print(
            f"  ⚠ WARNING: config.toml [{engine}.command].host 缺失或为占位符 ({host!r})，"
            "serve_command 将使用该值；请核对 config.toml。",
            file=_sys.stderr,
        )
    if model and served_name:
        try:
            serve_command = render_serve_command(
                model,
                served_name,
                host,
                port,
                best.get("params", {}),
                config_fixed_params,
                parameter_metadata=parameter_metadata,
            )
            command = shlex.join(serve_command)
            command_data = {"argv": serve_command, "command": command}
            if serve_env:
                command_data["env"] = serve_env
            write_json(run_dir / "serve_command.json", command_data)
            shell_lines = ["#!/usr/bin/env bash"]
            for name, value in sorted(serve_env.items()):
                shell_lines.append(f"unset {name}" if value is None else f"export {name}={shlex.quote(value)}")
            shell_lines.append(command)
            (run_dir / "serve_command.sh").write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
        except Exception as exc:
            parser.exit(1, f"Cannot export serve command: {exc}\n")

    write_json(
        run_dir / "optimizer_config_handoff.json",
        {
            "source": "optix-assistant",
            "recommended_params": best.get("params", {}),
            "parameter_metadata": parameter_metadata,
            "fitness": best.get("fitness"),
            "performance": best.get("performance"),
            "slo_violated": best.get("slo_violated"),
            "slo_compliant": bool(compliant),
            "selection": selection,
            "config_fixed_params": config_fixed_params,
            "serve_command": serve_command,
            **({"serve_env": serve_env} if serve_env else {}),
        },
    )


if __name__ == "__main__":
    main()
