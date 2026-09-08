"""Ownership: agent. Export best config, serve command and handoff files."""

import argparse
import shlex
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

    JSON-container flags (--speculative-config / --compilation-config) are NEVER
    dropped here: their dict may carry static keys (e.g. speculative method) that
    the search space cannot express, and render_serve_command merges them with the
    search-side sub-keys (aligns with custom_command merge).
    """
    name = flag.lstrip("-").replace("-", "_").lower()
    return name in {n.lower() for n in search_names}


#: 搜索参数名 -> vLLM 真实 flag 名（与 optix/config/custom_command.py FLAG_NAME_MAP 对齐）
_FLAG_NAME_MAP = {
    "tp": "tensor-parallel-size",
    "dp": "data-parallel-size",
    "pp": "pipeline-parallel-size",
    # 主模型 eager/图模式开关 -> 顶层 --enforce-eager（一义一名）
    "enforce_eager": "enforce-eager",
}

#: 作为 JSON 容器子键渲染的搜索参数 -> (容器 flag, json key)
#: 与 custom_command.JSON_SUBKEY_MAP 对齐：投机容器内 eager 用独立名
#: speculative_enforce_eager（容器原子性，须与 method/model 底座成对）
_JSON_SUBKEY_MAP = {
    "cudagraph_mode": ("compilation-config", "cudagraph_mode"),
    "cudagraph_capture_sizes": ("compilation-config", "cudagraph_capture_sizes"),
    "num_speculative_tokens": ("speculative-config", "num_speculative_tokens"),
    "speculative_enforce_eager": ("speculative-config", "enforce_eager"),
}

#: benchmark-only 参数，不进入 vLLM serve 命令
_BENCHMARK_ONLY = {"concurrency", "requestrate"}


def _param_to_flag(name: str) -> str:
    """Map a search-param name to the real vLLM CLI flag name (no '--')."""
    if name in _FLAG_NAME_MAP:
        return _FLAG_NAME_MAP[name]
    if name in _JSON_SUBKEY_MAP:
        return _JSON_SUBKEY_MAP[name][0]
    return name.lower().replace("_", "-")


def render_serve_command(
    model: str,
    served_name: str,
    host: str,
    port: str,
    params: dict,
    fixed_flags: list,
) -> list:
    """Render a full ``vllm serve`` command from best params + config.toml fixed flags.

    - ``params``: search-space best values (recommended_params)
    - ``fixed_flags``: config.toml ``others`` flags not covered by search
      (e.g. --seed, --quantization ascend)

    Returns the argv list (first token is ``vllm``). JSON-subkey params
    (num_speculative_tokens / cudagraph_mode / cudagraph_capture_sizes /
    speculative_enforce_eager) merge into their container flag; plain params
    render as their own ``--flag value``. `fixed_flags` container dicts act as
    the static base (e.g. speculative method), search sub-keys override — the
    same merge semantics as optix custom_command.
    """
    cmd = ["vllm", "serve", model, "--served-model-name", served_name, "--host", host, "--port", port]

    # 1. Merge container flags: fixed(others) dict is the base, search sub-keys
    #    override (same precedence as custom_command). Plain fixed flags stay
    #    for the dedupe pass at the end.
    containers: dict = {}
    plain: list = []
    fixed_plain: list = []
    for entry in fixed_flags or []:
        flag = entry.get("flag", "")
        value = entry.get("value")
        cname = flag.lstrip("-")
        if cname in {"speculative-config", "compilation-config"} and isinstance(value, str):
            try:
                import json as _json

                containers.setdefault(cname, {}).update(_json.loads(value))
            except (ValueError, TypeError):
                fixed_plain.append(entry)  # 畸形容器值保持原样输出
        else:
            fixed_plain.append(entry)

    for name, value in (params or {}).items():
        if name.lower() in _BENCHMARK_ONLY:
            continue
        if value is None or value == "":
            continue
        if name in _JSON_SUBKEY_MAP:
            flag, key = _JSON_SUBKEY_MAP[name]
            # 若 value 是 JSON 字符串（如 "[1,6,12]"）解析回数组
            if isinstance(value, str):
                try:
                    import json as _json

                    value = _json.loads(value)
                except (ValueError, TypeError):
                    pass
            containers.setdefault(flag, {})[key] = value
        else:
            plain.append((name, value))

    # Container atomicity: num_speculative_tokens==0 means
    # speculation OFF -> drop the whole container; a merged speculative-config
    # without method/model is an orphan -> drop too (legal for every model).
    for flag, sub in list(containers.items()):
        if flag == "speculative-config":
            ns = sub.get("num_speculative_tokens")
            if ns == 0 or not (sub.get("method") or sub.get("model")):
                containers.pop(flag, None)

    for flag, sub in sorted(containers.items()):
        import json as _json

        cmd.extend([f"--{flag}", _json.dumps(sub, ensure_ascii=False)])

    # 2. Plain search flags
    for name, value in plain:
        flag = f"--{_param_to_flag(name)}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    # 3. Remaining fixed config.toml others flags (dedupe vs search)
    search_flag_names = {f"--{_param_to_flag(n)}" for n, _ in plain}
    search_flag_names.update(f"--{f}" for f in containers)
    for entry in fixed_plain:
        flag = entry.get("flag", "")
        value = entry.get("value")
        if flag in search_flag_names:
            continue
        if value is not None:
            cmd.extend([flag, str(value)])
        else:
            cmd.append(flag)

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Export the best agent optimizer result.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="config.toml 路径（默认 <cwd>/optix/config.toml），用于导出引擎级固定参数",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trials = []
    for path in sorted(run_dir.glob("results.round-*.json")):
        trials.extend(read_json(path).get("trials", []))
    trials = [
        trial
        for trial in trials
        if not trial.get("early_exit") and not (trial.get("performance") or {}).get("early_exit")
    ]
    if not trials:
        raise SystemExit("No completed non-early-exit trial results found.")
    # Hard-constraint preference: prefer the best SLO-compliant trial; only fall
    # back to pure fitness when no trial meets the hard constraints.
    compliant = [t for t in trials if t.get("slo_violated") is False]
    pool = compliant if compliant else trials
    best = min(pool, key=lambda item: item.get("fitness", float("inf")))
    best["slo_compliant"] = bool(compliant)
    write_json(run_dir / "best_result.json", best)
    best["trial_count"] = len(trials)

    round_numbers = sorted(
        {int(p.stem.split("-")[-1]) for p in run_dir.glob("results.round-*.json") if p.stem.split("-")[-1].isdigit()}
    )
    best["round_count"] = round_numbers[-1] if round_numbers else 0

    # 引擎级固定参数：config.toml others 中未被搜索空间覆盖的 flag + 反转验证状态
    context = {}
    context_path = run_dir / "context.json"
    if context_path.exists():
        try:
            context = read_json(context_path)
        except Exception:
            context = {}
    search_names = {p["name"] for p in (context.get("search_space") or {}).get("parameters", [])} | {
        c["name"] for c in (context.get("search_space") or {}).get("constants", [])
    }
    config_fixed_params = []
    config_path = Path(args.config) if args.config else Path.cwd() / "optix" / "config.toml"
    if config_path.exists():
        try:
            import tomllib

            with open(config_path, "rb") as f:
                toml_data = tomllib.load(f)
            engine = (context.get("engine") or "vllm").lower()
            others = str(((toml_data.get(engine) or {}).get("command") or {}).get("others") or "")
            config_fixed_params = [
                entry
                for entry in _parse_others_flags(others)
                if not _flag_covered_by_search(entry["flag"], search_names)
            ]
            for entry in config_fixed_params:
                entry["source"] = f"config.toml [{engine}.command].others"
                entry["verified"] = False
        except Exception:
            config_fixed_params = []

    tested_values = {}
    for t in trials:
        for name, value in (t.get("params") or {}).items():
            tested_values.setdefault(name, set()).add(value)
    fixed_verification = []
    for entry in context.get("engine_fixed_params") or []:
        if not entry.get("flip_required"):
            continue
        search_name = entry.get("search_name")
        verified = False
        if search_name:
            tested = tested_values.get(search_name, set())
            verified = any(str(v) != str(entry.get("value")) for v in tested)
        fixed_verification.append({**entry, "verified": verified})

    # 渲染完整 vllm serve 命令：best 参数 + config.toml 固定 flag
    serve_command = []
    # command 骨架从 config.toml [<engine>.command] 读取
    cmd_toml = {}
    if config_path.exists():
        try:
            import tomllib

            with open(config_path, "rb") as f:
                _toml = tomllib.load(f)
            cmd_toml = (_toml.get(engine) or {}).get("command") or {}
        except Exception:
            cmd_toml = {}
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
            )
            write_json(
                run_dir / "serve_command.json",
                {
                    "argv": serve_command,
                    "command": " ".join(serve_command),
                },
            )
            # shell 可执行版：对含空格的 token 加单引号（JSON 容器值等）
            _shell_cmd = []
            for _tok in serve_command:
                if any(ch.isspace() for ch in _tok):
                    _shell_cmd.append("'" + _tok.replace("'", "'\\''") + "'")
                else:
                    _shell_cmd.append(_tok)
            (run_dir / "serve_command.sh").write_text(
                "#!/usr/bin/env bash\n" + " ".join(_shell_cmd) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            import sys as _sys

            _sys.stderr.write(f"[export_best_config] Warning: failed to render serve command: {exc}\n")

    write_json(
        run_dir / "optimizer_config_handoff.json",
        {
            "source": "optix-assistant",
            "recommended_params": best.get("params", {}),
            "fitness": best.get("fitness"),
            "performance": best.get("performance"),
            "slo_violated": best.get("slo_violated"),
            "slo_compliant": bool(compliant),
            "selection": "slo_compliant_best" if compliant else "fallback_fitness_best",
            "config_fixed_params": config_fixed_params,
            "engine_fixed_params_verification": fixed_verification,
            "serve_command": serve_command,
        },
    )


if __name__ == "__main__":
    main()
