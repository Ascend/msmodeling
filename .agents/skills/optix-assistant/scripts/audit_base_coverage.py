#!/usr/bin/env python3
"""④ 审计：案例底座集合 vs TOML 实际生效集合，输出缺口清单（只读）。

真源全部现读现取，不复制常量：
  - optix/optimizer/agentic/pd.py      -> PD_OWNED_FLAGS（框架占用；写进 others 会直接 ValueError）
  - optix/optimizer/agentic/pd.py      -> OWNED_ENV、TOPOLOGY_FIELDS
  - optix/serve_plan.py                -> JSON_SUBKEY_MAP / BENCHMARK_ONLY_FIELDS / TOPOLOGY_ONLY_FIELDS

TOML 侧"实际生效集合"的构造与运行时同规则：
  - [<engine>.command].others          -> 顶层 flag（shlex token 化）
  - [[vllm_pd.target_field]] run 型     -> JSON_SUBKEY_MAP ? 容器子键 : --name
  - [[vllm_pd.target_field]] env 型     -> 环境变量名
  - [vllm_pd.env*] / [nodes.env] / [proxy.env] / [<engine>.env] -> 环境变量名
  - bool False 的 flag 型字段不生效（与 _field_to_cli_flag 一致）

退出码：存在未被解释的缺口 -> 1（可直接接进 preflight 门禁）
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from copy import deepcopy
from pathlib import Path

#: 编排/代理决定、注定不该抄进 others 的 flag（OWNED_FLAGS 之外的补充，需案例逐条确认）
ORCHESTRATION_FLAGS = {
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--api-server-count",
    "--distributed-executor-backend",
    "--prefiller-hosts",
    "--prefiller-ports",
    "--decoder-hosts",
    "--decoder-ports",
}
#: 案例里出现但必须先做方法卡/互斥裁决、不允许自动采用的 flag
#: 注意：--additional-config 已从本集合移除——它的子键现在有结构化承载通道
#: （子键 ∈ JSON_SUBKEY_MAP：单角色走 role 后缀 target_field、双侧走共享容器合并），
#: 不再需要"整容器先裁决"；未登记的子键仍会在容器子键分支被判 needs_review。
#: 本集合必须与 case_base_rules.json 的 review_flags 保持一致。
REVIEW_FLAGS = {
    "--enforce-eager",
    "--compilation-config",
    "--async-scheduling",
    "--speculative-config",
}
#: 工具链/节点特定 env：值含占位符或由环境决定，不搬
ENV_VALUE_PLACEHOLDER = re.compile(r"\$\{|xxxx|xxx\b|nic_name|local_ip|path/to", re.IGNORECASE)
ENV_TOOLCHAIN = {"LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "PYTHONHASHSEED", "PROMETHEUS_MULTIPROC_DIR"}


def _load_toml(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        import tomllib

        return tomllib.loads(raw.decode("utf-8"))
    except ImportError:
        pass
    try:
        import tomli

        return tomli.loads(raw.decode("utf-8"))
    except ImportError:
        print("WARNING: 无 tomllib/tomli，使用内置精简解析器（仅支持本仓 config.toml 子集）")
        return _mini_toml(raw.decode("utf-8"))


def _mini_toml(text: str) -> dict:
    """极简 TOML：section / [[array of table]] / key = 标量。仅作兜底。"""
    root: dict = {}
    cur: dict = root
    for raw in text.splitlines():
        line = raw.split("#")[0].strip() if not raw.strip().startswith('others = "') else raw.strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            path = line[2:-2].split(".")
            cur = root
            for p in path[:-1]:
                cur = cur.setdefault(p, {})
            arr = cur.setdefault(path[-1], [])
            cur = {}
            arr.append(cur)
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = root
            for p in line[1:-1].split("."):
                cur = cur.setdefault(p, {})
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        elif v in ("true", "false"):
            v = v == "true"
        else:
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
        cur[k] = v
    return root


def _read_literal_set(py_file: Path, var: str) -> set[str]:
    """从 optix 源码里读 `VAR = { "a", "b" }` 字面量集合（现读真源，不复制常量）。"""
    if not py_file.exists():
        raise SystemExit(f"ERROR: 找不到真源文件 {py_file}")
    text = py_file.read_text(encoding="utf-8")
    m = re.search(rf"^{var}\s*(?::[^=]+)?=\s*\{{(.*?)\}}", text, re.S | re.M)
    if not m:
        raise SystemExit(f"ERROR: {py_file} 中未找到 {var} 字面量集合（源码结构可能已变）")
    items = re.findall(r'"([^"]+)"', m.group(1))
    if not items:
        raise SystemExit(f"ERROR: {var} 解析为空（源码结构可能已变）")
    return set(items)


def _read_json_subkey_map(py_file: Path) -> dict[str, tuple[str, str]]:
    text = py_file.read_text(encoding="utf-8")
    m = re.search(r"^JSON_SUBKEY_MAP\s*=\s*\{(.*?)^\}", text, re.S | re.M)
    if not m:
        raise SystemExit(f"ERROR: {py_file} 中未找到 JSON_SUBKEY_MAP")
    out: dict[str, tuple[str, str]] = {}
    for name, container, key in re.findall(r'"([^"]+)"\s*:\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', m.group(1)):
        out[name] = (container, key)
    if not out:
        raise SystemExit("ERROR: JSON_SUBKEY_MAP 解析为空")
    return out


def _norm_flag(name: str) -> str:
    n = name.strip()
    return n if n.startswith("--") else "--" + n.lstrip("-")


def _param_to_cli(name: str, subkey_map: dict[str, tuple[str, str]]) -> str:
    if name in subkey_map:
        return "--" + subkey_map[name][0]
    return "--" + name.lower().replace("_", "-")


def _split_role(name: str) -> tuple[str, str | None]:
    for role in ("prefill", "decode"):
        if name.endswith("_" + role):
            return name[: -len("_" + role)], role
    return name, None


def _map_name_for(subkey_map: dict[str, tuple[str, str]], container: str, key: str) -> str | None:
    """(容器 flag, 容器内 JSON key) → JSON_SUBKEY_MAP 的字段名（渲染契约要求的那个名字）。

    必须反向查表而不是拿 JSON key 直接当字段名：两者并不总是同名，例如
    (speculative-config, enforce_eager) 的字段名是 `speculative_enforce_eager`
    （见 serve_plan.FLAG_NAME_MAP 注释「一义一名不混用」）。用 JSON key 直接判会把它误判成未登记。
    """
    want = (container.lstrip("-"), key)
    for name, pair in subkey_map.items():
        if (pair[0].lstrip("-"), pair[1]) == want:
            return name
    return None


def effective_from_toml(cfg: dict, engine: str, subkey_map: dict, benchmark_only: set, topology_only: set) -> dict:
    """构造"实际会生效"的 flag/env/容器子键集合。"""
    flags: dict[str, str] = {}
    containers: dict[str, dict] = {}
    env: dict[str, str] = {}
    search_subkeys: dict[str, set] = {}
    notes: list[str] = []

    others = ""
    for sect in (engine, "vllm_pd", "vllm"):
        node = cfg.get(sect) or {}
        if isinstance(node, dict):
            val = (node.get("command") or {}).get("others") if isinstance(node.get("command"), dict) else None
            if val:
                others = val
                notes.append(f"底座 others 取自 [{sect}.command]")
                break
        val = node.get("vllm_others") if isinstance(node, dict) else None
        if val:
            others = val
            notes.append(f"底座 others 取自 [{sect}].vllm_others")
            break
    if others:
        try:
            toks = shlex.split(others)
        except ValueError as exc:
            raise SystemExit(f"ERROR: others 无法 shlex 解析: {exc}")
        i = 0
        while i < len(toks):
            if toks[i].startswith("--"):
                value = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
                flags[toks[i]] = value
                if value.lstrip().startswith("{"):
                    try:
                        containers[toks[i]] = json.loads(value)
                    except ValueError:
                        containers[toks[i]] = {"__unparsed__": value}
                i += 2 if value else 1
            else:
                i += 1

    # 搜索维度（target_field）
    tf_blocks = []
    for sect in ("vllm_pd", "vllm"):
        node = cfg.get(sect) or {}
        tfs = node.get("target_field") if isinstance(node, dict) else None
        if tfs:
            tf_blocks.extend(tfs)
    ranges: dict[tuple, dict] = {}
    subkey_ranges: dict[tuple, dict] = {}
    for f in tf_blocks or []:
        name = str(f.get("name") or "")
        if not name:
            continue
        base, role = _split_role(name)
        if base.upper() in benchmark_only | topology_only:
            continue
        pos = str(f.get("config_position") or "run")
        value = f.get("value")
        if pos == "env":
            env[base] = f"target_field(role={role or '-'})"
            continue
        if isinstance(value, bool) and value is False:
            notes.append(f"{base}: bool False → 不产生 flag（与 _field_to_cli_flag 一致）")
            continue
        flag = _param_to_cli(base, subkey_map)
        flags.setdefault(flag, f"target_field(role={role or '-'})")
        entry = {
            "name": name,
            "role": role,
            "min": f.get("min"),
            "max": f.get("max"),
            "value": value,
            "dtype": f.get("dtype"),
            "choices": f.get("dtype_param") if f.get("dtype") == "enum" else None,
        }
        if base in subkey_map:
            container, json_key = subkey_map[base]
            search_subkeys.setdefault(flag, set()).add(json_key)
            subkey_ranges[(container, json_key)] = entry
        else:
            # 角色后缀字段必须按 (flag, role) 分别记录，否则 P/D 区间会互相覆盖
            ranges[(flag, role or "")] = entry

    # env 块
    def walk_env(node, path: str):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if isinstance(v, dict):
                walk_env(v, f"{path}.{k}" if path else k)
            elif isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k)):
                env.setdefault(str(k), path or "env")

    for sect, node in cfg.items():
        if not isinstance(node, dict):
            continue
        if sect == "vllm_pd" or sect.startswith("vllm"):
            for k, v in node.items():
                if k == "env" or k.startswith("env_") or k == "command":
                    if k == "command" and isinstance(v, dict):
                        walk_env(v.get("env"), f"{sect}.command.env")
                    else:
                        walk_env(v, f"{sect}.{k}")
        if sect == "cluster":
            for h in node.get("hosts") or []:
                walk_env(h.get("env"), "cluster.hosts.env")
    # nodes / proxy 的 env（vllm_pd.nodes 是数组）
    pd = cfg.get("vllm_pd") or {}
    for key in ("nodes",):
        for n in pd.get(key) or []:
            walk_env(n.get("env"), f"vllm_pd.{key}.env")
    walk_env((pd.get("proxy") or {}).get("env"), "vllm_pd.proxy.env")
    walk_env((pd.get("nodes") or {}).get("env") if isinstance(pd.get("nodes"), dict) else None, "vllm_pd.nodes.env")

    return {
        "flags": flags,
        "containers": containers,
        "env": env,
        "search_subkeys": search_subkeys,
        "ranges": ranges,
        "subkey_ranges": subkey_ranges,
        "notes": notes,
    }


def _range_gap(rng: dict, value) -> str | None:
    """案例取值是否落在当前搜索区间/枚举内；返回差异描述或 None。"""
    if value is None:
        return None
    choices = rng.get("choices")
    if choices:
        v = str(value)
        if v not in [str(c) for c in choices]:
            return f"案例取值 {v} 不在枚举 {choices} 内"
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = rng.get("min"), rng.get("max")
    if isinstance(lo, (int, float)) and v < float(lo):
        return f"案例取值 {value} 低于搜索下界 {lo}（当前 value={rng.get('value')}）"
    if isinstance(hi, (int, float)) and v > float(hi):
        return f"案例取值 {value} 高于搜索上界 {hi}（当前 value={rng.get('value')}）"
    return None


def _pd_role_effective(cfg, engine, subkey_map, benchmark_only, topology_only):
    """Keep static and per-role values separate; role fields override common values."""
    static = deepcopy(cfg)
    for section in ("vllm", "vllm_pd"):
        node = static.get(section) or {}
        node.pop("target_field", None)
        for key in list(node):
            if key.startswith("env_") or key in {"nodes", "proxy"}:
                node.pop(key, None)
    base = effective_from_toml(static, engine, subkey_map, benchmark_only, topology_only)
    result = {
        role: {"flags": dict(base["flags"]), "containers": deepcopy(base["containers"]), "env": {}}
        for role in ("prefill", "decode")
    }
    for role in result:
        for section in ("vllm", "vllm_pd"):
            node = cfg.get(section) or {}
            for env in (
                (node.get("command") or {}).get("env") or {},
                node.get("env") or {},
                node.get("env_" + role) or {},
            ):
                if isinstance(env, dict):
                    result[role]["env"].update({k: v for k, v in env.items() if not isinstance(v, dict)})
            nested = node.get("env") or {}
            if isinstance(nested, dict) and isinstance(nested.get(role), dict):
                result[role]["env"].update(nested[role])
        nodes = [n for n in (cfg.get("vllm_pd") or {}).get("nodes", []) if n.get("role") == role]
        for key in {k for n in nodes for k in (n.get("env") or {})}:
            values = [n.get("env", {}).get(key, result[role]["env"].get(key)) for n in nodes]
            result[role]["env"][key] = values[0] if all(v == values[0] for v in values) else object()
        fields = []
        for section in ("vllm", "vllm_pd"):
            fields.extend((cfg.get(section) or {}).get("target_field") or [])
        # Common fields first, then role-specific fields regardless of source order.
        for field in sorted(fields, key=lambda f: bool(_split_role(str(f.get("name") or ""))[1])):
            name, side = _split_role(str(field.get("name") or ""))
            if not name or side not in (None, "", role) or name.upper() in benchmark_only | topology_only:
                continue
            value = field.get("value")
            if str(field.get("config_position") or "run") == "env":
                result[role]["env"][name] = value
            elif name in subkey_map:
                container, key = subkey_map[name]
                result[role]["containers"].setdefault("--" + container, {})[key] = value
            else:
                flag = _param_to_cli(name, subkey_map)
                if value is False or value is None:
                    result[role]["flags"].pop(flag, None)
                else:
                    result[role]["flags"][flag] = "" if value is True else value
    return result


def _same_case_value(actual, expected):
    if isinstance(actual, bool):
        return str(actual).lower() == str(expected).lower()
    if expected is None:
        return actual in (None, "", True)
    return str(actual) == str(expected)


def _numeric_or_choice(value, rng):
    if rng.get("choices"):
        return str(value) in {str(choice) for choice in rng["choices"]}
    try:
        number = float(value)
        return float(rng["min"]) <= number <= float(rng["max"])
    except (TypeError, ValueError, OverflowError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="案例底座 vs TOML 生效集合 差集审计（只读）")
    ap.add_argument("--config", required=True, help="config.toml 绝对路径")
    ap.add_argument("--case-json", required=True, help="case_base_extract.py 的输出")
    ap.add_argument("--optix-root", required=True, help="msmodeling-x-pd-agent-search 仓根")
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--out", default="base_coverage.json")
    ap.add_argument("--all-roles", action="store_true", help="不区分角色，按并集比对（默认即并集，保留开关）")
    args = ap.parse_args()

    root = Path(args.optix_root)
    owned_flags = _read_literal_set(root / "optix/optimizer/agentic/pd.py", "PD_OWNED_FLAGS")
    owned_env = _read_literal_set(root / "optix/optimizer/agentic/pd.py", "OWNED_ENV")
    topology_fields = _read_literal_set(root / "optix/optimizer/agentic/pd.py", "TOPOLOGY_FIELDS")
    subkey_map = _read_json_subkey_map(root / "optix/serve_plan.py")
    benchmark_only = _read_literal_set(root / "optix/serve_plan.py", "BENCHMARK_ONLY_FIELDS")

    cfg = _load_toml(Path(args.config))
    case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    eff = effective_from_toml(cfg, args.engine, subkey_map, benchmark_only, topology_fields)

    case_flags: dict = case["union"]["flags"]
    case_env: dict = case["union"]["env"]
    case_containers: dict = case["union"]["containers"]
    # 子键的角色维度来自 case_base_extract 的 union.container_subkeys（containers 是扁平合并，
    # 会丢掉角色信息）；PD 的角色差异只出现在子键级，例如 D 侧独有的 recompute_scheduler_enable。
    case_subkeys = (case.get("union") or {}).get("container_subkeys") or {}
    # 已解析出真实子键的容器：这些容器 flag 的差异由同名 container_subkey 行逐子键承载，
    # flag 级不应再报缺口（否则同一件事被记两遍，且整容器字面量写共享 others 会丢角色差异）。
    parsed_containers = {
        cont
        for cont, subs in case_containers.items()
        if subs and "__unparsed__" not in subs and (case_subkeys.get(cont) or {})
    }

    rows: list[dict] = []

    def add(kind, key, roles, case_value, status, category, reason, action="", **extra):
        row = {
            "kind": kind,
            "key": key,
            "roles": roles,
            "case_value": case_value,
            "status": status,
            "category": category,
            "reason": reason,
            "action": action,
        }
        row.update(extra)
        rows.append(row)

    pd_roles = (
        _pd_role_effective(cfg, args.engine, subkey_map, benchmark_only, topology_fields)
        if isinstance(cfg.get("vllm_pd"), dict)
        else None
    )

    # ---- flags ----
    for raw, meta in case_flags.items():
        flag = _norm_flag(raw)
        roles = sorted(set(meta["roles"]))
        if flag in owned_flags:
            add(
                "flag",
                flag,
                roles,
                meta["values"],
                "skip",
                "framework_owned",
                "框架占用（pd_adapter.OWNED_FLAGS），写进 others 会直接 ValueError",
                "不抄",
            )
            continue
        if flag in ORCHESTRATION_FLAGS:
            add("flag", flag, roles, meta["values"], "skip", "orchestration", "编排/代理决定，非服务参数", "不抄")
            continue
        if flag in parsed_containers:
            # 容器 flag：案例 JSON 已解析到子键级 → 差异由 container_subkey 行承载。
            # 整容器写共享 others 会把 P/D 抹成同值（--additional-config 正是这么丢的 D 侧开关）。
            add(
                "flag",
                flag,
                roles,
                meta["values"],
                "skip",
                "subkey_level",
                "JSON 容器 flag：角色差异由同名 container_subkey 行逐子键承载（整容器字面量进共享 others 会丢角色差异）",
                "见同名 container_subkey 行",
            )
            continue
        if pd_roles and roles and set(roles) <= set(pd_roles):
            if (
                set(roles) == set(pd_roles)
                and all(flag not in pd_roles[role]["flags"] for role in roles)
                and len({str(meta["values"].get(role)) for role in roles}) == 1
            ):
                add(
                    "flag",
                    flag,
                    roles,
                    meta["values"],
                    "missing",
                    "needs_review" if flag in REVIEW_FLAGS else "auto_adopt",
                    "两侧均缺失",
                    "按共同配置裁决",
                )
                continue
            for role in roles:
                expected = meta["values"].get(role)
                actual = pd_roles[role]["flags"].get(flag)
                if flag not in pd_roles[role]["flags"]:
                    add(
                        "flag",
                        flag,
                        [role],
                        {role: expected},
                        "missing",
                        "needs_review" if flag in REVIEW_FLAGS else "auto_adopt",
                        "该角色未配置",
                        "按角色裁决",
                    )
                elif not _same_case_value(actual, expected):
                    rng = eff["ranges"].get((flag, role)) or eff["ranges"].get((flag, ""))
                    gap = _range_gap(rng, expected) if rng else None
                    if gap:
                        add("flag", flag, [role], expected, "range_gap", "range_mismatch", gap, "评估固定约束")
                    elif (
                        rng
                        and (
                            rng.get("choices")
                            or isinstance(rng.get("min"), (int, float))
                            and isinstance(rng.get("max"), (int, float))
                        )
                        and _numeric_or_choice(expected, rng)
                    ):
                        add(
                            "flag",
                            flag,
                            [role],
                            {role: expected},
                            "adopted",
                            "range_covered",
                            "该角色搜索范围覆盖案例值",
                            "-",
                        )
                    else:
                        add(
                            "flag",
                            flag,
                            [role],
                            {role: expected},
                            "missing",
                            "needs_review",
                            "该角色的实际值与案例不同",
                            "人工裁决",
                        )
                else:
                    add("flag", flag, [role], {role: expected}, "adopted", "present", "该角色实际值匹配", "-")
            continue
        if flag in eff["flags"]:
            add("flag", flag, roles, meta["values"], "adopted", "present", f"已生效（{eff['flags'][flag]}）", "-")
            if flag not in eff["search_subkeys"]:  # 容器型 flag 的区间由子键检查负责
                for r_, v_ in meta["values"].items():
                    rng = eff["ranges"].get((flag, r_)) or eff["ranges"].get((flag, ""))
                    if not rng:
                        continue
                    gap = _range_gap(rng, v_)
                    if gap:
                        add(
                            "flag",
                            flag,
                            [r_],
                            v_,
                            "range_gap",
                            "range_mismatch",
                            gap,
                            "案例取值未被搜索区间覆盖：评估是否放宽 min/max（走 --target-field）",
                        )
            continue
        cat = "needs_review" if flag in REVIEW_FLAGS else "auto_adopt"
        add(
            "flag",
            flag,
            roles,
            meta["values"],
            "missing",
            cat,
            "案例存在、当前 TOML 无对应底座/维度",
            "先做方法卡+互斥裁决，再经 config_writer 结构化写入底座"
            if cat == "needs_review"
            else "可经 config_writer --set '<engine>.command.others.<flag>=<JSON>' 写入底座",
        )

    # ---- 容器子键 ----
    for cont, subs in case_containers.items():
        if cont in owned_flags:
            add("container", cont, [], subs, "skip", "framework_owned", "框架占用", "不抄")
            continue
        live = set((eff["containers"].get(cont) or {}).keys()) | set(eff["search_subkeys"].get(cont) or set())
        for sk, sv in subs.items():
            if sk == "__unparsed__":
                add(
                    "container_subkey",
                    f"{cont}.<unparsed>",
                    [],
                    sv,
                    "missing",
                    "needs_review",
                    "案例容器无法解析为 JSON",
                    "人工核对",
                )
                continue
            meta = (case_subkeys.get(cont) or {}).get(sk) or {}
            sub_roles = sorted(set(meta.get("roles") or []))
            sub_values = {r: v for r, v in (meta.get("values") or {}).items()}
            # 渲染契约用的是 JSON_SUBKEY_MAP 的字段名（与容器内 JSON key 未必同名）
            field_name = _map_name_for(subkey_map, cont, sk)
            extra = {"role_values": sub_values}
            if field_name:
                extra["field_name"] = field_name
            if pd_roles and sub_roles and set(sub_roles) <= set(pd_roles):
                for role in sub_roles:
                    values = pd_roles[role]["containers"].get(cont, {})
                    expected = sub_values.get(role)
                    extra_role = {"role_values": {role: expected}}
                    if field_name:
                        extra_role["field_name"] = field_name
                    if sk not in values:
                        add(
                            "container_subkey",
                            f"{cont}:{sk}",
                            [role],
                            expected,
                            "missing",
                            "auto_adopt" if field_name else "needs_review",
                            "该角色未配置子键",
                            "按角色裁决",
                            **extra_role,
                        )
                    elif not _same_case_value(values[sk], expected):
                        add(
                            "container_subkey",
                            f"{cont}:{sk}",
                            [role],
                            expected,
                            "missing",
                            "needs_review",
                            "该角色子键值与案例不同",
                            "人工裁决",
                            **extra_role,
                        )
                    else:
                        add(
                            "container_subkey",
                            f"{cont}:{sk}",
                            [role],
                            expected,
                            "adopted",
                            "present",
                            "该角色子键值匹配",
                            "-",
                            **extra_role,
                        )
                continue
            if sk in live:
                where = "搜索维度" if sk in (eff["search_subkeys"].get(cont) or set()) else "静态底座"
                add(
                    "container_subkey",
                    f"{cont}:{sk}",
                    sub_roles,
                    sv,
                    "adopted",
                    "present",
                    f"已生效（{where}）",
                    "-",
                    **extra,
                )
                rng = eff["subkey_ranges"].get((cont, sk))
                gap = _range_gap(rng, sv) if rng else None
                if gap:
                    add(
                        "container_subkey",
                        f"{cont}:{sk}",
                        sub_roles,
                        sv,
                        "range_gap",
                        "range_mismatch",
                        gap,
                        "案例取值未被搜索维度覆盖：评估是否调整枚举/区间",
                        **extra,
                    )
            else:
                # 子键在 JSON_SUBKEY_MAP 内 → 有承载通道（单角色走 role 后缀 target_field、
                # 双侧走共享 others 容器合并），属可自动装配；不在表内只能整容器静态下发。
                registered = field_name is not None
                if registered:
                    cat = "auto_adopt"
                    act = (
                        "单角色 → role 后缀 target_field（写入器自动）；双侧 → 共享容器合并"
                        if sub_roles
                        else "经 config_writer 合并进容器底座（写入器自动）"
                    )
                else:
                    cat = "needs_review"
                    act = (
                        "子键不在 JSON_SUBKEY_MAP：不能作为角色维度，"
                        "需人工改走整容器 str 枚举 target_field 或 shared others 容器合并"
                    )
                add(
                    "container_subkey",
                    f"{cont}:{sk}",
                    sub_roles,
                    sv,
                    "missing",
                    cat,
                    "案例容器的子键在当前底座与搜索维度里都不存在",
                    act,
                    **extra,
                )

    # ---- env ----
    for key, meta in case_env.items():
        roles = sorted(set(meta["roles"]))
        vals = list(meta["values"].values())
        if key in owned_env:
            add(
                "env", key, roles, meta["values"], "skip", "framework_owned", "框架注入（agentic.pd.OWNED_ENV）", "不抄"
            )
            continue
        if key in ENV_TOOLCHAIN:
            add("env", key, roles, meta["values"], "skip", "toolchain", "工具链/环境特定", "不抄")
            continue
        if any(isinstance(v, str) and ENV_VALUE_PLACEHOLDER.search(v) for v in vals):
            add(
                "env",
                key,
                roles,
                meta["values"],
                "skip",
                "node_placeholder",
                "值为节点占位符（ip/nic/路径），随部署变化",
                "不抄",
            )
            continue
        if pd_roles and roles and set(roles) <= set(pd_roles):
            for role in roles:
                expected = meta["values"].get(role)
                actual = pd_roles[role]["env"]
                if key not in actual:
                    add(
                        "env",
                        key,
                        [role],
                        {role: expected},
                        "missing",
                        "auto_adopt",
                        "该角色未配置环境变量",
                        "按角色写入",
                    )
                elif not _same_case_value(actual[key], expected):
                    add(
                        "env",
                        key,
                        [role],
                        {role: expected},
                        "missing",
                        "needs_review",
                        "该角色环境变量值与案例不同",
                        "人工裁决",
                    )
                else:
                    add("env", key, [role], {role: expected}, "adopted", "present", "该角色环境变量匹配", "-")
            continue
        if key in eff["env"]:
            add("env", key, roles, meta["values"], "adopted", "present", f"已生效（{eff['env'][key]}）", "-")
        else:
            add(
                "env",
                key,
                roles,
                meta["values"],
                "missing",
                "auto_adopt",
                "案例 env 在当前 TOML 未落位",
                "写入 [vllm_pd.env_prefill]/[vllm_pd.env_decode]/[nodes.env]（值必须是字符串）",
            )

    missing = [r for r in rows if r["status"] == "missing"]
    range_gaps = [r for r in rows if r["status"] == "range_gap"]
    result = {
        "config": args.config,
        "case": case["case_file"],
        "truth_sources": {
            "owned_flags": sorted(owned_flags),
            "owned_env": sorted(owned_env),
            "json_subkey_map": {k: list(v) for k, v in subkey_map.items()},
        },
        "effective": {
            "flags": eff["flags"],
            "env": eff["env"],
            "containers": eff["containers"],
            "search_subkeys": {k: sorted(v) for k, v in eff["search_subkeys"].items()},
            "notes": eff["notes"],
        },
        "counts": {
            "case_flags": len(case_flags),
            "case_env": len(case_env),
            "effective_flags": len(eff["flags"]),
            "effective_env": len(eff["env"]),
            "adopted": len([r for r in rows if r["status"] == "adopted"]),
            "skipped": len([r for r in rows if r["status"] == "skip"]),
            "missing": len(missing),
            "range_gap": len(range_gaps),
        },
        "items": rows,
        "missing": missing,
        "range_gaps": range_gaps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    c = result["counts"]
    print("=" * 78)
    print(f"案例 : {case['case_file']}")
    print(f"配置 : {args.config}")
    print(f"真源 : OWNED_FLAGS={len(owned_flags)} OWNED_ENV={len(owned_env)} JSON_SUBKEY_MAP={len(subkey_map)}")
    print(
        f"统计 : 案例 flag={c['case_flags']} env={c['case_env']} | "
        f"生效 flag={c['effective_flags']} env={c['effective_env']} | "
        f"已覆盖={c['adopted']} 合理跳过={c['skipped']} 缺口={c['missing']} 区间不覆盖={c['range_gap']}"
    )
    print("=" * 78)
    if missing:
        print("\n【缺口清单】（status=missing）")
        print(f"{'kind':16} {'key':46} {'cat':13} {'role':16} 案例值")
        print("-" * 130)
        for r in sorted(missing, key=lambda x: (x["kind"], x["key"])):
            cv = json.dumps(r["case_value"], ensure_ascii=False)
            print(f"{r['kind']:16} {r['key']:46} {r['category']:13} {','.join(r['roles']):16} {cv[:44]}")
    if range_gaps:
        print("\n【案例取值未被搜索区间覆盖】")
        for r in range_gaps:
            print(f"  {r['key']:44} role={','.join(r['roles']):9} {r['reason']}")
    print("\n【合理跳过】")
    for r in rows:
        if r["status"] == "skip":
            print(f"  {r['kind']:16} {r['key']:46} {r['category']:18} {r['reason'][:52]}")
    print(f"\nwritten: {args.out}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
