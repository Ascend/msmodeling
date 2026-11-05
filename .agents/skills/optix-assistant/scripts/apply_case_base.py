#!/usr/bin/env python3
"""③ 结构化底座写入器：把审计缺口里的 auto_adopt 项写成 config_writer 调用并执行。

输入 : --coverage base_coverage.json （④ 的产物）
       --rules    case_base_rules.json（分类与携带通道规则）
       --config   <目标 config.toml>（应当是"新 run 的基础配置副本"）
输出 : 命令清单（stdout，逐条可复制）+ base_apply_plan.json

安全设计（对应 skill 的冻结契约与已知坑）：
1. **默认干跑**：不加 --apply 只打印/落盘计划，绝不写文件。
2. **拒绝原地补写冻结配置**：目标路径 == 审计产物里记录的 config 路径时直接报错，
   除非显式 --allow-in-place；推荐用 --copy-from <冻结配置> --config <新路径> 先复制。
3. **永不自动写入**：命中 OWNED_FLAGS（框架占用，写进 others 会 ValueError）、
   orchestration_flags、review_flags 的项一律只出建议，不执行。
4. **P/D 取值不同不猜**：[<engine>.command].others 是共享通道，双侧取值不一致的项转人工。
5. **env 值必须是字符串**：写入后重新解析 TOML 断言 str，失败即报错（整份 PD 配置会被 schema
   静默拒绝并退回 localhost，这是已踩过的坑）。
6. **渲染同源**：直接 importlib 载入 audit_base_coverage.py 复用其渲染函数，
   保证"审计看到的集合"与"写入改动的对象"不会漂移。
7. **flag 只走 --append-others**：它的语义是 token 级并入；--set-others 是整体覆盖，
   会把底座既有的 --quantization / --speculative-config 一起抹掉（实测入口语义差异，勿混用）。
8. **--apply 强制要求 --optix-root**：OWNED_FLAGS 护栏必须现读真源，否则退化为空集合。
9. **调用 config_writer 时 cwd 必须是仓根**：容器默认 cwd=/workspace 时 `import optix` 命中另一个
   checkout（/data1/ssq/msmodeling-x），API 不一致 → check_service_semantics 抛 ImportError。
10. **不用 --check**：实测 PD 配置上 --check 必失败（pydantic Settings 报 cluster/vllm_pd
   "Extra inputs are not permitted"），与写入内容无关；改用 --print，验收靠写后 TOML 回读 + 重跑 ④。
11. **落地前断言 argv 形状**：每条 argv 的 --config 必须正好是目标路径，且不得含非字符串参数
   （历史上曾把已解析的 dict 传给 --config，str(dict) 变成巨型伪路径，config_writer 报
   "File name too long" 而全线中止）。

实测验收（45.24，冻结底座 config.pd-qwen3500n256-20261023.toml）：
adopted 14→24 / missing 14→4（全部 needs_review）/ range_gap 2→0 / 8 个 env 全为 str。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import tempfile
import os
import sys
import time
from pathlib import Path

DEFAULT_PY = sys.executable
DEFAULT_CW = str(Path(__file__).resolve().with_name("config_writer.py"))


# --------------------------------------------------------------------------- 工具


def _load_module(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"ERROR: 找不到 {path}（写入器需要它复用渲染逻辑）")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: 无法载入模块 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shell(argv: list[str]) -> str:
    return shlex.join(argv)


def _num(x):
    f = float(x)
    return int(f) if f == int(f) else f


def _scalar_for_cli(v) -> str:
    """标量转 CLI 文本：bool 用 TOML/JSON 风格的 true/false，其余 str()。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _infer_dtype(v) -> str:
    """按案例值推断 target_field 的 dtype（bool 必须先判，bool 是 int 的子类）。"""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    text = str(v).strip()
    if text.lower() in ("true", "false"):
        return "bool"
    try:
        int(text)
        return "int"
    except ValueError:
        pass
    try:
        float(text)
        return "float"
    except ValueError:
        return "str"


def _upper_snake(flag: str) -> str:
    """--max-model-len -> MAX_MODEL_LEN（target_field 命名：flag 转大写下划线）。"""
    return flag.lstrip("-").replace("-", "_").upper()


def _bounds_args(dtype: str, value) -> list[str]:
    """数值 dtype 必须显式写 min/max：缺省时运行期把域当成 0..100，
    会把 16384 这类值判成越界（check_search_space 报 "value is outside configured range"）。
    案例只给单点参考值时按 min=max=value 冻结该点，后续可由搜索空间显式放宽。
    """
    if dtype not in ("int", "float"):
        return []
    text = _scalar_for_cli(value)
    return [f"min={text}", f"max={text}"]


def _others_text(doc: dict, engine: str) -> str:
    """按 engine 优先、再回退 vllm/vllm_pd 找底座 others。"""
    for sect in (engine, "vllm", "vllm_pd"):
        node = doc.get(sect) or {}
        if isinstance(node, dict) and isinstance(node.get("command"), dict):
            cand = node["command"].get("others")
            if cand:
                return str(cand)
    return ""


def _dot_get(doc: dict, path: list[str]):
    cur = doc
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _find_tf_block(cfg: dict, name: str) -> tuple[str, dict] | None:
    """返回 (所在段名, target_field 块)；段名用于 --target-field engine=。"""
    for sect in ("vllm_pd", "vllm"):
        node = cfg.get(sect) or {}
        for blk in node.get("target_field") or []:
            if str(blk.get("name") or "") == name:
                return sect, blk
    return None


# --------------------------------------------------------------------------- 计划


def build_plan(audit, rules, coverage, cfg, eff, truth, engine, py, cw, cfg_path, subkey_map=None):
    owned_flags = truth["owned_flags"]
    orch = set(rules.get("orchestration_flags") or [])
    review = set(rules.get("review_flags") or [])
    pol = rules.get("adopt_policy") or {}
    env_pol = pol.get("env") or {}
    subkey_map = subkey_map or {}
    role_split_flags = set(rules.get("role_split_flags") or [])
    has_pd = isinstance(cfg.get("vllm_pd"), dict)
    env_path_tpl = env_pol.get("path_template_pd") if has_pd else env_pol.get("path_template_single")
    role_sides = env_pol.get("role_sides") or ["prefill", "decode"]

    actions: list[dict] = []
    manual: list[dict] = []

    def act(kind, key, category, argv, reason, verify, roles=None):
        actions.append(
            {
                "id": len(actions) + 1,
                "kind": kind,
                "key": key,
                "category": category,
                "roles": roles or [],
                "argv": argv,
                "shell": _shell(argv),
                "reason": reason,
                "verify": verify,
            }
        )

    def man(kind, key, category, why, suggested=""):
        manual.append({"kind": kind, "key": key, "category": category, "why": why, "suggested": suggested})

    def base_argv(*extra: str) -> list[str]:
        # 注意 1：这里必须用「目标配置的路径字符串」，不能传 cfg（那是已解析的 dict，
        # str(dict) 会让 config_writer 收到一个巨大的伪路径 -> "File name too long"）。
        # 注意 2：**不能用 --check**。实测（见 s243_check_probe）：PD 配置上 --check 必失败——
        #   pydantic Settings 报 "cluster / vllm_pd: Extra inputs are not permitted"，
        #   即 --check 走的是单机 Settings 模式，不认 PD 的 cluster+vllm_pd 段。
        #   该失败与写入内容无关（原样 --check-only 也报），故改用 --print（写完打印相关段），
        #   真实验收靠写完后的 TOML 回读自检 + 重跑 ④ 审计。
        return [py, cw, "--config", str(cfg_path), *extra, "--print"]

    # ---- 1. flags ----
    for row in coverage.get("missing", []):
        if row.get("kind") != "flag":
            continue
        flag = row["key"] if row["key"].startswith("--") else "--" + row["key"].lstrip("-")
        roles = row.get("roles") or []
        if has_pd and flag in role_split_flags:
            continue  # The role-specific pass handles these without changing shared others.
        if row.get("category") != "auto_adopt":
            man(
                "flag",
                flag,
                row.get("category", "?"),
                f"category={row.get('category')}：需按方法卡/互斥关系裁决",
                suggested=_shell(base_argv("--append-others", f"{engine}={flag}")),
            )
            continue
        if flag in owned_flags or flag in orch or flag in review:
            man(
                "flag",
                flag,
                "policy_denied",
                "命中框架占用/编排/需裁决集合，策略禁止自动写入 others",
                suggested=f"# {flag} 需人工确认（框架占用写进 others 会直接 ValueError）",
            )
            continue
        values = {r: v for r, v in (row.get("case_value") or {}).items()}
        distinct = {v for v in values.values()}
        if has_pd and set(roles) != set(role_sides):
            man("flag", flag, "needs_review", "单侧参数不能写入共享 others，否则会影响另一角色")
            continue
        if len(distinct) > 1:
            man(
                "flag",
                flag,
                "role_divergent",
                "P/D 取值不同，[<engine>.command].others 为共享通道，无法按角色区分",
                suggested=f"# {flag} P={values.get('prefill')} D={values.get('decode')} 需人工决定",
            )
            continue
        value = next(iter(distinct)) if distinct else None
        text = flag if value in (None, "") else f"{flag} {value}"
        # 必须用 --append-others（token 级并入），不能用 --set-others：后者是整体覆盖语义，
        # 会把底座里既有的 --quantization / --speculative-config 等一起抹掉。
        act(
            "flag",
            flag,
            "auto_adopt",
            base_argv("--append-others", f"{engine}={text}"),
            f"案例底座补齐（token 级幂等并入 others，不覆盖既有 flag）：{text}",
            {"type": "others_token", "token": flag, "expect_value": None if value in (None, "") else str(value)},
            roles,
        )

    # ---- 1b. 角色分侧 flag（role_split_flags）----
    # 这类 flag 即使案例双侧同值也要建 role 后缀 target_field：一旦只写进共享 others，
    # P/D 就再也不能分侧，且 ④ 审计会按 flag 名把它记成 adopted（D 侧差异被吞掉）。
    # 注意：这里扫 items（全集），因为这些 flag 往往已经是 adopted（在共享 others 里）。
    for row in coverage.get("items") or []:
        if row.get("kind") != "flag":
            continue
        flag = row["key"] if row["key"].startswith("--") else "--" + row["key"].lstrip("-")
        if flag not in role_split_flags or flag in owned_flags or flag in orch:
            continue
        for role, raw in (row.get("case_value") or {}).items():
            if role not in role_sides or raw in (None, ""):
                continue
            name = f"{_upper_snake(flag)}_{role}"
            if _find_tf_block(cfg, name):
                continue  # 幂等
            dtype = _infer_dtype(raw)
            argv = [
                py,
                cw,
                "--config",
                str(cfg_path),
                "--target-field",
                "engine=vllm_pd",
                f"name={name}",
                "config_position=run",
                f"dtype={dtype}",
                f"value={_scalar_for_cli(raw)}",
                *_bounds_args(dtype, raw),
                "--print",
            ]
            act(
                "flag_role_split",
                f"{flag}:{role}",
                "role_split",
                argv,
                f"{flag} ∈ role_split_flags → 为 {role} 建 {name}={raw}，使其可按角色区分；"
                "共享 others 里的同名字面量保留不动（plain flag 去重后搜索侧胜）",
                {"type": "role_target_field", "name": name, "expect": raw, "dtype": dtype},
                [role],
            )

    # ---- 2. 容器子键 ----
    for row in coverage.get("missing", []):
        if row.get("kind") != "container_subkey":
            continue
        key = row["key"]
        if ":" not in key:
            man("container_subkey", key, row.get("category", "?"), "无法解析容器:子键", "")
            continue
        container, subkey = key.rsplit(":", 1)
        container = container if container.startswith("--") else "--" + container.lstrip("-")
        roles = [r for r in (row.get("roles") or []) if r in role_sides]
        role_values = row.get("role_values") or {}
        # 单角色子键取该角色的值（扁平 union 里可能是另一角色的值）
        if len(set(roles)) == 1 and role_values.get(roles[0]) is not None:
            value = role_values[roles[0]]
        else:
            value = row.get("case_value")
        key_no_dash = container.lstrip("-")
        structured = base_argv("--set", f"{engine}.command.others.{key_no_dash}={json.dumps({subkey: value})}")
        # 字段名必须是 JSON_SUBKEY_MAP 里的名字（pd.py 靠它反查容器/JSON key）：
        # 例如案例子键 enforce_eager 对应字段名 speculative_enforce_eager（一义一名）。
        base_name = row.get("field_name") or subkey
        if container in owned_flags:
            man("container_subkey", key, "policy_denied", "容器为框架占用，不可写入 others", "")
            continue
        if base_name not in subkey_map:
            # 不在 JSON_SUBKEY_MAP：渲染层无法把该子键作为角色/搜索维度承载，只能整容器静态下发
            man(
                "container_subkey",
                key,
                "needs_review",
                f"子键 {subkey}（字段名 {base_name}）不在 JSON_SUBKEY_MAP：不能按角色承载；"
                f"若 {container} 只能走共享 others，双侧都会生效",
                suggested=_shell(structured),
            )
            continue
        if len(set(roles)) == 1:
            # 单角色需要的容器子键 → role 后缀 target_field（渲染成容器的单个 JSON key，只落在该角色）
            role = roles[0]
            name = f"{base_name}_{role}"
            if _find_tf_block(cfg, name):
                continue  # 幂等：已在位，交给 ④ 复检判 adopted
            dtype = _infer_dtype(value)
            argv = [
                py,
                cw,
                "--config",
                str(cfg_path),
                "--target-field",
                "engine=vllm_pd",
                f"name={name}",
                "config_position=run",
                f"dtype={dtype}",
                f"value={_scalar_for_cli(value)}",
                *_bounds_args(dtype, value),
                "--print",
            ]
            act(
                "container_subkey_role",
                key,
                "auto_adopt",
                argv,
                f"仅 {role} 侧需要的容器子键（字段名 {base_name} ∈ JSON_SUBKEY_MAP）→ 写入 {name}，"
                f"渲染为 {container} '{{\"{subkey}\": ...}}' 只出现在 {role}",
                {"type": "role_target_field", "name": name, "expect": value, "dtype": dtype},
                [role],
            )
            continue
        if container in review:
            man(
                "container_subkey",
                key,
                "container_needs_review",
                f"容器 {container} 在 review_flags 内：整容器的取舍需先裁决",
                suggested=_shell(structured),
            )
            continue
        # P/D 都要（或未标注）→ 共享通道可表达
        act(
            "container_subkey",
            key,
            "auto_adopt",
            structured,
            f"结构化合并容器子键：{container} += {json.dumps({subkey: value})}",
            {"type": "container_subkey", "container": container, "subkey": subkey, "expect": value},
            roles or both_roles(role_sides),
        )

    # ---- 3. env ----
    for row in coverage.get("missing", []):
        if row.get("kind") != "env":
            continue
        var = row["key"]
        if row.get("category") != "auto_adopt":
            man("env", var, row.get("category", "?"), "需人工裁决", "")
            continue
        sides = [s for s in (row.get("roles") or []) if s in role_sides]
        values = row.get("case_value") or {}
        for side in sides or role_sides:
            v = values.get(side)
            if v is None:
                continue
            path = env_path_tpl.format(role=side, VAR=var, engine=engine)
            path_list = path.split(".")
            act(
                "env",
                var,
                "auto_adopt",
                base_argv("--set", f"{path}={json.dumps(str(v))}"),
                f"案例 env 落位到 {path} = {v}（JSON 引号包裹保证 TOML 字符串）",
                {"type": "env", "path": path_list, "expect": str(v)},
                [side],
            )

    # ---- 4. 区间缺口 ----
    for row in coverage.get("range_gaps", []):
        flag = row["key"]
        roles = row.get("roles") or []
        role = roles[0] if roles else ""
        rng = eff["ranges"].get((flag, role)) or eff["ranges"].get((flag, ""))
        if not rng:
            man(
                "range_gap",
                flag,
                "range_mismatch",
                "找不到对应的 target_field（可能该维度由 env 承载或命名不含角色后缀）",
                f"# 案例值 {row.get('case_value')}：{row.get('reason')}",
            )
            continue
        name = rng["name"]
        found = _find_tf_block(cfg, name)
        if not found:
            man(
                "range_gap",
                flag,
                "range_mismatch",
                "配置里找不到同名 target_field 块（可能与审计读到的不是同一份配置）",
                "",
            )
            continue
        tf_sect, blk = found
        if str(blk.get("dtype") or "") == "enum":
            man(
                "range_gap",
                flag,
                "enum_manual",
                "枚举型维度：是否扩展 choices 需人工决定（自动扩展会改变搜索语义）",
                f"# {name} 现枚举 {blk.get('dtype_param')}，案例值 {row.get('case_value')} 不在其中",
            )
            continue
        try:
            case_val = _num(row.get("case_value"))
        except (TypeError, ValueError):
            man("range_gap", flag, "range_mismatch", f"案例值无法数值化：{row.get('case_value')}", "")
            continue
        old_min, old_max = blk.get("min"), blk.get("max")
        workflow_path = Path(cfg_path).parent / "pd_workflow.json"
        fixed_params = (
            json.loads(workflow_path.read_text(encoding="utf-8")).get("fixed_params", {})
            if workflow_path.exists()
            else {}
        )
        if name in fixed_params or blk.get("constant") is not None or (old_min is not None and old_min == old_max):
            man("range_gap", flag, "needs_review", "案例值与固定约束冲突，不能自动扩展", "")
            continue
        new_min = case_val if isinstance(old_min, (int, float)) and case_val < old_min else old_min
        new_max = case_val if isinstance(old_max, (int, float)) and case_val > old_max else old_max
        if new_min == old_min and new_max == old_max:
            continue
        if not isinstance(new_min, (int, float)) and not isinstance(new_max, (int, float)):
            man("range_gap", flag, "range_mismatch", "该 target_field 既无 min 也无 max（非数值区间），需人工处理", "")
            continue
        fields = [
            # 用块实际所在段，避免在 [vllm] 与 [vllm_pd] 并存时写错分区
            f"engine={tf_sect}",
            f"name={name}",
            f"config_position={blk.get('config_position', 'run')}",
        ]
        if "dtype" in blk:
            fields.append(f"dtype={blk['dtype']}")
        if "value" in blk:
            fields.append(f"value={_scalar_for_cli(blk['value'])}")
        if isinstance(new_min, (int, float)):
            fields.append(f"min={new_min}")
        if isinstance(new_max, (int, float)):
            fields.append(f"max={new_max}")
        if blk.get("dtype_param"):
            fields.append("dtype_param=" + json.dumps(blk["dtype_param"], ensure_ascii=False))
        argv = [py, cw, "--config", str(cfg_path), "--target-field", *fields, "--print"]
        act(
            "range_gap",
            name,
            "widen_range",
            argv,
            f"把案例取值 {case_val} 并入区间：[{old_min}, {old_max}] -> [{new_min}, {new_max}]（value 保持不变）",
            {"type": "target_field", "name": name, "min": new_min, "max": new_max},
            [role] if role else [],
        )

    return actions, manual


def both_roles(role_sides: list[str]) -> list[str]:
    return list(role_sides)


# --------------------------------------------------------------------------- 执行与校验


def run_actions(actions: list[dict], cfg_path: Path, dry: bool, cwd: str = "", backup: bool = True) -> None:
    if dry:
        return
    backup_path = None
    if backup:
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_path = cfg_path.with_name(cfg_path.name + f".bak-{ts}")
        shutil.copy2(cfg_path, backup_path)
        print(f"[backup] {backup_path}")
    for a in actions:
        print(f"[run {a['id']:>2}] {a['shell']}")
        # cwd 必须是 optix 仓根：容器默认 cwd=/workspace 会让 `import optix` 命中
        # 另一个 checkout（实测解析到 /data1/ssq/msmodeling-x），其 API 与本仓不一致，
        # 直接在 check_service_semantics 抛 ImportError。
        proc = subprocess.run(a["argv"], capture_output=True, text=True, cwd=cwd or None)
        a["returncode"] = proc.returncode
        a["stdout"] = (proc.stdout or "").strip()[-500:]
        if proc.returncode != 0:
            print(proc.stdout or "", proc.stderr or "")
            detail = f"备份在 {backup_path}" if backup_path else "目标配置未发布"
            raise SystemExit(
                f"ERROR: 第 {a['id']} 条写入失败（exit={proc.returncode}）。已停止后续写入；"
                f"{detail}，请人工核对后再继续"
            )


def verify(audit, cfg_path: Path, actions: list[dict], engine: str) -> list[str]:
    """写后自检：env 必须是 str、others token 必须在位、target_field 区间已放宽。"""
    doc = audit._load_toml(cfg_path)
    problems: list[str] = []
    for a in actions:
        v = a["verify"]
        if v["type"] == "env":
            got = _dot_get(doc, v["path"])
            if not isinstance(got, str):
                problems.append(
                    f"[env] {'.'.join(v['path'])} 期望字符串 {v['expect']!r}，实际 {got!r}（类型 {type(got).__name__}）"
                    " -> 值可能被写成非字符串，PD schema 会静默拒绝；请改用 --set 传 JSON 引号包裹的值"
                )
            elif got != v["expect"]:
                problems.append(f"[env] {'.'.join(v['path'])} 期望 {v['expect']!r}，实际 {got!r}")
        elif v["type"] == "others_token":
            toks = shlex.split(_others_text(doc, engine)) if _others_text(doc, engine) else []
            if v["token"] not in toks:
                problems.append(f"[others] 未找到 {v['token']}（当前 others: {_others_text(doc, engine)[:120]}）")
            elif v.get("expect_value") and v["expect_value"] not in toks:
                problems.append(f"[others] {v['token']} 的值 {v['expect_value']} 未在位")
        elif v["type"] == "container_subkey":
            toks = shlex.split(_others_text(doc, engine)) if _others_text(doc, engine) else []
            idx = toks.index(v["container"]) if v["container"] in toks else -1
            if idx < 0 or idx + 1 >= len(toks):
                problems.append(f"[container] {v['container']} 不在 others 中")
            else:
                try:
                    payload = json.loads(toks[idx + 1])
                except ValueError:
                    payload = {}
                if payload.get(v["subkey"]) != v["expect"]:
                    problems.append(
                        f"[container] {v['container']}.{v['subkey']} 期望 {v['expect']!r}，实际 {payload.get(v['subkey'])!r}"
                    )
        elif v["type"] == "target_field":
            found = _find_tf_block(doc, v["name"])
            blk = found[1] if found else {}
            if blk.get("min") != v["min"] or blk.get("max") != v["max"]:
                problems.append(
                    f"[target_field] {v['name']} 区间期望 [{v['min']}, {v['max']}]，实际 [{blk.get('min')}, {blk.get('max')}]"
                )
        elif v["type"] == "role_target_field":
            found = _find_tf_block(doc, v["name"])
            if not found:
                problems.append(f"[target_field] 未创建 {v['name']}（role 后缀字段）")
                continue
            blk = found[1]
            got = blk.get("value")
            if str(got).lower() != str(v["expect"]).lower():
                problems.append(f"[target_field] {v['name']} 期望 {v['expect']!r}，实际 {got!r}")
            if str(blk.get("dtype") or "") != v["dtype"]:
                problems.append(f"[target_field] {v['name']} dtype 期望 {v['dtype']!r}，实际 {blk.get('dtype')!r}")
            if str(blk.get("config_position") or "run") != "run":
                problems.append(
                    f"[target_field] {v['name']} config_position 期望 run，实际 {blk.get('config_position')!r}"
                )
    return problems


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="按审计缺口生成/执行 config_writer 写入（默认干跑）")
    ap.add_argument("--coverage", required=True, help="base_coverage.json")
    ap.add_argument("--config", required=True, help="目标 config.toml（建议为新 run 的副本）")
    ap.add_argument("--copy-from", default="", help="先把该文件复制为 --config（推荐：冻结配置 -> 新路径）")
    ap.add_argument("--force-copy", action="store_true", help="目标已存在时仍覆盖复制")
    ap.add_argument("--rules", default="", help="规则表，默认与脚本同目录的 case_base_rules.json")
    ap.add_argument("--audit-script", default="", help="audit_base_coverage.py，默认同目录")
    ap.add_argument("--optix-root", default="", help="仓根；区间缺口处理需要（读真源与 target_field 命名）")
    ap.add_argument("--engine", default="vllm", help="others 所在引擎段（默认 vllm）")
    ap.add_argument("--python", default=DEFAULT_PY, help="跑 config_writer 的 python")
    ap.add_argument("--config-writer", default=DEFAULT_CW, help="config_writer.py 路径")
    ap.add_argument("--out", default="", help="计划产物，默认 <coverage 同目录>/base_apply_plan.json")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认干跑）")
    ap.add_argument(
        "--allow-in-place", action="store_true", help="允许直接改审计产物里记录的那份配置（违反冻结契约，慎用）"
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    rules_path = Path(args.rules) if args.rules else here / "case_base_rules.json"
    audit_path = Path(args.audit_script) if args.audit_script else here / "audit_base_coverage.py"
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    coverage = json.loads(Path(args.coverage).read_text(encoding="utf-8"))
    audit = _load_module(audit_path, "audit_base_coverage")

    cfg_path = Path(args.config)
    frozen = Path(coverage.get("config", ""))
    src = Path(args.copy_from) if args.copy_from else cfg_path
    if not src.is_file():
        print(f"ERROR: 源配置不存在: {src}")
        return 2
    if args.copy_from and cfg_path.exists() and not args.force_copy:
        print(f"ERROR: 目标已存在 {cfg_path}；如要覆盖请加 --force-copy")
        return 2
    if frozen and cfg_path.resolve() == frozen.resolve() and not args.allow_in_place:
        print(f"ERROR: 目标是审计记录的那份冻结配置（{frozen}）。")
        return 2
    if args.copy_from and cfg_path.resolve() == src.resolve():
        print("ERROR: 目标与源配置相同")
        return 2

    # 真源 + 现值渲染（与审计同源）
    if args.apply and not args.optix_root:
        print("ERROR: --apply 必须同时提供 --optix-root。")
        print("       原因：OWNED_FLAGS 安全护栏与区间缺口处理都依赖从仓里现读真源；")
        print("       缺失时护栏会退化为空集合，可能把框架占用的 flag 写进 others（pd_adapter 会直接 ValueError）。")
        return 2
    if not args.optix_root:
        print("WARNING: 未提供 --optix-root，区间缺口将无法自动处理（其余项不受影响）")
    truth = {"owned_flags": set()}
    eff = {"ranges": {}}
    subkey_map: dict = {}
    if args.optix_root:
        root = Path(args.optix_root)
        truth["owned_flags"] = audit._read_literal_set(root / "optix/optimizer/agentic/pd.py", "PD_OWNED_FLAGS")
        subkey_map = audit._read_json_subkey_map(root / "optix/serve_plan.py")
        benchmark_only = audit._read_literal_set(root / "optix/serve_plan.py", "BENCHMARK_ONLY_FIELDS")
        topology_only = audit._read_literal_set(root / "optix/optimizer/agentic/pd.py", "TOPOLOGY_FIELDS")
        cfg_doc = audit._load_toml(src)
        eff = audit.effective_from_toml(cfg_doc, args.engine, subkey_map, benchmark_only, topology_only)
    cfg_doc = audit._load_toml(src)

    actions, manual = build_plan(
        audit,
        rules,
        coverage,
        cfg_doc,
        eff,
        truth,
        args.engine,
        args.python,
        args.config_writer,
        cfg_path,
        subkey_map=subkey_map,
    )

    out = Path(args.out) if args.out else Path(args.coverage).resolve().parent / "base_apply_plan.json"
    plan = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(cfg_path),
        "source_coverage": str(args.coverage),
        "frozen_config": str(frozen),
        "engine": args.engine,
        "mode": "apply" if args.apply else "dry_run",
        "summary": {"auto_actions": len(actions), "manual": len(manual)},
        "actions": actions,
        "manual": manual,
    }

    print("=" * 100)
    print(f"目标配置 : {cfg_path}")
    print(f"审计产物 : {args.coverage}   模式: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"计划     : 自动写入 {len(actions)} 条 / 人工裁决 {len(manual)} 条")
    print("=" * 100)
    for a in actions:
        print(f"\n[{a['id']:>2}] ({a['kind']}/{a['category']}) {a['key']}  roles={a['roles'] or '-'}")
        print(f"     why : {a['reason']}")
        print(f"     cmd : {a['shell']}")
    if manual:
        print("\n【人工裁决 / 策略不自动写入】")
        for m in manual:
            print(f"  - ({m['kind']}/{m['category']}) {m['key']}\n      why: {m['why']}")
            if m.get("suggested"):
                print(f"      cmd: {m['suggested']}")

    if out.resolve() in {src.resolve(), cfg_path.resolve()} or (
        out.exists() and any(path.exists() and os.path.samefile(out, path) for path in (src, cfg_path))
    ):
        print("ERROR: 计划产物路径不能与源配置或目标配置相同")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nplan written: {out}")

    if not args.apply:
        print("\n（干跑结束，未修改任何文件。确认后加 --apply 执行）")
        return 0

    # 前置断言：每条 argv 的 --config 必须正好是目标路径。
    # （曾踩坑：build_plan 误传已解析的 dict，str(dict) 变成巨型伪路径，
    #   config_writer 报 "File name too long"，写入在第 1 条即全线中止。）
    bad = []
    for a in actions:
        av = a["argv"]
        if "--config" not in av or av[av.index("--config") + 1] != str(cfg_path):
            bad.append(a["id"])
        if any(t.startswith("{") or t.startswith("[") for t in av):
            bad.append(a["id"])
    if bad:
        print(f"ERROR: 以下动作的 argv 里 --config 不是目标路径或混入了非字符串参数: {sorted(set(bad))}")
        print("       拒绝执行（这属于脚本自身 bug，不要用 --apply 绕过）")
        return 2

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".case-base-", suffix=".toml", dir=cfg_path.parent)
    os.close(fd)
    staged = Path(tmp_name)
    try:
        shutil.copy2(src, staged)
        staged_actions = []
        for action in actions:
            item = dict(action)
            item["argv"] = list(action["argv"])
            item["argv"][item["argv"].index("--config") + 1] = str(staged)
            item["shell"] = _shell(item["argv"])
            staged_actions.append(item)
        run_actions(staged_actions, staged, dry=False, cwd=args.optix_root, backup=False)
        problems = verify(audit, staged, staged_actions, args.engine)
        if problems:
            print("\n【写后自检失败】")
            for problem in problems:
                print("  -", problem)
            return 1
        if cfg_path.exists():
            backup_path = cfg_path.with_name(cfg_path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
            shutil.copy2(cfg_path, backup_path)
        os.replace(staged, cfg_path)
    finally:
        staged.unlink(missing_ok=True)
    print("\n【写后自检通过】env 均为字符串、others token 在位、区间已放宽")
    print("请重跑 ④ 复检：")
    print(
        f"  {args.python} {audit_path} --config {cfg_path} --case-json <case_base_raw.json> "
        f"--optix-root {args.optix_root or '<仓根>'} --out <base_coverage.json>"
    )
    print("期望：missing 数从当前值降到仅剩 needs_review 项（容器/互斥类）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
