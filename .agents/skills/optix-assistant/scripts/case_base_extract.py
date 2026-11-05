#!/usr/bin/env python3
"""Extract a case's `## 启动脚本` into structured flag/env/container sets (read-only).

Input : a case-knowledge .md (e.g. cases/official/0.23.0/XXX__pd-1p1d.md)
Output: case_base_raw.json -- flags/env/containers grouped by role (prefill/decode)

Design constraints:
- Read-only. Writes nothing inside the repository (--out defaults to cwd).
- Parse failures must be explicit, never a silent empty set (a silent empty
  result would make the downstream base-coverage audit falsely green).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

#: 标题行（### Prefill Node / #### 5.4.1 Prefill Node ...）
HEADING_RE = re.compile(r"^\s{0,3}#{2,6}\s*(.+?)\s*$")
#: 任何以 ``` 开头的行都视为 fence（官方案例里存在 "```." 这种带点的畸形收尾）
FENCE_RE = re.compile(r"^\s*```(.*)$")
EXPORT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
ROLE_RE = re.compile(r"(prefill|prefiller|decode|decoder)", re.IGNORECASE)
#: 容器型 flag（值是 JSON dict，需按子键比对）
CONTAINER_FLAGS = (
    "--speculative-config",
    "--compilation-config",
    "--additional-config",
    "--kv-transfer-config",
)


def parse_case(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 只处理 `## 启动脚本` 小节：从该标题到下一个同级 `## ` 标题为止
    start = end = None
    for i, ln in enumerate(lines):
        m = HEADING_RE.match(ln)
        if m and "启动脚本" in m.group(1):
            start = i + 1
            for j in range(start, len(lines)):
                if re.match(r"^##\s+\S", lines[j]):
                    end = j
                    break
            end = end if end is not None else len(lines)
            break
    if start is None:
        raise SystemExit(f"ERROR: {path} 中未找到 '## 启动脚本' 小节")

    blocks: list[dict] = []
    cur_heading = ""
    i = start
    while i < end:
        ln = lines[i]
        m = HEADING_RE.match(ln)
        if m:
            cur_heading = m.group(1)
            i += 1
            continue
        f = FENCE_RE.match(ln)
        if f:
            lang = f.group(1).strip()
            j = i + 1
            body: list[str] = []
            closed = False
            while j < end:
                if FENCE_RE.match(lines[j]):
                    closed = True
                    break
                body.append(lines[j])
                j += 1
            if not closed:
                print(f"WARNING: {path} 第 {i + 1} 行 code fence 在小节内未闭合，已跳过该块")
                break
            block = _parse_block(body, lang, cur_heading)
            if block:
                blocks.append(block)
            i = j + 1
            continue
        i += 1

    if not blocks:
        raise SystemExit(f"ERROR: {path} 的启动脚本小节里没有解析到任何包含 vllm serve 的代码块")
    return {"case_file": str(path), "blocks": blocks, "union": _union(blocks)}


def _parse_block(body: list[str], lang: str, heading: str) -> dict | None:
    env: dict[str, str] = {}
    cmd_lines: list[str] = []
    collecting = False
    quote: str | None = None
    for raw in body:
        stripped = raw.strip()
        if not collecting:
            if stripped.startswith("vllm serve"):
                collecting = True
                quote = None
            else:
                m = EXPORT_RE.match(raw)
                if m and not stripped.startswith("#"):
                    env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
                continue
        cmd_lines.append(stripped)
        # 续行判定：行尾反斜杠，或引号尚未闭合（JSON 容器跨行时不带反斜杠）
        for ch in stripped:
            if quote is None:
                if ch in ("'", '"'):
                    quote = ch
            elif ch == quote:
                quote = None
        if not stripped.endswith("\\") and quote is None:
            collecting = False
    if not cmd_lines:
        return None

    joined = " ".join(line.rstrip("\\").strip() for line in cmd_lines)
    try:
        tokens = shlex.split(joined)
    except ValueError as exc:
        raise SystemExit(f"ERROR: 无法解析 case 启动命令（shlex）: {exc}")

    model = tokens[2] if len(tokens) > 2 and tokens[0] == "vllm" and tokens[1] == "serve" else ""
    flags: dict[str, str | None] = {}
    containers: dict[str, dict] = {}
    idx = 1
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.startswith("--"):
            value = None
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                value = tokens[idx + 1]
                idx += 1
            flags[tok] = value
            if tok in CONTAINER_FLAGS and value and value.lstrip().startswith("{"):
                try:
                    containers[tok] = json.loads(value)
                except ValueError:
                    containers[tok] = {"__unparsed__": value}
        idx += 1

    role = None
    m = ROLE_RE.search(heading or "")
    if m:
        role = "prefill" if m.group(1).lower().startswith("pre") else "decode"
    if role is None:
        kv = containers.get("--kv-transfer-config", {})
        kv_role = str(kv.get("kv_role", ""))
        if "producer" in kv_role:
            role = "prefill"
        elif "consumer" in kv_role:
            role = "decode"
    return {
        "role": role or "unknown",
        "heading": heading,
        "lang": lang,
        "model": model,
        "flags": flags,
        "containers": containers,
        "env": env,
    }


def _union(blocks: list[dict]) -> dict:
    flags: dict[str, dict] = {}
    env: dict[str, dict] = {}
    containers: dict[str, dict] = {}
    # 容器子键的角色维度（container_subkeys[容器][子键] = {values:{role:值}, roles:[...]}）：
    # containers 是扁平合并（保留给旧调用），会丢掉"哪个角色带了这个子键"，
    # 而 PD 角色差异恰恰只在子键级出现（如 D 侧独有的 recompute_scheduler_enable）。
    container_subkeys: dict[str, dict] = {}
    for b in blocks:
        for k, v in b["flags"].items():
            flags.setdefault(k, {"values": {}, "roles": []})
            flags[k]["values"][b["role"]] = v
            flags[k]["roles"].append(b["role"])
        for k, v in b["containers"].items():
            containers.setdefault(k, {}).update(v)
            for sub_key, sub_val in v.items():
                slot = container_subkeys.setdefault(k, {}).setdefault(sub_key, {"values": {}, "roles": []})
                slot["values"][b["role"]] = sub_val
                slot["roles"].append(b["role"])
        for k, v in b["env"].items():
            env.setdefault(k, {"values": {}, "roles": []})
            env[k]["values"][b["role"]] = v
            env[k]["roles"].append(b["role"])
    return {"flags": flags, "env": env, "containers": containers, "container_subkeys": container_subkeys}


def main() -> int:
    ap = argparse.ArgumentParser(description="从 case-knowledge 启动脚本抽取 flag/env 集合（只读）")
    ap.add_argument("--case", required=True, help="case .md 绝对路径")
    ap.add_argument("--out", default="case_base_raw.json")
    args = ap.parse_args()

    path = Path(args.case)
    if not path.exists():
        print(f"ERROR: case 文件不存在: {path}")
        return 2
    result = parse_case(path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    u = result["union"]
    print(f"case      : {path}")
    print(f"blocks    : {[(b['role'], b['heading'][:34]) for b in result['blocks']]}")
    print(f"flags     : {len(u['flags'])}  env: {len(u['env'])}  containers: {sorted(u['containers'])}")
    print(f"written   : {out}")
    for role in ("prefill", "decode"):
        rflags = [k for k, v in u["flags"].items() if role in v["roles"]]
        renv = [k for k, v in u["env"].items() if role in v["roles"]]
        print(f"  [{role}] flags={len(rflags)} env={len(renv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
