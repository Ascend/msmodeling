#!/usr/bin/env python3
"""R0 底座裁决硬门禁：案例底座里未被消解的 missing/needs_review 项必须闭环，否则禁止开跑。

为什么需要它：④ 审计早就把 `--additional-config:recompute_scheduler_enable` 判成
missing/needs_review 并给出建议命令，清单也写进了产物与 README——但"人工裁决"这一步
从来没人执行，缺口就一路被继承到每一轮候选与最终 best（实测：13 份 P/D 启动脚本里
`--additional-config` 出现 0 次）。门禁把"建议"变成"阻断"。

闭环的两条合法路径（对应 case_base_rules.json）：
  1. 改造成可自动装配：子键登记进 `optix/serve_plan.py` 的 `JSON_SUBKEY_MAP`（单角色
     走 role 后缀 target_field、双侧走共享容器合并），或把 flag 加进 `role_split_flags`；
     重跑流水线后该行变 adopted；
  2. 在 decisions 文件里逐条裁决：verdict(adopted|not_adopted) + owner + reason + decided_at。

用法：
  # R0 装配后（④ 复检产物）
  python3 check_base_review.py --audit <AUDIT_DIR>/out/base_coverage.after.json --out-dir <AUDIT_DIR>/out
  # 开跑前（run 目录里放了 base_coverage.after.json / base_review_decisions.json）
  python3 check_base_review.py --run-dir <run dir>
  # 显式放行（打印 WARNING 并写进报告）
  python3 check_base_review.py --run-dir <run dir> --allow-unreviewed

退出码：0 = 闭环（或显式放行）；3 = 存在未闭环项（禁止开跑）；2 = 用法问题

被 run_pd_round.py 通过 evaluate() 复用（同一个判断，两处入口，避免口径漂移）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

VALID_VERDICTS = {"adopted", "not_adopted"}
REQUIRED_DECISION_FIELDS = ("key", "verdict", "owner", "reason", "decided_at")

DECISIONS_TEMPLATE = {
    "_README": [
        "R0 案例底座裁决记录。base_coverage 里每一条 missing 行都必须在这里留下裁决，",
        "否则 check_base_review.py 会以退出码 3 阻断开跑。",
        "verdict=adopted 表示已落到底座（请写明写在哪里），not_adopted 表示明确不采用（写明理由）。",
        "owner 必须是人或角色名，reason 必须写出依据（方法卡/实测/互斥关系），decided_at 用 YYYY-MM-DD。",
    ],
    "decisions": [],
}


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decision_index(decisions_path, out_dir: Path) -> tuple[dict, list[str]]:
    """返回 {(kind,key): decision, ('',key): decision} 与非法条目说明。缺文件时写出模板。"""
    path = Path(decisions_path) if decisions_path else None
    if path is None or not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        template = out_dir / "base_review_decisions.json"
        if not template.exists():
            template.write_text(json.dumps(DECISIONS_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
        return {}, []
    data = load_json(path)
    entries = data.get("decisions") if isinstance(data, dict) else data
    index: dict = {}
    problems: list[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            problems.append(f"条目不是对象: {entry!r}")
            continue
        missing = [f for f in REQUIRED_DECISION_FIELDS if not str(entry.get(f) or "").strip()]
        if missing:
            problems.append(f"{entry.get('key') or entry!r}: 缺字段 {missing}（不算闭环）")
            continue
        if str(entry["verdict"]).strip() not in VALID_VERDICTS:
            problems.append(
                f"{entry['key']}: verdict 必须是 {sorted(VALID_VERDICTS)}，收到 {entry['verdict']!r}（不算闭环）"
            )
            continue
        key = str(entry["key"]).strip()
        index[("", key)] = entry
        kind = str(entry.get("kind") or "").strip()
        if kind:
            index[(kind, key)] = entry
    return index, problems


def unresolved_rows(coverage: dict, index: dict) -> list[dict]:
    out = []
    for row in coverage.get("missing") or []:
        key = str(row.get("key") or "")
        kind = str(row.get("kind") or "")
        if ("", key) in index or (kind, key) in index:
            continue
        out.append(row)
    return out


def evaluate(audit_path, decisions_path=None) -> dict:
    """只读评估：返回 {coverage_present, coverage_path, decisions_path, coverage, rows, problems}。"""
    path = Path(audit_path)
    result = {
        "coverage_present": False,
        "coverage_path": path,
        "decisions_path": Path(decisions_path) if decisions_path else None,
        "coverage": {},
        "rows": [],
        "problems": [],
    }
    if not path.exists():
        return result
    coverage = load_json(path)
    index, problems = decision_index(result["decisions_path"], path.resolve().parent)
    result.update(coverage_present=True, coverage=coverage, rows=unresolved_rows(coverage, index), problems=problems)
    return result


def run_dir_artifacts(run_dir) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    return run_dir / "base_coverage.after.json", run_dir / "base_review_decisions.json"


def render_report(result: dict, cfg_path: str = "", allowed: bool = False) -> str:
    coverage = result.get("coverage") or {}
    rows = result["rows"]
    problems = result["problems"]
    title = "# R0 底座裁决未闭环（本次已显式放行 --allow-unreviewed）" if allowed else "# R0 底座裁决未闭环 —— 禁止开跑"
    lines = [
        title,
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 审计产物: {result['coverage_path']}",
        f"- 配置: {cfg_path or coverage.get('config', '?')}",
        f"- 审计计数: {json.dumps(coverage.get('counts') or {}, ensure_ascii=False)}",
        f"- 未闭环项: {len(rows)}",
        "",
    ]
    if problems:
        lines += ["## 裁决记录里的问题（这些条目不算闭环）", ""] + [f"- {p}" for p in problems] + [""]
    lines += [
        "## 未闭环清单",
        "",
        "| kind | key | category | roles | 案例值 | 原因 | 建议动作 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cv = json.dumps(r.get("case_value"), ensure_ascii=False)
        lines.append(
            f"| {r.get('kind')} | `{r.get('key')}` | {r.get('category')} | "
            f"{','.join(r.get('roles') or []) or '-'} | `{cv[:60]}` | {r.get('reason')} | {r.get('action')} |"
        )
    lines += [
        "",
        "## 怎么闭环",
        "",
        "1. **改造成可自动装配**：子键登记进 `optix/serve_plan.py` 的 `JSON_SUBKEY_MAP`（单角色 → role 后缀",
        "   target_field；双侧 → 共享容器合并），或把 flag 加进 `case_base_rules.json` 的 `role_split_flags`；",
        "   然后重跑 `run_casebase_pipeline.sh`（`APPLY=1`），该行会变 adopted。",
        "2. **人工裁决**：在同目录 `base_review_decisions.json` 里逐条写",
        "   `{\"kind\":..., \"key\":..., \"verdict\":\"adopted|not_adopted\", \"owner\":\"...\", \"reason\":\"依据...\",",
        "   \"decided_at\":\"YYYY-MM-DD\"}`，再重跑本门禁。",
        "3. 无论哪条路，R0 结束时都要把 `base_coverage.after.json` 与 `base_review_decisions.json` 复制进 run",
        "   目录，否则 `run_pd_round.py` 的底座门禁看不到闭环证据（缺产物时会告警）。",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict, out_dir, allowed: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = str((result.get("coverage") or {}).get("config") or "")
    report = out_dir / "BASE_REVIEW_REQUIRED.md"
    report.write_text(render_report(result, cfg, allowed), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="R0 底座裁决硬门禁（只读审计产物，阻断未闭环开跑）")
    ap.add_argument("--audit", default="", help="base_coverage*.json（④ 产物）")
    ap.add_argument("--run-dir", default="", help="run 目录：读 <run>/base_coverage.after.json 与裁决文件")
    ap.add_argument("--decisions", default="", help="裁决记录 json（缺省与 --audit 同目录）")
    ap.add_argument("--out-dir", default="", help="报告/模板输出目录（缺省 = 审计产物同目录）")
    ap.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="显式放行未闭环项（打印 WARNING 并写进报告，退出码 0；默认禁止开跑）",
    )
    args = ap.parse_args()

    if args.run_dir:
        audit, decisions = run_dir_artifacts(args.run_dir)
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.run_dir)
        if args.decisions:
            decisions = Path(args.decisions)
    elif args.audit:
        audit = Path(args.audit)
        out_dir = Path(args.out_dir) if args.out_dir else audit.resolve().parent
        decisions = Path(args.decisions) if args.decisions else out_dir / "base_review_decisions.json"
    else:
        print("ERROR: 需要 --audit 或 --run-dir")
        return 2

    result = evaluate(audit, decisions)

    if not result["coverage_present"]:
        print(f"审计产物: {audit}（不存在）")
        if args.allow_unreviewed:
            print("WARNING: 无 R0 案例底座产物，--allow-unreviewed 放行。")
            return 0
        print("ERROR: 该 run 没做过 R0 案例底座装配/裁决，按门禁规则禁止开跑。")
        print(
            "       先跑: APPLY=1 SRC=<冻结 config.toml> NEW=<新底座 config.toml> "
            "bash <SKILL_ROOT>/scripts/run_casebase_pipeline.sh"
        )
        print("       确有理由于非案例装配的 run，请显式加 --allow-unreviewed。")
        return 3

    print(f"审计产物 : {audit}")
    print(f"裁决记录 : {decisions if Path(decisions).exists() else '（缺省，按空处理；模板已生成）'}")
    print(f"计数     : {json.dumps((result['coverage'].get('counts') or {}), ensure_ascii=False)}")
    print(f"未闭环   : {len(result['rows'])}")
    for r in result["rows"]:
        print(f"  - [{r.get('kind')}] {r.get('key')}  category={r.get('category')} roles={r.get('roles') or '-'}")
    for p in result["problems"]:
        print(f"  ! 裁决记录问题: {p}")

    if result["rows"] or result["problems"]:
        report = write_report(result, out_dir, args.allow_unreviewed)
        print(f"报告: {report}")
        if not args.allow_unreviewed:
            print("\n门禁结果: 阻断（退出码 3）。未闭环项必须按报告里的两条路径之一处理。")
            return 3
        print("\nWARNING: --allow-unreviewed 显式放行，未闭环项被带进 run（裁决与责任人须可追溯）。")
        return 0
    print("\n门禁结果: 通过（案例底座的 missing/needs_review 项已全部闭环）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
