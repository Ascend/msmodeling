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

"""Ownership: script. Dimension coverage matrix for one agent-mode run.

Answers the question the round-count based convergence check cannot answer:
*are the movable dimensions actually explored?*

For every movable ``context.search_space.parameters`` entry the script counts how
many executed trials moved it off its default, how many distinct values were
tried, and whether the run declared it ``retired`` (with a reason) in
``dimension_index.json``.  A dimension that was never moved and never retired is
an ``uncovered`` dimension: claiming convergence with uncovered dimensions is a
process gap, not a result.

Read-only by default; ``--init-index`` writes a skeleton ``dimension_index.json``
that the agent fills in (reason is mandatory for ``retired``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# run 目录发现逻辑与 metrics_gate 等脚本共享，抽到 optix_assistant_common
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optix_assistant_common import find_run_dir  # noqa: E402

INDEX_FILE = "dimension_index.json"
#: 单维度 trial 上限的缺省值：context 里没有 budgets.max_trials 时才使用。
DEFAULT_MAX_TRIALS_PER_DIM = 3
#: 自动分配时的下限：每个维度至少要留 2 个点才能看出趋势。
MIN_TRIALS_PER_DIM = 2


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def movable_parameters(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parameters the search can actually move (no ``constant``, min < max)."""
    space = context.get("search_space") or {}
    result: List[Dict[str, Any]] = []
    for param in space.get("parameters") or []:
        name = str(param.get("name") or "").strip()
        if not name:
            continue
        if param.get("constant") is not None:
            continue
        low, high = param.get("min"), param.get("max")
        try:
            if low is not None and high is not None and float(low) == float(high):
                continue
        except (TypeError, ValueError):
            pass
        result.append(param)
    return result


def load_trials(run_dir: Path) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("results.round-*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        rnd = int(payload.get("round") or path.stem.split("-")[-1])
        for trial in payload.get("trials") or []:
            if trial.get("error"):
                continue
            trials.append(
                {"round": rnd, "params": trial.get("params") or {}, "candidate_id": trial.get("candidate_id")}
            )
    return trials


def same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return left == right


def build_matrix(
    context: Dict[str, Any],
    trials: Sequence[Dict[str, Any]],
    index: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for param in movable_parameters(context):
        name = param["name"]
        default = param.get("default")
        reported = [t for t in trials if name in t["params"]]
        off_default = [t for t in reported if not same(t["params"][name], default)]
        values = sorted({t["params"][name] for t in off_default}, key=lambda v: (str(type(v)), str(v)))
        counts: Dict[str, int] = {}
        for trial in off_default:
            counts[str(trial["params"][name])] = counts.get(str(trial["params"][name]), 0) + 1
        entry = index.get(name) or {}
        status = str(entry.get("status") or "").strip().lower()
        reason = str(entry.get("reason") or "").strip()
        if status == "retired":
            verdict = "retired" if reason else "retired_without_reason"
        elif not off_default:
            verdict = "uncovered"
        elif len(values) == 1 and len(off_default) >= 3:
            # 只有一个非默认值出现 ≥3 次 —— 这不是"探索过"，是空间 default 与实际使用值不一致
            verdict = "pinned_off_default"
        elif len(values) == 1:
            verdict = "under_tested"
        else:
            verdict = "covered"
        rows.append(
            {
                "name": name,
                "default": default,
                "min": param.get("min"),
                "max": param.get("max"),
                "dtype": param.get("dtype"),
                "reported_trials": len(reported),
                "off_default_trials": len(off_default),
                "rounds": [t["round"] for t in off_default],
                "values": values,
                "value_counts": counts,
                "declared_status": status or None,
                "reason": reason or None,
                "verdict": verdict,
            }
        )
    return rows


def resolve_budget_cap(context: Dict[str, Any], dimensions: int, explicit: Optional[int]) -> Tuple[int, str]:
    """单维度 trial 上限：显式值优先，否则按总预算在可动维度间均分。

    将每维度上限固定为 3，在预算大、维度少（或反之）的场景都会失真，
    所以默认从 ``context.budgets.max_trials`` 推导；预算缺失才回落到常量。
    """
    if explicit is not None:
        return max(1, explicit), "--max-trials-per-dim 显式指定"
    budget = (context.get("budgets") or {}).get("max_trials")
    try:
        budget_value = int(budget)
    except (TypeError, ValueError):
        budget_value = 0
    if budget_value > 0 and dimensions > 0:
        cap = max(MIN_TRIALS_PER_DIM, math.ceil(budget_value / dimensions))
        return cap, f"按预算 max_trials={budget_value} / {dimensions} 个可动维度均分"
    return DEFAULT_MAX_TRIALS_PER_DIM, f"无 budgets.max_trials，用缺省值 {DEFAULT_MAX_TRIALS_PER_DIM}"


def cmd(args: argparse.Namespace) -> int:
    run_dir = find_run_dir(args.run_dir)
    context = read_json(run_dir / "context.json")
    trials = load_trials(run_dir)
    index_path = run_dir / INDEX_FILE
    index: Dict[str, Any] = read_json(index_path) if index_path.exists() else {}

    rows = build_matrix(context, trials, index)
    if not rows:
        raise SystemExit("ERROR: context.search_space has no movable parameter")
    cap, cap_source = resolve_budget_cap(context, len(rows), args.max_trials_per_dim)

    findings: List[str] = []
    for row in rows:
        explored = len(row["values"]) >= 2
        if explored and row["off_default_trials"] > cap:
            row["verdict"] = "over_budget"
            detail = ", ".join(f"{key}×{count}" for key, count in sorted(row["value_counts"].items()))
            findings.append(
                f"{row['name']}: {row['off_default_trials']} 个 trial 偏离默认值"
                f"（{detail}），超过单维度上限 {cap}（{cap_source}）"
            )
        elif row["verdict"] == "pinned_off_default":
            single = next(iter(row["value_counts"]))
            findings.append(
                f"{row['name']}: {row['off_default_trials']} 个 trial 固定为 {single}，"
                f"与空间 default={row['default']} 不一致 —— 该维度从未被移动，"
                "应修正 default 或标 retired/constant，不能算作已探索"
            )
        elif row["verdict"] in ("uncovered", "under_tested", "retired_without_reason"):
            detail = {
                "uncovered": "从未偏离默认值",
                "under_tested": "只有 1 个 trial 偏离默认值，不足以判定",
                "retired_without_reason": "声明了 retired 但没写理由",
            }[row["verdict"]]
            findings.append(f"{row['name']}: {detail}")

    payload = {
        "run_dir": str(run_dir),
        "trials": len(trials),
        "movable_dimensions": len(rows),
        "max_trials_per_dim": cap,
        "max_trials_per_dim_source": cap_source,
        "covered": sum(1 for r in rows if r["verdict"] == "covered"),
        "retired": sum(1 for r in rows if r["verdict"] == "retired"),
        "findings": findings,
        "dimensions": rows,
    }

    if args.init_index:
        skeleton = {
            name: {
                "status": row["verdict"] if row["verdict"] in ("covered", "retired") else "active",
                "reason": row["reason"] or "",
            }
            for name, row in ((r["name"], r) for r in rows)
        }
        existing = read_json(index_path) if index_path.exists() else {}
        if existing and not args.force_index:
            # 已有 index（agent 可能已填写 retired/reason）：只补缺失维度，
            # 不覆盖已填写条目；全量重建需显式 --force-index
            merged = dict(existing)
            for name, info in skeleton.items():
                merged.setdefault(name, info)
            write_json(index_path, merged)
            print(f"✓ merged skeleton into {index_path}（保留已填写条目；--force-index 全量重建）", file=sys.stderr)
        else:
            write_json(index_path, skeleton)
            print(
                f"✓ wrote skeleton {index_path}" + ("（--force-index 全量重建）" if existing else ""), file=sys.stderr
            )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"run-dir      : {run_dir}")
        print(f"trials       : {len(trials)}   movable dims: {len(rows)}")
        print(f"covered/retired: {payload['covered']}/{payload['retired']}")
        print(f"单维度上限   : {cap}（{cap_source}）")
        print()
        header = f"{'dimension':<38} {'default':>9} {'moved':>5} {'values':>22}  verdict"
        print(header)
        print("-" * (len(header) + 10))
        for row in rows:
            values = ",".join(f"{v}x{row['value_counts'].get(str(v), 0)}" for v in row["values"][:3]) or "-"
            print(
                f"{row['name'][:38]:<38} {str(row['default'])[:9]:>9} {row['off_default_trials']:>5} "
                f"{values[:22]:>22}  {row['verdict']}"
            )
        print()
        if findings:
            print("FINDINGS:")
            for item in findings:
                print(f"  - {item}")
            print()
            print(
                "VERDICT: 维度覆盖不完整 —— 不得宣称收敛；"
                "未覆盖维度须补测，或写入 dimension_index.json 的 retired 并给出理由。"
            )
        else:
            print("VERDICT: 维度覆盖完整")

    if findings:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dimension coverage matrix for one agent-mode run directory.")
    parser.add_argument("--run-dir", default="", help="run directory (default: unique dir under CWD)")
    parser.add_argument(
        "--max-trials-per-dim",
        type=int,
        default=None,
        help="trial budget per dimension off its default value "
        "(default: derive from context.budgets.max_trials / movable dimensions)",
    )
    parser.add_argument(
        "--init-index",
        action="store_true",
        help=f"write a skeleton {INDEX_FILE}（已有 index 时默认合并保留已填写条目）",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="with --init-index: rebuild the skeleton from scratch (discards filled entries)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return cmd(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
