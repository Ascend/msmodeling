# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the software at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""SKILL 收尾步骤：把一次完整运行（模式B 生成 + 模式A 验证）整理为 Markdown 报告。

报告写入 SKILL 目录（``reports/<db>-run-report.md``），内容涵盖：
1. 产物文件说明（性能数据库位置、mock 库作用、日志、生成报告与后续用途）；
2. 本次执行的仿真命令清单；
3. 每个 case-phase 的 Top-N 计算算子命中情况（通信算子排除，见
   verify_db_coverage.COMM_OP_TOKENS——通信时长由 base HCCL alpha-beta 模型给出，
   wrapper 插值主动跳过查库）；
4. MISS 计算算子总览：以算子为粒度聚合（避免逐 case 重复），按 Top-10 内
   最大 analytic total 降序罗列算子、MISS 原因、根因方向、MISS 次数与 case 场景；
   不含修复建议（根因方向是静态映射，处置方式见 SKILL.md "常见 MISS 处置"）。

输入为 verify_db_coverage.py 产出的 verify-summary.json 与可选的
shape-generation-report.json；本脚本不执行任何仿真。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent

# MISS 原因 → (根因方向, 修复建议)。第 4 节仅展示根因方向（索引 0）；修复建议保留在
# 映射中供人工参照（与 SKILL.md "常见 MISS 处置" 一致并扩充）。
MISS_SUGGESTIONS: dict[str, tuple[str, str]] = {
    "shape_mismatch": (
        "数据库缺该 shape 行",
        "确认场景仍在 cases yaml 后重跑 generate_shape_db.py（已有库会自动追加，无需重新 parse），"
        "或加大 --rows；随后重新 mock 并 verify",
    ),
    "per-shape": (
        "该算子部分 shape 未命中",
        "把缺失 shape 对应的场景保留在 cases yaml 重跑 generate（追加行），重新 mock 并 verify",
    ),
    "elementwise_interpolation_failed": (
        "元素级算子插值失败（常见：dtype 严格不匹配或对应 dtype 模板行缺失）",
        "核对 demand dtype 与数据库模板行 dtype 是否严格相等；在 op_mapping/实测侧补齐对应 dtype 的"
        "模板行后重跑 generate（追加）并重新 mock",
    ),
    "outside_axis_boundary": (
        "查询点在实测轴边界之外（插值不外推）",
        "在实测范围内补边界点（注意 ATTN_SEQ 平方复杂度轴的采样代价），重跑 generate 并重新 mock",
    ),
    "compute_csv_not_found": (
        "该 kernel 的实测 CSV 整库缺失",
        "提供覆盖该 kernel 的 NPU kernel_details.csv 并 --force-parse（或新建库）；无实测数据时本地无法补齐",
    ),
    "attention_index_unavailable": (
        "attention 索引/实测数据缺失（无法定位 attention 行）",
        "核对 op_mapping 的 attention 映射与 runtime metadata 列；补充覆盖该 kernel 的 NPU 实测数据",
    ),
    "wrapper_unmapped": (
        "op_mapping 未映射该 wrapper 算子",
        "用 op-mapping-generator 核对并补 op_mapping.yaml 映射后重跑 verify",
    ),
    "wrapper_communication_owned_by_base": (
        "通信算子时长由 base HCCL alpha-beta 模型给出，wrapper 插值主动跳过查库",
        "正常现象，无需处理（top-total 口径已排除通信算子）",
    ),
    "compute_multidim_interpolation_failed": (
        "M/K/N 网格缺列（sweep workload 部分失败）",
        "降低 --max-workers 重跑 generate_shape_db.py（缓存续跑），再重新 mock 复验",
    ),
    "target_unavailable": (
        "查询无法构造有效查询目标（dtype/字段不在支持集合）",
        "检查 demand 的 dtype 与 runtime 字段（如 pack int4 权重需还原逻辑 K）；详见生成报告 demand_ledger 的拒绝明细",
    ),
    "insufficient_filtered_candidates": (
        "候选行被过滤后不足（预过滤条件过严或模板行覆盖不足）",
        "读取 shape-generation-report.json 的 demand_ledger 定位该算子的 demand 拒绝明细",
    ),
    "regime_key_unmatched": (
        "regime 分档键不匹配（查询 regime 与数据库行 regime 不一致）",
        "核对查询 regime 字段（分组大小/对称性等）与数据库模板行 regime 是否一致",
    ),
}

_SUGGESTION_FALLBACK = (
    "原因未收录",
    "读取 shape-generation-report.json 的 demand_ledger 定位该算子的 demand 拒绝/未命中明细，"
    "再按 SKILL.md 常见 MISS 处置表处理",
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the end-of-run Markdown report for the shape-generate + verify workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--database-path", type=Path, required=True, help="Clean source database (<DB>) built by generate_shape_db."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="verify_db_coverage.py summary. Default: <DB>_mockverify-verify-logs/verify-summary.json",
    )
    parser.add_argument(
        "--generation-report",
        type=Path,
        default=None,
        help="generate_shape_db report JSON. Default: <DB parent>/<DB name>-shape-generation-report.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Output Markdown report. Default: <SKILL>/reports/<DB name>-run-report.md",
    )
    return parser


_REPO_ROOT = Path(__file__).resolve().parents[4]


def rel_path(path: Path | str) -> str:
    """Render a path relative to the repo root when possible (readable report tables)."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def artifact_rows(
    db: Path, mock_dir: Path, log_dir: Path, cases_yaml: Path, generation_report: Path | None
) -> list[tuple[str, str, str]]:
    rows = [
        (
            f"`{rel_path(db)}`",
            "本次构建的性能数据库：实测解析行 + 输入 case demand 驱动的 shape 行（新增行 latency 为空/0）",
            "交给 tools/perf_data_collection/start_microbench.py 或 op_replay/<KernelType>_run.py 做"
            "正式算子回放（只补空时长行）；之后 text_generate --profiling-database 直接消费；"
            "后续新增 case 再次运行 generate 时会直接在该库追加，无需重新解析",
        ),
        (
            f"`{rel_path(db / 'op_mapping.yaml')}`",
            "TensorCast op ↔ NPU kernel 映射（MISS 归类与回放支持判定依据）",
            "映射缺口用 op-mapping-generator 补充",
        ),
        (
            f"`{rel_path(mock_dir)}`",
            "mock 验证库：源库副本，仅为缺失/<=0 的时长列填占位时长（100us），模拟回放完成",
            "仅用于本地命中率/插值验证；禁止回写正式库或用于任何发布流程",
        ),
        (
            f"`{rel_path(log_dir)}/<case>-<phase>_run.log`",
            "每个 case-phase 的 text_generate 完整回显日志（Top-N 命中判定的原始依据）",
            "审计用；本报告第 3/4/5 节均从这些日志解析",
        ),
        (
            f"`{rel_path(log_dir / 'verify-summary.json')}`",
            "verify_db_coverage 的结构化汇总（命令、Top-N 命中、MISS 原因）",
            "本报告与后续 diff 分析的机器可读输入",
        ),
        (
            f"`{rel_path(cases_yaml)}`",
            "输入 workload 定义（case id = workload 1-based 顺序号）",
            "后续追加场景直接编辑该文件并重跑 generate（追加模式）",
        ),
    ]
    if generation_report is not None:
        rows.append(
            (
                f"`{rel_path(generation_report)}`",
                "shape 生成报告：demand 落库明细、拒绝原因（projection/validation_rejected）、"
                "unsupported_kernel_demands",
                "MISS 根因定位与 op_mapping/回放支持 gap 分析的输入",
            )
        )
    return rows


def extract_comm_note(entry: dict) -> str | None:
    """Extract the communication-op exclusion description from the top-total info line."""
    for info_line in entry.get("info", []):
        if "communication op(s) excluded" in info_line and "(" in info_line and "):" in info_line:
            return info_line[info_line.index("(") + 1 : info_line.rindex("):")]
    return None


def _miss_names_from_parsed(phases: list[dict], limit: int = 3) -> str:
    """非 top-total 口径下的 MISS 算子摘要（去重排序，默认取前 3 个）。"""
    names = sorted(
        {
            str(miss.get("func_name", ""))
            for phase in phases
            for miss in phase.get("misses") or []
            if miss.get("func_name")
        }
    )
    return "、".join(names[:limit]) if names else "MISS 明细见第 3/4 节"


def _format_duration(seconds: float) -> str:
    """按量级自适应展示时长（秒输入），与 verify/runtime 日志的 ms/us 切换保持一致。"""
    if seconds >= 5e-4:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds * 1e6:.3f}us"


def render_miss_in_top(entry: dict, top_n: int | None) -> list[str]:
    """MISS 的计算算子表：仅列出全体算子 Top-N 中参与判定的 MISS 计算算子。"""
    top_ops = entry.get("top_ops") or []
    misses = [op for op in top_ops if op.get("judged", True) and not op.get("hit")]
    label = f"case TOP{top_n}" if top_n else "case TOP10"
    if not misses:
        return [f"- {label}中无 MISS 的计算算子。"]
    lines = [f"- {label}中MISS的计算算子如下", ""]
    lines.append("| # | 算子 | empirical total | analytic total | calls | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for rank, op in enumerate(misses, 1):
        lines.append(
            f"| {rank} | `{op['name']}` | {_format_duration(op['empirical'])} "
            f"| {_format_duration(op['analytic'])} | {op['calls']} | MISS |"
        )
    return lines


def render_miss_from_parsed(entry: dict) -> list[str]:
    misses = entry.get("misses") or []
    if not misses:
        return ["- 本 phase 无 MISS 记录。"]
    counter: dict[tuple[str, str], int] = {}
    for miss in misses:
        key = (str(miss.get("func_name", "")), str(miss.get("reason", "")))
        counter[key] = counter.get(key, 0) + 1
    lines = ["- MISS 清单（MISSes/MISS shapes 解析口径）："]
    for (func_name, reason), count in sorted(counter.items()):
        lines.append(f"  - `{func_name}` reason={reason} x{count}")
    return lines


def render_miss_by_operator(summary: dict) -> list[str]:
    """MISS 算子粒度聚合。

    top-total 口径：按 Top-10 内最大 analytic total 降序，无表格逐项罗列（只聚合
    Top-10 MISS 计算算子的 MISS 记录；每个算子下按 MISS 次数降序列出各原因）。
    其他口径（replay/all）：summary 无 top_ops，改为直接聚合 misses 清单，
    避免误报"所有 Top-10 计算算子均命中"。
    标题附占比视角：同 phase 全体 Top-N 算子中该算子的最大 analytic 占 Top-1 的比例
    （verify-summary 无全量算子总时长，用 Top-1 作基准）。
    """
    scope = summary.get("miss_scope", "top-total")
    if scope != "top-total":
        lines = [
            f"> 验收口径为 `{scope}`，本节按 MISS 清单（MISSes/MISS shapes 解析）聚合，不含 Top-N analytic 排序信息。"
        ]
        agg: dict[str, dict[str, list]] = {}
        for phase in summary["phases"]:
            for miss in phase.get("misses") or []:
                name = str(miss.get("func_name", ""))
                if not name:
                    continue
                bucket = agg.setdefault(name, {}).setdefault(str(miss.get("reason", "")), [set(), 0])
                bucket[0].add(phase["label"])
                bucket[1] += 1
        if not agg:
            lines.append("- 所有 case-phase 均无 MISS 记录。")
            return lines
        for name in sorted(agg):
            lines.append("")
            lines.append(f"### {name}")
            reasons = agg[name]
            for reason, (phases, count) in sorted(reasons.items(), key=lambda kv: -kv[1][1]):
                direction = MISS_SUGGESTIONS.get(reason, _SUGGESTION_FALLBACK)[0]
                ps = sorted(phases)
                scene = f"{len(ps)} 个（{'、'.join(ps[:3])}等）" if len(ps) > 3 else f"{len(ps)} 个（{'、'.join(ps)}）"
                lines.append(f"- MISS 原因：{reason}")
                lines.append(f"  根因方向：{direction}")
                lines.append(f"  MISS次数：{count} 次，MISS的case场景：{scene}")
        return lines
    op_analytic: dict[str, float] = {}
    phase_top1: dict[str, float] = {}
    agg: dict[str, dict[str, list]] = {}
    for phase in summary["phases"]:
        top_analytics = [float(op["analytic"]) for op in phase.get("top_ops") or []]
        if top_analytics:
            phase_top1[phase["label"]] = max(top_analytics)
        for op in phase.get("top_ops") or []:
            if op.get("judged", True) and not op.get("hit"):
                name = op["name"]
                op_analytic[name] = max(op_analytic.get(name, 0.0), float(op["analytic"]))
        for miss in phase.get("misses") or []:
            name = str(miss.get("func_name", ""))
            if name not in op_analytic:
                continue
            bucket = agg.setdefault(name, {}).setdefault(str(miss.get("reason", "")), [set(), 0])
            bucket[0].add(phase["label"])
            bucket[1] += 1
    if not op_analytic:
        return ["", f"- 所有 TOP{summary.get('top') or 10} 计算算子均命中，无 MISS。"]
    label_n = summary.get("top") or 10
    lines = []
    for name in sorted(op_analytic, key=lambda n: -op_analytic[n]):
        base_phase = None
        for phase in summary["phases"]:
            for op in phase.get("top_ops") or []:
                if (
                    op.get("judged", True)
                    and not op.get("hit")
                    and op["name"] == name
                    and float(op["analytic"]) == op_analytic[name]
                ):
                    base_phase = phase["label"]
                    break
            if base_phase:
                break
        ratio = ""
        if base_phase and phase_top1.get(base_phase):
            pct = op_analytic[name] / phase_top1[base_phase] * 100
            ratio = f"，占 {base_phase} Top-1 的 {pct:.1f}%"
        lines.append("")
        lines.append(f"### {name}（TOP{label_n} 内最大 analytic total {_format_duration(op_analytic[name])}{ratio}）")
        reasons = agg.get(name, {})
        for reason, (phases, count) in sorted(reasons.items(), key=lambda kv: -kv[1][1]):
            direction = MISS_SUGGESTIONS.get(reason, _SUGGESTION_FALLBACK)[0]
            ps = sorted(phases)
            scene = f"{len(ps)} 个（{'、'.join(ps[:3])}等）" if len(ps) > 3 else f"{len(ps)} 个（{'、'.join(ps)}）"
            lines.append(f"- MISS 原因：{reason}")
            lines.append(f"  根因方向：{direction}")
            lines.append(f"  MISS次数：{count} 次，MISS的case场景：{scene}")
    return lines


def render_generation_overview(generation_report: Path | None) -> list[str]:
    if generation_report is None or not generation_report.is_file():
        return ["- 生成报告不存在（生成阶段未运行或路径变更）。"]
    data = json.loads(generation_report.read_text(encoding="utf-8"))
    lines = [f"- 报告文件：`{rel_path(generation_report)}`"]
    totals = data.get("totals") or {}
    for key, value in totals.items():
        lines.append(f"- {key}: {value}")
    unsupported = data.get("unsupported_kernel_demands") or []
    if unsupported:
        lines.append(f"- unsupported_kernel_demands: {len(unsupported)} 项（详见报告 JSON）")
    return lines


def render_header_summary(summary: dict, top_n: int | None) -> list[str]:
    """头部总判定摘要 + cases yaml workload 摘要（一眼结论，不翻到文末）。"""
    phases = summary["phases"]
    fail_phases = [p["label"] for p in phases if p.get("passed") is False]
    passed = not fail_phases
    if passed:
        verdict = f"**ALL PASS**（{len(phases)} 个 phase 全部命中）"
    else:
        scope = summary.get("miss_scope", "top-total")
        if scope == "top-total":
            op_max: dict[str, float] = {}
            for phase in phases:
                for op in phase.get("top_ops") or []:
                    if op.get("judged", True) and not op.get("hit"):
                        name = op["name"]
                        op_max[name] = max(op_max.get(name, 0.0), float(op["analytic"]))
            top3 = sorted(op_max, key=op_max.get, reverse=True)[:3]
            top3_desc = "、".join(top3) if top3 else "MISS 明细见第 3/4 节"
            verdict = (
                f"**FAIL**（{len(fail_phases)}/{len(phases)} 个 phase 存在 Top-{top_n} 计算算子 roofline 回退，"
                f"最大缺口算子：{top3_desc}）"
            )
        else:
            # replay/all 口径下 top_ops 恒为空：缺口从 misses 清单聚合，不能沿用 Top-N roofline 文案
            top3_desc = _miss_names_from_parsed(phases)
            verdict = f"**FAIL**（{len(fail_phases)}/{len(phases)} 个 phase 存在回放/查库缺口，MISS 算子：{top3_desc}）"
    lines = [f"- 总判定：{verdict}"]
    cases_yaml = Path(summary.get("cases_yaml", ""))
    workloads: list[dict] = []
    if cases_yaml.is_file():
        try:
            import yaml

            data = yaml.safe_load(cases_yaml.read_text(encoding="utf-8")) or {}
            workloads = data.get("workloads") or []
        except Exception:
            workloads = []
    if workloads:
        parts = []
        for w in workloads:
            in_len = w.get("input_length")
            out_len = w.get("output_length")
            parts.append(
                f"输入{in_len}/输出{out_len}" if in_len is not None and out_len is not None else str(w.get("name", "?"))
            )
        parallel = ""
        first = workloads[0]
        tp = first.get("tp_sizes")
        ep = first.get("ep_sizes")
        quant = first.get("quantize_linear_action")
        bits = []
        if tp:
            bits.append(f"TP{'/'.join(map(str, tp))}")
        if ep:
            bits.append(f"EP{'/'.join(map(str, ep))}")
        if quant:
            bits.append(str(quant))
        parallel = "；" + "、".join(bits) if bits else ""
        lines.append(f"- cases 摘要：{len(workloads)} 个 workload（{'、'.join(parts)}）{parallel}")
    return lines


def fold_phases(summary: dict) -> list[list[dict]]:
    """第 3 节折叠：判定结果完全一致的同 phase 分组合并展示。

    签名覆盖判定依据的全部来源：top-total 口径为排除说明 + Top-N MISS 表；
    replay/all 口径下 top_ops 恒为空，因此还须覆盖 misses 清单与 passed，
    否则不同 case 的不同 MISS 集合会被误折叠为"判定结果完全相同"。
    Returns 分组列表；每组用组内第一个 phase 渲染，其余在标题行追加别名。
    """
    groups: list[list[dict]] = []
    sig_by_phase: dict[str, str] = {}
    group_idx_by_phase: dict[str, int] = {}
    for entry in summary["phases"]:
        miss_view = [
            (op["name"], op["empirical"], op["analytic"], op["calls"], op.get("judged", True), op.get("hit"))
            for op in entry.get("top_ops") or []
        ]
        misses_view = [
            (str(m.get("func_name", "")), str(m.get("reason", "")), m.get("shape")) for m in entry.get("misses") or []
        ]
        phase = str(entry.get("phase", ""))
        sig = repr((extract_comm_note(entry), miss_view, misses_view, entry.get("passed")))
        if phase and sig_by_phase.get(phase) == sig:
            groups[group_idx_by_phase[phase]].append(entry)
            continue
        group_idx_by_phase[phase] = len(groups)
        groups.append([entry])
        sig_by_phase[phase] = sig
    return groups


def main() -> None:
    args = build_argparser().parse_args()
    db = args.database_path.resolve()
    if not (db / "op_mapping.yaml").is_file():
        raise SystemExit(f"not a profiling database (missing op_mapping.yaml): {db}")
    summary_path = (
        args.summary_json or db.parent / f"{db.name}_mockverify-verify-logs" / "verify-summary.json"
    ).resolve()
    if not summary_path.is_file():
        raise SystemExit(
            f"verify summary not found: {summary_path}\n"
            "run verify_db_coverage.py first (it writes verify-summary.json into the log dir)"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generation_report = (args.generation_report or db.parent / f"{db.name}-shape-generation-report.json").resolve()
    report_path = (args.report_path or SKILL_DIR / "reports" / f"{db.name}-run-report.md").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(summary["phases"][0]["log_file"]).parent if summary["phases"] else db.parent
    cases_yaml = Path(summary.get("cases_yaml", ""))
    scope = summary.get("miss_scope", "top-total")
    top_n = summary.get("top")

    lines: list[str] = []
    lines.append(f"# 性能数据库 Shape 生成与命中验证报告：`{db.name}`")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 性能数据库：`{rel_path(db)}`")
    lines.append(f"- cases yaml：`{rel_path(cases_yaml)}`")
    lines.append(
        f"- 验收口径：`{scope}`" + (f"（全体算子 Top-{top_n}，通信算子不参与判定）" if scope == "top-total" else "")
    )
    lines.extend(render_header_summary(summary, top_n))

    lines.append("")
    lines.append("## 1. 产物文件说明")
    lines.append("")
    lines.append("| 文件 | 作用 | 后续用途 |")
    lines.append("|---|---|---|")
    for path, role, usage in artifact_rows(
        db,
        db.parent / f"{db.name}_mockverify",
        log_dir,
        cases_yaml,
        generation_report if generation_report.is_file() else None,
    ):
        lines.append(f"| {path} | {role} | {usage} |")

    lines.append("")
    lines.append("## 2. 仿真命令")
    for entry in summary["phases"]:
        lines.append("")
        lines.append(f"### {entry['label']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(str(part) for part in entry["command"]))
        lines.append("```")

    lines.append("")
    lines.append("## 3. 仿真判定结果")
    lines.append("")
    all_passed = True
    lines.append(
        "> 通信算子（all_reduce/all_gather/all_to_all/reduce_scatter 等）已排除：其时长由 base"
        " HCCL alpha-beta 模型给出，wrapper 插值主动跳过查库，不参与命中判定。"
    )
    if scope != "top-total":
        lines.append(
            f"> 注意：本次验收口径为 `{scope}`，下方展示 MISS 清单而非 Top-N 计算算子表"
            "（top_ops 仅在 top-total 口径下生成）。"
        )
    for group in fold_phases(summary):
        first = group[0]
        all_passed &= all(e.get("passed") is not False for e in group)
        lines.append("")
        if len(group) > 1:
            aliases = "、".join(e["label"] for e in group[1:])
            lines.append(f"### {first['label']}（判定结果与 {aliases} 完全相同）")
            logs = "；".join(f"`{rel_path(e['log_file'])}`" for e in group[1:])
            lines.append(f"- 其余 phase 日志：{logs}")
        else:
            lines.append(f"### {first['label']}，日志 `{rel_path(first['log_file'])}`")
        comm_note = extract_comm_note(first)
        if comm_note:
            lines.append(f"- 通信算子排除说明：`{comm_note}`")
        if scope == "top-total":
            lines.extend(render_miss_in_top(first, top_n))
        else:
            lines.extend(render_miss_from_parsed(first))

    lines.append("")
    if scope == "top-total":
        lines.append("## 4. MISS 计算算子总览（按 analytic 总时长降序）")
    else:
        lines.append(f"## 4. MISS 总览（{scope} 口径，按算子聚合）")
    lines.extend(render_miss_by_operator(summary))

    lines.append("")
    lines.append("## 5. 生成阶段概览")
    lines.extend(render_generation_overview(generation_report if generation_report.is_file() else None))

    lines.append("")
    lines.append("## 结论")
    lines.append("")
    fail_labels = [p["label"] for p in summary["phases"] if p.get("passed") is False]
    fail_labels_desc = "、".join(fail_labels) if len(fail_labels) <= 4 else "全部 phase"
    if not fail_labels:
        if scope == "top-total":
            lines.append(
                f"**ALL PASS**：{len(summary['phases'])} 个 case-phase 的 Top-{top_n} 计算算子均命中数据库，无 roofline 回退。"
            )
        else:
            lines.append(f"**ALL PASS**：{len(summary['phases'])} 个 case-phase 无查库/回放缺口（{scope} 口径）。")
    else:
        if scope == "top-total":
            op_max: dict[str, float] = {}
            for phase in summary["phases"]:
                for op in phase.get("top_ops") or []:
                    if op.get("judged", True) and not op.get("hit"):
                        op_max[op["name"]] = max(op_max.get(op["name"], 0.0), float(op["analytic"]))
            top3 = sorted(op_max, key=op_max.get, reverse=True)[:3]
            top3_desc = "、".join(top3) if top3 else "MISS 明细见第 3/4 节"
            lines.append(
                f"**FAIL**：{len(fail_labels)}/{len(summary['phases'])} 个 case-phase 存在 Top-{top_n} 计算算子 roofline 回退"
                f"（{fail_labels_desc}）；"
                f"优先处理缺口 {top3_desc}。按第 4 节根因方向对照 SKILL.md 「常见 MISS 处置」处理"
                "（补行/op_mapping/实测数据），重跑 generate（追加）→ mock → verify 后再生成本报告。"
            )
        else:
            lines.append(
                f"**FAIL**：{len(fail_labels)}/{len(summary['phases'])} 个 case-phase 存在回放/查库缺口"
                f"（{fail_labels_desc}）；MISS 算子：{_miss_names_from_parsed(summary['phases'])}。"
                "按第 4 节根因方向对照 SKILL.md 「常见 MISS 处置」处理"
                "（补行/op_mapping/实测数据），重跑 generate（追加）→ mock → verify 后再生成本报告。"
            )
    lines.append("")
    lines.append("> 本报告为本地验证产物，禁止提交仓库或外发。")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[generate_run_report] report written: {report_path}")
    print(f"[generate_run_report] overall: {'ALL PASS' if all_passed else 'FAIL (see 结论 section)'}")


if __name__ == "__main__":
    main()
