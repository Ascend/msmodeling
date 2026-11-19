#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# MindStudio project. Licensed under Mulan PSL v2 (同仓内其他脚本).
# -------------------------------------------------------------------------
"""从 optix agent 寻优 run 目录提取负结果（retired 维度 + 对照实测），
结构化写入 case-knowledge 的 negative 分区，供后续 run 的 Round 0 按
场景指纹检索（对应 agent.md「历史负知识引用」与 data-analysis.md
「知识参考优先级」第 5 条 S6 口径指纹规则）。

数据来源（全部只读）：
- dimension_index.json        : status=retired 的参数与 reason
- results.round-*.json        : 候选 params 与 performance（对照幅度）
- noise_band.json             : σ 判据（优先；历史 run 回退 context.noise_band）
- context.json                : 场景字段（指纹结构化直读来源）
- config.toml（尽力解析）      : 模型路径、并行、benchmark 口径

条目判据（防止把噪声波动误记成参数无效结论入库）：
- 对照优先取值分组（同 run 单变量对照）；分组不足时回退全 run 最优 trial
  作对照（条目标注「非同轮单变量对照」，confidence 最高记 medium）；
- 对照幅度（默认吞吐）相对差 |Δ|/基准 ≥ 2σ（noise_band 对应指标）；
  σ 缺失时退回 --min-ratio 绝对阈值判定；
- 不满足时跳过并提示（--force 可入库，条目标 confidence: low）。

用法：
  python export_negative_knowledge.py --run-dir <run目录> [--dry-run] [--force]
  python export_negative_knowledge.py --run-dir <run目录> --knowledge-root <case-knowledge目录>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# noise_band 校验与 metrics_gate compare 共用同一实现（格式/符号/上限统一把关）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optix_assistant_common import extract_noise_band, validate_noise_sigma  # noqa: E402

try:  # Python 3.11+
    import tomllib  # type: ignore

    _HAS_TOMLLIB = True
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore
    _HAS_TOMLLIB = False

try:
    import tomli  # type: ignore

    _HAS_TOMLLIB = True
    _TOMLI = True
except ImportError:  # pragma: no cover
    _TOMLI = False

SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE = SKILL_ROOT / "knowledge" / "case-knowledge"

PERF_KEY_THROUGHPUT = "generate_speed"
# 延迟类指标越小越好：对照选择（best/worst/fallback/坏组）的比较方向随之翻转
LOWER_IS_BETTER_METRICS = {"time_per_output_token", "time_to_first_token"}


def is_lower_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER_METRICS


SIGMA_KEYS = {
    PERF_KEY_THROUGHPUT: ("generate_speed_sigma", "吞吐"),
    "time_per_output_token": ("tpot_sigma", "TPOT"),
    "time_to_first_token": ("ttft_sigma", "TTFT"),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] 无法读取 {path.name}: {exc}", file=sys.stderr)
        return None


def load_toml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        if _HAS_TOMLLIB and tomllib is not None:
            with open(path, "rb") as fh:
                return tomllib.load(fh)
        if _TOMLI and tomli is not None:  # type: ignore[name-defined]
            with open(path, "rb") as fh:
                return tomli.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[warn] 无法解析 {path.name}: {exc}", file=sys.stderr)
    return None


def collect_results(run_dir: Path) -> List[Dict[str, Any]]:
    """汇总全部 results.round-N.json 的 trials（附 round 标注）。"""
    trials: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("results.round-*.json")):
        data = load_json(path)
        if not data:
            continue
        for trial in data.get("trials", []) or []:
            item = dict(trial)
            item["_round"] = data.get("round")
            trials.append(item)
    return trials


def retired_params(dim_index: Dict[str, Any]) -> List[Tuple[str, str]]:
    out = []
    for name, info in (dim_index or {}).items():
        if isinstance(info, dict) and info.get("status") == "retired":
            out.append((name, str(info.get("reason", ""))))
    return out


def group_trials_by_param(trials: List[Dict[str, Any]], param: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials:
        params = trial.get("params") or {}
        # PD 分侧维度（*_prefill/_decode）与单机维度统一按子串匹配取值
        value = None
        if param in params:
            value = params[param]
        else:
            for key in params:
                if key == param or key.startswith(param):
                    value = params[key]
                    break
        if value is None:
            continue
        groups.setdefault(f"{value}", []).append(trial)
    return groups


def best_of(groups: Dict[str, List[Dict[str, Any]]], metric: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """按 metric 最优取各组最优 trial（吞吐取最大，延迟类取最小）。"""
    best = None
    for value, members in groups.items():
        for trial in members:
            perf = trial.get("performance") or {}
            raw = perf.get(metric)
            if raw is None:
                continue
            try:
                num = float(raw)
            except (TypeError, ValueError):
                continue
            if best is None or (num < best[1] if is_lower_better(metric) else num > best[1]):
                best = (value, num, trial)  # type: ignore[assignment]
    if best is None:
        return None
    return best[0], best[2]  # type: ignore[misc]


def worst_of(groups: Dict[str, List[Dict[str, Any]]], metric: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    worst = None
    for value, members in groups.items():
        for trial in members:
            perf = trial.get("performance") or {}
            raw = perf.get(metric)
            if raw is None:
                continue
            try:
                num = float(raw)
            except (TypeError, ValueError):
                continue
            if worst is None or (num > worst[1] if is_lower_better(metric) else num < worst[1]):
                worst = (value, num, trial)  # type: ignore[assignment]
    if worst is None:
        return None
    return worst[0], worst[2]  # type: ignore[misc]


def median_trial(members: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    """组内 metric 中位数对应的实测 trial。

    对照用中位数而不是极值：最差/最优单样本会把两侧噪声都算进 gap，
    组内样本越多 gap 越大，真实效应为 0 也可能越过 2σ 入库。
    """
    values: List[Tuple[float, Dict[str, Any]]] = []
    for m in members:
        perf = m.get("performance") or {}
        raw = perf.get(metric)
        if raw is None:
            continue
        try:
            values.append((float(raw), m))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    med = statistics.median(v for v, _ in values)
    return min(values, key=lambda vm: abs(vm[0] - med))[1]


def fallback_reference(
    trials: List[Dict[str, Any]], param: str, metric: str, bad_value: str
) -> Optional[Dict[str, Any]]:
    """取值分组不足时的对照：全 run 中该参数未取坏值 trial 的 metric 中位数（取最接近中位数的实测 trial）。

    params 显式键缺失或取值 != bad_value 的 trial 按中位数挑选；
    调用方须在条目中标注「非同轮单变量对照」。找不到返回 None。
    """
    pool: List[Dict[str, Any]] = []
    for trial in trials:
        params = trial.get("params") or {}
        value = None
        if param in params:
            value = params[param]
        else:
            for key in params:
                if key == param or key.startswith(param):
                    value = params[key]
                    break
        if value is not None and f"{value}" == bad_value:
            continue
        pool.append(trial)
    return median_trial(pool, metric)


def relative_gap(good: float, bad: float, metric: str) -> float:
    """bad 相对 good 的劣化幅度（正数 = 更差）。"""
    if metric == PERF_KEY_THROUGHPUT:
        if good <= 0:
            return 0.0
        return (good - bad) / good
    # 延迟类指标劣化 = (bad - good) / good
    if good <= 0:
        return 0.0
    return (bad - good) / good


def extract_fingerprint(run_dir: Path) -> Dict[str, Any]:
    """场景指纹：模型/引擎/硬件/拓扑/负载口径。多来源尽力提取，缺项标 unknown。"""
    fp: Dict[str, Any] = {
        "model": "unknown",
        "vllm_ascend": "unknown",
        "hardware": "unknown",
        "mode": "unknown",
        "input_len": None,
        "output_len": None,
        "concurrency": None,
        "request_rate": None,
        "prefix_cache": None,
        "design_hit_rate": None,
        "temperature": "unknown",
        "context_len": None,
    }
    ctx = load_json(run_dir / "context.json") or {}
    text_dump = json.dumps(ctx, ensure_ascii=False)
    dim = load_json(run_dir / "dimension_index.json") or {}
    dim_dump = json.dumps(dim, ensure_ascii=False)
    model = None
    # 结构化直读优先（context.model / model_info）；正则回退只用于无结构化来源的历史 run
    struct_model = ctx.get("model") or ctx.get("model_info")
    if isinstance(struct_model, dict):
        struct_model = struct_model.get("name") or struct_model.get("model")
    if isinstance(struct_model, str) and struct_model.strip():
        model = struct_model.strip()
        if "/" in model:
            model = Path(model).name
    if model is None:
        for pattern in (r"[A-Za-z0-9_.\-]*Qwen[A-Za-z0-9_.\-]*", r"[A-Za-z0-9_.\-]*DeepSeek[A-Za-z0-9_.\-]*"):
            m = re.search(pattern, text_dump)
            if m and "path" not in m.group(0).lower():
                model = m.group(0)
                break
    if model:
        fp["model"] = model
    # config.toml 尽力解析（仅 serve 命令口径；run 目录无 config.toml 时走 context 直读）
    cfg = None
    for cand in (
        sorted(run_dir.parent.glob("config*.toml"))
        + sorted((run_dir).glob("config*.toml"))
        + sorted((run_dir / "..").glob("config*.toml"))
    ):
        cfg = load_toml(cand)
        if cfg:
            break
    others = ""
    if cfg:
        vllm_cmd = (cfg.get("vllm") or {}).get("command") or {}
        others = str(vllm_cmd.get("others", ""))
        m = re.search(r"(?:serve\s+)?(/[^\s\"']+)", others)
        if m and fp["model"] == "unknown":
            fp["model"] = Path(m.group(1)).name
        if "--enable-prefix-caching" in others:
            fp["prefix_cache"] = "on"
        elif "--no-enable-prefix-caching" in others:
            fp["prefix_cache"] = "off"
        m_ctx = re.search(r"--max-model-len\s+(\d+)", others)
        if m_ctx:
            fp["context_len"] = int(m_ctx.group(1))
    # 硬件 tag：context.hardware 直读（1n8npu-64gb 风格；不推测型号）
    hw = ctx.get("hardware") or {}
    num_nodes = hw.get("num_nodes")
    num_per_node = hw.get("num_per_node") or hw.get("world_size")
    mem = hw.get("single_card_memory_gb")
    hw_bits = []
    if num_nodes and num_per_node:
        hw_bits.append(f"{num_nodes}n{num_per_node}npu")
    if mem:
        hw_bits.append(f"{float(mem):.0f}gb")
    if hw_bits:
        fp["hardware"] = "-".join(hw_bits)
    # 部署形态：PD 硬证据 > config others TP > search_space.constants TP > 单机推断
    tp = None
    m_tp = re.search(r"--tensor-parallel-size\s+(\d+)", others)
    if m_tp:
        tp = int(m_tp.group(1))
    else:
        for const in (ctx.get("search_space") or {}).get("constants") or []:
            if isinstance(const, dict) and str(const.get("name", "")).upper() == "TENSOR_PARALLEL_SIZE":
                tp = const.get("default") if const.get("default") is not None else const.get("constant")
                break
    pd_flag = (
        "kv_producer" in text_dump
        or "kv_consumer" in text_dump
        or "prefill_world_size" in dim_dump
        or "prefill_world_size" in text_dump
    )
    if pd_flag:
        fp["mode"] = "pd"
    elif tp:
        fp["mode"] = f"single-tp{tp}"
    elif num_nodes == 1 and num_per_node:
        fp["mode"] = f"single-tp{num_per_node}"
    # 负载口径：context.workload 直读优先，数据集文件名正则回退
    wl = ctx.get("workload") or {}
    if wl.get("input_len_avg") is not None:
        fp["input_len"] = wl.get("input_len_avg")
    if wl.get("output_len_avg") is not None:
        fp["output_len"] = wl.get("output_len_avg")
    # 并发与请求频率：search_space.constants 直读优先
    for const in (ctx.get("search_space") or {}).get("constants") or []:
        if not isinstance(const, dict):
            continue
        name = str(const.get("name", "")).upper()
        val = const.get("default") if const.get("default") is not None else const.get("constant")
        if name == "CONCURRENCY" and val is not None and fp["concurrency"] is None:
            fp["concurrency"] = val
        if name == "REQUESTRATE" and val is not None and fp["request_rate"] is None:
            fp["request_rate"] = val
    # noise_band（σ 判据）：统一入口 noise_band.json 优先（SKILL.md「PD 分离部署的强制约束」第 4 条），
    # 兼容未包装 dict 与 {"noise_band": {...}} 两种包装；历史 run 回退 context.noise_band
    nb, _band_source = extract_noise_band(run_dir)
    if nb:
        fp["_noise_band"] = nb
    # 引擎版本：context → dimension_index reason 文本
    m_ver = re.search(r"0\.\d{2}\.\d[^\"'\s,}\]]*", text_dump) or re.search(r"0\.\d{2}\.\d[^\"'\s,}\]]*", dim_dump)
    if m_ver:
        fp["vllm_ascend"] = m_ver.group(0)
    # 数据集文件名里的口径（in3500-num256-repeatRate0.9 等）
    m_ds = re.search(r"in(\d+)-num(\d+)[^\s\"]*?(?:repeatRate([\d.]+))?", text_dump)
    if m_ds:
        if fp["input_len"] is None:
            fp["input_len"] = int(m_ds.group(1))
        if m_ds.group(3):
            fp["design_hit_rate"] = float(m_ds.group(3))
    m_out = re.search(r"max_tokens[\"':=\s]+(\d+)", text_dump)
    if m_out and fp["output_len"] is None:
        fp["output_len"] = int(m_out.group(1))
    return fp


def fmt_perf(perf: Dict[str, Any]) -> str:
    gen = perf.get(PERF_KEY_THROUGHPUT)
    ttft = perf.get("time_to_first_token")
    tpot = perf.get("time_per_output_token")
    parts = []
    if gen is not None:
        parts.append(f"吞吐 {float(gen):.1f} tok/s")
    if ttft is not None:
        parts.append(f"TTFT {float(ttft):.3f}s")
    if tpot is not None:
        parts.append(f"TPOT {float(tpot) * 1000:.1f}ms")
    return "、".join(parts) or "无指标"


def build_entry_md(
    param: str,
    reason: str,
    fp: Dict[str, Any],
    good: Tuple[str, Dict[str, Any]],
    bad: Tuple[str, Dict[str, Any]],
    gap: float,
    sigma_used: Optional[float],
    metric: str,
    run_id: str,
    confidence: str,
    ref_note: str = "",
) -> str:
    model = fp.get("model", "unknown")
    hardware = fp.get("hardware", "unknown")
    mode = fp.get("mode", "unknown")
    vllm_ascend = fp.get("vllm_ascend", "unknown")
    wl_bits = []
    for key in (
        "input_len",
        "output_len",
        "concurrency",
        "request_rate",
        "prefix_cache",
        "design_hit_rate",
        "context_len",
    ):
        if fp.get(key) not in (None, "unknown"):
            wl_bits.append(f"{key}={fp[key]}")
    workload = ", ".join(wl_bits) or "unknown"
    cid_good, t_good = good
    cid_bad, t_bad = bad
    ref_line = f"**注意：{ref_note}。**" if ref_note else "对照为同 run 分组单变量对照。"
    table_title = "## 对照实测（跨轮对照）" if ref_note else "## 对照实测（同 run 单变量）"
    sigma_line = (
        f"σ={sigma_used:.4f}，阈值取 max(2σ, --min-ratio)" if sigma_used else "噪声判据无效或缺失，按绝对阈值判定"
    )
    metric_name = SIGMA_KEYS.get(metric, (metric, metric))[1]
    return f"""---
kind: run_negative_finding
parameter: "{param}"
model: "{model}"
vllm_ascend: "{vllm_ascend}"
hardware: "{hardware}"
mode: "{mode}"
workload: "{{{workload}}}"
metric: "{metric}"
degradation: "{gap * 100:.1f}%"
confidence: {confidence}
source_run: "{run_id}"
---

# 负结果：{param}（{mode} / {model}）

## 结论

在上述场景指纹下开启 `{param}` 为**明确负收益**：{metric_name}劣化 {gap * 100:.1f}%，
远超 2σ 判据（{sigma_line}）。

{table_title}

| 取值 | 候选 | {metric_name} |
|---|---|---|
| 基准（{cid_good}） | {fmt_perf(t_good.get("performance") or {})} | 优 |
| 开启（{cid_bad}） | {fmt_perf(t_bad.get("performance") or {})} | 劣化 {gap * 100:.1f}% |

对照数据来源：`results.round-*.json`（source_run: {run_id}）。{ref_line}

## 机制推断（agent 记录，未经 profiler 证实）

> {reason or '（run 目录 reason 为空）'}

## 未验证场景（引用前必读）

本结论**仅绑定上述场景指纹**。以下场景的边际效应未知，引用时不得外推：

- 部署形态不同（{mode} → PD / 多机 / 其他并行）；
- 负载口径不同（输入/输出长度、并发、到达模式、prefix cache 结构）；
- 引擎版本不同（{vllm_ascend} → 其他版本，参数可能已迁移或行为变化）。

引用规则见 data-analysis.md「知识参考优先级」第 5/6 条：指纹一致才可作
「已被复测否定」的强先验；指纹不一致降级「仅参考」，如需采纳须先做单变量复验。
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 run 目录导出负结果知识条目到 case-knowledge negative 分区")
    parser.add_argument("--run-dir", required=True, help="run 目录（.agent_optimizer/runs/<run_id>）")
    parser.add_argument("--knowledge-root", default=str(DEFAULT_KNOWLEDGE), help="case-knowledge 根目录")
    parser.add_argument("--out", default=None, help="覆盖输出目录（默认 <knowledge-root>/cases/negative）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要生成的条目，不落盘")
    parser.add_argument("--force", action="store_true", help="证据不足 2σ 时仍入库（条目标 confidence: low）")
    parser.add_argument("--min-ratio", type=float, default=0.10, help="σ 缺失时的绝对劣化阈值（默认 0.10）")
    parser.add_argument("--metric", default=PERF_KEY_THROUGHPUT, help="判定幅度用的指标（默认 generate_speed）")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[error] run 目录不存在: {run_dir}", file=sys.stderr)
        return 3
    run_id = run_dir.name

    dim = load_json(run_dir / "dimension_index.json")
    if dim is None:
        print(f"[error] 缺 dimension_index.json: {run_dir}", file=sys.stderr)
        return 3
    retired = retired_params(dim)
    if not retired:
        print("[no-op] 无 retired 条目，无负结果可沉淀（exit 0）")
        return 0

    trials = collect_results(run_dir)
    if not trials:
        print("[error] 无 results.round-*.json，无法构建对照", file=sys.stderr)
        return 3

    fp = extract_fingerprint(run_dir)
    nb = fp.get("_noise_band") or {}
    sigma = None
    noise_usable = False
    noise_broken = False
    sig_key = SIGMA_KEYS.get(args.metric)
    raw_sigma = nb.get(sig_key[0]) if sig_key else None
    if raw_sigma is not None:
        # 与 metrics_gate compare 共用校验：格式非法/负数/超可信上限不再静默采纳
        check = validate_noise_sigma(raw_sigma)
        if check["status"] == "error":
            noise_broken = True
            hint = (
                "请重新校准，compare 侧可用 --noise-ack-wide 人工确认后继续"
                if check["ackable"]
                else "修复 noise_band.json 后重跑"
            )
            print(
                f"[warn] noise_band.{sig_key[0]} = {raw_sigma!r} {check['error']}，"
                f"2σ 判据无效，按 --min-ratio 绝对阈值执行（{hint}）",
                file=sys.stderr,
            )
        elif check["status"] == "zero":
            noise_broken = True
            print(
                f"[warn] noise_band.{sig_key[0]} = 0 {check['warning']}，按 --min-ratio 绝对阈值执行",
                file=sys.stderr,
            )
        else:
            sigma = check["sigma"]
            noise_usable = True

    out_dir = Path(args.out) if args.out else Path(args.knowledge_root) / "cases" / "negative"
    # max(2σ, --min-ratio)：σ 校准异常小（如 0.1%）时 2σ 判据会把噪声级差异放进
    # 知识库并标 high —— 绝对阈值回退，2σ 只在更高时收紧
    threshold = max(2 * sigma, args.min_ratio) if noise_usable else args.min_ratio

    written, skipped, lowconf = [], [], []
    for param, reason in retired:
        groups = group_trials_by_param(trials, param)
        ref_note = ""
        if len(groups) < 2:
            members = next(iter(groups.values())) if groups else None
            picked = median_trial(members, args.metric) if members else None
            bad = (next(iter(groups)), picked) if picked is not None else None
            if bad is None:
                skipped.append((param, "results 中无该参数的取值分组，无法构建对照"))
                continue
            ref = fallback_reference(trials, param, args.metric, bad[0])
            if ref is None:
                skipped.append((param, f"该参数仅 {bad[0]} 取值有实测，全 run 无对照 trial（该参数取值固定未变）"))
                continue
            good = ("reference", ref)
            ref_note = (
                f"对照为全 run 未取坏值 trial 的中位数 {ref.get('candidate_id', '?')}"
                f"（round {ref.get('_round', '?')}），非同轮单变量对照"
            )
        else:
            good = best_of(groups, args.metric)
            bad = None
            # 锚定 1：reason 里的 candidate_id（agent 记录的劣化 trial）
            m_cid = re.search(r"\br\d+-[a-z0-9]+\b", reason)
            if m_cid and good:
                for value, members in groups.items():
                    for t in members:
                        if t.get("candidate_id") == m_cid.group(0) and value != good[0]:
                            bad = (value, t)
                            break
                    if bad:
                        break
            # 锚定 2：组间对照——排除 good 组，取组内最优最低的组为坏组
            if bad is None and good:
                cand = None
                for value, members in groups.items():
                    if value == good[0]:
                        continue
                    g = best_of({value: members}, args.metric)
                    if g is None:
                        continue
                    num = float((g[1].get("performance") or {}).get(args.metric))
                    # 比较必须内联在 or 的右侧：拆成独立赋值会在 cand 首次出现前
                    # 求值 cand[1]，reason 无候选 ID 时（锚定 2 是唯一路径）直接 TypeError
                    if cand is None or (num > cand[1] if is_lower_better(args.metric) else num < cand[1]):
                        cand = (value, num, g[1])
                if cand:
                    bad = (cand[0], cand[2])
            if not good or not bad or good[0] == bad[0]:
                skipped.append((param, "对照分组退化（无有效指标或同值）"))
                continue
        g_num = float((good[1].get("performance") or {}).get(args.metric))
        b_num = float((bad[1].get("performance") or {}).get(args.metric))
        gap = relative_gap(g_num, b_num, args.metric)
        if gap < threshold and not args.force:
            skipped.append((param, f"劣化 {gap * 100:.1f}% 低于判据 {threshold * 100:.1f}%（2σ 或 --min-ratio）"))
            continue
        confidence = "high" if noise_usable else ("low" if noise_broken else "medium")
        if ref_note and confidence == "high":
            # 非同轮单变量对照：置信度最高记 medium
            confidence = "medium"
        if args.force and gap < threshold:
            confidence = "low"
        md = build_entry_md(
            param, reason, fp, good, bad, gap, sigma if sigma else None, args.metric, run_id, confidence, ref_note
        )
        model_name = str(fp.get("model", "unknown"))
        if "--" in model_name:
            # 去厂商前缀（Eco-Tech--Qwen3.5-... → Qwen3.5-...），对齐官方案例
            # {权重}__{硬件}__{形态}.md 命名，保证 search --model <前缀> 可命中
            model_name = model_name.split("--")[-1]
        fname = f"{model_name}__{fp.get('hardware', 'unknown')}__{fp.get('mode', 'unknown')}__neg__{param}.md"
        fname = re.sub(r"[^\w.\-]", "_", fname)
        if args.dry_run:
            print(f"[dry-run] 将写入 {out_dir / fname}（劣化 {gap * 100:.1f}%）")
            written.append(fname)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / fname).write_text(md, encoding="utf-8")
        written.append(fname)
        if confidence == "low":
            lowconf.append(fname)

    print(f"[done] retired={len(retired)} written={len(written)} skipped={len(skipped)}")
    for f in written:
        print(f"  + {f}")
    for p, why in skipped:
        print(f"  - 跳过 {p}: {why}")
    if args.dry_run and written:
        print("[dry-run] 未落盘；去掉 --dry-run 实际写入")
        return 0
    if not written and skipped:
        print("[hint] 全部跳过：如确需入库用 --force（条目标 confidence: low）")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
