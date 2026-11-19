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

"""Ownership: script. Metric gates for agent-mode PD/standard run directories.

Two read-only subcommands (never writes into the run dir):

``breakdown``
    Decompose the recorded fitness into its four weighted terms and report the
    share taken by the two latency terms.  Below ``--degraded-share`` (default
    10%) the objective has collapsed onto throughput and every further round is
    ranking noise; the workflow must not claim a latency win.

``compare``
    Estimate the noise floor from repeated measurements of the same arm, derive
    how many samples per arm a target effect needs, and classify the observed
    difference as ``real_gain`` / ``within_noise`` / ``insufficient``.

Both subcommands only use the structured artefacts already on disk
(``results.round-N.json``, ``context.json``, ``config*.toml``), so they can run
before, during and after a search without touching the deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# run 目录发现逻辑与 check_dimensions 等脚本共享，抽到 optix_assistant_common
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optix_assistant_common import extract_noise_band, find_run_dir, validate_noise_sigma  # noqa: E402

#: Fitness weights used by ``PerformanceTuner`` (they sum to 1).
WEIGHT_GEN = 0.4
WEIGHT_TTFT = 0.2
WEIGHT_TPOT = 0.3
WEIGHT_SUCC = 0.1

#: ``d2`` factors: expected range of n standard-normal draws, used to turn an
#: observed range into a sigma estimate when only a handful of samples exist.
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

#: two-sided normal quantiles for the usual alpha levels
Z_ALPHA = {0.10: 1.644854, 0.05: 1.959964, 0.02: 2.326348, 0.01: 2.575829}
#: one-sided normal quantiles for the usual power levels
Z_POWER = {0.80: 0.841621, 0.90: 1.281552, 0.95: 1.644854}

DEFAULT_GEN_TARGET = 5300.0
DEFAULT_DELTA = 0.01
DEFAULT_DEGRADED_SHARE = 0.10

#: 同配置重复组的相对 σ 超过这个量级时，说明该组根本不是"重复测量"
#: （典型情形：两个实验配置组的 params 恰好完全一致，被当成一组），不能拿来当噪声底。
#: 阈值可由 ``--max-plausible-sigma`` 覆盖。
MAX_PLAUSIBLE_SIGMA = 0.05

#: ``|rho(fitness, 吞吐)|`` 超过该值时，fitness 排名与吞吐排名等价（延迟信息已丢失）。
DEFAULT_RANK_COLLAPSE_RHO = 0.9


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_toml_scalars(path: Path, keys: Iterable[str]) -> Dict[str, float]:
    """Read a few top-level numeric keys without pulling in a TOML dependency."""
    wanted = set(keys)
    found: Dict[str, float] = {}
    if not path.exists():
        return found
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]").strip()
            continue
        if "=" not in line or section:
            # 只取顶层标量：section 内的同名键不属于顶层口径
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name not in wanted:
            continue
        try:
            found[name] = float(value.split("#")[0].strip())
        except ValueError:
            continue
    return found


def read_toml_section_scalars(path: Path, sections: Sequence[str], keys: Iterable[str]) -> Dict[str, float]:
    """读取指定 ``[section]`` 下的数值键（浅层：section 名精确匹配，忽略嵌套子表）。

    PD 配置的 SLO 在 ``[pd_disagg.<phase>]`` 段内（fitness 按 pd_phase 的
    ``PerformanceTuner`` 消费阶段键），read_toml_scalars 只取顶层标量会直接跳过
    ——breakdown 对这类配置拿不到 SLO，直接报 cannot resolve 并挡住轮次。
    """
    wanted = set(keys)
    wanted_sections = set(sections)
    found: Dict[str, float] = {}
    if not path.exists():
        return found
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]").strip()
            continue
        if "=" not in line or section not in wanted_sections:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name not in wanted:
            continue
        try:
            found[name] = float(value.split("#")[0].strip())
        except ValueError:
            continue
    return found


def parse_rounds(spec: str) -> Optional[set]:
    """``"1-5,10"`` -> {1..5, 10}; empty -> None (all rounds)."""
    if not spec:
        return None
    rounds: set = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            rounds.update(range(int(lo), int(hi) + 1))
        else:
            rounds.add(int(chunk))
    return rounds or None


def load_trials(run_dir: Path, rounds: Optional[set] = None) -> List[Dict[str, Any]]:
    """Successful trials in round order (failed/early-exit trials are skipped)."""
    trials: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("results.round-*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        rnd = int(payload.get("round") or path.stem.split("-")[-1])
        if rounds is not None and rnd not in rounds:
            continue
        for trial in payload.get("trials") or []:
            perf = trial.get("performance") or {}
            fitness = trial.get("fitness")
            gen = perf.get("generate_speed")
            if trial.get("error") or fitness is None or gen in (None, 0) or trial.get("early_exit"):
                continue
            trials.append(
                {
                    "round": rnd,
                    "candidate_id": trial.get("candidate_id") or f"round-{rnd}",
                    "params": trial.get("params") or {},
                    "fitness": float(fitness),
                    "generate_speed": float(gen),
                    "ttft": float(perf.get("time_to_first_token") or 0.0),
                    "tpot": float(perf.get("time_per_output_token") or 0.0),
                    "success_rate": float(perf.get("success_rate") or 0.0),
                    "elapsed_seconds": float(trial.get("elapsed_seconds") or 0.0),
                }
            )
    return trials


def load_context(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "context.json"
    return read_json(path) if path.exists() else {}


def resolve_slo(run_dir: Path, config: Optional[Path]) -> Tuple[Optional[float], Optional[float], str]:
    """ttft_slo/tpot_slo from an explicit config, else from the run's config*.toml."""
    candidates: List[Path] = []
    if config is not None:
        candidates.append(config)
    else:
        context = load_context(run_dir)
        source = ((context.get("source") or {}).get("config_toml")) or ""
        if source:
            candidates.append(Path(source))
        candidates.extend(sorted(run_dir.glob("config*.toml")))
    for path in candidates:
        values = read_toml_scalars(path, ("ttft_slo", "tpot_slo"))
        if "tpot_slo" in values:
            return values.get("ttft_slo"), values.get("tpot_slo"), str(path)
        # PD 场景（检视 202103638）：fitness 消费 [pd_disagg.<phase>].tpot_slo，
        # 顶层键不生效——顶层缺失时按阶段键解析（decode 优先），否则 breakdown
        # 拿不到 SLO 会直接挡住轮次。
        for phase in ("decode", "prefill"):
            phase_values = read_toml_section_scalars(path, (f"pd_disagg.{phase}",), ("ttft_slo", "tpot_slo"))
            if "tpot_slo" in phase_values:
                merged = dict(values)
                merged.update(phase_values)
                return merged.get("ttft_slo"), merged.get("tpot_slo"), f"{path} [pd_disagg.{phase}]"
        if values:
            return values.get("ttft_slo"), values.get("tpot_slo"), str(path)
    return None, None, ""


#: ``PerformanceTuner.__init__`` 的默认参数：config 没写这些键时 optix 用的就是它们。
OBJECTIVE_KEYS = {
    "generate_speed_target": "gen_target",
    "ttft_penalty": "ttft_penalty",
    "tpot_penalty": "tpot_penalty",
}
OBJECTIVE_DEFAULTS = {"gen_target": 5300.0, "ttft_penalty": 3.0, "tpot_penalty": 3.0}


def resolve_objective(run_dir: Path, config: Optional[Path]) -> Tuple[Dict[str, float], str]:
    """目标函数的全部系数：与 ``PerformanceTuner`` 同源，缺失才回落到类默认值。

    只读 SLO 不读 penalty/target 会让四分量拆解与真实 fitness 对不上，
    拆出来的占比也就不能用来判断"延迟项是否已失效"。
    """
    objective = dict(OBJECTIVE_DEFAULTS)
    candidates: List[Path] = []
    if config is not None:
        candidates.append(config)
    else:
        context = load_context(run_dir)
        source = ((context.get("source") or {}).get("config_toml")) or ""
        if source:
            candidates.append(Path(source))
        candidates.extend(sorted(run_dir.glob("config*.toml")))
    used = ""
    for path in candidates:
        values = read_toml_scalars(path, tuple(OBJECTIVE_KEYS))
        if values:
            for key, name in OBJECTIVE_KEYS.items():
                if key not in values:
                    continue
                if values[key] < 0:
                    print(
                        f"WARNING: {path}: {key}={values[key]!r} 为负值，惩罚项/目标值不接受负数，已忽略并沿用默认值",
                        file=sys.stderr,
                    )
                    continue
                # >= 0：显式写 0 是合法配置（如 tpot_penalty = 0.0），> 0 会把它当成
                # 「未配置」而沿用默认值，四分量拆解与真实 fitness 对不上。
                objective[name] = values[key]
            used = str(path)
            break
    return objective, used


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def breakdown_trial(
    trial: Dict[str, Any],
    ttft_slo: float,
    tpot_slo: float,
    gen_target: float,
    ttft_penalty: float,
    tpot_penalty: float,
) -> Dict[str, float]:
    cost_gen = WEIGHT_GEN * (gen_target / trial["generate_speed"])
    cost_ttft = WEIGHT_TTFT * safe_exp(ttft_penalty * (trial["ttft"] / ttft_slo - 1)) if ttft_slo > 0 else 0.0
    cost_tpot = WEIGHT_TPOT * safe_exp(tpot_penalty * (trial["tpot"] / tpot_slo - 1)) if tpot_slo > 0 else 0.0
    sr = trial["success_rate"]
    if sr > 0:
        cost_succ = WEIGHT_SUCC * safe_exp(5.0 * (1.0 / sr - 1.0))
    else:
        cost_succ = float("inf")
    total = cost_gen + cost_ttft + cost_tpot + cost_succ
    return {
        "gen": cost_gen,
        "ttft": cost_ttft,
        "tpot": cost_tpot,
        "succ": cost_succ,
        "total": total,
        "delay_share": (cost_ttft + cost_tpot) / total if total else 0.0,
    }


def rank(values: Sequence[float]) -> List[float]:
    """平均秩，用于 Spearman（并列取平均秩）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """排名相关性；样本 < 3 或某侧为常数时返回 None（无法判断）。"""
    if len(left) != len(right) or len(left) < 3:
        return None
    rank_left, rank_right = rank(left), rank(right)
    try:
        mean_left = statistics.fmean(rank_left)
        mean_right = statistics.fmean(rank_right)
        covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(rank_left, rank_right))
        variance_left = sum((a - mean_left) ** 2 for a in rank_left)
        variance_right = sum((b - mean_right) ** 2 for b in rank_right)
    except (TypeError, ValueError):
        return None
    if variance_left <= 0 or variance_right <= 0:
        return None
    return covariance / math.sqrt(variance_left * variance_right)


def cmd_breakdown(args: argparse.Namespace) -> int:
    run_dir = find_run_dir(args.run_dir)
    trials = load_trials(run_dir, parse_rounds(args.rounds))
    if not trials:
        raise SystemExit("ERROR: no successful trial found in the given rounds")
    config_path = Path(args.config) if args.config else None
    objective, objective_source = resolve_objective(run_dir, config_path)
    ttft_slo, tpot_slo, source = resolve_slo(run_dir, config_path)
    ttft_slo = float(args.ttft_slo or ttft_slo or 0.0)
    tpot_slo = float(args.tpot_slo or tpot_slo or 0.0)
    if tpot_slo <= 0:
        raise SystemExit(
            "ERROR: cannot resolve tpot_slo; pass --tpot-slo or --config "
            "（PD 场景按 [pd_disagg.<phase>].tpot_slo 解析，确认轮次 config 带阶段 SLO）"
        )
    # ttft_slo 缺失或 <=0 合法：TTFT 不作为门禁约束，fitness 拆解时该项按 0 计
    gen_target = float(args.gen_target or objective["gen_target"])
    ttft_penalty = float(args.ttft_penalty if args.ttft_penalty is not None else objective["ttft_penalty"])
    tpot_penalty = float(args.tpot_penalty if args.tpot_penalty is not None else objective["tpot_penalty"])

    rows: List[Dict[str, Any]] = []
    for trial in trials:
        parts = breakdown_trial(trial, ttft_slo, tpot_slo, gen_target, ttft_penalty, tpot_penalty)
        rows.append({"trial": trial, "parts": parts})

    # success_rate 缺失或为 0 时 succ 惩罚项溢出（total=inf、delay_share=0），
    # 这样的行不能参与占比/排名判定，否则真实数据里的失败 trial 会被误判成
    # 「低延迟占比」，并把 degraded 判定拖向错误结论。
    usable_rows = [r for r in rows if math.isfinite(r["parts"]["total"])]
    if not usable_rows:
        raise SystemExit("ERROR: 全部 trial 不可用（success_rate<=0 或惩罚项溢出），fitness 拆解无法进行")
    if len(usable_rows) < len(rows):
        print(
            f"WARNING: {len(rows) - len(usable_rows)}/{len(rows)} 个 trial 不可用"
            "（success_rate<=0 或惩罚项溢出），已从拆解判定中剔除",
            file=sys.stderr,
        )

    # 延迟项是否还有区分力：占比只是代理指标，真正的判据是"排名是否等价于吞吐"
    rho_gen = spearman(
        [r["trial"]["fitness"] for r in usable_rows], [r["trial"]["generate_speed"] for r in usable_rows]
    )
    rho_tpot = spearman([r["trial"]["fitness"] for r in usable_rows], [r["trial"]["tpot"] for r in usable_rows])
    low_share = [r for r in usable_rows if r["parts"]["delay_share"] < args.degraded_share]
    rank_collapsed = rho_gen is not None and abs(rho_gen) >= args.rank_collapse_rho
    degraded = low_share and (rho_gen is None or rank_collapsed)

    payload = {
        "run_dir": str(run_dir),
        "slo_source": source,
        "objective_source": objective_source,
        "ttft_slo": ttft_slo,
        "tpot_slo": tpot_slo,
        "gen_target": gen_target,
        "ttft_penalty": ttft_penalty,
        "tpot_penalty": tpot_penalty,
        "degraded_share": args.degraded_share,
        "rank_collapse_rho": args.rank_collapse_rho,
        "rho_fitness_vs_generate_speed": None if rho_gen is None else round(rho_gen, 4),
        "rho_fitness_vs_tpot": None if rho_tpot is None else round(rho_tpot, 4),
        "trials": [
            {
                "round": r["trial"]["round"],
                "candidate_id": r["trial"]["candidate_id"],
                "fitness": round(r["trial"]["fitness"], 6),
                "usable": math.isfinite(r["parts"]["total"]),
                "recomputed": round(r["parts"]["total"], 6) if math.isfinite(r["parts"]["total"]) else None,
                "delay_share": round(r["parts"]["delay_share"], 4) if math.isfinite(r["parts"]["total"]) else None,
                "shares": (
                    {key: round(r["parts"][key] / r["parts"]["total"], 4) for key in ("gen", "ttft", "tpot", "succ")}
                    if math.isfinite(r["parts"]["total"])
                    else None
                ),
            }
            for r in rows
        ],
        "low_delay_share_rounds": [r["trial"]["round"] for r in low_share],
        "degraded": bool(degraded),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        # JSON 模式也必须带上退出码：否则 --json 会成为绕过门禁的途径
        return 3 if degraded else 0

    print(f"run-dir      : {run_dir}")
    print(f"SLO 来源     : {source or '(cli)'}  ttft_slo={ttft_slo}s tpot_slo={tpot_slo}s")
    print(f"目标函数来源 : {objective_source or '(PerformanceTuner 默认值)'}")
    print(f"weights      : gen={WEIGHT_GEN} ttft={WEIGHT_TTFT} tpot={WEIGHT_TPOT} succ={WEIGHT_SUCC}")
    print(f"gen target   : {gen_target}   penalties: ttft={ttft_penalty} tpot={tpot_penalty}")
    print()
    header = f"{'round':>5} {'candidate':<26} {'fitness':>9} {'recalc':>9} {'gen%':>6} {'ttft%':>6} {'tpot%':>6} {'succ%':>6} {'delay%':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        t, p = r["trial"], r["parts"]
        if not math.isfinite(p["total"]):
            print(
                f"{t['round']:>5} {t['candidate_id'][:26]:<26} {t['fitness']:>9.4f} "
                f"{'unusable':>9}  （success_rate<=0 或惩罚项溢出，已剔除）"
            )
            continue
        share = {k: (p[k] / p["total"] if p["total"] else 0.0) for k in ("gen", "ttft", "tpot", "succ")}
        print(
            f"{t['round']:>5} {t['candidate_id'][:26]:<26} {t['fitness']:>9.4f} {p['total']:>9.4f} "
            f"{share['gen'] * 100:>5.1f}% {share['ttft'] * 100:>5.1f}% {share['tpot'] * 100:>5.1f}% "
            f"{share['succ'] * 100:>5.1f}% {p['delay_share'] * 100:>6.1f}%"
        )
    print()
    worst = min(usable_rows, key=lambda r: r["parts"]["delay_share"])
    print(
        f"延迟项占比最低: R{worst['trial']['round']} {worst['parts']['delay_share'] * 100:.1f}%"
        f"  （阈值 {args.degraded_share * 100:.0f}%，低于阈值的轮次 {len(low_share)}/{len(usable_rows)}）"
    )
    rho_text = "样本不足" if rho_gen is None else f"{rho_gen:.2f}"
    tpot_text = "样本不足" if rho_tpot is None else f"{rho_tpot:.2f}"
    print(
        f"排名一致性   : ρ(fitness, 吞吐)={rho_text}  ρ(fitness, TPOT)={tpot_text}"
        f"  （|ρ| ≥ {args.rank_collapse_rho:g} 视为与吞吐排名等价）"
    )
    if degraded:
        print(
            "VERDICT: DEGRADED —— 延迟项在总分中占比低于阈值，且 fitness 排名与吞吐排名等价；"
            "本 run 的 fitness 只反映吞吐，不得据此宣称延迟收益。"
            "先按环境准备流程与用户确认 SLO / 目标吞吐量级。"
        )
        return 3
    if low_share:
        print(
            "VERDICT: WEAK —— 延迟项占比偏低，但排名仍在延迟上有区分（ρ 未达等价阈值）；"
            "结论需同时给出延迟与吞吐，不能只报吞吐。"
        )
        return 0
    print("VERDICT: OK —— 延迟项占比达标，fitness 可用于多目标排名")
    return 0


def sigma_from_range(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    spread = max(values) - min(values)
    return spread / D2.get(n, 3.078)


def sigma_mad(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    return 1.4826 * mad


def arm_key(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


NOISE_BAND_KEYS = {
    "fitness": ("fitness_sigma",),
    "generate_speed": ("generate_speed_sigma", "fitness_sigma"),
    "tpot": ("tpot_sigma",),
    "ttft": ("ttft_sigma",),
}


def load_noise_band(
    run_dir: Path, metric: str, max_sigma: float, ack_wide: str = ""
) -> Tuple[float, str, List[str], List[str], bool]:
    """noise_band 校准值的守门入口（compare 用；校验逻辑与导出共用 optix_assistant_common）。

    σ 是 Round 0 写入的普通 JSON 值，没有可信区间校验时，一次校准抖动或格式错误
    就能让整个 run 判不出任何差异，或反向把噪声级差异判成 real_gain。规则：
    - band/键缺失 → ``(0.0, "", [], [], False)``：noise_floor 回退同配置重复组池化
    - ``0 < σ ≤ max_sigma`` → 正常返回
    - ``σ = 0`` → 继续但 WARNING（确定性测量，需确认校准没漏测波动）
    - ``σ < 0`` 或超上限 → 默认拒绝（errors 非空，调用方 exit 2）；``--noise-ack-wide``
      人工确认后返回 ``band_acked=True``，后续判定给出 noise_dominant/unmeasurable 终止判定
    - 格式非法（字符串/列表/bool/NaN）→ 明确 ERROR 且不可 ack：数据坏了必须修

    注意：校验失败时**不得**静默退回同配置重复组池化估值继续判——band 正是被
    ``max()`` 取大的那一个，退回会把 σ 低估约两个数量级（实测 50% → 0.24%），
    把 within_noise 翻成 real_gain。
    """
    band, band_source = extract_noise_band(run_dir)
    errors: List[str] = []
    warnings: List[str] = []
    for key in NOISE_BAND_KEYS.get(metric, ()):
        if band.get(key) is None:
            continue
        source = f"{band_source}.{key}"
        check = validate_noise_sigma(band[key], max_sigma)
        if check["status"] == "error":
            if check["ackable"]:
                if ack_wide.strip():
                    warnings.append(
                        f"{source} = {band[key]!r}{check['error']}；已按 --noise-ack-wide 记录人工确认：{ack_wide.strip()}"
                    )
                    return check["sigma"], source, [], warnings, True
                errors.append(
                    f"{source} = {band[key]!r}{check['error']} —— 请重新校准，或加 --noise-ack-wide \"<依据>\" 人工确认后继续"
                )
            else:
                errors.append(
                    f"{source} = {band[key]!r}{check['error']} —— 校准数据损坏，"
                    "修复 noise_band.json 后重跑（该项不可 ack）"
                )
            return 0.0, "", errors, warnings, False
        if check["warning"]:
            warnings.append(f"{source} = {band[key]!r}{check['warning']}")
        return check["sigma"], source, [], warnings, False
    return 0.0, "", [], [], False


def noise_floor(
    run_dir: Path,
    trials: Sequence[Dict[str, Any]],
    getter,
    metric: str,
    max_plausible: float = MAX_PLAUSIBLE_SIGMA,
    band_sigma: Optional[float] = None,
    band_source: str = "",
) -> Tuple[float, str]:
    """本 run 的噪声底：优先 noise_band（经 load_noise_band 校验后注入），否则用同配置重复组池化。

    band_sigma/band_source 由 ``load_noise_band`` 校验后传入（格式/符号/上限都在
    那里把关，校验失败时 cmd_compare 直接 exit 2，不会走到这里）；band 缺失时
    回退同配置重复组池化。单样本配置组自己的极差是 0，直接拿它当 σ 会把任何差异都
    判成"显著"——噪声底必须来自"同一配置重复测"的样本。
    """
    candidates: List[Tuple[float, str]] = []
    if band_sigma is not None and band_source:
        candidates.append((band_sigma, band_source))

    pooled: List[float] = []
    skipped = 0
    groups: Dict[str, List[float]] = {}
    for trial in trials:
        groups.setdefault(arm_key(trial["params"]), []).append(getter(trial))
    for values in groups.values():
        if len(values) < 2:
            continue
        median = statistics.median(values)
        if not median:
            continue
        sigma = max(sigma_from_range(values), sigma_mad(values)) / median
        if sigma > max_plausible:
            # 组内差异远超噪声量级 —— 这组 params 相同但实际上是不同实验（或跨了配置变更），
            # 拿来估噪声会使判定阈值过宽，无法区分任何真实差异。
            skipped += 1
            continue
        pooled.append(sigma)
    if pooled:
        # 取各重复组相对 σ 的中位数：比 max 稳，不会被某一对噪声样本把门放得太宽
        note = f"同配置重复组池化（{len(pooled)} 组，取中位数）"
        if skipped:
            note += f"；{skipped} 组相对差异 > {max_plausible * 100:.0f}% 已按混入不同实验剔除"
        candidates.append((statistics.median(pooled), note))
    elif skipped:
        candidates.append((0.0, f"无可用的同配置重复组（{skipped} 组差异过大被剔除）"))

    if not candidates:
        return 0.0, "无（单样本配置组且缺 noise_band —— 结论只能是 insufficient）"
    # noise_band 来自 Round 0 的同配置连测，权威性高于事后池化；两者取大者以防低估
    value, source = max(candidates, key=lambda item: item[0])
    return value, source


def short_key(key: str) -> str:
    """长参数名的缩写：``GPU_MEMORY_UTILIZATION_decode`` -> ``GMU_decode``。"""
    parts = [part for part in key.split("_") if part]
    if len(parts) < 3:
        return key
    return "".join(part[0] for part in parts[:-1]) + "_" + parts[-1]


def describe_arm(params: Dict[str, Any], reference: Optional[Dict[str, Any]]) -> str:
    """配置组标签：``<取值>@<参数缩写>``，其余差异只报个数。

    以前把若干 ``key=value`` 拼起来再截断，长键名会把取值截掉，不同配置组落到同一个
    标签上，聚合时互相覆盖 —— 配置组一多就会丢数据。取值放最前面，截断也不会丢区分度。
    """
    if not reference:
        return "reference"
    changed = [(key, params.get(key)) for key in sorted(params) if reference.get(key) != params.get(key)]
    if not changed:
        return "(identical)"
    label = f"{changed[0][1]}@{short_key(changed[0][0])}"
    if len(changed) > 1:
        label += f" (+{len(changed) - 1})"
    return label


def cmd_compare(args: argparse.Namespace) -> int:
    run_dir = find_run_dir(args.run_dir)
    trials = load_trials(run_dir, parse_rounds(args.rounds))
    if len(trials) < 2:
        raise SystemExit("ERROR: need at least 2 successful trials")

    metric_getter = {
        "fitness": lambda t: t["fitness"],
        "generate_speed": lambda t: t["generate_speed"],
        "tpot": lambda t: t["tpot"],
        "ttft": lambda t: t["ttft"],
    }[args.metric]

    # grouping: explicit arm patterns win, otherwise group by identical params
    groups: Dict[str, List[Dict[str, Any]]] = {}
    if args.arm:
        for spec in args.arm:
            name, _, pattern = spec.partition("=")
            if not pattern:
                raise SystemExit(f"ERROR: --arm expects NAME=SUBSTRING, got {spec!r}")
            # 先精确匹配 candidate_id，避免 r3 顺带命中 r7-repeat-r3 这类前缀相同的历史配置组
            matched = [t for t in trials if t["candidate_id"] == pattern]
            if not matched:
                matched = [t for t in trials if pattern in t["candidate_id"]]
            if not matched:
                raise SystemExit(f"ERROR: arm {name!r} matched no trial candidate_id containing {pattern!r}")
            groups[name] = matched
    else:
        reference: Optional[Dict[str, Any]] = None
        for trial in trials:
            key = arm_key(trial["params"])
            if key not in groups:
                groups[key] = []
                if reference is None:
                    reference = trial["params"]
            groups[key].append(trial)
        renamed: Dict[str, List[Dict[str, Any]]] = {}
        used: Dict[str, int] = {}
        for index, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(t["round"] for t in kv[1]))):
            label = "reference" if index == 0 else describe_arm(members[0]["params"], reference)
            seen = used.get(label, 0) + 1
            used[label] = seen
            if seen > 1:
                label = f"{label}#{seen}"
            renamed[label] = members
        groups = renamed

    band_sigma, band_source, noise_errors, noise_warnings, band_acked = load_noise_band(
        run_dir, args.metric, args.max_plausible_sigma, getattr(args, "noise_ack_wide", "")
    )
    if noise_errors:
        for message in noise_errors:
            print(f"ERROR: {message}")
        print(
            "FAIL: noise_band 校准值异常 —— 不能静默退回池化估值继续判"
            "（band 正是被 max() 取大的那个，退回会把 σ 低估两个数量级、翻转结论）"
        )
        return 2
    for message in noise_warnings:
        print(f"WARNING: {message}", file=sys.stderr)
    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "metric": args.metric,
        "arms": {},
        "pairs": [],
        "noise_acked": band_acked,
    }
    floor_sigma, floor_source = noise_floor(
        run_dir,
        trials,
        metric_getter,
        args.metric,
        args.max_plausible_sigma,
        band_sigma=band_sigma if band_source else None,
        band_source=band_source,
    )
    report["sigma_floor"] = round(floor_sigma, 6)
    report["sigma_floor_source"] = floor_source
    for name, members in groups.items():
        values = [metric_getter(t) for t in members]
        median = statistics.median(values)
        sigma = max(sigma_from_range(values), sigma_mad(values))
        arm_sigma_rel = sigma / median if median else 0.0
        report["arms"][name] = {
            "n": len(values),
            "rounds": [t["round"] for t in members],
            "params": members[0]["params"],
            "median": round(median, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "relative_sigma": round(arm_sigma_rel, 6),
            "relative_sigma_used": round(max(arm_sigma_rel, floor_sigma), 6),
            "relative_range": round((max(values) - min(values)) / median, 6) if median else 0.0,
        }

    z_alpha = Z_ALPHA.get(args.alpha)
    z_power = Z_POWER.get(args.power)
    if z_alpha is None:
        raise SystemExit(f"ERROR: unsupported --alpha {args.alpha}; choose from {sorted(Z_ALPHA)}")
    if z_power is None:
        raise SystemExit(f"ERROR: unsupported --power {args.power}; choose from {sorted(Z_POWER)}")

    budgets = load_context(run_dir).get("budgets") or {}
    budget_raw = budgets.get("max_trials")
    try:
        budget_max = int(budget_raw) if budget_raw is not None else None
    except (TypeError, ValueError):
        budget_max = None

    names = list(report["arms"])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = report["arms"][names[i]], report["arms"][names[j]]
            if not left["median"] or not right["median"]:
                continue
            diff = right["median"] - left["median"]
            rel_diff = diff / left["median"]
            sigma_rel = math.hypot(left["relative_sigma_used"], right["relative_sigma_used"])
            # hypot 已合并两个配置组的 σ（等方差时=√2σ），标准式 n=(z)²(σ_a²+σ_b²)/Δ² 的系数
            # 在这里就是 (z)²；再乘 2 会把样本量需求翻倍（σ=Δ=0.5% 时 32→16）
            target = abs(args.delta)
            n_required = math.ceil(((z_alpha + z_power) ** 2) * (sigma_rel**2) / (target**2)) if sigma_rel else 2
            n_have = min(left["n"], right["n"])

            # paired estimate when the two arms alternate in adjacent rounds
            paired = _paired_sigma(groups[names[i]], groups[names[j]], metric_getter)
            n_paired = None
            if paired is not None and paired > 0:
                n_paired = math.ceil(((z_alpha + z_power) ** 2) * (paired**2) / (target**2))

            # 终止判定（检视 202107780）：人工确认接受的过宽 σ 与超预算的补测需求
            # 都不是可执行指令——给 noise_dominant/unmeasurable 让工作流终止并在
            # final_report 记录，而不是按 required_repetitions 补测几万次。
            if band_acked and 2 * sigma_rel >= target:
                verdict = "noise_dominant"
            elif abs(rel_diff) < 2 * sigma_rel:
                verdict = "within_noise"
            elif budget_max is not None and (n_paired or n_required) > budget_max:
                verdict = "unmeasurable"
            elif n_have < (n_paired or n_required):
                verdict = "insufficient"
            else:
                verdict = "real_gain"
            report["pairs"].append(
                {
                    "left": names[i],
                    "right": names[j],
                    "relative_diff": round(rel_diff, 6),
                    "relative_sigma": round(sigma_rel, 6),
                    "n_have": n_have,
                    "n_required": n_required,
                    "paired_relative_sigma": round(paired, 6) if paired is not None else None,
                    "n_required_paired": n_paired,
                    "target_delta": target,
                    "budget_max": budget_max,
                    "verdict": verdict,
                }
            )

    if not report["pairs"]:
        # 全部配对被跳过（配置组中位数为 0）或配置组不足 2 时，静默 exit 0 会让调用方把
        # 「无法比较」当成「比较通过」。显式给 verdict，并以不可比较退出。
        zero_arms = ", ".join(name for name, arm in report["arms"].items() if not arm["median"])
        if len(report["arms"]) < 2:
            reason = f"可用配置组不足 2（共 {len(report['arms'])} 个）：params 完全相同的 trial 会归入同一组"
        else:
            reason = (
                f"配置组中位数为 0（{zero_arms}）：--metric tpot/ttft 时缺失字段的 trial 会被记 0，"
                "中位数为 0 的配置组不参与配对"
            )
        report["verdict"] = "incomparable"
        report["reason"] = reason
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 3
        print(f"run-dir : {run_dir}")
        print(f"VERDICT: 无法比较 —— {reason}")
        return 3

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if any(p["verdict"] in ("noise_dominant", "unmeasurable") for p in report["pairs"]):
            return 4
        return 3 if any(p["verdict"] == "within_noise" for p in report["pairs"]) else 0

    print(f"run-dir : {run_dir}")
    print(f"metric  : {args.metric}   target delta: {args.delta * 100:.2f}%   alpha={args.alpha} power={args.power}")
    print(f"noise   : σ_floor={floor_sigma * 100:.2f}%  来源：{floor_source}")
    if not args.arm:
        print("配置组标签  : <取值>@<参数缩写>，相对 reference 的首个差异；(+N) = 另有 N 个参数不同")
    print()
    header = f"{'arm':<34} {'n':>3} {'median':>10} {'range%':>8} {'sigma%':>8}  rounds"
    print(header)
    print("-" * (len(header) + 12))
    for name, arm in report["arms"].items():
        print(
            f"{name[:34]:<34} {arm['n']:>3} {arm['median']:>10.4f} "
            f"{arm['relative_range'] * 100:>7.2f}% {arm['relative_sigma_used'] * 100:>7.2f}%  {arm['rounds']}"
        )
    print()
    for pair in report["pairs"]:
        print(
            f"{pair['left'][:26]:<26} vs {pair['right'][:26]:<26} diff {pair['relative_diff'] * 100:+.2f}%  "
            f"sigma {pair['relative_sigma'] * 100:.2f}%  n={pair['n_have']}/need {pair['n_required']} "
            f"(paired need {pair['n_required_paired']})  => {pair['verdict']}"
        )
    terminal = [p for p in report["pairs"] if p["verdict"] in ("noise_dominant", "unmeasurable")]
    if terminal:
        print()
        for pair in terminal:
            if pair["verdict"] == "noise_dominant":
                print(
                    f"VERDICT: {pair['left']} vs {pair['right']} —— noise_dominant："
                    f"2σ（{pair['relative_sigma'] * 100:.2f}%）≥ 目标效应"
                    f"（{pair['target_delta'] * 100:.2f}%），不可判定"
                )
            else:
                print(
                    f"VERDICT: {pair['left']} vs {pair['right']} —— unmeasurable：需要 "
                    f"每种配置需 {pair['n_required']} 次，超过 run 预算 max_trials={pair['budget_max']}，不可判定"
                )
        print("本轮噪声过大、不可判定 —— 建议终止 run 并在 final_report 记录，不要再按 n_required 补测。")
        return 4
    if any(p["verdict"] == "within_noise" for p in report["pairs"]):
        print()
        print("VERDICT: 存在 within_noise 判定 —— 结论必须写「不可分辨」，禁止写「提升 x%」。")
        return 3
    return 0


def _paired_sigma(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]], getter) -> Optional[float]:
    """Relative sigma of paired differences when the two arms interleave."""
    by_round = {t["round"]: getter(t) for t in left}
    diffs: List[float] = []
    for trial in right:
        previous = by_round.get(trial["round"] - 1)
        if previous is None:
            continue
        if not previous:
            continue
        diffs.append((getter(trial) - previous) / previous)
    if len(diffs) < 2:
        return None
    return statistics.stdev(diffs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fitness breakdown and noise gate for agent-mode runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--run-dir", default="", help="run directory (default: unique dir under CWD)")
    common_parent.add_argument("--rounds", default="", help="round filter, e.g. 1-5,10")
    common_parent.add_argument("--json", action="store_true", help="machine-readable output")

    bd = sub.add_parser("breakdown", parents=[common_parent], help="decompose fitness into four terms")
    bd.add_argument("--config", default="", help="config.toml holding ttft_slo/tpot_slo/penalties")
    bd.add_argument("--ttft-slo", type=float, default=0.0, help="override ttft_slo (seconds)")
    bd.add_argument("--tpot-slo", type=float, default=0.0, help="override tpot_slo (seconds)")
    bd.add_argument(
        "--gen-target",
        type=float,
        default=None,
        help=f"generate_speed target (default: config value, else {DEFAULT_GEN_TARGET})",
    )
    bd.add_argument("--ttft-penalty", type=float, default=None, help="ttft penalty coefficient (default: config/3.0)")
    bd.add_argument("--tpot-penalty", type=float, default=None, help="tpot penalty coefficient (default: config/3.0)")
    bd.add_argument(
        "--degraded-share",
        type=float,
        default=DEFAULT_DEGRADED_SHARE,
        help="latency share below which the objective counts as collapsed",
    )
    bd.add_argument(
        "--rank-collapse-rho",
        type=float,
        default=DEFAULT_RANK_COLLAPSE_RHO,
        help="|rho(fitness, throughput)| above which the ranking equals a throughput ranking",
    )
    bd.set_defaults(func=cmd_breakdown)

    cp = sub.add_parser("compare", parents=[common_parent], help="noise floor, sample size and verdict")
    cp.add_argument("--metric", default="generate_speed", choices=["fitness", "generate_speed", "tpot", "ttft"])
    cp.add_argument("--arm", action="append", help="NAME=CANDIDATE_ID_SUBSTRING; repeatable (default: group by params)")
    cp.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="effect worth concluding (relative)")
    cp.add_argument("--alpha", type=float, default=0.05)
    cp.add_argument("--power", type=float, default=0.80)
    cp.add_argument(
        "--max-plausible-sigma",
        type=float,
        default=MAX_PLAUSIBLE_SIGMA,
        help="relative sigma cap: same-params groups above it are mixed experiments; noise_band values above "
        "it are calibration failures (re-calibrate or pass --noise-ack-wide)",
    )
    cp.add_argument(
        "--noise-ack-wide",
        default="",
        help="noise_band σ 超上限/为负时的人工确认接受依据（确认人 + 原因）；"
        "确认后仍不可判定时给出 noise_dominant/unmeasurable 终止判定（退出码 4）",
    )
    cp.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
