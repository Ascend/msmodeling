"""Ownership: agent. Append round summaries into summary.md."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import read_json


def _build_trend(values: List[float], label: str) -> str:
    """Build a trend arrow from a list of values."""
    if len(values) < 2:
        return f"{label}: {values[0] if values else 'N/A'}"
    direction = "↓ 改善中" if values[-1] < values[0] else "↑ 恶化中" if values[-1] > values[0] else "→ 持平"
    arrows = " → ".join(
        f"{v:.1f}" if isinstance(v, float) and v < 1e6 else f"{v:.0f}" if isinstance(v, float) else str(v)
        for v in values
    )
    return f"{label}: {arrows} {direction}"


def _load_historical_trials(run_dir: Path, current_round: int) -> List[Dict[str, Any]]:
    """Load all completed rounds' trials."""
    all_trials = []
    for r in range(1, current_round + 1):
        results_path = run_dir / f"results.round-{r}.json"
        if results_path.exists():
            data = read_json(results_path)
            all_trials.extend(data.get("trials", []))
    return all_trials


def main():
    parser = argparse.ArgumentParser(description="Summarize an agent optimizer round.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round", required=True, type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / f"results.round-{args.round}.json"
    results = read_json(results_path) if results_path.exists() else {"trials": []}
    trials = sorted(
        [t for t in results.get("trials", []) if t.get("fitness", float("inf")) < float("inf")],
        key=lambda item: item.get("fitness", float("inf")),
    )

    # Load historical for trend
    historical = _load_historical_trials(run_dir, args.round)
    best_per_round = {}
    best_trial_per_round = {}
    for t in historical:
        r = t.get("metadata", {}).get("round_id", args.round)
        best_per_round.setdefault(r, float("inf"))
        if t.get("fitness", float("inf")) < best_per_round[r]:
            best_per_round[r] = t["fitness"]
            best_trial_per_round[r] = t

    # Build trend lines
    sorted_bests = [best_per_round[r] for r in sorted(best_per_round)]
    lines = [
        f"## Round {args.round}",
        "",
        "### 本轮结果",
        "| 候选 | fitness | throughput | TTFT | TPOT | SLO | 状态 |",
        "|------|---------|------------|------|------|-----|------|",
    ]
    for t in trials:
        perf = t.get("performance", {}) or {}
        fail_info = t.get("failure", {}) or {}
        state_label = (
            "✅" if t.get("fitness", float("inf")) < float("inf") else f"❌ {fail_info.get('category', 'failed')}"
        )
        slo_label = {False: "✅ 达标", True: "❌ 违例"}.get(t.get("slo_violated"), "-")
        lines.append(
            f"| {t.get('candidate_id')} | {t.get('fitness', 'inf')} "
            f"| {perf.get('generate_speed', '-')} tok/s "
            f"| {perf.get('time_to_first_token', '-')}ms "
            f"| {perf.get('time_per_output_token', '-')}ms "
            f"| {slo_label} "
            f"| {state_label} |"
        )

    lines.extend(
        [
            "",
            "### 趋势（自 Round 1 起）",
            _build_trend(sorted_bests, "- best_fitness"),
        ]
    )

    # Add throughput trend if available (use best candidate per round)
    throughputs = []
    for r in sorted(best_per_round):
        best_t = best_trial_per_round.get(r)
        if best_t:
            tp = (best_t.get("performance", {}) or {}).get("generate_speed")
            if tp:
                throughputs.append(tp)
    if throughputs:
        lines.append(_build_trend(throughputs, "- best_throughput (tok/s)"))

    # Best candidate
    if trials:
        best = trials[0]
        lines.extend(
            [
                "",
                "### 本轮最佳",
                f"- 候选: {best.get('candidate_id')}",
                f"- fitness: {best.get('fitness')}",
                f"- 参数: {json.dumps(best.get('params', {}))}",
            ]
        )

    # Hard-constraint view: best SLO-compliant trial by throughput (issue 1.2 —
    # fitness may be dominated by the throughput term and mis-rank violating configs)
    compliant = [t for t in trials if t.get("slo_violated") is False]
    if compliant:
        best_compliant = max(
            compliant,
            key=lambda t: (t.get("performance", {}) or {}).get("generate_speed", 0) or 0,
        )
        perf = best_compliant.get("performance", {}) or {}
        lines.extend(
            [
                "",
                "### 达标最优（硬约束 TPOT/TTFT）",
                f"- 候选: {best_compliant.get('candidate_id')}",
                f"- 吞吐: {perf.get('generate_speed')} tok/s",
                f"- TPOT: {perf.get('time_per_output_token')}ms | TTFT: {perf.get('time_to_first_token')}ms",
                f"- fitness: {best_compliant.get('fitness')}",
            ]
        )
    else:
        lines.extend(["", "### 达标最优（硬约束 TPOT/TTFT）", "- 本轮无 SLO 达标候选，导出请以 fitness 兜底并人工复核"])

    # Failure analysis — read structured `failure` block (raw logs are persisted
    # by the optix executor, never inlined into results)
    failed = [t for t in results.get("trials", []) if t.get("fitness", float("inf")) >= float("inf")]
    if failed:
        lines.extend(
            [
                "",
                "### 失败分析",
            ]
        )
        for t in failed:
            f = t.get("failure", {}) or {}
            offending = f.get("offending_params") or {}
            off_str = ", ".join(f"{k}={v}" for k, v in offending.items()) or "-"
            lines.append(
                f"- {t.get('candidate_id')}: {f.get('category', 'unknown')}/"
                f"{f.get('sub_category', '')} — suggested: {f.get('suggested_action', '-')} "
                f"| offending: {off_str}"
            )

    lines.extend(
        [
            "",
            "### 下轮方向建议",
            "<!-- agent must fill this section before writing candidates.round-N+1.json -->",
        ]
    )

    # Append to cumulative summary.md
    summary_path = run_dir / "summary.md"
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "# Agent Optimizer Summary\n\n"
    summary_path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
