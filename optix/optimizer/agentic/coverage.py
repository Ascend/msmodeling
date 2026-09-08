"""Search-space coverage checks for agent-mode convergence.

Issue 1.3 / requirement 1: convergence must not fire on result trend alone —
an engine-level blind spot (e.g. preset ``enforce_eager=true`` disabling CUDA
Graph) can keep every candidate on a fake plateau. Convergence is only allowed
when BOTH the trend condition (no improvement for N rounds) and the coverage
condition hold:

- at least ``min_rounds`` rounds executed and ``min_trials`` trials completed;
- every ``flip_required`` engine fixed param has been verified (the flipped
  value was actually tested by some candidate).

This module is intentionally stdlib-only so it can be unit-tested hermetically.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def _load_trials(run_dir: Path) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    run_dir = Path(run_dir)
    for results_path in sorted(run_dir.glob("results.round-*.json")):
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        trials.extend(data.get("trials", []))
    return trials


def engine_fixed_param_verification(run_dir: Path, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return per-param verification status for flip_required engine fixed params.

    A param is verified when any completed trial carried a value different from
    the preset default (e.g. ``enforce_eager=false`` vs preset ``true``).
    """
    fixed = context.get("engine_fixed_params") or []
    trials = _load_trials(run_dir)
    tested_values: Dict[str, set] = {}
    for trial in trials:
        for name, value in (trial.get("params") or {}).items():
            tested_values.setdefault(name, set()).add(value)

    results: List[Dict[str, Any]] = []
    for entry in fixed:
        if not entry.get("flip_required"):
            continue
        search_name = entry.get("search_name")
        default = entry.get("value")
        verified = False
        if search_name:
            tested = tested_values.get(search_name, set())
            verified = any(str(value) != str(default) for value in tested)
        results.append({**entry, "verified": verified})
    return results


def search_space_coverage(
    run_dir: Path,
    context: Dict[str, Any],
    min_rounds: int = 3,
    min_trials: int = 12,
) -> Dict[str, Any]:
    """Evaluate the coverage condition for convergence.

    Returns ``{"ok": bool, "rounds_done": int, "trials_done": int, "reasons": [...]}``.
    ``ok`` is True only when all gates pass; ``reasons`` lists every unmet gate.
    """
    trials = _load_trials(run_dir)
    rounds_done = len({(t.get("metadata") or {}).get("round_id") for t in trials})
    reasons: List[str] = []

    if rounds_done < min_rounds:
        reasons.append(f"已执行 {rounds_done} 轮 < min_rounds={min_rounds}")
    if len(trials) < min_trials:
        reasons.append(f"已完成 {len(trials)} 个 trial < min_trials={min_trials}")

    for entry in engine_fixed_param_verification(run_dir, context):
        if not entry["verified"]:
            reasons.append(
                f"引擎级固定参数 {entry.get('path')}={entry.get('value')!r} 未做反转验证"
                f"（需要 {entry.get('search_name')} 取相反值）"
            )

    return {
        "ok": not reasons,
        "rounds_done": rounds_done,
        "trials_done": len(trials),
        "reasons": reasons,
    }
