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

"""Search-space coverage checks for agent-mode convergence.

Convergence must not fire on result trend alone — it is only allowed when BOTH
the trend condition (no improvement for N rounds) and the coverage condition
hold:

- at least ``min_rounds`` rounds executed and ``min_trials`` trials completed.

Agent mode runs exactly one candidate per round (rounds ≡ trials), so the two
gates are equivalent and the larger one effectively wins. Only successful
trials (``error is None`` and finite ``fitness``) count, matching
``AgentOptimizer._scan_completed_rounds``.

This module is intentionally stdlib-only so it can be unit-tested hermetically.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _is_successful_trial(trial: Dict[str, Any]) -> bool:
    """与 orchestrator._scan_completed_rounds 同口径：无 error 且 fitness 为有限数值。"""
    if trial.get("error") is not None:
        return False
    fitness = trial.get("fitness")
    return isinstance(fitness, (int, float)) and math.isfinite(fitness)


def _load_trials(run_dir: Path) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    run_dir = Path(run_dir)
    for results_path in sorted(run_dir.glob("results.round-*.json")):
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        trials.extend(t for t in data.get("trials", []) if _is_successful_trial(t))
    return trials


def search_space_coverage(
    run_dir: Path,
    min_rounds: int = 4,
    min_trials: int = 4,
) -> Dict[str, Any]:
    """Evaluate the coverage condition for convergence.

    Agent 模式每轮固定 1 个 trial，min_rounds 与 min_trials 两个门等价，
    取较大者生效。

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

    return {
        "ok": not reasons,
        "rounds_done": rounds_done,
        "trials_done": len(trials),
        "reasons": reasons,
    }
