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

"""Offline verification planning; execution costs are supplied by callers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import heapq
from itertools import accumulate
import math
from operator import mul
from typing import Callable


@dataclass(frozen=True)
class AcceptanceProfile:
    conditional_probabilities: tuple[float, ...]

    @classmethod
    def from_rates(cls, rates: Sequence[float] | None, num_speculative_tokens: int) -> "AcceptanceProfile":
        if (
            isinstance(num_speculative_tokens, bool)
            or not isinstance(num_speculative_tokens, int)
            or num_speculative_tokens <= 0
        ):
            raise ValueError("num_speculative_tokens must be a positive integer")
        if rates is None or isinstance(rates, (str, bytes)) or len(rates) != num_speculative_tokens:
            raise ValueError(
                f"DSpark acceptance rate count ({0 if rates is None or isinstance(rates, (str, bytes)) else len(rates)}) "
                f"must match num_speculative_tokens ({num_speculative_tokens})"
            )
        if any(
            isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in rates
        ):
            raise ValueError("DSpark conditional probabilities must be finite numbers in [0, 1]")
        return cls(tuple(float(p) for p in rates))

    @property
    def survival(self) -> tuple[float, ...]:
        return tuple(accumulate(self.conditional_probabilities, mul))


@dataclass(frozen=True)
class VerificationPlan:
    prefixes: tuple[int, ...]
    expected_outputs: tuple[float, ...]

    @property
    def query_lens(self) -> tuple[int, ...]:
        return tuple(k + 1 for k in self.prefixes)

    @property
    def draft_tokens(self) -> int:
        return sum(self.prefixes)


@dataclass(frozen=True)
class PlanCost:
    latency_ms: float
    memory_left_gb: float


@dataclass(frozen=True)
class VerificationSelection:
    plan: VerificationPlan
    cost: PlanCost
    evaluated_candidates: int
    full_window_cost: PlanCost
    feasible: bool = True


class InfeasibleVerificationPlanError(ValueError):
    """Raised when no evaluated verification prefix satisfies memory and TPOT limits."""

    def __init__(self, message: str, selection: VerificationSelection):
        super().__init__(message)
        self.selection = selection


def select_verification_plan(
    survival: tuple[tuple[float, ...], ...],
    evaluate: Callable[[VerificationPlan], PlanCost],
    *,
    max_evaluations: int = 32,
    tpot_limit_ms: float | None = None,
    remaining_outputs: tuple[float, ...] | None = None,
) -> VerificationSelection:
    """Select within an explicit candidate set; do not assume monotonic costs."""
    if not survival or any(len(row) == 0 for row in survival):
        raise ValueError("verification plan needs at least one decode request and one speculative position")
    count, n = len(survival), len(survival[0])
    if any(len(row) != n for row in survival):
        raise ValueError("survival rows must be rectangular")
    if remaining_outputs is not None and len(remaining_outputs) != count:
        raise ValueError("remaining_outputs length must match decode request count")
    total = count * n
    cumulative = [tuple(accumulate(row, initial=1.0)) for row in survival]
    gains = survival
    if remaining_outputs is not None:
        cumulative = [tuple(min(e, remaining) for e in row) for row, remaining in zip(cumulative, remaining_outputs)]
        gains = tuple(tuple(b - a for a, b in zip(row, row[1:])) for row in cumulative)

    def make_plan(prefixes):
        outputs = tuple(cumulative[r][k] for r, k in enumerate(prefixes))
        return VerificationPlan(tuple(prefixes), outputs)

    def frontier(budgets):
        prefixes = [0] * count
        queue = [(-row[0], 0, r) for r, row in enumerate(gains)]
        heapq.heapify(queue)
        admitted = 0
        for budget in budgets:
            while admitted < budget:
                _, j, r = heapq.heappop(queue)
                prefixes[r] += 1
                admitted += 1
                if j + 1 < n:
                    heapq.heappush(queue, (-gains[r][j + 1], j + 1, r))
            yield make_plan(prefixes)

    costs = {}

    def visit(plan):
        if plan not in costs:
            costs[plan] = evaluate(plan)

    def score(plan):
        cost = costs[plan]
        if cost.memory_left_gb < 0 or cost.latency_ms <= 0 or any(e <= 0 for e in plan.expected_outputs):
            return (-math.inf, 0)
        if tpot_limit_ms is not None and any(cost.latency_ms / e > tpot_limit_ms for e in plan.expected_outputs):
            return (-math.inf, 0)
        return (sum(plan.expected_outputs) / cost.latency_ms, -plan.draft_tokens)

    if total + 1 <= max_evaluations:
        for plan in frontier(range(total + 1)):
            visit(plan)
    else:
        initial = max(2, max_evaluations // 2)
        homogeneous = len(set(survival)) == 1 and (remaining_outputs is None or len(set(remaining_outputs)) == 1)
        if homogeneous:
            samples = n + 1 if n + 1 <= max_evaluations else initial
            prefixes = [i * n // (samples - 1) for i in range(samples)]
            frontier_points = {k * count for k in prefixes}
            for k in prefixes:
                visit(make_plan((k,) * count))
        else:
            budgets = [i * total // (initial - 1) for i in range(initial)]
            frontier_points = set(budgets)
            for plan in frontier(budgets):
                visit(plan)
        while len(costs) < max_evaluations:
            best = max(costs, key=score)
            points = sorted(frontier_points)
            intervals = [(a, b) for a, b in zip(points, points[1:]) if b - a > 1]
            if not intervals:
                break
            a, b = min(
                intervals,
                key=lambda ab: (
                    min(abs(best.draft_tokens - ab[0]), abs(best.draft_tokens - ab[1])),
                    -(ab[1] - ab[0]),
                    ab[0],
                ),
            )
            mid = (a + b) // 2
            frontier_points.add(mid)
            visit(next(frontier([mid])))
    best = max(costs, key=score)
    full_cost = costs[make_plan((n,) * count)]
    if score(best)[0] == -math.inf:

        def fallback_key(plan):
            cost = costs[plan]
            mem_ok = cost.memory_left_gb >= 0
            if cost.latency_ms > 0 and all(e > 0 for e in plan.expected_outputs):
                worst_tpot = max(cost.latency_ms / e for e in plan.expected_outputs)
            else:
                worst_tpot = math.inf
            return (mem_ok, cost.memory_left_gb if not mem_ok else 0.0, -worst_tpot, -plan.draft_tokens)

        fallback = max(costs, key=fallback_key)
        infeasible = VerificationSelection(
            fallback,
            costs[fallback],
            len(costs),
            full_cost,
            feasible=False,
        )
        raise InfeasibleVerificationPlanError(
            "no verification prefix meets the memory and TPOT constraints",
            infeasible,
        )
    return VerificationSelection(
        best,
        costs[best],
        len(costs),
        full_cost,
        feasible=True,
    )
