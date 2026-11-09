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

"""Statistical proposer that widens the exact gate's proposal reach.

DISCIPLINE (non-negotiable): statistics NEVER serve residuals.  The exact
gate - zero residual on every captured sample - remains the ONLY serving
path; nothing this module returns is trusted until the caller re-verifies it
on all samples through the existing evaluator
(``formulas.evaluate_formula_safe``).  What statistics buy here:

1. Smarter proposals the hand enumeration misses: stepwise feature selection
   (BIC) over the FULL runtime feature table
   (``template_generate._build_features``) instead of the hand shortlist,
   MULTI widths 2..4 with optional integer offset, and integer coefficients
   solved exactly (fraction-free Cramer + small-coefficient enumeration).
2. Outlier detection that FLAGS capture anomalies instead of fitting through
   them: leave-one-out residuals flag suspect samples into the diagnostics.
   A flagged proposal is only servable when the exact gate still verifies it
   on ALL samples - flagged ones included.  Nothing is ever dropped
   silently.
3. Quantified extrapolation confidence (``extrapolation_confidence``) for
   formulas that ARE exact on all anchors but are evaluated beyond the
   anchor span.  Confidence tiers live in diagnostics only - no serving
   behaviour change beyond the existing ``extrapolated`` flag.

Every formula returned reuses the existing schema (LINEAR / MULTI), so the
exact gate evaluates proposals unchanged.

Architecture (reading guide)
----------------------------
Owns the statistical PROPOSER tier.  Input: ``(config, value)`` samples plus
runtime feature tables; output: ``(formula, diagnostics)`` pairs in the
standard schema (or ``(None, reason)``), recorded in :data:`PROPOSAL_LOG`.

Entry points:

- ``propose`` (L640)                  — the beam search over feature subsets
- ``extrapolation_confidence`` (L781) — confidence tiers for exact formulas
- ``_exact_integer_fits`` (L217) / ``_verify_fit`` (L213) — the exactness spine
- ``_loo_outliers`` (L474)            — the capture-anomaly flags

Read this if you want to understand why a formula was PROPOSED (the
diagnostics say so), what the confidence tiers mean, and why nothing here
can serve a residual: every proposal is re-verified by the caller.
"""

from __future__ import annotations

import json
import math
from itertools import combinations, product
from typing import Any, Mapping, Sequence

__all__ = [
    "PROPOSAL_LOG",
    "_bic",
    "extrapolation_confidence",
    "propose",
]

# ---------------------------------------------------------------- limits ---

_MAX_STEPWISE_COLUMNS = 64  # discovery pool cap (ALL varying columns ranked)
_EXHAUSTIVE_PAIR_POOL = 10  # exhaustive width-2 sweep over this pool
_RESIDUAL_SEEDS = 4  # width-1 residual anchors
_RESIDUAL_TOP = 5  # residual-ranked extensions per seed
_BEAM_WIDTH = 8  # surviving subsets per level
_BEAM_WIDTH4_SEEDS = 3  # triples extended to width 4
_PRUNE_FACTOR = 1e-6  # float-LS prune: exact fits imply RSS ~ 0
_MAX_EXACT_ATTEMPTS = 48  # exact-solve budget per dim per family
_SMALL_COEFF_LIMITS = {2: 8, 3: 3, 4: 2}  # |coeff| caps for enumeration
_SEED_ROWS = 8  # Cramer seed-row cap (mirrors builder discipline)
_OUTLIER_LOO_FACTOR = 6.0  # |LOO residual| > factor * robust scale -> flag
_MIN_OUTLIER_SCALE = 1.0
_MAX_PROPOSALS = 12
_LOG_MAX = 4096
_MEMO_MAX = 20000

PROPOSAL_LOG: list[dict[str, Any]] = []
_MEMO: dict[Any, list[tuple[dict[str, Any], dict[str, Any]]]] = {}


def _copy_proposals(
    proposals: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deep-copy proposals so callers can never mutate the memoized originals.

    Review issue #27: the memo used to share formula/diagnostics dicts with
    its callers, so a caller that tweaked a returned formula (e.g. bumped
    ``offset``) poisoned the process-wide cache - the next identical input
    came back with the mutated numbers AND the stale ``exact`` diagnostics.
    Both cache directions copy: the cold call stores a private copy and
    returns the originals, a cache hit hands out a fresh copy every time.
    """

    import copy

    return [(copy.deepcopy(formula), copy.deepcopy(diag)) for formula, diag in proposals]


# ------------------------------------------------------------------- BIC ---


def _bic(rss: float, k: int, n: int) -> float:
    """Gaussian BIC of a least-squares fit: ``n*ln(rss/n) + k*ln(n)``.

    A zero-RSS (exact) fit scores ``-inf`` - strictly better than any
    residual fit regardless of width.  Degenerate inputs score ``+inf``.
    """

    if n <= 0:
        return math.inf
    if rss <= 0.0:
        return -math.inf
    try:
        return n * math.log(rss / n) + k * math.log(n)
    except ValueError:  # pragma: no cover - rss>0 keeps the log defined
        return math.inf


# --------------------------------------------------- least squares (diag) --


def _ols_fit(columns: Sequence[Sequence[float]], y: Sequence[float]) -> tuple[list[float], float] | None:
    """Plain least squares via normal equations (diagnostics only - the
    exact coefficient solve below never uses these floats).
    """

    k = len(columns)
    if k == 0:
        mean = sum(y) / len(y)
        rss = sum((value - mean) ** 2 for value in y)
        return [mean], rss
    gram = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for row, target in zip(zip(*columns), y):
        for i in range(k):
            ri = row[i]
            rhs[i] += ri * target
            for j in range(i, k):
                gram[i][j] += ri * row[j]
    for i in range(k):
        for j in range(i):
            gram[i][j] = gram[j][i]
    solved = _gauss(gram, rhs)
    if solved is None:
        return None
    rss = 0.0
    for row, target in zip(zip(*columns), y):
        pred = sum(solved[i] * row[i] for i in range(k))
        rss += (target - pred) ** 2
    return solved, rss


def _gauss(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; ``None`` when singular."""

    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    scale = max((abs(value) for row in aug for value in row), default=0.0)
    if scale == 0.0:
        return None
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12 * scale:
            return None
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    solution = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = aug[r][n] - sum(aug[r][c] * solution[c] for c in range(r + 1, n))
        solution[r] = acc / aug[r][r]
    return solution


# --------------------------------------------- exact integer coefficient ---


def _det_bareiss(matrix: list[list[int]]) -> int:
    """Exact integer determinant (fraction-free Bareiss)."""

    m = len(matrix)
    aug = [list(row) for row in matrix]
    sign = 1
    prev = 1
    for col in range(m):
        pivot_row = next((r for r in range(col, m) if aug[r][col] != 0), None)
        if pivot_row is None:
            return 0
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            sign = -sign
        pivot = aug[col][col]
        for r in range(col + 1, m):
            for c in range(col + 1, m):
                aug[r][c] = (aug[r][c] * pivot - aug[r][col] * aug[col][c]) // prev
        prev = pivot
    return sign * aug[m - 1][m - 1]


def _cramer_solve(matrix: list[list[int]], rhs: list[int]) -> list[int] | None:
    """Exact integer Cramer solve; ``None`` unless every unknown divides."""

    det = _det_bareiss(matrix)
    if det == 0:
        return None
    m = len(matrix)
    solution: list[int] = []
    for col in range(m):
        swapped = [list(row) for row in matrix]
        for r in range(m):
            swapped[r][col] = rhs[r]
        num = _det_bareiss(swapped)
        if num % det:
            return None
        solution.append(num // det)
    return solution


def _column_rows(rows: Sequence[Sequence[int]], cols: Sequence[int], with_offset: bool) -> list[list[int]]:
    return [list(rows[i][c] for c in cols) + ([1] if with_offset else []) for i in range(len(rows))]


def _verify_fit(expanded: Sequence[Sequence[int]], y: Sequence[int], solution: Sequence[int]) -> bool:
    return all(sum(c * v for c, v in zip(solution, row)) == target for row, target in zip(expanded, y))


def _exact_integer_fits(
    cols: Sequence[int],
    rows: Sequence[Sequence[int]],
    y: Sequence[int],
    with_offset: bool,
) -> list[list[int]]:
    """All integer-coefficient fits of ``y`` on the selected columns.

    Two exact solvers, both verified on every row:
    (a) fraction-free Cramer over seed-row combinations (the builder's own
        discipline: solve on a few rows, cross-check on all), and
    (b) small-coefficient enumeration (rank-deficient feature columns make
        Cramer seeds singular even when a small-integer solution exists;
        the offset is derived exactly from row 0 for each coefficient try).
    Over-determination is required: unknowns + 1 <= n, so at least one row
    always falsifies - stat_fit never proposes an interpolation-grade fit.
    """

    k = len(cols)
    m = k + (1 if with_offset else 0)
    n = len(y)
    if m < 1 or n < m + 1:
        return []
    expanded = _column_rows(rows, cols, with_offset)
    fits: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def record(solution: list[int]) -> None:
        key = tuple(solution)
        if key not in seen:
            seen.add(key)
            fits.append(list(solution))

    seed_count = min(n, _SEED_ROWS)
    for seed in combinations(range(seed_count), m):
        matrix = [expanded[i] for i in seed]
        rhs = [y[i] for i in seed]
        solution = _cramer_solve(matrix, rhs)
        if solution is None:
            continue
        if _verify_fit(expanded, y, solution):
            record(solution)

    limit = _SMALL_COEFF_LIMITS.get(k)
    if limit is not None:
        for combo in product(range(-limit, limit + 1), repeat=k):
            if not any(combo):
                continue
            offset = y[0] - sum(c * v for c, v in zip(combo, expanded[0]))
            solution = list(combo) + ([offset] if with_offset else [])
            if _verify_fit(expanded, y, solution):
                record(solution)
    return fits


# ------------------------------------------------------- column building ---


def _feature_tables(configs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, int]]:
    try:
        from template_engine.formulas import _features_for  # noqa: PLC0415

        return [_features_for(config) for config in configs]
    except Exception:  # pragma: no cover - standalone use
        from template_generate import _build_features  # noqa: PLC0415

        return [_build_features(config) for config in configs]


def _build_columns(tables: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Common, varying, integer-valued feature columns; alias-collapsed."""

    common = [name for name in tables[0] if all(name in table for table in tables)]
    columns: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for name in common:
        column: list[int] = []
        ok = True
        for table in tables:
            value = table.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                ok = False
                break
            column.append(int(value))
        if not ok or len(set(column)) < 2:
            continue
        key = tuple(column)
        if key in seen:  # aliased features (e.g. "(B*S/TP)*H") collapse
            continue
        seen.add(key)
        columns[str(name)] = column
    return columns


def _prefilter(columns: dict[str, list[int]], y: Sequence[int], keep: int) -> list[str]:
    """Rank ALL varying columns by |corr| with the target (soft ordering).

    This orders the discovery pool only - it never EXCLUDES a column from
    residual-chasing (small-contribution features rank poorly against a
    dominant term but are exactly what the residual stages exist to find).
    """

    n = len(y)
    mean_y = sum(y) / n
    denom_y = math.sqrt(sum((value - mean_y) ** 2 for value in y)) or 1.0

    def abs_corr(column: list[int]) -> float:
        mean_x = sum(column) / n
        denom_x = math.sqrt(sum((value - mean_x) ** 2 for value in column))
        if denom_x == 0.0:
            return 0.0
        num = sum((column[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        return abs(num / (denom_x * denom_y))

    ranked = sorted(columns, key=lambda name: (-abs_corr(columns[name]), name))
    return ranked[:keep]


def _ols_rss(subset: Sequence[str], columns: dict[str, list[int]], y: Sequence[int]) -> float | None:
    """Least-squares RSS of ``y`` on the subset plus a constant column.

    Purely a discovery prune: an integer-exact relation implies rss ~ 0
    (float noise), so subsets with large RSS cannot admit an exact fit and
    never reach the exact solvers.  ``None`` marks a singular design.
    """

    design = [columns[name] for name in subset] + [[1.0] * len(y)]
    fit = _ols_fit(design, [float(v) for v in y])
    if fit is None:
        return None
    return max(fit[1], 0.0)


def _residual_after(subset: Sequence[str], columns: dict[str, list[int]], y: Sequence[int]) -> list[float] | None:
    """Residual of the least-squares fit (with intercept) of ``y``."""

    design = [columns[name] for name in subset] + [[1.0] * len(y)]
    fit = _ols_fit(design, [float(v) for v in y])
    if fit is None:
        return None
    coeffs = fit[0]
    residual = []
    for i, target in enumerate(y):
        pred = sum(c * columns[name][i] for c, name in zip(coeffs[:-1], subset)) + coeffs[-1]
        residual.append(target - pred)
    return residual


def _abs_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(ys)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = math.sqrt(sum((v - mean_x) ** 2 for v in xs) * sum((v - mean_y) ** 2 for v in ys))
    if denom == 0.0:
        return 0.0
    return abs(sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / denom)


def _candidate_subsets(
    names: Sequence[str],
    columns: dict[str, list[int]],
    y: Sequence[int],
) -> list[list[str]]:
    """Candidate feature subsets, width 2..4, cheapest-first.

    Residual-chasing discovery (statistics propose, the exact gate
    disposes): at every level ALL one-column extensions are scored by
    least-squares RSS and the best are kept globally - not per prefix - so
    small-contribution features (a ``+3*MTP`` beside a dominant ``2*B*S``)
    surface wherever they enter.  Every subset handed back is later solved
    in exact integer arithmetic and re-verified by the caller's exact
    gate; nothing here serves on statistics.
    """

    pool = list(names)[:_MAX_STEPWISE_COLUMNS]

    def rss_of(subset: Sequence[str]) -> float:
        rss = _ols_rss(subset, columns, y)
        return math.inf if rss is None else rss

    # Width 1 seeds (residual anchors); width 2 = exhaustive sweep over the
    # top-correlation pool UNION residual-top extensions of the seeds.  The
    # residual-guided pairs ALWAYS reach the width-3 expansion: pair-level
    # RSS alone lets near-proxy pairs (rss 0.28) crowd out true intermediate
    # pairs (their residual still hides a structured term, so their raw rss
    # looks worse).
    singles = sorted(pool, key=lambda name: (rss_of([name]), name))
    pair_pool: set[tuple[str, str]] = set()
    for a, b in combinations(pool[:_EXHAUSTIVE_PAIR_POOL], 2):
        pair_pool.add(tuple(sorted((a, b))))
    guided: list[list[str]] = []
    for seed in singles[:_RESIDUAL_SEEDS]:
        residual = _residual_after([seed], columns, y)
        if residual is None:
            continue
        ranked = sorted(
            (name for name in pool if name != seed),
            key=lambda name: (-_abs_corr(columns[name], residual), name),
        )
        for name in ranked[:_RESIDUAL_TOP]:
            pair = [seed, name] if name > seed else [name, seed]
            pair_pool.add((pair[0], pair[1]))
            guided.append(pair)
    width2 = sorted(
        ([a, b] for a, b in pair_pool),
        key=lambda subset: (rss_of(subset), subset),
    )[:_BEAM_WIDTH]
    expansion_pairs: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for subset in width2 + guided:
        key = (subset[0], subset[1])
        if key not in seen_pairs:
            seen_pairs.add(key)
            expansion_pairs.append(subset)
    # Width 3: every one-column extension of every expansion pair competes
    # globally; the exact-relation extensions sort to the very front.
    known: set[tuple[str, ...]] = set()
    triples: list[tuple[float, list[str]]] = []
    for prefix in expansion_pairs:
        if tuple(prefix) in known:
            continue
        known.add(tuple(prefix))
        for name in pool:
            if name in prefix:
                continue
            candidate = sorted(prefix + [name])
            if tuple(candidate) in known:
                continue
            known.add(tuple(candidate))
            triples.append((rss_of(candidate), candidate))
    triples.sort(key=lambda item: (item[0], item[1]))
    width2 = [list(p) for p in expansion_pairs[:_BEAM_WIDTH]]
    width3 = [subset for _rss, subset in triples[:_BEAM_WIDTH]]
    # Width 4: extend only the best triples (cost guard).
    quads: list[tuple[float, list[str]]] = []
    for prefix in width3[:_BEAM_WIDTH4_SEEDS]:
        for name in pool:
            if name in prefix:
                continue
            candidate = sorted(prefix + [name])
            if tuple(candidate) in known:
                continue
            known.add(tuple(candidate))
            quads.append((rss_of(candidate), candidate))
    quads.sort(key=lambda item: (item[0], item[1]))
    width4 = [subset for _rss, subset in quads[:_BEAM_WIDTH4_SEEDS]]
    # No width-1 re-entry: the hand enumeration (formulas Passes 1-3) already
    # tries every single-feature exact LINEAR over the full feature table
    # before the stat_fit seam is consulted, so a width-1 proposal here could
    # never pass the caller's exact gate.  (It existed for the DOMAIN_DIV
    # family, removed with issue #11.)
    return width2 + width3 + width4


# --------------------------------------------------------- LOO outliers ---


def _loo_outliers(
    columns: dict[str, list[int]],
    selected: Sequence[str],
    y: Sequence[int],
) -> list[int]:
    """Leave-one-out residual outlier FLAGS (never drops).

    Each sample is predicted by the least-squares fit trained on the other
    samples; residuals far above the robust scale flag capture anomalies.
    The flags only annotate diagnostics - the exact gate decides serving,
    and a proposal must be exact on ALL samples (flagged included).
    """

    n = len(y)
    k = len(selected)
    if n < 4 or k == 0 or k >= n - 1:
        return []
    residuals: list[float] = []
    for held_out in range(n):
        sub_cols = [[columns[name][i] for i in range(n) if i != held_out] for name in selected]
        sub_y = [y[i] for i in range(n) if i != held_out]
        fit = _ols_fit(sub_cols, [float(v) for v in sub_y])
        if fit is None:
            return []
        coeffs = fit[0]
        pred = sum(c * columns[name][held_out] for c, name in zip(coeffs, selected))
        residuals.append(y[held_out] - pred)
    # Per-sample robust scale: the median |LOO residual| of the OTHER
    # samples.  A capture anomaly inflates every fit it participates in, so
    # a shared scale would hide it; excluding the sample itself keeps the
    # scale clean and the anomaly stands out against its peers.
    abs_residuals = [abs(r) for r in residuals]
    flagged: list[int] = []
    for i, residual in enumerate(residuals):
        peers = sorted(value for j, value in enumerate(abs_residuals) if j != i)
        peer_scale = peers[len(peers) // 2] if peers else 0.0
        threshold = max(_OUTLIER_LOO_FACTOR * max(peer_scale, _MIN_OUTLIER_SCALE), _MIN_OUTLIER_SCALE)
        if abs(residual) > threshold:
            flagged.append(i)
    return flagged


# ------------------------------------------------------------ proposing ---


def _formula_from_fit(names: Sequence[str], solution: Sequence[int], with_offset: bool) -> dict[str, Any]:
    coeffs = [int(c) for c in solution]
    offset = coeffs.pop() if with_offset else 0
    if len(coeffs) == 1:
        return {"type": "LINEAR", "coeff": coeffs[0], "feat": names[0], "offset": offset}
    return {"type": "MULTI", "coeffs": coeffs, "feats": list(names), "offset": offset}


def _diagnostics(
    formula: dict[str, Any],
    columns: dict[str, list[int]],
    y: Sequence[int],
    solution: Sequence[int],
    with_offset: bool,
    exact: bool,
    family: str,
) -> dict[str, Any]:
    names = formula.get("feats") or ([formula["feat"]] if formula.get("feat") else [])
    k = len(names)
    unknowns = k + (1 if with_offset else 0)
    n = len(y)
    fit = _ols_fit([columns[name] for name in names], [float(v) for v in y]) if names else None
    rss = fit[1] if fit else None
    bic = _bic(rss, unknowns, n) if rss is not None and rss > 0 else (-math.inf if exact else math.inf)
    outliers = _loo_outliers(columns, names, list(y))
    slack = n - unknowns
    return {
        "source": "stat_fit",
        "family": family,
        "exact": exact,
        "n": n,
        "k": k,
        "feats": names,
        "coeffs": [int(c) for c in solution[:-1]] if with_offset else [int(c) for c in solution],
        "offset": int(solution[-1]) if with_offset else 0,
        "bic": bic,
        "rss": rss,
        "slack": slack,
        "tier": ("strong" if slack >= 2 else "medium") if exact else None,
        "outliers": outliers,
        "suspect": bool(outliers),
        "note": (f"stat_fit exact {family} fit (k={k}, offset={with_offset}, slack={slack}, outliers={outliers})"),
    }


def _linear_family(
    columns: dict[str, list[int]],
    y: Sequence[int],
    family: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Stepwise-selected exact MULTI fits (the widened proposal space).

    Discovery: exhaustive width-2/3 + beam width-4 over the correlation-
    ranked pool, pruned by the float-LS RSS (an exact relation implies
    RSS ~ 0, so pruned subsets cannot hold an exact fit).  Survivors go to
    the exact integer solvers; the exact gate re-verifies everything.
    """

    names = _prefilter(columns, y, _MAX_STEPWISE_COLUMNS)
    prune = _PRUNE_FACTOR * (1.0 + sum(value * value for value in y))
    rows = [list(values) for values in zip(*(columns[name] for name in names))]
    name_index = {name: i for i, name in enumerate(names)}
    survivors: list[tuple[float, list[str]]] = []
    for subset in _candidate_subsets(names, columns, y):
        rss = _ols_rss(subset, columns, y)
        if rss is not None and rss <= prune:
            survivors.append((rss, subset))
    survivors.sort(key=lambda item: (item[0], item[1]))
    proposals: list[tuple[dict[str, Any], dict[str, Any]]] = []
    attempts = 0
    for _rss, subset in survivors:
        if attempts >= _MAX_EXACT_ATTEMPTS:
            break
        attempts += 1
        col_ids = list(range(len(subset)))
        subset_rows = [[row[name_index[name]] for name in subset] for row in rows]
        for with_offset in (False, True):
            for solution in _exact_integer_fits(col_ids, subset_rows, list(y), with_offset):
                formula = _formula_from_fit(subset, solution, with_offset)
                proposals.append((formula, _diagnostics(formula, columns, y, solution, with_offset, True, family)))
    return proposals


def _canonical(formula: Mapping[str, Any]) -> str:
    return json.dumps(formula, sort_keys=True, default=str)


def propose(
    configs: Sequence[Mapping[str, Any]],
    values: Sequence[int],
    features: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Propose exact-capable formulas for one numeric quantity.

    ``configs``/``values`` are the (config, observed value) anchor pairs;
    ``features`` optionally carries the precomputed feature tables.  Returns
    ``[(formula_dict, diagnostics), ...]`` best-first: exact-on-all-samples
    proposals (ranked by width then BIC), then a few best-fitting
    non-exact proposals kept purely as diagnostics (residual shape + LOO
    outlier flags) for the caller's stat-reason note.  NOTHING here is
    served until the caller's exact gate verifies zero residual on every
    sample.
    """

    y = [int(value) for value in values]
    n = len(y)
    if n < 2 or len(set(y)) < 2:
        return []
    memo_key = (
        tuple(tuple(sorted((str(k), v) for k, v in config.items())) for config in configs),
        tuple(y),
    )
    cached = _MEMO.get(memo_key)
    if cached is not None:
        return _copy_proposals(cached)

    tables = list(features) if features is not None else _feature_tables(configs)
    if len(tables) != n:
        return []
    columns = _build_columns(tables)
    if not columns:
        return []

    exact_proposals = _linear_family(columns, y, "multi")
    # Belt-and-braces: integer verification already ran inside the solvers;
    # keep only proposals the column arithmetic still confirms.
    exact_proposals = [
        (formula, diag)
        for formula, diag in exact_proposals
        if diag["exact"]
        and all(
            sum(c * v for c, v in zip(diag["coeffs"], (columns[name][i] for name in diag["feats"]))) + diag["offset"]
            == y[i]
            for i in range(n)
        )
    ]
    exact_proposals.sort(key=lambda item: (item[1]["k"], item[1]["bic"]))

    # Non-exact diagnostics: the BIC-best residual fit + LOO outlier flags
    # scanned across the best subset of EACH width (a capture anomaly hides
    # the true relation; the width-1/2 subsets closest to it expose it), so
    # the caller's stat-reason can say WHY nothing served.
    diagnostic_proposals: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if not exact_proposals:
        names = _prefilter(columns, y, _MAX_STEPWISE_COLUMNS)
        scored: list[tuple[float, int, list[str]]] = []
        for subset in _candidate_subsets(names, columns, y):
            rss = _ols_rss(subset, columns, y)
            if rss is not None:
                scored.append((rss, len(subset), subset))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        best_per_width: dict[int, list[str]] = {}
        for _rss, width, subset in scored:
            best_per_width.setdefault(width, subset)
        outliers = sorted({i for subset in best_per_width.values() for i in _loo_outliers(columns, subset, y)})
        if scored:
            rss, _width, subset = scored[0]
            design = [columns[name] for name in subset] + [[1.0] * n]
            fit = _ols_fit(design, [float(v) for v in y])
            if fit is not None:
                coeffs = [float(c) for c in fit[0][:-1]]
                offset_float = float(fit[0][-1])
                residuals = [
                    y[i] - (sum(c * columns[name][i] for c, name in zip(coeffs, subset)) + offset_float)
                    for i in range(n)
                ]
                diagnostic_proposals.append(
                    (
                        {"type": "FAILED"},
                        {
                            "source": "stat_fit",
                            "family": "diagnostic",
                            "exact": False,
                            "n": n,
                            "k": len(subset),
                            "feats": list(subset),
                            "coeffs_float": [round(c, 6) for c in coeffs],
                            "bic": _bic(rss, len(subset) + 1, n),
                            "rss": rss,
                            "residuals": [round(r, 4) for r in residuals],
                            "outliers": outliers,
                            "suspect": bool(outliers),
                            "tier": None,
                            "note": (
                                f"stat_fit BIC-best non-exact fit {subset} "
                                f"(rss={rss:.4g}); LOO outliers={outliers} - "
                                "statistics never serve residuals"
                            ),
                        },
                    )
                )

    proposals = exact_proposals[:_MAX_PROPOSALS] + diagnostic_proposals
    unique: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for formula, diag in proposals:
        unique.setdefault(_canonical(formula) + "|" + str(diag.get("family")), (formula, diag))
    proposals = list(unique.values())

    if len(_MEMO) > _MEMO_MAX:
        _MEMO.clear()
    _MEMO[memo_key] = _copy_proposals(proposals)
    if len(PROPOSAL_LOG) < _LOG_MAX:
        PROPOSAL_LOG.append(
            {
                "n": n,
                "values": y,
                "exact_proposals": sum(1 for _, d in proposals if d.get("exact")),
                "winner": proposals[0][0] if proposals and proposals[0][1].get("exact") else None,
                "outliers": next((d["outliers"] for _, d in proposals if d.get("outliers")), []),
                "diagnostic_note": proposals[-1][1]["note"] if proposals and not proposals[0][1].get("exact") else None,
            }
        )
    return proposals


# ------------------------------------------------ extrapolation confidence --


def extrapolation_confidence(
    anchor_configs: Sequence[Mapping[str, Any]],
    target_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Quantified extrapolation confidence for an exact formula at a target.

    Pure diagnostics: anchor count, per-axis anchor span, and the target's
    normalised span leverage (distance from the span centre in units of the
    half-span; >1 means the target sits beyond the captured hull on some
    axis).  Tiers: ``strong`` (>=4 anchors, inside span), ``medium``
    (>=3 anchors, inside span), ``thin`` (everything else).  Consumers must
    not change serving behaviour on this - the existing ``extrapolated``
    flag remains the only gate-relevant marker.
    """

    axes = ("B", "S")
    anchors = {axis: [int(c.get(axis, 1) or 1) for c in anchor_configs] for axis in axes}
    target = {axis: int(target_config.get(axis, 1) or 1) for axis in axes}
    spans = {axis: [min(vals), max(vals)] for axis, vals in anchors.items()}
    leverages: list[float] = []
    inside = True
    for axis in axes:
        lo, hi = spans[axis]
        centre = (lo + hi) / 2.0
        half = max((hi - lo) / 2.0, 1.0)
        leverages.append(abs(target[axis] - centre) / half)
        if not lo <= target[axis] <= hi:
            inside = False
    count = len(anchor_configs)
    leverage = max(leverages) if leverages else 0.0
    if count >= 4 and inside:
        tier = "strong"
    elif count >= 3 and inside:
        tier = "medium"
    else:
        tier = "thin"
    return {
        "anchor_count": count,
        "span": spans,
        "inside_span": inside,
        "span_leverage": round(leverage, 4),
        "tier": tier,
        "note": (
            f"extrapolation tier={tier}: {count} anchors, span={spans}, leverage={leverage:.2f} (diagnostics only)"
        ),
    }
