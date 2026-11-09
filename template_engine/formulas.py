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

"""Exact shape-formula derivation for the template-graph builder.

This module implements the offline induction step described by the template
engine specification (section 17): given one numeric quantity observed across
several anchor configurations, derive the smallest formula that reproduces the
value exactly at every anchor.  Nothing is ever extrapolated silently — when no
candidate formula fits every sample the derivation returns ``FAILED`` and the
expansion side falls back to the template's original value.

Feature tables come from ``template_generate._build_features`` so that derived
formulas evaluate through the same runtime used by ``generate_ops_from_template``.

Architecture (reading guide)
----------------------------
Owns exact formula DERIVATION (spec §17) and the formula SCHEMA:
CONST / LINEAR / MULTI / DOMAIN_CEIL_DIV / EXACT_LOOKUP / FAILED.  Input:
``(config, value)`` samples; output: one formula dict exact on every sample,
or :data:`FAILED_FORMULA` (never a guess).

Issue #11 removed the never-delivered families (STEP, DOMAIN_DIV - zero
uses across all artifacts, so their fitters and evaluation branches are
gone; a future fit falls to FAILED -> EXACT_LOOKUP conservative semantics).

Entry points:

- ``derive_value_formula`` (L545)  — the pass cascade (the induction itself)
- ``evaluate_formula_safe`` (L633) — the runtime evaluator used by all consumers
- ``template_has_failed_formula`` (L646) / ``count_failed_formulas`` (L658)
- ``_stat_fit_fallback`` (L467)    — the seam into the ``stat_fit`` proposer tier

Read this if you want to understand what a formula object inside the pickle
means and how a value observed at the anchors becomes one.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from itertools import combinations
from typing import Any, Mapping, Sequence

from template_generate import MODEL_CONSTANTS, eval_formula

# Features allowed inside MULTI fits.  Keeping the list short bounds the
# pairwise search; every entry is an "atomic" quantity that plausibly drives a
# tensor dimension.
_MULTI_FEATURE_SHORTLIST = (
    "B",
    "S",
    "TP",
    "EP",
    "MOE_DP",
    "DCP",
    "WS",
    "MTP",
    "B*S",
    "H",
    "I",
    "MI",
    "E",
    # Paged-KV whitebox features (see template_generate._build_features).
    "B*(2*S+1+MTP)",
    "total_kv//4",
    "ceil(total_kv//4/128)",
    "S//256+1",
    "B*S/256",
    "S//256",
    "S//64",
    "S//4",
    "indexer_kv_blocks",
    "B*S*(S//256)",
    "B*S*(S//64)",
    # Parallel-product axes (RFC Phase D): expert-weight sharding dims such
    # as 3072*MOE_DP//TP are exact linear combinations of these; the 0/1
    # scatter/pad flags let families spanning a flip condition dims once
    # captures cover both sides.
    "TP*MOE_DP",
    "EP*MOE_DP",
    "E*MOE_DP/TP",
    "E*MOE_DP/EP",
    "E*EP/TP",
    "step((B*S/EP)%128==0)",
    "pad",
    "group_index",
    "list_index",
    "B*group_index",
    "B*list_index",
    "S*group_index",
    "S*list_index",
)

FAILED_FORMULA: dict[str, Any] = {"type": "FAILED"}


def const_formula(value: Any) -> dict[str, Any]:
    return {"type": "CONST", "value": value}


def _features_for(config: Mapping[str, Any]) -> dict[str, int]:
    """Build the feature table for one configuration (memoized)."""

    from template_generate import _build_features

    cache = _FEATURE_TABLE_CACHE
    fingerprint = frozenset(config.items())
    table = cache.get(fingerprint)
    if table is None:
        table = _build_features(config)
        if len(cache) > 4096:
            cache.clear()
        cache[fingerprint] = table
    return table


_FEATURE_TABLE_CACHE: dict[frozenset, dict[str, int]] = {}


def _linear_fit(samples: Sequence[tuple[dict[str, int], int]], feature: str) -> dict[str, Any] | None:
    """Try ``value == coeff * feature + offset`` with exact integer arithmetic."""

    pairs = [(sample[0].get(feature), sample[1]) for sample in samples]
    if any(f_value is None for f_value, _ in pairs):
        return None
    distinct = {f_value for f_value, _ in pairs}
    if len(distinct) < 2:
        return None
    ordered = sorted(distinct)
    f_low, f_high = ordered[0], ordered[1]
    v_low = next(value for f_value, value in pairs if f_value == f_low)
    v_high = next(value for f_value, value in pairs if f_value == f_high)
    delta_feature = f_high - f_low
    delta_value = v_high - v_low
    if delta_feature == 0 or delta_value % delta_feature != 0:
        return None
    coeff = delta_value // delta_feature
    offset = v_low - coeff * f_low
    for f_value, value in pairs:
        if coeff * f_value + offset != value:
            return None
    if coeff == 1 and offset == 0:
        return {"type": "LINEAR", "coeff": 1, "feat": feature, "offset": 0}
    return {"type": "LINEAR", "coeff": coeff, "feat": feature, "offset": offset}


def _multi_fit(samples: Sequence[tuple[dict[str, int], int]], pair: tuple[str, str]) -> dict[str, Any] | None:
    """Try ``value == w1 * f1 + w2 * f2`` via exact Cramer solve (no offset)."""

    first, second = pair
    rows = [(s[0], s[1]) for s in samples]
    usable = [row for row in rows if first in row[0] and second in row[0]]
    if len(usable) != len(samples) or len(usable) < 2:
        return None

    def solve2(row_a: tuple[dict[str, int], int], row_b: tuple[dict[str, int], int]):
        f1a, f2a = row_a[0][first], row_a[0][second]
        f1b, f2b = row_b[0][first], row_b[0][second]
        ya, yb = row_a[1], row_b[1]
        determinant = f1a * f2b - f1b * f2a
        if determinant == 0:
            return None
        if (ya * f2b - yb * f2a) % determinant or (f1a * yb - f1b * ya) % determinant:
            return None
        w1 = (ya * f2b - yb * f2a) // determinant
        w2 = (f1a * yb - f1b * ya) // determinant
        return w1, w2

    for index_a in range(len(usable)):
        for index_b in range(index_a + 1, len(usable)):
            solved = solve2(usable[index_a], usable[index_b])
            if solved is None:
                continue
            w1, w2 = solved
            if all(w1 * row[0][first] + w2 * row[0][second] == row[1] for row in usable):
                return {"type": "MULTI", "coeffs": [w1, w2], "feats": [first, second], "offset": 0}
    return None


def _varying_features(paired: Sequence[tuple[dict[str, int], int]], candidates: Sequence[str]) -> list[str]:
    """Features whose values actually move across the paired samples."""

    varying: list[str] = []
    for feature in candidates:
        values = {row[0].get(feature) for row in paired if feature in row[0]}
        if len(values) >= 2:
            varying.append(feature)
    return varying


def _solve3(equations: list[tuple[int, int, int, int]]) -> tuple[int, int, int] | None:
    """Exact integer 3x3 Cramer solve; ``None`` unless every unknown divides."""

    (fa, fb, fc, y1), (fd, fe, ff, y2), (fg, fh, fi, y3) = equations
    det = fa * (fe * fi - ff * fh) - fb * (fd * fi - ff * fg) + fc * (fd * fh - fe * fg)
    if det == 0:
        return None
    wx = y1 * (fe * fi - ff * fh) - fb * (y2 * fi - ff * y3) + fc * (y2 * fh - fe * y3)
    wy = fa * (y2 * fi - ff * y3) - y1 * (fd * fi - ff * fg) + fc * (fd * y3 - y2 * fg)
    wz = fa * (fe * y3 - y2 * fh) - fb * (fd * y3 - y2 * fg) + y1 * (fd * fh - fe * fg)
    if any(determinant % det for determinant in (wx, wy, wz)):
        return None
    return wx // det, wy // det, wz // det


def _multi_fit_offset(samples: Sequence[tuple[dict[str, int], int]], pair: tuple[str, str]) -> dict[str, Any] | None:
    """Try ``value == w1*f1 + w2*f2 + offset`` (needs >= 4 samples).

    Seeds come from the first six rows only: 3 unknowns solved from 3 rows,
    cross-checked on every row.  This bounds the per-dim cost on rich
    families (C(16,3) seeds per pair measured in build-time hours) while
    every accepted fit still verifies exactly on all samples.
    """

    first, second = pair
    usable = [row for row in samples if first in row[0] and second in row[0]]
    if len(usable) != len(samples) or len(usable) < 4:
        return None

    def features_of(row: tuple[dict[str, int], int]) -> tuple[int, int, int, int]:
        return row[0][first], row[0][second], 1, row[1]

    seed_count = min(len(usable), 6)
    for rows_a, rows_b, rows_c in combinations(range(seed_count), 3):
        equations = [features_of(usable[index]) for index in (rows_a, rows_b, rows_c)]
        solved = _solve3(equations)
        if solved is None:
            continue
        w1, w2, offset = solved
        if all(w1 * row[0][first] + w2 * row[0][second] + offset == row[1] for row in usable):
            return {"type": "MULTI", "coeffs": [w1, w2], "feats": [first, second], "offset": offset}
    return None


def _multi3_fit(samples: Sequence[tuple[dict[str, int], int]], triple: tuple[str, str, str]) -> dict[str, Any] | None:
    """Try ``value == w1*f1 + w2*f2 + w3*f3`` (needs >= 4 samples)."""

    name_a, name_b, name_c = triple
    usable = [row for row in samples if all(name in row[0] for name in (name_a, name_b, name_c))]
    if len(usable) != len(samples) or len(usable) < 4:
        return None

    def features_of(row: tuple[dict[str, int], int]) -> tuple[int, int, int, int]:
        return row[0][name_a], row[0][name_b], row[0][name_c], row[1]

    seed_count = min(len(usable), 6)
    for rows_a, rows_b, rows_c in combinations(range(seed_count), 3):
        equations = [features_of(usable[index]) for index in (rows_a, rows_b, rows_c)]
        solved = _solve3(equations)
        if solved is None:
            continue
        w1, w2, w3 = solved
        if all(w1 * row[0][name_a] + w2 * row[0][name_b] + w3 * row[0][name_c] == row[1] for row in usable):
            return {"type": "MULTI", "coeffs": [w1, w2, w3], "feats": [name_a, name_b, name_c], "offset": 0}
    return None


def _proportional_fit(samples: Sequence[tuple[dict[str, int], int]], feature: str) -> dict[str, Any] | None:
    pairs = [(sample[0].get(feature), sample[1]) for sample in samples]
    if any(f_value in (None, 0) for f_value, _ in pairs):
        return None
    f_first, v_first = pairs[0]
    if v_first % f_first != 0:
        return None
    coeff = v_first // f_first
    # coeff == 0 would bake the first sample's zero into a constant that
    # fails every other anchor — reject it outright (CONST covers it anyway).
    if coeff != 0 and all(coeff * f_value == value for f_value, value in pairs):
        if coeff == 1:
            return {"type": "LINEAR", "coeff": 1, "feat": feature, "offset": 0}
        return {"type": "LINEAR", "coeff": coeff, "feat": feature, "offset": 0}
    return None


def _ceil_div_fit(
    samples: Sequence[tuple[Mapping[str, Any], int]],
    feature_tables: Sequence[Mapping[str, int]],
) -> dict[str, Any] | None:
    """Try ``value == ceil(feature / divisor)`` for an integer divisor.

    Domain family for chunk/segment counts (review §3.3): the number of
    fixed-size chunks a variable-sized quantity is split into.  The divisor
    is recovered from the intersection of the per-sample feasibility ranges
    of ``ceil(f/d) == value`` — no brute-force scan needed.
    """

    if len(samples) < 2:
        return None
    for feature in feature_tables[0]:
        lo, hi = 2, None
        feasible = True
        for (config, value), table in zip(samples, feature_tables):
            f_value = table.get(feature)
            if f_value is None or f_value <= 0 or value <= 0:
                feasible = False
                break
            if value == 1:
                # ceil(f/d) == 1  ⟺  d >= f
                range_lo, range_hi = f_value, None
            else:
                # ceil(f/d) == v (v ≥ 2)  ⟺  f/v ≤ d < f/(v-1)
                range_lo = -(-f_value // value)
                range_hi = (f_value - 1) // (value - 1)
            lo = max(lo, range_lo)
            if range_hi is not None:
                hi = range_hi if hi is None else min(hi, range_hi)
            if hi is not None and lo > hi:
                feasible = False
                break
        if feasible and lo <= (hi if hi is not None else lo):
            divisor = max(lo, 2)
            if hi is None or divisor <= hi:
                return {"type": "DOMAIN_CEIL_DIV", "feat": feature, "divisor": divisor}
    return None


def _proposal_exact(formula: Mapping[str, Any], samples: Sequence[tuple[Mapping[str, Any], int]]) -> bool:
    """The existing exact gate: zero residual on EVERY sample.

    Shared by the semantic and statistical proposal tails below - a formula
    that misses a single anchor can never leave ``derive_value_formula``.
    """

    for config, value in samples:
        if evaluate_formula_safe(formula, config) != value:
            return False
    return True


def _stat_fit_fallback(
    samples: Sequence[tuple[Mapping[str, Any], int]],
    feature_tables: Sequence[Mapping[str, int]],
) -> tuple[dict[str, Any] | None, str]:
    """Proposal tail after the exact enumeration exhausts (stat_fit seam).

    The statistical proposer ``template_engine.stat_fit`` proposes candidates
    here.  Statistics NEVER serve residuals: every proposal must pass the
    exact gate above (0 residual on all samples, outliers included) before it
    can replace the FAILED marker.  On total failure the returned reason
    annotates the FAILED formula as a diagnostics-only ``stat_reason`` (the
    EXACT_LOOKUP conversion keeps the accounting unchanged); serving
    behaviour is untouched.

    Note: ``shape_rules.derive_shape_semantically`` is an OP-level API (it
    consumes an op carrying ``func``/``args`` and derives an output shape
    tree) - it is NOT a scalar-formula proposer over ``(config, value)``
    samples, so it has no call site here (review issue #25).  Its independent
    public capability remains independent of scalar fitting.
    """

    try:
        from template_engine.stat_fit import propose  # noqa: PLC0415
    except ImportError:  # pragma: no cover - stat_fit ships with this module
        return None, "stat_fit unavailable"
    except Exception as exc:  # noqa: BLE001 - proposer failures are never fatal
        return None, f"stat_fit error: {exc}"
    try:
        proposals = propose(
            [config for config, _ in samples],
            [int(value) for _, value in samples],
            list(feature_tables),
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"stat_fit error: {exc}"

    best_note = ""
    for formula, diagnostics in proposals:
        if not diagnostics.get("exact"):
            best_note = diagnostics.get("note", best_note)
            continue
        if _proposal_exact(formula, samples):
            return formula, diagnostics.get("note", "stat_fit exact proposal")
    outlier_note = ""
    for _, diagnostics in proposals:
        if diagnostics.get("outliers"):
            outlier_note = f"; LOO-flagged capture anomalies at samples {diagnostics['outliers']}"
            break
    return None, f"stat_fit: no proposal exact on all {len(samples)} samples{outlier_note}" + (
        f" ({best_note})" if best_note else ""
    )


_FORMULA_CACHE: ContextVar[dict | None] = ContextVar("template_formula_cache", default=None)


@contextmanager
def cache_value_formulas():
    """Reuse pure fits within one build, without sharing mutable formulas."""
    token = _FORMULA_CACHE.set({})
    try:
        yield
    finally:
        _FORMULA_CACHE.reset(token)


def derive_value_formula(samples: Sequence[tuple[Mapping[str, Any], int]]) -> dict[str, Any]:
    """Derive an exact formula, optionally reusing this build's identical fit."""
    cache = _FORMULA_CACHE.get()
    if cache is None:
        return _derive_value_formula_uncached(samples)
    key = (
        tuple(sorted(MODEL_CONSTANTS.items())),
        tuple((tuple(sorted(config.items())), int(value)) for config, value in samples),
    )
    try:
        hash(key)
    except TypeError:
        return _derive_value_formula_uncached(samples)
    if key not in cache:
        formula = _derive_value_formula_uncached(samples)
        if len(cache) >= 10000:
            return formula
        cache[key] = formula
    return deepcopy(cache[key])


def _derive_value_formula_uncached(samples: Sequence[tuple[Mapping[str, Any], int]]) -> dict[str, Any]:
    """Derive an exact formula for one numeric quantity.

    ``samples`` is a sequence of ``(config, observed_value)`` pairs.  The
    decision order follows the specification: constant, single-feature
    (identity, proportional, linear), then two-feature MULTI, and finally
    ``FAILED``.  Every candidate is verified against *all* samples; a formula
    that misses a single anchor is rejected.  After the hand enumeration
    exhausts, semantic shape rules and the statistical proposer get one
    chance to propose a formula that still passes the same exact gate
    (``_stat_fit_fallback``); only then does the derivation return
    ``FAILED`` (annotated with a diagnostics-only ``stat_reason``).
    """

    if not samples:
        return dict(FAILED_FORMULA)
    feature_tables = [_features_for(config) for config, _ in samples]
    values = [int(value) for _, value in samples]

    if all(value == values[0] for value in values):
        return const_formula(values[0])

    paired = list(zip(feature_tables, values))
    # Pass 1: identity on each feature (dict order approximates simplicity).
    for feature in feature_tables[0]:
        if all(table.get(feature) == value for table, value in paired):
            return {"type": "LINEAR", "coeff": 1, "feat": feature, "offset": 0}
    # Pass 2: integer proportional.
    for feature in feature_tables[0]:
        fit = _proportional_fit(paired, feature)
        if fit is not None:
            return fit
    # Pass 3: general single-feature linear.
    for feature in feature_tables[0]:
        fit = _linear_fit(paired, feature)
        if fit is not None:
            return fit
    # Pass 4: chunk/segment ceil-division family (review §3.3).  The
    # former domain-division and step families (issue #11) were removed:
    # zero delivered uses, and a future fit of those shapes now falls to
    # FAILED -> EXACT_LOOKUP conservative semantics.
    # Single-feature candidates intentionally precede the two-feature MULTI:
    # when both fit every anchor exactly, the single-feature formula is the
    # more constrained hypothesis and extrapolates more reliably to unseen
    # (B, S) combinations (e.g. ceil(S/256) chunk counts beating a spurious
    # -B+S/128 MULTI that threads the same three anchors).
    ceil_fit = _ceil_div_fit(samples, feature_tables)
    if ceil_fit is not None:
        return ceil_fit
    # Pass 5: two-feature MULTI on the atomic shortlist.
    available = [name for name in _MULTI_FEATURE_SHORTLIST if name in feature_tables[0]]
    for pair in combinations(available, 2):
        fit = _multi_fit(paired, pair)
        if fit is not None:
            return fit
    # Pass 6: two-feature MULTI with offset, then three-feature MULTI.  Both
    # solve 3 unknowns, so they require >= 4 samples for at least one anchor
    # to cross-check the Cramer solve (3 equations for 3 unknowns would
    # overfit identically).  Only features that actually vary across the
    # samples enter the search - constant features cannot move the value and
    # would explode the combination space.
    varying = _varying_features(paired, available)
    if len(samples) >= 4 and len(varying) >= 2:
        for pair in combinations(varying, 2):
            fit = _multi_fit_offset(paired, pair)
            if fit is not None:
                return fit
        if len(varying) >= 3:
            for triple in combinations(varying, 3):
                fit = _multi3_fit(paired, triple)
                if fit is not None:
                    return fit
    # Exact enumeration exhausted: semantic rules, then stat_fit, each gated
    # by the same 0-residual check on all samples.
    fallback, stat_reason = _stat_fit_fallback(samples, feature_tables)
    if fallback is not None:
        return fallback
    failed = dict(FAILED_FORMULA)
    if stat_reason:
        # Diagnostics-only annotation (the EXACT_LOOKUP conversion drops
        # unknown keys; the stat_fit PROPOSAL_LOG is the durable record).
        failed["stat_reason"] = stat_reason
    return failed


def evaluate_formula_safe(formula: Mapping[str, Any] | None, config: Mapping[str, Any]) -> Any:
    """Evaluate a derived formula, returning ``None`` on FAILED/unsupported."""

    if formula is None:
        return None
    if formula.get("type") == "FAILED":
        return None
    try:
        return eval_formula(dict(formula), config)
    except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None


def template_has_failed_formula(formula: Any) -> bool:
    """Recursively detect ``FAILED`` markers inside a formula tree."""

    if isinstance(formula, Mapping):
        if formula.get("type") == "FAILED":
            return True
        return any(template_has_failed_formula(value) for value in formula.values())
    if isinstance(formula, (list, tuple, set)):
        return any(template_has_failed_formula(value) for value in formula)
    return False


def count_failed_formulas(formula: Any) -> int:
    """Count ``FAILED`` leaves inside a formula tree (for coverage reports)."""

    if isinstance(formula, Mapping):
        if formula.get("type") == "FAILED":
            return 1
        return sum(count_failed_formulas(value) for value in formula.values())
    if isinstance(formula, (list, tuple, set)):
        return sum(count_failed_formulas(value) for value in formula)
    return 0


def model_constant_names() -> tuple[str, ...]:
    return tuple(sorted(MODEL_CONSTANTS))


__all__ = [
    "FAILED_FORMULA",
    "const_formula",
    "count_failed_formulas",
    "derive_value_formula",
    "evaluate_formula_safe",
    "model_constant_names",
    "template_has_failed_formula",
]
