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

"""One body resolver shared by coverage and expansion.

New templates enumerate discrete cells; within a cell they partition op
structure and tensor reference topology. Exact anchors, a unique containing
B/S span resolve multiple candidates.
Unresolved structure switches refuse. Legacy mode predicates remain readable.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from .mode_router import mode_features

#: Config features that pin the structure mode independently of (B, S).
#: Everything derived from B/S (scatter/pad, per-rank token windows,
#: b-oddity) is intentionally absent: those belong to the (B, S) span
#: discipline, and a discrete misroute would silently swap structures.
DISCRETE_FEATURES: tuple[str, ...] = (
    "TP",
    "EP",
    "MD",
    "DCP",
    "mtp",
    "dfc",
    "dec",
    "qlin",
    "qattn",
    "unroll",
    "md_eq_tp",
    "md_gt_tp",
)

#: Selection status returned by :func:`select_body`.
STATUS_SINGLE = "single"
STATUS_MATCHED = "predicate"
STATUS_EXACT = "exact_member"
STATUS_SPAN = "span_contained"
STATUS_AMBIGUOUS = "mode_ambiguous"
STATUS_NONE = "no_candidate"
STATUS_BRANCH_MISSING = 'prefill_structure_branch'
STATUS_STRUCTURE_MISSING = 'structure_bucket_missing'


def discrete_cell(config: Mapping[str, Any]) -> tuple[Any, ...]:
    """The captured configuration axes that B/S fitting must never cross."""
    return (
        *(int(config.get(name, 1)) for name in ("TP", "EP", "MOE_DP", "DCP")),
        int(config.get("num_mtp_tokens", 0) or 0),
        bool(config.get("enable_dispatch_ffn_combine", False)),
        bool(config.get("is_decode", False)),
        config.get("quantize_linear_action", "DISABLED"),
        config.get("quantize_attention_action", "DISABLED"),
        config.get("do_compile"),
    )


def _norm(value: Any) -> Any:
    """JSON-round-trip safe normalisation (lists -> tuples, bools stay)."""

    if isinstance(value, list):
        return tuple(_norm(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_norm(item) for item in value)
    return value


def base_key(group_key: Any) -> tuple[Any, ...]:
    """The merge-scope tuple of a (possibly mode-tagged) group key.

    Serving only ever compares the merge scope, so every historical key
    layout stays readable (review issue #3 - flat keys):

    - bare merge scope (a plain ``group(4)``, e.g. validation-internal
      single-group templates) - returned as-is;
    - group-head layouts - ``group(4) + ...``: the delivered v0.9 artifact
      nests the cell (``group(4) + (cell,) + mode``), the current builder
      concatenates it (``group(4) + cell(10) [+ mode]``) - the scope is
      the literal first four components;
    - flat (new): ``cell(10) [+ mode]`` - every merge-scope axis is
      derivable from the cell (quant actions at 7/8, decode at 6, par_cat
      as the >1 projection of the parallel values at 0..3), so nothing is
      stored twice and the scope is derived on read.
    """

    key = tuple(_norm(group_key))
    if len(key) == 4:
        return key
    if len(key) in (10, 11):
        cell = key[:10]
        par_dimensions = (cell[0] > 1, cell[1] > 1, cell[2] > 1, cell[3] > 1)
        par_cat: Any = "single" if sum(par_dimensions) <= 1 else par_dimensions
        return (cell[7], cell[8], par_cat, cell[6])
    return key[:4]


def structure_fingerprint(ops, *, parameter_lists=False):
    """Operator/reference topology; scalar parameters and shapes may vary."""
    from .compare_ops import tensor_alias_paths
    from .mode_router import mode_fingerprint

    def structure(value: Any) -> Any:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return ("tensor", len(value.shape), str(value.dtype), bool(value.requires_grad))
        if isinstance(value, Mapping):
            return ("dict", tuple((key, structure(value[key])) for key in sorted(value, key=str)))
        if isinstance(value, (list, tuple)):
            # Request phase metadata is a repeated scalar parameter, not a
            # sequence of operators/tensor references. Its length belongs to
            # the parameter formula (e.g. ceil(B/DP)), including in Prefill.
            if parameter_lists and value and all(type(item) is int for item in value):
                return (type(value).__name__, 'integer_parameters')
            if parameter_lists and value and all(type(item) is bool and item is value[0] for item in value):
                return (type(value).__name__, 'repeated_bool', value[0])
            return (type(value).__name__, tuple(structure(item) for item in value))
        # Integer leaves are fitted; other leaves must be constant within a body.
        return ("int",) if type(value) is int else (type(value).__name__, repr(value))

    return (
        mode_fingerprint(ops),
        tuple(structure((op.args, op.kwargs or {}, op.out)) for op in ops),
        tuple(tensor_alias_paths(ops).items()),
    )


def partition_by_mode(captures: Sequence[Any], *, parameter_lists=None) -> OrderedDict[tuple[str, ...], list[Any]]:
    """Partition captures by structure-mode fingerprint (stable order)."""
    if parameter_lists is None:
        parameter_lists = all(not c.config.get('is_decode', False) for c in captures)
    partitions: OrderedDict[tuple[str, ...], list[Any]] = OrderedDict()
    for capture in captures:
        signature = structure_fingerprint(capture.ops, parameter_lists=parameter_lists)
        partitions.setdefault(signature, []).append(capture)
    return partitions


def discrete_predicate(captures: Sequence[Any]) -> dict[str, list[Any]]:
    """Set-valued conjunction over :data:`DISCRETE_FEATURES` for members."""

    values: dict[str, set[Any]] = {feat: set() for feat in DISCRETE_FEATURES}
    for capture in captures:
        feats = mode_features(capture.config)
        for feat in DISCRETE_FEATURES:
            values[feat].add(feats.get(feat))
    return {feat: sorted(vals, key=str) for feat, vals in values.items()}


def _config_matches_predicate(predicate: Mapping[str, Sequence[Any]], config: Mapping[str, Any]) -> bool:
    feats = mode_features(config)
    for feat, allowed in predicate.items():
        if feats.get(feat) not in set(allowed):
            return False
    return True


def _same_config(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return discrete_cell(a) == discrete_cell(b) and all(
        int(a.get(name, 0)) == int(b.get(name, 0)) for name in ("B", "S")
    )


def _span_contains(members: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    """Whether the members' (B, S) span contains the target under span discipline.

    Same rules as the (B, S) gate: a rank-1 span (all members share the
    target's B, or all share its S) needs >=2 anchors; a rank-2 bbox needs
    >=4 anchors with both dimensions varying (the 3-anchor rank-2 fit is
    under-determined and mispredicted interior points - P2 evidence).
    """

    points = {(int(m.get("B", 0)), int(m.get("S", 0))) for m in members}
    if not points:
        return False
    tb, ts = int(config.get("B", 0)), int(config.get("S", 0))
    if (tb, ts) in points:
        return True
    same_b = {s for b, s in points if b == tb}
    if len(same_b) >= 2 and min(same_b) <= ts <= max(same_b):
        return True
    same_s = {b for b, s in points if s == ts}
    if len(same_s) >= 2 and min(same_s) <= tb <= max(same_s):
        return True
    if len(points) >= 4 and len({b for b, _ in points}) >= 2 and len({s for _, s in points}) >= 2:
        bs = {b for b, _ in points}
        ss = {s for _, s in points}
        return min(bs) <= tb <= max(bs) and min(ss) <= ts <= max(ss)
    return False


def select_body(
    groups: Mapping[Any, Any],
    target_key4: tuple[Any, ...],
    config: Mapping[str, Any],
) -> tuple[Any, Any, str]:
    """Resolve the mode body for ``config``.

    Returns ``(key, group, status)``.  ``key``/``group`` are ``None`` when
    no candidate shares the base key; ``status`` distinguishes a clean
    single candidate, a predicate match, an exact-anchor hit, an
    unresolvable tie (:data:`STATUS_AMBIGUOUS`) and an empty candidate set.
    """

    target = tuple(_norm(target_key4))

    def domain_matches(group):
        from .serve_rules import prefill_window_blocker
        from .structure_buckets import bucket_matches

        meta = group.get("mode_meta") or {}
        if meta.get('structure_bucket'):
            return bucket_matches(group, config)
        if meta.get('prefill_window_branch') and prefill_window_blocker(group, config):
            return False
        return True

    scoped = [
        (key, group)
        for key, group in groups.items()
        if base_key(key) == target
        and ("discrete_cell" not in group or tuple(_norm(group["discrete_cell"])) == discrete_cell(config))
    ]
    candidates = [(key, group) for key, group in scoped if domain_matches(group)]
    if not candidates:
        if any((group.get('mode_meta') or {}).get('structure_bucket') for _, group in scoped):
            return None, None, STATUS_STRUCTURE_MISSING
        if any((group.get('mode_meta') or {}).get('prefill_window_branch') for _, group in scoped):
            return None, None, STATUS_BRANCH_MISSING
        return None, None, STATUS_NONE
    if len(candidates) == 1:
        key, group = candidates[0]
        if not _config_matches_predicate((group.get("mode_meta") or {}).get("discrete_predicate", {}), config):
            return None, None, STATUS_NONE
        return key, group, STATUS_SINGLE

    eligible = [
        (key, group)
        for key, group in candidates
        if _config_matches_predicate((group.get("mode_meta") or {}).get("discrete_predicate", {}), config)
    ]
    if not eligible:
        return None, None, STATUS_NONE
    if len(eligible) == 1:
        return eligible[0][0], eligible[0][1], STATUS_MATCHED

    exact_hits = [
        (key, group)
        for key, group in eligible
        if any(_same_config(member, config) for member in (group.get("mode_meta") or {}).get("member_configs", []))
    ]
    if len(exact_hits) == 1:
        return exact_hits[0][0], exact_hits[0][1], STATUS_EXACT

    # A rectangle of observed B/S values cannot prove structure membership.
    # New fixed buckets resolve by official predicates or exact anchors only.
    if any((group.get('mode_meta') or {}).get('structure_bucket') for _, group in eligible):
        return eligible[0][0], eligible[0][1], STATUS_AMBIGUOUS

    # Span-containment tiebreak: when several mode cells share the config's
    # discrete features, the cell whose member (B, S) span contains the
    # target under span discipline owns the target - the identical
    # structure fingerprints of its anchors are empirical evidence the
    # (B, S)-dependent driver does not switch inside that span.  Zero or
    # several containing cells stay ambiguous (honest; never guessed).
    containing = [
        (key, group)
        for key, group in eligible
        if _span_contains((group.get("mode_meta") or {}).get("member_configs", []), config)
    ]
    if len(containing) == 1:
        return containing[0][0], containing[0][1], STATUS_SPAN
    return eligible[0][0], eligible[0][1], STATUS_AMBIGUOUS
