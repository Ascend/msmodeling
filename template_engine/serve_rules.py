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

"""Serve-admission rules: the single implementation home for the rules that
decide whether a template can serve a config.

Ownership (review issue #6 - judgement convergence): the *coverage gate*
(:mod:`template_engine.coverage`) is the ONLY runtime adjudicator - it
executes these rules once and produces the structured verdict (``exact_group``
/ ``interpolated`` / ``fallback_group`` + ``blocker``); the *expander*
(:mod:`template_generate`) delegates strict-mode admission to the gate and its
gate-internal call passes ``prevalidated=True`` so the rules run exactly once
per serve.  Everything here is a pure rule over (template structure, config):
no capture, no expansion, no I/O.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------------------
# Key vocabulary (list/tuple normalisation + distance) shared by the gate and expander
# --------------------------------------------------------------------------------------


def prefill_window_tokens(config):
    """Packed prefill length passed as sl by the official input generator."""
    from .structure_buckets import parallel_geometry

    _, local_batch = parallel_geometry(config)
    return local_batch * int(config['S'])


def prefill_window_regime(config, window):
    """Official sliding-window prefill write branch over packed tokens."""
    tokens = prefill_window_tokens(config)
    if tokens <= window:
        return 'short'
    return 'aligned' if tokens % window == 0 else 'wrapped'


def prefill_structure_domain(config, window):
    """Official window, empty compressed-stream and empty expert conditions.

    Empty slices can be removed by compilation. These conditions must not
    be crossed by an otherwise identical captured operator body.
    """
    from .parameter_features import prefill_routing_features

    parameters = dict(config.get('model_parameters') or ())
    tokens = prefill_window_tokens(config)
    ratios = sorted({value for name, value in parameters.items() if name.startswith('COMPRESS_RATIO/')})
    features = prefill_routing_features(
        int(config['B']),
        int(config['S']),
        int(config.get('TP', 1)),
        int(config.get('EP', 1)),
        int(config.get('MOE_DP', 1)),
        parameters,
    )
    domain = {
        'window': prefill_window_regime(config, window) if window else None,
        'empty_compressed': [ratio for ratio in ratios if tokens < ratio and int(config['S']) != 1],
        'empty_experts': sorted(name for name, value in features.items() if '/input_expert/' in name and value == 0),
    }
    if ratios:
        from tensor_cast.layers.moe_layer import assign_experts

        if features:
            start, count = assign_experts(parameters['E'], int(config.get('EP', 1)), 0)
            domain['local_empty_experts'] = [
                index for index in range(start, start + count) if features.get(f'moe_route/input_expert/{index}') == 0
            ]
        # Deepseek V4's attention wrapper uses per-request query_len < 5;
        # compressor separately distinguishes concrete single-token queries.
        sequence = int(config['S'])
        domain['query_kind'] = 'single' if sequence == 1 else 'short' if sequence < 5 else 'normal'
        domain['packed_single_token'] = tokens == 1
    return domain


def domain_matches(expected, actual):
    """Compare the conditions recorded by a bucket, including new conditions.

    Previously shipped buckets retain their recorded condition vocabulary.
    Newly built buckets record every current discriminator.
    """
    # Dispatch send sizes remain a fixed scalar list; only this rank's expert
    # slices affect its compiled operator stream. Keep the historical global
    # condition for shipped descriptors, while new descriptors use local slices.
    return all(
        actual.get(key) == value
        for key, value in expected.items()
        if key != 'empty_experts' or 'local_empty_experts' not in expected
    )


def prefill_window_blocker(group, config):
    branch = (group.get('mode_meta') or {}).get('prefill_window_branch')
    if branch and not config.get('is_decode'):
        domains = branch.get('domains')
        actual = prefill_structure_domain(config, int(branch['window']))
        if domains is not None and not any(domain_matches(domain, actual) for domain in domains):
            return 'prefill_structure_branch: no body for window/compressed-stream/expert domain'
        regime = prefill_window_regime(config, int(branch['window']))
        if regime not in branch['regimes']:
            return f'prefill_window_write_branch: missing {regime} body'
    alignment = (group.get('mode_meta') or {}).get('prefill_window_alignment')
    if alignment and not config.get('is_decode'):
        sequence = int(config['S'])
        if sequence <= alignment or sequence % alignment:
            return f'prefill_window_write_branch: template requires S>{alignment} and S%{alignment}=0'
    return None


def signature_fields(group):
    fields = (
        'par_sig',
        'scatter_cat',
        'pad_cat',
        'num_mtp_tokens',
        'enable_dispatch_ffn_combine',
        'is_decode',
        'do_compile',
    )
    meta = group.get('mode_meta') or {}
    if meta.get('parameterized_prefill') or meta.get('structure_bucket'):
        return tuple(field for field in fields if field not in ('scatter_cat', 'pad_cat'))
    return fields


def _normalise_key(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_key(item) for item in value)
    return value


def _key_equal(left: Any, right: Any) -> bool:
    return _normalise_key(left) == _normalise_key(right)


def _group_distance(target: Sequence[Any], candidate: Sequence[Any]) -> float:
    if len(target) != len(candidate):
        return float(abs(len(target) - len(candidate)) + 100)
    distance = 0.0
    for target_value, candidate_value in zip(target, candidate):
        if _key_equal(target_value, candidate_value):
            continue
        if isinstance(target_value, Number) and isinstance(candidate_value, Number):
            distance += abs(float(target_value) - float(candidate_value))
        else:
            distance += 1.0
    return distance


# --------------------------------------------------------------------------------------
# Signature categories (scatter/pad) - the MoE dispatch geometry of a config
# --------------------------------------------------------------------------------------


def _scatter_category(batch: int, sequence: int, ep: int) -> bool:
    from template_generate import _floor_div  # noqa: PLC0415 - avoid an import cycle

    tokens_per_ep = _floor_div(batch * sequence, ep) if ep > 0 else 0
    return tokens_per_ep % 128 == 0


def _pad_cat_from_config(B: int, S: int, TP: int, EP: int, constants=None) -> bool:
    """Predict whether MoE expert padding is present for a target config."""

    from template_generate import MODEL_CONSTANTS  # noqa: PLC0415 - avoid an import cycle

    if TP <= 1:
        return False
    legacy = constants is None
    constants = MODEL_CONSTANTS if legacy else constants
    if not constants.get("E") or not constants.get("E_per_tok"):
        return False
    from fractions import Fraction

    k = constants["E"] // constants["E_per_tok"] if legacy else Fraction(constants["E"], constants["E_per_tok"])
    tp_eff = 1 if EP <= 1 else TP
    numerator, denominator = B * S * EP, tp_eff * k
    if denominator <= 0:
        return False
    if numerator % denominator != 0:
        return True
    return (numerator // denominator) % TP != 0


def _signature_categories(config: Mapping[str, Any]) -> tuple[bool, bool]:
    """Signature (scatter, pad) categories for a capture/target config.

    Decode forwards dispatch only their query tokens (``num_mtp_tokens + 1``
    per request), not the context length - categorising decode on ``B * S``
    mis-tags captures whose S happens to be a multiple of 128 and then
    rejects same-structure targets (v4 decode-anchor lesson); prefill keeps
    the full ``B * S``.
    """

    batch = int(config.get("B", 0) or 0)
    ep = int(config.get("EP", 1) or 1)
    if bool(config.get("is_decode", False)):
        sequence = int(config.get("num_mtp_tokens", 0) or 0) + 1
    else:
        sequence = int(config.get("S", 0) or 0)
    return (
        _scatter_category(batch, sequence, ep),
        _pad_cat_from_config(
            batch,
            sequence,
            int(config.get("TP", 1) or 1),
            ep,
            dict(config["model_parameters"]) if "model_parameters" in config else None,
        ),
    )


# --------------------------------------------------------------------------------------
# Parallel-variant selection (shared by the scoped lookup walk and the expander)
# --------------------------------------------------------------------------------------


def _moe_tp(par_sig: Sequence[int]) -> int:
    from template_generate import _floor_div  # noqa: PLC0415 - avoid an import cycle

    tp, ep, moe_dp, dcp = (int(value) for value in par_sig)
    work_size = max(tp, ep, dcp) * moe_dp
    denominator = moe_dp * ep
    return int(_floor_div(work_size, denominator)) if denominator else 0


def _parallel_dimension_distance(target: int, candidate: int) -> int:
    if target == candidate:
        return 0
    if (target > 1) == (candidate > 1):
        return 1
    return 2


def _match_parallel_variant(
    variants: Sequence[Mapping[str, Any]],
    tp: int,
    ep: int,
    md: int,
    dcp: int,
    target_scatter_cat: bool | None = None,
    target_pad_cat: bool | None = None,
) -> Mapping[str, Any] | None:
    """Select the closest parallel variant using the documented penalties."""

    target_sig = (tp, ep, md, dcp)
    best: tuple[tuple[float, int, float], Mapping[str, Any]] | None = None
    for variant in variants:
        for raw_key in variant.get("var_keys", []):
            if len(raw_key) != 3:
                continue
            candidate_sig = tuple(int(value) for value in raw_key[0])
            dimension_distance = sum(
                _parallel_dimension_distance(target_value, candidate_value)
                for target_value, candidate_value in zip(target_sig, candidate_sig)
            )
            category_match = sum(
                (target_value > 1) == (candidate_value > 1)
                for target_value, candidate_value in zip(target_sig, candidate_sig)
            )
            value_distance = sum(
                abs(target_value - candidate_value) for target_value, candidate_value in zip(target_sig, candidate_sig)
            )
            distance = float(dimension_distance)
            if _moe_tp(target_sig) != _moe_tp(candidate_sig):
                distance += 20
            if target_scatter_cat is not None and bool(raw_key[1]) != target_scatter_cat:
                distance += 10
            if target_pad_cat is not None and bool(raw_key[2]) != target_pad_cat:
                distance += 5
            score = (distance, -category_match, float(value_distance))
            if best is None or score < best[0]:
                best = (score, variant)
    return best[1] if best else None


def _select_parallel_variant(
    variants: Sequence[Mapping[str, Any]],
    tp: int,
    ep: int,
    md: int,
    dcp: int,
    scatter_cat: bool,
    pad_cat: bool,
) -> tuple[Mapping[str, Any] | None, bool]:
    target_key = ((tp, ep, md, dcp), scatter_cat, pad_cat)
    target_sig = target_key[0]
    for variant in variants:
        if any(_key_equal(raw_key, target_key) for raw_key in variant.get("var_keys", [])):
            return variant, False
    for variant in variants:
        if any(_key_equal(_normalise_key(raw_key)[0], target_sig) for raw_key in variant.get("var_keys", [])):
            return variant, False
    return _match_parallel_variant(variants, tp, ep, md, dcp, scatter_cat, pad_cat), True


# --------------------------------------------------------------------------------------
# EXACT_LOOKUP admission rule (the scoped walk is the only form: it visits exactly
# the nodes expansion would reach, so sibling variants never block a config)
# --------------------------------------------------------------------------------------

_LOOKUP_FIELDS = (
    "B",
    "S",
    "TP",
    "EP",
    "MOE_DP",
    "DCP",
    "num_mtp_tokens",
    "enable_dispatch_ffn_combine",
    "is_decode",
    "do_compile",
    "quantize_linear_action",
    "quantize_attention_action",
)

#: Canonical defaults for absent optional fields; TP/EP/MOE_DP/DCP default to 1.
_LOOKUP_DEFAULTS: dict[str, Any] = {
    "B": None,
    "S": None,
    "do_compile": None,
    "num_mtp_tokens": 0,
    "enable_dispatch_ffn_combine": False,
    "is_decode": False,
    "quantize_linear_action": "DISABLED",
    "quantize_attention_action": "DISABLED",
}


def _lookup_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical identity view of a config for EXACT_LOOKUP matching."""

    return {field: config[field] if field in config else _LOOKUP_DEFAULTS.get(field, 1) for field in _LOOKUP_FIELDS}


def exact_lookup_covered_scoped(structure: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    """True when ``config`` satisfies every EXACT_LOOKUP dim expansion reaches.

    Walks the structure the way expansion does: at every node carrying
    ``parallel_variants`` only the variant selected for ``config`` is
    visited, and ``moe_fold`` nodes visit only the dispatch-ffn-combine
    branch the config expands into.  A config must therefore satisfy only
    the lookup dims it can actually reach, not those of sibling variants
    (the md1-anchor-blocked-by-md2/md4-variant lesson).  Groups whose
    entries carry exact-lookup dimensions are only verified at their
    captured configurations; any other target is refused by the caller
    (coverage verdict downgrade / strict expansion error).
    """

    tp = int(config.get("TP", 1))
    ep = int(config.get("EP", 1))
    md = int(config.get("MOE_DP", 1))
    dcp = int(config.get("DCP", 1))
    scatter_cat, pad_cat = _signature_categories(config)
    view = _lookup_config_view(config)

    def matches(candidate: Any) -> bool:
        if not isinstance(candidate, Mapping):
            return False
        match = _lookup_config_view(candidate.get("match", {}))
        return all(view.get(key) == value for key, value in match.items())

    def covered(node: Any) -> bool:
        if isinstance(node, Mapping):
            if str(node.get("type", "")).upper() == "EXACT_LOOKUP":
                return any(matches(candidate) for candidate in node.get("entries", ()))
            variants = node.get("parallel_variants")
            if variants is not None:
                selected, _ = _select_parallel_variant(variants, tp, ep, md, dcp, scatter_cat, pad_cat)
                entries = selected.get("entries", []) if selected is not None else []
                return all(covered(entry) for entry in entries)
            return all(covered(value) for value in node.values())
        if isinstance(node, (list, tuple)):
            return all(covered(item) for item in node)
        return True

    return covered(structure)


# --------------------------------------------------------------------------------------
# (B, S) span discipline
# --------------------------------------------------------------------------------------


def bs_span_blocker(
    points: set[tuple[int, int]],
    batch: int,
    sequence: int,
    formulas_exact: bool = True,
) -> str | None:
    """(B, S) extrapolation discipline for a signature's captured points.

    Returns the refusing blocker, or ``None`` when the target rides
    verified formulas.  Gate v2 (pilot-evidenced): <2 points -> every
    (B,S)-dependent dim is CONST-frozen (``bs_samples``); a rank-1
    COLLINEAR span is token-algebraic ON the line when formulas are exact
    (0 shape mismatches at (3,768) from a (2,512)/(4,1024) fit), off the
    line the B-only vs B*S dependence is unidentifiable (18 mismatches at
    (8,512)) -> ``bs_offline``; a rank-2 span identifies both axes ->
    serve anywhere when formulas are exact, else fall back to the
    bounding box (``bs_envelope``); an axis-aligned rank-1 span keeps the
    frozen-axis rule (``bs_axis_frozen``).  Rank-2 spans under 4 anchors
    under-determine the feature family (53 interior mismatches on the
    dfc-on pilot) -> ``bs_offline`` unless over-determined.
    """

    if (batch, sequence) in points:
        return None
    if len(points) < 2:
        return "bs_samples"
    same_b = len({b for b, _ in points}) == 1
    same_s = len({s for _, s in points}) == 1
    if same_b or same_s:
        frozen_b = next(iter(points))[0]
        frozen_s = next(iter(points))[1]
        if (same_b and batch != frozen_b) or (same_s and sequence != frozen_s):
            return "bs_axis_frozen"
        if not formulas_exact and not (
            min(b for b, _ in points) <= batch <= max(b for b, _ in points)
            and min(s for _, s in points) <= sequence <= max(s for _, s in points)
        ):
            return "bs_envelope"
        return None
    pts = sorted(points)
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    collinear = all((x - x1) * (y2 - y1) == (y - y1) * (x2 - x1) for x, y in pts[1:-1]) and (x2 != x1 or y2 != y1)
    if collinear:
        on_line = (batch - x1) * (y2 - y1) == (sequence - y1) * (x2 - x1)
        if on_line and formulas_exact:
            return None
        return "bs_offline"
    if len(points) >= 4:
        if formulas_exact:
            return None
        if not (
            min(b for b, _ in points) <= batch <= max(b for b, _ in points)
            and min(s for _, s in points) <= sequence <= max(s for _, s in points)
        ):
            return "bs_envelope"
        return None
    return "bs_offline"


#: Historic private name (the rule used to live in the expander module).
_bs_span_blocker = bs_span_blocker


__all__ = [
    "bs_span_blocker",
    "exact_lookup_covered_scoped",
]
