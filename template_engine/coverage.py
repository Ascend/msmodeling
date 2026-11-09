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

"""Select one fixed operator body and report whether a configuration is covered.

``exact_group`` denotes an observed anchor; ``interpolated`` denotes a new
B/S point served by verified formulas. ``fallback_group`` is an unsupported
configuration, never permission to select a different structure by proximity.
The gate optionally returns the exact operator stream used by callers.
"""

from __future__ import annotations

from typing import Any, Mapping

from template_generate import TemplateExpandError, generate_ops_from_template
from template_engine.serve_rules import (
    _bs_span_blocker,
    _group_distance,
    _key_equal,
    _normalise_key,
    _signature_categories,
    exact_lookup_covered_scoped,
)


def _group_key_of(config: Mapping[str, Any]) -> tuple[Any, ...]:
    """Mirror template_generate.generate_ops_from_template's target key."""

    parallel_dimensions = (
        int(config.get("TP", 1)) > 1,
        int(config.get("EP", 1)) > 1,
        int(config.get("MOE_DP", 1)) > 1,
        int(config.get("DCP", 1)) > 1,
    )
    par_cat: Any = "single" if sum(parallel_dimensions) <= 1 else parallel_dimensions
    return (
        config.get("quantize_linear_action", "DISABLED"),
        config.get("quantize_attention_action", "DISABLED"),
        par_cat,
        bool(config.get("is_decode", False)),
    )


def _select_group(
    groups: Mapping[Any, Any],
    target_key: tuple[Any, ...],
    discrete_target: tuple[Any, ...] | None = None,
):
    """Legacy projection scan kept for the ``nearest_distance`` diagnostic.

    Flat group keys are discrete cells (+ a mode suffix), while
    ``target_key`` is the 4-axis legacy projection - comparing them
    cross-format made every distance the bogus ``107`` length penalty
    (review issue #28).  Distances are now computed in the SAME key space
    only: a group's 10-axis cell view competes against the config's
    ``discrete_cell`` target, its 4-axis merge-scope view (``base_key``)
    against the legacy projection, and cross-format pairs are skipped.
    ``exact`` requires a same-space equality; with no same-space candidate
    the distance stays ``None`` (diagnostics omitted by contract).
    """

    from .body_select import base_key

    targets = [target for target in (discrete_target, target_key) if target is not None]
    exact = None
    fallback = None
    fallback_distance = None
    for raw_key, group in groups.items():
        normalised = tuple(_normalise_key(raw_key))
        views = {normalised}
        if len(normalised) >= 10:
            views.add(normalised[:10])
        scope_view = base_key(normalised)
        if isinstance(scope_view, tuple):
            views.add(scope_view)
        for target in targets:
            for view in views:
                if len(view) != len(target):
                    continue
                if _key_equal(view, target):
                    # exact hit: there is no "nearest fallback" to report -
                    # a stale distance from a sibling scan would mislabel a
                    # known unit (review issue #28)
                    return group, None, None
                distance = _group_distance(target, view)
                if fallback_distance is None or distance < fallback_distance:
                    fallback, fallback_distance = group, distance
            if exact is not None:
                break
        if exact is not None:
            break
    return exact, fallback, fallback_distance


def _parallel_variants_of(group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [variant for entry in group.get('entries', []) for variant in entry.get('parallel_variants', [])]


def _served_member_keys(group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    served = group.get("served_keys")
    return [item for item in served if isinstance(item, Mapping)] if isinstance(served, list) else []


def describe_coverage(
    template: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    return_ops: bool = False,
) -> dict[str, Any]:
    """Summarise what expanding ``config`` from ``template`` would use.

    The verdict is ``exact_group`` only when the group key matches AND the
    full (par_sig, scatter_cat, pad_cat, num_mtp_tokens, dfc) signature is
    served by a real capture in that group AND that capture's (B, S) matches
    the target AND the target MTP count does not exceed the trained maximum.
    A served signature at an unseen (B, S) reports ``interpolated`` (formula
    extrapolation).  Everything else reports ``fallback_group`` — the
    expander would otherwise silently reuse a nearest variant/structure
    (review P0-1/P1-2/P1-4).
    """

    groups = template.get("groups", {})
    from .serve_rules import signature_fields
    from .model_parameters import bind_parameters

    config = bind_parameters(template, config)
    from .structure_buckets import prefill_anchor_config

    config = prefill_anchor_config(groups, config)
    config_full = {
        "quantize_linear_action": "DISABLED",
        "quantize_attention_action": "DISABLED",
        **config,
    }
    target_key = _group_key_of(config_full)
    # Mode-granular body selection: a base group may carry several structure
    # bodies (mode fingerprints).  Resolve by discrete-mode predicate with
    # exact-anchor tiebreak; an unresolvable tie refuses honestly.  No body
    # sharing the base key falls through to the legacy "group_key" blocker.
    from .body_select import (
        STATUS_AMBIGUOUS,
        STATUS_BRANCH_MISSING,
        STATUS_STRUCTURE_MISSING,
        discrete_cell,
        select_body,
    )

    _legacy_exact, _legacy_fallback, fallback_distance = _select_group(
        groups, target_key, tuple(discrete_cell(config_full))
    )
    sel_key, selected, sel_status = select_body(groups, target_key, config_full)
    if sel_status in (STATUS_BRANCH_MISSING, STATUS_STRUCTURE_MISSING):
        return {
            'verdict': 'fallback_group',
            'blocker': sel_status,
            'expandable': False,
            'op_count': 0,
            'failed_formulas': 0,
            'reason': 'No fixed operator/reference bucket matches the target execution conditions.',
        }
    if sel_status == STATUS_AMBIGUOUS:
        return {
            "config": {k: config_full[k] for k in sorted(config_full)},
            "target_group_key": str(target_key),
            "verdict": "fallback_group",
            "blocker": "mode_ambiguous",
            "reason": (
                "several structure bodies share this config's discrete features; "
                "the (B,S)-dependent structure driver is not yet characterised - "
                "capture this exact config to disambiguate"
            ),
            "group_key": [list(sel_key)],
            "nearest_distance": fallback_distance,
            "variant_mode": "ambiguous",
        }

    served = _served_member_keys(selected) if selected is not None else []
    # Lazy-shard templates: the variant/MTP introspection below reads the
    # group's body keys (suffix_mtp0 / mtp_structure) - attach the selected
    # group's body blob on first need.  No-op on materialised templates.
    if selected is not None:
        from .zero_meta import ensure_group_body  # noqa: PLC0415

        ensure_group_body(template, sel_key)
    from .serve_rules import prefill_window_blocker

    branch_blocker = prefill_window_blocker(selected, config_full) if selected is not None else None
    if branch_blocker:
        return {
            'verdict': 'fallback_group',
            'blocker': branch_blocker,
            'expandable': False,
            'op_count': 0,
            'failed_formulas': 0,
            'reason': branch_blocker,
        }
    batch = int(config_full["B"])
    sequence = int(config_full["S"])
    tp = int(config_full.get("TP", 1))
    ep = int(config_full.get("EP", 1))
    md = int(config_full.get("MOE_DP", 1))
    dcp = int(config_full.get("DCP", 1))
    num_mtp = int(config_full.get("num_mtp_tokens", 0))
    dfc = bool(config_full.get("enable_dispatch_ffn_combine", False))
    # Decode targets categorise on their query-token geometry (the forward's
    # dispatch payload), matching the decode-aware builder tags - deriving
    # them from the context length mis-tagged the v4 decode anchors and then
    # rejected same-structure targets at off-multiple-of-128 contexts.
    target_scatter, target_pad = _signature_categories(config_full)
    target_member = {
        "par_sig": [tp, ep, md, dcp],
        "scatter_cat": target_scatter,
        "pad_cat": target_pad,
        "num_mtp_tokens": num_mtp,
        "enable_dispatch_ffn_combine": dfc,
        "is_decode": bool(config_full.get("is_decode", False)),
        "do_compile": config_full.get("do_compile"),
    }

    signature_served = False
    bs_served = False
    served_carry_bs = False
    sig_bs_points: set[tuple[int, int]] = set()
    if selected is not None:
        for item in served:
            carries = "B" in item and "S" in item
            served_carry_bs = served_carry_bs or carries
            if all(
                _key_equal(_normalise_key(item.get(field)), _normalise_key(target_member[field]))
                for field in signature_fields(selected)
            ):
                signature_served = True
                if carries:
                    try:
                        sig_bs_points.add((int(item["B"]), int(item["S"])))
                    except (TypeError, ValueError):
                        pass
                if carries and _key_equal(item.get("B"), batch) and _key_equal(item.get("S"), sequence):
                    bs_served = True
                    break

    bs_samples_insufficient = False
    bs_outside_envelope = False
    bs_collinear_span = False
    bs_axis_frozen = False
    exact_lookup_only = False
    blocker: str | None = None
    if not groups:
        verdict = "empty"
        blocker = "empty"
    elif selected is None:
        verdict = "fallback_group"
        blocker = "group_key"
    elif not signature_served:
        # Group key matches but this parallel/MTP/scatter/pad signature was
        # never captured: the expander would silently reuse the nearest
        # variant.
        verdict = "fallback_group"
        blocker = "signature"
    elif not exact_lookup_covered_scoped(selected, config_full):
        # The reachable variant for this config carries exact-lookup dims
        # (dims no formula explains): only captured configurations whose
        # lookup entries hit are verified, everything else must be refused
        # (P0-1 discipline).  Scoped = sibling variants the config would
        # never expand into do not block it (md1-anchor lesson).
        verdict = "fallback_group"
        exact_lookup_only = True
        blocker = "exact_lookup"
    elif served_carry_bs and not bs_served:
        # Signature-scoped (B, S) span discipline, gate v2: on-line targets
        # of an exact-formula collinear span ride the token-algebraic
        # formulas (pilot: 0 shape mismatches at (3,768) from a
        # (2,512)/(4,1024) fit); off-line targets are unidentifiable (18
        # shape mismatches at (8,512)) until a non-collinear anchor joins.
        mode_meta = selected.get("mode_meta") if isinstance(selected, Mapping) else None
        formulas_exact = not (isinstance(mode_meta, Mapping) and mode_meta.get("failed_formulas"))
        from .structure_buckets import parameter_config

        coordinates = parameter_config(config_full)
        coordinate_points = {(int(parameter_config({**config_full, 'B': b, 'S': s})['B']), s) for b, s in sig_bs_points}
        if config_full.get('structure_policy') == 'fixed_structure_v1' and not config_full.get('is_decode'):
            from .structure_buckets import parallel_geometry

            coordinates = {**config_full, 'B': parallel_geometry(config_full)[1]}
            coordinate_points = {(parallel_geometry({**config_full, 'B': b})[1], s) for b, s in sig_bs_points}
        span_blocker = _bs_span_blocker(coordinate_points, int(coordinates['B']), sequence, formulas_exact)
        if span_blocker == "bs_samples":
            verdict = "fallback_group"
            bs_samples_insufficient = True
            blocker = "bs_samples"
        elif span_blocker == "bs_offline":
            verdict = "fallback_group"
            bs_collinear_span = True
            blocker = "bs_offline"
        elif span_blocker == "bs_axis_frozen":
            verdict = "fallback_group"
            bs_axis_frozen = True
            blocker = "bs_axis_frozen"
        elif span_blocker == "bs_envelope":
            verdict = "fallback_group"
            bs_outside_envelope = True
            blocker = "bs_envelope"
        else:
            # Signature served, span verified for this target: the
            # expansion rides the group's exact-integer formulas.
            # Disclosed as its own tier so sweeps can weigh interpolated
            # answers explicitly.
            verdict = "interpolated"
    else:
        verdict = "exact_group"

    variant_mode = "no_variants"
    if selected is not None:
        variants = _parallel_variants_of(selected)
        if variants:
            variant_keys: list[tuple[Any, Any, Any]] = []
            for variant in variants:
                for raw_key in variant.get("var_keys", []):
                    if isinstance(raw_key, (list, tuple)) and len(raw_key) == 3:
                        variant_keys.append(
                            (
                                tuple(int(value) for value in raw_key[0]),
                                bool(raw_key[1]),
                                bool(raw_key[2]),
                            )
                        )
            target_triple = (
                tuple(target_member["par_sig"]),
                bool(target_scatter),
                target_member["pad_cat"],
            )
            if target_triple in variant_keys:
                variant_mode = "exact_variant"
            elif verdict == "exact_group":
                variant_mode = "base_entries"
            else:
                variant_mode = "unmatched_variant"

    result: dict[str, Any] = {
        "config": {k: config_full[k] for k in sorted(config_full)},
        "target_group_key": str(target_key),
        "signature_served": signature_served,
        "bs_served": bs_served,
        "verdict": verdict,
        "blocker": blocker,
        "nearest_distance": fallback_distance,
        "variant_mode": variant_mode,
        "exact_lookup_only": exact_lookup_only,
        "mtp_bs_uncovered": False,
        "bs_samples_insufficient": bs_samples_insufficient,
        "bs_collinear_span": bs_collinear_span,
        "bs_axis_frozen": bs_axis_frozen,
        "bs_outside_envelope": bs_outside_envelope,
        # Verified-extrapolation discipline: interpolated serves ride
        # over-determined exact formulas beyond the captured (B, S) span;
        # consumers can demand anchor confirmation for flagged configs.
        "extrapolated": verdict == "interpolated",
    }
    if verdict == "interpolated":
        # Quantified extrapolation confidence (stat_fit): anchor count and
        # span leverage of the captured points behind the exact formulas
        # this serve rides on.  Diagnostics only - the verdict, blocker and
        # extrapolated flag above remain the only gate-relevant fields.
        from template_engine.stat_fit import extrapolation_confidence  # noqa: PLC0415

        result["extrapolation_confidence"] = extrapolation_confidence(
            [{"B": point[0], "S": point[1]} for point in sorted(sig_bs_points)],
            config_full,
        )
    if selected is not None:
        meta_coverage = template.get("template_meta", {}).get("coverage", {})
        mode_meta = selected.get("mode_meta") or {}
        result["failed_formulas"] = mode_meta.get("failed_formulas", meta_coverage.get("failed_formulas", 0))
        result["mode_tag"] = mode_meta.get("tag")
        result['structure_bucket_id'] = (mode_meta.get('structure_bucket') or {}).get('id')
        result["trained_mtp"] = selected.get("trained_mtp")

    # Validate the same full-template routing and strict expansion used by
    # serving.  prevalidated=True: THIS gate just adjudicated the config -
    # the expander must not re-run the admission rules (single-judgement
    # discipline, review issue #6); its light misuse assertion still runs.
    # Optionally hand those exact ops to the caller for one replay.
    if verdict not in ("exact_group", "interpolated"):
        result.update(op_count=0, expandable=False)
        return result
    try:
        generated = generate_ops_from_template(template, config_full, strict=True, prevalidated=True)
    except TemplateExpandError as exc:
        result["op_count"] = 0
        result["expandable"] = False
        result["blocker"] = "expansion_failed"
        result["verdict"] = "fallback_group"
        result["reason"] = f"validating expansion refused: {exc}"
        return result
    result["op_count"] = len(generated)
    structure_bucket = (selected.get('mode_meta') or {}).get('structure_bucket') if selected else None
    if structure_bucket:
        from .body_select import structure_fingerprint
        from .structure_buckets import bucket_id

        emitted_id = bucket_id(selected['discrete_cell'], structure_fingerprint(generated, parameter_lists=True))
        if emitted_id != structure_bucket['id']:
            result.update(
                verdict='fallback_group',
                blocker='structure_bucket_changed',
                op_count=0,
                expandable=False,
                reason='Expansion changed the operator/reference structure of the selected fixed bucket.',
            )
            return result
    result["expandable"] = bool(generated)
    if return_ops and generated:
        result["ops"] = generated
    if not generated and verdict in ("exact_group", "interpolated"):
        # Lie guard: a served verdict whose body expands to NOTHING is the
        # census empty-op defect (verdict exact_group with op_count=0 - an
        # unreachable/degenerate structure body silently serving 0 ops).
        # Downgrade to the honest refusal; the expander would emit nothing,
        # so "served" would be a lie no matter the verdict.
        result["verdict"] = "fallback_group"
        result["blocker"] = "empty_expansion"
        result["reason"] = (
            "served body expanded to an empty op list; the structure body "
            "cannot materialise this target - capture the config or rebuild"
        )
    return result


def serve_verdict(coverage: Mapping[str, Any]) -> dict[str, Any]:
    """Project a :func:`describe_coverage` result onto the admission decision.

    Review issue #6 (diagnostics decoupled from judgement): callers that only
    need "can this be served, and if not why" read THIS structure - three
    fields, no display booleans.  The full coverage mapping stays available
    for diagnostics; this projection is the contract for the single
    judgement entry point.

    Returns ``{"servable": bool, "verdict": str, "blocker": str | None}``.
    """

    verdict = str(coverage.get("verdict"))
    failed = int(coverage.get("failed_formulas", 0) or 0)
    servable = verdict == "exact_group" or (verdict == "interpolated" and failed == 0)
    return {
        "servable": servable,
        "verdict": verdict,
        "blocker": None if servable else (coverage.get("blocker") or verdict),
    }


def parse_sweep(spec: str) -> list[dict[str, Any]]:
    """Parse ``B=1:4,S=512:4096:512``-style sweeps into config dicts.

    ``name=start:stop[:step]`` enumerates inclusive integer ranges; a bare
    ``name=value`` fixes one value.
    """

    axes: list[tuple[str, list[int]]] = []
    for chunk in spec.split(","):
        name, _, raw = chunk.partition("=")
        name = name.strip()
        raw = raw.strip()
        if ":" in raw:
            parts = [int(part) for part in raw.split(":")]
            start, stop = parts[0], parts[1]
            step = parts[2] if len(parts) > 2 else 1
            if step <= 0:
                raise ValueError(f"sweep step must be positive: {chunk!r}")
            axes.append((name, list(range(start, stop + 1, step))))
        else:
            axes.append((name, [int(raw)]))
    combos: list[dict[str, Any]] = [{}]
    for name, values in axes:
        combos = [{**combo, name: value} for combo in combos for value in values]
    return combos


def _parse_target_spec(spec: str) -> dict[str, Any]:
    """Parse a "tp=1,ep=1,B=2,S=512" style target spec into a config dict."""

    typed: dict[str, Any] = {}
    mapping = {
        "B": "B",
        "S": "S",
        "tp": "TP",
        "TP": "TP",
        "ep": "EP",
        "EP": "EP",
        "md": "MOE_DP",
        "moe_dp": "MOE_DP",
        "MOE_DP": "MOE_DP",
        "dcp": "DCP",
        "DCP": "DCP",
        "mtp": "num_mtp_tokens",
        "num_mtp_tokens": "num_mtp_tokens",
        "quant_linear": "quantize_linear_action",
        "quant_attention": "quantize_attention_action",
    }
    for chunk in spec.split(","):
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if key in {"is_decode", "do_compile", "enable_dispatch_ffn_combine", "dfc"}:
            normalized = value.lower()
            if normalized not in {"1", "true", "on", "yes", "0", "false", "off", "no"}:
                raise ValueError(f"invalid boolean: {chunk!r}")
            typed["enable_dispatch_ffn_combine" if key == "dfc" else key] = normalized in {"1", "true", "on", "yes"}
            continue
        target_key = mapping.get(key)
        if target_key is None:
            typed[key] = value
        elif target_key.startswith("quantize_"):
            typed[target_key] = value.upper()
        else:
            typed[target_key] = int(value)
    typed.setdefault("quantize_linear_action", "DISABLED")
    typed.setdefault("quantize_attention_action", "DISABLED")
    return typed


def main(argv=None):
    from cli.inference.template_graph_diagnostics import coverage_main

    return coverage_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
