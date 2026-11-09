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

"""template_graph: G(params) -> computation graph (customer MVP v0.9).

Pure-function semantics per https://gitcode.com/Ascend/msmodeling/issues/591
(discrete-structure pivot): the structure dimension is an enumerated router over
captured (TP,EP,...) grid cells, the (B,S) dimension is continuous via
per-body symbolic formulas.  No cache tier, no capture fallback --
uncovered configs are refused with anchors_required (never guessed).

Pipeline (build time, offline):
    corpus -> structure enumeration -> per-body (B,S) formula fitting
           -> template_graph.json.xz (bodies + router + discipline)

Pipeline (run time, pure software):
    config -> router (predicate) -> body -> formula evaluation -> ops
"""

from __future__ import annotations

from .graph_codec import read_artifact_bytes

import sys
import time
from pathlib import Path
from typing import Any, Mapping

TP_GRID = (1, 2, 4, 8)
EP_GRID = (1, 2, 4, 8)


class TemplateGraphRefusal(Exception):
    """Raised when G(params) leaves the verified domain (never guesses)."""

    def __init__(self, config: Mapping[str, Any], blocker: str, anchors: list[dict[str, Any]] | None = None):
        self.config = dict(config)
        self.blocker = blocker
        self.anchors = anchors or []
        hint = ""
        if self.config.get("do_compile") is None and blocker in {
            "group_key",
            "signature",
            "compile provenance mismatch",
        }:
            hint = "; compile status unknown; use capture_config(key, cache_dir) to load manifest provenance"
        super().__init__(f"uncovered config {self.config}: {blocker}{hint}")


def _cell(tp: int, ep: int) -> str:
    return f"tp{tp}_ep{ep}"


def build_template_graph(
    out_path: str | Path | None = "template_graph.json.xz",
    cache_dir: str | Path = "compile_ops_cache_v4_serving",
    num_hidden_layers: int | None = None,
    keys: list[str] | None = None,
    do_compile: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the customer template graph from the capture corpus.

    ``keys`` restricts the corpus (P2 hold-out experiments build from the
    enumeration subset only); default is every cached capture.
    """
    from .cache_io import capture_config, load_capture
    from .body_select import discrete_cell
    from .template_builder import (
        _Capture,
        build_fixed_template,
        discover_cache_keys,
    )

    if keys is None:
        keys = discover_cache_keys(cache_dir)

    captures = []
    for key in keys:
        config = capture_config(key, cache_dir)
        if do_compile is not None and config.get("do_compile") is not do_compile:
            continue
        loaded = load_capture(key, cache_dir)
        if loaded is None:
            continue
        ops, exec_time = loaded
        captures.append(_Capture(key=key, config=config, ops=ops, exec_time=exec_time))
    modes = {c.config.get("do_compile") for c in captures}
    if len(modes) > 1:
        raise ValueError("build COMPILE on/off independently; pass do_compile or --compile-mode")
    if not captures:
        raise ValueError("no captures with matching compile provenance; legacy keys are not classified automatically")
    identities = {c.config.get('model_id') for c in captures if c.config.get('model_id')}
    if len(identities) > 1:
        raise ValueError('Build each model independently; capture corpus mixes model identities')
    artifact_mode = next(iter(modes))
    keys = [c.key for c in captures]
    template, report = build_fixed_template(captures)

    structures: list[dict[str, Any]] = []
    covered_cells: set[tuple[int, int, bool]] = set()
    for group_key, group in template["groups"].items():
        cells: set[tuple[int, int, bool]] = set()
        bs: list[tuple[int, int]] = []
        flags: dict[str, Any] = {}
        for served in group.get("served_keys", []):
            # served entries are dicts: par_sig=(TP,EP,MOE_DP,DCP) + B/S
            tp, ep, moe_dp, dcp = served["par_sig"]
            is_decode = bool(served.get("is_decode", False))
            cells.add((tp, ep, is_decode))
            covered_cells.add((tp, ep, is_decode))
            bs.append((served["B"], served["S"]))
            # Flat group keys ARE the discrete cell (+ an optional mode
            # suffix): the quant actions live at positions 7/8, not 0/1
            # (those are TP/EP - review issue #28).  Bare 4-axis validation
            # keys carry no quant axes and report None.
            quant_linear = group_key[7] if len(group_key) > 8 else None
            quant_attention = group_key[8] if len(group_key) > 8 else None
            flags = {
                "MOE_DP": moe_dp,
                "DCP": dcp,
                "num_mtp_tokens": served["num_mtp_tokens"],
                "enable_dispatch_ffn_combine": served["enable_dispatch_ffn_combine"],
                "quantize_linear_action": quant_linear,
                "quantize_attention_action": quant_attention,
            }
        structures.append(
            {
                "id": repr(group_key),
                "group_key": group_key,
                "structure_bucket_id": (group.get('mode_meta', {}).get('structure_bucket') or {}).get('id'),
                "cells": sorted(list(c) for c in cells),
                "bs_points": sorted(bs),
                "flags": flags,
            }
        )

    anchors_required = []
    for tp in TP_GRID:
        for ep in EP_GRID:
            for is_decode in (False, True):
                if (tp, ep, is_decode) in covered_cells:
                    continue
                regime = "unrolled" if (ep >= 2 and tp >= 2 * ep) else "fused"
                anchors_required.append(
                    {
                        "cell": _cell(tp, ep),
                        "is_decode": is_decode,
                        "regime": regime,
                        "rule": "EP>=2 and TP>=2*EP -> unrolled body (empirical grid)",
                        "reason": "capture independent B/S anchors, fit, then validate against held-out graphs",
                    }
                )

    # Stamp provenance: the serving-semantics capture CLI (which charges
    # query = mtp + 1 on decode) is also the only writer of capture
    # sidecars, so a decode corpus WITH sidecars is a decode corpus in the
    # optimizer-aligned convention.  A legacy-convention corpus (v3:
    # decode captured with query = S) has no sidecars and stays unstamped,
    # keeping the runtime gates conservative there.
    model_meta = _collect_model_meta(cache_dir, keys)
    # Depth is a property of the model, not of this engine: take it from the
    # capture sidecars instead of assuming one architecture's layer count.
    if num_hidden_layers is None:
        depths = set()
        for entry in (model_meta or {}).values():
            if not isinstance(entry, Mapping):
                continue
            try:
                depth = int(entry.get("num_hidden_layers") or 0)
            except (TypeError, ValueError):
                continue
            if depth > 0:
                depths.add(depth)
        if len(depths) == 1:
            num_hidden_layers = depths.pop()
        elif len(depths) > 1:
            raise ValueError(f"corpus mixes layer counts {sorted(depths)}: pass num_hidden_layers explicitly")
        else:
            num_hidden_layers = 0
        if num_hidden_layers <= 0:
            raise ValueError(
                "num_hidden_layers is unknown for this corpus: capture with sidecars "
                "(--meta) or pass num_hidden_layers explicitly"
            )
    decode_stamp = (
        "mtp_plus_1"
        if any(c.config.get("is_decode") for c in captures)
        and all(c.config.get("decode_query_semantics") == "mtp_plus_1" for c in captures if c.config.get("is_decode"))
        else None
    )

    # Model identity comes from the corpus itself, never from a hard-coded
    # name: a template built from one model must not silently serve another.
    # Mixed corpora keep every id and are refused at serving time.
    model_ids = sorted({str(c.config.get("model_id") or "unknown") for c in captures})
    template_graph = {
        "meta": {
            "model": model_ids[0] if len(model_ids) == 1 else "mixed",
            "model_ids": model_ids,
            "do_compile": artifact_mode,
            "num_hidden_layers": num_hidden_layers,
            "format": "template_graph/fixed-structure-v1",
            "semantics": "G(params) -> ops | refusal; fixed operator/reference buckets with parameter formulas",
            "corpus": list(keys),
            "build_notes": report.notes[:50],
            "model_meta": model_meta,
            # Runtime gates serve decode through coverage only when this
            # stamp is present; unstamped templates keep the conservative
            # decode refusal (legacy decode anchors had query = S
            # semantics and would mislead the optimizer's decode probes).
            "decode_query_semantics": decode_stamp,
        },
        "router": {
            # The older TP/EP projection is retained for readers; it is not
            # coverage of the full quantization/MTP/DCP/DFC/compile grid.
            "discrete_cells_captured": sorted({discrete_cell(c.config) for c in captures}, key=str),
            "structures": structures,
            "grid_cells_covered": sorted([list(c) for c in covered_cells]),
            "anchors_required": anchors_required,
            "regime_predicate": "unrolled iff EP>=2 and TP>=2*EP (empirical, see plan revisions)",
        },
        "discipline": {
            "refusal": "uncovered configs raise/return covered=false with anchors_required; never guessed",
            "bs_envelope": "per-structure (B,S) points; formulas extrapolate only inside fitted domain",
        },
        "engine": template,
    }
    from .zero_meta import _save_template_lazy

    if out_path is not None:
        _save_template_lazy(template_graph, out_path)
    stats = {
        "corpus": len(captures),
        "structures": len(structures),
        "cells_covered": len(covered_cells),
        "anchors_required": len(anchors_required),
        "failed_formulas": report.failed_formulas,
        "notes": len(report.notes),
    }
    return template_graph, stats


def load_template_graph(path: str | Path = "template_graph.json.xz") -> dict[str, Any]:
    from .zero_meta import _load_template_any

    return _load_template_any(path)


def _collect_model_meta(cache_dir: str | Path | None, keys: list[str] | None = None) -> dict[str, Any]:
    """Aggregate canonical ``<key>.json.xz.meta`` sidecars.

    First-seen wins: every capture of one weight key carries identical
    weight/kv metadata (the axes that change them are pinned by the
    key).  Empty when no sidecars exist - the corpus predates them and
    candidates keep taking the legacy build+wrap path.
    """

    from .graph_codec import decode_document

    if not cache_dir:
        return {}
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return {}
    model_meta: dict[str, Any] = {}
    # Sidecars live next to their capture, including inside per-model
    # sub-directories, so both levels are scanned.
    if keys is None:
        sidecars = sorted(cache_path.glob("*.json.xz.meta")) + sorted(cache_path.glob("*/*.json.xz.meta"))
    else:
        sidecars = []
        for key in keys:
            direct = cache_path / f"{key}.json.xz.meta"
            sidecars.append(direct)
            sidecars.extend(sorted(cache_path.glob(f"*/{key}.json.xz.meta")))
    for sidecar in sidecars:
        try:
            meta = decode_document(read_artifact_bytes(sidecar), "model_meta/1")
        except Exception:  # noqa: BLE001 - a broken sidecar never blocks the build
            continue
        weight_key = meta.get("weight_key") if isinstance(meta, dict) else None
        if weight_key and weight_key not in model_meta:
            model_meta[weight_key] = meta
    return model_meta


def generate(
    template_graph: Mapping[str, Any],
    config: Mapping[str, Any],
    num_hidden_layers: int | None = None,
    return_ops: bool = False,
) -> dict[str, Any]:
    """G(params): config -> ops and graph-generation timing. Raises TemplateGraphRefusal
    outside the verified domain (planning-safe; never captures).

    Ops are omitted by default. Independent graph comparisons must pass
    ``return_ops=True`` and compare ``result["ops"]`` against held-out captures.
    """
    started = time.perf_counter()
    if num_hidden_layers is None:
        num_hidden_layers = int(template_graph.get("meta", {}).get("num_hidden_layers") or 0)
        if num_hidden_layers <= 0:
            raise TemplateGraphRefusal(config, "template declares no num_hidden_layers")
    declared_depth = template_graph.get("meta", {}).get("num_hidden_layers")
    if (
        type(declared_depth) is not int
        or declared_depth <= 0
        or type(num_hidden_layers) is not int
        or num_hidden_layers <= 0
        or num_hidden_layers != declared_depth
    ):
        raise TemplateGraphRefusal(config, "layer count differs from the fixed-depth artifact")
    if "do_compile" in template_graph.get("meta", {}) and template_graph["meta"]["do_compile"] is not config.get(
        "do_compile"
    ):
        raise TemplateGraphRefusal(config, "compile provenance mismatch")
    # Model identity gate: a template carries the model it was induced from,
    # so serving a different model's config is refused instead of silently
    # returning another architecture's graph.
    served_models = template_graph.get("meta", {}).get("model_ids")
    requested_model = config.get("model_id")
    if served_models and requested_model:
        if str(requested_model) not in {str(item) for item in served_models}:
            raise TemplateGraphRefusal(config, "model identity mismatch")
    from .cache_io import build_key_and_config
    from .coverage import describe_coverage

    key, key_config = build_key_and_config(config)
    identities = [m for m in (served_models or []) if m != 'unknown']
    identity = requested_model or (identities[0] if len(identities) == 1 else None)
    if identity:
        key_config['model_id'] = identity
    if 'device' in config:
        key_config['device'] = config['device']
    if 'model_parameters' in config:
        key_config['model_parameters'] = config['model_parameters']
    engine = template_graph["engine"]
    from .model_parameters import bind_parameters

    key_config = bind_parameters(engine, key_config)
    coverage = describe_coverage(engine, key_config, return_ops=True)
    verdict = str(coverage.get("verdict"))
    failed = int(coverage.get("failed_formulas", 0) or 0)
    # Discipline: an exact anchor is served when its reachable variant's
    # lookups all hit (checked by coverage's scoped gate) - the group-global
    # failed-formula count belongs to sibling variants it never expands
    # into.  Interpolation stays conservative: group formulas must be clean.
    serve = verdict == "exact_group" or (verdict == "interpolated" and failed == 0)
    if serve:
        ops = coverage["ops"]
        out = {
            "covered": True,
            "source": "template_graph",
            "coverage": verdict,
            # Verified-extrapolation discipline: an interpolated (formula-
            # extrapolated, not a captured anchor) serve is flagged so
            # callers/consumers can distinguish anchor replays from
            # extrapolations riding over-determined exact formulas.
            "extrapolated": verdict == "interpolated",
            "mode_tag": coverage.get("mode_tag"),
            "structure_bucket_id": coverage.get('structure_bucket_id'),
            "structure_policy": engine.get('template_meta', {}).get('structure_policy'),
            "key": key,
            "op_count": len(ops),
            "wall_time_s": round(time.perf_counter() - started, 3),
        }
        if return_ops:
            out["ops"] = ops
        return out
    anchors = _anchors_for(template_graph, key_config)
    raise TemplateGraphRefusal(
        key_config, coverage.get("blocker") or (f"failed_formulas={failed}" if failed else verdict), anchors
    )


def _anchors_for(template_graph: Mapping[str, Any], key_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tp, ep = int(key_config["TP"]), int(key_config["EP"])
    is_decode = bool(key_config.get("is_decode", False))
    regime = "unrolled" if (ep >= 2 and tp >= 2 * ep) else "fused"
    return [
        {
            "cell": _cell(tp, ep),
            "is_decode": is_decode,
            "regime": regime,
            "reason": "capture independent B/S anchors and validate generated graphs",
            "suggested_command": f"capture B={key_config['B']} S={key_config['S']} tp={tp} ep={ep}",
        }
    ]


def main(argv=None):
    from cli.inference.template_graph import legacy_graph_main

    return legacy_graph_main(argv)


if __name__ == '__main__':
    sys.exit(main())
