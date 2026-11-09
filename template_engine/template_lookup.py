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

"""Shared graph supply: exact capture cache, covered template, native capture.

Only legitimate coverage misses reach native capture. Missing or malformed
explicit template files and internal failures propagate. ``ensure_ops`` is
the typed compatibility adapter; ``lookup_ops`` owns the routing decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .cache_io import load_capture


def ensure_ops(
    B: int,
    S: int,
    *,
    is_decode: bool = False,
    tp: int = 1,
    ep: int = 1,
    moe_dp: int = 1,
    dcp: int = 1,
    do_compile: bool = False,
    num_mtp_tokens: int = 0,
    quantize_linear_action: str = "DISABLED",
    quantize_attention_action: str = "DISABLED",
    enable_dispatch_ffn_combine: bool = False,
    num_hidden_layers_override: int | None = None,
    cache_dir: str | Path | None = None,
    model_id: str | None = None,
    template_path: str | Path | None = None,
    template: Mapping[str, Any] | None = None,
    allow_capture: bool = True,
) -> dict[str, Any] | None:
    """Return ``{"key", "ops", "exec_time_s", "source"}`` for one config.

    ``source`` is ``"cache"``, ``"template"``, or ``"captured"``.  Template
    hits carry ``exec_time_s=None`` because a template stores the anchor
    capture's timing, not the target's — callers that need timing must run
    the capture.  With ``allow_capture=False`` the function returns ``None``
    instead of capturing (used to probe template coverage only).
    ``template`` accepts an in-memory template (build output) and takes
    precedence over ``template_path`` — this is the single coverage-judged
    supply path; there is no second config-level entry that re-implements
    the gate.
    """

    # The request config is canonicalised directly from the typed arguments —
    # no build_cache_key -> parse_cache_key string round-trip (issue #9: one
    # normalization entry, cache_io.normalize_config for dict callers).
    config = {
        "TP": tp,
        "EP": ep,
        "MOE_DP": moe_dp,
        "DCP": dcp,
        "B": B,
        "S": S,
        "num_mtp_tokens": num_mtp_tokens,
        "enable_dispatch_ffn_combine": enable_dispatch_ffn_combine,
        "is_decode": is_decode,
        "do_compile": do_compile,
        "quantize_linear_action": quantize_linear_action,
        "quantize_attention_action": quantize_attention_action,
    }
    if model_id is not None:
        config['model_id'] = model_id
    result = lookup_ops(
        config,
        template=template,
        template_path=template_path,
        cache_dir=cache_dir,
        allow_capture=allow_capture,
        layers=num_hidden_layers_override or 0,
    )
    return result if result['source'] != 'uncovered' else None


def lookup_ops(
    config,
    *,
    template=None,
    template_path=None,
    cache_dir=None,
    allow_capture=True,
    layers=0,
    capture_fn=None,
    persist=False,
):
    """One cache/template/native routing implementation for all callers."""
    from .cache_io import build_key_and_config, cache_variant_dir, save_capture
    from .zero_meta import load_template_engine

    key, canonical = build_key_and_config(config)
    config = {**config, **canonical}
    model_id = config.get('model_id')
    cache_dir = cache_variant_dir(cache_dir, layers)
    cached = load_capture(key, cache_dir, model_id=model_id) if cache_dir is not None else None
    if cached is not None:
        ops, duration = cached
        return dict(key=key, ops=ops, exec_time_s=duration, source='cache')
    if template is None and template_path is not None:
        template = load_template_engine(template_path)
    blocker = 'not_captured'
    if template is not None and not layers:
        from .coverage import describe_coverage, serve_verdict

        coverage = describe_coverage(template, config, return_ops=True)
        admission = serve_verdict(coverage)
        if admission['servable']:
            return dict(key=key, ops=coverage.pop('ops'), exec_time_s=None, source='template', coverage=coverage)
        blocker = admission['blocker']
    if not allow_capture:
        return dict(key=key, ops=None, exec_time_s=None, source='uncovered', blocker=blocker)
    if capture_fn is None:
        from .capture_ops import capture as capture_fn

    ops, duration = capture_fn(
        config['B'],
        config['S'],
        is_decode=config['is_decode'],
        tp=config['TP'],
        ep=config['EP'],
        moe_dp=config['MOE_DP'],
        dcp=config['DCP'],
        do_compile=bool(config.get('do_compile', False)),
        num_mtp_tokens=config['num_mtp_tokens'],
        quantize_linear_action=config['quantize_linear_action'],
        quantize_attention_action=config['quantize_attention_action'],
        enable_dispatch_ffn_combine=config['enable_dispatch_ffn_combine'],
        num_hidden_layers_override=layers,
        **({'model_id': model_id} if model_id else {}),
    )
    if persist and cache_dir is not None:
        save_capture(key, ops, duration, cache_dir, model_id=model_id)
    return dict(key=key, ops=ops, exec_time_s=duration, source='captured')
