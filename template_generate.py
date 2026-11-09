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

"""Expand a serialized model graph template into ``OpInvokeInfo`` objects.

The training-side template builder is intentionally outside this module.  This
module only implements the runtime half of the template workflow:

``load_template(path)`` -> restore serialized ``torch.ops`` handles
``generate_ops_from_template(template, config)`` -> expand formulas and loops

Template pickle files are executable Python serialization artifacts.  Only load
files from a trusted source.
"""

from __future__ import annotations

import functools
import logging
import math
import numbers
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

# Admission-rule layer (review issue #6 - judgement convergence): the rule
# implementations (signature categories, variant selection, exact-lookup
# walks, (B, S) span discipline, key vocabulary) moved to the judgement-side
# module; this expander imports them back for its own internals and keeps
# them importable from here for existing callers.
from template_engine.serve_rules import (  # noqa: E402 - after tensor_cast imports
    _group_distance,
    _key_equal,
    _lookup_config_view,
    _normalise_key,
    _pad_cat_from_config,
    _select_parallel_variant,
    _signature_categories,
)


class TemplateExpandError(RuntimeError):
    """Raised by strict expansion when the target config is not covered."""


MODEL_CONSTANTS = {
    "H": 7168,
    "I": 18432,
    "MI": 3072,
    "V": 129280,
    "E": 384,
    "E_per_tok": 6,
    "CR": 128,
}

# Retained shape-formula vocabulary (issue #11): the delivered artifacts use
# exactly CONST / LINEAR / MULTI / DOMAIN_CEIL_DIV / EXACT_LOOKUP (+ the
# stride-only CONTIGUOUS marker and the internal FAILED sentinel).  STEP,
# STEP_BILINEAR, MULTI_CANDIDATE and DOMAIN_DIV had zero artifact uses and no
# live fitter after the census, so their evaluation branches were removed -
# a template carrying one of them now refuses honestly at expansion time
# (the artifact-level gate test keeps them out of new artifacts).
_SHAPE_FORMULA_TYPES = {
    "CONST",
    "LINEAR",
    "MULTI",
    "EXACT_LOOKUP",
    "DOMAIN_CEIL_DIV",
    "FAILED",
}


def _integer_if_close(value: Any) -> Any:
    """Convert numeric values close to an integer without changing other values."""

    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        return value
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return value
    if math.isfinite(numeric_value) and abs(numeric_value - round(numeric_value)) < 1e-9:
        return int(round(numeric_value))
    return value


def _config_int(config: Mapping[str, Any], key: str, default: int = 1) -> int:
    value = config.get(key, default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _floor_div(numerator: int | float, denominator: int | float) -> int | float:
    if denominator == 0:
        return 0
    return numerator // denominator


def _ceil_div(numerator: int | float, denominator: int | float) -> int | float:
    if denominator <= 0:
        return 0
    return -(-numerator // denominator)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(_ceil_div(value, multiple) * multiple)


def _build_features(config: Mapping[str, Any]) -> Mapping[Any, Any]:
    """Feature table for ``config``'s shape formulas (memoized, read-only).

    The table is a pure function of the eight config fields read below and
    the artifact-bound model parameters (legacy artifacts use ``MODEL_CONSTANTS``).  Expansion evaluates tens of
    thousands of formulas per point against the same config, so the table
    is built once per distinct field tuple and returned as a read-only
    mapping - an accidental mutation raises instead of silently poisoning
    a cached table shared across evaluations.
    """

    from template_engine.structure_buckets import parameter_config

    config = parameter_config(config)
    base = _build_features_cached(
        _config_int(config, "B"),
        _config_int(config, "S"),
        _config_int(config, "TP"),
        _config_int(config, "MOE_DP"),
        _config_int(config, "EP"),
        _config_int(config, "DCP"),
        _config_int(config, "num_mtp_tokens", 0),
        bool(config.get("is_decode", False)),
        tuple(tuple(pair) for pair in config.get("model_parameters", tuple(MODEL_CONSTANTS.items()))),
        "model_parameters" in config,
        config.get('structure_policy') == 'fixed_structure_v1',
    )
    if "group_index" not in config and "list_index" not in config:
        return base
    features = dict(base)
    b, s = _config_int(config, "B"), _config_int(config, "S")
    for name in ("group_index", "list_index"):
        index = int(config.get(name, 0))
        features[name] = index
        features[f"B*{name}"] = b * index
        features[f"S*{name}"] = s * index
        features[f"{name}//B"] = index // max(b, 1)
        features[f"{name}%B"] = index % max(b, 1)
        features[f"{name}%B==0"] = int(index % max(b, 1) == 0)
        for offset in (-1, 0, 1):
            features[f"{name}<B{offset:+d}"] = int(index < b + offset)
        features[f"{name}==0"] = int(index == 0)
    width = max(1, int(config.get("list_width", 1)))
    list_index = int(config.get("list_index", 0))
    features["list_index//width"] = list_index // width
    features["list_index%width"] = list_index % width
    features["list_index//width==0"] = int(list_index // width == 0)
    features["list_index//width<B-1"] = int(list_index // width < b - 1)
    return features


@functools.lru_cache(maxsize=64)
def _build_features_cached(
    b: int,
    s: int,
    tp: int,
    moe_dp: int,
    ep: int,
    dcp: int,
    mtp: int,
    decode: bool,
    model_parameters: tuple,
    declared_parameters: bool,
    fixed_structure: bool = False,
) -> Mapping[Any, Any]:
    """Build the feature dictionary used by shape formulas.

    Feature names deliberately retain the spelling used by the training-side
    template generator.  In particular, names contain ``/``, ``*`` and ``+``
    rather than being parsed as Python expressions.
    """

    ws = max(tp, ep, dcp) * moe_dp

    constants = dict(model_parameters)
    h, i, mi, vocab, experts, experts_per_token, compress_ratio = (
        constants.get(name, 1) for name in ('H', 'I', 'MI', 'V', 'E', 'E_per_tok', 'CR')
    )

    features: dict[Any, Any] = {
        "1": 1,
        "B": b,
        "S": s,
        "TP": tp,
        "MOE_DP": moe_dp,
        "EP": ep,
        "DCP": dcp,
        "WS": ws,
        "MTP": mtp,
        "MTP+1": mtp + 1,
        "B*S": b * s,
        "H": h,
        "I": i,
        "MI": mi,
        "V": vocab,
        "E": experts,
        "E_per_tok": experts_per_token,
        # Some early template snapshots used this shorter spelling.
        "Eper": experts_per_token,
        "CR": compress_ratio,
    }

    # Data dimensions divided by parallel dimensions.
    for denominator_name, denominator in (
        ("TP", tp),
        ("MOE_DP", moe_dp),
        ("EP", ep),
        ("WS", ws),
    ):
        features[f"B*S/{denominator_name}"] = _floor_div(b * s, denominator)
        features[f"B/{denominator_name}"] = _floor_div(b, denominator)
        features[f"S/{denominator_name}"] = _floor_div(s, denominator)

    features["S/CR"] = _floor_div(s, compress_ratio)
    # Compressor outputs use floor(packed_tokens / ratio), independently
    # of paged-KV allocation. Both ratios occur in the V4 semantic op.
    # Without these primitives, aligned anchors can accidentally select
    # a proportional KV-page ceil that adds rows at a larger B or S.
    features["B*S//4"] = _floor_div(b * s, 4)
    features["B*S//128"] = _floor_div(b * s, 128)
    # Prefill indexer allocation first floors compressed slots, then ceils
    # pages. B reserve tokens disappear for B<4 but can add a page at B>=4.
    # See input_generator._resolve_sparse_attention_indexer_num_blocks.
    prefill_slots = _floor_div(b * (s + 1 + mtp), 4)
    features["prefill_indexer_kv_blocks"] = max(1, _ceil_div(prefill_slots, 128))

    # Paged-KV whitebox features.
    # ``generate_inputs`` sizes the paged pools from
    #   max_context_length = seq_len + max(MTP, dflash) + 1
    #   total_kv_tokens    = max_context_length * batch_size
    # (tensor_cast/core/input_generator.py).  The template capture contract
    # (template_engine.capture_ops) builds prefill requests with
    # seq_len = context + query = 2*S; decode requests keep query_len=S but
    # run with context_length = S+256, so seq_len = 2*S+256.  Decode captures
    # live in their own expander groups (the group key carries the decode
    # marker), so the shared feature keys never mix the two semantics.
    # DeepSeek-V4 compresses the SFA indexer slots with compress_ratio=4
    # before the block_size=128 ceil (``_resolve_sparse_attention_indexer_num_blocks``).
    decode_context = 256 if decode else 0
    total_kv = b * (2 * s + decode_context + 1 + mtp)
    indexer_blocks = max(1, -(-_floor_div(total_kv, 4) // 128))
    features["B*(2*S+1+MTP)"] = total_kv
    features["total_kv//4"] = _floor_div(total_kv, 4)
    features["ceil(total_kv//4/128)"] = indexer_blocks
    features["ceil(total_kv//4/128)+B"] = indexer_blocks + b
    features["S//256+1"] = _floor_div(s, 256) + 1
    features["B*S/256"] = _floor_div(b * s, 256)
    # SFA indexer cache widths observed on card-1 captures: one family has
    # width = 128 * (S // 256) and another width = 128 + S // 64; their row
    # strides follow tokens * width.  The compound features below let the
    # exact-integer fits express those layouts without guesswork.
    s_per_256 = _floor_div(s, 256)
    s_per_64 = _floor_div(s, 64)
    features["S//256"] = s_per_256
    features["S//64"] = s_per_64
    features["S//4"] = _floor_div(s, 4)
    features["B*S*(S//256)"] = _floor_div(b * s, 1) * s_per_256
    features["B*S*(S//64)"] = b * s * s_per_64

    # Serving-cell SFA indexer page blocks (the (N,128,512) caches): the
    # serving capture contract gives each decode request a context of S plus
    # a 512-token decode reserve (per-request seq_len = S+512) and the varlen
    # tail adds B*(1+MTP); `_resolve_sparse_attention_indexer_num_blocks`
    # then compresses slots by ratio 4 and ceils to 128-token blocks.
    # Verified exactly on 9 real captures - B2 S{512,640,768,896,1024,777}
    # (DISABLED) and B1S512/B3S512/B4S1024 (W8A8) - including the
    # page-aligned S=768 point (640 slots = exactly 5 blocks) that the
    # former indexer_topk2_width//118 approximation mis-fit at S=777.
    indexer_kv_total = b * (s + 512) + b * (1 + mtp)
    features["indexer_kv_blocks"] = max(1, -(-_floor_div(indexer_kv_total, 4) // 128))

    # Model constants divided by parallel dimensions.
    for constant_name, constant in (
        ("H", h),
        ("I", i),
        ("MI", mi),
        ("E", experts),
    ):
        for denominator_name, denominator in (
            ("TP", tp),
            ("MOE_DP", moe_dp),
            ("WS", ws),
        ):
            features[f"{constant_name}/{denominator_name}"] = _floor_div(constant, denominator)
        if constant_name == "E":
            features["E/EP"] = _floor_div(constant, ep)

    # The training-side name is intentionally ``B*S/TP*H``.  Its historical
    # formula divides by TP twice, as recorded in the task specification.
    features["B*S/TP*H"] = _floor_div(b * s * h, tp * tp)
    features["(B*S/TP)*H"] = features["B*S/TP*H"]
    features["B*H/TP"] = _floor_div(b * h, tp)
    features["B*H/MOE_DP"] = _floor_div(b * h, moe_dp)
    features["B*H/WS"] = _floor_div(b * h, ws)

    # Batch/parallel products.
    features.update(
        {
            "B*TP": b * tp,
            "B*MOE_DP": b * moe_dp,
            "B*WS": b * ws,
            "S*TP": s * tp,
        }
    )

    # Alignment features.
    aligned_b_moe_dp = _ceil_to_multiple(b, moe_dp)
    aligned_b_ep = _ceil_to_multiple(b, ep)
    aligned_b_ws = _ceil_to_multiple(b, ws)
    aligned_bs_ws = _ceil_to_multiple(b * s, ws)
    features.update(
        {
            "ceil(B/MOE_DP)*MOE_DP": aligned_b_moe_dp,
            "ceil(B/WS)*WS": aligned_b_ws,
            "ceil(B*S/WS)*WS": aligned_bs_ws,
            "ceil(B/MOE_DP)*S": aligned_b_moe_dp * s,
            "ceil(B/WS)*S": aligned_b_ws * s,
            "ceil(B/TP)*S": _ceil_to_multiple(b, tp) * s,
            "ceil(B/MOE_DP)": _ceil_div(b, moe_dp),
            "ceil(B/WS)": _ceil_div(b, ws),
            "ceil(B/TP)": _ceil_div(b, tp),
            "ceil(B/EP)": _ceil_div(b, ep),
            "ceil(B/EP)*S": _ceil_div(b, ep) * s,
            "ceil(B/MOE_DP)*MOE_DP*S": aligned_b_moe_dp * s,
            "alBMOE_DP": aligned_b_moe_dp,
            "alBEP": aligned_b_ep,
            "alBMOE_DP*S*Eper": aligned_b_moe_dp * s * experts_per_token,
            "alBEP*S*Eper": aligned_b_ep * s * experts_per_token,
        }
    )

    # Decode dimensions use max rather than sum to model the largest local
    # work item.  Include the unscaled and S/H/I/MI-scaled variants.
    max_dimensions = {
        "MOE_DP": moe_dp,
        "TP": tp,
        "EP": ep,
        "WS": ws,
    }
    for dimension_name, dimension in max_dimensions.items():
        maximum = max(b, dimension)
        features[f"max(B,{dimension_name})"] = maximum
        features[f"max(B,{dimension_name})*S"] = maximum * s
        features[f"max(B,{dimension_name})*H"] = maximum * h
        features[f"max(B,{dimension_name})*I"] = maximum * i
        features[f"max(B,{dimension_name})*MI"] = maximum * mi

    # Expert token and all-to-all features.
    features.update(
        {
            "B*E_per_tok": b * experts_per_token,
            "B*E_per_tok*S": b * experts_per_token * s,
            "B*E_per_tok*S/EP": _floor_div(b * experts_per_token * s, ep),
            "B*E_per_tok*S/TP": _floor_div(b * experts_per_token * s, tp),
            "ceil(B*E_per_tok/EP)*S": _ceil_div(b * experts_per_token, ep) * s,
            "ceil(B*E_per_tok/TP)*S": _ceil_div(b * experts_per_token, tp) * s,
            "ceil(B*E_per_tok/WS)*S": _ceil_div(b * experts_per_token, ws) * s,
            "ceil(B*E_per_tok/EP)": _ceil_div(b * experts_per_token, ep),
            "ceil(B*E_per_tok/MOE_DP)*S": _ceil_div(b * experts_per_token, moe_dp) * s,
            "min(B,EP)": min(b, ep),
        }
    )
    min_b_ep = min(b, ep)
    features["B*E_per_tok*S/min(B,EP)/EP"] = (
        _floor_div(_floor_div(b * experts_per_token * s, min_b_ep), ep) if min_b_ep else 0
    )

    expert_padding_denominator_ep = ep * experts
    expert_padding_denominator_tp = tp * experts
    features.update(
        {
            "ceil(B*S*E_per_tok/(EP*E))*E": _ceil_div(b * s * experts_per_token, expert_padding_denominator_ep)
            * experts,
            "ceil(B*S*E_per_tok/(EP*E))*(E/EP)": _ceil_div(b * s * experts_per_token, expert_padding_denominator_ep)
            * _floor_div(experts, ep),
            "ceil(B*S*E_per_tok/(TP*E))*E": _ceil_div(b * s * experts_per_token, expert_padding_denominator_tp)
            * experts,
        }
    )

    # KV-cache scatter windows and the joint TP/EP embedding path.
    bs_over_ep = _floor_div(b * s, ep)
    scatter_remainder = bs_over_ep % 128 if ep else 0
    features.update(
        {
            "B*S/EP": bs_over_ep,
            "(B*S/EP)%128": scatter_remainder,
            "step((B*S/EP)%128==0)": 1 if ep == 0 or scatter_remainder == 0 else 0,
            "B*S*min(TP,EP)/EP": _floor_div(b * s * min(tp, ep), ep),
            "max(S, B*S*min(TP,EP)/EP)": max(s, _floor_div(b * s * min(tp, ep), ep)),
        }
    )

    # MoE expert token-count formulas.
    moe_base = max(_floor_div(b * s, ws), _floor_div(s, tp))
    moe_expert_tokens = _floor_div(moe_base * experts_per_token * ep, experts)
    alignment_denominator = b * ep
    alignment_factor = max(1, int(_floor_div(tp, alignment_denominator))) if alignment_denominator else 1
    features.update(
        {
            "max(B*S/WS, S/TP)": moe_base,
            "moe_expert_tokens": moe_expert_tokens,
            "moe_expert_tokens_aligned": moe_expert_tokens * alignment_factor,
            "ceil(B*TP/WS)": _ceil_div(b * tp, ws),
            "ceil(B*TP/WS)+1": _ceil_div(b * tp, ws) + 1,
        }
    )

    # Parallel-product features (RFC Phase D): expert-weight sharding dims
    # such as ``3072*MOE_DP//TP`` are exact linear combinations of these, so
    # MULTI fits can express them once captures decorrelate TP and MOE_DP.
    features["TP*MOE_DP"] = tp * moe_dp
    features["EP*MOE_DP"] = ep * moe_dp
    features["TP*EP"] = tp * ep
    features["E*MOE_DP/TP"] = _floor_div(experts * moe_dp, tp)
    features["E*MOE_DP/EP"] = _floor_div(experts * moe_dp, ep)
    features["E*EP/TP"] = _floor_div(experts * ep, tp)
    # Scatter/pad 0-1 features: signature-level semantics as fit inputs so a
    # family spanning a scatter/pad flip can condition dims on them instead
    # of relying on the group key.
    features["pad"] = 1 if _pad_cat_from_config(b, s, tp, ep, constants) else 0

    # V4 lightning-indexer top-k width, whitebox from the op semantics
    # (tensor_cast/ops/deepseek_v4.py): ``topk = min(topk_limit, seq_lens.max() // 4)``
    # with topk_limit = 1024 and the capture-contract request seq_len
    # ``2*S + decode_context`` (256 for decode, 0 for prefill).  This replaces
    # the former EXACT_LOOKUP dims on the decode indexer path: the apparent
    # "+64 offset vanishing at S=2048" was the topk_limit clamp, not a
    # semantic break, so decode (B, S) points extrapolate by formula.
    indexer_seq = 2 * s + decode_context
    features["indexer_seq"] = indexer_seq
    features["indexer_topk_width"] = min(1024, _floor_div(indexer_seq, 4))
    # The second sparse layer's indexer sees +512 raw context tokens
    # (+128 compressed slots) relative to the first; exact across the whole
    # decode anchor envelope, including the topk_limit-clamped point.
    features["indexer_topk2_width"] = features["indexer_topk_width"] + 128
    # Element counts / leading strides of the topk tensors are the product of
    # the packed token rows (B*S) and the topk width - a product no linear
    # combination can express, hence these whitebox product features.
    features["indexer_topk_numel"] = b * s * features["indexer_topk_width"]
    features["indexer_topk2_numel"] = b * s * features["indexer_topk2_width"]

    if not decode and declared_parameters:
        from template_engine.parameter_features import prefill_routing_features, prefill_cache_features

        features.update(prefill_routing_features(b, s, tp, ep, moe_dp, constants))
        features.update(prefill_cache_features(b, s, tp, ep, moe_dp, mtp, constants))
    if decode and declared_parameters and fixed_structure:
        from template_engine.parameter_features import prefill_cache_features, prefill_routing_features

        features.update(prefill_routing_features(b, mtp + 1, tp, ep, moe_dp, constants))
        cache = prefill_cache_features(b, s, tp, ep, moe_dp, mtp, constants, decode=True)
        features.update({'decode/' + name.removeprefix('prefill/'): value for name, value in cache.items()})

    # Missing architecture fields are not inferred from another model. They
    # must not introduce model-specific hypotheses into a new template.
    missing = set(MODEL_CONSTANTS) - set(constants)
    if missing:
        import re

        features = {
            name: value
            for name, value in features.items()
            if not any(re.search(r'(?<![A-Za-z_])' + field + r'(?![A-Za-z_])', str(name)) for field in missing)
        }
        # These derived aliases depend on expert counts rather than spelling E.
        if {'E', 'E_per_tok'} & missing:
            for name in list(features):
                if 'moe_expert' in str(name) or name == 'pad' or 'Eper' in str(name):
                    features.pop(name)
    if declared_parameters and (not decode or fixed_structure):
        # Official semantic features precede empirical vocabulary. In
        # particular, aligned anchors must not select a B-only approximation
        # of local-batch page rounding or finite-token routing imbalance.
        semantic = {
            key: value for key, value in features.items() if str(key).startswith(('prefill/', 'decode/', 'moe_route/'))
        }
        features = {**semantic, **features}
    return MappingProxyType(features)


def _lookup_feature(features: Mapping[Any, Any], name: Any) -> Any:
    if name in features:
        return features[name]
    text_name = str(name)
    if text_name in features:
        return features[text_name]
    # Honest refusal: an unevaluable dim must surface as TemplateExpandError
    # (the coverage/expansion contract) - a bare KeyError here escaped as a
    # crash on malformed builder output instead of a refusal.
    raise TemplateExpandError(f"Unknown template feature: {name!r}")


def _formula_param_value(param: Any, config: Mapping[str, Any], features: Mapping[Any, Any]) -> Any:
    if param is None:
        return 1
    if param in features:
        return features[param]
    if str(param) in features:
        return features[str(param)]
    return config.get(param, config.get(str(param), 1))


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == ">":
        return actual > expected
    if op == "<=":
        return actual <= expected
    if op == ">=":
        return actual >= expected
    if op == "<":
        return actual < expected
    if op == "!=":
        return actual != expected
    if op == "==":
        return actual == expected
    raise ValueError(f"Unsupported template condition operator: {op!r}")


# --- Single formula-evaluator implementation (issue #11) --------------------
#
# One evaluator per retained shape-formula type, dispatched through
# ``_SHAPE_FORMULA_EVALUATORS``.  Both consumers route through this table:
# ``eval_formula`` (shape dimensions, stride formulas, graph-program integer
# formulas) and ``_eval_repeat_formula`` (loop/count/condition formulas,
# which delegate uppercase shape types and reuse the LINEAR/MULTI/FAILED
# implementations for their feature-based lowercase spellings).  Behaviour
# is unchanged - the branches below are the former ``eval_formula`` bodies,
# moved verbatim.  MULTI evaluates directly on the features ``eval_formula``
# already fetched instead of re-deriving them per call (review issue #30).


def _eval_const_formula(formula: Mapping[str, Any], config: Mapping[str, Any], features: Mapping[Any, Any]) -> Any:
    return _integer_if_close(formula.get("value"))


def _eval_failed_formula(formula: Mapping[str, Any], config: Mapping[str, Any], features: Mapping[Any, Any]) -> Any:
    # Internal builder sentinel: never ships in a delivered artifact (the
    # builder converts FAILED dims to EXACT_LOOKUP before serialization), but
    # the derivation pipeline and the coverage gates speak this marker.
    return None


def _eval_exact_lookup_formula(
    formula: Mapping[str, Any], config: Mapping[str, Any], features: Mapping[Any, Any]
) -> Any:
    # The dim is only known at captured configurations: replay the value
    # of the exactly-matching sample, otherwise report unknown (the
    # coverage/strict gates refuse such targets before expansion).
    config_view = _lookup_config_view(config)
    for candidate in formula.get("entries", ()):  # noqa: B007
        match = _lookup_config_view(candidate.get("match", {}))
        if all(config_view.get(key) == value for key, value in match.items()):
            return _integer_if_close(candidate.get("value"))
    return None


def _eval_feature_linear_formula(
    formula: Mapping[str, Any], config: Mapping[str, Any], features: Mapping[Any, Any]
) -> Any:
    value = float(formula.get("coeff", 1)) * _lookup_feature(features, formula.get("feat")) + formula.get("offset", 0)
    return _integer_if_close(value)


def _eval_multi_formula(formula: Mapping[str, Any], config: Mapping[str, Any], features: Mapping[Any, Any]) -> Any:
    offset = formula.get("offset", 0)
    coeffs = formula.get("coeffs", [])
    feature_names = formula.get("feats", formula.get("features", []))
    if isinstance(coeffs, Mapping):
        return _integer_if_close(
            offset
            + sum(
                float(coefficient) * _lookup_feature(features, feature_name)
                for feature_name, coefficient in coeffs.items()
            )
        )
    return _integer_if_close(
        offset
        + sum(
            float(coefficient) * _lookup_feature(features, feature_name)
            for coefficient, feature_name in zip(coeffs, feature_names)
        )
    )


def _eval_domain_ceil_div_formula(
    formula: Mapping[str, Any], config: Mapping[str, Any], features: Mapping[Any, Any]
) -> Any:
    feature_value = _lookup_feature(features, formula.get("feat"))
    divisor = int(formula.get("divisor", 1))
    if divisor <= 0:
        return None
    if formula.get('require_divisible') and int(feature_value) % divisor:
        return None
    return _integer_if_close(-(-int(feature_value) // divisor))


_SHAPE_FORMULA_EVALUATORS = {
    "CONST": _eval_const_formula,
    "FAILED": _eval_failed_formula,
    "EXACT_LOOKUP": _eval_exact_lookup_formula,
    "LINEAR": _eval_feature_linear_formula,
    "MULTI": _eval_multi_formula,
    "DOMAIN_CEIL_DIV": _eval_domain_ceil_div_formula,
}


def eval_formula(formula: Any, config: Mapping[str, Any]) -> Any:
    """Evaluate a shape formula from an entry's ``*_f`` fields."""

    if formula is None:
        return None
    if isinstance(formula, bool) or isinstance(formula, numbers.Number):
        return _integer_if_close(formula)
    if not isinstance(formula, Mapping):
        return formula

    evaluator = _SHAPE_FORMULA_EVALUATORS.get(str(formula.get("type", "CONST")).upper())
    if evaluator is None:
        raise ValueError(f"Unsupported shape formula type: {formula.get('type')!r}")
    return evaluator(formula, config, _build_features(config))


def _eval_repeat_formula(formula: Any, config: Mapping[str, Any]) -> Any:
    """Evaluate a loop/count/condition formula."""

    if formula is None:
        return None
    if isinstance(formula, bool) or isinstance(formula, numbers.Number):
        return _integer_if_close(formula)
    if not isinstance(formula, Mapping):
        return formula

    raw_type = str(formula.get("type", "linear"))
    if raw_type in _SHAPE_FORMULA_TYPES:
        # Shared dispatch (issue #11): uppercase shape types in count
        # positions evaluate through the single shape-formula table.
        return _SHAPE_FORMULA_EVALUATORS[raw_type](formula, config, _build_features(config))
    formula_type = raw_type.lower()
    features = _build_features(config)

    if formula_type == "failed":
        return _eval_failed_formula(formula, config, features)
    if formula_type == "linear":
        param = formula.get("param", "B")
        value = float(formula.get("coeff", 1)) * config.get(param, 1) + formula.get("offset", 0)
        return _integer_if_close(value)
    if formula_type == "linear_padded":
        b = _config_int(config, "B")
        pad_param = formula.get("pad_param", formula.get("param", "MOE_DP"))
        padding = _config_int(config, str(pad_param), 1)
        effective_b = _ceil_to_multiple(b, padding) if padding > 1 else b
        value = float(formula.get("coeff", 1)) * effective_b + formula.get("offset", 0)
        return _integer_if_close(value)
    if formula_type == "linear_feat":
        # Same schema as shape LINEAR - one shared implementation.
        return _eval_feature_linear_formula(formula, config, features)
    if formula_type == "step_bilinear":
        param_value = _formula_param_value(formula.get("param"), config, features)
        if param_value < formula.get("threshold", 0):
            value = float(formula.get("low_coeff", 0)) * _lookup_feature(
                features, formula.get("low_feat")
            ) + formula.get("low_offset", 0)
        else:
            value = float(formula.get("high_coeff", 1)) * _lookup_feature(
                features, formula.get("high_feat")
            ) + formula.get("high_offset", 0)
        return _integer_if_close(value)
    if formula_type == "multi":
        # Same schema as shape MULTI - one shared implementation.
        return _eval_multi_formula(formula, config, features)
    if formula_type == "compound_condition":
        conditions = formula.get("conditions", [])
        result = all(_compare(config.get(c.get("param"), 1), c.get("op", "=="), c.get("value")) for c in conditions)
        return 1.0 if result else 0.0
    if formula_type == "feature_condition":
        return 1.0 if _lookup_feature(features, formula.get("feat")) > 0.5 else 0.0
    raise ValueError(f"Unsupported repeat formula type: {formula.get('type')!r}")


def _evaluate_count_formula(formula: Any, config: Mapping[str, Any], default: int = 0) -> int:
    if formula is None:
        return default
    value = _eval_repeat_formula(formula, config)
    if value is None:
        return 0
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError, OverflowError):
        return default


def _tensor_sizes(values: Sequence[Any], *, strict: bool) -> tuple[int, ...]:
    if strict:
        for value in values:
            if (
                not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or float(value) != int(value)
                or value < 0
            ):
                raise TemplateExpandError(f"invalid tensor dimension/stride: {value!r}")
    return tuple(int(round(float(value))) for value in values)


def rebuild_tensor(orig: torch.Tensor, new_shape: Sequence[Any], *, strict: bool = False) -> torch.Tensor:
    """Rebuild a meta tensor with the requested shape and compatible strides."""

    shape = _tensor_sizes(new_shape, strict=strict)
    if any(dimension < 0 for dimension in shape):
        return orig
    if tuple(orig.shape) == shape:
        return orig

    old_shape = tuple(int(dimension) for dimension in orig.shape)
    old_stride = tuple(int(stride) for stride in orig.stride())
    if orig.is_contiguous() or len(old_stride) != len(shape) or len(old_shape) != len(shape):
        stride: list[int] = [0] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            stride[index] = running
            running *= max(shape[index], 1)
    else:
        stride = []
        for old_dimension, old_value, old_stride_value in zip(old_shape, shape, old_stride):
            if old_dimension == 0:
                stride.append(old_stride_value)
            else:
                stride.append(int(round(old_stride_value * old_value / old_dimension)))

    return torch.empty_strided(
        shape,
        tuple(stride),
        dtype=orig.dtype,
        device="meta",
        requires_grad=orig.requires_grad,
    )


def _apply_tensor_shape_stride(
    orig: torch.Tensor,
    formulas: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    strict: bool = False,
) -> torch.Tensor:
    """Apply an explicit ``{"shape": [...], "stride": [...]}`` mapping to a tensor.

    Used for non-contiguous views (slices/permutations of the paged-KV pool)
    whose strides do not follow from the shape alone.  A missing ``shape`` key
    keeps the anchor shape; a missing ``stride`` key falls back to
    ``rebuild_tensor``'s stride handling.  Any ``None`` evaluation keeps the
    anchor tensor unchanged.
    """

    shape_formulas = formulas.get("shape")
    if shape_formulas:
        new_shape = [eval_formula(formula, config) for formula in shape_formulas]
        if any(dimension is None for dimension in new_shape):
            if strict:
                raise TemplateExpandError("tensor dimension formula is unresolved")
            return orig
    else:
        new_shape = [int(dimension) for dimension in orig.shape]
    stride_formulas = formulas.get("stride")
    if not stride_formulas:
        return rebuild_tensor(orig, new_shape, strict=strict)
    shape = _tensor_sizes(new_shape, strict=strict)
    new_stride: list[int | None] = []
    for dim, formula in enumerate(stride_formulas):
        if isinstance(formula, Mapping) and formula.get("type") == "CONTIGUOUS":
            # Structurally derived: a stride dim that equaled the contiguous
            # product at every capture is the shape product at any target
            # (pilot: fitted stride extrapolation produced 364544 vs the
            # contiguous 322560 at an untrained point).
            product = 1
            for size in shape[dim + 1 :]:
                product *= size
            new_stride.append(product)
        else:
            new_stride.append(eval_formula(formula, config))
    if any(value is None for value in new_stride):
        if strict:
            raise TemplateExpandError("tensor stride formula is unresolved")
        return orig
    stride = _tensor_sizes(new_stride, strict=strict)
    if any(dimension < 0 for dimension in shape) or any(value < 0 for value in stride):
        return orig
    if tuple(orig.shape) == shape and tuple(orig.stride()) == stride:
        return orig
    return torch.empty_strided(
        shape,
        stride,
        dtype=orig.dtype,
        device="meta",
        requires_grad=orig.requires_grad,
    )


def apply_formula_to_val(
    val: Any,
    formulas: Any,
    config: Mapping[str, Any],
    *,
    strict: bool = False,
) -> Any:
    """Recursively apply shape formulas to tensors, containers and scalars."""

    if formulas is None:
        return val
    if isinstance(val, torch.Tensor):
        if not formulas:
            return val
        if isinstance(formulas, Mapping) and ("shape" in formulas or "stride" in formulas):
            return _apply_tensor_shape_stride(val, formulas, config, strict=strict)
        if not isinstance(formulas, (list, tuple)):
            formulas = [formulas]
        new_shape = [eval_formula(formula, config) for formula in formulas]
        if any(dimension is None for dimension in new_shape):
            if strict:
                raise TemplateExpandError("tensor dimension formula is unresolved")
            return val
        return rebuild_tensor(val, new_shape, strict=strict)

    if isinstance(val, list):
        if isinstance(formulas, Mapping):
            formula_type = str(formulas.get("type", "")).lower()
            if formula_type == "variable_list":
                count = _evaluate_count_formula(formulas.get("count_formula"), config)
                reference = val[0] if val else None
                per_index = formulas.get("element_formulas")
                if isinstance(per_index, list):
                    # Indexed form: element i may differ from element 0 (e.g.
                    # TP group lists whose first entry flags the local rank).
                    out = []
                    for index in range(count):
                        element_formula = (
                            per_index[index] if index < len(per_index) else formulas.get("element_formula")
                        )
                        indexed_config = {**config, "__list_index__": index}
                        out.append(apply_formula_to_val(reference, element_formula, indexed_config, strict=strict))
                    return out
                return [
                    apply_formula_to_val(reference, formulas.get("element_formula"), config, strict=strict)
                    for _ in range(count)
                ]
            if formula_type == "binary_int_list":
                count = _evaluate_count_formula(formulas.get("count_formula"), config)
                length = int(formulas.get("length", len(val)))
                return [1 if index < count else 0 for index in range(max(0, length))]
            if formula_type == "binary_tensor_list":
                count = _evaluate_count_formula(formulas.get("count_formula"), config)
                length = int(formulas.get("length", len(val)))
                other_dims = formulas.get("other_dims")
                result = []
                for index in range(max(0, length)):
                    reference = val[index] if index < len(val) else (val[0] if val else None)
                    if not isinstance(reference, torch.Tensor):
                        result.append(reference)
                        continue
                    if other_dims is None:
                        dimensions = list(reference.shape[1:])
                    else:
                        dimensions = [int(round(float(dimension))) for dimension in other_dims]
                    result.append(rebuild_tensor(reference, [1 if index < count else 0, *dimensions]))
                return result
        if not formulas:
            return val
        return [apply_formula_to_val(item, formula, config, strict=strict) for item, formula in zip(val, formulas)]

    if isinstance(val, tuple):
        result = apply_formula_to_val(list(val), formulas, config, strict=strict)
        return tuple(result) if isinstance(result, list) else result

    if isinstance(val, dict):
        if formulas is None:
            return val
        return {
            key: apply_formula_to_val(
                item, formulas.get(key) if isinstance(formulas, Mapping) else None, config, strict=strict
            )
            for key, item in val.items()
        }

    if isinstance(formulas, Mapping):
        evaluated = eval_formula(formulas, config)
        if strict and evaluated is None and not (formulas.get("type") == "CONST" and formulas.get("value") is None):
            raise TemplateExpandError(f"scalar formula is unresolved: {formulas!r}")
        if (
            strict
            and type(val) is int
            and (
                not isinstance(evaluated, numbers.Real)
                or not math.isfinite(float(evaluated))
                or float(evaluated) != int(evaluated)
            )
        ):
            raise TemplateExpandError("integer scalar formula is not integral")
        return val if evaluated is None else evaluated
    return val


def _restore_op(func: Any) -> Any:
    if not isinstance(func, str):
        return func
    name = func[10:] if func.startswith("torch.ops.") else func
    parts = name.split(".")
    if len(parts) < 2:
        return func
    namespace, op_name = parts[0], parts[1]
    overload = ".".join(parts[2:]) if len(parts) > 2 else "default"
    try:
        namespace_obj = getattr(torch.ops, namespace)
        op = getattr(namespace_obj, op_name)
        return getattr(op, overload)
    except (AttributeError, RuntimeError, TypeError):
        return func


def _restore_template_functions(value: Any) -> Any:
    if isinstance(value, dict):
        if "func" in value:
            value["func"] = _restore_op(value["func"])
        for child in value.values():
            _restore_template_functions(child)
        return value
    if isinstance(value, list):
        for child in value:
            _restore_template_functions(child)
        return value
    if isinstance(value, tuple):
        return tuple(_restore_template_functions(child) for child in value)
    return value


def load_template(path: str | Path) -> dict:
    """Load a template artifact and restore every serialized ``func`` field.

    Loading goes through the single sniffing boundary
    (``template_engine.zero_meta._load_template_any``): plain/zlib/xz pickles
    and lazy-shard bodies all load here, and the template_graph wrapper shape
    is unwrapped to its engine so every supply entry sees the same mapping
    regardless of the delivered file shape (review issue #22).
    """

    from template_engine.zero_meta import _load_template_any

    template = _load_template_any(path)
    if isinstance(template, dict) and isinstance(template.get("engine"), dict):
        template = template["engine"]
    if not isinstance(template, dict):
        raise TypeError(f"Template root must be a dict, got {type(template).__name__}")
    return _restore_template_functions(template)


def _has_failed_formula(formula: Any) -> bool:
    if isinstance(formula, Mapping):
        formula_type = str(formula.get("type", "")).upper()
        if formula_type in {"FAILED", "EXACT_LOOKUP"}:
            return True
        return any(_has_failed_formula(value) for value in formula.values())
    if isinstance(formula, (list, tuple, set)):
        return any(_has_failed_formula(value) for value in formula)
    return False


def _func_name(func: Any) -> str:
    return str(func)


def _region_kind(func: Any) -> str | None:
    name = _func_name(func)
    if "_internal_mark_region_begin" in name:
        return "begin"
    if "_internal_mark_region_end" in name:
        return "end"
    if "_internal_copy_region" in name:
        return "copy"
    return None


def _region_arg_index(args: Any) -> int | None:
    if not isinstance(args, (list, tuple)):
        return None
    if len(args) > 1 and isinstance(args[1], numbers.Integral):
        return 1
    # ``_internal_copy_region_v2`` carries its id after the output template.
    if len(args) > 2 and isinstance(args[2], numbers.Integral):
        return 2
    return None


def _shape_of(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(dimension) for dimension in value.shape)
    return None


def _replace_sequence_item(sequence: Any, index: int, value: Any) -> Any:
    if isinstance(sequence, tuple):
        result = list(sequence)
        result[index] = value
        return tuple(result)
    if isinstance(sequence, list):
        result = list(sequence)
        result[index] = value
        return result
    return sequence


def _reassign_region_ids(ops: list[OpInvokeInfo]) -> list[OpInvokeInfo]:
    """Pair begin/end/copy markers by shape and assign fresh region IDs.

    Ops whose region markers are already consistent (unique begin ids, ends
    closing a marked region, copies referencing a marked region) pass through
    unchanged: the template remap already assigned exact per-region ids, and a
    shape-based reassignment would collapse same-shape regions - every copy
    would point at the last closed region, silently rewiring which region
    body each copy duplicates.
    """

    def _marker(op: OpInvokeInfo) -> tuple[str, Any] | None:
        kind = _region_kind(op.func)
        if kind is None or not isinstance(op.args, (list, tuple)) or not op.args:
            return None
        arg_index = _region_arg_index(op.args)
        if arg_index is None:
            return None
        return kind, op.args[arg_index]

    begin_ids: set[Any] = set()
    for op in ops:
        marker = _marker(op)
        if marker is None:
            continue
        kind, region_id = marker
        if kind == "begin":
            if region_id in begin_ids:
                break
            begin_ids.add(region_id)
        elif region_id not in begin_ids:
            break
    else:
        return ops

    next_id = 1_000_000
    open_by_shape: dict[tuple[int, ...] | None, list[int]] = {}
    closed_by_shape: dict[tuple[int, ...] | None, list[int]] = {}

    def allocate() -> int:
        nonlocal next_id
        result = next_id
        next_id += 1
        return result

    for op in ops:
        kind = _region_kind(op.func)
        if kind is None or not isinstance(op.args, (list, tuple)):
            continue
        arg_index = _region_arg_index(op.args)
        if arg_index is None or not op.args:
            continue
        shape = _shape_of(op.args[0])
        if kind == "begin":
            region_id = allocate()
            open_by_shape.setdefault(shape, []).append(region_id)
        elif kind == "end":
            stack = open_by_shape.get(shape, [])
            region_id = stack.pop() if stack else allocate()
            closed_by_shape.setdefault(shape, []).append(region_id)
        else:
            closed = closed_by_shape.get(shape, [])
            open_regions = open_by_shape.get(shape, [])
            region_id = closed[-1] if closed else (open_regions[-1] if open_regions else allocate())
        op.args = _replace_sequence_item(op.args, arg_index, region_id)
        op.cache_key = op.compute_cache_key()
    return ops


def generate_ops_from_template(
    template: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    strict: bool = False,
    prevalidated: bool = False,
) -> list[OpInvokeInfo]:
    """Expand the closest template group for ``target_config`` into ops.

    With ``strict=True`` the expansion refuses to guess: admission is
    delegated to the coverage gate (:func:`template_engine.coverage.describe_coverage`),
    the single adjudicator of "can this config be served" (review issue #6).
    The refusal carries the gate's blocker code.  Strict mode stays the safe
    path for callers that do not run the gate first (review P0-1/P2-2);
    templates built before served_keys existed are rejected.

    ``prevalidated=True`` marks a gate-internal call: the gate already ran
    the full admission for this exact (template, config) pair, so only a
    light misuse assertion (signature membership + folded count ceiling)
    runs - the span/lookup/MTP scans are NOT repeated.  Outside the gate,
    do not pass it: the assertion is deliberately weaker than full admission.

    Every expansion owns its operation records and meta tensors. Runtime
    replay mutates both, so sharing cached operation objects is unsafe.
    """

    if not isinstance(template, Mapping):
        raise TypeError(f"Template must be a mapping, got {type(template).__name__}")
    groups = template.get("groups", {})
    if not isinstance(groups, Mapping) or not groups:
        return []
    if template.get("template_meta", {}).get("format") in (
        "capture_induced_fixed_cell_v2",
        "capture_induced_fixed_structure_v1",
    ):
        strict = True

    try:
        return _generate_ops_from_template_uncached(template, target_config, strict=strict, prevalidated=prevalidated)
    except TemplateExpandError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RuntimeError) as exc:
        raise TemplateExpandError(f"invalid template expansion: {exc}") from exc


def _generate_ops_from_template_uncached(
    template: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    strict: bool = False,
    prevalidated: bool = False,
) -> list[OpInvokeInfo]:
    """Expansion body of :func:`generate_ops_from_template` (uncached)."""

    groups = template.get("groups", {})
    if not isinstance(groups, Mapping) or not groups:
        return []

    from template_engine.model_parameters import bind_parameters

    config = bind_parameters(template, target_config)
    from template_engine.structure_buckets import prefill_anchor_config

    config = prefill_anchor_config(groups, config)
    tp = _config_int(config, "TP")
    ep = _config_int(config, "EP")
    moe_dp = _config_int(config, "MOE_DP")
    dcp = _config_int(config, "DCP")
    scatter_cat, pad_cat = _signature_categories(config)
    parallel_dimensions = (tp > 1, ep > 1, moe_dp > 1, dcp > 1)
    par_cat: Any = "single" if sum(parallel_dimensions) <= 1 else parallel_dimensions
    # The merge scope (this 4-tuple) is the serving comparison key: scatter/
    # pad are signature-level fields checked against served_keys below, not
    # group boundaries, so scatter-flipped targets of a known structure still
    # reach their group (they are then refused by the signature check unless
    # a flipped capture trained that signature).  It is derived here from the
    # config and on read from stored cells (body_select.base_key); the builder
    # does not materialise it anywhere (review issue #13).
    target_key = (
        config.get("quantize_linear_action", "DISABLED"),
        config.get("quantize_attention_action", "DISABLED"),
        par_cat,
        bool(config.get("is_decode", False)),
    )

    group_items = list(groups.items())
    # Mode-granular body selection (shared selection layer, also used by the
    # coverage gate): resolve the config's structure body by discrete
    # predicate with exact-anchor tiebreak; an unresolvable tie refuses
    # honestly (strict defers the refusal wording to the gate below).
    from template_engine.body_select import (
        STATUS_AMBIGUOUS,
        STATUS_NONE,
        STATUS_BRANCH_MISSING,
        STATUS_STRUCTURE_MISSING,
        select_body,
    )

    body_key, selected_group, body_status = select_body(groups, target_key, config)
    if body_status == STATUS_AMBIGUOUS:
        if strict:
            if prevalidated:
                # Misuse guard: a prevalidated call must come from the gate,
                # which never hands down an ambiguous selection.
                raise TemplateExpandError(
                    "strict expansion refused [blocker=mode_ambiguous]: mode-ambiguous "
                    f"target key {target_key!r} (prevalidated call from outside the gate?)"
                )
        else:
            selected_group = None
    elif body_status in (STATUS_NONE, STATUS_BRANCH_MISSING, STATUS_STRUCTURE_MISSING):
        if strict:
            if prevalidated:
                # Misuse guard: same contract - the gate never approves a
                # config with no candidate group.
                raise TemplateExpandError(
                    f"strict expansion refused [blocker=group_key]: no exact group for target "
                    f"key {target_key!r} (prevalidated call from outside the gate?)"
                )
        else:
            nearest = (
                min(
                    group_items,
                    key=lambda item: _group_distance(target_key, _normalise_key(item[0])),
                )
                if group_items
                else None
            )
            if nearest is not None:
                body_key, selected_group = nearest[0], nearest[1]
            else:
                selected_group = None

    if strict and not prevalidated:
        # Judgement convergence (review issue #6): the expander no longer
        # owns admission.  The coverage gate is the single adjudicator -
        # delegate the FULL admission decision to it and refuse with its
        # blocker taxonomy (one wording for one fact: mode_ambiguous,
        # group_key, signature, exact_lookup, mtp_bs, bs_* ...).  The
        # gate's own expansion call comes back here with prevalidated=True
        # (light assertion only), so the rules run exactly once per
        # admission.  The approved fall-through still performs THIS call's
        # own expansion: operation records and meta tensors are
        # per-expansion state and must never be shared between calls.
        from template_engine.coverage import describe_coverage  # noqa: PLC0415

        coverage_view = describe_coverage(template, config)
        verdict = str(coverage_view.get("verdict"))
        failed = int(coverage_view.get("failed_formulas", 0) or 0)
        servable = verdict == "exact_group" or (verdict == "interpolated" and failed == 0)
        if not servable:
            blocker = coverage_view.get("blocker") or verdict
            raise TemplateExpandError(
                f"strict expansion refused [blocker={blocker}] verdict={verdict}: "
                f"{coverage_view.get('reason', 'target is not servable from this template')}"
            )
    if not isinstance(selected_group, Mapping):
        return []

    # Lazy-shard templates carry group bodies as compressed blobs; the
    # selected group's body must be materialised before any structure read
    # (the strict guards walk EXACT_LOOKUP dims inside the body).  No-op on
    # fully materialised templates.
    from template_engine.zero_meta import ensure_group_body  # noqa: PLC0415

    ensure_group_body(template, body_key)
    if strict:
        # Gate-internal call (prevalidated=True): the gate already
        # adjudicated this exact (template, config) pair.  Keep a light
        # internal assertion for caller misuse - the O(served) signature
        # membership and the folded count ceiling - but never repeat the
        # full span/lookup/MTP scans (single-judgement discipline).
        from template_engine.serve_rules import prefill_window_blocker

        branch_blocker = prefill_window_blocker(selected_group, config)
        if branch_blocker:
            raise TemplateExpandError(branch_blocker)
        served = selected_group.get("served_keys")
        target_member = {
            "par_sig": [tp, ep, moe_dp, dcp],
            "scatter_cat": scatter_cat,
            # pad_cat must use the SAME decode-aware categorisation the
            # builder wrote into served_keys and the gate's target_member
            # uses (_signature_categories): a decode forward dispatches
            # only its query tokens, so deriving pad from the context
            # length mis-tagged decode anchors.
            "pad_cat": pad_cat,
            "num_mtp_tokens": int(config.get("num_mtp_tokens", 0) or 0),
            "enable_dispatch_ffn_combine": bool(config.get("enable_dispatch_ffn_combine", False)),
            "is_decode": bool(config.get("is_decode", False)),
            "do_compile": config.get("do_compile"),
        }
        served_exact = False
        if isinstance(served, list):
            for item in served:
                if not isinstance(item, Mapping):
                    continue
                from template_engine.serve_rules import signature_fields

                if all(_key_equal(item.get(field), target_member[field]) for field in signature_fields(selected_group)):
                    served_exact = True
                    break
        if not served_exact:
            raise TemplateExpandError(
                f"strict expansion refused [blocker=signature]: {target_member!r} is not "
                "served by any capture in this group (prevalidated call from outside the gate?)"
            )
    region_id_map: dict[tuple[Any, Any], int] = {}
    next_region_id = [1_000_000]
    output: list[OpInvokeInfo] = []
    tensor_pool: dict[int, torch.Tensor] = {}
    tensor_origins: dict[int, str] = {}

    def wire_tensors(base: Any, value: Any, path: str) -> Any:
        """Preserve captured references while keeping this expansion isolated."""
        if isinstance(base, torch.Tensor) and isinstance(value, torch.Tensor):
            key = id(base)
            previous = tensor_pool.get(key)
            if previous is not None:
                if (
                    previous.shape != value.shape
                    or previous.stride() != value.stride()
                    or previous.dtype != value.dtype
                    or previous.requires_grad != value.requires_grad
                ):
                    raise TemplateExpandError(
                        "inconsistent formulas for one captured tensor identity: "
                        f"{tensor_origins[key]} -> {path}: "
                        f"{tuple(previous.shape)}/{previous.stride()} vs {tuple(value.shape)}/{value.stride()}"
                    )
                return previous
            cloned = torch.empty_strided(
                tuple(value.shape),
                value.stride(),
                dtype=value.dtype,
                device="meta",
                requires_grad=value.requires_grad,
            )
            tensor_pool[key] = cloned
            tensor_origins[key] = path
            return cloned
        if isinstance(base, (list, tuple)) and isinstance(value, (list, tuple)):
            # Scalar metadata containers have no producer/consumer identities.
            # Their formula-derived length may exceed the anchor's length.
            if not any(True for _ in _iter_base_tensors(base)) and not any(True for _ in _iter_base_tensors(value)):
                return value
            if len(value) > len(base):
                raise TemplateExpandError("tensor container exceeds the captured structure")
            items = [wire_tensors(old, new, f"{path}[{index}]") for index, (old, new) in enumerate(zip(base, value))]
            return tuple(items) if isinstance(value, tuple) else items
        if isinstance(base, Mapping) and isinstance(value, Mapping):
            return {key: wire_tensors(base[key], item, f"{path}[{key!r}]") for key, item in value.items()}
        return value

    def _iter_base_tensors(value: Any):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _iter_base_tensors(item)
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from _iter_base_tensors(item)

    def remap_region_args(base_args: Any, args: Any, region_map: dict[tuple[Any, Any], int], replica=None) -> Any:
        if not isinstance(base_args, (list, tuple)) or not isinstance(args, (list, tuple)):
            return args
        base_index = _region_arg_index(base_args)
        if base_index is None or base_index >= len(args):
            return args
        original_id = base_args[base_index]
        shape = _shape_of(base_args[0]) if base_args else None
        # Input/output shapes may differ inside a valid region (e.g. DP
        # transforms). Identity belongs to the captured region and explicit
        # repetition namespace, never to its boundary tensor shape.
        region_key = (int(original_id), replica)
        if region_key not in region_map:
            region_map[region_key] = next_region_id[0]
            next_region_id[0] += 1
        if os.environ.get("TEMPLATE_REGION_TRACE"):
            kind = "copy" if len(base_args) > 2 and isinstance(base_args[1], torch.Tensor) else "begin"
            logging.getLogger(__name__).debug(
                "[region-remap] %s base_id=%s shape=%s -> %s",
                kind,
                original_id,
                shape,
                region_map[region_key],
            )
        return _replace_sequence_item(args, base_index, region_map[region_key])

    def selected_formula(entry: Mapping[str, Any], normal_name: str, cross_name: str, extrapolated: bool) -> Any:
        if extrapolated and cross_name in entry:
            candidate = entry[cross_name]
            if not _has_failed_formula(candidate):
                return candidate
        return entry.get(normal_name)

    def emit_one(
        entry: Mapping[str, Any],
        region_map: dict[tuple[Any, Any], int],
        extrapolated: bool = False,
    ) -> None:
        base_args = entry.get("base_args", ())
        base_kwargs = entry.get("base_kwargs", {})
        base_out = entry.get("base_out")
        args_formula = selected_formula(entry, "args_f", "cross_args_f", extrapolated)
        kwargs_formula = selected_formula(entry, "kwargs_f", "cross_kwargs_f", extrapolated)
        out_formula = selected_formula(entry, "out_f", "cross_out_f", extrapolated)
        args = apply_formula_to_val(base_args, args_formula, config, strict=strict)
        kwargs = apply_formula_to_val(base_kwargs, kwargs_formula, config, strict=strict)
        out = apply_formula_to_val(base_out, out_formula, config, strict=strict)

        label = f"op{len(output)}:{entry.get('func')}"
        args = wire_tensors(base_args, args, f"{label}.args")
        kwargs = wire_tensors(base_kwargs, kwargs, f"{label}.kwargs")
        out = wire_tensors(base_out, out, f"{label}.out")

        # Producer alias discipline (build-side ``out_alias_prev``): the
        # capture reuses the previous op's out object as this op's out, so
        # expansion must wire the SAME object - a fresh tensor would change
        # the runtime's identity-based dependency/DCE topology and shift the
        # performance replay.  Shape must agree; otherwise keep the fresh
        # tensor (defensive - an inconsistent pair never gets the flag).
        if entry.get("out_alias_prev") and output:
            prev_out = output[-1].out
            if hasattr(prev_out, "shape") and hasattr(out, "shape") and tuple(prev_out.shape) == tuple(out.shape):
                out = prev_out

        func = _restore_op(entry.get("func"))
        if func is None:
            raise ValueError("Template entry is missing func")
        if _region_kind(func) is not None:
            base_index = _region_arg_index(base_args)
            if base_index is not None and isinstance(base_args[base_index], numbers.Integral):
                args = remap_region_args(base_args, args, region_map)
        output.append(OpInvokeInfo(func, args, kwargs, out))

    for entry in selected_group['entries']:
        variants = entry.get('parallel_variants')
        if variants is None:
            emit_one(entry, region_id_map)
        else:
            variant, extrapolated = _select_parallel_variant(variants, tp, ep, moe_dp, dcp, scatter_cat, pad_cat)
            if variant is None:
                raise TemplateExpandError('No parallel variant matches the template')
            for item in variant.get('entries', []):
                emit_one(item, region_id_map, extrapolated)

    return _reassign_region_ids(output)


__all__ = [
    "MODEL_CONSTANTS",
    "apply_formula_to_val",
    "eval_formula",
    "generate_ops_from_template",
    "load_template",
    "rebuild_tensor",
]
