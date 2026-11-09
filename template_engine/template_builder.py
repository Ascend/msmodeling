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

"""Partition captures into fixed operator/reference buckets and fit parameters."""

from __future__ import annotations

from .graph_codec import read_artifact_bytes

import copy
import re
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from template_generate import (
    _region_arg_index,
    _region_kind,
)

from .cache_io import capture_config, load_capture
from .formulas import (
    cache_value_formulas,
    derive_value_formula,
)

_KEY_PATTERN = re.compile(
    r"^tp(?P<tp>\d+)_ep(?P<ep>\d+)_md(?P<md>\d+)_B(?P<B>\d+)_S(?P<S>\d+?)"
    r"(?:_mtp(?P<mtp>\d+))?(?:_dcp(?P<dcp>\d+))?(?P<dec>_dec)?"
    r"(?:_cmp_(?P<compile>on|off))?_dfc_(?P<dfc>on|off)"
    r"(?:_(?:(?P<q_disabled>DISABLED)|q(?P<q_lin>.+?)_(?P<q_attn>[A-Za-z0-9]+)))?"
    r"_L1$"
)


class TemplateBuildError(ValueError):
    """Raised when captures cannot be assembled into a template."""


def parse_cache_key(key: str) -> dict[str, Any]:
    """Parse a Card 1 cache key into an expander-side config mapping.

    Legacy keys carry ``_DISABLED_L1`` (quantization disabled); keys of other
    quantization families use ``_q{LINEAR}_{ATTENTION}_L1`` (see
    ``template_engine.cache_io.build_cache_key``).
    """

    match = _KEY_PATTERN.match(key)
    if match is None:
        raise TemplateBuildError(f"unrecognised cache key {key!r}")
    if match.group("q_disabled"):
        linear_action, attention_action = "DISABLED", "DISABLED"
    elif match.group("q_lin"):
        linear_action, attention_action = match.group("q_lin"), match.group("q_attn")
    else:
        linear_action, attention_action = "DISABLED", "DISABLED"
    return {
        "TP": int(match.group("tp")),
        "EP": int(match.group("ep")),
        "MOE_DP": int(match.group("md")),
        "DCP": int(match.group("dcp") or 1),
        "B": int(match.group("B")),
        "S": int(match.group("S")),
        "num_mtp_tokens": int(match.group("mtp") or 0),
        "enable_dispatch_ffn_combine": match.group("dfc") == "on",
        "is_decode": match.group("dec") is not None,
        "do_compile": None if match.group("compile") is None else match.group("compile") == "on",
        "quantize_linear_action": linear_action,
        "quantize_attention_action": attention_action,
    }


def _parallel_signature(config: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (config["TP"], config["EP"], config["MOE_DP"], config["DCP"])


def _categories(config: Mapping[str, Any]) -> tuple[bool, bool]:
    """Signature (scatter, pad) categories, decode-aware.

    Decode forwards dispatch only their query tokens, so the categories
    must not be derived from the context length (see
    ``template_generate._signature_categories``).
    """

    from template_generate import _signature_categories

    return _signature_categories(config)


def _op_func(op: Any) -> str:
    return str(op.func)


def _op_args(op: Any) -> Any:
    return op.args


def _op_kwargs(op: Any) -> Any:
    return op.kwargs


def _op_out(op: Any) -> Any:
    return op.out


def _shape_fingerprint(value: Any) -> tuple[Any, ...]:
    """Structure/shape fingerprint used for cross-config op alignment."""

    if isinstance(value, torch.Tensor):
        return ("T", tuple(int(dim) for dim in value.shape), str(value.dtype))
    if isinstance(value, dict):
        return ("D", tuple(sorted((str(key), _shape_fingerprint(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_shape_fingerprint(item) for item in value))
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("S", type(value).__name__, value)
    return ("X", repr(value))


def _op_fingerprint(op: Any) -> tuple[Any, ...]:
    func = _op_func(op)
    args = _op_args(op)
    if _region_kind(func) is not None:
        # Region IDs are unstable across captures; drop them from the
        # fingerprint (and from shape hashing of the marker payload).
        args = _without_region_id(args)
    return (
        func,
        _shape_fingerprint(args),
        _shape_fingerprint(_op_kwargs(op)),
        _shape_fingerprint(_op_out(op)),
    )


def _without_region_id(args: Any) -> Any:
    if not isinstance(args, (list, tuple)):
        return args
    index = _region_arg_index(args)
    if index is None:
        return args
    result = list(args)
    if index < len(result):
        result[index] = "<region-id>"
    return tuple(result) if isinstance(args, tuple) else result


@dataclass
class _Capture:
    key: str
    config: dict[str, Any]
    ops: list[Any]
    exec_time: float


@dataclass
class BuildReport:
    """Coverage disclosure for one template build."""

    source_keys: list[str] = field(default_factory=list)
    families: list[dict[str, Any]] = field(default_factory=list)
    derived_formulas: int = 0
    failed_formulas: int = 0
    identity_only_families: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_keys": self.source_keys,
            "families": self.families,
            "derived_formulas": self.derived_formulas,
            "failed_formulas": self.failed_formulas,
            "identity_only_families": self.identity_only_families,
            "notes": self.notes,
        }


def _failed_to_lookup(formula: Mapping[str, Any], samples: Sequence[tuple[Mapping[str, Any], int]]) -> Any:
    """Turn a FAILED scalar formula into an exact per-config lookup table.

    A FAILED dimension is only verified at the captured configurations: the
    lookup replays the observed value for an exactly-matching config and
    reports unknown otherwise (coverage/strict refuse such targets).  This
    keeps multi-anchor families exact on every captured anchor instead of
    freezing everyone to the anchor-first base value.
    """

    if formula.get("type") != "FAILED":
        return formula
    entries: list[dict[str, Any]] = []
    for config, value in samples:
        match = dict(config)
        if any(entry["match"] == match for entry in entries):
            continue
        entries.append({"match": match, "value": int(value)})
    if not entries:
        return formula
    return {"type": "EXACT_LOOKUP", "entries": entries}


def _contiguous_stride(tail: Any) -> int:
    """The contiguous stride for a dim whose trailing shape sizes are ``tail``."""

    product = 1
    for size in tail:
        product *= int(size)
    return product


def _routing_vector_features(values, configs):
    """Match an entire official split vector, never unrelated scalar columns."""
    if any(c.get('is_decode') or not c.get('model_parameters') for c in configs):
        return None
    if not values or not all(isinstance(v, (list, tuple)) for v in values):
        return None
    if not values[0] or len({len(v) for v in values}) != 1:
        return None
    tensors = all(all(isinstance(item, torch.Tensor) and item.ndim > 0 for item in v) for v in values)
    integers = all(all(type(item) is int for item in v) for v in values)
    if not tensors and not integers:
        return None
    from .formulas import _features_for

    vectors = [[int(item.shape[0]) if tensors else item for item in v] for v in values]
    tables = [_features_for(c) for c in configs]
    for role in ('merged_expert', 'input_device', 'output_device', 'input_expert'):
        prefix = f'moe_route/{role}/'
        names = [prefix + str(i) for i in range(len(vectors[0]))]
        if any(prefix + str(len(names)) in table for table in tables):
            continue
        if all(
            all(table.get(name) == value for name, value in zip(names, vector))
            for table, vector in zip(tables, vectors)
        ):
            return names, tensors
    return None


def _derive_formulas_tree(
    anchor_value: Any,
    sample_values: Sequence[Any],
    configs: Sequence[Mapping[str, Any]],
    report: BuildReport,
    path: str,
) -> Any:
    """Derive a formula tree mirroring ``anchor_value``; ``None`` keeps value.

    Convention: ``sample_values[0]`` is the anchor's own value and
    ``configs[0]`` is the anchor's config, so ``zip(configs, sample_values)``
    is always aligned.
    """

    if isinstance(anchor_value, torch.Tensor):
        if sample_values and not all(isinstance(item, torch.Tensor) for item in sample_values):
            report.notes.append(f"{path}: tensor/scalar mismatch across anchors; keeping anchor value")
            return None
        if anchor_value.ndim == 0:
            return None

        def derive_tensor_dims(value_of, stride: bool = False) -> tuple[list[Any], bool, bool]:
            formulas: list[Any] = []
            any_derived = False
            any_failed = False
            for dim in range(anchor_value.ndim):
                samples = [(config, int(value_of(tensor, dim))) for config, tensor in zip(configs, sample_values)]
                if (
                    stride
                    and samples
                    and all(
                        int(tensor.stride()[dim]) == _contiguous_stride(tensor.shape[dim + 1 :])
                        for _, tensor in zip(configs, sample_values)
                    )
                ):
                    # Structurally determined: the stride equals the
                    # contiguous product at every capture, so it equals it at
                    # any target - no fit, no extrapolation risk, and the
                    # emit side derives it from the generated shape.
                    formulas.append({"type": "CONTIGUOUS", "dim": dim})
                    any_derived = True
                    continue
                formula = derive_value_formula(samples)
                if formula.get("type") == "FAILED":
                    if stride and all(c.get('structure_policy') == 'fixed_structure_v1' for c in configs):
                        # Performance replay consumes shapes, not physical layout.
                        # An unidentifiable layout must not freeze valid shapes
                        # to the training coordinates.
                        formula = {'type': 'CONTIGUOUS', 'dim': dim}
                    else:
                        formula = _failed_to_lookup(formula, samples)
                formulas.append(formula)
                if formula.get("type") == "CONST":
                    continue
                any_derived = True
                if formula.get("type") in {"FAILED", "EXACT_LOOKUP"}:
                    any_failed = True
            return formulas, any_derived, any_failed

        shape_formulas, shape_derived, shape_failed = derive_tensor_dims(lambda t, dim: t.shape[dim])
        stride_formulas, stride_derived, stride_failed = derive_tensor_dims(lambda t, dim: t.stride()[dim], stride=True)

        if not shape_derived and not stride_derived:
            return None
        report.derived_formulas += 1
        if shape_failed or stride_failed:
            # A dimension that no formula explains becomes an EXACT_LOOKUP:
            # every captured anchor still replays exactly (lookup hit), and
            # unseen configs are refused by the coverage/strict gates instead
            # of being silently filled with the anchor-first base value.
            report.failed_formulas += 1
            report.notes.append(f"{path}: dimension formula kept as EXACT_LOOKUP")
        if not stride_derived:
            # Review P2-4: keep per-dimension CONST strides alongside the
            # shape formulas instead of letting rebuild_tensor guess the
            # layout for non-contiguous views.  (shape_derived is guaranteed
            # True here: both-false returned None above.)
            return {"shape": shape_formulas, "stride": stride_formulas}
        payload: dict[str, Any] = {"stride": stride_formulas}
        if shape_derived:
            payload["shape"] = shape_formulas
        return payload

    if isinstance(anchor_value, (list, tuple)):
        if sample_values and not all(isinstance(item, (list, tuple)) for item in sample_values):
            report.notes.append(f"{path}: container mismatch across anchors; keeping anchor value")
            return None
        lengths = [len(item) for item in sample_values]
        vector = _routing_vector_features(sample_values, configs)
        if vector is not None:
            names, tensors = vector
            if not tensors:
                report.derived_formulas += len(names)
                return [{'type': 'LINEAR', 'feat': name, 'coeff': 1, 'offset': 0} for name in names]
            result = []
            for index, name in enumerate(names):
                reference = anchor_value[index]
                item = (
                    _derive_formulas_tree(
                        reference, [v[index] for v in sample_values], configs, report, f'{path}[{index}]'
                    )
                    or {}
                )
                shape = list(item.get('shape', [{'type': 'CONST', 'value': int(d)} for d in reference.shape]))
                shape[0] = {'type': 'LINEAR', 'feat': name, 'coeff': 1, 'offset': 0}
                result.append({**item, 'shape': shape})
            return result
        if len(set(lengths)) > 1:
            # Variable-length list: derive the count and reuse the reference
            # element structure.  This covers e.g. split/chunk outputs whose
            # length tracks a configuration axis.
            count_samples = list(zip(configs, lengths))
            count_formula = derive_value_formula(count_samples)
            if count_formula.get("type") == "FAILED":
                count_formula = _failed_to_lookup(count_formula, count_samples)
                report.failed_formulas += 1
                report.notes.append(f"{path}: variable list length kept as EXACT_LOOKUP")
            else:
                report.derived_formulas += 1
            element_formula = None
            per_index_formulas: list[Any] | None = None
            if anchor_value and all(lengths):
                first_elements = [value[0] for value in sample_values]
                # first_elements already follows the anchor-first convention.
                element_formula = _derive_formulas_tree(
                    first_elements[0], first_elements, configs, report, f"{path}[0]"
                )
                # A repeated scalar is one parameter law over a symbolic
                # count. Per-index fits would freeze the sparsely observed
                # tail (e.g. query lengths at larger local batch sizes).
                uniform_scalars = all(
                    values
                    and type(values[0]) in (int, bool)
                    and all(type(item) is type(values[0]) and item == values[0] for item in values)
                    for values in sample_values
                )
                if uniform_scalars:
                    return {'type': 'variable_list', 'count_formula': count_formula, 'element_formula': element_formula}
                # Indexed derivation: when captured lists of different lengths
                # hold non-uniform elements (e.g. TP group flags where entry 0
                # differs), derive each index from the samples that have it.
                max_length = max(lengths)
                if max_length > 1:
                    candidates: list[Any] = []
                    exact = True
                    for index in range(max_length):
                        index_anchor = anchor_value[index] if index < len(anchor_value) else None
                        index_samples = [
                            value[index] for value, length in zip(sample_values, lengths) if length > index
                        ]
                        index_configs = [config for config, length in zip(configs, lengths) if length > index]
                        if index_anchor is None or not index_samples:
                            if not index_samples:
                                exact = False
                                break
                            # Anchor list shorter than this index: promote
                            # the first sample that has it (anchor-first
                            # convention stays: configs[0] pairs with it).
                            index_anchor = index_samples[0]
                        index_formula = _derive_formulas_tree(
                            index_anchor, index_samples, index_configs, report, f"{path}[{index}]"
                        )
                        one_sample_index = len(index_samples) == 1
                        shapeless_partial = (
                            isinstance(index_formula, Mapping)
                            and hasattr(index_anchor, "shape")
                            and "shape" not in index_formula
                        )
                        if (
                            index_formula is None
                            or (
                                isinstance(index_formula, Mapping)
                                and index_formula.get("type") == "FAILED"
                                and one_sample_index
                            )
                            or (shapeless_partial and one_sample_index)
                        ):
                            if hasattr(index_anchor, "shape"):
                                # Tensor constant at this index: pin the
                                # anchor's exact shape/stride 鈥?a 1-sample
                                # index cannot be extrapolated (an
                                # under-derived partial payload without a
                                # shape would fall back to another index's
                                # shape at expansion; B4 chunk elements
                                # showed this).
                                index_formula = {
                                    "shape": [{"type": "CONST", "value": int(dim)} for dim in index_anchor.shape],
                                    "stride": [
                                        {"type": "CONST", "value": int(stride)} for stride in index_anchor.stride()
                                    ],
                                }
                            else:
                                index_formula = {"type": "CONST", "value": index_anchor}
                        elif isinstance(index_formula, Mapping) and index_formula.get("type") == "FAILED":
                            exact = False
                            break
                        candidates.append(index_formula)
                    if exact:
                        per_index_formulas = candidates
            formula: dict[str, Any] = {"type": "variable_list", "count_formula": count_formula}
            if per_index_formulas is not None:
                formula["element_formulas"] = per_index_formulas
            elif element_formula is not None:
                formula["element_formula"] = element_formula
            return formula
        element_formulas = []
        derived = False
        for index, anchor_element in enumerate(anchor_value):
            element_samples = [value[index] for value in sample_values]
            element_formula = _derive_formulas_tree(
                anchor_element, element_samples, configs, report, f"{path}[{index}]"
            )
            if element_formula is not None:
                derived = True
            element_formulas.append(element_formula)
        if not derived:
            return None
        return element_formulas

    if isinstance(anchor_value, dict):
        keys = set(anchor_value)
        for value in sample_values:
            if not isinstance(value, dict) or set(value) != keys:
                report.notes.append(f"{path}: dict keys differ across anchors; keeping anchor value")
                return None
        key_formulas: dict[str, Any] = {}
        derived = False
        for key, anchor_item in anchor_value.items():
            item_formula = _derive_formulas_tree(
                anchor_item, [value[key] for value in sample_values], configs, report, f"{path}.{key}"
            )
            if item_formula is not None:
                derived = True
                key_formulas[key] = item_formula
        if not derived:
            return None
        return key_formulas

    if isinstance(anchor_value, bool) or not isinstance(anchor_value, int):
        # float/str/bool/dtype/device/None: require exact constancy, never
        # extrapolate.  Only exact integers participate in formula fits.
        for value in sample_values:
            if type(value) is not type(anchor_value) or value != anchor_value:
                raise TemplateBuildError(f"{path}: non-integer scalar differs across anchors")
        return None

    samples = list(zip(configs, [int(value) for value in sample_values]))
    formula = derive_value_formula(samples)
    if formula.get("type") == "CONST":
        return None
    if formula.get("type") == "FAILED":
        formula = _failed_to_lookup(formula, samples)
        report.failed_formulas += 1
    else:
        report.derived_formulas += 1
    return formula


def _build_entry(
    anchor_op: Any,
    sample_ops: Sequence[Any],
    configs: Sequence[Mapping[str, Any]],
    report: BuildReport,
    op_index: int,
    prev_anchor_op: Any = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "func": _op_func(anchor_op),
        "base_args": tuple(anchor_op.args) if isinstance(anchor_op.args, tuple) else anchor_op.args,
        "base_kwargs": anchor_op.kwargs,
        "base_out": anchor_op.out,
    }

    # Producer alias discipline: when the captured graph reuses the PREVIOUS
    # op's out object as this op's out (inplace writers like
    # apply_rope_inplace, region-end markers carrying the region result),
    # record it so expansion wires the same object identity.  The runtime
    # builds its dependency/DCE topology from tensor object identity, so a
    # silently materialised fresh out changes the replay's event structure
    # and shifts the performance estimate (C3: +2.6% from 18 lost aliases).
    # Single-point rule (issue #19): the ANCHOR pair's out identity decides -
    # the former per-sample shape/identity layers were redundant with the
    # structure bucketing (alias topology is part of the family fingerprint)
    # and with expansion's shape-consistency guard, which keeps the fresh out
    # whenever the aliased pair's shapes disagree (the only real risk).
    if prev_anchor_op is not None and anchor_op.out is prev_anchor_op.out:
        entry["out_alias_prev"] = True

    region_skip: set[str] = set()
    region_positions: list[int] = []
    if _region_kind(entry["func"]) is not None:
        index = _region_arg_index(entry["base_args"])
        if index is not None:
            region_skip.add(f"args[{index}]")
            region_positions.append(index)

    # Anchor-first convention: sample_values[0] is the anchor's own value and
    # configs[0] is the anchor config (see _derive_formulas_tree).
    args_samples = [entry["base_args"], *[op.args for op in sample_ops]]
    kwargs_samples = [entry["base_kwargs"], *[op.kwargs for op in sample_ops]]
    out_samples = [entry["base_out"], *[op.out for op in sample_ops]]

    # Region ids are raw ``id(module)`` addresses: exclude them from
    # derivation BEFORE running it (neutralise to a constant) so they neither
    # fail derivation nor inflate the failure report.
    if region_positions:

        def _neutralise_region_ids(sample: Any) -> Any:
            sample = list(sample)
            for position in region_positions:
                if position < len(sample):
                    sample[position] = 0
            return sample

        derive_anchor_args = _neutralise_region_ids(entry["base_args"])
        args_samples = [_neutralise_region_ids(sample) for sample in args_samples]
    else:
        derive_anchor_args = entry["base_args"]

    args_f = _derive_formulas_tree(derive_anchor_args, args_samples, configs, report, f"op{op_index}.args")
    if isinstance(args_f, list):
        for index in range(len(args_f)):
            if f"args[{index}]" in region_skip:
                args_f[index] = None
    kwargs_f = _derive_formulas_tree(entry["base_kwargs"], kwargs_samples, configs, report, f"op{op_index}.kwargs")
    if isinstance(kwargs_f, dict):
        for key in list(kwargs_f):
            if f"kwargs.{key}" in region_skip:
                kwargs_f[key] = None
    out_f = _derive_formulas_tree(entry["base_out"], out_samples, configs, report, f"op{op_index}.out")

    # The compressor schema identifies its cache and ratio. A constant
    # observed page count must not freeze this known cache-capacity law.
    if configs[0].get('structure_policy') == 'fixed_structure_v1' and entry['func'] == 'tensor_cast.compressor.default':
        from .formulas import _features_for

        samples = [anchor_op, *sample_ops]
        if all(len(op.args) > 5 and isinstance(op.args[1], torch.Tensor) and op.args[1].ndim == 3 for op in samples):
            ratio, indexer = samples[0].args[2], samples[0].args[5]
            phase = 'decode' if configs[0].get('is_decode') else 'prefill'
            role = 'index_blocks' if indexer else 'blocks'
            feature = f'{phase}/cache/{ratio}/{role}'
            if all(
                op.args[2] == ratio and op.args[5] == indexer and _features_for(c).get(feature) == op.args[1].shape[0]
                for op, c in zip(samples, configs)
            ):
                entry['cache_role_feature'] = feature
            compressed = f'prefill/cache/{ratio}/compressed_tokens'
            if phase == 'prefill' and all(
                isinstance(op.out, (list, tuple))
                and isinstance(op.out[0], torch.Tensor)
                and op.out[0].ndim == 3
                and op.args[2] == ratio
                and _features_for(c).get(compressed) == op.out[0].shape[1]
                for op, c in zip(samples, configs)
            ):
                # The compressor schema determines this dimension from its
                # packed input and ratio. An unrelated routing feature can
                # agree at every anchor while rounding differently elsewhere.
                out_f = out_f if isinstance(out_f, list) else [None] * len(anchor_op.out)
                output_formula = out_f[0] if isinstance(out_f[0], dict) else {}
                shape = output_formula.setdefault(
                    'shape', [{'type': 'CONST', 'value': int(n)} for n in anchor_op.out[0].shape]
                )
                shape[1] = {'type': 'LINEAR', 'feat': compressed, 'coeff': 1, 'offset': 0}
                out_f[0] = output_formula

    if configs[0].get('structure_policy') == 'fixed_structure_v1' and not configs[0].get('is_decode'):
        from .formulas import _features_for

        aligned = [anchor_op, *sample_ops]
        feature = None
        if entry['func'] == 'tensor_cast.init_routing_v2.default':
            feature = 'moe_route/routed_total'
        elif entry['func'] == 'tensor_cast.all_to_all.default':
            dispatch = (
                prev_anchor_op is not None
                and _op_func(prev_anchor_op) == 'tensor_cast.init_routing_v2.default'
                and anchor_op.args[0] is prev_anchor_op.out
            )
            direction = 'output' if dispatch else 'input'
            if all(
                list(op.args[1])
                == [_features_for(c).get(f'moe_route/{direction}_device/{i}') for i in range(len(op.args[1]))]
                for op, c in zip(aligned, configs)
            ):
                feature = f'moe_route/{direction}_total'
        if feature and all(
            isinstance(op.out, torch.Tensor) and op.out.ndim == 2 and _features_for(c).get(feature) == op.out.shape[0]
            for op, c in zip(aligned, configs)
        ):
            # init_routing emits topk_indices.numel() rows; each all_to_all
            # emits sum(output_split_sizes). Pin these schema-owned totals
            # before consumers share their producer's tensor formula.
            out_f = out_f if isinstance(out_f, dict) else {}
            shape = out_f.setdefault('shape', [{'type': 'CONST', 'value': int(n)} for n in anchor_op.out.shape])
            shape[0] = {'type': 'LINEAR', 'feat': feature, 'coeff': 1, 'offset': 0}

    # Learn an operator-local relation instead of independently fitting a
    # slice bound that is the input dimension at every training point.
    if entry['func'] == 'aten.slice.Tensor' and not configs[0].get('is_decode'):
        aligned = [anchor_op, *sample_ops]
        axes = []
        ends = []
        for sample in aligned:
            args = sample.args
            if (
                not isinstance(args, (list, tuple))
                or len(args) < 4
                or not isinstance(args[0], torch.Tensor)
                or args[0].ndim == 0
                or args[2] != 0
                or (len(args) > 4 and args[4] != 1)
            ):
                break
            axis = int(args[1]) % args[0].ndim
            if args[3] != args[0].shape[axis] or not isinstance(sample.out, torch.Tensor):
                break
            if tuple(sample.out.shape) != tuple(args[0].shape):
                break
            axes.append(axis)
            ends.append(int(args[3]))
        if len(axes) == len(aligned) and len(set(axes)) == 1 and len(set(ends)) > 1:
            entry['full_slice_axis'] = axes[0]

    if entry['func'] == 'aten.split_with_sizes.default' and not configs[0].get('is_decode'):
        aligned = [anchor_op, *sample_ops]
        window = dict(configs[0].get('model_parameters') or ()).get('WINDOW')
        if window:
            from .formulas import _features_for

            axes = []
            for sample, config in zip(aligned, configs):
                args = sample.args
                if len(args) < 3 or not isinstance(args[0], torch.Tensor) or args[0].ndim == 0:
                    break
                axis = int(args[2]) % args[0].ndim
                cutoff = _features_for(config).get('prefill/window/cutoff')
                if args[0].shape[axis] != window or list(args[1]) != [window - cutoff, cutoff]:
                    break
                axes.append(axis)
            if len(axes) == len(aligned) and len(set(axes)) == 1:
                # The official circular-window write splits its tail at
                # WINDOW - packed_tokens % WINDOW. Aligned residues in the
                # training S values must not freeze or alias the cutoff to B.
                lengths = [
                    {'type': 'LINEAR', 'feat': f'prefill/window/{name}', 'coeff': 1, 'offset': 0}
                    for name in ('first', 'cutoff')
                ]
                args_f = args_f if isinstance(args_f, list) else [None] * len(anchor_op.args)
                args_f[1] = lengths
                out_f = out_f if isinstance(out_f, list) else [None] * len(anchor_op.out)
                for index, tensor in enumerate(anchor_op.out):
                    formula = out_f[index] if isinstance(out_f[index], dict) else {}
                    shape = formula.setdefault('shape', [{'type': 'CONST', 'value': int(n)} for n in tensor.shape])
                    shape[axes[0]] = copy.deepcopy(lengths[index])
                    out_f[index] = formula
        parts = []
        for sample in aligned:
            args = sample.args
            if (
                len(args) < 2
                or not isinstance(args[0], torch.Tensor)
                or args[0].ndim == 0
                or (len(args) > 2 and args[2] != 0)
                or not isinstance(args[1], (list, tuple))
                or len(args[1]) < 2
                or len(set(args[1])) != 1
                or sum(args[1]) != args[0].shape[0]
            ):
                break
            parts.append(len(args[1]))
        if len(parts) == len(aligned) and len(set(parts)) == 1:
            entry['equal_leading_split'] = parts[0]

    if configs[0].get('structure_policy') == 'fixed_structure_v1' and entry['func'] in {
        'tensor_cast.multihead_latent_attention.default',
        'tensor_cast.multihead_latent_attention_quant.default',
        'tensor_cast.mla_sparse_attention.default',
        'tensor_cast.mla_sparse_attention_quant.default',
    }:
        from .formulas import _features_for

        # InputGenerator owns these request metadata tensors: K+1 query
        # offsets and K sequence/query lengths, where K = ceil(B/DP).
        feature = ('decode' if configs[0].get('is_decode') else 'prefill') + '/local_batch'
        roles = {}
        for position, offset in ((5, 1), (6, 0), (7, 0)):
            if all(
                len(op.args) > position
                and isinstance(op.args[position], torch.Tensor)
                and op.args[position].ndim == 1
                and op.args[position].shape[0] == _features_for(c).get(feature, -2) + offset
                for op, c in zip([anchor_op, *sample_ops], configs)
            ):
                roles[position] = (feature, offset)
        if roles:
            entry['attention_metadata_roles'] = roles

    if args_f is not None:
        entry["args_f"] = args_f
    if kwargs_f is not None:
        entry["kwargs_f"] = kwargs_f
    if out_f is not None:
        entry["out_f"] = out_f
    return entry


def _share_tensor_formulas(entries):
    """A captured tensor has one formula owner; consumers reuse that law."""
    owners = {}
    tensors, edges, constraints = {}, {}, {}

    def cache_edge(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            tensors[id(a)], tensors[id(b)] = a, b
            edges.setdefault(id(a), set()).add(id(b))
            edges.setdefault(id(b), set()).add(id(a))

    for entry in entries:
        args, out = entry['base_args'], entry['base_out']
        if entry['func'] == 'tensor_cast.scatter_nd_update_mla.default' and len(args) > 1:
            cache_edge(args[1], out)
        if entry['func'] == 'tensor_cast.compressor.default' and len(args) > 1:
            if isinstance(out, (list, tuple)) and len(out) > 1:
                cache_edge(args[1], out[1])
            feature = entry.pop('cache_role_feature', None)
            if feature:
                constraints[id(args[1])] = (feature, 0)
                tensors[id(args[1])] = args[1]
        for position, role in entry.pop('attention_metadata_roles', {}).items():
            constraints[id(args[position])] = role
            tensors[id(args[position])] = args[position]
    for origin, (feature, offset) in constraints.items():
        pending, visited = [origin], set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            tensor = tensors[key]
            formula = {
                'shape': [{'type': 'CONST', 'value': int(n)} for n in tensor.shape],
                'stride': [{'type': 'CONST', 'value': int(n)} for n in tensor.stride()],
            }
            formula['shape'][0] = {'type': 'LINEAR', 'feat': feature, 'coeff': 1, 'offset': offset}
            if key in owners and owners[key] != formula:
                raise TemplateBuildError('Conflicting official tensor roles share a tensor')
            owners[key] = formula
            pending.extend(edges.get(key, ()))

    def visit(base, formula):
        if isinstance(base, torch.Tensor):
            return owners.setdefault(id(base), formula)
        if isinstance(base, (list, tuple)):
            # Variable tensor lists require a symbolic reference program;
            # fixed scalar lists have no tensor owners to reconcile.
            if isinstance(formula, Mapping) and formula.get('type') == 'variable_list':
                return formula
            if formula is not None and not isinstance(formula, (list, tuple)):
                return formula
            fields = formula if formula is not None else [None] * len(base)
            result = [visit(item, fields[i] if i < len(fields) else None) for i, item in enumerate(base)]
            return result if any(item is not None for item in result) else None
        if isinstance(base, Mapping):
            fields = formula if isinstance(formula, Mapping) else {}
            result = {key: visit(item, fields.get(key)) for key, item in base.items()}
            return {key: item for key, item in result.items() if item is not None} or None
        return formula

    for entry in entries:
        for base, formula_field in (('base_args', 'args_f'), ('base_kwargs', 'kwargs_f')):
            result = visit(entry[base], entry.get(formula_field))
            if result is None:
                entry.pop(formula_field, None)
            else:
                entry[formula_field] = result
        axis = entry.pop('full_slice_axis', None)
        if axis is not None:
            args_f = entry.get('args_f')
            if isinstance(args_f, list) and isinstance(args_f[0], Mapping):
                source = args_f[0]
                shape = source.get('shape')
                if shape is not None:
                    args_f[3] = copy.deepcopy(shape[axis])
                    entry['out_f'] = copy.deepcopy(source)
        parts = entry.pop('equal_leading_split', None)
        if parts is not None:
            args_f = entry.get('args_f')
            source = args_f[0] if isinstance(args_f, list) else None
            shape = source.get('shape') if isinstance(source, Mapping) else None
            if shape and shape[0].get('type') == 'LINEAR':
                expr = shape[0]
                coeff, offset = expr.get('coeff', 1), expr.get('offset', 0)
                divided = None
                if coeff % parts == 0 and offset % parts == 0:
                    divided = {**expr, 'coeff': coeff // parts, 'offset': offset // parts}
                elif coeff == 1 and offset == 0:
                    divided = {
                        'type': 'DOMAIN_CEIL_DIV',
                        'feat': expr['feat'],
                        'divisor': parts,
                        'require_divisible': True,
                    }
                if divided is not None:
                    args_f[1] = [copy.deepcopy(divided) for _ in range(parts)]
                    outputs = []
                    for _ in range(parts):
                        item = copy.deepcopy(source)
                        item['shape'][0] = copy.deepcopy(divided)
                        outputs.append(item)
                    entry['out_f'] = outputs
        result = visit(entry['base_out'], entry.get('out_f'))
        if result is None:
            entry.pop('out_f', None)
        else:
            entry['out_f'] = result


def _build_entry_list(
    anchor_capture: _Capture,
    family_captures: Sequence[_Capture],
    report: BuildReport,
) -> list[dict[str, Any]]:
    """Build the entry list for one structure family.

    ``family_captures`` all share the same op count and func sequence.  The
    first capture is the anchor providing base values; every additional capture
    contributes exact-fit samples for formula derivation.
    """

    # Anchor-first ordering: the anchor capture must be configs[0] so that
    # _derive_formulas_tree's zip(configs, sample_values) stays aligned.
    ordered_captures = [anchor_capture, *[item for item in family_captures if item is not anchor_capture]]
    configs = [capture.config for capture in ordered_captures]
    sample_ops_per_position: list[list[Any]] = [[] for _ in anchor_capture.ops]
    for capture in ordered_captures[1:]:
        for position, op in enumerate(capture.ops):
            sample_ops_per_position[position].append(op)

    entries: list[dict[str, Any]] = []
    prev_anchor_op: Any = None
    for index, anchor_op in enumerate(anchor_capture.ops):
        samples = sample_ops_per_position[index]
        entries.append(_build_entry(anchor_op, samples, configs, report, index, prev_anchor_op))
        prev_anchor_op = anchor_op
    if configs[0].get('model_parameters') and (
        not configs[0].get('is_decode') or configs[0].get('structure_policy') == 'fixed_structure_v1'
    ):
        _share_tensor_formulas(entries)
    return entries


def _sort_captures_for_anchor(captures: Sequence[_Capture]) -> _Capture:
    return sorted(captures, key=lambda capture: (capture.config["B"], capture.config["S"]))[0]


def _served_key(config: Mapping[str, Any]) -> dict[str, Any]:
    """The served-key record of one capture config (coverage guard metadata)."""

    from template_generate import _signature_categories

    scatter_cat, pad_cat = _signature_categories(config)
    return {
        "par_sig": list(_parallel_signature(config)),
        "scatter_cat": scatter_cat,
        "pad_cat": pad_cat,
        "num_mtp_tokens": config["num_mtp_tokens"],
        "enable_dispatch_ffn_combine": config["enable_dispatch_ffn_combine"],
        "is_decode": bool(config.get("is_decode", False)),
        "do_compile": config.get("do_compile"),
        # B/S of the capture: lets coverage separate exact anchor points
        # from formula-extrapolated (interpolated) targets.  Older
        # templates without these fields keep the legacy exact verdict.
        "B": config["B"],
        "S": config["S"],
    }


def _build_group(
    group_key: tuple[Any, ...],
    captures: Sequence[_Capture],
    report: BuildReport,
) -> dict[str, Any]:
    """Fit shape/parameter formulas within one fixed operator structure."""
    anchor = _sort_captures_for_anchor(captures)
    entries = _build_entry_list(anchor, captures, report)
    report.families.append(
        {
            "op_count": len(anchor.ops),
            "members": [capture.key for capture in captures],
            "configs": [dict(capture.config) for capture in captures],
        }
    )
    if len(captures) == 1:
        report.identity_only_families.append(anchor.key)
    group = {
        "entries": entries,
        "served_keys": [_served_key(capture.config) for capture in captures],
        "trained_mtp": int(anchor.config["num_mtp_tokens"]),
    }
    return group


def build_fixed_template(captures, *, template_meta=None):
    """Build fixed operator/reference buckets for all production entrypoints."""
    from .structure_buckets import FIXED_POLICY

    return build_template(captures, template_meta={**(template_meta or {}), 'structure_policy': FIXED_POLICY})


def build_template(
    captures: Sequence[_Capture],
    *,
    template_meta: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], BuildReport]:
    """Assemble captures into an expander-ready template dict.

    Captures are partitioned by the expander group key first, so a template
    can carry several quantization families side by side; the expander picks
    the group by the target config's quantization/structure actions.

    Production build, coverage and bootstrap entrypoints use
    build_fixed_template. No entrypoint trains dynamic folded structures.
    """

    if not captures:
        raise TemplateBuildError("no captures supplied")
    with cache_value_formulas():
        return _assemble_template(captures, template_meta=template_meta)


def _assemble_template(
    captures: Sequence[_Capture],
    *,
    template_meta: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], BuildReport]:
    """Build the groups/report (the ``build_template`` body)."""

    report = BuildReport(source_keys=[capture.key for capture in captures])

    from .body_select import discrete_cell
    from .structure_buckets import FIXED_POLICY, bucket_id, structure_domain

    fixed = (template_meta or {}).get('structure_policy') == FIXED_POLICY
    if fixed:
        captures = [
            _Capture(c.key, {**c.config, 'structure_policy': FIXED_POLICY}, c.ops, c.exec_time) for c in captures
        ]

    captures_by_group: dict[tuple[Any, ...], list[_Capture]] = {}
    for capture in captures:
        # Flat bucket key (review issue #3): the discrete cell is the single
        # registration of every captured axis; the merge scope (quant
        # actions, decode, par projection) is derivable from it on read
        # (body_select.base_key) instead of being stored twice.
        key = discrete_cell(capture.config)
        if fixed and capture.config.get('is_decode'):
            from .structure_buckets import parallel_geometry

            # A fixed Decode body is fitted at one per-rank request batch.
            # Equal operator fingerprints do not justify cross-K formulas.
            key += ('rank_batch', parallel_geometry(capture.config)[1])
        elif fixed and any(
            name.startswith('COMPRESS_RATIO/') for name, _ in capture.config.get('model_parameters', ())
        ):
            from .structure_buckets import parallel_geometry

            sequence = int(capture.config['S'])
            kind = 'single' if sequence == 1 else 'short' if sequence < 5 else 'normal'
            packed_single = parallel_geometry(capture.config)[1] * sequence == 1
            key += ('query_regime', kind, packed_single)
        captures_by_group.setdefault(key, []).append(capture)

    # Within each discrete cell, formulas may share only equivalent structures.
    from .body_select import discrete_predicate, partition_by_mode

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    cell_to_subkeys: dict[tuple[Any, ...], list[list[Any]]] = {}

    for base_key in sorted(captures_by_group, key=lambda item: str(item[0])):
        members_all = captures_by_group[base_key]
        # Include parameter containers, tensor ranks and reference topology.
        buckets: OrderedDict[tuple[str, ...], list[_Capture]] = partition_by_mode(
            members_all, parameter_lists=True if fixed else None
        )
        sub_keys: list[list[Any]] = []
        for index, (_sig, bucket_members) in enumerate(buckets.items()):
            bucket_members.sort(key=lambda capture: capture.key)
            cell = discrete_cell(bucket_members[0].config)
            if len(base_key) > len(cell) and bucket_members[0].config.get('is_decode'):
                tag = f'k{base_key[-1]}_m{index}'
            elif len(base_key) > len(cell):
                tag = f'q{base_key[-2]}_{int(base_key[-1])}_m{index}'
            else:
                tag = f'm{index}'
            sub_key = tuple(cell) + (tag,)
            failures_before = report.failed_formulas
            group = _build_group(
                sub_key,
                bucket_members,
                report,
            )
            parameters = dict(bucket_members[0].config.get('model_parameters') or ())
            window = (
                parameters.get('WINDOW', 0)
                if not bucket_members[0].config.get('is_decode')
                and any(_op_func(op) == 'tensor_cast.scatter_nd_update_mla.default' for op in bucket_members[0].ops)
                else 0
            )
            from .serve_rules import prefill_window_regime, prefill_structure_domain

            regimes = sorted({prefill_window_regime(c.config, window) for c in bucket_members}) if window else []
            domains = []
            if window:
                for capture in bucket_members:
                    domain = prefill_structure_domain(capture.config, window)
                    if domain not in domains:
                        domains.append(domain)
            # A branch predicate is usable only when no sibling structure
            # claims the same official branch. Otherwise retain ambiguity.
            branch_unique = bool(window) and not any(
                prefill_structure_domain(c.config, window) in domains
                for signature, captures_in_mode in buckets.items()
                if signature != _sig
                for c in captures_in_mode
            )
            group["mode_meta"] = {
                "tag": f"m{index}",
                "mode_ops": len(bucket_members[0].ops),
                "n_members": len(bucket_members),
                "member_keys": [capture.key for capture in bucket_members],
                "member_configs": [dict(capture.config) for capture in bucket_members],
                "discrete_predicate": discrete_predicate(bucket_members),
                "parameterized_prefill": (
                    (len(buckets) == 1 or branch_unique)
                    and not bucket_members[0].config.get('is_decode')
                    and bool(bucket_members[0].config.get('model_parameters'))
                    and (not window or branch_unique)
                ),
                'prefill_window_branch': {'window': window, 'regimes': regimes, 'domains': domains} if window else None,
                # Per-body failure count: the interpolated gate in
                # template_graph.generate reads this instead of the
                # group-global count (md1-anchor lesson - a sibling
                # mode's failures must not veto this body).
                "failed_formulas": report.failed_formulas - failures_before,
            }
            group["discrete_cell"] = discrete_cell(bucket_members[0].config)
            if fixed:
                fixed_domains = []
                for capture in bucket_members:
                    domain = structure_domain(capture.config, window)
                    if domain is not None and domain not in fixed_domains:
                        fixed_domains.append(domain)
                group['mode_meta']['structure_bucket'] = {
                    'id': bucket_id(group['discrete_cell'], _sig),
                    'window': window,
                    'domains': fixed_domains,
                    'op_count': len(bucket_members[0].ops),
                }
            groups[sub_key] = group
            sub_keys.append(list(sub_key))
        cell_to_subkeys.setdefault(discrete_cell(members_all[0].config), []).extend(sub_keys)

    # Parameter routing table (RFC zip-gap F1/F2): a capture's parallel/MTP
    # parameter tuple maps to the group keys trained from it.  Coverage and
    # external tools use it to answer "are these parameters captured at all"
    # without re-deriving families; the expander still selects groups by the
    # config-derived key (scatter/pad excluded), so routing is an index, not
    # a gate.  With mode-granular bodies one param tuple maps to every mode
    # body trained from it.
    routing: dict[str, list[Any]] = {}
    for capture in captures:
        config = capture.config
        param_key = (
            f"{config['TP']},{config['EP']},{config['MOE_DP']},{config['DCP']},"
            f"{config['num_mtp_tokens']},{bool(config['enable_dispatch_ffn_combine'])},"
            f"{bool(config.get('is_decode', False))}"
        )
        # The parameter projection deliberately ignores the quant axes, so
        # sibling cells (DISABLED vs FP8, ...) trained from the SAME parallel
        # parameters must AGGREGATE - a first-write index dropped every
        # later cell's group keys and pointed queries at only one of them
        # (review issue #28).
        bucket = routing.setdefault(param_key, [])
        for sub_key in cell_to_subkeys[discrete_cell(config)]:
            if sub_key not in bucket:
                bucket.append(sub_key)

    meta: dict[str, Any] = {
        "format": "capture_induced_fixed_structure_v1" if fixed else "capture_induced_fixed_cell_v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_keys": list(report.source_keys),
        "group_keys": [list(group_key) for group_key in groups],
        "routing": routing,
        "coverage": report.as_dict(),
    }
    parameters = {
        tuple(tuple(pair) for pair in c.config['model_parameters'])
        for c in captures
        if c.config.get('model_parameters') is not None
    }
    if len(parameters) > 1:
        raise ValueError('Capture corpus mixes model formula parameters')
    if parameters:
        if any(c.config.get('model_parameters') is None for c in captures):
            raise ValueError('Capture corpus mixes declared and missing model formula parameters')
        meta['model_parameters'] = parameters.pop()
    if template_meta:
        meta.update({key: value for key, value in template_meta.items() if key != "func"})

    template = {"groups": groups, "template_meta": meta}
    return template, report


def load_captures_from_cache(
    keys: Iterable[str],
    cache_dir: str | Path | None = None,
) -> list[_Capture]:
    """Load and normalise captures for ``keys`` from a Card 1 cache directory."""

    captures: list[_Capture] = []
    for key in keys:
        payload = load_capture(key, cache_dir)
        if payload is None:
            raise TemplateBuildError(f"capture cache miss for key {key!r} (dir={cache_dir})")
        ops, exec_time = payload
        captures.append(
            _Capture(key=key, config=capture_config(key, cache_dir), ops=list(ops), exec_time=float(exec_time))
        )
    return captures


def discover_cache_keys(cache_dir: str | Path) -> list[str]:
    """List parseable capture keys present in a cache directory."""

    root = Path(cache_dir)
    keys: list[str] = []
    if not root.is_dir():
        return keys
    # Model-scoped corpus layout: ``<cache_dir>/<model_slug>/<key>.json.xz``.
    # One sub-directory per model means two models whose full layout
    # spellings collide can never overwrite each other's captures.  Flat
    # files stay readable, so existing corpora keep working unchanged.
    paths = sorted(root.glob("*.json.xz")) + sorted(root.glob("*/*.json.xz"))
    for path in paths:
        key = path.name[: -len(".json.xz")]
        try:
            parse_cache_key(key)
        except TemplateBuildError:
            continue
        keys.append(key)
    return keys


def save_template(template: Mapping[str, Any], path: str | Path) -> Path:
    """Persist a template as a json+xz artifact (atomic write).

    JSON cannot instantiate arbitrary objects, so a template file is data,
    not executable serialization; the graph codec preserves the Meta
    placeholder tensors (shape/stride/dtype/alias) inside the tree.
    """

    from .graph_codec import PRESET_MAX, encode_document, save_json_xz

    return save_json_xz(encode_document(dict(template), "template_lazy/1", preset=PRESET_MAX), path)


def load_template(path: str | Path) -> dict[str, Any]:
    """Load a template saved by :func:`save_template`."""

    from .graph_codec import decode_document

    template = decode_document(read_artifact_bytes(Path(path)), "template_lazy/1")
    if not isinstance(template, dict):
        raise TemplateBuildError(f"template artifact {path} did not decode to a dict")
    return template


__all__ = [
    "BuildReport",
    "TemplateBuildError",
    "build_template",
    "discover_cache_keys",
    "load_captures_from_cache",
    "parse_cache_key",
    "save_template",
]
