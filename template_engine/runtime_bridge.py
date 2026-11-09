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

"""Adapt generated operators and request metadata for native ModelRunner input."""

from __future__ import annotations

from typing import Any


# Custom ops renamed upstream after the corpus capture runtime.  Captures
# record the dispatcher name of their runtime; serving on a newer runtime
# must resolve the current name for the same op.  The mapping is verified
# by hold-out bit-exact replays (see tests/regression/tensor_cast/
# test_template_fixed_cell.py) and must stay minimal: an entry here is a
# claim that two dispatcher names are the same op, nothing more.
_OP_RENAMES = {
    "tensor_cast.v4_clamped_swiglu": "tensor_cast.clamped_swiglu",
}


def _resolve_exact(name: str):
    obj: Any = __import__("torch").ops
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_func(name: str | Any):
    """Resolve an op name like ``aten.mm.default`` to its dispatcher op.

    Template expansion emits ``OpOverload`` objects directly; serialized
    captures carry strings.  Both are accepted.  Names recorded by an
    older runtime are resolved through ``_OP_RENAMES`` when the exact
    name no longer exists on the current dispatcher.
    """

    if not isinstance(name, str):
        if name is None:
            raise ValueError(f"invalid op func name: {name!r}")
        return name
    if not name.strip():
        raise ValueError(f"invalid op func name: {name!r}")
    try:
        return _resolve_exact(name)
    except AttributeError:
        stem, dot, suffix = name.rpartition(".")
        renamed = _OP_RENAMES.get(stem) if dot and suffix else None
        if renamed is None:
            raise
        return _resolve_exact(f"{renamed}.{suffix}")


def request_op_invoke_infos(ops, attention_meta, *, dcp_size=1):
    """Restore request-length values consumed by the official cost models.

    Tensor placeholders preserve shapes but cannot answer CPU value reads.
    Bind schema-owned metadata and every reference to those same tensors.
    """
    import torch
    from tensor_cast.runtime import OpInvokeInfo

    resolved_ops = [(resolve_func(op.func), op) for op in ops]
    replacements = {}
    for func, op in resolved_ops:
        schema = getattr(func, '_schema', None)
        for position, argument in enumerate(getattr(schema, 'arguments', ())):
            name = argument.name
            if name not in ('query_start_loc', 'seq_lens', 'query_lens'):
                continue
            source = op.args[position] if position < len(op.args) else (op.kwargs or {}).get(name)
            actual = getattr(attention_meta, name, None)
            if not isinstance(source, torch.Tensor) or not isinstance(actual, torch.Tensor):
                continue
            if (
                name == 'seq_lens'
                and dcp_size > 1
                and getattr(attention_meta, 'is_dcp_decode', False)
                and schema.name
                in (
                    'tensor_cast::multihead_latent_attention',
                    'tensor_cast::multihead_latent_attention_quant',
                    'tensor_cast::mla_sparse_attention',
                    'tensor_cast::mla_sparse_attention_quant',
                )
            ):
                actual = torch.clamp(torch.div(actual + dcp_size - 1, dcp_size, rounding_mode='floor'), min=1)
            if source.shape != actual.shape or source.dtype != actual.dtype:
                raise ValueError(f'template request metadata disagrees with generated inputs: {name}')
            replacements[id(source)] = actual

    def bind(value):
        if isinstance(value, torch.Tensor):
            return replacements.get(id(value), value)
        if isinstance(value, (list, tuple)):
            return type(value)(bind(item) for item in value)
        if isinstance(value, dict):
            return {key: bind(item) for key, item in value.items()}
        return value

    return [OpInvokeInfo(func, bind(op.args), bind(op.kwargs), bind(op.out)) for func, op in resolved_ops]
