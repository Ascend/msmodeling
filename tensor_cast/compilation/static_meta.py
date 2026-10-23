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

"""Exact operator capture for static, meta-only inference workloads.

This is a per-workload compiler, not an executable cache. The serving optimizer
caches the final hardware result by exact request shape. Every miss captures a
new graph with all metadata specialized, runs the normal fusion passes, executes
it under Runtime, and releases the graph. Model weights/buffers must be immutable.
"""

import copy
import dataclasses

import torch
from torch._guards import TracingContext
from torch._inductor.decomposition import select_decomp_table
from torch._subclasses.functional_tensor import dispatch_functionalize, FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes


class StaticMetaCaptureUnsupported(ValueError):
    """The input cannot be safely specialized by this compiler."""


def flatten_tensor_inputs(value):
    """Lift tensors from metadata without losing repeated tensor identities.

    Host values are specialized for this one graph. Dataclasses are copied
    without re-running __post_init__, which may contain host-only calculations.
    All fields and additional instance attributes are visited, including KV and
    speculative metadata; none of those tensors is accidentally frozen as a
    captured constant.
    """
    tensors = []
    indices = {}

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type not in ("meta", "cpu"):
                raise StaticMetaCaptureUnsupported("Static capture requires meta or host metadata tensors")
            if id(item) not in indices:
                indices[id(item)] = len(tensors)
                tensors.append(item)
            index = indices[id(item)]
            return lambda values: values[index]
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            names = dict.fromkeys(f.name for f in dataclasses.fields(item))
            names.update(dict.fromkeys(getattr(item, "__dict__", {})))
            children = {name: visit(getattr(item, name)) for name in names}

            def rebuild(values):
                result = copy.copy(item)
                for name, child in children.items():
                    object.__setattr__(result, name, child(values))
                return result

            return rebuild
        if isinstance(item, dict):
            children = {key: visit(child) for key, child in item.items()}
            return lambda values: {key: child(values) for key, child in children.items()}
        if isinstance(item, (tuple, list)):
            children = [visit(child) for child in item]
            if isinstance(item, tuple) and hasattr(item, "_fields"):
                return lambda values: type(item)(*(child(values) for child in children))
            return lambda values: type(item)(child(values) for child in children)
        if isinstance(item, (set, frozenset)):
            children = [visit(child) for child in item]
            return lambda values: type(item)(child(values) for child in children)
        if item is None or isinstance(item, (bool, int, float, str, torch.dtype, torch.device, slice)):
            return lambda values: item
        raise StaticMetaCaptureUnsupported(f"Unsupported static metadata type: {type(item).__name__}")

    rebuild = visit(value)
    return tensors, rebuild


class StaticMetaForward:
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend

    def __call__(self, **kwargs):
        # Do not mix this standalone capture with an enclosing Dynamo/AOT trace.
        if TracingContext.try_get() is not None:
            raise StaticMetaCaptureUnsupported("Static capture cannot nest inside a tracing context")
        parameters = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        state_tensors = (*parameters.values(), *buffers.values())
        if any(t.device.type != "meta" and not (t.device.type == "cpu" and t.ndim == 0) for t in state_tensors):
            raise StaticMetaCaptureUnsupported("Static capture requires meta model parameters and buffers")
        flat, rebuild = flatten_tensor_inputs((parameters, buffers, kwargs))
        frozen_ids = {id(t) for t in (*parameters.values(), *buffers.values())}
        num_frozen = sum(id(t) in frozen_ids for t in flat)
        output_spec = None

        def forward(*inputs):
            nonlocal output_spec
            params, state, requests = rebuild(inputs)
            output = torch.func.functional_call(self.model, (params, state), (), requests)
            values, output_spec = pytree.tree_flatten(output)
            return tuple(values)

        # Capture/rewrites must not contribute fake/compiler operations to the
        # enclosing simulator's event stream. Only the lowered graph is recorded.
        with _disable_current_modes(), torch.no_grad():
            functional = dispatch_functionalize(forward, FunctionalTensorMode())
            # Wrapper aliases (notably V4's shared indexer projections) can
            # outlive functional_call's name-based restoration. Restore every
            # registered state mapping, including duplicate module paths, so
            # no FunctionalTensor from this capture reaches the next workload.
            state = [
                (module, dict(module.__dict__), dict(module._parameters), dict(module._buffers))
                for module in self.model.modules()
            ]
            try:
                graph = make_fx(
                    functional,
                    decomposition_table=select_decomp_table(),
                    tracing_mode="fake",
                    _allow_non_fake_inputs=True,
                )(*flat)
            finally:
                for module, attributes, params, buffers in state:
                    module.__dict__.clear()
                    module.__dict__.update(attributes)
                    module._parameters.clear()
                    module._parameters.update(params)
                    module._buffers.clear()
                    module._buffers.update(buffers)
            graph = self.backend.compile_static_meta(graph, flat, num_frozen)
        outputs = graph(*flat[num_frozen:])
        return pytree.tree_unflatten(list(outputs), output_spec)
