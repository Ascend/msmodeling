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

"""Register final graph constants with the tracker of each actual invocation."""

import torch
from torch._inductor.freezing_utils import is_frozen_param
from torch.utils._pytree import tree_leaves


STATIC_WEIGHT_ORIGIN = "tensor_cast_static_weight_origin"
STATIC_WEIGHT_TENSOR_IDS = "tensor_cast_static_weight_tensor_ids"


def _iter_graph_modules(graph):
    """Yield a GraphModule and all nested GraphModules exactly once."""
    seen = set()
    for module in graph.modules():
        if isinstance(module, torch.fx.GraphModule) and id(module) not in seen:
            seen.add(id(module))
            yield module


def mark_frozen_weight_origins(graph):
    """Mark constants introduced by parameter freezing as already charged weights."""
    for graph_module in _iter_graph_modules(graph):
        for node in graph_module.graph.nodes:
            if node.op != "get_attr":
                continue
            value = graph_module
            for part in node.target.split("."):
                value = getattr(value, part)
            tensors = [tensor for tensor in tree_leaves(value) if isinstance(tensor, torch.Tensor)]
            if node.target.split(".")[-1].startswith("_frozen_param"):
                node.meta[STATIC_WEIGHT_ORIGIN] = True
                node.meta[STATIC_WEIGHT_TENSOR_IDS] = frozenset(id(tensor) for tensor in tensors)
                continue
            frozen_tensor_ids = frozenset(id(tensor) for tensor in tensors if is_frozen_param(tensor))
            if frozen_tensor_ids:
                node.meta[STATIC_WEIGHT_ORIGIN] = True
                node.meta[STATIC_WEIGHT_TENSOR_IDS] = frozen_tensor_ids


def inherit_static_weight_origin(node):
    """Propagate provenance through a constant-only graph expression."""
    if any(input_node.meta.get(STATIC_WEIGHT_ORIGIN, False) for input_node in node.all_input_nodes):
        node.meta[STATIC_WEIGHT_ORIGIN] = True


def register_graph_weight_aliases(graph):
    """Register compiler-proven weight constants for each graph invocation.

    Constant collection happens once after graph transformations. The wrapper
    also executes on compiler-cache hits, where a fresh tracker needs the same
    registrations. Proven compiler provenance handles new storage, while the
    tracker separately recognizes constants that still alias registered storage.
    """
    constants = {}
    weight_constants = {}
    for graph_module in _iter_graph_modules(graph):
        for node in graph_module.graph.nodes:
            if node.op == "get_attr" and node.users:
                value = graph_module
                for part in node.target.split("."):
                    value = getattr(value, part)
                for tensor in tree_leaves(value):
                    if isinstance(tensor, torch.Tensor):
                        constants[id(tensor)] = tensor
                        weight_tensor_ids = node.meta.get(STATIC_WEIGHT_TENSOR_IDS)
                        is_static_weight = (
                            id(tensor) in weight_tensor_ids
                            if weight_tensor_ids is not None
                            else node.meta.get(STATIC_WEIGHT_ORIGIN, False)
                        )
                        if is_static_weight:
                            weight_constants[id(tensor)] = tensor
    tensors = tuple(constants.values())
    proven_weights = tuple(weight_constants.values())

    def execute(*args, **kwargs):
        from ..runtime import current_runtime

        runtime = current_runtime()
        if tensors and runtime is not None and runtime.memory_tracker is not None:
            runtime.memory_tracker.register_compiled_weight_constants(proven_weights)
            runtime.memory_tracker.register_compiled_weight_aliases(tensors)
        return graph(*args, **kwargs)

    return execute
