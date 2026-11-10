# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modifications Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# SPDX-License-Identifier: BSD-3-Clause
# This file contains code derived from PyTorch
# (https://github.com/pytorch/pytorch). See the PyTorch LICENSE.

import functools
from typing import Any, Optional, Set

import torch
import torch.utils._pytree as pytree
from torch._inductor.freezing_utils import maybe_set_is_frozen_param
from torch.fx import GraphModule, Node


# Identity registries bridging constant folding and memory accounting.
#
# ``MemoryTracker`` classifies every consumed tensor against the model's
# registered ``named_parameters``/``named_buffers``.  Constant folding breaks
# that matching in both directions:
#
# * Folded constants (``_frozen_param*`` buffers created here or by the
#   inductor freezing fold) are *new* tensor objects consumed by ops but never
#   registered as params, so they would land in ``derived_resident`` and be
#   multiplied by region repeats, double-counting weight bytes.
# * Source tensors whose graph references were entirely replaced by folded
#   constants (the dead ``get_attr`` removals below) are no longer traced even
#   though their folded replacement *is* consumed, so they must not be counted
#   as unexecuted-resident.
#
# Keep the identities on the graph so cached forwards can register them
# with whichever Runtime executes the graph.
def register_graph_folded_constants(gm: GraphModule) -> None:
    from ..runtime import current_runtime

    runtime = current_runtime()
    if runtime is not None and runtime.memory_tracker is not None:
        runtime.memory_tracker.register_folded_constants(gm._folded_constant_obj_ids, gm._folded_source_obj_ids)


def replace_node_with_constant(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    constant: Any,  # Can be Tensor or other constant types like tuple
    name: Optional[str] = None,
) -> None:
    """
    Replaces a node in the graph with a 'get_attr' node pointing to
    a new constant value registered on the GraphModule.
    """
    g = gm.graph

    if name:
        qualname = name
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0  # type: ignore[assignment]
        i = gm._frozen_param_count

        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1  # type: ignore[assignment, operator]

        gm._frozen_param_count = i + 1  # type: ignore[assignment, operator]

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)
        new_input_node.name = node.name

    # Register the constant on the module
    # We use register_buffer for Tensors and simple setattr for others
    if isinstance(constant, torch.Tensor):
        gm.register_buffer(qualname, constant)
        # mark any constants created during freezing
        maybe_set_is_frozen_param(constant)
    else:
        # needed to suppress `does not reference an nn.Module...` warning
        # for non-tensor constants (like tuples of meta tensors)
        setattr(gm, qualname, constant)


class MetaConstantFolder(torch.fx.Interpreter):
    """
    A simplified FX Interpreter that executes operations on 'meta' tensors
    and records the results.

    It identifies nodes that can be pre-computed (folded) because
    all of their inputs are constants (either 'get_attr' nodes
    pointing to meta tensors or other primitive constants like ints).
    """

    def __init__(self, gm: GraphModule):
        super().__init__(gm)
        # Stores {node: folded_value} for nodes that can be replaced
        self.node_replacements: dict[Node, Any] = {}
        # A marker for values that are not constant (e.g., placeholders)
        self.unknown_value = object()

    def run(self) -> None:
        """
        Runs the interpreter over the graph.
        """
        # Seed the environment with `unknown_value` for all placeholders
        env: dict[Node, Any] = {}
        for n in self.module.graph.find_nodes(op="placeholder"):
            env[n] = self.unknown_value

        # Run the interpreter, populating self.env and self.node_replacements
        super().run(initial_env=env)

    def run_node(self, node: Node) -> Any:
        """
        Executes a single node.
        """
        # 1. Handle nodes that are not foldable
        if node.op == "placeholder":
            return self.unknown_value  # This value is not known

        if node.op == "output":
            # Just execute the output node to complete the trace
            return super().run_node(node)

        # 2. Fetch arguments from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(node)

        # 3. Check if all inputs are known constants
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)
        inputs_are_constants = True

        for inp in flattened_inputs:
            if inp is self.unknown_value:
                # Depends on a placeholder
                inputs_are_constants = False
                break
            if isinstance(inp, torch.Tensor) and inp.device.type != "meta":
                # Depends on a non-meta (e.g., 'cpu'/'cuda') tensor
                inputs_are_constants = False
                break

        # 4. Handle 'get_attr' (our constant sources)
        if node.op == "get_attr":
            val = super().run_node(node)
            # Only treat it as a constant if it's a meta tensor
            # or not a tensor at all (e.g., a float)
            if isinstance(val, torch.Tensor) and val.device.type != "meta":
                return self.unknown_value
            return val

        # 5. If inputs aren't constants, this node can't be folded
        if not inputs_are_constants:
            return self.unknown_value

        # 6. All inputs are constants (meta tensors or primitives).
        # We can try to execute the node.
        try:
            # This executes the op, e.g., torch.add(meta_t1, meta_t2)
            # which will return a new meta tensor.
            out = super().run_node(node)
        except Exception:
            # If the meta op fails (not implemented, etc.), we can't fold it
            return self.unknown_value

        # 7. Check that the output is also a meta tensor (or primitive)
        flattened_outputs = pytree.arg_tree_leaves(out)
        for o in flattened_outputs:
            if isinstance(o, torch.Tensor) and o.device.type != "meta":
                # The op produced a non-meta tensor. Don't fold.
                return self.unknown_value

        # 8. Success! This node is a constant. Record it for replacement.
        # We only need to replace 'call_function' nodes.
        if node.op == "call_function":
            self.node_replacements[node] = out

        # Return the computed value so it can be used by subsequent nodes
        return out


def _remove_dead_get_attrs(gm: GraphModule, source_ids: Set[int]) -> None:
    for child in gm.children():
        if isinstance(child, GraphModule):
            _remove_dead_get_attrs(child, source_ids)

    dead_nodes = [node for node in gm.graph.find_nodes(op="get_attr") if not node.users]
    for node in dead_nodes:
        # A dead get_attr means every consumer of this param/buffer was folded
        # into a constant: record the source tensor's identity so the memory
        # tracker can skip it in the unexecuted-resident account.  Resolve
        # before ``delattr``; dotted targets traverse submodule attributes.
        try:
            source = functools.reduce(getattr, str(node.target).split("."), gm)
        except (AttributeError, KeyError, TypeError):
            source = None
        if isinstance(source, torch.Tensor):
            source_ids.add(id(source))
        if hasattr(gm, node.target):
            delattr(gm, node.target)
        gm.graph.erase_node(node)


def _collect_folded_constant_ids(gm: GraphModule) -> Set[int]:
    """Collect ids of every ``_frozen_param*`` tensor reachable from ``gm``.

    The inductor freezing fold (``torch._inductor.constant_folding``) runs
    inside ``freeze()`` *before* this pass and registers its folded constants
    with the same ``_frozen_param`` prefix; this scan covers those in addition
    to the constants registered by ``replace_node_with_constant`` above.
    """
    constant_ids = set()
    for module in gm.modules():
        for mapping in (getattr(module, "_buffers", {}), vars(module)):
            for name, value in mapping.items():
                if name.startswith("_frozen_param") and isinstance(value, torch.Tensor):
                    constant_ids.add(id(value))
    return constant_ids


def fold_meta_constants(gm: GraphModule) -> GraphModule:
    """
    Performs constant folding on a GraphModule with 'meta' device tensors.

    This function will:
    1. Run the MetaConstantFolder to find all foldable nodes.
    2. Replace foldable nodes with 'get_attr' nodes pointing to new buffers.
    3. Clean up the graph by removing dead nodes.

    Args:
        gm: The GraphModule to process.
    """
    # Disable any dispatch modes (like FakeTensorMode)
    with torch.utils._python_dispatch._disable_current_modes():
        folder = MetaConstantFolder(gm)
        folder.run()

        # Replace all foldable nodes with new constants
        for node, constant in folder.node_replacements.items():
            replace_node_with_constant(gm, node, constant)

    # Clean up dead constants before DCE recursively lints child GraphModules.
    source_ids = set()
    _remove_dead_get_attrs(gm, source_ids)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    gm._folded_constant_obj_ids = _collect_folded_constant_ids(gm)
    gm._folded_source_obj_ids = source_ids
    return gm
