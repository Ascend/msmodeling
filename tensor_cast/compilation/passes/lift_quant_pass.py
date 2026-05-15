import logging
import operator
from typing import Any, Dict, Tuple

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)


class LiftCombineQuantPass(TensorCastGraphModulePass):
    _SWAPPABLE_OPS = {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    }

    _QUANTIZE_OPS = [
        torch.ops.tensor_cast.quantize.default,
        torch.ops.tensor_cast.dynamic_quantize_asymmetric.default,
        torch.ops.tensor_cast.dynamic_quantize_symmetric.default,
        torch.ops.tensor_cast.dynamic_quantize_mxfp4.default,
    ]

    """
    An FX graph pass that lifts `tensor_cast.quantize` operations as early
    as possible and combines identical quantization operations into one.
    This makes later fusions possible such as rms_norm+quant fusion.

    The pass works as follows:
    1. Iterates through all nodes in the graph to find `quantize` calls.
    2. For each `quantize` call, it traces its input backward through a chain
       of "swappable" ops (like reshape, view, transpose).
    3. It determines the true, pre-view-change input tensor.
    4. It uses a cache to see if this tensor has already been quantized with the
       same scale/offset.
       - If yes, it reuses the existing quantized tensor.
       - If no, it inserts a new `quantize` op right after the true input tensor
         and adds it to the cache.
    5. It rebuilds the chain of swappable ops on top of the (new or cached)
       quantized tensor.
    6. It replaces the original `quantize` node's uses with the final node of
       the rebuilt chain.
    7. Finally, it removes all the old, now-unused nodes.
    """

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        # TODO(jgong5): we should apply AOT dispatch to normalize the graph
        # before going through the remaining passes like this so that
        # the graph would only contain call_function not call_method
        # and the target would always be the op override like
        # torch.ops.tensor_cast.quantize.default
        logger.debug("Running LiftCombineQuantPass.........")

        def is_swappable(node: fx.Node) -> bool:
            """Checks if a node represents a swappable operation."""
            if node.op == "call_function":
                return node.target in self._SWAPPABLE_OPS
            return False

        def is_multi_output_node(node: fx.Node) -> bool:
            """Checks if a node produces multiple outputs, i.e. used by getitem."""
            return any(user.op == "call_function" and user.target == operator.getitem for user in node.users)

        graph = gm.graph

        for quantize_op in self._QUANTIZE_OPS:
            # Cache to store (target, args, kwargs) -> new_quantize_node
            # This enables combining identical quantization ops.
            node_cache: Dict[Tuple[Any, Any, Any], fx.Node] = {}
            # Iterate over a copy of nodes, as we'll be modifying the graph
            for node in graph.find_nodes(op="call_function", target=quantize_op):
                is_multi_output = is_multi_output_node(node)
                original_quant_node = node

                # --- Phase 1: Lifting ---
                # Traverse up the graph through swappable ops to find the earliest
                # possible point to place the quantization op.
                current_input = original_quant_node.args[0]
                swappable_ops_chain = []

                while is_swappable(current_input):
                    # As a safety check, we only lift through an op if it has a single user.
                    # Lifting an op with multiple users would affect other paths in the graph.
                    if len(current_input.users) > 1:
                        break

                    swappable_ops_chain.append(current_input)
                    current_input = current_input.args[0]

                # --- Phase 2: Combination / Creation ---
                # Get the arguments of the original quantization operation.
                args = original_quant_node.args
                kwargs = original_quant_node.kwargs

                # Create a unique key for the quantization operation.
                cache_key = (
                    original_quant_node.target,
                    (current_input, *args[1:]),
                    kwargs,
                )

                if cache_key in node_cache:
                    # An identical quantization op has already been created. Reuse it.
                    lifted_quant_node = node_cache[cache_key]
                    if is_multi_output:
                        new_getitem_nodes = {}
                        for user in list(lifted_quant_node.users):
                            if user.op == "call_function" and user.target == operator.getitem:
                                idx = user.args[1]
                                new_getitem_nodes.setdefault(idx, user)
                        quantized_tensor_node = new_getitem_nodes[0]
                    else:
                        quantized_tensor_node = lifted_quant_node
                else:
                    # This is the first time we see this quantization. Create a new node.
                    with graph.inserting_after(current_input):
                        lifted_quant_node = graph.call_function(
                            quantize_op,
                            args=(current_input, *args[1:]),
                            kwargs=kwargs,
                        )
                    # Add the new node to the cache for future reuse.
                    node_cache[cache_key] = lifted_quant_node
                    if is_multi_output:
                        new_getitem_nodes = {}
                        for user in list(original_quant_node.users):
                            if user.op == "call_function" and user.target == operator.getitem:
                                idx = user.args[1]
                                if idx not in new_getitem_nodes:
                                    with graph.inserting_after(lifted_quant_node):
                                        new_getitem_nodes[idx] = graph.call_function(
                                            operator.getitem,
                                            args=(lifted_quant_node, idx),
                                        )
                        assert 0 in new_getitem_nodes, (
                            f"Expected accessing to first output of {original_quant_node} but got {new_getitem_nodes}"
                        )
                        quantized_tensor_node = new_getitem_nodes[0]
                    else:
                        quantized_tensor_node = lifted_quant_node

                # --- Phase 3: Rebuilding ---
                # Re-apply the chain of swappable ops after the new quantization node.
                final_node = quantized_tensor_node
                for swappable_node in reversed(swappable_ops_chain):
                    key = (
                        swappable_node.target,
                        (final_node, *swappable_node.args[1:]),
                        swappable_node.kwargs,
                    )
                    if key in node_cache:
                        final_node = node_cache[key]
                        continue
                    with graph.inserting_after(final_node):
                        # Create a new swappable op node with the same arguments,
                        # but with its input connected to the previous node in our new chain.
                        call = getattr(graph, swappable_node.op)
                        final_node = call(
                            swappable_node.target,
                            args=(final_node, *swappable_node.args[1:]),
                            kwargs=swappable_node.kwargs,
                        )
                        node_cache[key] = final_node

                # --- Phase 4: Targeted Replacement ---
                # This is the key change. We replace users of each output separately.
                if is_multi_output:
                    for user in list(original_quant_node.users):
                        if user.op == "call_function" and user.target == operator.getitem:
                            # Get the index this user is accessing (0 for tensor, 1 for scale, etc.)
                            idx = user.args[1]

                            if idx == 0:
                                # Users of the quantized tensor should now use the output of our rebuilt chain.
                                user.replace_all_uses_with(final_node)
                            else:
                                assert idx in new_getitem_nodes
                                user.replace_all_uses_with(new_getitem_nodes[idx])
                else:
                    original_quant_node.replace_all_uses_with(final_node)

        # Turn on DCE before recompile.
        graph.eliminate_dead_code()
        gm.recompile()
        return gm
