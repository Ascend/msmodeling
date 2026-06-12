from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.freezing_utils import maybe_set_is_frozen_param
from torch.fx import GraphModule, Node


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

    # Clean up the graph
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm
