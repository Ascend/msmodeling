import operator

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort


class GroupedMatmulSwigluPass(TensorCastGraphModulePass):
    # Map original operators to their fused counterparts
    _op_map = {
        torch.ops.tensor_cast.grouped_matmul.default: (torch.ops.tensor_cast.grouped_matmul_swiglu.default),
        torch.ops.tensor_cast.grouped_matmul_quant.default: (torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default),
        torch.ops.tensor_cast.grouped_matmul_quant_int4.default: (
            torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default
        ),
        torch.ops.tensor_cast.grouped_matmul_mxfp4.default: (torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default),
        torch.ops.tensor_cast.grouped_matmul_fp8.default: (torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default),
    }

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        # Iterate over all nodes to find swiglu nodes as entry points.
        # Use list() to create a copy since the graph may be modified during iteration.
        for node in list(graph.nodes):
            if not self._is_valid_swiglu_start(node):
                continue

            gate_node, up_node = node.args

            # 1. Validate getitem nodes
            if not self._is_valid_getitem_pair(gate_node, up_node):
                continue

            # 2. Validate Split node (compatible with split.Tensor and split_with_sizes)
            split_node = gate_node.args[0]
            if not self._is_valid_split_node(split_node, up_node):
                continue

            # 3. Validate Matmul node
            matmul_node = split_node.args[0]
            if not self._is_valid_matmul_node(matmul_node):
                continue

            # 4. Check user counts to ensure no other operations reference these intermediate nodes
            if not self._check_user_counts(split_node, gate_node, up_node):
                continue

            # --- Perform Fusion ---
            fused_target = self._op_map[matmul_node.target]

            # Construct new node arguments by reusing matmul inputs.
            # Assumes grouped_matmul signature is (input, weight, bias?, ...)
            # and the fused op has the same signature, handling swiglu internally.
            new_args = tuple(matmul_node.args)

            with graph.inserting_before(node):
                fused_node = graph.create_node(
                    "call_function",
                    fused_target,
                    args=new_args,
                    kwargs=matmul_node.kwargs,  # Preserve kwargs if any
                )

            # Replace all uses of the swiglu node with the new fused node
            node.replace_all_uses_with(fused_node)
            modified = True

        if modified:
            stable_topo_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm

    def _is_valid_swiglu_start(self, node: fx.Node) -> bool:
        """Check if the node is a standard swiglu call."""
        if node.op != "call_function":
            return False
        if node.target != torch.ops.tensor_cast.swiglu.default:
            return False
        if len(node.args) != 2:
            return False
        return True

    def _is_valid_getitem_pair(self, gate_node: fx.Node, up_node: fx.Node) -> bool:
        """Check if nodes are two getitem operations with indices 0 and 1."""
        if not isinstance(gate_node, fx.Node) or not isinstance(up_node, fx.Node):
            return False

        if gate_node.target != operator.getitem or up_node.target != operator.getitem:
            return False

        # getitem args are typically (source_node, index)
        if len(gate_node.args) != 2 or len(up_node.args) != 2:
            return False

        idx_gate = gate_node.args[1]
        idx_up = up_node.args[1]

        # Must be indices 0 and 1 (order does not matter)
        indices = sorted([idx_gate, idx_up])
        return indices == [0, 1]

    def _is_valid_split_node(self, split_node: fx.Node, up_node: fx.Node) -> bool:
        """
        Validate split node, compatible with both split.Tensor and split_with_sizes.
        Example split.Tensor: target=torch.ops.aten.split.Tensor, args=(768, -1)
        """
        if not isinstance(split_node, fx.Node):
            return False

        # Ensure up_node also originates from the same split_node
        if up_node.args[0] != split_node:
            return False

        target = split_node.target

        # Case A: aten.split_with_sizes.default
        if target == torch.ops.aten.split_with_sizes.default:
            # Args are typically (input, sizes_list, dim)
            if len(split_node.args) < 3:
                return False
            dim = split_node.args[2] if len(split_node.args) > 2 else 0
            sizes = split_node.args[1]

            # Verify split occurs on the last dimension (-1)
            try:
                input_tensor = split_node.args[0]
                if "val" in input_tensor.meta:
                    max_dim = input_tensor.meta["val"].dim() - 1
                    if dim not in [-1, max_dim]:
                        return False
                else:
                    if dim != -1:
                        return False
            except Exception:
                if dim != -1:
                    return False

            if isinstance(sizes, (list, tuple)) and len(sizes) == 2:
                return True
            return False

        # Case B: aten.split.Tensor
        elif target == torch.ops.aten.split.Tensor:
            # Args are typically (input, split_size, dim)
            if len(split_node.args) < 3:
                return False

            dim = split_node.args[2]

            # Verify dimension is the last one (-1)
            if dim != -1:
                try:
                    if "val" in split_node.meta:
                        max_dim = split_node.meta["val"].dim() - 1
                        if dim != max_dim:
                            return False
                    else:
                        return False  # Conservative rejection if meta info is missing
                except Exception:
                    return False

            # Verify split count implicitly via user count check (performed earlier)
            # split.Tensor(size, dim) splits into N chunks; we expect exactly 2 users (gate, up)
            return True

        return False

    def _is_valid_matmul_node(self, node: fx.Node) -> bool:
        """Check if the node is a supported grouped_matmul variant."""
        if not isinstance(node, fx.Node):
            return False
        if node.op != "call_function":
            return False
        return node.target in self._op_map

    def _check_user_counts(self, split_node: fx.Node, gate_node: fx.Node, up_node: fx.Node) -> bool:
        """
        Ensure intermediate nodes are not used by other operations to guarantee fusion safety.
        """
        # split node must only be used by the two getitem nodes
        if len(split_node.users) != 2:
            return False
        # getitem nodes must only be used by the swiglu node
        if len(gate_node.users) != 1 or len(up_node.users) != 1:
            return False
        return True
