import operator

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort


class GroupedMatmulSwigluPass(TensorCastGraphModulePass):
    _op_map = {
        torch.ops.tensor_cast.grouped_matmul.default: torch.ops.tensor_cast.grouped_matmul_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant.default: torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4.default: torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4.default: torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_fp8.default: torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default,
    }

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            if not self._is_valid_swiglu_start(node):
                continue

            gate_node, up_node = node.args[0], node.args[1]

            if not self._is_valid_getitem_pair(gate_node, up_node):
                continue

            split_node = gate_node.args[0]
            if not self._is_valid_split_node(split_node, up_node):
                continue

            matmul_node = split_node.args[0]
            if not self._is_valid_matmul_node(matmul_node):
                continue

            if not self._check_user_counts(split_node, gate_node, up_node):
                continue

            fused_target = self._op_map.get(matmul_node.target)
            if fused_target is None:
                continue

            with graph.inserting_before(node):
                fused_node = graph.create_node(
                    "call_function",
                    fused_target,
                    args=tuple(matmul_node.args),
                    kwargs=matmul_node.kwargs,
                )

            node.replace_all_uses_with(fused_node)
            modified = True

        if modified:
            stable_topo_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm

    def _is_valid_swiglu_start(self, node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        if node.target != torch.ops.tensor_cast.swiglu.default:
            return False
        if len(node.args) != 2:
            return False
        return True

    def _is_valid_getitem_pair(self, gate_node: fx.Node, up_node: fx.Node) -> bool:
        if not isinstance(gate_node, fx.Node) or not isinstance(up_node, fx.Node):
            return False

        if gate_node.target != operator.getitem or up_node.target != operator.getitem:
            return False

        if len(gate_node.args) != 2 or len(up_node.args) != 2:
            return False

        idx_gate = gate_node.args[1]
        idx_up = up_node.args[1]

        indices = sorted([idx_gate, idx_up])
        return indices == [0, 1]

    def _is_valid_split_node(self, split_node: fx.Node, up_node: fx.Node) -> bool:
        if not isinstance(split_node, fx.Node):
            return False

        if up_node.args[0] != split_node:
            return False

        target = split_node.target

        if target == torch.ops.aten.split_with_sizes.default:
            if len(split_node.args) < 3:
                return False
            dim = split_node.args[2] if len(split_node.args) > 2 else 0
            sizes = split_node.args[1]

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

        elif target == torch.ops.aten.split.Tensor:
            if len(split_node.args) < 3:
                return False

            dim = split_node.args[2]

            if dim != -1:
                try:
                    if "val" in split_node.meta:
                        max_dim = split_node.meta["val"].dim() - 1
                        if dim != max_dim:
                            return False
                    else:
                        return False
                except Exception:
                    return False

            return True

        return False

    def _is_valid_matmul_node(self, node: fx.Node) -> bool:
        if not isinstance(node, fx.Node):
            return False
        if node.op != "call_function":
            return False
        return node.target in self._op_map

    def _check_user_counts(self, split_node: fx.Node, gate_node: fx.Node, up_node: fx.Node) -> bool:
        if len(split_node.users) != 2:
            return False
        if len(gate_node.users) != 1 or len(up_node.users) != 1:
            return False
        return True
