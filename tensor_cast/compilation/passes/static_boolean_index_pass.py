import operator

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass


_FALSE_MASK_TARGET = torch.ops.tensor_cast.static_false_mask.default
_INVERT_TARGETS = {
    operator.invert,
    torch.ops.aten.bitwise_not.default,
}
_PRESERVING_METHODS = {
    "bool",
    "clone",
    "contiguous",
    "detach",
    "expand",
    "permute",
    "repeat",
    "reshape",
    "to",
    "transpose",
    "view",
}
_PRESERVING_TARGETS = {
    torch.ops.aten.clone.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.repeat.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.view.default,
}


class StaticBooleanIndexPass(TensorCastGraphModulePass):
    """Lower Boolean indexing when mask value provenance is statically known."""

    def __init__(self):
        self.modified = False

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            source, mask = self._get_boolean_index_args(node)
            if source is None or mask is None:
                continue

            mask_is_false = self._is_false_mask(mask)
            if mask_is_false is None:
                continue

            if mask_is_false:
                with graph.inserting_before(node):
                    empty = graph.call_function(torch.ops.aten.slice.Tensor, args=(source, 0, 0, 0))
                    source_value = source.meta.get("val")
                    if isinstance(source_value, torch.Tensor):
                        empty.meta["val"] = source_value[:0]
                node.replace_all_uses_with(empty)
            else:
                node.replace_all_uses_with(source)
            graph.erase_node(node)
            modified = True

        self.modified = modified
        if modified:
            graph.eliminate_dead_code()
            gm.recompile()

        return gm

    @staticmethod
    def _get_boolean_index_args(node: fx.Node) -> tuple[fx.Node | None, fx.Node | None]:
        if node.target == operator.getitem and len(node.args) == 2:
            source, mask = node.args
        elif node.target == torch.ops.aten.index.Tensor and len(node.args) == 2:
            source, indices = node.args
            if len(indices) != 1:
                return None, None
            mask = indices[0]
        else:
            return None, None

        if not isinstance(source, fx.Node) or not isinstance(mask, fx.Node):
            return None, None
        return source, mask

    def _is_false_mask(self, node: fx.Node) -> bool | None:
        if node.target == _FALSE_MASK_TARGET:
            return True
        if node.target in _INVERT_TARGETS:
            mask_is_false = self._is_false_mask(node.args[0])
            return None if mask_is_false is None else not mask_is_false
        if node.op == "call_method" and node.target in _PRESERVING_METHODS:
            return self._is_false_mask(node.args[0])
        if node.target in _PRESERVING_TARGETS:
            return self._is_false_mask(node.args[0])
        if node.target == operator.getitem:
            return self._is_false_mask(node.args[0])
        if node.target == torch.ops.aten.bitwise_and.Tensor:
            left_is_false = self._is_false_mask(node.args[0])
            right_is_false = self._is_false_mask(node.args[1])
            if left_is_false is True or right_is_false is True:
                return True
        return None
