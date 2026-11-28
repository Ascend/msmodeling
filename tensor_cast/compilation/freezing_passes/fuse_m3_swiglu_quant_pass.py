import operator

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass

_M3_SWIGLU = torch.ops.tensor_cast.m3_swiglu.default
_M3_SWIGLU_QUANT = torch.ops.tensor_cast.m3_swiglu_quant.default
_DYNAMIC_QUANT = torch.ops.tensor_cast.dynamic_quantize_symmetric.default
_DEFAULT_GROUP_SIZE = 128


class FuseM3SwigluQuantPass(TensorCastGraphModulePass):
    """Fuse ``m3_swiglu`` + ``dynamic_quantize_symmetric`` into ``m3_swiglu_quant``.

    Also removes redundant ``dynamic_quantize_symmetric`` when the input is already
    produced by ``m3_swiglu_quant`` (maps to silu_and_mul_post_quant on device).
    """

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != _DYNAMIC_QUANT:
                continue
            input_node = node.args[0]
            if not isinstance(input_node, fx.Node):
                continue

            m3_swiglu_node = self._unwrap_m3_swiglu(input_node)
            if m3_swiglu_node is not None and m3_swiglu_node.target == _M3_SWIGLU_QUANT:
                self._rewire_to_existing_m3_swiglu_quant(graph, node, m3_swiglu_node)
                modified = True
                continue

            if m3_swiglu_node is not None and m3_swiglu_node.target == _M3_SWIGLU and len(m3_swiglu_node.args) >= 4:
                self._fuse_swiglu_and_quant(graph, m3_swiglu_node, node)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()
        return gm

    @staticmethod
    def _unwrap_m3_swiglu(node: fx.Node) -> fx.Node | None:
        if node.op != "call_function":
            return None
        if node.target == _M3_SWIGLU or node.target == _M3_SWIGLU_QUANT:
            return node
        if node.target == operator.getitem and isinstance(node.args[0], fx.Node):
            if node.args[0].target in (_M3_SWIGLU, _M3_SWIGLU_QUANT):
                return node.args[0]
        return None

    @staticmethod
    def _get_or_create_getitem(graph: fx.Graph, src: fx.Node, index: int) -> fx.Node:
        for user in src.users:
            if user.op == "call_function" and user.target == operator.getitem and user.args[1] == index:
                return user
        with graph.inserting_after(src):
            return graph.call_function(operator.getitem, args=(src, index))

    def _rewire_to_existing_m3_swiglu_quant(self, graph: fx.Graph, quant_node: fx.Node, m3sq_node: fx.Node) -> None:
        quant_tensor = self._get_or_create_getitem(graph, m3sq_node, 0)
        quant_scale = self._get_or_create_getitem(graph, m3sq_node, 1)
        self._replace_quant_node(graph, quant_node, quant_tensor, quant_scale)

    def _fuse_swiglu_and_quant(self, graph: fx.Graph, swiglu_node: fx.Node, quant_node: fx.Node) -> None:
        gate, up, alpha, limit = swiglu_node.args[:4]
        group_size = (
            swiglu_node.args[4]
            if len(swiglu_node.args) >= 5
            else quant_node.kwargs.get("group_size", _DEFAULT_GROUP_SIZE)
        )
        with graph.inserting_before(swiglu_node):
            m3sq_node = graph.call_function(
                _M3_SWIGLU_QUANT,
                args=(gate, up, alpha, limit, group_size),
            )
        quant_tensor = self._get_or_create_getitem(graph, m3sq_node, 0)
        quant_scale = self._get_or_create_getitem(graph, m3sq_node, 1)
        self._replace_quant_node(graph, quant_node, quant_tensor, quant_scale)
        if len(swiglu_node.users) == 0:
            graph.erase_node(swiglu_node)

    @staticmethod
    def _replace_quant_node(
        graph: fx.Graph,
        quant_node: fx.Node,
        quant_tensor: fx.Node,
        quant_scale: fx.Node,
    ) -> None:
        for user in list(quant_node.users):
            if user.op == "call_function" and user.target == operator.getitem:
                replacement = quant_tensor if user.args[1] == 0 else quant_scale
                user.replace_all_uses_with(replacement)
                graph.erase_node(user)
            else:
                user.replace_input_with(quant_node, quant_tensor)
        graph.erase_node(quant_node)
